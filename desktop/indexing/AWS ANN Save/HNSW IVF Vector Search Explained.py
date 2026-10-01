# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,HNSW & IVF Vector Search Explained
# MAGIC %md
# MAGIC # Vector Search Index Algorithms: HNSW & IVF Explained
# MAGIC
# MAGIC Interactive guide to the two most common ANN (Approximate Nearest Neighbor) index algorithms used in vector search systems like Databricks Vector Search, OpenSearch Serverless, FAISS, and Milvus.
# MAGIC
# MAGIC **What you'll learn:**
# MAGIC - How HNSW (Hierarchical Navigable Small-World) graphs work — layers, M, efSearch
# MAGIC - How IVF (Inverted File) indexes work — clustering, nlist, nprobe
# MAGIC - Hands-on tuning: adjust parameters and see recall/latency trade-offs in real time
# MAGIC - Practical guidance for configuring Vector Search & OpenSearch indexes
# MAGIC
# MAGIC **Key concepts at a glance:**
# MAGIC
# MAGIC | Concept | HNSW | IVF |
# MAGIC |---|---|---|
# MAGIC | Structure | Proximity graph (hierarchical layers) | Clustered inverted lists |
# MAGIC | Build param | `M` (max connections per node) | `nlist` (number of clusters) |
# MAGIC | Query param | `efSearch` (candidate exploration width) | `nprobe` (clusters to search) |
# MAGIC | Navigation | Greedy graph traversal across layers | Find nearest centroids → search those lists |
# MAGIC | Memory | Higher (stores graph edges) | Lower (stores cluster assignments) |
# MAGIC | Recall vs speed | Tune efSearch ↑ for recall | Tune nprobe ↑ for recall |

# COMMAND ----------

# DBTITLE 1,Setup & Ground Truth
# ── Setup: imports and synthetic vector dataset ─────────────────────
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict, deque

np.random.seed(42)

# Generate synthetic embedding vectors (simulating document chunks)
# In real life these would be 768-dim or 1536-dim embeddings
VECTOR_DIM = 128
NUM_VECTORS = 2000
NUM_QUERIES = 100

vectors = np.random.randn(NUM_VECTORS, VECTOR_DIM).astype(np.float32)
queries = np.random.randn(NUM_QUERIES, VECTOR_DIM).astype(np.float32)

# Normalize for cosine similarity
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

print(f"Dataset: {NUM_VECTORS} vectors, dim={VECTOR_DIM}")
print(f"Queries: {NUM_QUERIES}")
print(f"Vectors normalized: {np.allclose(np.linalg.norm(vectors, axis=1), 1.0)}")

# ── Ground truth: exact nearest neighbors (brute force) ────────────────
def brute_force_knn(query_vectors, db_vectors, k=10):
    """Exact k-NN via dot product (cosine similarity for normalized vectors)."""
    results = []
    for q in query_vectors:
        sims = db_vectors @ q
        top_k_idx = np.argpartition(-sims, k)[:k]
        top_k_idx = top_k_idx[np.argsort(-sims[top_k_idx])]
        results.append(top_k_idx)
    return np.array(results)

ground_truth = brute_force_knn(queries, vectors, k=10)
print(f"Ground truth computed: {ground_truth.shape} (queries × k)")
print("✅ Setup complete")

# COMMAND ----------

# DBTITLE 1,How HNSW Works
# MAGIC %md
# MAGIC ## How HNSW Works
# MAGIC
# MAGIC **HNSW = Hierarchical Navigable Small-World graph**
# MAGIC
# MAGIC The index is a multi-layer graph where:
# MAGIC - **Layer 0 (bottom):** Contains ALL vectors. Each node connects to up to `M × 2` neighbors.
# MAGIC - **Layer 1, 2, ... (upper layers):** Progressively sparser subsets of nodes. Each node connects to up to `M` neighbors.
# MAGIC - A node present in layer L is also present in all layers below (0 to L).
# MAGIC - Layer assignment is probabilistic: `P(node in layer L) = exp(-L / mL)` where `mL = 1/ln(M)`.
# MAGIC
# MAGIC ### Search algorithm:
# MAGIC ```
# MAGIC 1. Start at the entry point in the TOP layer
# MAGIC 2. Greedily move to the nearest neighbor in the current layer
# MAGIC 3. When no improvement is found, descend to the layer below
# MAGIC 4. At Layer 0, switch to efSearch-bounded exploration:
# MAGIC    - Maintain a candidate list of size efSearch
# MAGIC    - Explore neighbors of the closest unexplored candidate
# MAGIC 5. Return the Top-K from the final candidate list
# MAGIC ```
# MAGIC
# MAGIC ### Key parameters:
# MAGIC
# MAGIC | Parameter | What it controls | Effect of increasing |
# MAGIC |---|---|---|
# MAGIC | `M` | Max connections per node per layer (graph density) | Higher recall, more memory, slower build |
# MAGIC | `efConstruction` | Candidate list size during index build | Better quality graph, slower build |
# MAGIC | `efSearch` | Candidate list size during query | Higher recall, slower query |
# MAGIC
# MAGIC **Important:** `efSearch ≥ k` (you must explore at least as many candidates as you want to return).
# MAGIC
# MAGIC **Not hash partitioning!** HNSW navigates a proximity graph — it doesn't hash vectors into buckets. The graph structure encodes geometric proximity directly.

# COMMAND ----------

# DBTITLE 1,HNSW Implementation & Build
# ── Simplified HNSW Implementation ────────────────────────────────────
# Educational implementation showing the core algorithm structure.
# Production systems (FAISS, hnswlib) use optimized C++ with pruning heuristics.

class SimpleHNSW:
    """Simplified HNSW index for educational purposes.

    Parameters:
        M:          Max connections per node in upper layers (Layer 0 gets 2×M)
        ef_construction: Candidate list size during graph building
    """

    def __init__(self, dim, M=16, ef_construction=64):
        self.dim = dim
        self.M = M
        self.M_max0 = 2 * M  # Layer 0 gets more connections
        self.ef_construction = ef_construction
        self.ml = 1.0 / np.log(M) if M > 1 else 1.0
        self.vectors = None
        self.graph = []          # graph[layer] = {node_id: [neighbor_ids]}
        self.entry_point = None
        self.max_layer = -1

    def _distance(self, a, b):
        return 1.0 - np.dot(a, b)  # cosine distance for normalized vectors

    def _random_layer(self):
        return int(-np.log(np.random.random()) * self.ml)

    def _search_layer(self, query_vec, entry_points, layer, ef):
        """Greedy search within a single layer. Returns ef closest nodes."""
        visited = set(entry_points)
        candidates = [(self._distance(query_vec, self.vectors[ep]), ep) for ep in entry_points]
        candidates.sort()
        results = list(candidates)

        while candidates:
            dist, node = candidates.pop(0)
            if results and dist > results[-1][0] and len(results) >= ef:
                break
            for neighbor in self.graph[layer].get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    d = self._distance(query_vec, self.vectors[neighbor])
                    if len(results) < ef or d < results[-1][0]:
                        candidates.append((d, neighbor))
                        results.append((d, neighbor))
                        results.sort()
                        if len(results) > ef:
                            results = results[:ef]
        return results

    def build(self, vectors):
        """Build the HNSW index."""
        self.vectors = vectors
        n = len(vectors)
        self.graph = [defaultdict(list) for _ in range(1)]  # start with layer 0
        self.entry_point = 0
        self.max_layer = 0

        for i in range(n):
            level = self._random_layer()
            # Extend graph layers if needed
            while len(self.graph) <= level:
                self.graph.append(defaultdict(list))

            # Insert at each layer from level down to 0
            ep = [self.entry_point]
            for layer in range(min(level, self.max_layer), -1, -1):
                search_results = self._search_layer(vectors[i], ep, layer, self.ef_construction)
                # Select M nearest as neighbors
                M = self.M_max0 if layer == 0 else self.M
                neighbors = [node for _, node in search_results[:M]]
                self.graph[layer][i] = neighbors
                # Add reverse edges
                for nb in neighbors:
                    self.graph[layer][nb].append(i)
                    # Prune if too many connections
                    max_conn = self.M_max0 if layer == 0 else self.M
                    if len(self.graph[layer][nb]) > max_conn:
                        # Keep closest max_conn neighbors
                        nb_vec = vectors[nb]
                        all_nbs = self.graph[layer][nb]
                        all_nbs.sort(key=lambda x: self._distance(nb_vec, vectors[x]))
                        self.graph[layer][nb] = all_nbs[:max_conn]
                ep = [node for _, node in search_results[:1]]

            if level > self.max_layer:
                self.max_layer = level
                self.entry_point = i

    def search(self, query_vec, k=10, ef_search=None):
        """Search for k nearest neighbors."""
        if ef_search is None:
            ef_search = max(k, 50)
        ef_search = max(ef_search, k)

        # Descend from top layer to layer 1 (greedy search)
        ep = [self.entry_point]
        for layer in range(self.max_layer, 0, -1):
            results = self._search_layer(query_vec, ep, layer, 1)
            if results:
                ep = [results[0][1]]

        # Layer 0: ef_search-bounded search
        results = self._search_layer(query_vec, ep, 0, ef_search)
        results.sort()
        return [node for _, node in results[:k]]


# ── Build HNSW with different M values ────────────────────────────────
print("Building HNSW indexes with different M values...")
hnsw_indexes = {}
for M in [4, 8, 16, 32]:
    t0 = time.time()
    idx = SimpleHNSW(dim=VECTOR_DIM, M=M, ef_construction=64)
    idx.build(vectors)
    build_time = time.time() - t0
    hnsw_indexes[M] = idx
    # Count edges in layer 0
    total_edges = sum(len(nbrs) for nbrs in idx.graph[0].values())
    print(f"  M={M:2d} | build={build_time:.2f}s | layer0_edges={total_edges} | max_layer={idx.max_layer}")

print("\n✅ HNSW indexes built")

# COMMAND ----------

# DBTITLE 1,HNSW Tuning: M & efSearch
# ── HNSW Tuning: M and efSearch impact on recall and latency ──────────

def evaluate_hnsw(hnsw_idx, k=10, ef_search=50):
    """Evaluate HNSW index: returns recall@k and avg query time (ms)."""
    correct = 0
    total_time = 0
    for i, q in enumerate(queries):
        t0 = time.time()
        result = hnsw_idx.search(q, k=k, ef_search=ef_search)
        total_time += time.time() - t0
        # recall = fraction of ground truth found
        gt_set = set(ground_truth[i][:k])
        result_set = set(result[:k])
        correct += len(gt_set & result_set)
    recall = correct / (NUM_QUERIES * k)
    avg_ms = (total_time / NUM_QUERIES) * 1000
    return recall, avg_ms


# ── Tune efSearch: recall vs latency trade-off ───────────────────────
print("=" * 70)
print("HNSW Tuning: efSearch vs Recall (M=16, k=10)")
print("=" * 70)
print(f"{'efSearch':>10} {'Recall@10':>12} {'Latency (ms)':>14} {'Speedup vs BF':>15}")
print("-" * 55)

ef_values = [10, 20, 30, 50, 75, 100, 150, 200, 300]
hnsw_results = []

# Brute force baseline timing
t0 = time.time()
for q in queries:
    sims = vectors @ q
    top_k = np.argpartition(-sims, 10)[:10]
bf_time = (time.time() - t0) / NUM_QUERIES * 1000

for ef in ef_values:
    recall, latency_ms = evaluate_hnsw(hnsw_indexes[16], k=10, ef_search=ef)
    speedup = bf_time / latency_ms if latency_ms > 0 else float('inf')
    hnsw_results.append((ef, recall, latency_ms, speedup))
    print(f"{ef:>10} {recall:>12.4f} {latency_ms:>14.2f} {speedup:>14.1f}x")

print(f"\n  Brute force baseline: {bf_time:.2f} ms/query")
print(f"  ⚡ efSearch=10 is {bf_time / hnsw_results[0][2]:.1f}x faster but only {hnsw_results[0][1]*100:.1f}% recall")
print(f"  🎯 efSearch=200 gives {hnsw_results[6][1]*100:.1f}% recall at {hnsw_results[6][2]:.2f} ms/query")

# ── Tune M: graph connectivity impact (fixed efSearch=50) ─────────────
print("\n" + "=" * 70)
print("HNSW Tuning: M vs Recall (efSearch=50, k=10)")
print("=" * 70)
print(f"{'M':>6} {'Recall@10':>12} {'Latency (ms)':>14} {'Layer0 edges':>14}")
print("-" * 50)

for M in [4, 8, 16, 32]:
    recall, latency_ms = evaluate_hnsw(hnsw_indexes[M], k=10, ef_search=50)
    edges = sum(len(nbrs) for nbrs in hnsw_indexes[M].graph[0].values())
    print(f"{M:>6} {recall:>12.4f} {latency_ms:>14.2f} {edges:>14}")

print("\n💡 Higher M = denser graph = better recall but more memory and slower build")
print("💡 Higher efSearch = more candidates explored = better recall but slower query")

# COMMAND ----------

# DBTITLE 1,How IVF Works
# MAGIC %md
# MAGIC ## How IVF Works
# MAGIC
# MAGIC **IVF = Inverted File index**
# MAGIC
# MAGIC The index partitions vector space into clusters using k-means:
# MAGIC - **Training:** Run k-means with `nlist` centroids on the dataset
# MAGIC - **Assignment:** Each vector is assigned to its nearest centroid → creates `nlist` inverted lists
# MAGIC - **Query:**
# MAGIC   1. Compute distance from query to ALL `nlist` centroids
# MAGIC   2. Select the `nprobe` nearest centroids
# MAGIC   3. Search only the vectors in those `nprobe` inverted lists (brute force within each)
# MAGIC   4. Return Top-K across searched lists
# MAGIC
# MAGIC ### Key parameters:
# MAGIC
# MAGIC | Parameter | What it controls | Effect of increasing |
# MAGIC |---|---|---|
# MAGIC | `nlist` | Number of clusters/centroids | Finer partitioning, smaller lists, but more centroid comparisons |
# MAGIC | `nprobe` | Number of clusters to search at query time | Higher recall, slower query |
# MAGIC
# MAGIC **Trade-off:** `nprobe=nlist` → exact search (100% recall, no speedup). `nprobe=1` → fastest but may miss neighbors in adjacent clusters.
# MAGIC
# MAGIC ### HNSW vs IVF summary:
# MAGIC
# MAGIC | Aspect | HNSW | IVF |
# MAGIC |---|---|---|
# MAGIC | Approach | Navigate proximity graph | Search selected clusters |
# MAGIC | Build | Insert nodes incrementally | Train k-means centroids |
# MAGIC | Query speed | Fast (graph hops) | Fast if nprobe ≪ nlist |
# MAGIC | Recall tuning | efSearch ↑ | nprobe ↑ |
# MAGIC | Memory | Higher (graph edges) | Lower (cluster assignments) |
# MAGIC | Best for | High-recall, low-latency | Large datasets, memory-constrained |

# COMMAND ----------

# DBTITLE 1,IVF Implementation & Build
# ── Simplified IVF Implementation ────────────────────────────────────

class SimpleIVF:
    """Simplified IVF index for educational purposes.

    Parameters:
        nlist: Number of clusters (inverted lists)
    """

    def __init__(self, dim, nlist=64):
        self.dim = dim
        self.nlist = nlist
        self.centroids = None
        self.inverted_lists = None  # list of arrays: inverted_lists[c] = vector indices in cluster c
        self.vectors = None

    def _kmeans(self, vectors, k, max_iter=20):
        """Simple k-means clustering."""
        n = len(vectors)
        # Initialize: random points as centroids
        idx = np.random.choice(n, min(k, n), replace=False)
        centroids = vectors[idx].copy()

        for _ in range(max_iter):
            # Assign each vector to nearest centroid
            dists = 1.0 - vectors @ centroids.T  # cosine distance
            assignments = np.argmin(dists, axis=1)
            # Update centroids
            for c in range(k):
                mask = assignments == c
                if mask.any():
                    centroids[c] = vectors[mask].mean(axis=0)
                    centroids[c] /= np.linalg.norm(centroids[c])
        return centroids, assignments

    def build(self, vectors):
        """Build IVF index: cluster vectors into nlist inverted lists."""
        self.vectors = vectors
        self.centroids, assignments = self._kmeans(vectors, self.nlist)
        self.inverted_lists = [np.where(assignments == c)[0] for c in range(self.nlist)]

    def search(self, query_vec, k=10, nprobe=8):
        """Search: find nprobe nearest centroids, search only those lists."""
        # Distance to all centroids
        cent_dists = 1.0 - query_vec @ self.centroids.T
        nearest_clusters = np.argpartition(cent_dists, min(nprobe, len(cent_dists) - 1))[:nprobe]

        # Gather candidates from selected clusters
        candidates = []
        for c in nearest_clusters:
            candidates.extend(self.inverted_lists[c])

        if not candidates:
            return []

        # Brute force within candidates
        candidates = np.array(candidates)
        sims = self.vectors[candidates] @ query_vec
        top_k_local = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        top_k_local = top_k_local[np.argsort(-sims[top_k_local])]
        return candidates[top_k_local].tolist()


# ── Build IVF with different nlist values ────────────────────────────
print("Building IVF indexes with different nlist values...")
ivf_indexes = {}
for nlist in [16, 32, 64, 128, 256]:
    t0 = time.time()
    idx = SimpleIVF(dim=VECTOR_DIM, nlist=nlist)
    idx.build(vectors)
    build_time = time.time() - t0
    ivf_indexes[nlist] = idx
    avg_list_size = np.mean([len(l) for l in idx.inverted_lists])
    print(f"  nlist={nlist:3d} | build={build_time:.2f}s | avg_list_size={avg_list_size:.1f} vectors")

print("\n✅ IVF indexes built")

# COMMAND ----------

# DBTITLE 1,IVF Tuning: nprobe & nlist
# ── IVF Tuning: nprobe vs recall/latency ──────────────────────────────

def evaluate_ivf(ivf_idx, k=10, nprobe=8):
    """Evaluate IVF index: returns recall@k and avg query time (ms)."""
    correct = 0
    total_time = 0
    for i, q in enumerate(queries):
        t0 = time.time()
        result = ivf_idx.search(q, k=k, nprobe=nprobe)
        total_time += time.time() - t0
        gt_set = set(ground_truth[i][:k])
        result_set = set(result[:k])
        correct += len(gt_set & result_set)
    recall = correct / (NUM_QUERIES * k)
    avg_ms = (total_time / NUM_QUERIES) * 1000
    return recall, avg_ms


# ── Tune nprobe for nlist=64 ─────────────────────────────────────────
print("=" * 70)
print("IVF Tuning: nprobe vs Recall (nlist=64, k=10)")
print("=" * 70)
print(f"{'nprobe':>8} {'% scanned':>12} {'Recall@10':>12} {'Latency (ms)':>14} {'Speedup vs BF':>15}")
print("-" * 65)

nprobe_values = [1, 2, 4, 8, 16, 32, 64]
ivf_results = []

for nprobe in nprobe_values:
    recall, latency_ms = evaluate_ivf(ivf_indexes[64], k=10, nprobe=nprobe)
    pct_scanned = (nprobe / 64) * 100
    speedup = bf_time / latency_ms if latency_ms > 0 else float('inf')
    ivf_results.append((nprobe, recall, latency_ms, speedup, pct_scanned))
    print(f"{nprobe:>8} {pct_scanned:>11.1f}% {recall:>12.4f} {latency_ms:>14.2f} {speedup:>14.1f}x")

print(f"\n  Brute force baseline: {bf_time:.2f} ms/query")
print(f"  ⚡ nprobe=1 scans only {1/64*100:.1f}% of data → {ivf_results[0][1]*100:.1f}% recall")
print(f"  🎯 nprobe=16 (25% of clusters) → {ivf_results[4][1]*100:.1f}% recall at {ivf_results[4][2]:.2f} ms/query")
print(f"  📌 nprobe=64 (= nlist) → {ivf_results[-1][1]*100:.1f}% recall (exact search, no speedup)")

# ── Tune nlist: cluster count impact (fixed nprobe=8) ────────────────
print("\n" + "=" * 70)
print("IVF Tuning: nlist vs Recall (nprobe=8, k=10)")
print("=" * 70)
print(f"{'nlist':>8} {'nprobe/nlist':>14} {'Recall@10':>12} {'Latency (ms)':>14}")
print("-" * 52)

for nlist in [16, 32, 64, 128, 256]:
    nprobe = min(8, nlist)
    recall, latency_ms = evaluate_ivf(ivf_indexes[nlist], k=10, nprobe=nprobe)
    ratio = f"{nprobe}/{nlist}"
    print(f"{nlist:>8} {ratio:>14} {recall:>12.4f} {latency_ms:>14.2f}")

print("\n💡 More clusters (higher nlist) = smaller lists = faster per-cluster search")
print("💡 But same nprobe scans fewer vectors → recall may drop unless you increase nprobe too")

# COMMAND ----------

# DBTITLE 1,Visualization: Recall vs Latency
# ── Visualization: Recall vs Latency trade-offs ──────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ── Plot 1: HNSW efSearch trade-off ───────────────────────────────────
ax = axes[0]
efs = [r[0] for r in hnsw_results]
recalls = [r[1] for r in hnsw_results]
latencies = [r[2] for r in hnsw_results]

ax.plot(latencies, recalls, 'bo-', linewidth=2, markersize=8)
for i, (ef, rec, lat) in enumerate(zip(efs, recalls, latencies)):
    if i % 2 == 0 or i == len(efs) - 1:
        ax.annotate(f'ef={ef}', (lat, rec), textcoords='offset points', xytext=(5, 5), fontsize=8)
ax.set_xlabel('Query Latency (ms)', fontsize=11)
ax.set_ylabel('Recall@10', fontsize=11)
ax.set_title('HNSW: efSearch Trade-off (M=16)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# ── Plot 2: IVF nprobe trade-off ──────────────────────────────────────
ax = axes[1]
nps = [r[0] for r in ivf_results]
recalls_ivf = [r[1] for r in ivf_results]
latencies_ivf = [r[2] for r in ivf_results]

ax.plot(latencies_ivf, recalls_ivf, 'rs-', linewidth=2, markersize=8)
for i, (np_val, rec, lat) in enumerate(zip(nps, recalls_ivf, latencies_ivf)):
    ax.annotate(f'nprobe={np_val}', (lat, rec), textcoords='offset points', xytext=(5, 5), fontsize=8)
ax.set_xlabel('Query Latency (ms)', fontsize=11)
ax.set_ylabel('Recall@10', fontsize=11)
ax.set_title('IVF: nprobe Trade-off (nlist=64)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# ── Plot 3: HNSW vs IVF comparison ────────────────────────────────────
ax = axes[2]
ax.plot(latencies, recalls, 'bo-', linewidth=2, markersize=8, label='HNSW (M=16)')
ax.plot(latencies_ivf, recalls_ivf, 'rs-', linewidth=2, markersize=8, label='IVF (nlist=64)')
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Exact (brute force)')
ax.set_xlabel('Query Latency (ms)', fontsize=11)
ax.set_ylabel('Recall@10', fontsize=11)
ax.set_title('HNSW vs IVF: Recall vs Latency', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

plt.tight_layout()
plt.show()

print("\n📊 Key insight: At equal recall levels, HNSW typically achieves lower latency")
print("   because graph traversal prunes the search space more efficiently than cluster scanning.")
print("   IVF is simpler and uses less memory, but needs higher nprobe to match HNSW recall.")

# COMMAND ----------

# DBTITLE 1,Practical Tuning Guide
# ── Practical guidance for real vector search systems ────────────────

print("=" * 70)
print("PRACTICAL TUNING GUIDE")
print("=" * 70)

print("""
┌────────────────────────────────────────────────────────────────────┐
│                    Databricks Vector Search                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Databricks VS auto-selects the index algorithm (HNSW or IVF)      │
│  based on the index configuration. Key knobs:                      │
│                                                                    │
│  • similarity: cosine, l2, dot_product                            │
│  • num_results: default Top-K (maps to efSearch internally)        │
│  • For Delta Sync indexes: the underlying engine auto-tunes       │
│    HNSW parameters based on dataset size                           │
│                                                                    │
│  Tips:                                                             │
│  • For RAG: efSearch ~50-100 typically gives >95% recall           │
│  • For large datasets (>1M vectors): consider IVF-flat or IVF-PQ  │
│  • Monitor recall by comparing VS results vs brute-force sample   │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│                    OpenSearch k-NN (used by Bedrock KB)             │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  OpenSearch supports two engines:                                  │
│                                                                    │
│  1. nmslib / faiss → HNSW                                          │
│     • parameters: M, ef_construction, ef_search                    │
│     • M: 16-48 (default 16), ef_construction: 100-512             │
│     • ef_search: set at query time (min 100 for good recall)       │
│                                                                    │
│  2. faiss → IVF                                                    │
│     • parameters: nlist, nprobe                                    │
│     • nlist: sqrt(N) is a common heuristic                         │
│     • nprobe: nlist/10 for ~90% recall (tune up for more)         │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│                    General Recommendations                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  • Start with HNSW (M=16, ef_construction=200) for <10M vectors   │
│  • Switch to IVF when memory is constrained or >10M vectors       │
│  • For RAG pipelines: target 90-95% recall (not 100%)             │
│    — the LLM tolerates slightly imperfect retrieval               │
│  • Benchmark on YOUR data: recall depends on vector distribution  │
│  • Rebuild index when data changes significantly (drift)           │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
""")

# ── Quick reference: parameter cheat sheet ──────────────────────────
print("""
QUICK REFERENCE TABLE
=====================

| Use Case           | Algorithm | M/nlist    | efSearch/nprobe | Expected Recall |
|-------------------|-----------|------------|-----------------|-----------------|
| RAG (<1M docs)     | HNSW      | M=16       | ef=50-100       | 90-98%          |
| RAG (1M-10M docs)  | HNSW      | M=32       | ef=100-200      | 90-95%          |
| Large scale (>10M) | IVF       | nlist=4096 | nprobe=64-256   | 85-95%          |
| Memory constrained | IVF       | nlist=1024 | nprobe=32-64    | 80-90%          |
| High precision     | HNSW      | M=48       | ef=500          | 98-100%         |
""")

print("✅ Notebook complete — HNSW and IVF explained with hands-on tuning!")