# Advanced RAG — Hands-On Labs

Seven runnable labs for **Advanced RAG Techniques**, built on the synthetic **Helix Medical /
HelixRAG** mini-corpus. Everything runs **locally** — the LLM, the judge, and the embeddings are
all served by a local [Ollama](https://ollama.com) container, so there is **no API key to set up
and nothing to pay for.**

| Lab | Topic |
|-----|-------|
| 01 | Eval-first: build the baseline & the smoking-gun stale-guideline retrieval |
| 02 | Chunking showdown: fixed vs recursive vs semantic + contextual prepending |
| 03 | Hybrid search (BM25 + dense + RRF) with cross-encoder re-ranking |
| 04 | Query enhancement: rewrite, multi-query fan-out, HyDE |
| 05 | Self-grading loop: Self-RAG / CRAG catches the contraindication |
| 06 | Production readiness: eval gate, cost routing + caching, tracing, governance |
| 07 | Capstone: assemble the full production `HelixRAG` and clear the mandate |

## Quickstart

**Prerequisite — a container engine.** Use **Docker Desktop**, or if you can't (licensing, locked-down
machine), the free, open-source **[Rancher Desktop](https://rancherdesktop.io)** instead. Rancher
Desktop ships the same `docker` CLI + Compose, so every command below is identical — just select the
**`dockerd (moby)`** engine in its Settings (not `containerd`) for full Compose compatibility, and give
the VM ~4–6 GB RAM. Nothing else in the labs changes: the container only runs the model server, so the
notebooks behave the same either way.

**1. Start the local model server** (first run downloads ~3.6 GB of models, then they're cached):

```bash
cd labs
docker compose up -d
docker compose logs -f ollama-pull   # wait until you see "All models pulled — labs are ready."
```

This serves three models on `http://localhost:11434`:
- `qwen2.5:3b` — generation + LLM-as-judge
- `qwen2.5:1.5b` — the cheap tier for the Lab 6 cost-routing demo
- `qwen3-embedding:0.6b` — embeddings (1024-dim)

**2. Install the Python dependencies:**

```bash
pip install -r requirements.txt
```

**3. Run the labs.** Open the notebooks in `notebooks/` (Jupyter, VS Code, etc.) and run top to
bottom, or execute headless:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/Lab-01-Eval-First-Baseline.ipynb
```

The notebooks import the shared corpus from [`helix_corpus.py`](helix_corpus.py). Swap that file
for your own corpus and a 300-pair golden set to use these labs on real data.

## Configuration

All models are overridable by environment variable (defaults shown):

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `GEN_MODEL` | `qwen2.5:3b` | Answer generation |
| `JUDGE_MODEL` | `qwen2.5:3b` | LLM-as-judge (faithfulness, relevance) |
| `SMALL_MODEL` | `qwen2.5:1.5b` | Cheap tier for cost routing |
| `EMB_MODEL` | `qwen3-embedding:0.6b` | Embeddings |

Want stronger answers and have the RAM/patience? Pull a bigger model and point at it:
`docker compose exec ollama ollama pull qwen2.5:7b && export GEN_MODEL=qwen2.5:7b`.

## Notes

- **CPU-only is fine.** These models are sized to run on a laptop without a GPU; the golden-set
  loops take a few minutes.
- The **cross-encoder reranker** (Lab 3) runs via `sentence-transformers` (`ms-marco-MiniLM-L-6-v2`,
  ~80 MB, downloaded on first use) — reranking isn't served through Ollama's API.
- To rebuild the notebooks from source: `python build_notebooks.py`.
