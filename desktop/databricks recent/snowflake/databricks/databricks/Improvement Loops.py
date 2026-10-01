# Databricks notebook source
# DBTITLE 1,Introduction: Learning from Production Experience
# MAGIC %md
# MAGIC # Improvement Loops: Building Agents That Learn from Experience
# MAGIC
# MAGIC This notebook demonstrates how to build **continuous improvement systems** that automatically learn from production failures and successes. No agentic system is perfect on day one—what matters is whether it can systematically learn from mistakes and get better over time.
# MAGIC
# MAGIC ## Why Improvement Loops Matter
# MAGIC
# MAGIC Even well-designed agents will encounter:
# MAGIC - **Edge cases** not covered in initial testing
# MAGIC - **User patterns** different from what you anticipated  
# MAGIC - **Environmental shifts** (new products, policy changes, seasonal variations)
# MAGIC - **Emergent failure modes** that only appear at scale
# MAGIC
# MAGIC Without structured learning, these failures repeat indefinitely. With improvement loops, each failure becomes training data for the next iteration.
# MAGIC
# MAGIC ## The Improvement Cycle
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────┐
# MAGIC │  Production │ ← Agents handle real requests
# MAGIC │  Monitoring │
# MAGIC └──────┬──────┘
# MAGIC        │ Collect failures (errors, low ratings, escalations)
# MAGIC        ▼
# MAGIC ┌─────────────┐
# MAGIC │   Issue     │ ← Cluster similar failures
# MAGIC │  Detection  │   Identify patterns
# MAGIC └──────┬──────┘
# MAGIC        │
# MAGIC        ▼
# MAGIC ┌─────────────┐
# MAGIC │    HITL     │ ← Humans review & annotate
# MAGIC │   Review    │   Decide on fixes
# MAGIC └──────┬──────┘
# MAGIC        │
# MAGIC        ▼
# MAGIC ┌─────────────┐
# MAGIC │ Refinement  │ ← Update prompts, tools, examples
# MAGIC │  & Testing  │   Validate improvements
# MAGIC └──────┬──────┘
# MAGIC        │
# MAGIC        ▼
# MAGIC ┌─────────────┐
# MAGIC │   Deploy    │ ← Shadow mode → A/B test → Full rollout
# MAGIC │  & Monitor  │
# MAGIC └──────┬──────┘
# MAGIC        │
# MAGIC        └──────► Back to production monitoring
# MAGIC ```
# MAGIC
# MAGIC ## What We'll Build
# MAGIC
# MAGIC 1. **Feedback Pipeline** — Collect failures from monitoring systems
# MAGIC 2. **Issue Clustering** — Group similar failures using embeddings
# MAGIC 3. **HITL Review Interface** — Triage and annotate failure clusters
# MAGIC 4. **Automated Refinement** — Pattern-based prompt improvements
# MAGIC 5. **Shadow Deployments** — Safe validation before serving users
# MAGIC 6. **A/B Testing** — Statistical comparison of variants
# MAGIC 7. **Bayesian Bandits** — Adaptive experiments that converge faster
# MAGIC 8. **In-Context Learning** — Session-level adaptation
# MAGIC 9. **End-to-End Integration** — Close the loop automatically
# MAGIC
# MAGIC ## Running Example
# MAGIC
# MAGIC We'll use our **customer support agent** from the Validation and Monitoring notebooks:
# MAGIC - Handles order status, returns, product questions
# MAGIC - Routes to specialists when needed
# MAGIC - Learns from failures to improve responses
# MAGIC
# MAGIC Let's build systems that make agents continuously better.

# COMMAND ----------

# DBTITLE 1,1. Feedback Pipeline: Collecting Failures at Scale
# MAGIC %md
# MAGIC ## 1. Feedback Pipeline: Collecting Failures at Scale
# MAGIC
# MAGIC The first step in any improvement loop is **capturing what went wrong**. Production monitoring generates vast amounts of data—we need to filter for actionable failures:
# MAGIC
# MAGIC ### Failure Types to Collect
# MAGIC
# MAGIC 1. **Hard Failures** — Exceptions, timeouts, tool errors
# MAGIC 2. **Quality Failures** — Low user ratings, thumbs down
# MAGIC 3. **Escalations** — Requests routed to humans
# MAGIC 4. **Policy Violations** — Unsafe outputs, PII leaks
# MAGIC 5. **Silent Failures** — High latency, unclear responses
# MAGIC
# MAGIC ### Data to Capture
# MAGIC
# MAGIC For each failure:
# MAGIC - User input (anonymized if needed)
# MAGIC - Agent output (partial or complete)
# MAGIC - Trace data (reasoning steps, tool calls)
# MAGIC - Context (user history, session state)
# MAGIC - Failure mode (error type, rating, escalation reason)
# MAGIC - Timestamp and environment
# MAGIC
# MAGIC Let's build a feedback collector that integrates with monitoring systems.

# COMMAND ----------

# DBTITLE 1,Feedback Collector Implementation
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import json

class FailureType(Enum):
    """Types of failures to collect for improvement."""
    EXCEPTION = "exception"
    LOW_RATING = "low_rating"
    ESCALATION = "escalation"
    POLICY_VIOLATION = "policy_violation"
    HIGH_LATENCY = "high_latency"
    UNCLEAR_RESPONSE = "unclear_response"

@dataclass
class FailureRecord:
    """Structured failure data for analysis."""
    failure_id: str
    timestamp: datetime
    failure_type: FailureType
    
    # Input/output
    user_input: str
    agent_output: Optional[str]
    
    # Trace data
    reasoning_steps: List[str]
    tool_calls: List[Dict[str, Any]]
    
    # Context
    session_id: str
    user_history: List[str]
    
    # Failure details
    error_message: Optional[str] = None
    rating: Optional[int] = None
    escalation_reason: Optional[str] = None
    latency_ms: Optional[int] = None
    
    # Metadata
    agent_version: str = "v1.0"
    environment: str = "production"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "failure_id": self.failure_id,
            "timestamp": self.timestamp.isoformat(),
            "failure_type": self.failure_type.value,
            "user_input": self.user_input,
            "agent_output": self.agent_output,
            "reasoning_steps": self.reasoning_steps,
            "tool_calls": self.tool_calls,
            "session_id": self.session_id,
            "user_history": self.user_history,
            "error_message": self.error_message,
            "rating": self.rating,
            "escalation_reason": self.escalation_reason,
            "latency_ms": self.latency_ms,
            "agent_version": self.agent_version,
            "environment": self.environment
        }

class FeedbackCollector:
    """Collects failures from production monitoring."""
    
    def __init__(self, storage_path: str = "failures.jsonl"):
        self.storage_path = storage_path
        self.buffer: List[FailureRecord] = []
        self.buffer_size = 100
    
    def record_exception(self, 
                        user_input: str,
                        error: Exception,
                        trace_data: Dict[str, Any],
                        session_id: str) -> FailureRecord:
        """Record an exception failure."""
        record = FailureRecord(
            failure_id=f"exc_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            failure_type=FailureType.EXCEPTION,
            user_input=user_input,
            agent_output=None,
            reasoning_steps=trace_data.get("reasoning_steps", []),
            tool_calls=trace_data.get("tool_calls", []),
            session_id=session_id,
            user_history=trace_data.get("user_history", []),
            error_message=str(error)
        )
        self._add_to_buffer(record)
        return record
    
    def record_low_rating(self,
                         user_input: str,
                         agent_output: str,
                         rating: int,
                         trace_data: Dict[str, Any],
                         session_id: str) -> FailureRecord:
        """Record a low user rating."""
        record = FailureRecord(
            failure_id=f"rating_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            failure_type=FailureType.LOW_RATING,
            user_input=user_input,
            agent_output=agent_output,
            reasoning_steps=trace_data.get("reasoning_steps", []),
            tool_calls=trace_data.get("tool_calls", []),
            session_id=session_id,
            user_history=trace_data.get("user_history", []),
            rating=rating
        )
        self._add_to_buffer(record)
        return record
    
    def record_escalation(self,
                         user_input: str,
                         agent_output: str,
                         escalation_reason: str,
                         trace_data: Dict[str, Any],
                         session_id: str) -> FailureRecord:
        """Record an escalation to human."""
        record = FailureRecord(
            failure_id=f"esc_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            failure_type=FailureType.ESCALATION,
            user_input=user_input,
            agent_output=agent_output,
            reasoning_steps=trace_data.get("reasoning_steps", []),
            tool_calls=trace_data.get("tool_calls", []),
            session_id=session_id,
            user_history=trace_data.get("user_history", []),
            escalation_reason=escalation_reason
        )
        self._add_to_buffer(record)
        return record
    
    def _add_to_buffer(self, record: FailureRecord):
        """Add record to buffer and flush if needed."""
        self.buffer.append(record)
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Write buffer to storage."""
        if not self.buffer:
            return
        
        with open(self.storage_path, 'a') as f:
            for record in self.buffer:
                f.write(json.dumps(record.to_dict()) + '\n')
        
        print(f"Flushed {len(self.buffer)} failure records to {self.storage_path}")
        self.buffer.clear()
    
    def load_failures(self, limit: Optional[int] = None) -> List[FailureRecord]:
        """Load failure records from storage."""
        failures = []
        try:
            with open(self.storage_path, 'r') as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                    data = json.loads(line)
                    failures.append(self._dict_to_record(data))
        except FileNotFoundError:
            print(f"No failure records found at {self.storage_path}")
        
        return failures
    
    def _dict_to_record(self, data: Dict[str, Any]) -> FailureRecord:
        """Convert dict back to FailureRecord."""
        return FailureRecord(
            failure_id=data["failure_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            failure_type=FailureType(data["failure_type"]),
            user_input=data["user_input"],
            agent_output=data.get("agent_output"),
            reasoning_steps=data.get("reasoning_steps", []),
            tool_calls=data.get("tool_calls", []),
            session_id=data["session_id"],
            user_history=data.get("user_history", []),
            error_message=data.get("error_message"),
            rating=data.get("rating"),
            escalation_reason=data.get("escalation_reason"),
            latency_ms=data.get("latency_ms"),
            agent_version=data.get("agent_version", "v1.0"),
            environment=data.get("environment", "production")
        )

# Initialize collector
collector = FeedbackCollector(storage_path="/tmp/agent_failures.jsonl")

print("✅ Feedback Collector initialized")
print("\nCollector captures:")
print("  • Exceptions and errors")
print("  • Low user ratings (1-2 stars)")
print("  • Escalations to humans")
print("  • Policy violations")
print("  • High latency responses")

# COMMAND ----------

# DBTITLE 1,Test: Collect Sample Failures
# Simulate collecting various failure types from production

# 1. Exception failure
try:
    # Simulate agent encountering an error
    raise ValueError("Order lookup failed: database timeout")
except Exception as e:
    collector.record_exception(
        user_input="What's the status of my order #12345?",
        error=e,
        trace_data={
            "reasoning_steps": [
                "User asked about order status",
                "Attempted to call order_lookup tool",
                "Database connection timed out"
            ],
            "tool_calls": [{"tool": "order_lookup", "args": {"order_id": "12345"}}],
            "user_history": ["Previous: Asked about return policy"]
        },
        session_id="sess_001"
    )

# 2. Low rating failure
collector.record_low_rating(
    user_input="Can I return this item after 60 days?",
    agent_output="Our return policy allows returns within 30 days of purchase.",
    rating=1,  # User gave thumbs down
    trace_data={
        "reasoning_steps": [
            "User asked about return timeframe",
            "Retrieved policy: 30-day window",
            "Generated response"
        ],
        "tool_calls": [{"tool": "get_policy", "args": {"policy_type": "returns"}}],
        "user_history": []
    },
    session_id="sess_002"
)

# 3. Escalation failure
collector.record_escalation(
    user_input="I received the wrong item and need a refund immediately",
    agent_output="I understand your frustration. Let me connect you with a specialist who can process refunds.",
    escalation_reason="Refund request exceeds agent authority",
    trace_data={
        "reasoning_steps": [
            "User reported wrong item received",
            "User requested immediate refund",
            "Refund amount likely >$100 threshold",
            "Escalating to human specialist"
        ],
        "tool_calls": [
            {"tool": "check_order", "args": {"issue_type": "wrong_item"}},
            {"tool": "escalate_to_human", "args": {"reason": "refund_request"}}
        ],
        "user_history": ["Previous: Order placed 3 days ago"]
    },
    session_id="sess_003"
)

# 4. Another low rating - different issue
collector.record_low_rating(
    user_input="Do you have this in blue?",
    agent_output="I apologize, but I don't have information about product colors in my current system.",
    rating=2,
    trace_data={
        "reasoning_steps": [
            "User asked about color availability",
            "Attempted to query product catalog",
            "Color field not available in tool response",
            "Generated fallback response"
        ],
        "tool_calls": [{"tool": "product_search", "args": {"query": "blue"}}],
        "user_history": ["Previous: Browsing product catalog"]
    },
    session_id="sess_004"
)

# 5. Similar escalation - also about refunds
collector.record_escalation(
    user_input="The product arrived damaged, I want my money back",
    agent_output="I'm sorry about the damaged product. Let me transfer you to our returns team.",
    escalation_reason="Damage claim requires human verification",
    trace_data={
        "reasoning_steps": [
            "User reported damaged product",
            "User requested refund",
            "Damage claims require photo verification",
            "Escalating to returns specialist"
        ],
        "tool_calls": [
            {"tool": "check_order", "args": {"issue_type": "damaged"}},
            {"tool": "escalate_to_human", "args": {"department": "returns"}}
        ],
        "user_history": []
    },
    session_id="sess_005"
)

# Flush to storage
collector.flush()

print("\n✅ Collected 5 sample failures:")
print("  • 1 exception (database timeout)")
print("  • 2 low ratings (policy unclear, missing color info)")
print("  • 2 escalations (both refund-related)")
print("\n➡️  Next: Cluster similar failures to find patterns")

# COMMAND ----------

# DBTITLE 1,2. Issue Clustering: Finding Failure Patterns
# MAGIC %md
# MAGIC ## 2. Issue Clustering: Finding Failure Patterns
# MAGIC
# MAGIC Individual failures tell us *what* went wrong. **Clusters** tell us *patterns*—recurring issues that justify systematic fixes.
# MAGIC
# MAGIC ### Why Cluster?
# MAGIC
# MAGIC Instead of fixing failures one-by-one:
# MAGIC - **Pattern Recognition** — "20 failures all about refunds" → systemic issue
# MAGIC - **Prioritization** — Fix the cluster affecting 100 users, not 3
# MAGIC - **Root Cause Analysis** — Similar failures often share root causes
# MAGIC - **Efficient Review** — Humans review clusters, not individual failures
# MAGIC
# MAGIC ### Approach
# MAGIC
# MAGIC 1. **Embed Failures** — Convert text (user input + error) to vectors using sentence-transformers
# MAGIC 2. **Cluster** — Apply HDBSCAN (finds variable-sized clusters + outliers) or k-means
# MAGIC 3. **Summarize** — Generate human-readable cluster descriptions
# MAGIC 4. **Visualize** — Plot in 2D to see failure landscape
# MAGIC
# MAGIC ### What Makes a Good Cluster?
# MAGIC
# MAGIC - **Coherent** — Failures share a common root cause
# MAGIC - **Actionable** — Clear fix strategy (e.g., "add refund policy to knowledge base")
# MAGIC - **Significant** — Impacts enough users to justify effort
# MAGIC
# MAGIC Let's build the clustering system.

# COMMAND ----------

# DBTITLE 1,Issue Clustering Implementation
# MAGIC %pip install -q sentence-transformers scikit-learn hdbscan matplotlib umap-learn
# MAGIC
# MAGIC import numpy as np
# MAGIC from sentence_transformers import SentenceTransformer
# MAGIC from sklearn.cluster import KMeans
# MAGIC import hdbscan
# MAGIC from collections import Counter, defaultdict
# MAGIC from typing import List, Dict, Tuple
# MAGIC import matplotlib.pyplot as plt
# MAGIC import umap
# MAGIC
# MAGIC class FailureClusterer:
# MAGIC     """Clusters similar failures using embeddings."""
# MAGIC     
# MAGIC     def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
# MAGIC         """Initialize with a sentence transformer model.
# MAGIC         
# MAGIC         all-MiniLM-L6-v2: Fast, good balance of speed and quality.
# MAGIC         """
# MAGIC         print(f"Loading embedding model: {model_name}...")
# MAGIC         self.model = SentenceTransformer(model_name)
# MAGIC         self.embeddings = None
# MAGIC         self.cluster_labels = None
# MAGIC         self.failures = None
# MAGIC         print("✅ Model loaded")
# MAGIC     
# MAGIC     def embed_failures(self, failures: List[FailureRecord]) -> np.ndarray:
# MAGIC         """Convert failures to embeddings.
# MAGIC         
# MAGIC         We embed: user_input + failure context (error message, escalation reason, etc.)
# MAGIC         This captures both what the user asked and why it failed.
# MAGIC         """
# MAGIC         texts = []
# MAGIC         for f in failures:
# MAGIC             # Combine user input with failure context
# MAGIC             text_parts = [f"User: {f.user_input}"]
# MAGIC             
# MAGIC             if f.error_message:
# MAGIC                 text_parts.append(f"Error: {f.error_message}")
# MAGIC             if f.escalation_reason:
# MAGIC                 text_parts.append(f"Escalation: {f.escalation_reason}")
# MAGIC             if f.rating and f.rating <= 2:
# MAGIC                 text_parts.append(f"Low rating: {f.rating}/5")
# MAGIC             
# MAGIC             texts.append(" | ".join(text_parts))
# MAGIC         
# MAGIC         print(f"Embedding {len(texts)} failure descriptions...")
# MAGIC         self.embeddings = self.model.encode(texts, show_progress_bar=True)
# MAGIC         self.failures = failures
# MAGIC         return self.embeddings
# MAGIC     
# MAGIC     def cluster_hdbscan(self, min_cluster_size: int = 2) -> np.ndarray:
# MAGIC         """Cluster using HDBSCAN (finds variable-sized clusters).
# MAGIC         
# MAGIC         HDBSCAN advantages:
# MAGIC         - Automatically finds number of clusters
# MAGIC         - Handles outliers (label = -1)
# MAGIC         - Finds clusters of varying density
# MAGIC         
# MAGIC         min_cluster_size: Minimum failures to form a cluster
# MAGIC         """
# MAGIC         if self.embeddings is None:
# MAGIC             raise ValueError("Must call embed_failures first")
# MAGIC         
# MAGIC         print(f"\nClustering with HDBSCAN (min_cluster_size={min_cluster_size})...")
# MAGIC         clusterer = hdbscan.HDBSCAN(
# MAGIC             min_cluster_size=min_cluster_size,
# MAGIC             metric='euclidean',
# MAGIC             cluster_selection_method='eom'  # Excess of Mass
# MAGIC         )
# MAGIC         self.cluster_labels = clusterer.fit_predict(self.embeddings)
# MAGIC         
# MAGIC         n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
# MAGIC         n_outliers = sum(1 for label in self.cluster_labels if label == -1)
# MAGIC         
# MAGIC         print(f"✅ Found {n_clusters} clusters")
# MAGIC         print(f"  • {n_outliers} outliers (unique failures)")
# MAGIC         
# MAGIC         return self.cluster_labels
# MAGIC     
# MAGIC     def cluster_kmeans(self, n_clusters: int = 3) -> np.ndarray:
# MAGIC         """Cluster using k-means (specify number of clusters).
# MAGIC         
# MAGIC         Use when you know roughly how many failure types to expect.
# MAGIC         """
# MAGIC         if self.embeddings is None:
# MAGIC             raise ValueError("Must call embed_failures first")
# MAGIC         
# MAGIC         print(f"\nClustering with k-means (n_clusters={n_clusters})...")
# MAGIC         kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
# MAGIC         self.cluster_labels = kmeans.fit_predict(self.embeddings)
# MAGIC         
# MAGIC         print(f"✅ Created {n_clusters} clusters")
# MAGIC         return self.cluster_labels
# MAGIC     
# MAGIC     def get_cluster_summaries(self) -> Dict[int, Dict[str, any]]:
# MAGIC         """Generate human-readable summaries for each cluster."""
# MAGIC         if self.cluster_labels is None or self.failures is None:
# MAGIC             raise ValueError("Must cluster first")
# MAGIC         
# MAGIC         summaries = {}
# MAGIC         
# MAGIC         for cluster_id in set(self.cluster_labels):
# MAGIC             cluster_failures = [
# MAGIC                 f for f, label in zip(self.failures, self.cluster_labels)
# MAGIC                 if label == cluster_id
# MAGIC             ]
# MAGIC             
# MAGIC             if not cluster_failures:
# MAGIC                 continue
# MAGIC             
# MAGIC             # Gather statistics
# MAGIC             failure_types = [f.failure_type.value for f in cluster_failures]
# MAGIC             type_counts = Counter(failure_types)
# MAGIC             
# MAGIC             # Sample inputs
# MAGIC             sample_inputs = [f.user_input for f in cluster_failures[:3]]
# MAGIC             
# MAGIC             # Common errors/reasons
# MAGIC             errors = [f.error_message for f in cluster_failures if f.error_message]
# MAGIC             escalation_reasons = [f.escalation_reason for f in cluster_failures if f.escalation_reason]
# MAGIC             
# MAGIC             summaries[cluster_id] = {
# MAGIC                 "size": len(cluster_failures),
# MAGIC                 "failure_types": dict(type_counts),
# MAGIC                 "sample_inputs": sample_inputs,
# MAGIC                 "common_errors": Counter(errors).most_common(3) if errors else [],
# MAGIC                 "common_escalations": Counter(escalation_reasons).most_common(3) if escalation_reasons else [],
# MAGIC                 "failures": cluster_failures
# MAGIC             }
# MAGIC         
# MAGIC         return summaries
# MAGIC     
# MAGIC     def visualize_clusters(self, figsize=(12, 8)):
# MAGIC         """Visualize clusters in 2D using UMAP."""
# MAGIC         if self.embeddings is None or self.cluster_labels is None:
# MAGIC             raise ValueError("Must cluster first")
# MAGIC         
# MAGIC         print("\nReducing to 2D for visualization...")
# MAGIC         reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=5)
# MAGIC         embedding_2d = reducer.fit_transform(self.embeddings)
# MAGIC         
# MAGIC         plt.figure(figsize=figsize)
# MAGIC         
# MAGIC         # Plot each cluster with different color
# MAGIC         unique_labels = set(self.cluster_labels)
# MAGIC         colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
# MAGIC         
# MAGIC         for label, color in zip(unique_labels, colors):
# MAGIC             if label == -1:
# MAGIC                 # Outliers in gray
# MAGIC                 color = 'lightgray'
# MAGIC                 label_name = 'Outliers'
# MAGIC                 marker = 'x'
# MAGIC             else:
# MAGIC                 label_name = f'Cluster {label}'
# MAGIC                 marker = 'o'
# MAGIC             
# MAGIC             mask = self.cluster_labels == label
# MAGIC             plt.scatter(
# MAGIC                 embedding_2d[mask, 0],
# MAGIC                 embedding_2d[mask, 1],
# MAGIC                 c=[color],
# MAGIC                 label=f"{label_name} (n={sum(mask)})",
# MAGIC                 alpha=0.7,
# MAGIC                 s=100,
# MAGIC                 marker=marker
# MAGIC             )
# MAGIC         
# MAGIC         plt.xlabel('UMAP Dimension 1', fontsize=12)
# MAGIC         plt.ylabel('UMAP Dimension 2', fontsize=12)
# MAGIC         plt.title('Failure Clusters in Embedding Space', fontsize=14, fontweight='bold')
# MAGIC         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# MAGIC         plt.grid(True, alpha=0.3)
# MAGIC         plt.tight_layout()
# MAGIC         
# MAGIC         display(plt.gcf())
# MAGIC         plt.close()
# MAGIC         
# MAGIC         print("✅ Visualization complete")
# MAGIC
# MAGIC # Initialize clusterer
# MAGIC clusterer = FailureClusterer()
# MAGIC
# MAGIC print("\n➡️  Ready to cluster failures")

# COMMAND ----------

# DBTITLE 1,Test: Cluster Sample Failures
# Load failures and cluster them
failures = collector.load_failures()
print(f"Loaded {len(failures)} failures\n")

# Embed failures
embeddings = clusterer.embed_failures(failures)

# Cluster with HDBSCAN (automatically finds clusters)
labels = clusterer.cluster_hdbscan(min_cluster_size=2)

# Get cluster summaries
summaries = clusterer.get_cluster_summaries()

print("\n" + "="*60)
print("CLUSTER SUMMARIES")
print("="*60)

for cluster_id, summary in sorted(summaries.items()):
    if cluster_id == -1:
        print(f"\n🔍 OUTLIERS (n={summary['size']})")
    else:
        print(f"\n📦 CLUSTER {cluster_id} (n={summary['size']})")
    
    print(f"  Failure types: {summary['failure_types']}")
    
    print(f"\n  Sample inputs:")
    for i, inp in enumerate(summary['sample_inputs'], 1):
        print(f"    {i}. '{inp}'")
    
    if summary['common_errors']:
        print(f"\n  Common errors:")
        for error, count in summary['common_errors']:
            print(f"    • {error} ({count}x)")
    
    if summary['common_escalations']:
        print(f"\n  Common escalations:")
        for reason, count in summary['common_escalations']:
            print(f"    • {reason} ({count}x)")

print("\n" + "="*60)

# Visualize
clusterer.visualize_clusters()

print("\n✅ Clustering complete!")
print("\n💡 Insights:")
print("  • Cluster 0: Both failures are refund/return escalations")
print("  • Other failures are outliers (unique issues)")
print("\n➡️  Next: Build HITL interface for humans to review clusters")

# COMMAND ----------

# DBTITLE 1,3. HITL Review Interface
# MAGIC %md
# MAGIC ## 3. Human-in-the-Loop Review Interface
# MAGIC
# MAGIC Clusters identify *patterns*, but humans decide *what to do about them*. A good HITL interface helps reviewers:
# MAGIC
# MAGIC 1. **Understand the cluster** — What's the common failure mode?
# MAGIC 2. **Diagnose root cause** — Is it a prompt issue? Missing tool? Knowledge gap?
# MAGIC 3. **Decide on fix strategy** — Update prompt? Add examples? Build new tool?
# MAGIC 4. **Prioritize** — Which clusters impact users most?
# MAGIC
# MAGIC ### Review Workflow
# MAGIC
# MAGIC ```
# MAGIC Cluster → Reviewer sees:
# MAGIC   • Cluster size & failure types
# MAGIC   • Sample failures with full context
# MAGIC   • Suggested root cause (from pattern analysis)
# MAGIC   → Reviewer annotates:
# MAGIC      • Root cause (confirmed/corrected)
# MAGIC      • Fix strategy (prompt, tool, knowledge, escalate)
# MAGIC      • Priority (critical, high, medium, low)
# MAGIC ```
# MAGIC
# MAGIC Let's build a simple review interface.

# COMMAND ----------

# DBTITLE 1,HITL Review Implementation
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List
import json

class RootCause(Enum):
    """Common root causes for agent failures."""
    PROMPT_UNCLEAR = "prompt_unclear"
    MISSING_KNOWLEDGE = "missing_knowledge"
    TOOL_ERROR = "tool_error"
    POLICY_GAP = "policy_gap"
    AMBIGUOUS_INPUT = "ambiguous_input"
    SCOPE_EXCEEDED = "scope_exceeded"
    OTHER = "other"

class FixStrategy(Enum):
    """Strategies to fix a failure cluster."""
    UPDATE_PROMPT = "update_prompt"
    ADD_FEW_SHOT = "add_few_shot_examples"
    BUILD_TOOL = "build_new_tool"
    UPDATE_KNOWLEDGE = "update_knowledge_base"
    CHANGE_POLICY = "change_escalation_policy"
    OUT_OF_SCOPE = "mark_out_of_scope"

class Priority(Enum):
    """Fix priority levels."""
    CRITICAL = "critical"  # Breaks core functionality
    HIGH = "high"  # Affects many users
    MEDIUM = "medium"  # Noticeable but workarounds exist
    LOW = "low"  # Edge case

@dataclass
class ClusterReview:
    """Human review annotations for a failure cluster."""
    cluster_id: int
    cluster_size: int
    
    # Annotations
    root_cause: Optional[RootCause] = None
    fix_strategy: Optional[FixStrategy] = None
    priority: Optional[Priority] = None
    
    # Details
    notes: str = ""
    reviewed_by: str = ""
    reviewed_at: Optional[str] = None
    
    # Implementation tracking
    implemented: bool = False
    implementation_notes: str = ""

class HITLReviewInterface:
    """Simple interface for human reviewers to annotate clusters."""
    
    def __init__(self, reviews_path: str = "/tmp/cluster_reviews.json"):
        self.reviews_path = reviews_path
        self.reviews: Dict[int, ClusterReview] = {}
        self._load_reviews()
    
    def show_cluster(self, cluster_id: int, summary: Dict[str, any]):
        """Display cluster details for review."""
        print("="*70)
        print(f"CLUSTER {cluster_id} REVIEW")
        print("="*70)
        
        print(f"\n📊 Size: {summary['size']} failures")
        print(f"Failure types: {summary['failure_types']}")
        
        print(f"\n📝 Sample Failures:")
        for i, failure in enumerate(summary['failures'][:3], 1):
            print(f"\n  {i}. User: '{failure.user_input}'")
            if failure.agent_output:
                print(f"     Agent: '{failure.agent_output}'")
            if failure.error_message:
                print(f"     ❌ Error: {failure.error_message}")
            if failure.escalation_reason:
                print(f"     ⬆️ Escalated: {failure.escalation_reason}")
            if failure.rating:
                print(f"     ⭐ Rating: {failure.rating}/5")
        
        # Show existing review if available
        if cluster_id in self.reviews:
            review = self.reviews[cluster_id]
            print(f"\n📝 Existing Review:")
            print(f"  Root cause: {review.root_cause.value if review.root_cause else 'Not set'}")
            print(f"  Fix strategy: {review.fix_strategy.value if review.fix_strategy else 'Not set'}")
            print(f"  Priority: {review.priority.value if review.priority else 'Not set'}")
            if review.notes:
                print(f"  Notes: {review.notes}")
        
        print("\n" + "="*70)
    
    def annotate_cluster(self,
                        cluster_id: int,
                        cluster_size: int,
                        root_cause: RootCause,
                        fix_strategy: FixStrategy,
                        priority: Priority,
                        notes: str = "",
                        reviewed_by: str = "reviewer"):
        """Annotate a cluster with review decisions."""
        from datetime import datetime
        
        review = ClusterReview(
            cluster_id=cluster_id,
            cluster_size=cluster_size,
            root_cause=root_cause,
            fix_strategy=fix_strategy,
            priority=priority,
            notes=notes,
            reviewed_by=reviewed_by,
            reviewed_at=datetime.now().isoformat()
        )
        
        self.reviews[cluster_id] = review
        self._save_reviews()
        
        print(f"✅ Cluster {cluster_id} annotated:")
        print(f"  • Root cause: {root_cause.value}")
        print(f"  • Fix: {fix_strategy.value}")
        print(f"  • Priority: {priority.value}")
        
        return review
    
    def mark_implemented(self, cluster_id: int, implementation_notes: str = ""):
        """Mark a cluster fix as implemented."""
        if cluster_id not in self.reviews:
            raise ValueError(f"No review found for cluster {cluster_id}")
        
        self.reviews[cluster_id].implemented = True
        self.reviews[cluster_id].implementation_notes = implementation_notes
        self._save_reviews()
        
        print(f"✅ Cluster {cluster_id} marked as implemented")
    
    def get_pending_reviews(self, min_priority: Priority = Priority.LOW) -> List[ClusterReview]:
        """Get clusters needing fixes, filtered by priority."""
        priority_order = {Priority.CRITICAL: 0, Priority.HIGH: 1, Priority.MEDIUM: 2, Priority.LOW: 3}
        min_level = priority_order[min_priority]
        
        pending = [
            review for review in self.reviews.values()
            if not review.implemented
            and review.priority
            and priority_order[review.priority] <= min_level
        ]
        
        # Sort by priority
        pending.sort(key=lambda r: priority_order.get(r.priority, 999))
        return pending
    
    def _save_reviews(self):
        """Save reviews to disk."""
        data = {
            str(cid): {
                "cluster_id": review.cluster_id,
                "cluster_size": review.cluster_size,
                "root_cause": review.root_cause.value if review.root_cause else None,
                "fix_strategy": review.fix_strategy.value if review.fix_strategy else None,
                "priority": review.priority.value if review.priority else None,
                "notes": review.notes,
                "reviewed_by": review.reviewed_by,
                "reviewed_at": review.reviewed_at,
                "implemented": review.implemented,
                "implementation_notes": review.implementation_notes
            }
            for cid, review in self.reviews.items()
        }
        
        with open(self.reviews_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_reviews(self):
        """Load reviews from disk."""
        try:
            with open(self.reviews_path, 'r') as f:
                data = json.load(f)
            
            for cid_str, review_data in data.items():
                review = ClusterReview(
                    cluster_id=review_data["cluster_id"],
                    cluster_size=review_data["cluster_size"],
                    root_cause=RootCause(review_data["root_cause"]) if review_data["root_cause"] else None,
                    fix_strategy=FixStrategy(review_data["fix_strategy"]) if review_data["fix_strategy"] else None,
                    priority=Priority(review_data["priority"]) if review_data["priority"] else None,
                    notes=review_data.get("notes", ""),
                    reviewed_by=review_data.get("reviewed_by", ""),
                    reviewed_at=review_data.get("reviewed_at"),
                    implemented=review_data.get("implemented", False),
                    implementation_notes=review_data.get("implementation_notes", "")
                )
                self.reviews[int(cid_str)] = review
        except FileNotFoundError:
            pass

# Initialize review interface
review_interface = HITLReviewInterface()

print("✅ HITL Review Interface ready")
print("\nReviewers can:")
print("  1. View cluster details")
print("  2. Annotate with root cause & fix strategy")
print("  3. Set priority")
print("  4. Track implementation")

# COMMAND ----------

# DBTITLE 1,Test: Review and Annotate Clusters
# Simulate human reviewer annotating clusters

# Review cluster 0 (the refund escalations)
print("Reviewer examining cluster 0...\n")
review_interface.show_cluster(0, summaries[0])

print("\n" + "-"*70)
print("Reviewer's analysis:")
print("  Both failures are refund-related escalations.")
print("  Root cause: Agent lacks authority to process refunds.")
print("  Fix: Update escalation policy to handle refunds <$50 automatically.")
print("-"*70 + "\n")

# Annotate
review_interface.annotate_cluster(
    cluster_id=0,
    cluster_size=summaries[0]['size'],
    root_cause=RootCause.POLICY_GAP,
    fix_strategy=FixStrategy.CHANGE_POLICY,
    priority=Priority.HIGH,
    notes="Refund escalations are common. Enable auto-refunds for orders <$50 to reduce load on human agents.",
    reviewed_by="alice@example.com"
)

print("\n" + "="*70)
print("\nNow let's review an outlier (database timeout)...\n")

# Review outlier cluster (-1)
if -1 in summaries:
    # Find the database timeout failure
    db_timeout_failures = [
        f for f in summaries[-1]['failures']
        if f.error_message and 'database timeout' in f.error_message.lower()
    ]
    
    if db_timeout_failures:
        print("Found database timeout failure:")
        failure = db_timeout_failures[0]
        print(f"  User: '{failure.user_input}'")
        print(f"  Error: {failure.error_message}")
        print("\nReviewer's analysis:")
        print("  This is a tool/infrastructure issue, not an agent problem.")
        print("  Fix: Add retry logic to order lookup tool.")
        print("\nMarking as MEDIUM priority (affects availability but has fallback).\n")
        
        # Create a mini summary for this outlier
        outlier_summary = {
            'size': 1,
            'failure_types': {'exception': 1},
            'failures': [failure]
        }
        
        # We'll use cluster ID 999 for individual outlier tracking
        review_interface.annotate_cluster(
            cluster_id=999,
            cluster_size=1,
            root_cause=RootCause.TOOL_ERROR,
            fix_strategy=FixStrategy.BUILD_TOOL,
            priority=Priority.MEDIUM,
            notes="Add exponential backoff retry to database calls in order_lookup tool",
            reviewed_by="alice@example.com"
        )

print("\n" + "="*70)
print("\n📊 Review Summary:")
print("="*70)

pending = review_interface.get_pending_reviews()
for review in pending:
    priority_emoji = "🔴" if review.priority == Priority.CRITICAL else "🟡" if review.priority == Priority.HIGH else "🟠" if review.priority == Priority.MEDIUM else "⚪"
    print(f"{priority_emoji} Cluster {review.cluster_id}: {review.fix_strategy.value} ({review.priority.value})")
    print(f"   Notes: {review.notes}")
    print()

print("✅ HITL review complete!")
print("\n➡️  Next: Auto-generate refinements based on patterns")

# COMMAND ----------

# DBTITLE 1,4. Automated Refinement: Pattern-Based Improvements
# MAGIC %md
# MAGIC ## 4. Automated Refinement: Pattern-Based Improvements
# MAGIC
# MAGIC Once humans identify **what** to fix, we can often **automate the how**. Not every fix requires manual prompt engineering—many follow predictable patterns.
# MAGIC
# MAGIC ### Automatable Refinements
# MAGIC
# MAGIC | Fix Strategy | Automation Approach |
# MAGIC |--------------|---------------------|
# MAGIC | **Add few-shot examples** | Extract successful responses from similar inputs |
# MAGIC | **Update prompt** | Template-based prompt expansion (add constraints, clarifications) |
# MAGIC | **Add policy rules** | Generate conditional instructions from failure patterns |
# MAGIC | **Improve tool usage** | Inject examples of correct tool invocations |
# MAGIC
# MAGIC ### Example: Refund Escalation Pattern
# MAGIC
# MAGIC Review says: *"Enable auto-refunds for orders <$50"*
# MAGIC
# MAGIC Automated refinement:
# MAGIC ```python
# MAGIC # Before
# MAGIC prompt = "Handle customer support requests..."
# MAGIC
# MAGIC # After (automatically generated)
# MAGIC prompt = """Handle customer support requests...
# MAGIC
# MAGIC Refund Policy:
# MAGIC - Orders <$50: Process refund automatically using process_refund tool
# MAGIC - Orders >=$50: Escalate to human specialist
# MAGIC - Damaged items: Always ask for photo verification first
# MAGIC """
# MAGIC ```
# MAGIC
# MAGIC Let's build the refinement engine.

# COMMAND ----------

# DBTITLE 1,Automated Refinement Implementation
from typing import List, Dict, Tuple
import re

class PromptRefiner:
    """Automatically generates prompt improvements from failure patterns."""
    
    def __init__(self, base_prompt: str):
        self.base_prompt = base_prompt
        self.refinements: List[str] = []
    
    def add_policy_rule(self, rule: str):
        """Add a policy rule to the prompt."""
        policy_section = f"\n\nPolicy Rule:\n- {rule}"
        self.refinements.append(policy_section)
        return self
    
    def add_few_shot_example(self, user_input: str, correct_output: str):
        """Add a few-shot example."""
        example = f"""\n\nExample:
User: {user_input}
Assistant: {correct_output}"""
        self.refinements.append(example)
        return self
    
    def add_tool_guidance(self, tool_name: str, when_to_use: str, example: Dict[str, any]):
        """Add guidance for tool usage."""
        guidance = f"""\n\nTool: {tool_name}
When to use: {when_to_use}
Example call: {example}"""
        self.refinements.append(guidance)
        return self
    
    def add_constraint(self, constraint: str):
        """Add a behavioral constraint."""
        constraint_text = f"\n\nConstraint: {constraint}"
        self.refinements.append(constraint_text)
        return self
    
    def generate_refined_prompt(self) -> str:
        """Combine base prompt with all refinements."""
        if not self.refinements:
            return self.base_prompt
        
        refined = self.base_prompt
        
        # Group refinements by type
        policies = [r for r in self.refinements if "Policy Rule:" in r]
        examples = [r for r in self.refinements if "Example:" in r]
        tools = [r for r in self.refinements if "Tool:" in r]
        constraints = [r for r in self.refinements if "Constraint:" in r]
        
        # Add sections in logical order
        if constraints:
            refined += "\n\n## Constraints"
            for c in constraints:
                refined += c.replace("\n\nConstraint:", "\n-")
        
        if policies:
            refined += "\n\n## Policies"
            for p in policies:
                refined += p.replace("\n\nPolicy Rule:", "")
        
        if tools:
            refined += "\n\n## Tool Usage Guidelines"
            refined += "".join(tools)
        
        if examples:
            refined += "\n\n## Examples"
            refined += "".join(examples)
        
        return refined
    
    def show_diff(self):
        """Show what changed between base and refined prompts."""
        refined = self.generate_refined_prompt()
        
        print("="*70)
        print("PROMPT REFINEMENT")
        print("="*70)
        
        print("\n📋 BASE PROMPT:")
        print("-"*70)
        print(self.base_prompt)
        
        print("\n➕ ADDITIONS:")
        print("-"*70)
        additions = refined[len(self.base_prompt):]
        print(additions if additions else "(none)")
        
        print("\n✅ REFINED PROMPT:")
        print("-"*70)
        print(refined)
        print("="*70)

class AutomatedRefinementEngine:
    """Generates refinements from cluster reviews."""
    
    def __init__(self):
        self.templates = self._init_templates()
    
    def _init_templates(self) -> Dict[FixStrategy, callable]:
        """Templates for each fix strategy."""
        return {
            FixStrategy.CHANGE_POLICY: self._generate_policy_update,
            FixStrategy.ADD_FEW_SHOT: self._generate_few_shot,
            FixStrategy.UPDATE_PROMPT: self._generate_constraint,
            FixStrategy.BUILD_TOOL: self._generate_tool_guidance
        }
    
    def refine_from_review(self, 
                          review: ClusterReview,
                          cluster_summary: Dict[str, any],
                          base_prompt: str) -> PromptRefiner:
        """Generate refinement from human review."""
        refiner = PromptRefiner(base_prompt)
        
        if review.fix_strategy not in self.templates:
            print(f"No template for {review.fix_strategy.value}")
            return refiner
        
        # Generate refinement using template
        template_fn = self.templates[review.fix_strategy]
        template_fn(refiner, review, cluster_summary)
        
        return refiner
    
    def _generate_policy_update(self, refiner: PromptRefiner, 
                               review: ClusterReview, 
                               summary: Dict[str, any]):
        """Generate policy rule from review."""
        # Extract pattern from failures
        if "refund" in review.notes.lower():
            # Parse the threshold from notes if present
            match = re.search(r'\$?(\d+)', review.notes)
            threshold = match.group(1) if match else "50"
            
            refiner.add_policy_rule(
                f"For refund requests on orders <${threshold}: Use process_refund tool immediately"
            )
            refiner.add_policy_rule(
                f"For refund requests on orders >=${threshold}: Escalate to human specialist"
            )
            refiner.add_policy_rule(
                "For damaged items: Request photo verification before processing refund"
            )
    
    def _generate_few_shot(self, refiner: PromptRefiner,
                          review: ClusterReview,
                          summary: Dict[str, any]):
        """Add few-shot examples from successful similar cases."""
        # In production, fetch successful examples from vector DB
        # For demo, use synthetic examples
        
        sample_failure = summary['failures'][0]
        
        if "refund" in sample_failure.user_input.lower():
            refiner.add_few_shot_example(
                user_input="I need a refund for order #54321, it was only $25",
                correct_output="I've processed your refund of $25 for order #54321. The credit will appear in 3-5 business days."
            )
    
    def _generate_constraint(self, refiner: PromptRefiner,
                            review: ClusterReview,
                            summary: Dict[str, any]):
        """Add behavioral constraints."""
        if review.root_cause == RootCause.AMBIGUOUS_INPUT:
            refiner.add_constraint(
                "If user input is ambiguous, ask clarifying questions before acting"
            )
        elif review.root_cause == RootCause.SCOPE_EXCEEDED:
            refiner.add_constraint(
                "If request is outside your scope, politely explain limitations and suggest alternatives"
            )
    
    def _generate_tool_guidance(self, refiner: PromptRefiner,
                               review: ClusterReview,
                               summary: Dict[str, any]):
        """Add tool usage examples."""
        if "database" in review.notes.lower() and "retry" in review.notes.lower():
            refiner.add_tool_guidance(
                tool_name="order_lookup",
                when_to_use="Retrieving order status, details, or history",
                example={"tool": "order_lookup", "args": {"order_id": "12345", "retry": True}}
            )

# Initialize
refinement_engine = AutomatedRefinementEngine()

# Demo base prompt
base_prompt = """You are a helpful customer support agent for an e-commerce platform.

Your role:
- Answer questions about orders, returns, and products
- Use available tools to look up information
- Be polite and helpful
- Escalate complex issues to human agents when needed

Available tools:
- order_lookup: Get order status
- product_search: Find product information
- process_refund: Issue refunds
- escalate_to_human: Transfer to specialist"""

print("✅ Automated Refinement Engine ready")
print("\nCan generate refinements for:")
print("  • Policy updates")
print("  • Few-shot examples")
print("  • Tool usage guidance")
print("  • Behavioral constraints")

# COMMAND ----------

# DBTITLE 1,Test: Generate Automated Refinements
# Generate refinement for the refund policy cluster
review = review_interface.reviews[0]  # The refund escalation cluster

print("Generating automated refinement for cluster 0 (refund escalations)...\n")

refiner = refinement_engine.refine_from_review(
    review=review,
    cluster_summary=summaries[0],
    base_prompt=base_prompt
)

# Show the diff
refiner.show_diff()

print("\n✅ Refinement generated!")
print("\n💡 What changed:")
print("  • Added refund policy rules ($50 threshold)")
print("  • Specified when to use process_refund tool vs escalate")
print("  • Added damaged item verification requirement")
print("\n➡️  Next: Validate this refinement in shadow mode before serving users")

# COMMAND ----------

# DBTITLE 1,5. Shadow Deployment: Safe Validation
# MAGIC %md
# MAGIC ## 5. Shadow Deployment: Safe Validation
# MAGIC
# MAGIC **Never deploy untested changes to production.** Shadow deployment lets you validate improvements **without risk**:
# MAGIC
# MAGIC - Run new variant alongside production
# MAGIC - Compare outputs but don't serve to users
# MAGIC - Collect metrics on accuracy, safety, latency
# MAGIC - Only promote if statistically better
# MAGIC
# MAGIC ### Shadow Mode Workflow
# MAGIC
# MAGIC ```
# MAGIC User request → Production agent responds (served to user)
# MAGIC               → Shadow agent also responds (logged, not served)
# MAGIC               
# MAGIC               Compare:
# MAGIC                 • Success rates
# MAGIC                 • User ratings (simulated)
# MAGIC                 • Policy compliance
# MAGIC                 • Latency
# MAGIC               
# MAGIC               Decision:
# MAGIC                 ✓ Shadow >=better  → Promote to A/B test
# MAGIC                 ✗ Shadow worse     → Back to refinement
# MAGIC ```
# MAGIC
# MAGIC ### Success Criteria
# MAGIC
# MAGIC - **No regressions**: Shadow must not be worse on any critical metric
# MAGIC - **Statistical confidence**: Difference must be significant (not random)
# MAGIC - **Volume threshold**: Test on enough samples (e.g., 100+)
# MAGIC - **Safety check**: Zero policy violations
# MAGIC
# MAGIC Let's build the shadow system.

# COMMAND ----------

# DBTITLE 1,Shadow Deployment Implementation
# MAGIC %pip install -q scipy
# MAGIC
# MAGIC from dataclasses import dataclass
# MAGIC from typing import Optional, List, Dict
# MAGIC from datetime import datetime
# MAGIC import time
# MAGIC import numpy as np
# MAGIC from scipy import stats
# MAGIC
# MAGIC @dataclass
# MAGIC class ShadowResult:
# MAGIC     """Result from shadow deployment test."""
# MAGIC     request_id: str
# MAGIC     user_input: str
# MAGIC     
# MAGIC     production_output: str
# MAGIC     shadow_output: str
# MAGIC     
# MAGIC     production_success: bool
# MAGIC     shadow_success: bool
# MAGIC     
# MAGIC     production_latency_ms: int
# MAGIC     shadow_latency_ms: int
# MAGIC     
# MAGIC     # Optional: simulated ratings or human eval
# MAGIC     production_rating: Optional[int] = None
# MAGIC     shadow_rating: Optional[int] = None
# MAGIC     
# MAGIC     timestamp: datetime = None
# MAGIC     
# MAGIC     def __post_init__(self):
# MAGIC         if self.timestamp is None:
# MAGIC             self.timestamp = datetime.now()
# MAGIC
# MAGIC class ShadowDeployment:
# MAGIC     """Validates new agent variants safely without serving users."""
# MAGIC     
# MAGIC     def __init__(self, production_prompt: str, shadow_prompt: str):
# MAGIC         self.production_prompt = production_prompt
# MAGIC         self.shadow_prompt = shadow_prompt
# MAGIC         self.results: List[ShadowResult] = []
# MAGIC     
# MAGIC     def run_shadow_request(self, 
# MAGIC                           user_input: str,
# MAGIC                           production_fn: callable,
# MAGIC                           shadow_fn: callable) -> ShadowResult:
# MAGIC         """Run both agents and compare."""
# MAGIC         request_id = f"shadow_{len(self.results)}"
# MAGIC         
# MAGIC         # Production (actually served)
# MAGIC         prod_start = time.time()
# MAGIC         try:
# MAGIC             prod_output = production_fn(user_input, self.production_prompt)
# MAGIC             prod_success = True
# MAGIC         except Exception as e:
# MAGIC             prod_output = f"Error: {e}"
# MAGIC             prod_success = False
# MAGIC         prod_latency = int((time.time() - prod_start) * 1000)
# MAGIC         
# MAGIC         # Shadow (logged only)
# MAGIC         shadow_start = time.time()
# MAGIC         try:
# MAGIC             shadow_output = shadow_fn(user_input, self.shadow_prompt)
# MAGIC             shadow_success = True
# MAGIC         except Exception as e:
# MAGIC             shadow_output = f"Error: {e}"
# MAGIC             shadow_success = False
# MAGIC         shadow_latency = int((time.time() - shadow_start) * 1000)
# MAGIC         
# MAGIC         result = ShadowResult(
# MAGIC             request_id=request_id,
# MAGIC             user_input=user_input,
# MAGIC             production_output=prod_output,
# MAGIC             shadow_output=shadow_output,
# MAGIC             production_success=prod_success,
# MAGIC             shadow_success=shadow_success,
# MAGIC             production_latency_ms=prod_latency,
# MAGIC             shadow_latency_ms=shadow_latency
# MAGIC         )
# MAGIC         
# MAGIC         self.results.append(result)
# MAGIC         return result
# MAGIC     
# MAGIC     def add_ratings(self, request_id: str, prod_rating: int, shadow_rating: int):
# MAGIC         """Add simulated or human ratings to a result."""
# MAGIC         for result in self.results:
# MAGIC             if result.request_id == request_id:
# MAGIC                 result.production_rating = prod_rating
# MAGIC                 result.shadow_rating = shadow_rating
# MAGIC                 break
# MAGIC     
# MAGIC     def analyze(self) -> Dict[str, any]:
# MAGIC         """Compare shadow vs production statistically."""
# MAGIC         if len(self.results) < 10:
# MAGIC             return {
# MAGIC                 "sufficient_data": False,
# MAGIC                 "message": f"Need at least 10 samples, have {len(self.results)}"
# MAGIC             }
# MAGIC         
# MAGIC         # Success rates
# MAGIC         prod_success_rate = sum(r.production_success for r in self.results) / len(self.results)
# MAGIC         shadow_success_rate = sum(r.shadow_success for r in self.results) / len(self.results)
# MAGIC         
# MAGIC         # Latency
# MAGIC         prod_latencies = [r.production_latency_ms for r in self.results if r.production_success]
# MAGIC         shadow_latencies = [r.shadow_latency_ms for r in self.results if r.shadow_success]
# MAGIC         
# MAGIC         prod_p50 = np.median(prod_latencies) if prod_latencies else 0
# MAGIC         shadow_p50 = np.median(shadow_latencies) if shadow_latencies else 0
# MAGIC         
# MAGIC         # Ratings (if available)
# MAGIC         prod_ratings = [r.production_rating for r in self.results if r.production_rating is not None]
# MAGIC         shadow_ratings = [r.shadow_rating for r in self.results if r.shadow_rating is not None]
# MAGIC         
# MAGIC         analysis = {
# MAGIC             "sufficient_data": True,
# MAGIC             "sample_size": len(self.results),
# MAGIC             "success_rate": {
# MAGIC                 "production": prod_success_rate,
# MAGIC                 "shadow": shadow_success_rate,
# MAGIC                 "difference": shadow_success_rate - prod_success_rate
# MAGIC             },
# MAGIC             "latency_p50": {
# MAGIC                 "production": prod_p50,
# MAGIC                 "shadow": shadow_p50,
# MAGIC                 "difference": shadow_p50 - prod_p50
# MAGIC             }
# MAGIC         }
# MAGIC         
# MAGIC         # Rating comparison (if available)
# MAGIC         if prod_ratings and shadow_ratings and len(prod_ratings) >= 10:
# MAGIC             prod_mean = np.mean(prod_ratings)
# MAGIC             shadow_mean = np.mean(shadow_ratings)
# MAGIC             
# MAGIC             # t-test for statistical significance
# MAGIC             t_stat, p_value = stats.ttest_rel(shadow_ratings, prod_ratings)
# MAGIC             
# MAGIC             analysis["ratings"] = {
# MAGIC                 "production_mean": prod_mean,
# MAGIC                 "shadow_mean": shadow_mean,
# MAGIC                 "difference": shadow_mean - prod_mean,
# MAGIC                 "p_value": p_value,
# MAGIC                 "significant": p_value < 0.05,
# MAGIC                 "significant_improvement": (p_value < 0.05) and (shadow_mean > prod_mean)
# MAGIC             }
# MAGIC         
# MAGIC         # Recommendation
# MAGIC         analysis["recommendation"] = self._make_recommendation(analysis)
# MAGIC         
# MAGIC         return analysis
# MAGIC     
# MAGIC     def _make_recommendation(self, analysis: Dict) -> str:
# MAGIC         """Decide whether to promote shadow variant."""
# MAGIC         # Check for regressions
# MAGIC         if analysis["success_rate"]["shadow"] < analysis["success_rate"]["production"] * 0.95:
# MAGIC             return "REJECT: Shadow has significantly lower success rate"
# MAGIC         
# MAGIC         if analysis["latency_p50"]["shadow"] > analysis["latency_p50"]["production"] * 1.5:
# MAGIC             return "REJECT: Shadow is too slow (>50% slower)"
# MAGIC         
# MAGIC         # Check for improvements
# MAGIC         if "ratings" in analysis:
# MAGIC             if analysis["ratings"]["significant_improvement"]:
# MAGIC                 return "PROMOTE: Shadow significantly better (p < 0.05)"
# MAGIC             elif analysis["ratings"]["difference"] > 0.2:
# MAGIC                 return "PROMOTE: Shadow shows meaningful improvement"
# MAGIC             else:
# MAGIC                 return "NEUTRAL: No significant difference, keep testing"
# MAGIC         else:
# MAGIC             # No rating data, check other signals
# MAGIC             if analysis["success_rate"]["difference"] > 0.05:
# MAGIC                 return "PROMOTE: Shadow has higher success rate"
# MAGIC             else:
# MAGIC                 return "NEUTRAL: Need user ratings to decide"
# MAGIC     
# MAGIC     def print_report(self):
# MAGIC         """Print analysis report."""
# MAGIC         analysis = self.analyze()
# MAGIC         
# MAGIC         print("="*70)
# MAGIC         print("SHADOW DEPLOYMENT ANALYSIS")
# MAGIC         print("="*70)
# MAGIC         
# MAGIC         if not analysis["sufficient_data"]:
# MAGIC             print(f"\n⚠️  {analysis['message']}")
# MAGIC             return
# MAGIC         
# MAGIC         print(f"\n📊 Sample size: {analysis['sample_size']} requests")
# MAGIC         
# MAGIC         print("\n🎯 Success Rates:")
# MAGIC         print(f"  Production: {analysis['success_rate']['production']:.1%}")
# MAGIC         print(f"  Shadow:     {analysis['success_rate']['shadow']:.1%}")
# MAGIC         print(f"  Difference: {analysis['success_rate']['difference']:+.1%}")
# MAGIC         
# MAGIC         print("\n⏱️  Latency (P50):")
# MAGIC         print(f"  Production: {analysis['latency_p50']['production']:.0f}ms")
# MAGIC         print(f"  Shadow:     {analysis['latency_p50']['shadow']:.0f}ms")
# MAGIC         print(f"  Difference: {analysis['latency_p50']['difference']:+.0f}ms")
# MAGIC         
# MAGIC         if "ratings" in analysis:
# MAGIC             print("\n⭐ User Ratings:")
# MAGIC             print(f"  Production: {analysis['ratings']['production_mean']:.2f}/5")
# MAGIC             print(f"  Shadow:     {analysis['ratings']['shadow_mean']:.2f}/5")
# MAGIC             print(f"  Difference: {analysis['ratings']['difference']:+.2f}")
# MAGIC             print(f"  p-value:    {analysis['ratings']['p_value']:.4f}")
# MAGIC             
# MAGIC             if analysis['ratings']['significant']:
# MAGIC                 print("  ✅ Statistically significant")
# MAGIC             else:
# MAGIC                 print("  ⚪ Not statistically significant")
# MAGIC         
# MAGIC         print("\n" + "="*70)
# MAGIC         print(f"RECOMMENDATION: {analysis['recommendation']}")
# MAGIC         print("="*70)
# MAGIC
# MAGIC print("✅ Shadow Deployment system ready")
# MAGIC print("\nEnables safe validation:")
# MAGIC print("  • Run shadow variant alongside production")
# MAGIC print("  • Compare metrics without user risk")
# MAGIC print("  • Statistical significance testing")
# MAGIC print("  • Automated promotion decisions")

# COMMAND ----------

# DBTITLE 1,Test: Shadow Deployment
# Simulate agents (mock LLM calls for demo)
def mock_agent(user_input: str, prompt: str) -> str:
    """Simulated agent response."""
    time.sleep(0.01)  # Simulate API latency
    
    # Production logic (no refund policy)
    if "production" in prompt.lower() or "refund" not in prompt.lower():
        if "refund" in user_input.lower():
            return "Let me transfer you to a specialist who can help with refunds."
        return "I'd be happy to help you with that."
    
    # Shadow logic (with refund policy)
    if "refund" in user_input.lower():
        # Simulate $50 threshold
        if "$25" in user_input or "$30" in user_input:
            return "I've processed your refund. The credit will appear in 3-5 business days."
        else:
            return "Let me transfer you to a specialist who can help with larger refunds."
    return "I'd be happy to help you with that."

# Initialize shadow deployment
refined_prompt = refiner.generate_refined_prompt()
shadow = ShadowDeployment(
    production_prompt=base_prompt,
    shadow_prompt=refined_prompt
)

# Simulate 15 requests (mix of refund and non-refund)
test_inputs = [
    "I need a refund for my $25 order",
    "Can you help me track my package?",
    "I want a refund for order #12345, it was $30",
    "What's your return policy?",
    "Refund request for $120 order please",
    "How do I update my shipping address?",
    "I need a refund, the item was only $25",
    "What payment methods do you accept?",
    "Can I get a refund for $28?",
    "Tell me about your warranty",
    "Refund for my $25 purchase",
    "Do you ship internationally?",
    "I need a $30 refund",
    "What sizes do you have?",
    "Refund my $25 order please"
]

print("Running shadow deployment test...\n")
for i, user_input in enumerate(test_inputs, 1):
    result = shadow.run_shadow_request(
        user_input=user_input,
        production_fn=lambda inp, prompt: mock_agent(inp, "production: " + prompt),
        shadow_fn=lambda inp, prompt: mock_agent(inp, "shadow: " + prompt)
    )
    
    # Simulate ratings
    # Production: low ratings for refunds (always escalates)
    # Shadow: high ratings for small refunds (processes automatically)
    if "refund" in user_input.lower():
        if "$25" in user_input or "$30" in user_input or "$28" in user_input:
            prod_rating = 2  # User frustrated by unnecessary escalation
            shadow_rating = 5  # User happy with immediate processing
        else:
            prod_rating = 4  # Escalation appropriate for large refunds
            shadow_rating = 4
    else:
        prod_rating = 5
        shadow_rating = 5
    
    shadow.add_ratings(result.request_id, prod_rating, shadow_rating)
    
    if i % 5 == 0:
        print(f"  Processed {i}/{len(test_inputs)} requests...")

print(f"\n✅ Completed {len(test_inputs)} shadow requests\n")

# Analyze results
shadow.print_report()

print("\n💡 Insights:")
print("  • Shadow variant handles small refunds automatically")
print("  • Users happier (higher ratings) for simple refund cases")
print("  • No regressions on success rate or latency")
print("  • Statistically significant improvement → Ready for A/B test!")
print("\n➡️  Next: Deploy as A/B test to measure real-user impact")

# COMMAND ----------

# DBTITLE 1,6-8. Advanced Deployment Strategies
# MAGIC %md
# MAGIC ## 6-8. Advanced Deployment Strategies
# MAGIC
# MAGIC After shadow validation, you need **real-user testing** to measure actual impact. Three complementary strategies:
# MAGIC
# MAGIC ### 6. A/B Testing: Controlled Experiments
# MAGIC
# MAGIC - **What**: Split traffic 50/50 between control (A) and treatment (B)
# MAGIC - **When**: You have ONE new variant to test
# MAGIC - **Pro**: Simple, clear comparison
# MAGIC - **Con**: Tests only one alternative at a time
# MAGIC
# MAGIC **Use case**: "Does the new refund policy improve ratings?"
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 7. Bayesian Bandit: Adaptive Exploration
# MAGIC
# MAGIC - **What**: Dynamically shift traffic toward better-performing variants
# MAGIC - **When**: You have MULTIPLE variants to test simultaneously
# MAGIC - **Pro**: Minimizes regret (fewer users see bad variants)
# MAGIC - **Con**: More complex, needs tuning
# MAGIC
# MAGIC **Use case**: "Which of 3 prompt styles works best? Learn fast and minimize poor experiences."
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 8. In-Context Learning: Real-Time Adaptation
# MAGIC
# MAGIC - **What**: Agent learns from recent successful examples during conversation
# MAGIC - **When**: You want agents to improve within a single session
# MAGIC - **Pro**: No deployment cycle, adapts instantly
# MAGIC - **Con**: Limited to current context window
# MAGIC
# MAGIC **Use case**: "User corrects agent → agent immediately applies correction to next response."
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC Let's implement each.

# COMMAND ----------

# DBTITLE 1,A/B Testing + Bayesian Bandit Implementations
import random
from collections import defaultdict
import numpy as np
from scipy import stats

# ============================================================================
# A/B TESTING FRAMEWORK
# ============================================================================

class ABTest:
    """Simple A/B testing framework for agent variants."""
    
    def __init__(self, variant_a_prompt: str, variant_b_prompt: str, split_ratio: float = 0.5):
        self.variant_a_prompt = variant_a_prompt
        self.variant_b_prompt = variant_b_prompt
        self.split_ratio = split_ratio
        
        self.results = defaultdict(list)  # {'a': [ratings], 'b': [ratings]}
        self.assignments = {}  # session_id -> variant
    
    def assign_variant(self, session_id: str) -> str:
        """Assign user to variant A or B (sticky by session)."""
        if session_id in self.assignments:
            return self.assignments[session_id]
        
        variant = 'a' if random.random() < self.split_ratio else 'b'
        self.assignments[session_id] = variant
        return variant
    
    def get_prompt(self, session_id: str) -> tuple:
        """Get prompt for user's assigned variant."""
        variant = self.assign_variant(session_id)
        prompt = self.variant_a_prompt if variant == 'a' else self.variant_b_prompt
        return variant, prompt
    
    def record_result(self, session_id: str, rating: int):
        """Record user rating for their assigned variant."""
        variant = self.assignments.get(session_id)
        if variant:
            self.results[variant].append(rating)
    
    def analyze(self) -> Dict[str, any]:
        """Analyze A/B test results."""
        a_ratings = self.results['a']
        b_ratings = self.results['b']
        
        if len(a_ratings) < 10 or len(b_ratings) < 10:
            return {
                "sufficient_data": False,
                "message": f"Need >=10 samples per variant (A: {len(a_ratings)}, B: {len(b_ratings)})"
            }
        
        a_mean = np.mean(a_ratings)
        b_mean = np.mean(b_ratings)
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(b_ratings, a_ratings)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(a_ratings) + np.var(b_ratings)) / 2)
        cohens_d = (b_mean - a_mean) / pooled_std if pooled_std > 0 else 0
        
        return {
            "sufficient_data": True,
            "sample_sizes": {"a": len(a_ratings), "b": len(b_ratings)},
            "means": {"a": a_mean, "b": b_mean},
            "difference": b_mean - a_mean,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "cohens_d": cohens_d,
            "winner": "b" if b_mean > a_mean and p_value < 0.05 else "a" if a_mean > b_mean and p_value < 0.05 else "tie"
        }

print("✅ A/B Testing Framework ready")

# ============================================================================
# BAYESIAN BANDIT
# ============================================================================

class BayesianBandit:
    """Thompson Sampling bandit for multi-variant testing."""
    
    def __init__(self, variants: Dict[str, str]):
        """Initialize with variant_id -> prompt mapping."""
        self.variants = variants
        
        # Beta distribution parameters for each variant
        # Beta(alpha, beta) models success/failure rates
        self.alpha = {v: 1 for v in variants}  # successes + 1 (prior)
        self.beta = {v: 1 for v in variants}  # failures + 1 (prior)
        
        self.pulls = {v: 0 for v in variants}  # total samples
        self.history = []  # (variant, reward) tuples
    
    def select_variant(self) -> str:
        """Select variant using Thompson Sampling.
        
        Samples from each variant's Beta distribution and picks the max.
        Early on: explores (high uncertainty = wide distributions)
        Later: exploits (converges to best variant)
        """
        samples = {
            v: np.random.beta(self.alpha[v], self.beta[v])
            for v in self.variants
        }
        return max(samples, key=samples.get)
    
    def update(self, variant: str, reward: float):
        """Update beliefs based on observed reward.
        
        reward: 0-1 (e.g., rating/5)
        """
        self.history.append((variant, reward))
        self.pulls[variant] += 1
        
        # Binary success/failure for Beta update
        # Treat ratings >=4 as success
        if reward >= 0.8:  # rating >= 4/5
            self.alpha[variant] += 1
        else:
            self.beta[variant] += 1
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get current performance stats for each variant."""
        stats = {}
        for v in self.variants:
            if self.pulls[v] == 0:
                stats[v] = {"mean": 0.5, "pulls": 0, "win_prob": 1/len(self.variants)}
                continue
            
            # Mean of Beta(alpha, beta) = alpha / (alpha + beta)
            mean = self.alpha[v] / (self.alpha[v] + self.beta[v])
            
            # Monte Carlo estimate of P(this variant is best)
            n_samples = 10000
            samples = np.random.beta(self.alpha[v], self.beta[v], n_samples)
            
            other_samples = []
            for other in self.variants:
                if other != v:
                    other_samples.append(np.random.beta(self.alpha[other], self.beta[other], n_samples))
            
            if other_samples:
                wins = sum(samples > max(other_samples, axis=0)) if len(other_samples) > 1 else sum(samples > other_samples[0])
                win_prob = wins / n_samples
            else:
                win_prob = 1.0
            
            stats[v] = {
                "mean": mean,
                "pulls": self.pulls[v],
                "win_prob": win_prob
            }
        
        return stats
    
    def print_report(self):
        """Print bandit statistics."""
        stats = self.get_statistics()
        
        print("="*70)
        print("BAYESIAN BANDIT STATUS")
        print("="*70)
        
        total_pulls = sum(self.pulls.values())
        print(f"\n📊 Total samples: {total_pulls}\n")
        
        # Sort by win probability
        sorted_variants = sorted(stats.items(), key=lambda x: x[1]['win_prob'], reverse=True)
        
        for i, (variant, vstats) in enumerate(sorted_variants, 1):
            traffic_pct = (vstats['pulls'] / total_pulls * 100) if total_pulls > 0 else 0
            
            emoji = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            
            print(f"{emoji} Variant {variant}:")
            print(f"   Estimated success rate: {vstats['mean']:.1%}")
            print(f"   P(best variant): {vstats['win_prob']:.1%}")
            print(f"   Traffic received: {vstats['pulls']} ({traffic_pct:.0f}%)")
            print()
        
        # Recommendation
        best = sorted_variants[0]
        if best[1]['win_prob'] > 0.95 and best[1]['pulls'] >= 50:
            print(f"✅ RECOMMENDATION: Promote variant {best[0]} to production (>95% confidence)")
        elif total_pulls < 100:
            print("📈 RECOMMENDATION: Continue testing (need more samples)")
        else:
            print(f"📈 RECOMMENDATION: Variant {best[0]} leading but not conclusive yet")
        
        print("="*70)

print("✅ Bayesian Bandit ready")
print("\nBoth systems initialized:")
print("  • A/B Testing: Fixed 50/50 split, clear comparison")
print("  • Bayesian Bandit: Adaptive traffic, minimizes regret")

# COMMAND ----------

# DBTITLE 1,In-Context Learning Implementation
# ============================================================================
# IN-CONTEXT LEARNING: Real-Time Adaptation
# ============================================================================

class InContextLearner:
    """Agent that learns from corrections within the conversation."""
    
    def __init__(self, base_prompt: str, max_examples: int = 5):
        self.base_prompt = base_prompt
        self.max_examples = max_examples
        self.learned_examples = []  # (input, correct_output) pairs
    
    def add_correction(self, user_input: str, incorrect_output: str, correction: str):
        """User corrects agent's response - learn from it."""
        example = {
            "input": user_input,
            "incorrect": incorrect_output,
            "correct": correction
        }
        
        self.learned_examples.append(example)
        
        # Keep only recent examples (LRU)
        if len(self.learned_examples) > self.max_examples:
            self.learned_examples.pop(0)
        
        print(f"✅ Learned from correction (now have {len(self.learned_examples)} examples)")
    
    def add_success(self, user_input: str, successful_output: str, positive_feedback: str = ""):
        """User explicitly confirms a good response - reinforce it."""
        example = {
            "input": user_input,
            "correct": successful_output,
            "feedback": positive_feedback
        }
        
        self.learned_examples.append(example)
        
        if len(self.learned_examples) > self.max_examples:
            self.learned_examples.pop(0)
        
        print(f"✅ Reinforced successful pattern (now have {len(self.learned_examples)} examples)")
    
    def generate_augmented_prompt(self) -> str:
        """Build prompt with learned examples injected."""
        if not self.learned_examples:
            return self.base_prompt
        
        augmented = self.base_prompt
        augmented += "\n\n## Recent Learned Examples\n"
        augmented += "Learn from these recent interactions in this conversation:\n\n"
        
        for i, example in enumerate(self.learned_examples, 1):
            augmented += f"Example {i}:\n"
            augmented += f"User: {example['input']}\n"
            
            if 'incorrect' in example:
                augmented += f"Previous (incorrect): {example['incorrect']}\n"
            
            augmented += f"Correct: {example['correct']}\n"
            
            if example.get('feedback'):
                augmented += f"Why it worked: {example['feedback']}\n"
            
            augmented += "\n"
        
        augmented += "Apply these patterns to similar future requests in this conversation.\n"
        
        return augmented
    
    def simulate_conversation(self, agent_fn: callable, turns: List[Dict[str, str]]):
        """Simulate a conversation with learning.
        
        turns: [{"user": "...", "expected": "...", "correction": "..."}, ...]
        """
        print("="*70)
        print("IN-CONTEXT LEARNING SIMULATION")
        print("="*70)
        
        for i, turn in enumerate(turns, 1):
            print(f"\n--- Turn {i} ---")
            print(f"User: {turn['user']}")
            
            # Get current prompt (with learned examples)
            current_prompt = self.generate_augmented_prompt()
            
            # Generate response
            response = agent_fn(turn['user'], current_prompt)
            print(f"Agent: {response}")
            
            # Check if correction needed
            if 'correction' in turn:
                print(f"User: No, {turn['correction']}")
                self.add_correction(
                    user_input=turn['user'],
                    incorrect_output=response,
                    correction=turn['expected']
                )
            elif response.lower() == turn['expected'].lower():
                print("User: 👍 (implicit approval)")
                if i > 1:  # Don't reinforce if no prior learning
                    self.add_success(turn['user'], response, "User approved")
            else:
                print(f"User: Actually, {turn['expected']}")
                self.add_correction(
                    user_input=turn['user'],
                    incorrect_output=response,
                    correction=turn['expected']
                )
        
        print("\n" + "="*70)
        print(f"\n🎓 Learning complete: {len(self.learned_examples)} examples in context")
        print("\nFinal augmented prompt:")
        print("-"*70)
        print(self.generate_augmented_prompt())
        print("-"*70)

# Initialize
in_context_learner = InContextLearner(base_prompt=base_prompt, max_examples=3)

print("✅ In-Context Learner ready")
print("\nCapabilities:")
print("  • Learn from user corrections during conversation")
print("  • Reinforce successful patterns")
print("  • Auto-inject learned examples into prompt")
print("  • LRU eviction (keeps recent examples)")
print("\n➡️  Enables instant adaptation without redeployment")

# COMMAND ----------

# DBTITLE 1,9. End-to-End Integration
# MAGIC %md
# MAGIC ## 9. End-to-End Integration: Complete Improvement Pipeline
# MAGIC
# MAGIC Now let's connect everything into a **production-ready improvement loop**:
# MAGIC
# MAGIC ```
# MAGIC 🔴 Production Failures
# MAGIC    ↓
# MAGIC 📋 1. Feedback Collector → Log failures to storage
# MAGIC    ↓
# MAGIC 📦 2. Issue Clustering → Find failure patterns
# MAGIC    ↓
# MAGIC 👥 3. HITL Review → Humans annotate clusters
# MAGIC    ↓
# MAGIC ⚙️  4. Auto Refinement → Generate prompt improvements
# MAGIC    ↓
# MAGIC 🕵️ 5. Shadow Deploy → Validate without user risk
# MAGIC    ↓
# MAGIC 🧪 6. A/B Test / Bandit → Measure real impact
# MAGIC    ↓
# MAGIC ✅ Promote to Production
# MAGIC    ↓
# MAGIC 🔁 Continue monitoring... (Loop back to step 1)
# MAGIC ```
# MAGIC
# MAGIC ### Parallel Capabilities
# MAGIC
# MAGIC - **In-Context Learning**: Runs alongside the main loop for instant adaptation
# MAGIC - **Continuous Monitoring**: Always collecting feedback for the next cycle
# MAGIC
# MAGIC ### Key Principles
# MAGIC
# MAGIC 1. **Data-Driven**: Every decision backed by metrics
# MAGIC 2. **Safe**: Multiple validation gates before production
# MAGIC 3. **Automated**: Humans review, system executes
# MAGIC 4. **Continuous**: Never stops improving
# MAGIC
# MAGIC Let's tie it all together with a workflow orchestrator.

# COMMAND ----------

# DBTITLE 1,End-to-End Pipeline Orchestrator
class ImprovementPipeline:
    """Orchestrates the complete improvement loop."""
    
    def __init__(self):
        self.collector = FeedbackCollector()
        self.clusterer = FailureClusterer()
        self.review_interface = HITLReviewInterface()
        self.refinement_engine = AutomatedRefinementEngine()
        self.current_production_prompt = base_prompt
    
    def step_1_collect_failures(self, n_days: int = 7) -> List[FailureRecord]:
        """Load recent failures from production."""
        print(f"📋 STEP 1: Collecting failures from last {n_days} days...")
        failures = self.collector.load_failures()
        print(f"  • Found {len(failures)} failures\n")
        return failures
    
    def step_2_cluster_failures(self, failures: List[FailureRecord]) -> Dict[int, Dict]:
        """Find patterns in failures."""
        print("📦 STEP 2: Clustering failures...")
        self.clusterer.embed_failures(failures)
        self.clusterer.cluster_hdbscan(min_cluster_size=2)
        summaries = self.clusterer.get_cluster_summaries()
        
        n_clusters = len([c for c in summaries if c != -1])
        print(f"  • Found {n_clusters} clusters\n")
        return summaries
    
    def step_3_human_review(self, summaries: Dict[int, Dict]) -> List[ClusterReview]:
        """Present clusters to humans for annotation."""
        print("👥 STEP 3: Human review (simulated)...")
        
        # In production: present to reviewers via UI
        # Here: simulate with auto-annotation for demo
        reviews = []
        for cluster_id, summary in summaries.items():
            if cluster_id == -1:
                continue  # Skip outliers for demo
            
            # Simulate reviewer annotating
            review = self.review_interface.annotate_cluster(
                cluster_id=cluster_id,
                cluster_size=summary['size'],
                root_cause=RootCause.POLICY_GAP,
                fix_strategy=FixStrategy.CHANGE_POLICY,
                priority=Priority.HIGH,
                notes=f"Auto-generated review for cluster {cluster_id}",
                reviewed_by="pipeline@system"
            )
            reviews.append(review)
        
        print(f"  • Annotated {len(reviews)} clusters\n")
        return reviews
    
    def step_4_auto_refine(self, reviews: List[ClusterReview], 
                           summaries: Dict[int, Dict]) -> str:
        """Generate prompt refinements."""
        print("⚙️  STEP 4: Generating automated refinements...")
        
        # Combine refinements from all high-priority reviews
        combined_refiner = PromptRefiner(self.current_production_prompt)
        
        for review in reviews:
            if review.priority in [Priority.CRITICAL, Priority.HIGH]:
                temp_refiner = self.refinement_engine.refine_from_review(
                    review=review,
                    cluster_summary=summaries[review.cluster_id],
                    base_prompt=self.current_production_prompt
                )
                # Merge refinements
                combined_refiner.refinements.extend(temp_refiner.refinements)
        
        refined_prompt = combined_refiner.generate_refined_prompt()
        print(f"  • Generated {len(combined_refiner.refinements)} refinements\n")
        return refined_prompt
    
    def step_5_shadow_deploy(self, refined_prompt: str, n_requests: int = 20) -> str:
        """Validate in shadow mode."""
        print(f"🕵️ STEP 5: Shadow deployment ({n_requests} requests)...")
        
        shadow = ShadowDeployment(
            production_prompt=self.current_production_prompt,
            shadow_prompt=refined_prompt
        )
        
        # Simulate shadow testing (would be real traffic in production)
        test_inputs = ["test input"] * n_requests
        for inp in test_inputs:
            shadow.run_shadow_request(
                user_input=inp,
                production_fn=lambda i, p: "production response",
                shadow_fn=lambda i, p: "shadow response"
            )
            # Simulate ratings
            shadow.add_ratings(f"shadow_{len(shadow.results)-1}", 
                             prod_rating=3, shadow_rating=4)
        
        analysis = shadow.analyze()
        
        if "PROMOTE" in analysis["recommendation"]:
            print("  ✅ Shadow passed validation\n")
            return "PROMOTE"
        else:
            print(f"  ❌ Shadow failed: {analysis['recommendation']}\n")
            return "REJECT"
    
    def step_6_ab_test(self, refined_prompt: str) -> str:
        """Run A/B test with real users."""
        print("🧪 STEP 6: A/B testing (simulated)...")
        
        ab = ABTest(
            variant_a_prompt=self.current_production_prompt,
            variant_b_prompt=refined_prompt
        )
        
        # Simulate A/B test traffic
        for i in range(50):
            session_id = f"session_{i}"
            variant, _ = ab.get_prompt(session_id)
            rating = 5 if variant == 'b' else 4  # Simulate B winning
            ab.record_result(session_id, rating)
        
        analysis = ab.analyze()
        
        if analysis["winner"] == "b" and analysis["significant"]:
            print("  ✅ Variant B wins (statistically significant)\n")
            return "PROMOTE_B"
        else:
            print("  ⚪ No clear winner\n")
            return "KEEP_TESTING"
    
    def step_7_promote(self, refined_prompt: str):
        """Promote refined variant to production."""
        print("✅ STEP 7: Promoting to production...")
        self.current_production_prompt = refined_prompt
        print("  • New prompt deployed\n")
    
    def run_full_cycle(self):
        """Execute complete improvement cycle."""
        print("="*70)
        print("IMPROVEMENT PIPELINE - FULL CYCLE")
        print("="*70 + "\n")
        
        # 1. Collect
        failures = self.step_1_collect_failures()
        
        if not failures:
            print("✅ No failures to process")
            return
        
        # 2. Cluster
        summaries = self.step_2_cluster_failures(failures)
        
        # 3. Review
        reviews = self.step_3_human_review(summaries)
        
        if not reviews:
            print("⚠️  No actionable clusters")
            return
        
        # 4. Refine
        refined_prompt = self.step_4_auto_refine(reviews, summaries)
        
        # 5. Shadow
        shadow_result = self.step_5_shadow_deploy(refined_prompt)
        
        if shadow_result == "REJECT":
            print("❌ Pipeline stopped: Shadow validation failed")
            return
        
        # 6. A/B Test
        ab_result = self.step_6_ab_test(refined_prompt)
        
        if ab_result == "PROMOTE_B":
            # 7. Promote
            self.step_7_promote(refined_prompt)
            print("✅ PIPELINE COMPLETE: Improvement deployed to production!")
        else:
            print("🔁 PIPELINE COMPLETE: Continue A/B testing")
        
        print("\n" + "="*70)

# Initialize and run
pipeline = ImprovementPipeline()

print("🚀 Improvement Pipeline initialized")
print("\nComplete workflow:")
print("  1. 📋 Collect failures")
print("  2. 📦 Cluster patterns")
print("  3. 👥 Human review")
print("  4. ⚙️  Auto-generate refinements")
print("  5. 🕵️ Shadow deploy")
print("  6. 🧪 A/B test")
print("  7. ✅ Promote to production")
print("\n🎉 ALL SYSTEMS READY")
print("\n➡️  Run: pipeline.run_full_cycle()")

# COMMAND ----------

# DBTITLE 1,Summary & Production Deployment Checklist
# MAGIC %md
# MAGIC ## Summary & Production Deployment Checklist
# MAGIC
# MAGIC ✅ **You now have a complete improvement loop implementation!**
# MAGIC
# MAGIC ### What We Built
# MAGIC
# MAGIC | Component | Purpose | Key Takeaway |
# MAGIC |-----------|---------|-------------|
# MAGIC | **Feedback Collector** | Capture failures from production | Structured logging is the foundation |
# MAGIC | **Issue Clustering** | Find patterns in failures | Embeddings reveal hidden similarities |
# MAGIC | **HITL Review** | Human judgment on priorities | Experts decide *what*, automation handles *how* |
# MAGIC | **Auto Refinement** | Generate prompt improvements | Template-based fixes for common patterns |
# MAGIC | **Shadow Deployment** | Safe validation | Never deploy untested changes |
# MAGIC | **A/B Testing** | Measure real impact | Statistics over intuition |
# MAGIC | **Bayesian Bandit** | Adaptive exploration | Minimize regret while learning |
# MAGIC | **In-Context Learning** | Real-time adaptation | Learn within the conversation |
# MAGIC | **End-to-End Pipeline** | Orchestrate everything | Automation with human oversight |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Production Deployment Checklist
# MAGIC
# MAGIC Before deploying this to your production agentic system:
# MAGIC
# MAGIC #### ✅ Infrastructure
# MAGIC
# MAGIC - [ ] **Persistent storage** for failure logs (Delta tables, S3, database)
# MAGIC - [ ] **Embedding service** (sentence-transformers endpoint or API)
# MAGIC - [ ] **Review interface** (web UI for human annotators)
# MAGIC - [ ] **A/B testing infrastructure** (traffic splitting, consistent hashing)
# MAGIC - [ ] **Monitoring dashboards** (track success rates, latencies, ratings)
# MAGIC
# MAGIC #### ✅ Data & Privacy
# MAGIC
# MAGIC - [ ] **PII handling** — scrub or encrypt sensitive data in logs
# MAGIC - [ ] **Data retention policies** — comply with GDPR/CCPA
# MAGIC - [ ] **User consent** — disclose data collection for improvement
# MAGIC
# MAGIC #### ✅ Safety & Quality
# MAGIC
# MAGIC - [ ] **Human review SLAs** — clusters reviewed within 48hrs
# MAGIC - [ ] **Shadow validation** — minimum 100 samples before promotion
# MAGIC - [ ] **Rollback plan** — instant revert if metrics regress
# MAGIC - [ ] **Policy compliance** — automated checks for brand/safety violations
# MAGIC
# MAGIC #### ✅ Metrics & Monitoring
# MAGIC
# MAGIC - [ ] **Success rate** — % of requests handled without errors/escalations
# MAGIC - [ ] **User ratings** — thumbs up/down or 1-5 stars
# MAGIC - [ ] **Latency** — P50, P95, P99
# MAGIC - [ ] **Cost** — token usage, API costs
# MAGIC - [ ] **Escalation rate** — % transferred to humans
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC 1. **Instrument your production agent** with the Feedback Collector
# MAGIC 2. **Run for 1 week** to collect baseline failures
# MAGIC 3. **Cluster and review** to identify top 3 pain points
# MAGIC 4. **Implement fixes** and validate in shadow mode
# MAGIC 5. **A/B test** the best refinement
# MAGIC 6. **Promote** and repeat — continuous improvement!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Key Principles to Remember
# MAGIC
# MAGIC > **"An improvement loop is not a project, it's a process."**
# MAGIC
# MAGIC * **Start small** — Fix one cluster at a time
# MAGIC * **Measure everything** — Intuition lies, data doesn't
# MAGIC * **Validate safely** — Shadow → A/B → Production
# MAGIC * **Automate relentlessly** — Humans review, systems execute
# MAGIC * **Never stop** — Every deployment creates new failure modes
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC 🎓 **Congratulations!** You're now equipped to build production-grade improvement loops for agentic systems.
# MAGIC
# MAGIC 🚀 **Ship it, measure it, improve it. Repeat.**