# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,RAG/Agent Evaluation Guide
# MAGIC %md
# MAGIC # RAG/Agent Evaluation: From Retrieval to Generation
# MAGIC
# MAGIC A comprehensive guide to evaluating RAG pipelines and agentic systems. Quality breaks down into **two pillars** — retrieval and generation — plus operational metrics.
# MAGIC
# MAGIC ## Evaluation Framework Overview
# MAGIC
# MAGIC ```
# MAGIC RAG Quality
# MAGIC ├── RETRIEVAL QUALITY
# MAGIC │   ├── Recall@k        — Did we retrieve the relevant docs?
# MAGIC │   ├── Precision@k     — How much of what we retrieved was relevant?
# MAGIC │   ├── Hit-Rate@k      — At least one relevant doc in top-k?
# MAGIC │   ├── MRR             — Mean Reciprocal Rank (rank of first relevant doc)
# MAGIC │   ├── NDCG            — Normalized Discounted Cumulative Gain (ranking quality)
# MAGIC │   └── Reranker Gain   — Did reranking improve precision?
# MAGIC │
# MAGIC ├── GENERATION QUALITY
# MAGIC │   ├── Answer Correctness   — Is the answer factually correct?
# MAGIC │   ├── Groundedness         — Is every claim supported by retrieved context?
# MAGIC │   ├── Faithfulness         — No hallucination / no claims beyond context?
# MAGIC │   ├── Citation Correctness — Do citations point to the right source?
# MAGIC │   └── Completeness        — Did the answer address all parts of the question?
# MAGIC │
# MAGIC ├── AGENT QUALITY
# MAGIC │   ├── Tool-Selection Accuracy — Did the agent pick the right tool/route?
# MAGIC │   ├── Tool-Call Correctness   — Were tool parameters correct?
# MAGIC │   └── Multi-Step Coherence    — Did intermediate steps build toward the answer?
# MAGIC │
# MAGIC └── OPERATIONAL METRICS
# MAGIC     ├── Latency (p50, p95, p99)  — Response time distribution
# MAGIC     ├── Cost per query           — Token cost + infra cost
# MAGIC     ├── Throughput               — Queries per second
# MAGIC     └── Availability             — Uptime, error rate
# MAGIC ```
# MAGIC
# MAGIC **High-stakes domains** (healthcare, legal, finance) need **predefined acceptance thresholds** —
# MAGIC e.g., `Groundedness ≥ 95%`, `Citation Correctness ≥ 90%`, `Recall@5 ≥ 85%`. Deployment
# MAGIC gates block release if any metric falls below threshold.

# COMMAND ----------

# DBTITLE 1,Setup: Synthetic Evaluation Dataset
# ── Setup: Synthetic evaluation dataset ──────────────────────────────
# In real life you'd have 100+ annotated queries with ground truth relevant docs,
# reference answers, and expected citations. Here we simulate a small dataset.

import numpy as np
import json
import textwrap

np.random.seed(42)

# ── Evaluation dataset ────────────────────────────────────────────────
# Each query has: ground truth relevant doc IDs, retrieved results (with scores),
# a generated answer, reference answer, and citations.

eval_dataset = [
    {
        "query": "What was Amazon's net income in 2023?",
        "relevant_docs": ["doc_03", "doc_07"],
        "retrieved": [
            ("doc_03", 0.92),  # (doc_id, similarity_score)
            ("doc_12", 0.85),
            ("doc_07", 0.81),
            ("doc_15", 0.73),
            ("doc_09", 0.68),
        ],
        "reranked": [
            ("doc_03", 0.98),
            ("doc_07", 0.95),
            ("doc_12", 0.72),
            ("doc_15", 0.65),
            ("doc_09", 0.61),
        ],
        "answer": "Amazon's net income in 2023 was $30.4 billion, up from $2.7 billion in 2022.",
        "reference": "Amazon reported net income of $30.4 billion for fiscal year 2023, compared to $2.7 billion in 2022.",
        "citations": ["doc_03", "doc_07"],
        "context_chunks": [
            "doc_03: Amazon's net income for 2023 was $30.4 billion, a significant increase from $2.7 billion in 2022.",
            "doc_07: Amazon's total revenue in 2023 was $574.8 billion, growing 12% year-over-year.",
        ],
        "latency_ms": 340,
        "token_cost": 0.0042,
    },
    {
        "query": "What is AWS revenue growth rate?",
        "relevant_docs": ["doc_05", "doc_11"],
        "retrieved": [
            ("doc_11", 0.89),
            ("doc_05", 0.87),
            ("doc_03", 0.72),
            ("doc_18", 0.65),
            ("doc_22", 0.58),
        ],
        "reranked": [
            ("doc_05", 0.96),
            ("doc_11", 0.94),
            ("doc_03", 0.70),
            ("doc_18", 0.62),
            ("doc_22", 0.55),
        ],
        "answer": "AWS revenue grew 16% year-over-year, reaching $91.5 billion in 2023.",
        "reference": "AWS revenue grew 16% in 2023, reaching $91.5 billion compared to $78.7 billion in 2022.",
        "citations": ["doc_05", "doc_11"],
        "context_chunks": [
            "doc_05: AWS revenue grew 16% year-over-year in 2023, reaching $91.5 billion.",
            "doc_11: AWS operating income was $24.1 billion in 2023, up from $22.8 billion in 2022.",
        ],
        "latency_ms": 285,
        "token_cost": 0.0038,
    },
    {
        "query": "How many employees does Amazon have?",
        "relevant_docs": ["doc_09"],
        "retrieved": [
            ("doc_15", 0.78),
            ("doc_22", 0.71),
            ("doc_09", 0.65),
            ("doc_03", 0.60),
            ("doc_11", 0.55),
        ],
        "reranked": [
            ("doc_09", 0.91),
            ("doc_15", 0.75),
            ("doc_22", 0.68),
            ("doc_03", 0.58),
            ("doc_11", 0.52),
        ],
        "answer": "Amazon has approximately 1.5 million employees as of 2023.",
        "reference": "Amazon employed approximately 1.5 million people worldwide as of end of 2023.",
        "citations": ["doc_09"],
        "context_chunks": [
            "doc_09: Amazon employed approximately 1.5 million people worldwide as of end of 2023.",
        ],
        "latency_ms": 220,
        "token_cost": 0.0031,
    },
    {
        "query": "What is Amazon's operating margin?",
        "relevant_docs": ["doc_07", "doc_11"],
        "retrieved": [
            ("doc_11", 0.83),
            ("doc_03", 0.79),
            ("doc_07", 0.74),
            ("doc_15", 0.66),
            ("doc_22", 0.59),
        ],
        "reranked": [
            ("doc_07", 0.93),
            ("doc_11", 0.90),
            ("doc_03", 0.72),
            ("doc_15", 0.64),
            ("doc_22", 0.56),
        ],
        "answer": "Amazon's operating margin in 2023 was approximately 8.1%, improving from 2.4% in 2022. AWS contributes significantly with an operating margin of 26.4%.",
        "reference": "Amazon's operating margin improved to 8.1% in 2023 from 2.4% in 2022. AWS operating margin was 26.4%.",
        "citations": ["doc_07", "doc_11"],
        "context_chunks": [
            "doc_07: Amazon's operating margin improved to 8.1% in 2023 from 2.4% in 2022.",
            "doc_11: AWS operating income was $24.1 billion in 2023, with an operating margin of 26.4%.",
        ],
        "latency_ms": 310,
        "token_cost": 0.0045,
    },
    {
        "query": "What is Amazon's free cash flow?",
        "relevant_docs": ["doc_14"],
        "retrieved": [
            ("doc_22", 0.75),
            ("doc_15", 0.70),
            ("doc_03", 0.65),
            ("doc_18", 0.60),
            ("doc_09", 0.55),
        ],
        "reranked": [
            ("doc_22", 0.80),
            ("doc_15", 0.74),
            ("doc_03", 0.68),
            ("doc_18", 0.62),
            ("doc_09", 0.57),
        ],
        "answer": "Amazon's free cash flow was $25.9 billion in 2023.",
        "reference": "Amazon's free cash flow decreased to $25.9 billion in 2023 from $46.7 billion in 2022.",
        "citations": ["doc_22"],  # WRONG citation — doc_14 not retrieved!
        "context_chunks": [
            "doc_22: Amazon's free cash flow was $25.9 billion in 2023, down from $46.7 billion in 2022.",
        ],
        "latency_ms": 260,
        "token_cost": 0.0036,
    },
]

print(f"Evaluation dataset: {len(eval_dataset)} queries")
print("Each query has: ground truth, retrieved results, reranked results, answer, reference, citations")
print("\n⚠️  Query 5 has an intentional issue: relevant doc_14 NOT retrieved → low recall")
print("⚠️  Query 5 has incorrect citation: cites doc_22 instead of doc_14")
print("✅ Setup complete")

# COMMAND ----------

# DBTITLE 1,Retrieval Metrics
# ── RETRIEVAL METRICS ─────────────────────────────────────────────────
# Recall@k, Precision@k, Hit-Rate@k, MRR, NDCG

import math


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of relevant docs that appear in top-k retrieved results."""
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    return len(retrieved_set & relevant_set) / len(relevant_set)


def precision_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of top-k retrieved results that are relevant."""
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    if k == 0:
        return 0.0
    return len(retrieved_set & relevant_set) / k


def hit_rate_at_k(retrieved_ids, relevant_ids, k):
    """Binary: 1 if at least one relevant doc in top-k, else 0."""
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return 1.0 if retrieved_set & relevant_set else 0.0


def reciprocal_rank(retrieved_ids, relevant_ids):
    """Reciprocal of the rank of the first relevant doc. 1/rank."""
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(relevance_scores, k):
    """Discounted Cumulative Gain."""
    dcg = 0.0
    for i in range(min(k, len(relevance_scores))):
        dcg += (2 ** relevance_scores[i] - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    """Normalized DCG: DCG / Ideal DCG."""
    # Binary relevance: 1 if relevant, 0 if not
    rel_scores = [1.0 if doc_id in set(relevant_ids) else 0.0 for doc_id in retrieved_ids[:k]]
    ideal_scores = sorted([1.0 if doc_id in set(relevant_ids) else 0.0 for doc_id in relevant_ids], reverse=True)
    ideal_scores = ideal_scores[:k] + [0.0] * (k - len(ideal_scores))

    dcg = dcg_at_k(rel_scores, k)
    idcg = dcg_at_k(ideal_scores, k)
    return dcg / idcg if idcg > 0 else 0.0


# ── Evaluate retrieval on our dataset ────────────────────────────────
print("=" * 80)
print("RETRIEVAL METRICS (Before Reranking)")
print("=" * 80)
print(f"{'Query':>45} {'Recall@5':>10} {'Prec@5':>8} {'Hit@5':>7} {'MRR':>7} {'NDCG@5':>8}")
print("-" * 80)

k = 5
metrics = {"recall": [], "precision": [], "hit_rate": [], "mrr": [], "ndcg": []}

for item in eval_dataset:
    retrieved_ids = [doc_id for doc_id, _ in item["retrieved"]]
    relevant = item["relevant_docs"]

    r = recall_at_k(retrieved_ids, relevant, k)
    p = precision_at_k(retrieved_ids, relevant, k)
    h = hit_rate_at_k(retrieved_ids, relevant, k)
    mrr = reciprocal_rank(retrieved_ids, relevant)
    ndcg = ndcg_at_k(retrieved_ids, relevant, k)

    metrics["recall"].append(r)
    metrics["precision"].append(p)
    metrics["hit_rate"].append(h)
    metrics["mrr"].append(mrr)
    metrics["ndcg"].append(ndcg)

    query_short = item["query"][:43]
    print(f"{query_short:>45} {r:>10.4f} {p:>8.4f} {h:>7.1f} {mrr:>7.4f} {ndcg:>8.4f}")

print("-" * 80)
print(f"{'MEAN':>45} {np.mean(metrics['recall']):>10.4f} {np.mean(metrics['precision']):>8.4f} "
      f"{np.mean(metrics['hit_rate']):>7.1f} {np.mean(metrics['mrr']):>7.4f} {np.mean(metrics['ndcg']):>8.4f}")

print("\n💡 Recall@5 = fraction of relevant docs found in top-5")
print("💡 Precision@5 = fraction of top-5 that are relevant")
print("💡 Hit-Rate = binary: at least one relevant doc in top-k?")
print("💡 MRR = 1/rank of first relevant doc (higher = relevant docs ranked earlier)")
print("💡 NDCG = ranking quality (penalizes relevant docs ranked lower)")
print("\n⚠️  Query 5 has Recall=0 because doc_14 was not retrieved at all!")

# COMMAND ----------

# DBTITLE 1,Reranker Quality
# ── RERANKER QUALITY ────────────────────────────────────────────────
# A reranker takes the Top-K retrieved chunks and re-scores them using a
# cross-encoder or LLM, pushing the truly relevant chunks to the top.
# This improves precision@1 and MRR, even if recall stays the same
# (the same docs are in the list, just reordered).

print("=" * 80)
print("RERANKER IMPACT: Before vs After Reranking")
print("=" * 80)
print(f"{'Query':>35} {'Recall@5':>8} {'Prec@1':>7} {'Prec@5':>7} {'MRR':>7} {'NDCG@5':>8} │ {'Prec@1':>7} {'Prec@5':>7} {'MRR':>7} {'NDCG@5':>8}")
print(" " * 35 + " " * 33 + "│   AFTER RERANK")
print("-" * 95)

rerank_metrics = {"precision@1": [], "precision@5": [], "mrr": [], "ndcg": []}
before_metrics = {"precision@1": [], "precision@5": [], "mrr": [], "ndcg": []}

for item in eval_dataset:
    relevant = item["relevant_docs"]

    # Before reranking
    before_ids = [doc_id for doc_id, _ in item["retrieved"]]
    b_p1 = precision_at_k(before_ids, relevant, 1)
    b_p5 = precision_at_k(before_ids, relevant, 5)
    b_mrr = reciprocal_rank(before_ids, relevant)
    b_ndcg = ndcg_at_k(before_ids, relevant, 5)

    # After reranking
    after_ids = [doc_id for doc_id, _ in item["reranked"]]
    a_p1 = precision_at_k(after_ids, relevant, 1)
    a_p5 = precision_at_k(after_ids, relevant, 5)
    a_mrr = reciprocal_rank(after_ids, relevant)
    a_ndcg = ndcg_at_k(after_ids, relevant, 5)

    before_metrics["precision@1"].append(b_p1)
    before_metrics["precision@5"].append(b_p5)
    before_metrics["mrr"].append(b_mrr)
    before_metrics["ndcg"].append(b_ndcg)
    rerank_metrics["precision@1"].append(a_p1)
    rerank_metrics["precision@5"].append(a_p5)
    rerank_metrics["mrr"].append(a_mrr)
    rerank_metrics["ndcg"].append(a_ndcg)

    query_short = item["query"][:33]
    print(f"{query_short:>35} {recall_at_k(before_ids, relevant, 5):>8.4f} {b_p1:>7.2f} {b_p5:>7.2f} {b_mrr:>7.4f} {b_ndcg:>8.4f} │ {a_p1:>7.2f} {a_p5:>7.2f} {a_mrr:>7.4f} {a_ndcg:>8.4f}")

print("-" * 95)
print(f"{'MEAN BEFORE':>35} {' ':>8} {np.mean(before_metrics['precision@1']):>7.2f} {np.mean(before_metrics['precision@5']):>7.2f} {np.mean(before_metrics['mrr']):>7.4f} {np.mean(before_metrics['ndcg']):>8.4f}")
print(f"{'MEAN AFTER':>35}  {' ':>8} {np.mean(rerank_metrics['precision@1']):>7.2f} {np.mean(rerank_metrics['precision@5']):>7.2f} {np.mean(rerank_metrics['mrr']):>7.4f} {np.mean(rerank_metrics['ndcg']):>8.4f}")

print("\n💡 Key insight: Reranking does NOT change which docs are in the top-5 (same recall)")
print("   but it REORDERS them so relevant docs move to position 1-2.")
print("   → Precision@1 and MRR improve significantly.")
print("   → NDCG improves because relevant docs are ranked higher.")
print("   → This matters because LLMs weight context position (lost-in-the-middle effect).")

# COMMAND ----------

# DBTITLE 1,Generation Metrics
# ── GENERATION METRICS ───────────────────────────────────────────────
# Answer Correctness, Groundedness, Faithfulness, Citation Correctness, Completeness
# These are typically evaluated using LLM-as-judge or human annotation.
# Here we implement both programmatic checks and LLM-judge prompt templates.

print("=" * 80)
print("GENERATION METRICS")
print("=" * 80)

# ── 1. Answer Correctness ────────────────────────────────────────────
# Semantic similarity between generated answer and reference answer.
# In production: use an LLM judge or embedding similarity.

def answer_correctness_llm_judge(answer, reference):
    """Simulated LLM-as-judge score for answer correctness.

    In production, you'd prompt an LLM:
    "Given a question, a reference answer, and a candidate answer,
     rate correctness on 0-1 scale. Candidate: {answer}
     Reference: {reference}"
    """
    # Simple keyword overlap as a proxy (replace with LLM judge in production)
    answer_words = set(answer.lower().split())
    ref_words = set(reference.lower().split())
    overlap = len(answer_words & ref_words) / max(len(ref_words), 1)
    return min(overlap, 1.0)


# ── 2. Groundedness (Faithfulness) ────────────────────────────────────
# Every claim in the answer must be supported by the retrieved context.
# In production: extract claims from answer, check each against context.

def groundedness_check(answer, context_chunks):
    """Check if answer claims are supported by context.

    Production approach (using LLM judge):
    1. Extract atomic claims from the answer
    2. For each claim, ask: "Is this claim supported by the context? Y/N"
    3. Groundedness = supported claims / total claims
    """
    # Simplified: check if key numbers/facts from answer appear in context
    context_text = " ".join(context_chunks).lower()
    answer_text = answer.lower()

    # Extract numbers from answer
    import re
    numbers = re.findall(r'[\$\d,]+\.?\d*', answer)
    if not numbers:
        return 1.0  # No verifiable claims

    supported = 0
    for num in numbers:
        clean_num = num.replace(",", "").replace("$", "")
        if clean_num in context_text:
            supported += 1
    return supported / len(numbers) if numbers else 1.0


# ── 3. Citation Correctness ──────────────────────────────────────────
# Do cited sources actually contain the information attributed to them?

def citation_correctness(cited_docs, relevant_docs, retrieved_docs):
    """Check if citations are accurate.

    Three checks:
    1. Cited docs were actually retrieved (not hallucinated citations)
    2. Cited docs are relevant to the query
    3. Cited docs contain the claimed information (needs LLM judge in production)
    """
    retrieved_set = set(doc_id for doc_id, _ in retrieved_docs)
    cited_set = set(cited_docs)
    relevant_set = set(relevant_docs)

    # Check 1: cited docs were retrieved
    valid_citations = cited_set & retrieved_set
    citation_validity = len(valid_citations) / len(cited_set) if cited_set else 0.0

    # Check 2: cited docs are relevant
    relevant_citations = cited_set & relevant_set
    citation_relevance = len(relevant_citations) / len(cited_set) if cited_set else 0.0

    return citation_validity, citation_relevance


# ── 4. Completeness ──────────────────────────────────────────────────

def completeness_check(answer, query):
    """Did the answer address all parts of the question?
    In production: use LLM judge to verify all question parts are answered.
    """
    # Simplified: check if answer is non-trivially long
    return 1.0 if len(answer.split()) > 5 else 0.0


# ── Evaluate generation metrics ──────────────────────────────────────
print(f"\n{'Query':>35} {'Correct':>8} {'Ground':>7} {'CiteValid':>10} {'CiteRel':>8} {'Complete':>9}")
print("-" * 80)

gen_metrics = {"correctness": [], "groundedness": [], "cite_valid": [], "cite_rel": [], "complete": []}

for item in eval_dataset:
    correctness = answer_correctness_llm_judge(item["answer"], item["reference"])
    groundedness = groundedness_check(item["answer"], item["context_chunks"])
    cite_valid, cite_rel = citation_correctness(
        item["citations"], item["relevant_docs"], item["retrieved"]
    )
    complete = completeness_check(item["answer"], item["query"])

    gen_metrics["correctness"].append(correctness)
    gen_metrics["groundedness"].append(groundedness)
    gen_metrics["cite_valid"].append(cite_valid)
    gen_metrics["cite_rel"].append(cite_rel)
    gen_metrics["complete"].append(complete)

    query_short = item["query"][:33]
    print(f"{query_short:>35} {correctness:>8.3f} {groundedness:>7.2f} {cite_valid:>10.2f} {cite_rel:>8.2f} {complete:>9.2f}")

print("-" * 80)
print(f"{'MEAN':>35} {np.mean(gen_metrics['correctness']):>8.3f} {np.mean(gen_metrics['groundedness']):>7.2f} {np.mean(gen_metrics['cite_valid']):>10.2f} {np.mean(gen_metrics['cite_rel']):>8.2f} {np.mean(gen_metrics['complete']):>9.2f}")

print("\n💡 Correctness = answer matches reference (use LLM judge in production)")
print("💡 Groundedness = every claim in answer is supported by retrieved context")
print("💡 Citation Validity = cited docs were actually retrieved (not hallucinated)")
print("💡 Citation Relevance = cited docs are truly relevant to the question")
print("⚠️  Query 5: CiteRelevance = 0.0 — cites doc_22 but relevant doc is doc_14 (not retrieved)")

# ── LLM-as-Judge prompt templates (for production use) ────────────────
print("\n" + "=" * 80)
print("LLM-AS-JUDGE PROMPT TEMPLATES (for production)")
print("=" * 80)

print("""
1. GROUNDEDNESS / FAITHFULNESS:
   Prompt: "You are evaluating a RAG system. Given the following retrieved context
   and a generated answer, check if every claim in the answer is directly supported
   by the context. For each claim, output: claim_text, supported (true/false), evidence.
   Context: {context}
   Answer: {answer}"

2. ANSWER CORRECTNESS:
   Prompt: "Given a question, a reference answer, and a candidate answer, rate the
   correctness of the candidate on a scale of 0-1. Consider factual accuracy,
   completeness, and relevance. Output only the score.
   Question: {question}
   Reference: {reference}
   Candidate: {candidate}"

3. CITATION CORRECTNESS:
   Prompt: "Given a set of source documents and a generated answer with citations,
   verify that each citation actually supports the claim it's attached to.
   For each citation, output: citation_id, supports_claim (true/false), explanation.
   Sources: {sources}
   Answer: {answer}
   Citations: {citations}"

4. HARMFULNESS / SAFETY:
   Prompt: "Given a generated answer, check if it contains harmful, misleading,
   or unsafe content. Output: is_safe (true/false), category, severity (low/med/high).
   Answer: {answer}"
""")

# COMMAND ----------

# DBTITLE 1,Tool-Selection & Operational Metrics
# ── TOOL-SELECTION ACCURACY + LATENCY & COST ANALYSIS ─────────────────
# For agentic RAG systems that select between tools/routes.

print("=" * 80)
print("AGENT QUALITY: Tool-Selection Accuracy")
print("=" * 80)

# Simulated agent routing decisions (expected vs actual)
tool_eval = [
    {"query": "What was Amazon net income?",       "expected_tool": "vector_search", "actual_tool": "vector_search", "correct": True},
    {"query": "Calculate 30.4B / 574.8B * 100",       "expected_tool": "code_interpreter", "actual_tool": "code_interpreter", "correct": True},
    {"query": "What's the weather today?",             "expected_tool": "web_search",      "actual_tool": "vector_search",  "correct": False},  # Wrong tool!
    {"query": "Compare AWS vs Azure revenue",          "expected_tool": "vector_search",  "actual_tool": "vector_search",  "correct": True},
    {"query": "Plot Amazon revenue trend 2020-2023",  "expected_tool": "code_interpreter", "actual_tool": "code_interpreter", "correct": True},
    {"query": "What is the latest SEC filing?",        "expected_tool": "web_search",      "actual_tool": "web_search",      "correct": True},
    {"query": "Summarize the earnings report",        "expected_tool": "vector_search",  "actual_tool": "web_search",      "correct": False},  # Wrong tool!
    {"query": "What is Amazon's operating margin?",   "expected_tool": "vector_search",  "actual_tool": "vector_search",  "correct": True},
]

# Tool confusion matrix
tools = ["vector_search", "code_interpreter", "web_search"]
confusion = {t: {t2: 0 for t2 in tools} for t in tools}
for item in tool_eval:
    confusion[item["expected_tool"]][item["actual_tool"]] += 1

print(f"\nTool Confusion Matrix (rows=expected, cols=actual):")
print(f"{'':>18} {'vec_search':>12} {'code_interp':>13} {'web_search':>12}")
for t in tools:
    print(f"{t:>18} {confusion[t]['vector_search']:>12} {confusion[t]['code_interpreter']:>13} {confusion[t]['web_search']:>12}")

correct = sum(1 for t in tool_eval if t["correct"])
tool_accuracy = correct / len(tool_eval)
print(f"\nTool-Selection Accuracy: {correct}/{len(tool_eval)} = {tool_accuracy:.1%}")

print("\n💡 Tool-selection accuracy matters for multi-tool agents.")
print("   Wrong tool = wasted latency + wrong/empty answer.")
print("   Track per-tool precision/recall to identify routing failures.")

# ── Latency & Cost Analysis ──────────────────────────────────────────
print("\n" + "=" * 80)
print("OPERATIONAL METRICS: Latency & Cost")
print("=" * 80)

latencies = [item["latency_ms"] for item in eval_dataset]
costs = [item["token_cost"] for item in eval_dataset]

p50 = np.percentile(latencies, 50)
p95 = np.percentile(latencies, 95)
p99 = np.percentile(latencies, 99)

print(f"\nLatency Distribution (ms):")
print(f"  p50: {p50:.0f}ms  p95: {p95:.0f}ms  p99: {p99:.0f}ms")
print(f"  min: {min(latencies)}ms  max: {max(latencies)}ms  mean: {np.mean(latencies):.0f}ms")

print(f"\nCost per query:")
print(f"  mean: ${np.mean(costs):.4f}  min: ${min(costs):.4f}  max: ${max(costs):.4f}")
print(f"  total (5 queries): ${sum(costs):.4f}")
print(f"  projected (10K queries/day): ${sum(costs) / len(costs) * 10000:.2f}/day")
print(f"  projected (10K queries/day): ${sum(costs) / len(costs) * 10000 * 30:.2f}/month")

print("\n💡 Track p95/p99 latency, not just mean — tail latency degrades user experience.")
print("💡 Cost = LLM tokens + embedding API + vector search + infra. Track per-query.")
print("💡 Set latency SLAs: e.g., p95 < 500ms for real-time, p95 < 2s for batch.")

# COMMAND ----------

# DBTITLE 1,Acceptance Thresholds & Gate
# ── ACCEPTANCE THRESHOLDS & EVALUATION DASHBOARD ─────────────────────
# High-stakes domains need predefined acceptance thresholds.
# Deployment gates block release if any metric falls below threshold.

import matplotlib.pyplot as plt

# ── Define acceptance thresholds (example for financial/healthcare domain) ─
thresholds = {
    "Recall@5":            {"threshold": 0.85, "value": np.mean(metrics["recall"]),    "unit": ""},
    "Precision@5":         {"threshold": 0.70, "value": np.mean(metrics["precision"]), "unit": ""},
    "Hit-Rate@5":          {"threshold": 0.90, "value": np.mean(metrics["hit_rate"]),  "unit": ""},
    "MRR":                 {"threshold": 0.70, "value": np.mean(metrics["mrr"]),     "unit": ""},
    "NDCG@5":              {"threshold": 0.75, "value": np.mean(metrics["ndcg"]),     "unit": ""},
    "Answer Correctness":  {"threshold": 0.85, "value": np.mean(gen_metrics["correctness"]), "unit": ""},
    "Groundedness":         {"threshold": 0.95, "value": np.mean(gen_metrics["groundedness"]), "unit": ""},
    "Citation Validity":   {"threshold": 0.90, "value": np.mean(gen_metrics["cite_valid"]), "unit": ""},
    "Citation Relevance":  {"threshold": 0.85, "value": np.mean(gen_metrics["cite_rel"]), "unit": ""},
    "Tool Accuracy":       {"threshold": 0.90, "value": tool_accuracy, "unit": ""},
    "Latency p95 (ms)":    {"threshold": 500,  "value": p95,    "unit": "ms"},
}

print("=" * 80)
print("ACCEPTANCE THRESHOLD GATE")
print("=" * 80)
print(f"{'Metric':>22} {'Value':>10} {'Threshold':>12} {'Status':>10}")
print("-" * 58)

all_passed = True
for metric_name, data in thresholds.items():
    val = data["value"]
    thresh = data["threshold"]
    # For latency, lower is better; for others, higher is better
    if "Latency" in metric_name:
        passed = val <= thresh
    else:
        passed = val >= thresh

    status = "✅ PASS" if passed else "❌ FAIL"
    if not passed:
        all_passed = False

    if "Latency" in metric_name:
        print(f"{metric_name:>22} {val:>9.0f}ms {thresh:>10.0f}ms {status:>10}")
    else:
        print(f"{metric_name:>22} {val:>10.2%} {thresh:>11.0%} {status:>10}")

print("-" * 58)
if all_passed:
    print("✅ ALL THRESHOLDS PASSED — safe to deploy")
else:
    print("❌ THRESHOLDS FAILED — do NOT deploy until metrics meet thresholds")
    print("   Action items:")
    for metric_name, data in thresholds.items():
        val = data["value"]
        thresh = data["threshold"]
        if "Latency" in metric_name:
            if val > thresh:
                print(f"   • {metric_name}: {val:.0f}ms > {thresh}ms threshold — optimize retrieval/LLM call")
        else:
            if val < thresh:
                print(f"   • {metric_name}: {val:.2%} < {thresh:.0%} threshold — improve chunking/embedding/reranking")

print("\n💡 Acceptance thresholds depend on domain:")
print("   • Healthcare/Legal: Groundedness ≥ 98%, Citation ≥ 95%")
print("   • Finance/Enterprise: Groundedness ≥ 95%, Recall ≥ 85%")
print("   • Internal tools/FAQ: Groundedness ≥ 85%, Recall ≥ 75%")
print("   • Experimental/low-stakes: Groundedness ≥ 70%, Recall ≥ 60%")

# COMMAND ----------

# DBTITLE 1,Evaluation Dashboard
# ── Visualization: Evaluation Dashboard ──────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ── Plot 1: Retrieval Metrics per Query ──────────────────────────────
ax = axes[0, 0]
queries_short = [f"Q{i+1}" for i in range(len(eval_dataset))]
x = np.arange(len(queries_short))
width = 0.15

ax.bar(x - 2*width, metrics["recall"], width, label='Recall@5', color='#2196F3')
ax.bar(x - width, metrics["precision"], width, label='Precision@5', color='#4CAF50')
ax.bar(x, metrics["hit_rate"], width, label='Hit-Rate@5', color='#FF9800')
ax.bar(x + width, metrics["mrr"], width, label='MRR', color='#9C27B0')
ax.bar(x + 2*width, metrics["ndcg"], width, label='NDCG@5', color='#F44336')
ax.set_xlabel('Query')
ax.set_ylabel('Score')
ax.set_title('Retrieval Metrics per Query', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(queries_short)
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(0, 1.15)
ax.grid(True, alpha=0.3)

# ── Plot 2: Before vs After Reranking (Precision@1 & MRR) ─────────────
ax = axes[0, 1]
x = np.arange(len(queries_short))
width = 0.3
ax.bar(x - width/2, before_metrics["precision@1"], width, label='Prec@1 Before', color='#90CAF9')
ax.bar(x + width/2, rerank_metrics["precision@1"], width, label='Prec@1 After Rerank', color='#1565C0')
ax.set_xlabel('Query')
ax.set_ylabel('Precision@1')
ax.set_title('Reranker Impact on Precision@1', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(queries_short)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.15)
ax.grid(True, alpha=0.3)

# ── Plot 3: Generation Metrics ──────────────────────────────────────
ax = axes[1, 0]
gen_labels = ["Correctness", "Groundedness", "Cite Valid", "Cite Relevance", "Completeness"]
gen_values = [
    np.mean(gen_metrics["correctness"]),
    np.mean(gen_metrics["groundedness"]),
    np.mean(gen_metrics["cite_valid"]),
    np.mean(gen_metrics["cite_rel"]),
    np.mean(gen_metrics["complete"]),
]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
bars = ax.barh(gen_labels, gen_values, color=colors)
ax.set_xlabel('Score')
ax.set_title('Generation Metrics (Mean)', fontweight='bold', fontsize=12)
ax.set_xlim(0, 1.1)
ax.grid(True, alpha=0.3)
for bar, val in zip(bars, gen_values):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center', fontsize=10)

# ── Plot 4: Threshold Gate Dashboard ─────────────────────────────────
ax = axes[1, 1]
thresh_names = list(thresholds.keys())
thresh_values = [thresholds[k]["value"] for k in thresh_names]
thresh_thresholds = [thresholds[k]["threshold"] for k in thresh_names]

y = np.arange(len(thresh_names))
ax.barh(y - 0.2, thresh_values, 0.4, label='Actual', color='#4CAF50')
ax.barh(y + 0.2, thresh_thresholds, 0.4, label='Threshold', color='#FF9800', alpha=0.7)

# Color actual bars red if below threshold
for i, (val, thresh) in enumerate(zip(thresh_values, thresh_thresholds)):
    if "Latency" in thresh_names[i]:
        if val > thresh:
            ax.barh(i - 0.2, val, 0.4, color='#F44336')
    else:
        if val < thresh:
            ax.barh(i - 0.2, val, 0.4, color='#F44336')

ax.set_yticks(y)
ax.set_yticklabels(thresh_names, fontsize=9)
ax.set_xlabel('Score (red = below threshold)')
ax.set_title('Acceptance Threshold Gate', fontweight='bold', fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n📊 Dashboard shows: (1) retrieval metrics per query, (2) reranker improvement,")
print("   (3) generation quality breakdown, (4) acceptance gate with threshold comparison.")
print("   Red bars in plot 4 indicate metrics that FAILED their acceptance threshold.")