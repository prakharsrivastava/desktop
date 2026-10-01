# Databricks notebook source
# DBTITLE 1,MLflow Evaluation & RAGAS Title
# MAGIC %md
# MAGIC # MLflow Evaluation & RAGAS: All Scenarios for RAG, Agents & Tools
# MAGIC
# MAGIC Comprehensive evaluation notebook covering MLflow GenAI built-in scorers, RAGAS metrics, agent/tool-call evaluation, multi-turn conversations, production monitoring, and run comparison — aligned with the *Evaluation for LLM Applications* framework.

# COMMAND ----------

# DBTITLE 1,Setup: Install MLflow + RAGAS
# ─── Setup: Install MLflow (GenAI) ───
# RAGAS library has a langchain_community dependency conflict in this environment.
# This notebook implements all RAGAS metrics manually with enhanced algorithms.
%pip install --upgrade "mlflow[databricks]" --quiet
%pip uninstall ragas langchain-community langchain-core -y --quiet 2>/dev/null || true

import mlflow
import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional

# MLflow experiment setup (Databricks auto-configures tracking)
mlflow.set_experiment("/Users/prakhar1207srivastava@gmail.com/llm-evaluation-ragas")
print(f"MLflow version: {mlflow.__version__}")

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Foundations & Observability
# MAGIC %md
# MAGIC ## 1. Foundations of LLM Evaluation
# MAGIC
# MAGIC **Intrinsic vs Extrinsic Evaluation**
# MAGIC * *Intrinsic*: Measures the model in isolation (perplexity, BLEU, benchmark accuracy). Quick and automated, but may not reflect real-world usefulness.
# MAGIC * *Extrinsic*: Tests the model within a specific application (user satisfaction, task success, latency). More realistic, harder and more expensive.
# MAGIC
# MAGIC **What makes an LLM "good"?** Accuracy, Helpfulness, Safety, Latency.
# MAGIC
# MAGIC **Challenges**: Subjectivity, Open-endedness, Context sensitivity, Scalability.
# MAGIC
# MAGIC ## 2. Instrumentation & Observability
# MAGIC
# MAGIC MLflow tracing captures:
# MAGIC * **Inputs**: User prompts, system instructions, context documents, model parameters
# MAGIC * **Outputs**: Generated text, confidence scores
# MAGIC * **Metadata**: Timestamps, latency, token usage, session IDs, user IDs
# MAGIC
# MAGIC Every trace creates a complete record enabling systematic error analysis and continuous improvement.

# COMMAND ----------

# DBTITLE 1,Post-Restart Imports
# Post-restart: re-import everything
import mlflow
import os
import json
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

# Re-set experiment after restart
mlflow.set_experiment("/Users/prakhar1207srivastava@gmail.com/llm-evaluation-ragas")

# Import MLflow GenAI scorers
try:
    from mlflow.genai.scorers import (
        RelevanceToQuery, Safety, Guidelines, scorer
    )
    from mlflow.entities import SpanType, Feedback
    print("MLflow GenAI scorers loaded")
except ImportError as e:
    print(f"Some scorers not available: {e}")

print(f"MLflow {mlflow.__version__} ready")
print("Using enhanced manual RAGAS-equivalent metrics (all 5 metrics)")

# COMMAND ----------

# DBTITLE 1,Build Demo RAG Pipeline with Tracing
# ─── Build a Demo RAG Pipeline with MLflow Tracing ───

# Mock knowledge base
KNOWLEDGE_BASE = [
    {"id": "doc1", "text": "MLflow is an open-source platform for managing the ML lifecycle, including experimentation, reproducibility, and deployment."},
    {"id": "doc2", "text": "Databricks Lakehouse combines data warehouses and data lakes into a single platform with Delta Lake at its core."},
    {"id": "doc3", "text": "Vector search enables semantic similarity matching by embedding text into high-dimensional vectors."},
    {"id": "doc4", "text": "RAG (Retrieval-Augmented Generation) combines a retrieval system with a generative LLM to produce grounded answers."},
    {"id": "doc5", "text": "MLflow GenAI evaluation provides built-in LLM-judge scorers for relevance, groundedness, safety, and guidelines."},
]

# Simulated retrieval (mock for demo)
@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_docs(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Mock retriever: keyword-based scoring for demo."""
    query_words = set(query.lower().split())
    scored = []
    for doc in KNOWLEDGE_BASE:
        doc_words = set(doc["text"].lower().split())
        overlap = len(query_words & doc_words)
        scored.append({**doc, "score": overlap / max(len(query_words), 1)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

# Simulated LLM generation (mock for demo)
@mlflow.trace(span_type=SpanType.LLM)
def generate_response(query: str, retrieved_docs: List[Dict]) -> str:
    """Mock generator: synthesizes answer from retrieved context."""
    context = " ".join([d["text"] for d in retrieved_docs])
    if "mlflow" in query.lower():
        return "Based on retrieved documents: MLflow is an open-source platform for managing the ML lifecycle, including experimentation and deployment."
    elif "rag" in query.lower():
        return "Based on retrieved documents: RAG combines a retrieval system with a generative LLM to produce grounded, fact-based answers."
    elif "vector" in query.lower():
        return "Based on retrieved documents: Vector search enables semantic similarity matching using high-dimensional vector embeddings."
    else:
        return f"Based on retrieved documents: {context[:200]}"

# Full RAG pipeline with session tracking
@mlflow.trace(span_type=SpanType.CHAIN)
def rag_pipeline(query: str, user_id: str = "eval_user", session_id: str = "session_001") -> Dict[str, Any]:
    """End-to-end RAG pipeline with MLflow tracing."""
    mlflow.update_current_trace(metadata={
        "mlflow.trace.user": user_id,
        "mlflow.trace.session": session_id,
    })
    retrieved = retrieve_docs(query)
    response = generate_response(query, retrieved)
    return {
        "query": query,
        "response": response,
        "retrieved_contexts": [d["text"] for d in retrieved],
        "retrieved_doc_ids": [d["id"] for d in retrieved],
    }

# Test the pipeline
result = rag_pipeline("What is MLflow?")
print(json.dumps(result, indent=2))

# COMMAND ----------

# DBTITLE 1,RAGAS Evaluation Scenarios
# MAGIC %md
# MAGIC ## 3. RAGAS Evaluation Scenarios
# MAGIC
# MAGIC RAGAS (RAG Assessment) provides LLM-judge-based metrics for RAG evaluation:
# MAGIC
# MAGIC | Metric | What It Measures | Needs Ground Truth? |
# MAGIC |---|---|---|
# MAGIC | **Faithfulness** | Is the answer grounded in retrieved context? (no hallucination) | No |
# MAGIC | **Answer Relevancy** | Is the answer relevant to the question? | No |
# MAGIC | **Context Precision** | Are relevant chunks ranked higher in retrieval? | Yes |
# MAGIC | **Context Recall** | Did retrieval capture all info needed for the ground truth answer? | Yes |
# MAGIC | **Context Entity Recall** | Did retrieval capture entities from the ground truth? | Yes |
# MAGIC
# MAGIC **RAGAS vs MLflow Built-in**: RAGAS uses its own LLM judge; MLflow scorers use Databricks-hosted judges. Both can run side by side.

# COMMAND ----------

# DBTITLE 1,RAGAS Evaluation All Metrics
# ─── RAGAS Evaluation: All Metrics ───

# Build evaluation dataset with ground truth
eval_questions = [
    {"question": "What is MLflow?",
     "ground_truth": "MLflow is an open-source platform for managing the ML lifecycle, including experimentation, reproducibility, and deployment."},
    {"question": "What is RAG?",
     "ground_truth": "RAG combines a retrieval system with a generative LLM to produce grounded answers."},
    {"question": "How does vector search work?",
     "ground_truth": "Vector search enables semantic similarity matching by embedding text into high-dimensional vectors."},
    {"question": "What is Databricks Lakehouse?",
     "ground_truth": "Databricks Lakehouse combines data warehouses and data lakes with Delta Lake at its core."},
]

# Run RAG pipeline on all questions
rag_results = []
for item in eval_questions:
    output = rag_pipeline(item["question"])
    rag_results.append({
        "question": item["question"],
        "answer": output["response"],
        "contexts": output["retrieved_contexts"],
        "ground_truth": item["ground_truth"],
    })

print(f"Generated {len(rag_results)} RAG outputs for evaluation")

# ─── Scenario 1: RAGAS Evaluation ───
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy,
        context_precision, context_recall
    )
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    # Build RAGAS samples
    samples = []
    for r in rag_results:
        samples.append(SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        ))

    eval_dataset = EvaluationDataset(samples=samples)

    # Run RAGAS evaluation with all metrics
    ragas_result = ragas_evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    print("\n=== RAGAS Evaluation Results ===")
    print(ragas_result)

    if hasattr(ragas_result, 'to_pandas'):
        ragas_df = ragas_result.to_pandas()
    else:
        ragas_df = pd.DataFrame(ragas_result.scores)
    display(ragas_df)

except ImportError:
    print("RAGAS not available — using manual fallback metrics (see below)")
except Exception as e:
    print(f"RAGAS evaluation error: {e}")
    print("This may require LLM API access. Falling back to manual metrics.")

# ─── Enhanced RAGAS-Equivalent Metrics (all 5) ───
# Implements: faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall

def normalize_text(text: str) -> set:
    """Tokenize and normalize text to lowercase word set."""
    return set(re.findall(r'\b\w+\b', text.lower()))

def extract_entities(text: str) -> set:
    """Simple entity extraction: capitalized words, acronyms, numbers."""
    entities = set()
    # Capitalized words (proper nouns)
    entities.update(re.findall(r'\b[A-Z][a-z]+\b', text))
    # Acronyms (all caps)
    entities.update(re.findall(r'\b[A-Z]{2,}\b', text))
    # Numbers
    entities.update(re.findall(r'\b\d+\.?\d*\b', text))
    return set(e.lower() for e in entities)

def ragas_faithfulness(answer: str, contexts: List[str]) -> float:
    """Faithfulness: fraction of answer claims supported by retrieved context."""
    answer_words = normalize_text(answer)
    context_words = normalize_text(" ".join(contexts))
    if not answer_words:
        return 0.0
    supported = answer_words & context_words
    return len(supported) / len(answer_words)

def ragas_answer_relevancy(question: str, answer: str) -> float:
    """Answer Relevancy: how relevant is the answer to the question."""
    q_words = normalize_text(question)
    a_words = normalize_text(answer)
    if not q_words:
        return 0.0
    # Use Jaccard similarity for relevancy
    intersection = len(q_words & a_words)
    union = len(q_words | a_words)
    return intersection / union if union > 0 else 0.0

def ragas_context_precision(question: str, contexts: List[str], ground_truth: str) -> float:
    """Context Precision: are relevant chunks ranked higher? Uses precision@k."""
    gt_words = normalize_text(ground_truth)
    precision_scores = []
    for i, ctx in enumerate(contexts):
        ctx_words = normalize_text(ctx)
        if not ctx_words:
            continue
        relevance = len(ctx_words & gt_words) / len(gt_words) if gt_words else 0
        precision_scores.append((i + 1, relevance))
    if not precision_scores:
        return 0.0
    # Average precision: weighted by rank position
    total = 0.0
    relevant_count = 0
    for rank, rel in precision_scores:
        if rel > 0:
            relevant_count += 1
            total += rel * (1.0 / rank)
    return total / max(len(precision_scores), 1)

def ragas_context_recall(ground_truth: str, contexts: List[str]) -> float:
    """Context Recall: did retrieval capture all info needed for ground truth?"""
    gt_words = normalize_text(ground_truth)
    all_ctx_words = normalize_text(" ".join(contexts))
    if not gt_words:
        return 0.0
    covered = gt_words & all_ctx_words
    return len(covered) / len(gt_words)

def ragas_context_entity_recall(ground_truth: str, contexts: List[str]) -> float:
    """Context Entity Recall: did retrieval capture entities from ground truth?"""
    gt_entities = extract_entities(ground_truth)
    ctx_entities = extract_entities(" ".join(contexts))
    if not gt_entities:
        return 1.0  # No entities to recall
    recalled = gt_entities & ctx_entities
    return len(recalled) / len(gt_entities)

# Run all 5 RAGAS-equivalent metrics
ragas_results = []
for r in rag_results:
    f = ragas_faithfulness(r["answer"], r["contexts"])
    ar = ragas_answer_relevancy(r["question"], r["answer"])
    cp = ragas_context_precision(r["question"], r["contexts"], r["ground_truth"])
    cr = ragas_context_recall(r["ground_truth"], r["contexts"])
    cer = ragas_context_entity_recall(r["ground_truth"], r["contexts"])
    ragas_results.append({
        "question": r["question"],
        "faithfulness": round(f, 3),
        "answer_relevancy": round(ar, 3),
        "context_precision": round(cp, 3),
        "context_recall": round(cr, 3),
        "context_entity_recall": round(cer, 3),
    })

ragas_df = pd.DataFrame(ragas_results)
print("\n=== RAGAS-Equivalent Metrics (All 5) ===")
display(ragas_df)

# Summary statistics
print("\n=== Metric Summary ===")
summary = ragas_df.drop(columns=["question"]).agg(['mean', 'std', 'min', 'max']).round(3)
display(summary)

# COMMAND ----------

# DBTITLE 1,MLflow Built-in RAG Scorers
# MAGIC %md
# MAGIC ## 4. MLflow GenAI Built-in RAG Scorers
# MAGIC
# MAGIC MLflow provides LLM-judge scorers that run on Databricks-hosted foundation models:
# MAGIC
# MAGIC | Scorer | Category | What It Judges |
# MAGIC |---|---|---|
# MAGIC | `RelevanceToQuery` | Response Quality | Is the response relevant to the user's query? |
# MAGIC | `Safety` | Safety | Does the response avoid harmful content? |
# MAGIC | `Guidelines` | Custom | Does the response follow custom guidelines? |
# MAGIC
# MAGIC All built-in LLM judge scorers return **binary yes/no** with a rationale.

# COMMAND ----------

# DBTITLE 1,MLflow GenAI Evaluate with Built-in Scorers
# ─── Scenario 2: MLflow GenAI Evaluation with Built-in RAG Scorers ───

# Build eval data in MLflow format
mlflow_eval_data = []
for r in rag_results:
    mlflow_eval_data.append({
        "inputs": {"query": r["question"]},
        "outputs": {"response": r["answer"], "context": r["contexts"]},
        "expectations": {"expected_response": r["ground_truth"]},
    })

print(f"Eval data: {len(mlflow_eval_data)} samples")

# Predict function wrapping the RAG pipeline
# Note: predict_fn parameter name must match the key in the inputs dict
def rag_predict_fn(query: str) -> dict:
    result = rag_pipeline(query)
    return {"response": result["response"], "context": result["retrieved_contexts"]}

try:
    result = mlflow.genai.evaluate(
        data=mlflow_eval_data,
        predict_fn=rag_predict_fn,
        scorers=[
            RelevanceToQuery(),
            Safety(),
            Guidelines(
                name="rag_guidelines",
                guidelines=[
                    "The response must be grounded in retrieved context, not hallucinated.",
                    "The response should directly answer the user's question.",
                ],
            ),
        ],
    )
    print("\n=== MLflow GenAI Evaluation Results ===")
    print(result)
    if hasattr(result, 'summary'):
        display(result.summary)
    elif hasattr(result, 'tables'):
        for t in result.tables:
            display(t)
except Exception as e:
    print(f"MLflow evaluate error: {e}")
    print("Ensure mlflow[databricks] is installed and a warehouse is available for LLM judges.")

# COMMAND ----------

# DBTITLE 1,Agent & Tool Call Evaluation
# MAGIC %md
# MAGIC ## 5. Agent & Tool Call Evaluation
# MAGIC
# MAGIC Evaluating agents that use tools requires checking:
# MAGIC * **Tool selection correctness**: Did the agent pick the right tool?
# MAGIC * **Tool input correctness**: Were the right parameters passed?
# MAGIC * **Tool output handling**: Did the agent use the tool output correctly?
# MAGIC * **End-to-end task completion**: Did the full pipeline solve the user's problem?
# MAGIC
# MAGIC MLflow tracing captures tool calls as `SpanType.TOOL` spans, and custom scorers can inspect the trace.

# COMMAND ----------

# DBTITLE 1,Agent & Tool Evaluation with Custom Scorers
# ─── Scenario 3: Agent & Tool Call Evaluation ───

# Mock tools for the agent
@mlflow.trace(span_type=SpanType.TOOL)
def search_tool(query: str) -> str:
    """Mock search tool."""
    if "weather" in query.lower():
        return json.dumps({"location": "SF", "temp": 65, "condition": "foggy"})
    elif "stock" in query.lower():
        return json.dumps({"ticker": "DBRX", "price": 42.50, "change": "+2.3%"})
    return json.dumps({"results": "No results found"})

@mlflow.trace(span_type=SpanType.TOOL)
def calculator_tool(expression: str) -> str:
    """Mock calculator tool."""
    try:
        result = eval(expression)  # Demo only — never eval untrusted input in prod
        return str(result)
    except:
        return "Error: invalid expression"

@mlflow.trace(span_type=SpanType.CHAIN)
def agent_router(query: str, session_id: str = "agent_001") -> Dict[str, Any]:
    """Mock agent that routes to the right tool."""
    mlflow.update_current_trace(metadata={
        "mlflow.trace.session": session_id,
        "mlflow.trace.user": "eval_user",
    })
    q_lower = query.lower()
    if "weather" in q_lower:
        tool_result = search_tool(query)
        return {"response": f"Weather: {json.loads(tool_result)['condition']}, {json.loads(tool_result)['temp']}F", "tool_used": "search_tool"}
    elif "stock" in q_lower:
        tool_result = search_tool(query)
        return {"response": f"Stock price: ${json.loads(tool_result)['price']}", "tool_used": "search_tool"}
    elif any(op in query for op in ["+", "-", "*", "/"]):
        tool_result = calculator_tool(query)
        return {"response": f"Result: {tool_result}", "tool_used": "calculator_tool"}
    else:
        return {"response": "I don't know how to handle that.", "tool_used": "none"}

# ─── Custom scorer: tool selection correctness ───
@scorer
def tool_selection_correctness(inputs, outputs, trace) -> Feedback:
    """Check if the agent selected the correct tool for the query."""
    query = str(inputs.get("query", "")).lower()
    tool_used = outputs.get("tool_used", "none") if isinstance(outputs, dict) else "none"
    expected_tool = "none"
    if "weather" in query or "stock" in query:
        expected_tool = "search_tool"
    elif any(op in query for op in ["+", "-", "*", "/"]):
        expected_tool = "calculator_tool"
    correct = tool_used == expected_tool
    return Feedback(
        value=correct,
        rationale=f"Expected '{expected_tool}', got '{tool_used}' — {'correct' if correct else 'wrong tool'}",
    )

# ─── Custom scorer: response latency ───
@scorer
def latency_under_5s(inputs, outputs, trace) -> Feedback:
    """Check if the response was generated within 5 seconds."""
    duration_ms = trace.info.execution_duration if trace and trace.info else 0
    passed = duration_ms <= 5000
    return Feedback(
        value=passed,
        rationale=f"Latency: {duration_ms}ms {'within' if passed else 'exceeds'} 5s SLA",
    )

# Agent eval dataset
agent_eval_data = [
    {"inputs": {"query": "What's the weather in SF?"}},
    {"inputs": {"query": "What's the stock price of DBRX?"}},
    {"inputs": {"query": "Calculate 25 + 17"}},
    {"inputs": {"query": "Tell me a joke"}},  # No tool needed
]

try:
    agent_result = mlflow.genai.evaluate(
        data=agent_eval_data,
        predict_fn=lambda query: agent_router(query),
        scorers=[
            RelevanceToQuery(),
            Safety(),
            tool_selection_correctness,
            latency_under_5s,
        ],
    )
    print("\n=== Agent Evaluation Results ===")
    print(agent_result)
    if hasattr(agent_result, 'summary'):
        display(agent_result.summary)
except Exception as e:
    print(f"Agent evaluation error: {e}")
    # Manual fallback
    for item in agent_eval_data:
        out = agent_router(item["inputs"]["query"])
        print(f"Q: {item['inputs']['query']} -> Tool: {out['tool_used']}, Response: {out['response'][:80]}")

# COMMAND ----------

# DBTITLE 1,Multi-Turn Conversation Evaluation
# MAGIC %md
# MAGIC ## 6. Multi-Turn Conversation Evaluation
# MAGIC
# MAGIC For conversational agents, evaluate entire sessions (not individual turns). Requires traces tagged with `mlflow.trace.session`.
# MAGIC
# MAGIC **Built-in conversation scorers:**
# MAGIC * `ConversationCompleteness` — Did the conversation resolve the user's need?
# MAGIC * `UserFrustration` — Did the user show signs of frustration?
# MAGIC
# MAGIC MLflow groups traces sharing a session ID into one conversation and scores it as a whole.

# COMMAND ----------

# DBTITLE 1,Multi-Turn Conversation Eval Code
# ─── Scenario 4: Multi-Turn Conversation Evaluation ───

# Simulate a multi-turn conversation with session IDs
conversation_turns = [
    {"query": "What is MLflow?", "session_id": "conv_001"},
    {"query": "Can it track experiments?", "session_id": "conv_001"},
    {"query": "How do I log a model?", "session_id": "conv_001"},
    {"query": "What's the weather?", "session_id": "conv_002"},
    {"query": "And stock price of DBRX?", "session_id": "conv_002"},
]

# Generate traces for each turn
print("Generating multi-turn traces...")
for turn in conversation_turns:
    result = rag_pipeline(turn["query"], session_id=turn["session_id"])
    print(f"  [{turn['session_id']}] Q: {turn['query']} -> A: {result['response'][:60]}...")

# Search for traces and check session IDs
exp_id = mlflow.get_experiment_by_name("/Users/prakhar1207srivastava@gmail.com/llm-evaluation-ragas").experiment_id
traces = mlflow.search_traces(
    experiment_ids=[exp_id],
    max_results=20,
    return_type="list",
)

has_sessions = any(
    t.info.trace_metadata.get("mlflow.trace.session") for t in traces
) if traces else False

print(f"\nTotal traces: {len(traces) if traces else 0}")
print(f"Session IDs present: {has_sessions}")

if has_sessions:
    # Group by session
    sessions = {}
    for t in traces:
        sid = t.info.trace_metadata.get("mlflow.trace.session", "unknown")
        sessions.setdefault(sid, []).append(t)

    print(f"\nSessions found: {list(sessions.keys())}")
    for sid, session_traces in sessions.items():
        print(f"  {sid}: {len(session_traces)} turns")

    # ─── Multi-turn evaluation ───
    try:
        from mlflow.genai.scorers import ConversationCompleteness, UserFrustration

        conv_result = mlflow.genai.evaluate(
            data=traces,
            scorers=[ConversationCompleteness(), UserFrustration()],
        )
        print("\n=== Multi-Turn Conversation Evaluation ===")
        print(conv_result)
        if hasattr(conv_result, 'summary'):
            display(conv_result.summary)
    except ImportError:
        print("Conversation scorers not available in this MLflow version.")
    except Exception as e:
        print(f"Conversation evaluation error: {e}")
else:
    print("\nNo session IDs found — instrument mlflow.trace.session to enable multi-turn evaluation.")

# COMMAND ----------

# DBTITLE 1,Production Monitoring & Comparison
# MAGIC %md
# MAGIC ## 7. Production Monitoring & Run Comparison
# MAGIC
# MAGIC **Production Monitoring**: Register scorers that automatically evaluate future traces at a sampling rate.
# MAGIC * Create -> Register (persistent name) -> Start (sampling config)
# MAGIC
# MAGIC **Run Comparison**: Compare two evaluation runs to find regressions/improvements.
# MAGIC * Score deltas, trace-level diffs, root-cause attribution

# COMMAND ----------

# DBTITLE 1,Production Monitoring Scheduled Scorers
# ─── Scenario 5: Production Monitoring with Scheduled Scorers ───

try:
    from mlflow.genai.scorers import ScorerSamplingConfig, get_scorer, list_scorers

    experiment_name = "/Users/prakhar1207srivastava@gmail.com/llm-evaluation-ragas"
    mlflow.set_experiment(experiment_name)

    # Register and start a safety scorer
    safety_scorer = Safety().register(name="rag_safety_monitor")
    safety_scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=0.5))
    print("Safety scorer registered and started (sample_rate=0.5)")

    # Register and start a guidelines scorer
    conciseness_scorer = Guidelines(
        name="rag_conciseness",
        guidelines=["The response should be concise and directly answer the question."],
    ).register(name="rag_conciseness_monitor")
    conciseness_scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=0.3))
    print("Conciseness scorer registered and started (sample_rate=0.3)")

    # List all active scorers
    print("\n=== Active Scorers ===")
    for s in list_scorers():
        try:
            name = getattr(s, '_server_name', None) or getattr(s, 'name', str(s))
            sr = getattr(s, 'sample_rate', 'unknown')
            print(f"  {name}: sample_rate={sr}")
        except Exception:
            print(f"  {s}")

    # Validate: run scorer on existing traces
    sample_traces = mlflow.search_traces(
        experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id],
        max_results=5,
        return_type="list",
    )

    if sample_traces:
        validation_result = mlflow.genai.evaluate(
            data=sample_traces,
            scorers=[get_scorer(name="rag_safety_monitor")],
        )
        print("\n=== Validation Run ===")
        print(validation_result)

except Exception as e:
    print(f"Production monitoring setup: {e}")
    print("Note: Scheduled scorers require a running warehouse and may need admin permissions.")

# COMMAND ----------

# DBTITLE 1,Compare Evaluation Runs & Summary
# ─── Scenario 6: Compare Two Evaluation Runs ───

experiment_id = mlflow.get_experiment_by_name(
    "/Users/prakhar1207srivastava@gmail.com/llm-evaluation-ragas"
).experiment_id

runs = mlflow.search_runs(experiment_ids=[experiment_id], max_results=10)

print(f"Found {len(runs)} runs in experiment")
if len(runs) > 0:
    display(runs[["run_id", "status", "start_time"]].head(10))

    if len(runs) >= 2:
        baseline_run = runs.iloc[-1]  # older run
        candidate_run = runs.iloc[0]   # newer run

        print(f"\n=== Run Comparison ===")
        print(f"Baseline:  {baseline_run['run_id']}")
        print(f"Candidate: {candidate_run['run_id']}")

        # Compare metrics
        metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
        if metric_cols:
            print("\n--- Metric Deltas ---")
            comparison = pd.DataFrame({
                "metric": [c.replace("metrics.", "") for c in metric_cols],
                "baseline": [baseline_run.get(c, None) for c in metric_cols],
                "candidate": [candidate_run.get(c, None) for c in metric_cols],
            })
            comparison["delta"] = comparison["candidate"] - comparison["baseline"]
            comparison["direction"] = comparison["delta"].apply(
                lambda x: "improved" if x > 0 else ("regressed" if x < 0 else "unchanged")
            )
            display(comparison)
        else:
            print("No metrics found in runs. Run evaluation cells first to generate scored runs.")
    else:
        print("\nNeed at least 2 runs for comparison. Re-run evaluation cells to generate more runs.")
else:
    print("No runs found yet. Run the evaluation cells above to generate runs.")

# ─── Summary of All Scenarios ───
print("\n" + "=" * 60)
print("EVALUATION SCENARIOS COVERED")
print("=" * 60)
scenarios = [
    "1. RAGAS metrics (faithfulness, answer relevancy, context precision/recall)",
    "2. MLflow GenAI built-in RAG scorers (RelevanceToQuery, Safety, Guidelines)",
    "3. Agent & tool-call evaluation with custom scorers",
    "4. Multi-turn conversation evaluation (session-based)",
    "5. Production monitoring with scheduled scorers",
    "6. Run comparison for regression detection",
]
for s in scenarios:
    print(f"  done {s}")
print("=" * 60)

# COMMAND ----------

# DBTITLE 1,Systematic Error Analysis
# MAGIC %md
# MAGIC ## 8. Systematic Error Analysis
# MAGIC
# MAGIC From *Evaluation for LLM Applications* Chapter 3: Systematic analysis of errors to find patterns and root causes.
# MAGIC
# MAGIC **Key Steps:**
# MAGIC * Categorize errors by type (retrieval failure, hallucination, safety, latency, formatting)
# MAGIC * Identify patterns across traces (common failure modes, affected user segments)
# MAGIC * Attribute errors to pipeline components (retriever vs generator vs tool)
# MAGIC * Prioritize fixes by frequency and impact

# COMMAND ----------

# DBTITLE 1,Systematic Error Analysis Code
# ─── Scenario 7: Systematic Error Analysis ───
# Analyze MLflow traces for error patterns, categorize, and find root causes

experiment_name = "/Users/prakhar1207srivastava@gmail.com/llm-evaluation-ragas"
exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# Fetch all traces
all_traces = mlflow.search_traces(
    experiment_ids=[exp_id],
    max_results=100,
    return_type="list",
)
print(f"Total traces for analysis: {len(all_traces)}")

# Categorize traces by type and detect errors
def analyze_trace(trace) -> Dict[str, Any]:
    """Extract metrics and error indicators from a trace."""
    info = trace.info
    spans = trace.data.spans if trace.data else []
    
    # Extract session and user
    session = info.trace_metadata.get("mlflow.trace.session", "unknown")
    user = info.trace_metadata.get("mlflow.trace.user", "unknown")
    
    # Identify span types
    span_types = [s.span_type for s in spans] if spans else []
    has_retriever = any("RETRIEVER" in str(st) for st in span_types)
    has_llm = any("LLM" in str(st) for st in span_types)
    has_tool = any("TOOL" in str(st) for st in span_types)
    has_chain = any("CHAIN" in str(st) for st in span_types)
    
    # Extract output
    output = ""
    if trace.data:
        # Try different output attribute names across MLflow versions
        for attr in ["outputs", "predictions", "output"]:
            val = getattr(trace.data, attr, None)
            if val:
                output = str(val)
                break
        if not output and trace.data.spans:
            # Get output from last span
            last_span = trace.data.spans[-1]
            for attr in ["outputs", "output", "predictions"]:
                val = getattr(last_span, attr, None)
                if val:
                    output = str(val)
                    break
    
    # Detect error indicators
    error_type = "none"
    if not output or len(output) < 10:
        error_type = "empty_output"
    elif "error" in output.lower() or "failed" in output.lower():
        error_type = "error_in_output"
    elif not has_retriever and not has_tool:
        error_type = "no_retrieval"
    elif has_retriever and not has_llm:
        error_type = "retrieval_only_no_generation"
    
    # Duration
    duration_ms = info.execution_duration if hasattr(info, 'execution_duration') else 0
    latency_category = "fast" if duration_ms < 1000 else ("medium" if duration_ms < 5000 else "slow")
    
    return {
        "trace_id": trace.info.trace_id,
        "session": session,
        "user": user,
        "span_types": ",".join(set(str(st).split(".")[-1] for st in span_types)),
        "has_retriever": has_retriever,
        "has_llm": has_llm,
        "has_tool": has_tool,
        "error_type": error_type,
        "duration_ms": duration_ms,
        "latency_category": latency_category,
        "output_length": len(output),
    }

# Analyze all traces
analysis_data = [analyze_trace(t) for t in all_traces]
analysis_df = pd.DataFrame(analysis_data)

print("\n=== Trace Analysis Summary ===")
print(f"Total traces: {len(analysis_df)}")
print(f"Sessions: {analysis_df['session'].nunique()}")
print(f"Error types: {analysis_df['error_type'].value_counts().to_dict()}")
print(f"Latency categories: {analysis_df['latency_category'].value_counts().to_dict()}")

# Error pattern analysis
print("\n=== Error Distribution ===")
error_dist = analysis_df.groupby(["error_type", "session"]).size().reset_index(name="count")
display(error_dist)

# Span type distribution
print("\n=== Span Type Distribution ===")
span_dist = analysis_df["span_types"].value_counts().reset_index()
span_dist.columns = ["span_combination", "count"]
display(span_dist)

# Latency analysis
print("\n=== Latency Analysis ===")
latency_stats = analysis_df.groupby("latency_category")["duration_ms"].agg(["count", "mean", "max"]).round(2)
display(latency_stats)

# Component attribution
print("\n=== Component Attribution ===")
component_stats = pd.DataFrame({
    "component": ["retriever", "llm", "tool", "chain"],
    "traces_with_component": [
        analysis_df["has_retriever"].sum(),
        analysis_df["has_llm"].sum(),
        analysis_df["has_tool"].sum(),
        analysis_df["span_types"].str.contains("CHAIN").sum(),
    ],
})
component_stats["percentage"] = (component_stats["traces_with_component"] / len(analysis_df) * 100).round(1)
display(component_stats)

# COMMAND ----------

# DBTITLE 1,Retrieval Quality Evaluation
# MAGIC %md
# MAGIC ## 9. Retrieval Quality Evaluation
# MAGIC
# MAGIC From *Evaluation for LLM Applications* Chapter 5: Evaluate the retrieval component separately from generation.
# MAGIC
# MAGIC **Retrieval Metrics:**
# MAGIC * **Precision@k**: Fraction of retrieved docs that are relevant
# MAGIC * **Recall@k**: Fraction of all relevant docs that were retrieved
# MAGIC * **MRR (Mean Reciprocal Rank)**: Average rank position of first relevant doc
# MAGIC * **NDCG@k**: Normalized Discounted Cumulative Gain (accounts for ranking quality)

# COMMAND ----------

# DBTITLE 1,Retrieval Quality Metrics
# ─── Scenario 8: Retrieval Quality Evaluation ───
# Evaluate the retrieval component independently using IR metrics

def precision_at_k(retrieved, relevant, k=3):
    top_k = retrieved[:k]
    relevant_set = set(normalize_text(" ".join(relevant)))
    relevant_count = 0
    for doc in top_k:
        doc_words = normalize_text(doc)
        if len(doc_words & relevant_set) / max(len(relevant_set), 1) > 0.3:
            relevant_count += 1
    return relevant_count / k

def recall_at_k(retrieved, relevant, k=3):
    relevant_set = set(normalize_text(" ".join(relevant)))
    if len(relevant_set) == 0:
        return 1.0
    retrieved_words = set(normalize_text(" ".join(retrieved[:k])))
    return len(retrieved_words & relevant_set) / len(relevant_set)

def mean_reciprocal_rank(retrieved, relevant):
    relevant_set = set(normalize_text(" ".join(relevant)))
    for i, doc in enumerate(retrieved):
        doc_words = normalize_text(doc)
        if len(doc_words & relevant_set) / max(len(relevant_set), 1) > 0.3:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(retrieved, relevant, k=3):
    relevant_set = set(normalize_text(" ".join(relevant)))
    dcg = sum(len(normalize_text(d) & relevant_set) / max(len(relevant_set), 1) / np.log2(i + 2) for i, d in enumerate(retrieved[:k]))
    ideal = sorted([len(normalize_text(d) & relevant_set) / max(len(relevant_set), 1) for d in retrieved], reverse=True)[:k]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

retrieval_eval = []
for item in eval_questions:
    result = rag_pipeline(item["question"])
    retrieved = result["retrieved_contexts"]
    gt_words = normalize_text(item["ground_truth"])
    relevant_docs = [d["text"] for d in KNOWLEDGE_BASE if len(normalize_text(d["text"]) & gt_words) > 3]
    retrieval_eval.append({
        "query": item["question"],
        "precision@3": round(precision_at_k(retrieved, relevant_docs), 3),
        "recall@3": round(recall_at_k(retrieved, relevant_docs), 3),
        "mrr": round(mean_reciprocal_rank(retrieved, relevant_docs), 3),
        "ndcg@3": round(ndcg_at_k(retrieved, relevant_docs), 3),
        "retrieved_count": len(retrieved),
        "relevant_count": len(relevant_docs),
    })

retrieval_df = pd.DataFrame(retrieval_eval)
print("\n=== Retrieval Quality Metrics ===")
display(retrieval_df)
print("\n=== Retrieval Quality Summary ===")
for m in ["precision@3", "recall@3", "mrr", "ndcg@3"]:
    print(f"  {m}: {retrieval_df[m].mean():.3f}")

# COMMAND ----------

# DBTITLE 1,Combined RAG Evaluation
# MAGIC %md
# MAGIC ## 10. Combined Retrieval + Generation Evaluation
# MAGIC
# MAGIC From *Evaluation for LLM Applications* Chapter 5: Holistic evaluation of the full RAG pipeline.
# MAGIC
# MAGIC **Joint Metrics:**
# MAGIC * **Faithfulness**: Answer grounded in retrieved sources (no hallucination)
# MAGIC * **Source Attribution**: Can facts in the answer be traced back to documents?
# MAGIC * **End-to-End Success Rate**: Does the system solve the user's task?
# MAGIC * **Error Attribution**: Errors from retrieval vs generation vs both

# COMMAND ----------

# DBTITLE 1,Combined RAG Eval Code
# ─── Scenario 9: Combined Retrieval + Generation Evaluation ───
# Holistic evaluation of the full RAG pipeline

def source_attribution(answer, contexts):
    """Check if answer claims can be traced to source documents."""
    answer_sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if len(s.strip()) > 5]
    attributed = 0
    unattributed = 0
    for sent in answer_sentences:
        sent_words = normalize_text(sent)
        found = False
        for ctx in contexts:
            ctx_words = normalize_text(ctx)
            if len(sent_words & ctx_words) / max(len(sent_words), 1) > 0.5:
                found = True
                break
        if found:
            attributed += 1
        else:
            unattributed += 1
    total = attributed + unattributed
    return {"attributed": attributed, "unattributed": unattributed, "rate": round(attributed / total, 3) if total > 0 else 0.0}

def end_to_end_success(answer, ground_truth):
    """Check if the RAG system produced a correct answer."""
    gt_words = normalize_text(ground_truth)
    ans_words = normalize_text(answer)
    if not gt_words:
        return False
    return len(gt_words & ans_words) / len(gt_words) > 0.5

def attribute_error_source(answer, contexts, ground_truth):
    """Attribute errors to retrieval, generation, or both."""
    retrieval_ok = ragas_context_recall(ground_truth, contexts) > 0.3
    generation_ok = ragas_faithfulness(answer, contexts) > 0.3
    if retrieval_ok and generation_ok:
        return "no_error"
    elif not retrieval_ok and not generation_ok:
        return "both"
    elif not retrieval_ok:
        return "retrieval_failure"
    else:
        return "generation_failure"

combined_results = []
for r in rag_results:
    attr = source_attribution(r["answer"], r["contexts"])
    success = end_to_end_success(r["answer"], r["ground_truth"])
    error_src = attribute_error_source(r["answer"], r["contexts"], r["ground_truth"])
    combined_results.append({
        "question": r["question"],
        "faithfulness": round(ragas_faithfulness(r["answer"], r["contexts"]), 3),
        "attribution_rate": attr["rate"],
        "attributed_claims": attr["attributed"],
        "unattributed_claims": attr["unattributed"],
        "end_to_end_success": success,
        "error_source": error_src,
    })

combined_df = pd.DataFrame(combined_results)
print("\n=== Combined RAG Evaluation ===")
display(combined_df)

print("\n=== Error Source Distribution ===")
for source, count in combined_df["error_source"].value_counts().items():
    print(f"  {source}: {count} ({count/len(combined_df)*100:.0f}%)")
print(f"\nEnd-to-End Success Rate: {combined_df['end_to_end_success'].mean():.1%}")
print(f"Average Attribution Rate: {combined_df['attribution_rate'].mean():.1%}")

# COMMAND ----------

# DBTITLE 1,Human Review & Cost Optimization
# MAGIC %md
# MAGIC ## 11. Human Review & Cost Optimization
# MAGIC
# MAGIC From *Evaluation for LLM Applications* Chapter 7: Balance automated evaluation with human review and manage costs.
# MAGIC
# MAGIC **Strategies:**
# MAGIC * **Stratified Sampling**: Sample traces by error type, session, or score for human review
# MAGIC * **Cost-Aware Evaluation**: Use cheaper scorers for high-volume, expensive LLM-judges for sampled subset
# MAGIC * **Confidence-Based Filtering**: Only send low-confidence cases to human reviewers
# MAGIC * **Active Learning**: Prioritize traces where automated scorers disagree

# COMMAND ----------

# DBTITLE 1,Human Review & Cost Optimization Code
# ─── Scenario 10: Human Review & Cost Optimization ───
import random
random.seed(42)
np.random.seed(42)

# Simulate 100 traces with evaluation scores
large_eval = []
for i in range(100):
    r = random.choice(rag_results)
    f = max(0, min(1, ragas_faithfulness(r["answer"], r["contexts"]) + np.random.normal(0, 0.15)))
    ar = max(0, min(1, ragas_answer_relevancy(r["question"], r["answer"]) + np.random.normal(0, 0.1)))
    large_eval.append({
        "trace_id": f"trace_{i:03d}",
        "question": r["question"],
        "faithfulness": round(f, 3),
        "answer_relevancy": round(ar, 3),
        "auto_score": round((f + ar) / 2, 3),
        "session": f"session_{random.choice(['A', 'B', 'C', 'D'])}",
    })

large_df = pd.DataFrame(large_eval)

# ─── Strategy 1: Stratified Sampling ───
print("=== Strategy 1: Stratified Sampling ===")
large_df["quartile"] = pd.qcut(large_df["auto_score"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
stratified = large_df.groupby("quartile", group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=42))
print(f"Total traces: {len(large_df)}, Sampled: {len(stratified)} ({len(stratified)/len(large_df)*100:.0f}%)")
display(stratified[["trace_id", "question", "auto_score", "quartile"]])

# ─── Strategy 2: Confidence-Based Filtering ───
print("\n=== Strategy 2: Confidence-Based Filtering ===")
low_conf = large_df[large_df["auto_score"] < 0.5]
high_conf = large_df[large_df["auto_score"] >= 0.5]
print(f"High confidence (auto-passed): {len(high_conf)} ({len(high_conf)/len(large_df)*100:.0f}%)")
print(f"Low confidence (human review): {len(low_conf)} ({len(low_conf)/len(large_df)*100:.0f}%)")
if len(low_conf) > 0:
    display(low_conf[["trace_id", "question", "auto_score"]].head(10))

# ─── Strategy 3: Cost-Aware Evaluation Tiers ───
print("\n=== Strategy 3: Cost-Aware Evaluation Tiers ===")
cost_tiers = pd.DataFrame([
    {"tier": "Tier 1: Cheap (regex/keyword)", "cost_per_trace": "$0.00", "coverage": "100%"},
    {"tier": "Tier 2: LLM-Judge (sampled)", "cost_per_trace": "$0.02", "coverage": "30%"},
    {"tier": "Tier 3: Human Review (stratified)", "cost_per_trace": "$0.50", "coverage": "6%"},
    {"tier": "Tier 4: Expert Review (edge cases)", "cost_per_trace": "$5.00", "coverage": "1%"},
])
display(cost_tiers)
total_cost = len(large_df) * 0.30 * 0.02 + len(large_df) * 0.06 * 0.50 + len(large_df) * 0.01 * 5.00
print(f"\nTotal cost for {len(large_df)} traces: ${total_cost:.2f}")
print(f"vs. Full human review: ${len(large_df) * 0.50:.2f} (saves {(1 - total_cost/(len(large_df)*0.50))*100:.0f}%)")

# ─── Strategy 4: Active Learning ───
print("\n=== Strategy 4: Active Learning (Scorer Disagreement) ===")
large_df["disagreement"] = abs(large_df["faithfulness"] - large_df["answer_relevancy"])
disagree = large_df[large_df["disagreement"] > 0.3].sort_values("disagreement", ascending=False)
print(f"Traces with significant scorer disagreement: {len(disagree)}")
if len(disagree) > 0:
    display(disagree[["trace_id", "question", "faithfulness", "answer_relevancy", "disagreement"]].head(10))

# ─── Final Summary ───
print("\n" + "=" * 60)
print("ALL EVALUATION SCENARIOS IMPLEMENTED")
print("=" * 60)
all_scenarios = [
    "1. RAGAS metrics (faithfulness, answer relevancy, context precision/recall, entity recall)",
    "2. MLflow GenAI built-in scorers (RelevanceToQuery, Safety, Guidelines)",
    "3. Agent & tool-call evaluation with custom scorers",
    "4. Multi-turn conversation evaluation (session-based)",
    "5. Production monitoring with scheduled scorers",
    "6. Run comparison for regression detection",
    "7. Systematic error analysis (trace-level error patterns)",
    "8. Retrieval quality evaluation (precision@k, recall@k, MRR, NDCG)",
    "9. Combined retrieval + generation evaluation (faithfulness, attribution, success)",
    "10. Human review & cost optimization (stratified sampling, cost tiers, active learning)",
]
for s in all_scenarios:
    print(f"  done {s}")
print(f"\n  Total: {len(all_scenarios)} scenarios")
print("=" * 60)

# COMMAND ----------

# DBTITLE 1,Retrieval Quality Evaluation
# MAGIC %md
# MAGIC ## 9. Retrieval Quality Evaluation
# MAGIC
# MAGIC From *Evaluation for LLM Applications* Chapter 5: Evaluate the retrieval component separately from generation.
# MAGIC
# MAGIC **Retrieval Metrics:**
# MAGIC * **Precision@k**: Fraction of retrieved docs that are relevant
# MAGIC * **Recall@k**: Fraction of all relevant docs that were retrieved
# MAGIC * **MRR (Mean Reciprocal Rank)**: Average rank position of first relevant doc
# MAGIC * **NDCG@k**: Normalized Discounted Cumulative Gain (accounts for ranking quality)
# MAGIC * **Relevance Score**: How useful is each retrieved document for the query?

# COMMAND ----------

# DBTITLE 1,Retrieval Quality Metrics
# ─── Scenario 8: Retrieval Quality Evaluation ───
# Evaluate the retrieval component independently using IR metrics

def precision_at_k(retrieved: List[str], relevant: List[str], k: int = 3) -> float:
    """Precision@k: fraction of top-k retrieved docs that are relevant."""
    top_k = retrieved[:k]
    relevant_set = set(normalize_text(" ".join(relevant)))
    relevant_count = 0
    for doc in top_k:
        doc_words = normalize_text(doc)
        overlap = len(doc_words & relevant_set)
        if overlap / max(len(relevant_set), 1) > 0.3:  # threshold for relevance
            relevant_count += 1
    return relevant_count / k

def recall_at_k(retrieved: List[str], relevant: List[str], k: int = 3) -> float:
    """Recall@k: fraction of relevant docs that were retrieved in top-k."""
    relevant_set = set(normalize_text(" ".join(relevant)))
    all_relevant = len(relevant_set)
    if all_relevant == 0:
        return 1.0
    retrieved_words = set(normalize_text(" ".join(retrieved[:k])))
    covered = len(retrieved_words & relevant_set)
    return covered / all_relevant

def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """MRR: 1/rank of first relevant document."""
    relevant_set = set(normalize_text(" ".join(relevant)))
    for i, doc in enumerate(retrieved):
        doc_words = normalize_text(doc)
        overlap = len(doc_words & relevant_set)
        if overlap / max(len(relevant_set), 1) > 0.3:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int = 3) -> float:
    """NDCG@k: normalized discounted cumulative gain."""
    relevant_set = set(normalize_text(" ".join(relevant)))
    # DCG
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        doc_words = normalize_text(doc)
        relevance = len(doc_words & relevant_set) / max(len(relevant_set), 1)
        dcg += relevance / np.log2(i + 2)  # +2 because log2(1) = 0
    # IDCG (ideal DCG)
    ideal_relevances = sorted([len(normalize_text(d) & relevant_set) / max(len(relevant_set), 1) 
                               for d in retrieved], reverse=True)[:k]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_relevances))
    return dcg / idcg if idcg > 0 else 0.0

# Build retrieval evaluation dataset
retrieval_eval = []
for item in eval_questions:
    query = item["question"]
    gt = item["ground_truth"]
    # Get retrieved docs from RAG pipeline
    result = rag_pipeline(query)
    retrieved = result["retrieved_contexts"]
    # All docs in KB are potential relevant docs
    all_docs = [d["text"] for d in KNOWLEDGE_BASE]
    # Determine which docs are relevant based on ground truth overlap
    gt_words = normalize_text(gt)
    relevant_docs = [d for d in all_docs if len(normalize_text(d) & gt_words) > 3]
    
    p_at_3 = precision_at_k(retrieved, relevant_docs, k=3)
    r_at_3 = recall_at_k(retrieved, relevant_docs, k=3)
    mrr = mean_reciprocal_rank(retrieved, relevant_docs)
    ndcg = ndcg_at_k(retrieved, relevant_docs, k=3)
    
    retrieval_eval.append({
        "query": query,
        "precision@3": round(p_at_3, 3),
        "recall@3": round(r_at_3, 3),
        "mrr": round(mrr, 3),
        "ndcg@3": round(ndcg, 3),
        "retrieved_count": len(retrieved),
        "relevant_count": len(relevant_docs),
    })

retrieval_df = pd.DataFrame(retrieval_eval)
print("\n=== Retrieval Quality Metrics ===")
display(retrieval_df)

# Overall retrieval quality
print("\n=== Retrieval Quality Summary ===")
retrieval_summary = retrieval_df.drop(columns=["query", "retrieved_count", "relevant_count"]).mean().round(3)
for metric, value in retrieval_summary.items():
    print(f"  {metric}: {value}")

# COMMAND ----------

# DBTITLE 1,Combined RAG Evaluation
# MAGIC %md
# MAGIC ## 10. Combined Retrieval + Generation Evaluation
# MAGIC
# MAGIC From *Evaluation for LLM Applications* Chapter 5: Holistic evaluation of the full RAG pipeline.
# MAGIC
# MAGIC **Joint Metrics:**
# MAGIC * **Faithfulness**: Answer grounded in retrieved sources (no hallucination)
# MAGIC * **Source Attribution**: Can facts in the answer be traced back to documents?
# MAGIC * **End-to-End Success Rate**: Does the system solve the user's task?
# MAGIC * **Retrieval-Generation Gap**: Errors from retrieval vs generation vs both

# COMMAND ----------

# DBTITLE 1,Combined RAG Eval Code
# ─── Scenario 9: Combined Retrieval + Generation Evaluation ───
# Holistic evaluation of the full RAG pipeline

def source_attribution(answer: str, contexts: List[str]) -> Dict[str, Any]:
    """Check if answer claims can be traced to specific source documents."""
    answer_sentences = re.split(r'[.!?]+', answer)
    answer_sentences = [s.strip() for s in answer_sentences if len(s.strip()) > 5]
    
    attributed = 0
    unattributed = 0
    for sent in answer_sentences:
        sent_words = normalize_text(sent)
        found = False
        for ctx in contexts:
            ctx_words = normalize_text(ctx)
            overlap = len(sent_words & ctx_words)
            if overlap / max(len(sent_words), 1) > 0.5:
                found = True
                break
        if found:
            attributed += 1
        else:
            unattributed += 1
    
    total = attributed + unattributed
    return {
        "attributed_claims": attributed,
        "unattributed_claims": unattributed,
        "attribution_rate": round(attributed / total, 3) if total > 0 else 0.0,
    }

def end_to_end_success(question: str, answer: str, ground_truth: str) -> bool:
    """Check if the end-to-end RAG system produced a correct answer."""
    gt_words = normalize_text(ground_truth)
    ans_words = normalize_text(answer)
    if not gt_words:
        return False
    overlap = len(gt_words & ans_words) / len(gt_words)
    return overlap > 0.5

def attribute_error_source(question, answer, contexts, ground_truth):
    """Attribute errors to retrieval, generation, or both."""
    retrieval_ok = ragas_context_recall(ground_truth, contexts) > 0.3
    generation_ok = ragas_faithfulness(answer, contexts) > 0.3
    
    if retrieval_ok and generation_ok:
        return "no_error"
    elif not retrieval_ok and not generation_ok:
        return "both_retrieval_and_generation"
    elif not retrieval_ok:
        return "retrieval_failure"
    else:
        return "generation_failure"

# Run combined evaluation
combined_results = []
for r in rag_results:
    attr = source_attribution(r["answer"], r["contexts"])
    success = end_to_end_success(r["question"], r["answer"], r["ground_truth"])
    error_source = attribute_error_source(r["question"], r["answer"], r["contexts"], r["ground_truth"])
    faith = ragas_faithfulness(r["answer"], r["contexts"])
    
    combined_results.append({
        "question": r["question"],
        "faithfulness": round(faith, 3),
        "attribution_rate": attr["attribution_rate"],
        "attributed_claims": attr["attributed_claims"],
        "unattributed_claims": attr["unattributed_claims"],
        "end_to_end_success": success,
        "error_source": error_source,
    })

combined_df = pd.DataFrame(combined_results)
print("\n=== Combined RAG Evaluation ===")
display(combined_df)

# Error source distribution
print("\n=== Error Source Distribution ===")
error_sources = combined_df["error_source"].value_counts()
for source, count in error_sources.items():
    print(f"  {source}: {count} ({count/len(combined_df)*100:.0f}%)")

# Overall success rate
success_rate = combined_df["end_to_end_success"].mean()
print(f"\nEnd-to-End Success Rate: {success_rate:.1%}")
print(f"Average Attribution Rate: {combined_df['attribution_rate'].mean():.1%}")

# COMMAND ----------

# DBTITLE 1,Human Review & Cost Optimization
# MAGIC %md
# MAGIC ## 11. Human Review & Cost Optimization
# MAGIC
# MAGIC From *Evaluation for LLM Applications* Chapter 7: Balance automated evaluation with human review and manage costs.
# MAGIC
# MAGIC **Strategies:**
# MAGIC * **Stratified Sampling**: Sample traces by error type, session, or score for human review
# MAGIC * **Cost-Aware Evaluation**: Use cheaper scorers for high-volume, expensive LLM-judges for sampled subset
# MAGIC * **Confidence-Based Filtering**: Only send low-confidence cases to human reviewers
# MAGIC * **Active Learning**: Prioritize traces where automated scorers disagree

# COMMAND ----------

# DBTITLE 1,Human Review & Cost Optimization Code
# ─── Scenario 10: Human Review & Cost Optimization ───
# Implement stratified sampling, cost-aware evaluation, and confidence-based filtering

import random
random.seed(42)

# Simulate a larger set of evaluation results with confidence scores
np.random.seed(42)
large_eval_results = []
for i in range(100):
    r = random.choice(rag_results)
    # Add noise to simulate real-world variation
    f = max(0, min(1, ragas_faithfulness(r["answer"], r["contexts"]) + np.random.normal(0, 0.15)))
    ar = max(0, min(1, ragas_answer_relevancy(r["question"], r["answer"]) + np.random.normal(0, 0.1)))
    large_eval_results.append({
        "trace_id": f"trace_{i:03d}",
        "question": r["question"],
        "faithfulness": round(f, 3),
        "answer_relevancy": round(ar, 3),
        "auto_score": round((f + ar) / 2, 3),
        "session": f"session_{random.choice(['A', 'B', 'C', 'D'])}",
    })

large_df = pd.DataFrame(large_eval_results)

# ─── Strategy 1: Stratified Sampling for Human Review ───
print("=== Strategy 1: Stratified Sampling ===")
# Sample 20% of traces, stratified by auto_score quartiles
large_df["score_quartile"] = pd.qcut(large_df["auto_score"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
stratified_sample = large_df.groupby("score_quartile", group_keys=False).apply(
    lambda x: x.sample(frac=0.2, random_state=42)
)
print(f"Total traces: {len(large_df)}")
print(f"Sampled for human review: {len(stratified_sample)} ({len(stratified_sample)/len(large_df)*100:.0f}%)")
print(f"Sampling cost reduction: {(1 - len(stratified_sample)/len(large_df))*100:.0f}%")
display(stratified_sample[["trace_id", "question", "auto_score", "score_quartile"]])

# ─── Strategy 2: Confidence-Based Filtering ───
print("\n=== Strategy 2: Confidence-Based Filtering ===")
# Only send low-confidence cases (auto_score < 0.5) to human reviewers
low_confidence = large_df[large_df["auto_score"] < 0.5]
high_confidence = large_df[large_df["auto_score"] >= 0.5]
print(f"High confidence (auto-passed): {len(high_confidence)} ({len(high_confidence)/len(large_df)*100:.0f}%)")
print(f"Low confidence (sent to human): {len(low_confidence)} ({len(low_confidence)/len(large_df)*100:.0f}%)")
if len(low_confidence) > 0:
    print("\nLow-confidence traces for human review:")
    display(low_confidence[["trace_id", "question", "auto_score"]].head(10))

# ─── Strategy 3: Cost-Aware Evaluation Tiers ───
print("\n=== Strategy 3: Cost-Aware Evaluation Tiers ===")
cost_tiers = pd.DataFrame([
    {"tier": "Tier 1: Cheap (regex/keyword)", "scorers": "manual_faithfulness, manual_relevancy", "cost_per_trace": "$0.00", "coverage": "100%"},
    {"tier": "Tier 2: LLM-Judge (sampled)", "scorers": "MLflow Safety, Guidelines, RelevanceToQuery", "cost_per_trace": "$0.02", "coverage": "30%"},
    {"tier": "Tier 3: Human Review (stratified)", "scorers": "Human reviewer", "cost_per_trace": "$0.50", "coverage": "6%"},
    {"tier": "Tier 4: Expert Review (edge cases)", "scorers": "Domain expert", "cost_per_trace": "$5.00", "coverage": "1%"},
])
display(cost_tiers)

# Calculate total cost
total_cost = (
    len(large_df) * 0.00 +  # Tier 1: all traces
    len(large_df) * 0.30 * 0.02 +  # Tier 2: 30% sampled
    len(large_df) * 0.06 * 0.50 +  # Tier 3: 6% human
    len(large_df) * 0.01 * 5.00     # Tier 4: 1% expert
)
print(f"\nTotal evaluation cost for {len(large_df)} traces: ${total_cost:.2f}")
print(f"Average cost per trace: ${total_cost/len(large_df):.4f}")
print(f"vs. Full human review: ${len(large_df) * 0.50:.2f} (saves {(1 - total_cost/(len(large_df)*0.50))*100:.0f}%)")

# ─── Strategy 4: Active Learning (scorer disagreement) ───
print("\n=== Strategy 4: Active Learning (Scorer Disagreement) ===")
# Find traces where faithfulness and relevancy disagree significantly
large_df["score_disagreement"] = abs(large_df["faithfulness"] - large_df["answer_relevancy"])
disagreement = large_df[large_df["score_disagreement"] > 0.3].sort_values("score_disagreement", ascending=False)
print(f"Traces with significant scorer disagreement: {len(disagreement)}")
print("These traces should be prioritized for human review (model uncertainty).")
if len(disagreement) > 0:
    display(disagreement[["trace_id", "question", "faithfulness", "answer_relevancy", "score_disagreement"]].head(10))

# ─── Final Summary: All Scenarios ───
print("\n" + "=" * 60)
print("ALL EVALUATION SCENARIOS IMPLEMENTED")
print("=" * 60)
all_scenarios = [
    "1. RAGAS metrics (faithfulness, answer relevancy, context precision/recall, entity recall)",
    "2. MLflow GenAI built-in scorers (RelevanceToQuery, Safety, Guidelines)",
    "3. Agent & tool-call evaluation with custom scorers",
    "4. Multi-turn conversation evaluation (session-based)",
    "5. Production monitoring with scheduled scorers",
    "6. Run comparison for regression detection",
    "7. Systematic error analysis (trace-level error patterns)",
    "8. Retrieval quality evaluation (precision@k, recall@k, MRR, NDCG)",
    "9. Combined retrieval + generation evaluation (faithfulness, attribution, success)",
    "10. Human review & cost optimization (stratified sampling, cost tiers, active learning)",
]
for s in all_scenarios:
    print(f"  done {s}")
print(f"\n  Total: {len(all_scenarios)} scenarios")
print("=" * 60)