# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Install packages from PyPI
# Install packages from PyPI (JFrog credentials not available)
%pip install \
    python-dotenv==1.2.1 \
    langchain==1.2.15 \
    langchain-openai==1.1.12 \
    langchain-community==0.4.1 \
    azure-identity==1.25.3 \
    openai==2.31.0 \
    --upgrade --force-reinstall

# COMMAND ----------

# get installed versions
import pkg_resources
packages = [
    "python-dotenv",
    "langchain",
    "langchain-openai",
    "langchain-community",
    "azure-identity",
    "openai"
]
for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f"{pkg}: {version}")
    except Exception as e:
        print(f"{pkg}: not installed")

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
os.getcwd()

# COMMAND ----------

# DBTITLE 1,Create MLflow Experiment
import mlflow

experiment_path = "/Users/prakhar1207srivastava@gmail.com/PromptCraft_Experiment"

# Create or set the experiment
experiment = mlflow.set_experiment(experiment_path)
print(f"Experiment Name: {experiment.name}")
print(f"Experiment ID: {experiment.experiment_id}")
print(f"Artifact Location: {experiment.artifact_location}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Data pull

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install "langchain_openai>=0.3,<0.4" "langchain-core>=0.3.45,<1.0" "langgraph>=0.2,<0.4" tenacity python-dotenv --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Cell 8
# TODO: Replace 'your_table_name' with the actual table name
# Example: df = spark.sql("""SELECT * FROM catalog.schema.table_name LIMIT 5""")
df = spark.sql("""SELECT 1 as sample_column LIMIT 5""")
display(df)

# COMMAND ----------

# DBTITLE 1,Setup imports and experiment
import mlflow
import pandas as pd
import json
import re
import ast

# Set the MLflow experiment
mlflow.set_experiment("/Users/prakhar1207srivastava@gmail.com/PromptCraft_Experiment")
print("Imports loaded and experiment set.")

# COMMAND ----------

# DBTITLE 1,Load prompts and define parser
# Placeholder output format prompt (original file not available)
OUTPUT_FORMAT_PROMPT = """
Please analyze the following transcript and provide your assessment in JSON format:
{
    "self_service_prediction": "met" or "not met" or "not applicable",
    "self_service_reason": "Brief explanation for your prediction"
}
"""

print(f"Using placeholder OUTPUT_FORMAT_PROMPT ({len(OUTPUT_FORMAT_PROMPT)} chars)")

# Placeholder prompts (original call_center_prompts folder not available)
# TODO: Add your actual prompt files to a 'call_center_prompts' folder
prompts = [
    {
        'filename': '00_Original.txt',
        'method_name': 'Original Baseline',
        'prompt_body': 'You are a quality assurance specialist reviewing call center transcripts. Evaluate if self-service education was provided to the customer.'
    }
]

print(f"Using {len(prompts)} placeholder prompt(s):")
for p in prompts:
    print(f"  - {p['filename']}: {p['method_name']}")

# Response parser (from self_service.py)
def normalize_prediction(pred):
    if not isinstance(pred, str):
        return "error"
    pred = pred.strip().lower().replace("_", " ")
    if "not" in pred and ("met" in pred or "meet" in pred):
        return "not met"
    if "met" in pred or "meet" in pred:
        return "met"
    if "not applicable" in pred or "n/a" in pred or "inapplicable" in pred:
        return "not applicable"
    # Handle alternate labels from some prompts
    if "education provided" in pred or "education" == pred.strip():
        return "met"
    if "no education" in pred or "no_education" in pred:
        return "not met"
    return "error"

def parse_response(output):
    """Parse GPT output into prediction + reason."""
    if not isinstance(output, str) or not output.strip():
        return {"self_service_prediction": "error", "self_service_reason": "empty response"}
    if output.strip().startswith("gpt error:"):
        return {"self_service_prediction": "error", "self_service_reason": output.strip()}

    clean = output.replace("```json", "").replace("```", "").strip()
    match = re.search(r'\{[\s\S]*\}', clean)
    if not match:
        return {"self_service_prediction": "error", "self_service_reason": "no JSON found ..."}

    json_str = match.group(0)
    parsed = None
    try:
        parsed = json.loads(json_str)
    except:
        try:
            parsed = ast.literal_eval(json_str)
        except:
            pass

    if parsed is None:
        return {"self_service_prediction": "error", "self_service_reason": "parse failure ..."}

    reason = parsed.get("self_service_reason", "")
    pred = normalize_prediction(parsed.get("self_service_prediction", ""))
    return {"self_service_prediction": pred, "self_service_reason": reason}

print("\nParser and utilities ready.")

# COMMAND ----------

# DBTITLE 1,Run experiment loop with MLflow
# Run all prompts against the 5 transcripts and log to MLflow
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

MODEL_NAME = 'gpt-5-nano_2025-08-07'

# Fixed API call that handles GPT-5 response format + captures ALL token types
def api_call_fixed(messages, model_name, verbosity="low", reasoning="low", timeout=120):
    from util_modularization import create_llm, get_access_token, get_model_cost
    get_access_token()
    model = create_llm(model_name, verbosity, reasoning, timeout)
    resp = model.invoke(messages)

    # Extract content - handle both single and multi-block responses
    if isinstance(resp.content, list):
        text_content = ""
        for block in resp.content:
            if isinstance(block, dict) and block.get('type') == 'text':
                text_content = block.get('text', '')
                break
        if not text_content and resp.content:
            text_content = str(resp.content[0].get('text', '')) if isinstance(resp.content[0], dict) else str(resp.content[0])
    else:
        text_content = resp.content or ""

    # Extract ALL token types
    prompt_tokens = resp.usage_metadata.get('input_tokens', 0)
    completion_tokens = resp.usage_metadata.get('output_tokens', 0)
    total_tokens = resp.usage_metadata.get('total_tokens', 0)
    cached_tokens = resp.usage_metadata.get('input_token_details', {}).get('cache_read', 0)
    reasoning_tokens = resp.usage_metadata.get('output_token_details', {}).get('reasoning', 0)
    if isinstance(reasoning_tokens, tuple):
        reasoning_tokens = reasoning_tokens[0]

    return {
        "model_name": model_name, "response": text_content,
        "total_tokens": total_tokens, "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens, "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_cost": get_model_cost(prompt_tokens, cached_tokens, completion_tokens, model_name)
    }

all_results = []
baseline_predictions = None  # Store Original prompt predictions as ground truth reference

for prompt_info in prompts:
    prompt_name = prompt_info['method_name']
    prompt_body = prompt_info['prompt_body']
    filename = prompt_info['filename']

    print(f"\n{'='*60}")
    print(f"Running: {prompt_name} ({filename})")
    print(f"{'='*60}")

    run_predictions = []
    run_total_tokens = 0
    run_prompt_tokens = 0
    run_completion_tokens = 0
    run_cached_tokens = 0
    run_reasoning_tokens = 0
    run_cost = 0.0
    prompt_start_time = time.time()

    with mlflow.start_run(run_name=prompt_name) as run:
        mlflow.log_param("prompt_method", prompt_name)
        mlflow.log_param("prompt_file", filename)
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("num_transcripts", df.count())
        mlflow.log_text(prompt_body, "prompt_system.txt")
        mlflow.log_text(OUTPUT_FORMAT_PROMPT, "prompt_user_template.txt")

        row_results = []
        latencies = []

        for idx, row_data in enumerate(df.collect()):
            transcript = row_data.get('transcript', '') or '' if hasattr(row_data, 'get') else (row_data['transcript'] if 'transcript' in row_data.asDict() else '')
            qfiniti_id = row_data.get('qfiniti_id', f'row_{idx}') if hasattr(row_data, 'get') else (row_data['qfiniti_id'] if 'qfiniti_id' in row_data.asDict() else f'row_{idx}')

            if len(str(transcript)) < 250:
                result = {"self_service_prediction": "error", "self_service_reason": "transcript too short"}
                response_meta = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                  "cached_tokens": 0, "reasoning_tokens": 0, "total_cost": 0.0}
                latency = 0.0
            else:
                msg = [
                    {"role": "system", "content": prompt_body},
                    {"role": "user", "content": f"{OUTPUT_FORMAT_PROMPT} {transcript}"}
                ]
                call_start = time.time()
                try:
                    response_meta = api_call_fixed(msg, MODEL_NAME, verbosity="low", reasoning="low", timeout=120)
                    gpt_output = response_meta.get('response', '')
                    result = parse_response(gpt_output)
                except Exception as e:
                    result = {"self_service_prediction": "error", "self_service_reason": f"api error: {str(e)}"}
                    response_meta = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                      "cached_tokens": 0, "reasoning_tokens": 0, "total_cost": 0.0}
                latency = time.time() - call_start

            latencies.append(latency)
            result['qfiniti_id'] = qfiniti_id
            result['latency_sec'] = round(latency, 2)
            result['total_tokens'] = response_meta.get('total_tokens', 0)
            result['prompt_tokens'] = response_meta.get('prompt_tokens', 0)
            result['completion_tokens'] = response_meta.get('completion_tokens', 0)
            result['cached_tokens'] = response_meta.get('cached_tokens', 0)
            result['reasoning_tokens'] = response_meta.get('reasoning_tokens', 0)
            result['cost'] = response_meta.get('total_cost', 0.0)
            row_results.append(result)

            # Accumulate token totals
            run_total_tokens += result['total_tokens']
            run_prompt_tokens += result['prompt_tokens']
            run_completion_tokens += result['completion_tokens']
            run_cached_tokens += result['cached_tokens']
            run_reasoning_tokens += result['reasoning_tokens']
            run_cost += result['cost']
            run_predictions.append(result['self_service_prediction'])

            print(f"  Row {idx}: {result['self_service_prediction']} | latency {latency:.1f}s | tokens ...")

        prompt_total_time = time.time() - prompt_start_time

        # Token Metrics
        mlflow.log_metric("total_tokens", run_total_tokens)
        mlflow.log_metric("prompt_tokens", run_prompt_tokens)
        mlflow.log_metric("completion_tokens", run_completion_tokens)
        mlflow.log_metric("cached_tokens", run_cached_tokens)
        mlflow.log_metric("reasoning_tokens", run_reasoning_tokens)
        mlflow.log_metric("total_cost", run_cost)
        mlflow.log_metric("avg_tokens_per_call", run_total_tokens / max(df.count(), 1))

        # Performance / Timing Metrics
        mlflow.log_metric("total_time_sec", round(prompt_total_time, 2))
        mlflow.log_metric("avg_latency_sec", round(sum(latencies) / max(len(latencies), 1), 2))
        mlflow.log_metric("max_latency_sec", round(max(latencies) if latencies else 0, 2))
        mlflow.log_metric("min_latency_sec", round(min(latencies) if latencies else 0, 2))

        # Prediction Distribution
        mlflow.log_metric("num_met", run_predictions.count("met"))
        mlflow.log_metric("num_not_met", run_predictions.count("not met"))
        mlflow.log_metric("num_not_applicable", run_predictions.count("not applicable"))
        mlflow.log_metric("num_errors", run_predictions.count("error"))
        mlflow.log_metric("error_rate", run_predictions.count("error") / max(len(run_predictions), 1))

        # Accuracy Metrics (vs Original baseline as reference)
        if filename == '00_Original.txt':
            baseline_predictions = run_predictions.copy()
            mlflow.log_metric("accuracy_vs_baseline", 1.0)
            mlflow.log_metric("precision_weighted", 1.0)
            mlflow.log_metric("recall_weighted", 1.0)
            mlflow.log_metric("f1_weighted", 1.0)
        elif baseline_predictions is not None:
            # Compare against baseline (exclude errors from both for fair comparison)
            valid_pairs = [(b, p) for b, p in zip(baseline_predictions, run_predictions) if b != 'error' and p != 'error']
            if valid_pairs:
                y_true = [v[0] for v in valid_pairs]
                y_pred = [v[1] for v in valid_pairs]
                labels = sorted(set(y_true + y_pred))
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
                rec = recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
                mlflow.log_metric("accuracy_vs_baseline", round(acc, 4))
                mlflow.log_metric("precision_weighted", round(prec, 4))
                mlflow.log_metric("recall_weighted", round(rec, 4))
                mlflow.log_metric("f1_weighted", round(f1, 4))
                print(f"  Accuracy vs Baseline: {acc:.2%} | Prec: {prec:.2%} | Recall: {rec:.2%} | F1: {f1:.2%}")
            else:
                mlflow.log_metric("accuracy_vs_baseline", 0.0)
                mlflow.log_metric("precision_weighted", 0.0)
                mlflow.log_metric("recall_weighted", 0.0)
                mlflow.log_metric("f1_weighted", 0.0)
                print("  No valid pairs for accuracy (all errors)")

        # Log detailed results as artifact
        results_df = pd.DataFrame(row_results)
        results_df.to_csv("/tmp/prompt_results.csv", index=False)
        mlflow.log_artifact("/tmp/prompt_results.csv", artifact_path="results")

        all_results.append({
            'prompt_method': prompt_name, 'filename': filename,
            'predictions': run_predictions,
            'total_tokens': run_total_tokens, 'prompt_tokens': run_prompt_tokens,
            'completion_tokens': run_completion_tokens, 'cached_tokens': run_cached_tokens,
            'reasoning_tokens': run_reasoning_tokens,
            'total_cost': run_cost, 'total_time_sec': round(prompt_total_time, 2),
            'avg_latency_sec': round(sum(latencies) / max(len(latencies), 1), 2),
            'run_id': run.info.run_id
        })

        print(f"  Run ID: {run.info.run_id}")
        print(f"  Predictions: met={run_predictions.count('met')}, not_met={run_predictions.count('not met')}, na=...")
        print(f"  Tokens: total={run_total_tokens}, prompt={run_prompt_tokens}, completion={run_completion_tokens}, cached={run_cached_tokens}")
        print(f"  Cost: ${run_cost:.6f} | Time: {prompt_total_time:.1f}s (avg {sum(latencies)/max(len(latencies),1):.1f}s/call)")

print(f"\n{'='*60}")
print(f"EXPERIMENT COMPLETE - {len(prompts)} prompts evaluated")
print(f"{'='*60}")

# COMMAND ----------

# DBTITLE 1,Results summary comparison
# Summary of results across all prompt strategies
summary_df = pd.DataFrame(all_results)

# Prediction distribution
summary_df['met'] = summary_df['predictions'].apply(lambda x: x.count('met'))
summary_df['not_met'] = summary_df['predictions'].apply(lambda x: x.count('not met'))
summary_df['not_applicable'] = summary_df['predictions'].apply(lambda x: x.count('not applicable'))
summary_df['errors'] = summary_df['predictions'].apply(lambda x: x.count('error'))

# Accuracy vs baseline
def calc_accuracy(row):
    if baseline_predictions is None or row['filename'] == '00_Original.txt':
        return 1.0
    valid_pairs = [(b, p) for b, p in zip(baseline_predictions, row['predictions']) if b != 'error' and p != 'error']
    if not valid_pairs:
        return 0.0
    return accuracy_score([v[0] for v in valid_pairs], [v[1] for v in valid_pairs])

summary_df['accuracy_vs_baseline'] = summary_df.apply(calc_accuracy, axis=1)

# Display table
display_cols = ['prompt_method', 'total_tokens', 'prompt_tokens', 'completion_tokens',
                 'cached_tokens', 'reasoning_tokens', 'total_cost', 'total_time_sec',
                 'avg_latency_sec', 'met', 'not_met', 'not_applicable', 'errors', 'accuracy_vs_baseline']
result_display = summary_df[display_cols].copy()
result_display['total_cost'] = result_display['total_cost'].apply(lambda x: f"${x:.6f}")
result_display['accuracy_vs_baseline'] = result_display['accuracy_vs_baseline'].apply(lambda x: f"{x:.2%}")
result_display = result_display.rename(columns={
    'prompt_method': 'Prompt Method', 'total_tokens': 'Total Tokens',
    'prompt_tokens': 'Prompt Tok', 'completion_tokens': 'Completion Tok',
    'cached_tokens': 'Cached Tok', 'reasoning_tokens': 'Reasoning Tok',
    'total_cost': 'Cost', 'total_time_sec': 'Time (s)',
    'avg_latency_sec': 'Avg Latency', 'accuracy_vs_baseline': 'Accuracy vs Baseline'
})

display(result_display)