# Databricks notebook source
# MAGIC %md
# MAGIC ### A. Setup environment

# COMMAND ----------

# DBTITLE 1,Install dependencies - Databricks agents, SDK and MLflow
# MAGIC %pip install databricks-agents
# MAGIC %pip install databricks-sdk --upgrade
# MAGIC %pip install mlflow[databricks]
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### B. Run 'RAG Model' Notebook
# MAGIC
# MAGIC This creates all methods related to RAG steps, which can then be accessed in this notebook

# COMMAND ----------

# DBTITLE 1,Call RAG Model notebook
# MAGIC %run "./RAG Model"

# COMMAND ----------

# MAGIC %md
# MAGIC ### C. Create MLflow experiment

# COMMAND ----------

# DBTITLE 1,Define experiment name and create experiment
# Define experiment name
experimentName = "/Workspace/Databricks RAG/RAG Experiment"

# Create experiment
experimentId = mlflow.create_experiment(experimentName)

# COMMAND ----------

# MAGIC %md
# MAGIC ### D. Track and trace RAG pipeline using MLflow experiment

# COMMAND ----------

# DBTITLE 1,Create MLflow run to execute RAG pipeline
from mlflow.models import infer_signature

# Set active experiment
mlflow.set_experiment(experimentName)

# Define RAG pipeline parameters
numOfResults = 3
scoreThreshold = 0.8

# Create MLflow run
with mlflow.start_run(run_name="rag-pipeline-run") as run:

    # Input user query
    input = {"messages": [{"role": "user", "content": "What is the coverage under permanent loss of speech?"}]}

    query = input["messages"][0]["content"]

    # Retrieval: Extract chunks from vector search    
    chunks = retrieveChunks(query)

    # Prepare context by combining chunks
    chunksContext = prepareContext(chunks)

    # Augmentation: Prepare prompt with user query and context
    prompt = buildPrompt(query, chunksContext)

    # Generation: Call GenAI model
    answer = generateAnswer(systemMessage, prompt)

    print(answer)

    # Track parameters
    mlflow.log_param("model",           llmModelName)
    mlflow.log_param("top_k",           numOfResults)
    mlflow.log_param("score_threshold", scoreThreshold)

    # Track metrics
    metrics = {"retrieved_docs": len(chunks)}              # Other metrics like latency etc. can be logged here
    mlflow.log_metrics(metrics)

    # Trace RAG steps
    mlflow.log_text(query,         "query.txt")
    mlflow.log_text(chunksContext, "context.txt")
    mlflow.log_text(prompt,        "prompt.txt")
    mlflow.log_text(answer,        "response.txt")
    
    mlflow.log_dict({"citations": chunks}, "citations.json")

    # Print Run Id
    runId = run.info.run_id
    print(f"Run Id: {runId}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### E. Evaluate RAG pipeline using MLflow experiment

# COMMAND ----------

# DBTITLE 1,Evaluate RAG for correctness, relevance & safety
import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety, Correctness
import pandas as pd

# Set active experiment
mlflow.set_experiment(experimentName)

# Method to call RAG pipeline
def predict(messages):
    query = messages[0]["content"]
    answer = ragPipeline(query)
    return answer

# Define evaluation data
data = [
    {
        "inputs": {  "messages": [{"role": "user", "content": "What is the coverage under permanent loss of speech?"}]  },
        "expectations": {"expected_facts": ["70%"]}
    },
    {
        "inputs": {  "messages": [{"role": "user", "content": "How many days are covered under pre-hospitalization?"}]  },
        "expectations": {"expected_facts": ["90 days"]}
    },
    {
        "inputs": {  "messages": [{"role": "user", "content": "What policies are available from Contoso?"}]  },
        "expectations": {"expected_facts": ["contoso premier policy"]}        
    }
]

# Evaluate RAG pipeline
results = mlflow.genai.evaluate(
                                    data       = data,
                                    predict_fn = predict,
                                    scorers    = [
                                                    RelevanceToQuery(),     # Checks if response is relevant to query
                                                    Safety(),               # Checks for harmful content
                                                    Correctness()           # Checks for correctness against expected data
                                                 ]
                               )

# COMMAND ----------

# MAGIC %md
# MAGIC ### F. Register RAG pipeline as model in Unity Catalog using MLflow

# COMMAND ----------

# DBTITLE 1,Register RAG pipeline as Python Model
from mlflow.models import infer_signature

# Set active experiment
mlflow.set_experiment(experimentName)

# Create MLflow run
with mlflow.start_run(run_name="rag-pipeline-model-register") as run:
    
    # Log the model (Model gets registered if registered model name is passed)
    modelInfo = mlflow.pyfunc.log_model(
                                          name = "InsuranceRagModel",                # Model name in registry

                                          python_model = "RAG Model",                # Notebook with RAG Model details

                                          registered_model_name = "insurance.rag.insurancemodel", # Model name in catalog

                                          signature = infer_signature(input, answer) # Input and output format for model
                                       )

# Print Run Id and Model URI
print(f"MLflow Run: {modelInfo.run_id}")
print(f"Model URI: {modelInfo.model_uri}")

# COMMAND ----------

