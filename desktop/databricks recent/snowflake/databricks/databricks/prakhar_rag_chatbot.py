# Databricks notebook source
# DBTITLE 1,Prakhar RAG Chatbot
# MAGIC %md
# MAGIC # Prakhar RAG Chatbot
# MAGIC
# MAGIC Minimal RAG pipeline: 2 dummy chunks → vector search index → serving endpoint for chat. All resources prefixed with `prakhar_`.

# COMMAND ----------

# DBTITLE 1,Create Chunks Table
spark.sql("""
CREATE OR REPLACE TABLE workspace.default.prakhar_chunks (
  chunk_id STRING NOT NULL,
  text STRING,
  source STRING,
  category STRING,
  chunk_index INT
) USING DELTA
""")

spark.sql("""
INSERT INTO workspace.default.prakhar_chunks VALUES
('prakhar_chunk_001', 'Databricks is a unified data analytics platform that combines data engineering, data science, and business intelligence. It provides a collaborative workspace for processing big data using Apache Spark.', 'databricks_overview', 'platform', 0),
('prakhar_chunk_002', 'Vector Search in Databricks enables semantic search by converting text into embeddings and finding similar content. It supports Delta Sync indexes that automatically sync with source Delta tables.', 'vector_search_guide', 'features', 1)
""")

spark.sql("ALTER TABLE workspace.default.prakhar_chunks SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")

try:
    spark.sql("ALTER TABLE workspace.default.prakhar_chunks ADD CONSTRAINT prakhar_pk PRIMARY KEY (chunk_id)")
    print("Primary key constraint added")
except Exception as e:
    print(f"PK note: {e}")

display(spark.table("workspace.default.prakhar_chunks"))

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# No pip install needed — using databricks.sdk (pre-installed) for vector search

# COMMAND ----------

# DBTITLE 1,Create Vector Search Endpoint & Index
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    EndpointType, VectorIndexType, PipelineType,
    DeltaSyncVectorIndexSpecRequest, EmbeddingSourceColumn
)
import time

w = WorkspaceClient()

vs_endpoint_name = "prakhar_vs_endpoint"
vs_index_name = "workspace.default.prakhar_chunks_index"

# Create or get vector search endpoint
try:
    w.vector_search_endpoints.create_endpoint(name=vs_endpoint_name, endpoint_type=EndpointType.STANDARD)
    print(f"Creating endpoint: {vs_endpoint_name}")
except Exception as e:
    if "already" in str(e).lower():
        print(f"Endpoint {vs_endpoint_name} already exists")
    else:
        raise e

# Wait for endpoint to be ready
print("Waiting for endpoint...")
for i in range(20):
    time.sleep(15)
    try:
        status = w.vector_search_endpoints.get_endpoint(endpoint_name=vs_endpoint_name)
        state = status.endpoint_status.state
        print(f"  [{(i+1)*15}s] Endpoint state: {state}")
        if "ONLINE" in str(state):
            break
    except Exception as e:
        print(f"  [{(i+1)*15}s] Status: {e}")

# Create or get index
try:
    w.vector_search_indexes.get_index(index_name=vs_index_name)
    print(f"Index {vs_index_name} already exists")
except:
    print(f"Creating index: {vs_index_name}")
    w.vector_search_indexes.create_index(
        name=vs_index_name,
        endpoint_name=vs_endpoint_name,
        primary_key="chunk_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table="workspace.default.prakhar_chunks",
            embedding_source_columns=[
                EmbeddingSourceColumn(name="text", embedding_model_endpoint_name="databricks-gte-large-en")
            ],
            pipeline_type=PipelineType.TRIGGERED
        )
    )

# Trigger sync
try:
    w.vector_search_indexes.sync_index(index_name=vs_index_name)
    print("Sync triggered")
except Exception as e:
    print(f"Sync note: {e}")

# Wait for index to be ready
print("Waiting for index...")
for i in range(20):
    time.sleep(15)
    try:
        desc = w.vector_search_indexes.get_index(index_name=vs_index_name)
        state = desc.status.ready
        print(f"  [{(i+1)*15}s] Index ready: {state}")
        if state:
            break
    except Exception as e:
        print(f"  [{(i+1)*15}s] Status: {e}")

print(f"\nVector search endpoint: {vs_endpoint_name}")
print(f"Vector search index: {vs_index_name}")

# COMMAND ----------

# DBTITLE 1,Test Similarity Search
results = w.vector_search_indexes.query_index(
    index_name=vs_index_name,
    columns=["text", "source", "category", "chunk_index"],
    query_text="What is Databricks?",
    num_results=2
)

print("Similarity search results (top 2):")
for i, row in enumerate(results.result.data_array):
    print(f"\n--- Result {i+1} ---")
    print(f"Text: {row[0][:200]}...")
    print(f"Source: {row[1]}")
    print(f"Category: {row[2]}")
    print(f"Score: {row[4] if len(row) > 4 else 'N/A'}")

# COMMAND ----------

# DBTITLE 1,Build & Register RAG Model
import mlflow
from mlflow.models import infer_signature
import pandas as pd

class PrakharRAGChatbot(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import os
        self.host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
        self.token = os.environ.get("DATABRICKS_TOKEN", "")
        self.vs_endpoint = "prakhar_vs_endpoint"
        self.vs_index = "workspace.default.prakhar_chunks_index"
        self.llm_model = "databricks-meta-llama-3-3-70b-instruct"
    
    def predict(self, context, model_input, params=None):
        import os, requests
        if isinstance(model_input, pd.DataFrame):
            query = model_input.iloc[0]["query"]
        elif isinstance(model_input, dict):
            query = model_input.get("query", str(model_input))
        else:
            query = str(model_input)
        
        host = os.environ.get("DATABRICKS_HOST", self.host).rstrip("/")
        token = os.environ.get("DATABRICKS_TOKEN", self.token)
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        
        # Retrieve top 2 chunks via Vector Search REST API
        try:
            vs_url = f"{host}/api/2.0/vector-search/endpoints/{self.vs_endpoint}/indexes/{self.vs_index}/query"
            vs_resp = requests.post(vs_url, headers=headers, json={
                "columns": ["text", "source", "category", "chunk_index"],
                "query_text": query,
                "num_results": 2
            })
            chunks = vs_resp.json().get("result", {}).get("data_array", [])
            context_text = "\n\n".join([row[0] for row in chunks if row[0]])
        except Exception as e:
            context_text = ""
            chunks = []
        
        # Build prompt
        system_prompt = "You are a helpful assistant. Answer questions based on the provided context. Be concise and accurate. If the context is insufficient, say so."
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer based on the context above."
        
        # Call LLM via Foundation Model REST API
        llm_url = f"{host}/serving-endpoints/{self.llm_model}/invocations"
        llm_resp = requests.post(llm_url, headers=headers, json={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.1
        })
        answer = llm_resp.json()["choices"][0]["message"]["content"]
        
        return {"response": answer, "query": query, "num_sources": len(chunks)}

# Log model
mlflow.set_experiment("/Users/prakhar1207srivastava@gmail.com/prakhar_rag_chatbot")

with mlflow.start_run(run_name="prakhar_rag_chatbot_v4") as run:
    input_example = pd.DataFrame({"query": ["What is Databricks?"]})
    signature = infer_signature(
        model_input=input_example,
        model_output=pd.DataFrame({"response": ["Databricks is a unified analytics platform."], "query": ["What is Databricks?"], "num_sources": [2]})
    )
    
    model_info = mlflow.pyfunc.log_model(
        artifact_path="prakhar_rag_chatbot",
        python_model=PrakharRAGChatbot(),
        signature=signature,
        input_example=input_example,
    )
    
    model_uri = model_info.model_uri
    print(f"Model logged: {model_uri}")
    print(f"Run ID: {run.info.run_id}")
    
    # Register in UC
    model_name = "workspace.default.prakhar_rag_chatbot"
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Model registered: {model_name} (version {registered.version})")

# COMMAND ----------

# DBTITLE 1,Create Serving Endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
import time

w = WorkspaceClient()
serving_endpoint_name = "prakhar_chatbot_endpoint"
model_name = "workspace.default.prakhar_rag_chatbot"

try:
    w.serving_endpoints.create(
        name=serving_endpoint_name,
        config=EndpointCoreConfigInput(
            name=serving_endpoint_name,
            served_entities=[
                ServedEntityInput(
                    name="prakhar_rag_chatbot",
                    entity_name=model_name,
                    entity_version="3",
                    workload_size="Small",
                    scale_to_zero_enabled=True,
                )
            ]
        )
    )
    print(f"Creating serving endpoint: {serving_endpoint_name}")
except Exception as e:
    if "already" in str(e).lower():
        print(f"Endpoint {serving_endpoint_name} already exists — updating to version 3")
        w.serving_endpoints.update_config(
            name=serving_endpoint_name,
            served_entities=[
                ServedEntityInput(
                    name="prakhar_rag_chatbot",
                    entity_name=model_name,
                    entity_version="3",
                    workload_size="Small",
                    scale_to_zero_enabled=True,
                )
            ]
        )
        print("Config updated")
    else:
        print(f"Creation result: {e}")

# Wait for endpoint
print("Waiting for serving endpoint...")
for i in range(30):
    time.sleep(15)
    try:
        status = w.serving_endpoints.get(name=serving_endpoint_name)
        state = status.state.ready
        print(f"  [{(i+1)*15}s] State: {state}")
        if str(state.value) == "READY":
            break
    except Exception as e:
        print(f"  [{(i+1)*15}s] Status: {e}")

print(f"\nServing endpoint: {serving_endpoint_name}")
print(f"Model: {model_name}")

# COMMAND ----------

# DBTITLE 1,Test Chat Endpoint
import requests, json

host = dbutils.notebook.entry_point.getDbutils().notebook.getContext().apiUrl().get()
token = dbutils.notebook.entry_point.getDbutils().notebook.getContext().apiToken().get()

test_query = "What is Databricks Vector Search?"
response = requests.post(
    f"{host}/serving-endpoints/{serving_endpoint_name}/invocations",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json={"dataframe_records": [{"query": test_query}]}
)

result = response.json()
print(f"Query: {test_query}")
print(f"Response: {json.dumps(result, indent=2)}")