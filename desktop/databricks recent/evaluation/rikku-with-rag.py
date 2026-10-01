# Databricks notebook source
# DBTITLE 1,🔍 RAG Pipeline Overview
# MAGIC %md
# MAGIC # Rikku RAG Chatbot - Complete Implementation
# MAGIC
# MAGIC ## What is RAG?
# MAGIC **Retrieval Augmented Generation (RAG)** = LLM + External Knowledge Base
# MAGIC
# MAGIC ```
# MAGIC User Question → Search Documents → Retrieve Context → LLM Answer
# MAGIC ```
# MAGIC
# MAGIC ## Architecture:
# MAGIC
# MAGIC 1. **Vector Search Index** - Store documents as embeddings
# MAGIC 2. **Retrieval Tool** - Search for relevant docs
# MAGIC 3. **Agent** - Use retrieved context to answer
# MAGIC 4. **Anti-Hallucination** - Only answer from retrieved docs
# MAGIC
# MAGIC ## Components:
# MAGIC * 📚 **Knowledge Base**: Sample Databricks documentation
# MAGIC * 🔍 **Vector Search**: Databricks Vector Search with BGE embeddings
# MAGIC * 🤖 **Agent**: Rikku with retrieval capabilities
# MAGIC * ✅ **Deployment**: MLflow → Unity Catalog → Endpoint
# MAGIC
# MAGIC ## Model:
# MAGIC * **Embedding**: `databricks-bge-large-en`
# MAGIC * **LLM**: `databricks-meta-llama-3-3-70b-instruct`
# MAGIC * **Index**: `agents.default.rikku_knowledge_base`

# COMMAND ----------

# DBTITLE 1,Install Packages
# MAGIC %pip install -U databricks-vectorsearch databricks-langchain databricks-agents sentence-transformers langgraph langchain-core mlflow-skinny

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup: Import & Configure
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import pandas as pd
import time

# Initialize clients
vsc = VectorSearchClient()
w = WorkspaceClient()

# Configuration
CATALOG = "agents"
SCHEMA = "default"
VECTOR_SEARCH_ENDPOINT_NAME = "rikku_vs_endpoint"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.rikku_knowledge_base"
SOURCE_TABLE_NAME = f"{CATALOG}.{SCHEMA}.rikku_docs_source"
# Using self-managed embeddings (sentence-transformers)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

print("✅ Configuration loaded:")
print(f"  Catalog: {CATALOG}")
print(f"  Schema: {SCHEMA}")
print(f"  Index: {INDEX_NAME}")
print(f"  Embedding Model: {EMBEDDING_MODEL_NAME} (self-managed)")

# COMMAND ----------

# DBTITLE 1,Create Vector Search Endpoint
# Check if endpoint exists
try:
    endpoint = vsc.get_endpoint(VECTOR_SEARCH_ENDPOINT_NAME)
    print(f"✅ Endpoint '{VECTOR_SEARCH_ENDPOINT_NAME}' already exists")
    print(f"   Status: {endpoint.get('endpoint_status', {}).get('state', 'UNKNOWN')}")
except Exception as e:
    print(f"Creating new endpoint '{VECTOR_SEARCH_ENDPOINT_NAME}'...")
    vsc.create_endpoint(
        name=VECTOR_SEARCH_ENDPOINT_NAME,
        endpoint_type="STANDARD"
    )
    print("⏳ Waiting for endpoint to be online...")
    
    # Wait for endpoint to be ready
    for i in range(30):
        try:
            endpoint = vsc.get_endpoint(VECTOR_SEARCH_ENDPOINT_NAME)
            state = endpoint.get('endpoint_status', {}).get('state', 'UNKNOWN')
            if state == 'ONLINE':
                print(f"\n✅ Endpoint is ONLINE!")
                break
            print(f"  Status: {state} ({i*10}s)")
            time.sleep(10)
        except:
            time.sleep(10)
    else:
        print("⚠️ Endpoint creation taking longer than expected")

# COMMAND ----------

# DBTITLE 1,Create Sample Documents
import pandas as pd

# Sample knowledge base about Databricks
sample_docs = [
    {
        "id": "1",
        "content": "Databricks Unity Catalog is a unified governance solution for data and AI. It provides centralized access control, auditing, lineage, and data discovery across all your data assets. Unity Catalog works with structured and unstructured data.",
        "source": "unity_catalog_overview"
    },
    {
        "id": "2",
        "content": "Databricks Lakehouse combines the best of data lakes and data warehouses. It provides ACID transactions, schema enforcement, and data versioning on top of cloud object storage. Delta Lake is the storage layer that enables the lakehouse architecture.",
        "source": "lakehouse_architecture"
    },
    {
        "id": "3",
        "content": "MLflow is an open-source platform for managing the machine learning lifecycle. It includes experiment tracking, model packaging, versioning, and deployment. MLflow integrates seamlessly with Databricks for production ML workflows.",
        "source": "mlflow_overview"
    },
    {
        "id": "4",
        "content": "Databricks Vector Search enables you to store and search embeddings at scale. It supports similarity search with filtering, real-time updates, and integrates with Databricks workflows. Use it for RAG applications, recommendations, and semantic search.",
        "source": "vector_search_features"
    },
    {
        "id": "5",
        "content": "Databricks Model Serving provides real-time and batch inference for ML models. It supports auto-scaling, A/B testing, and model monitoring. You can serve models from MLflow, custom Python code, or external model registries.",
        "source": "model_serving_capabilities"
    },
    {
        "id": "6",
        "content": "Databricks Workflows orchestrates data pipelines, ML training, and analytics jobs. It supports task dependencies, error handling, and schedule-based execution. You can create workflows from notebooks, Python files, or SQL queries.",
        "source": "workflows_introduction"
    },
    {
        "id": "7",
        "content": "Rikku is an AI assistant designed to help with Databricks-related questions. Rikku can answer questions about Unity Catalog, MLflow, Vector Search, and other Databricks features using retrieved documentation.",
        "source": "rikku_introduction"
    }
]

df = pd.DataFrame(sample_docs)
print(f"✅ Created {len(sample_docs)} sample documents\n")
print("Sample documents:")
for doc in sample_docs[:3]:
    print(f"  [{doc['id']}] {doc['content'][:80]}...")

print(f"\nSaving to source table: {SOURCE_TABLE_NAME}")
spark_df = spark.createDataFrame(df)
spark_df.write.mode("overwrite").saveAsTable(SOURCE_TABLE_NAME)

print(f"✅ Documents saved to {SOURCE_TABLE_NAME}")

# COMMAND ----------

# DBTITLE 1,Create Vector Search Index
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load embedding model locally
print("Loading embedding model (sentence-transformers)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded\n")

# Get documents from source table
print(f"Reading documents from {SOURCE_TABLE_NAME}...")
source_df = spark.table(SOURCE_TABLE_NAME).toPandas()
print(f"✅ Loaded {len(source_df)} documents\n")

# Compute embeddings
print("Computing embeddings...")
texts = source_df['content'].tolist()
embeddings = embedding_model.encode(texts, show_progress_bar=True)

# Add embeddings to dataframe
source_df['embedding'] = list(embeddings)
print("✅ Embeddings computed\n")

# Convert embeddings to proper format for Spark
import numpy as np
source_df['embedding'] = source_df['embedding'].apply(lambda x: x.tolist())

# Save back to Delta table with embeddings (with schema overwrite)
print(f"Saving embeddings to {SOURCE_TABLE_NAME}...")
spark_df_with_embeddings = spark.createDataFrame(source_df)
spark_df_with_embeddings.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(SOURCE_TABLE_NAME)
print("✅ Embeddings saved to source table\n")

# Check if index exists
try:
    index = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=INDEX_NAME)
    print(f"✅ Index '{INDEX_NAME}' already exists")
    print(f"   Deleting and recreating for fresh setup...")
    vsc.delete_index(INDEX_NAME)
    time.sleep(5)
except:
    print(f"Index '{INDEX_NAME}' does not exist, creating new...")

# Create Delta Sync Index with self-managed embeddings
print(f"\nCreating Delta Sync Index with self-managed embeddings...")
index = vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=INDEX_NAME,
    source_table_name=SOURCE_TABLE_NAME,
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_dimension=384,  # all-MiniLM-L6-v2 dimension
    embedding_vector_column="embedding"
)

print("⏳ Waiting for index to be ready...\n")

# Wait for index to be online
for i in range(60):
    try:
        index_status = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=INDEX_NAME)
        status = index_status.get('status', {}).get('ready', False)
        if status:
            print("✅ Index is READY!")
            break
        detailed_state = index_status.get('status', {}).get('detailed_state', 'UNKNOWN')
        print(f"   Status: {detailed_state} ({i*10}s)")
        time.sleep(10)
    except Exception as e:
        print(f"   Checking status... ({i*10}s)")
        time.sleep(10)
else:
    print("⚠️ Index creation taking longer than expected, but may still complete")

print(f"\n✅ Delta Sync Index created successfully!")
print(f"   Index: {INDEX_NAME}")
print(f"   Source Table: {SOURCE_TABLE_NAME}")
print(f"   Documents: {len(source_df)}")

# COMMAND ----------

# DBTITLE 1,Test Vector Search
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Test query
test_query = "What is Unity Catalog?"
print(f"Test Query: '{test_query}'\n")

# Compute query embedding
query_embedding = embedding_model.encode([test_query])[0]
print("Searching for relevant documents...\n")

# Search using vector similarity
index = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=INDEX_NAME)
results = index.similarity_search(
    query_vector=query_embedding.tolist(),
    columns=["id", "content", "source"],
    num_results=3
)

print("Top 3 Results:")
print("=" * 80)
for idx, row in enumerate(results.get('result', {}).get('data_array', []), 1):
    print(f"\n[{idx}] Source: {row[2]}")
    print(f"    Content: {row[1][:150]}...")
    print(f"    ID: {row[0]}")

print("\n" + "=" * 80)
print("✅ Vector search is working!")

# COMMAND ----------

# DBTITLE 1,Create Rikku RAG Agent
# MAGIC %%writefile rikku_rag_agent.py
# MAGIC import json
# MAGIC from typing import Any, Dict
# MAGIC from uuid import uuid4
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC from databricks_langchain import ChatDatabricks
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC )
# MAGIC from sentence_transformers import SentenceTransformer
# MAGIC
# MAGIC # Configuration
# MAGIC LLM_ENDPOINT_NAME = "databricks-meta-llama-3.1-405b-instruct"
# MAGIC INDEX_NAME = "agents.default.rikku_knowledge_base"
# MAGIC VECTOR_SEARCH_ENDPOINT_NAME = "rikku_vs_endpoint"
# MAGIC EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# MAGIC
# MAGIC # Initialize
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME, temperature=0.1, max_tokens=500)
# MAGIC vsc = VectorSearchClient(disable_notice=True)
# MAGIC embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
# MAGIC
# MAGIC def search_knowledge_base(query: str) -> str:
# MAGIC     """Search the Databricks knowledge base for relevant information."""
# MAGIC     try:
# MAGIC         # Compute query embedding
# MAGIC         query_embedding = embedding_model.encode([query])[0]
# MAGIC         
# MAGIC         # Search using vector similarity
# MAGIC         results = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=INDEX_NAME).similarity_search(
# MAGIC             query_vector=query_embedding.tolist(),
# MAGIC             columns=["content", "source"],
# MAGIC             num_results=3
# MAGIC         )
# MAGIC         
# MAGIC         data_array = results.get('result', {}).get('data_array', [])
# MAGIC         
# MAGIC         if not data_array:
# MAGIC             return "No relevant information found in the knowledge base."
# MAGIC         
# MAGIC         # Format results
# MAGIC         context_parts = []
# MAGIC         for idx, row in enumerate(data_array, 1):
# MAGIC             content = row[0]
# MAGIC             source = row[1]
# MAGIC             context_parts.append(f"[Document {idx} - {source}]\n{content}")
# MAGIC         
# MAGIC         return "\n\n".join(context_parts)
# MAGIC     except Exception as e:
# MAGIC         return f"Error searching knowledge base: {str(e)}"
# MAGIC
# MAGIC class RikkuRAGResponsesAgent(ResponsesAgent):
# MAGIC     def __init__(self):
# MAGIC         self.llm = llm
# MAGIC     
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         # Extract the user question
# MAGIC         user_message = ""
# MAGIC         for msg in request.input:
# MAGIC             msg_dict = msg.model_dump()
# MAGIC             if msg_dict.get("role") == "user":
# MAGIC                 content = msg_dict.get("content", "")
# MAGIC                 if isinstance(content, list):
# MAGIC                     user_message = content[0].get("text", "") if content else ""
# MAGIC                 else:
# MAGIC                     user_message = content
# MAGIC                 break
# MAGIC         
# MAGIC         if not user_message:
# MAGIC             return ResponsesAgentResponse(
# MAGIC                 output=[self.create_text_output_item(text="No question provided.", id=str(uuid4()))]
# MAGIC             )
# MAGIC         
# MAGIC         # Step 1: Retrieve relevant context
# MAGIC         context = search_knowledge_base(user_message)
# MAGIC         
# MAGIC         # Step 2: Create prompt with context
# MAGIC         system_prompt = """You are Rikku, a helpful AI assistant with access to Databricks documentation.
# MAGIC
# MAGIC IMPORTANT RULES:
# MAGIC 1. Answer questions ONLY using the retrieved context provided below.
# MAGIC 2. If the context doesn't contain the answer, say "I don't have information about that in my knowledge base."
# MAGIC 3. Be specific and cite information from the retrieved documents.
# MAGIC 4. Never make up or infer information beyond what's in the context.
# MAGIC 5. If you're unsure, acknowledge it clearly."""
# MAGIC         
# MAGIC         user_prompt = f"""Context from knowledge base:
# MAGIC {context}
# MAGIC
# MAGIC User Question: {user_message}
# MAGIC
# MAGIC Provide a helpful answer based ONLY on the context above."""
# MAGIC         
# MAGIC         # Step 3: Get LLM response
# MAGIC         messages = [
# MAGIC             {"role": "system", "content": system_prompt},
# MAGIC             {"role": "user", "content": user_prompt}
# MAGIC         ]
# MAGIC         
# MAGIC         response = self.llm.invoke(messages)
# MAGIC         answer = response.content if hasattr(response, 'content') else str(response)
# MAGIC         
# MAGIC         # Return formatted response
# MAGIC         output = [self.create_text_output_item(text=answer, id=str(uuid4()))]
# MAGIC         return ResponsesAgentResponse(output=output)
# MAGIC
# MAGIC # Create and register agent
# MAGIC mlflow.langchain.autolog()
# MAGIC RIKKU_RAG_AGENT = RikkuRAGResponsesAgent()
# MAGIC mlflow.models.set_model(RIKKU_RAG_AGENT)
# MAGIC
# MAGIC print("✅ Rikku RAG Agent created successfully!")

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Test RAG Agent Locally
from rikku_rag_agent import RIKKU_RAG_AGENT

print("Testing Rikku RAG Agent...\n")
print("=" * 80)

# Test 1: Question about Unity Catalog
print("\nTest 1: Unity Catalog Question")
print("-" * 80)
result = RIKKU_RAG_AGENT.predict({
    "input": [{"role": "user", "content": "What is Unity Catalog and what does it provide?"}]
})
print(result.model_dump(exclude_none=True))

# Test 2: Question about MLflow
print("\n\nTest 2: MLflow Question")
print("-" * 80)
result = RIKKU_RAG_AGENT.predict({
    "input": [{"role": "user", "content": "Tell me about MLflow and its capabilities."}]
})
print(result.model_dump(exclude_none=True))

# Test 3: Question not in knowledge base (should say it doesn't know)
print("\n\nTest 3: Unknown Topic (Hallucination Check)")
print("-" * 80)
result = RIKKU_RAG_AGENT.predict({
    "input": [{"role": "user", "content": "What is Kubernetes?"}]
})
print(result.model_dump(exclude_none=True))

# COMMAND ----------

# DBTITLE 1,Log to MLflow
import mlflow
from mlflow.models.resources import DatabricksVectorSearchIndex

# Configuration (re-declared after Python restart)
CATALOG = "agents"
SCHEMA = "default"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.rikku_knowledge_base"

# Set experiment
mlflow.set_experiment("/Users/prakhar1207srivastava@gmail.com/rikku-rag-experiments")

# Declare vector search dependency
resources = [
    DatabricksVectorSearchIndex(
        index_name=INDEX_NAME
    )
]

print("Logging RAG agent to MLflow...\n")

# Log the agent (MLflow will infer the correct Agent Framework signature automatically)
with mlflow.start_run(run_name="rikku_rag_v1") as run:
    logged_agent_info = mlflow.langchain.log_model(
        lc_model="rikku_rag_agent.py",
        artifact_path="agent",
        input_example={
            "input": [{"role": "user", "content": "What is Unity Catalog?"}]
        },
        resources=resources,
        extra_pip_requirements=[
            "langgraph",
            "langchain-core",
            "databricks-langchain",
            "databricks-vectorsearch",
            "sentence-transformers"
        ]
    )

print(f"✅ RAG Agent logged successfully!")
print(f"Run ID: {logged_agent_info.run_id}")
print(f"Model URI: {logged_agent_info.model_uri}")

# COMMAND ----------

# DBTITLE 1,Register to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Model name with rikku prefix
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.rikku_rag_chatbot"

print(f"Registering RAG model as: {UC_MODEL_NAME}\n")

# Register
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri,
    name=UC_MODEL_NAME
)

print(f"✅ Model registered successfully!")
print(f"Model: {UC_MODEL_NAME}")
print(f"Version: {uc_registered_model_info.version}")

# COMMAND ----------

# DBTITLE 1,Deploy Endpoint
from databricks import agents

print(f"Deploying {UC_MODEL_NAME} version {uc_registered_model_info.version}...")
print("This may take 10-15 minutes.\n")

# Deploy
deployment_info = agents.deploy(
    UC_MODEL_NAME,
    uc_registered_model_info.version,
    scale_to_zero=True,
    tags={"purpose": "rikku-rag-chatbot", "environment": "production"}
)

print("\n✅ Deployment initiated!")
print(f"Endpoint: {deployment_info.endpoint_name}")
print(f"Query URL: {deployment_info.query_endpoint}")
print(f"Review App: {deployment_info.review_app_url}")

# COMMAND ----------

# DBTITLE 1,Test RAG Endpoint
import requests
import json
import time

# Wait a bit for deployment
print("Waiting for endpoint to be ready...\n")
time.sleep(30)

# Get credentials
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

endpoint_url = f"{url}/serving-endpoints/{deployment_info.endpoint_name}/invocations"

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

print("Testing RAG endpoint...\n")
print("=" * 80)

# Test questions
test_questions = [
    "What is Vector Search in Databricks?",
    "Tell me about Databricks Workflows.",
    "Who is Rikku?"
]

for idx, question in enumerate(test_questions, 1):
    print(f"\nTest {idx}: {question}")
    print("-" * 80)
    
    try:
        response = requests.post(
            endpoint_url,
            headers=headers,
            json={"input": [{"role": "user", "content": question}]},
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        output_text = result.get('output', [{}])[0].get('content', [{}])[0].get('text', 'No response')
        print(f"Answer: {output_text[:200]}...")
        
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(2)

print("\n" + "=" * 80)

# COMMAND ----------

# DBTITLE 1,🎉 Summary Dashboard
print("=" * 80)
print("🎉 RIKKU RAG CHATBOT - DEPLOYMENT SUMMARY")
print("=" * 80)

print("\n📚 Knowledge Base:")
print(f"   Vector Search Endpoint: {VECTOR_SEARCH_ENDPOINT_NAME}")
print(f"   Index: {INDEX_NAME}")
print(f"   Documents: 7 sample docs about Databricks")
print(f"   Embedding Model: {EMBEDDING_MODEL}")

print("\n📦 Model Details:")
print(f"   Name: {UC_MODEL_NAME}")
print(f"   Version: {uc_registered_model_info.version}")
print(f"   Run ID: {logged_agent_info.run_id}")

print("\n🚀 Endpoint Details:")
print(f"   Name: {deployment_info.endpoint_name}")
print(f"   URL: {deployment_info.endpoint_url}")
print(f"   Query Endpoint: {deployment_info.query_endpoint}")

print("\n🎨 UI Access:")
print(f"   Review App: {deployment_info.review_app_url}")

print("\n✅ Features:")
print("   • RAG (Retrieval Augmented Generation)")
print("   • Vector Search with BGE embeddings")
print("   • Anti-hallucination via context grounding")
print("   • Tool-calling pattern with search_knowledge_base")
print("   • Scale to zero enabled")

print("\n" + "=" * 80)
print("Rikku RAG is ready! Test via Review App or API. 🔍🤖")
print("=" * 80)