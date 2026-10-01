# Databricks notebook source
# Install (run once)
%pip install databricks-vectorsearch databricks-sdk mlflow sentence-transformers
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Cell 2
docs = [
    ("1", "Unity Catalog is a governance solution."),
    ("2", "MLflow manages ML lifecycle."),
    ("3", "Vector Search enables semantic retrieval.")
]

df = spark.createDataFrame(docs, ["id", "content"])
df.write.mode("overwrite").option("delta.enableChangeDataFeed", "true").saveAsTable("agents.default.rag_docs")

# Enable CDF on existing table (if it wasn't already enabled)
spark.sql("ALTER TABLE agents.default.rag_docs SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

pdf = spark.table("agents.default.rag_docs").toPandas()
pdf["embedding"] = pdf["content"].apply(lambda x: model.encode(x).tolist())

spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true").option("delta.enableChangeDataFeed", "true").saveAsTable("agents.default.rag_docs")

from databricks.vector_search.client import VectorSearchClient
from databricks.sdk.errors import ResourceConflict, BadRequest
import time

vsc = VectorSearchClient()

# Get existing endpoint or create new one
try:
    endpoints = vsc.list_endpoints().get('endpoints', [])
    if endpoints:
        endpoint_name = endpoints[0]['name']
        print(f"Using existing endpoint: {endpoint_name}")
    else:
        # No endpoints exist, create one
        endpoint_name = "rag_vs_endpoint"
        vsc.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
        print(f"Created endpoint: {endpoint_name}")
except (ResourceConflict, BadRequest) as e:
    # If creation fails, list and use first endpoint
    endpoints = vsc.list_endpoints().get('endpoints', [])
    endpoint_name = endpoints[0]['name']
    print(f"Using existing endpoint: {endpoint_name}")

# Check if index already exists
index_name = "agents.default.rag_index"
try:
    existing_index = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)
    print(f"Using existing index: {index_name}")
except Exception:
    # Index doesn't exist, create it
    vsc.create_delta_sync_index(
        endpoint_name=endpoint_name,
        index_name=index_name,
        source_table_name="agents.default.rag_docs",
        pipeline_type="TRIGGERED",
        primary_key="id",
        embedding_dimension=384,
        embedding_vector_column="embedding"
    )
    print(f"Created new index: {index_name}")

# Wait for index to be online/ready before syncing
print("Waiting for index to be online...")
index = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)

for i in range(60):
    status = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name).describe()
    index_status = status.get("status", {})
    
    if index_status.get("ready", False):
        print("Index is ready!")
        break
    elif i == 0:
        # On first iteration, try to sync if index is ONLINE but not ready
        if index_status.get("message") == "ONLINE":
            try:
                print("Syncing index...")
                index.sync()
            except Exception as e:
                print(f"Sync not yet available: {e}")
    
    time.sleep(2)

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

w = WorkspaceClient()

def simple_rag(question):
    # Compute question embedding
    question_embedding = model.encode(question).tolist()
    
    index = vsc.get_index(
        endpoint_name=endpoint_name,
        index_name=index_name
    )

    results = index.similarity_search(
        query_vector=question_embedding,
        columns=["content"],
        num_results=2
    )

    # data_array is a list of lists: [[content1], [content2], ...]
    context = "\n".join([r[0] for r in results["result"]["data_array"]])

    prompt = f"""
Answer only from context.

Context:
{context}

Question: {question}
"""

    response = w.serving_endpoints.query(
        name="databricks-meta-llama-3.1-405b-instruct",
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)]
    )

    return response.choices[0].message.content


print(simple_rag("What is Unity Catalog?"))

# Save model code to a Python file
model_code = '''
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from sentence_transformers import SentenceTransformer
import mlflow

class RAGModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """Load dependencies when model is loaded."""
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vsc = VectorSearchClient()
        self.w = WorkspaceClient()
        self.endpoint_name = "rag_endpoint"
        self.index_name = "agents.default.rag_index"
    
    def predict(self, context, model_input):
        """Run RAG for each question."""
        questions = model_input["question"] if isinstance(model_input, dict) else model_input
        results = []
        
        for question in questions:
            # Compute question embedding
            question_embedding = self.model.encode(question).tolist()
            
            # Get index
            index = self.vsc.get_index(
                endpoint_name=self.endpoint_name,
                index_name=self.index_name
            )
            
            # Search
            search_results = index.similarity_search(
                query_vector=question_embedding,
                columns=["content"],
                num_results=2
            )
            
            # Extract context
            context_text = "\\n".join([r[0] for r in search_results["result"]["data_array"]])
            
            # Query LLM
            prompt = f"""
Answer only from context.

Context:
{context_text}

Question: {question}
"""
            
            response = self.w.serving_endpoints.query(
                name="databricks-meta-llama-3.1-405b-instruct",
                messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)]
            )
            
            results.append(response.choices[0].message.content)
        
        return results

# Register the model instance
mlflow.models.set_model(RAGModel())
'''

# Write model code to file
with open("/tmp/rag_model.py", "w") as f:
    f.write(model_code)

import mlflow
import mlflow.pyfunc
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)

mlflow.set_experiment("/Users/prakhar1207srivastava@gmail.com/simple-rag")

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="rag_model",
        python_model="/tmp/rag_model.py",
        code_paths=["/tmp/rag_model.py"],
        input_example={"question": ["What is MLflow?"]},
        pip_requirements=[
            "databricks-vectorsearch",
            "databricks-sdk",
            "sentence-transformers"
        ],
        resources=[
            DatabricksVectorSearchIndex(index_name="agents.default.rag_index"),
            DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3.1-405b-instruct")
        ]
    )
    run_id = run.info.run_id
    print(f"Model logged with run_id: {run_id}")

mlflow.set_registry_uri("databricks-uc")

model_uri = f"runs:/{run_id}/rag_model"
registered_model = mlflow.register_model(model_uri, "agents.default.rag_chatbot")

print(f"Model registered: {registered_model.name} version {registered_model.version}")

# COMMAND ----------

# DBTITLE 1,Deploy Model Serving Endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

endpoint_name = "rag-chatbot-endpoint"

# Check if endpoint already exists
try:
    existing_endpoint = w.serving_endpoints.get(endpoint_name)
    print(f"Endpoint '{endpoint_name}' already exists. Updating to version 2...")
    
    # Update the endpoint with new model version
    w.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=[
            ServedEntityInput(
                entity_name="agents.default.rag_chatbot",
                entity_version="2",
                scale_to_zero_enabled=True,
                workload_size="Small"
            )
        ]
    )
    print(f"Updated endpoint '{endpoint_name}' to use version 2")
    
except Exception as e:
    # Endpoint doesn't exist, create it
    print(f"Creating new endpoint '{endpoint_name}'...")
    
    w.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name="agents.default.rag_chatbot",
                    entity_version="2",
                    scale_to_zero_enabled=True,
                    workload_size="Small"
                )
            ]
        )
    )
    print(f"Created endpoint '{endpoint_name}'")

# Wait for endpoint to be ready
import time

print("Waiting for endpoint to be ready...")
for i in range(90):
    endpoint = w.serving_endpoints.get(endpoint_name)
    state = endpoint.state.config_update if endpoint.state else None
    
    if state == "NOT_UPDATING" and endpoint.state.ready == "READY":
        print(f"\n✅ Endpoint is ready!")
        print(f"\nEndpoint URL: {endpoint.config.served_entities[0].entity_name}")
        print(f"Endpoint name: {endpoint_name}")
        break
    else:
        print(f"Status: {state}, Ready: {endpoint.state.ready if endpoint.state else 'Unknown'}")
        time.sleep(10)

print(f"\n🔗 View endpoint: https://dbc-948eca32-5f10.cloud.databricks.com/ml/endpoints/{endpoint_name}")

# COMMAND ----------

# DBTITLE 1,Test Serving Endpoint
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

endpoint_name = "rag-chatbot-endpoint"

# Check endpoint status
try:
    endpoint = w.serving_endpoints.get(endpoint_name)
    print(f"Endpoint Status: {endpoint.state.ready}")
    
    if endpoint.state.ready == "READY":
        # Test the endpoint
        print("\nTesting the endpoint...\n")
        
        test_question = "What is MLflow?"
        
        response = w.serving_endpoints.query(
            name=endpoint_name,
            inputs={"question": [test_question]}
        )
        
        print(f"Question: {test_question}")
        print(f"Answer: {response.predictions[0]}")
        
        # Try another question
        test_question_2 = "What does Vector Search do?"
        
        response_2 = w.serving_endpoints.query(
            name=endpoint_name,
            inputs={"question": [test_question_2]}
        )
        
        print(f"\nQuestion: {test_question_2}")
        print(f"Answer: {response_2.predictions[0]}")
    else:
        print(f"Endpoint is still deploying. Current state: {endpoint.state.config_update}")
        print("Please wait a few more minutes and re-run this cell.")
        
except Exception as e:
    print(f"Error: {e}")
    print("\nThe endpoint might still be deploying. Wait a few minutes and try again.")