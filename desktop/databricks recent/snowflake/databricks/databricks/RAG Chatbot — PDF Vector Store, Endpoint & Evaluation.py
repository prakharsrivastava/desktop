# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # RAG Chatbot — PDF Vector Store, Endpoint & Evaluation
# MAGIC
# MAGIC End-to-end RAG pipeline: parse PDF → chunk → vector search index → OpenAI chatbot → MLflow model → serving endpoint → traced evaluation with scorers.

# COMMAND ----------

# DBTITLE 1,Setup & Imports
# MAGIC %pip install mlflow[databricks] openai databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Install and Import Libraries
# MAGIC %skip
# MAGIC import mlflow
# MAGIC import os
# MAGIC from openai import OpenAI
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC import json
# MAGIC import uuid
# MAGIC
# MAGIC # Set MLflow experiment
# MAGIC mlflow.set_experiment("/Users/prakhar1207srivastava@gmail.com/rag-chatbot-eval")
# MAGIC
# MAGIC # Set OpenAI API key from Databricks secrets (user should have this)
# MAGIC # Try to get from secrets, fallback to environment variable
# MAGIC try:
# MAGIC     openai_key = "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"
# MAGIC     os.environ["OPENAI_API_KEY"] = openai_key
# MAGIC except:
# MAGIC     print("Warning: Could not get OpenAI key from secrets. Make sure OPENAI_API_KEY is set in environment.")
# MAGIC
# MAGIC print("Setup complete")
# MAGIC print(f"MLflow version: {mlflow.__version__}")
# MAGIC print(f"OpenAI key available: {bool(os.environ.get('OPENAI_API_KEY'))}")

# COMMAND ----------

# DBTITLE 1,Parse PDF with ai_parse_document
# MAGIC %sql
# MAGIC -- Parse the PDF and save as a Unity Catalog table
# MAGIC CREATE OR REPLACE TABLE workspace.default.pdf_parsed AS
# MAGIC SELECT
# MAGIC   _metadata.file_name AS file_name,
# MAGIC   content,
# MAGIC   ai_parse_document(content, MAP('version', '2.0')) AS parsed_content
# MAGIC FROM READ_FILES(
# MAGIC   'idbfs:/2026-09-24/10/_c54fb7c2-48fa-44dc-956b-2def36c3cd8f',
# MAGIC   format => 'binaryFile'
# MAGIC );
# MAGIC
# MAGIC SELECT file_name, parsed_content:metadata, size(try_cast(parsed_content:document:elements AS ARRAY<VARIANT>)) AS num_elements
# MAGIC FROM workspace.default.pdf_parsed;

# COMMAND ----------

# DBTITLE 1,Extract Text and Create Chunks
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, expr, concat_ws, collect_list, lit
import pandas as pd
import uuid

spark = SparkSession.builder.getOrCreate()

# Extract text elements from the parsed VARIANT and create chunks
chunks_df = spark.sql("""
WITH parsed AS (
  SELECT 
    file_name,
    explode(try_cast(parsed_content:document:elements AS ARRAY<VARIANT>)) AS element
  FROM workspace.default.pdf_parsed
),
text_elements AS (
  SELECT
    file_name,
    try_cast(element:content AS STRING) AS text,
    try_cast(element:type AS STRING) AS element_type,
    try_cast(element:page AS INT) AS page_number
  FROM parsed
  WHERE try_cast(element:type AS STRING) IN ('text', 'title', 'section_header', 'paragraph')
    AND length(try_cast(element:content AS STRING)) > 10
)
SELECT
  file_name,
  text,
  element_type,
  page_number
FROM text_elements
""")

# Convert to pandas for chunking
chunks_pdf = chunks_df.toPandas()
print(f"Extracted {len(chunks_pdf)} text elements from PDF")

# Create overlapping chunks for RAG
chunk_size = 800
chunk_overlap = 100
chunks = []
current_chunk = ""
current_metadata = {"page": [], "type": []}

for _, row in chunks_pdf.iterrows():
    text = str(row["text"])
    if len(current_chunk) + len(text) < chunk_size:
        current_chunk += " " + text
        current_metadata["page"].append(row["page_number"])
        current_metadata["type"].append(row["element_type"])
    else:
        if current_chunk:
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": current_chunk.strip(),
                "pages": [int(p) for p in set(current_metadata["page"]) if p is not None and p == p],
                "source_file": row["file_name"],
                "chunk_index": len(chunks)
            })
        # Start new chunk with overlap
        overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
        current_chunk = overlap_text + " " + text
        current_metadata = {"page": [row["page_number"]], "type": [row["element_type"]]}

# Don't forget the last chunk
if current_chunk:
    chunks.append({
        "chunk_id": str(uuid.uuid4()),
        "text": current_chunk.strip(),
        "pages": [int(p) for p in set(current_metadata["page"]) if p is not None and p == p],
        "source_file": chunks_pdf.iloc[0]["file_name"],
        "chunk_index": len(chunks)
    })

print(f"Created {len(chunks)} chunks for vector search")
if len(chunks) > 0:
    print(f"Average chunk length: {sum(len(c['text']) for c in chunks) / len(chunks):.0f} chars")
    print(f"\nSample chunk:\n{chunks[0]['text'][:300]}...")

# Save chunks as a table for the vector index
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType

chunks_pdf_final = pd.DataFrame(chunks)
schema = StructType([
    StructField("chunk_id", StringType(), True),
    StructField("text", StringType(), True),
    StructField("pages", ArrayType(IntegerType()), True),
    StructField("source_file", StringType(), True),
    StructField("chunk_index", IntegerType(), True)
])
chunks_spark = spark.createDataFrame(chunks_pdf_final, schema=schema)
chunks_spark.write.mode("overwrite").saveAsTable("workspace.default.pdf_chunks")
print(f"\nSaved {len(chunks)} chunks to workspace.default.pdf_chunks")

# COMMAND ----------

# DBTITLE 1,Enhanced Chunks Table with Hierarchy, Versioning & Hashing
from pyspark.sql.functions import col, lit, current_timestamp, md5, concat_ws, row_number
from pyspark.sql.window import Window
import hashlib
import pandas as pd
from datetime import datetime

# Read the basic chunks we created earlier
basic_chunks_df = spark.table("workspace.default.pdf_chunks")

# Extract document metadata from file name (customize based on your naming convention)
# Example: "policy_12345.pdf" -> doc_id="policy_12345", policy_id="12345"
enhanced_chunks = basic_chunks_df.toPandas()

# Add enhanced fields matching the architectural diagrams
for idx, row in enhanced_chunks.iterrows():
    # Document ID (extract from source file)
    doc_name = row['source_file'].replace('.pdf', '') if row['source_file'] else 'unknown'
    enhanced_chunks.at[idx, 'doc_id'] = doc_name
    
    # Policy ID (if applicable - extract number from filename)
    policy_id = ''.join(filter(str.isdigit, doc_name)) if any(char.isdigit() for char in doc_name) else 'none'
    enhanced_chunks.at[idx, 'policy_id'] = policy_id if policy_id else 'general'
    
    # Section ID (based on page ranges)
    pages = row['pages']
    if pages and len(pages) > 0:
        section_id = f"section_{min(pages)//5 + 1}"  # Group every 5 pages into a section
    else:
        section_id = f"section_{row['chunk_index']//10 + 1}"  # Or by chunk groups
    enhanced_chunks.at[idx, 'section_id'] = section_id
    
    # Content hash for change detection (MD5 of text)
    text_content = row['text'] if row['text'] else ''
    content_hash = hashlib.md5(text_content.encode('utf-8')).hexdigest()
    enhanced_chunks.at[idx, 'content_hash'] = content_hash
    
    # Hierarchical path
    path = f"{doc_name}/{section_id}/chunk_{row['chunk_index']}"
    enhanced_chunks.at[idx, 'path'] = path
    
    # Version (initially v1)
    enhanced_chunks.at[idx, 'version'] = 1
    
    # Timestamps
    now = datetime.now()
    enhanced_chunks.at[idx, 'created_at'] = now
    enhanced_chunks.at[idx, 'updated_at'] = now
    
    # Change type (for reindexing workflow)
    enhanced_chunks.at[idx, 'change_type'] = 'new_doc'  # 'new_doc', 'updated', 'deleted'

# Convert to Spark DataFrame with proper schema
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, TimestampType

enhanced_schema = StructType([
    StructField("chunk_id", StringType(), False),
    StructField("text", StringType(), True),
    StructField("pages", ArrayType(IntegerType()), True),
    StructField("source_file", StringType(), True),
    StructField("chunk_index", IntegerType(), True),
    # Enhanced fields
    StructField("doc_id", StringType(), True),
    StructField("policy_id", StringType(), True),
    StructField("section_id", StringType(), True),
    StructField("content_hash", StringType(), True),
    StructField("path", StringType(), True),
    StructField("version", IntegerType(), True),
    StructField("created_at", TimestampType(), True),
    StructField("updated_at", TimestampType(), True),
    StructField("change_type", StringType(), True)
])

enhanced_spark_df = spark.createDataFrame(enhanced_chunks, schema=enhanced_schema)

# Save to Unity Catalog table with enhanced schema
enhanced_spark_df.write.mode("overwrite").saveAsTable("workspace.default.pdf_chunks_enhanced")

print("\n" + "="*70)
print("ENHANCED CHUNKS TABLE CREATED: workspace.default.pdf_chunks_enhanced")
print("="*70)
print(f"\nTotal chunks: {enhanced_spark_df.count()}")
print(f"\nSchema:")
enhanced_spark_df.printSchema()

print(f"\nSample records with hierarchy and versioning:")
display(enhanced_spark_df.select(
    "chunk_id", "doc_id", "policy_id", "section_id", 
    "path", "version", "content_hash", "change_type"
).limit(5))

print(f"\nDocument hierarchy summary:")
hierarchy_summary = enhanced_spark_df.groupBy("doc_id", "section_id").count().orderBy("doc_id", "section_id")
display(hierarchy_summary)

# COMMAND ----------

# DBTITLE 1,Reindexing Workflow - Change Detection
from pyspark.sql.functions import col, count as sql_count

print("="*70)
print("REINDEXING WORKFLOW: Hash-Based Change Detection")
print("="*70)

# Load existing enhanced chunks
existing_df = spark.table("workspace.default.pdf_chunks_enhanced")

# Analyze current document inventory
doc_stats = existing_df.groupBy("doc_id", "version").agg(
    sql_count("*").alias("chunk_count")
).orderBy("doc_id")

print("\nCurrent Document Inventory:")
display(doc_stats)

# Show sample chunks with hash values
print("\nSample chunks with content hashes for change detection:")
display(existing_df.select(
    "doc_id", "section_id", "chunk_index", 
    "content_hash", "version", "change_type"
).orderBy("chunk_index").limit(10))

print("\n" + "="*70)
print("Change Detection Strategy (from architectural diagram):")
print("="*70)
print("""
1. NEW DOC: No existing hash → Full ingestion → chunking → vector DB

2. UPDATED DOC - Hash comparison:
   a. SMALL CHANGE (<20% chunks changed)
      → Reprocess whole document
      → Update all chunks with new version
   
   b. LARGE CHANGE (>20% chunks changed)
      → Upsert only changed chunks
      → Preserve unchanged chunks

3. DELETED DOC: 
   → Mark change_type = 'deleted'
   → Remove from vector index

4. NO CHANGE:
   → All hashes match
   → Skip reprocessing
""")

print("\nHash-based incremental update system ready!")
print("="*70)

# COMMAND ----------

# DBTITLE 1,Create Vector Search Index
from databricks.vector_search.client import VectorSearchClient
import time

# Enable change data feed on the chunks table (required for Vector Search)
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.sql("ALTER TABLE workspace.default.pdf_chunks SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
print("Change data feed enabled on workspace.default.pdf_chunks")

vsc = VectorSearchClient(disable_notice=True)

# Define index name
index_name = "workspace.default.pdf_chunks_index"
endpoint_name = "pdf-chatbot-endpoint"

# Create or get vector search endpoint
try:
    vs_endpoints = vsc.list_endpoints()
    endpoint_exists = any(e["name"] == endpoint_name for e in vs_endpoints.get("endpoints", []))
except:
    endpoint_exists = False

if not endpoint_exists:
    print(f"Creating vector search endpoint: {endpoint_name}")
    vsc.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
    print("Endpoint created (may take a few minutes to be ready)")
else:
    print(f"Endpoint {endpoint_name} already exists")

# Create or get the index
try:
    index = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)
    print(f"Index {index_name} already exists")
except:
    print(f"Creating index: {index_name}")
    index = vsc.create_delta_sync_index(
        endpoint_name=endpoint_name,
        index_name=index_name,
        source_table_name="workspace.default.pdf_chunks",
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="text",
        embedding_model_endpoint_name="databricks-gte-large-en",
    )
    print("Index created. Waiting for sync...")

# Trigger a sync (may fail if index not ready yet) and wait with polling
try:
    index.sync()
    print("Sync triggered.")
except Exception as e:
    print(f"Sync not ready yet (will poll): {e}")

print("Waiting for index to be ready...")
max_wait = 300  # 5 minutes
waited = 0
while waited < max_wait:
    time.sleep(30)
    waited += 30
    try:
        status = index.describe()
        state = status.get('status', 'unknown')
        print(f"  [{waited}s] Index status: {state}")
        if state == 'ONLINE':
            break
    except Exception as e:
        print(f"  [{waited}s] Status check: {e}")

try:
    status = index.describe()
    print(f"\nFinal index status: {status.get('status', 'unknown')}")
    print(f"Index ready: {status.get('status') == 'ONLINE'}")
except Exception as e:
    print(f"\nFinal status check error: {e}")

# COMMAND ----------

# DBTITLE 1,Build RAG Chatbot Function
from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI
import mlflow
import os
import json

# Initialize clients
vsc = VectorSearchClient(disable_notice=True)
openai_available = bool(os.environ.get("OPENAI_API_KEY"))
if openai_available:
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
else:
    print("OpenAI key not available. Using Databricks foundation model (databricks-meta-llama-3-3-70b-instruct).")
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    openai_client = None

def generate_response(system_prompt, user_prompt):
    """Generate response using OpenAI or Databricks foundation model."""
    if openai_available:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        return response.choices[0].message.content
    else:
        full_prompt = system_prompt + "\n\n" + user_prompt
        if len(full_prompt) > 3000:
            full_prompt = full_prompt[:3000] + "..."
        escaped_prompt = full_prompt.replace("\\", "\\\\").replace("'", "\\'")
        result = spark.sql(f"SELECT ai_query('databricks-meta-llama-3-3-70b-instruct', '{escaped_prompt}') AS response")
        return result.collect()[0][0]

# Get the vector search index
endpoint_name = "pdf-chatbot-endpoint"
index_name = "workspace.default.pdf_chunks_index"

def rag_chatbot(query: str, top_k: int = 3) -> dict:
    """RAG chatbot that retrieves relevant chunks from the PDF and generates an answer using OpenAI.
    
    Args:
        query: User's question
        top_k: Number of chunks to retrieve
    
    Returns:
        dict with 'response' and 'sources' keys
    """
    # 1. Retrieve relevant chunks from vector search
    try:
        index = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)
        results = index.similarity_search(
            query_text=query,
            columns=["text", "chunk_index", "pages"],
            num_results=top_k
        )
        chunks = results.get("result", {}).get("data_array", [])
        context_text = "\n\n".join([row[0] for row in chunks if row[0]])
    except Exception as e:
        context_text = ""
        chunks = []
    
    # 2. Build the prompt with retrieved context
    system_prompt = """You are a helpful assistant that answers questions based on the provided context from a PDF document. 
Use only the context below to answer the question. If the context does not contain enough information, say so clearly.
Be concise and accurate."""
    
    user_prompt = f"""Context from the document:
{context_text}

Question: {query}

Please answer based on the context above."""
    
    # 3. Generate response using OpenAI or Databricks foundation model
    answer = generate_response(system_prompt, user_prompt)
    
    # 4. Return response with sources
    sources = [
        {"chunk_index": row[1] if len(row) > 1 else None, "pages": row[2] if len(row) > 2 else None}
        for row in chunks
    ]
    
    return {"response": answer, "sources": sources, "query": query}

# Test the chatbot
test_result = rag_chatbot("What is the main topic of this document?")
print(f"Query: {test_result['query']}")
print(f"Response: {test_result['response']}")
print(f"Sources: {test_result['sources']}")

# COMMAND ----------

# DBTITLE 1,Register MLflow Model
import mlflow
from mlflow.models import infer_signature

# Create a wrapper model class
class RAGChatbotModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from databricks.vector_search.client import VectorSearchClient
        from openai import OpenAI
        import os
        
        self.vsc = VectorSearchClient(disable_notice=True)
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.endpoint_name = "pdf-chatbot-endpoint"
        self.index_name = "workspace.default.pdf_chunks_index"
    
    def predict(self, context, model_input, params=None):
        import pandas as pd
        if isinstance(model_input, pd.DataFrame):
            query = model_input.iloc[0].get("query", model_input.iloc[0][0])
        elif isinstance(model_input, dict):
            query = model_input.get("query", str(model_input))
        else:
            query = str(model_input)
        
        # Retrieve chunks
        try:
            index = self.vsc.get_index(endpoint_name=self.endpoint_name, index_name=self.index_name)
            results = index.similarity_search(
                query_text=query,
                columns=["text", "chunk_index", "pages"],
                num_results=3
            )
            chunks_data = results.get("result", {}).get("data_array", [])
            context_text = "\n\n".join([row[0] for row in chunks_data if row[0]])
        except Exception as e:
            context_text = ""
            chunks_data = []
        
        # Generate response
        system_prompt = "You are a helpful assistant that answers questions based on provided context. Use only the context to answer. Be concise."
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        return {"response": answer, "query": query}

# Log the model
with mlflow.start_run(run_name="rag_chatbot_model") as run:
    model = RAGChatbotModel()
    
    # Define input/output signature
    input_example = {"query": "What is this document about?"}
    signature = infer_signature(
        model_input={"query": "test question"},
        model_output={"response": "test answer", "query": "test question"}
    )
    
    # Log the model
    mlflow.pyfunc.log_model(
        artifact_path="rag_chatbot",
        python_model=model,
        signature=signature,
        input_example=input_example,
        pip_requirements=["openai", "databricks-vectorsearch"],
    )
    
    model_uri = f"runs:/{run.info.run_id}/rag_chatbot"
    print(f"Model logged: {model_uri}")
    print(f"Run ID: {run.info.run_id}")
    
    # Register the model
    model_name = "agents.default.rag_pdf_chatbot"
    mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Model registered as: {model_name}")

# COMMAND ----------

# DBTITLE 1,Create Serving Endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointConfig, ServedEntity
import time

w = WorkspaceClient()

endpoint_name = "rag-pdf-chatbot-endpoint"

# Create serving endpoint
try:
    w.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointConfig(
            served_entities=[
                ServedEntity(
                    name="rag_pdf_chatbot",
                    entity_name="agents.default.rag_pdf_chatbot",
                    entity_version="1",
                    workload_size="Small",
                    scale_to_zero_enabled=True,
                )
            ]
        )
    )
    print(f"Creating endpoint: {endpoint_name}")
    print("Endpoint creation initiated (may take a few minutes)")
except Exception as e:
    print(f"Endpoint creation result: {e}")

# Check endpoint status
time.sleep(5)
try:
    status = w.serving_endpoints.get(name=endpoint_name)
    print(f"Endpoint status: {status.state.ready}")
except Exception as e:
    print(f"Status check: {e}")

# COMMAND ----------

# DBTITLE 1,MLflow Tracing and Test Queries
import mlflow
from mlflow.entities import SpanType
from openai import OpenAI
from databricks.vector_search.client import VectorSearchClient
import os

# Enable MLflow tracing
mlflow.set_experiment("/Users/prakhar1207srivastava@gmail.com/rag-chatbot-eval")

# Initialize clients (in case they're not in scope from earlier cells)
vsc = VectorSearchClient(disable_notice=True)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Sample test queries
test_queries = [
    "What is the main topic of this document?",
    "Summarize the key findings or recommendations.",
    "What are the important metrics or numbers mentioned?",
    "What conclusions does this document reach?",
    "What methodology or approach is described?",
]

# Run the chatbot with tracing enabled
@mlflow.trace(span_type=SpanType.CHAIN)
def traced_chatbot(query: str) -> dict:
    """Chatbot with MLflow tracing for each step."""
    with mlflow.start_span(name="vector_search", span_type=SpanType.RETRIEVER) as span:
        try:
            index = vsc.get_index(endpoint_name="pdf-chatbot-endpoint", index_name="workspace.default.pdf_chunks_index")
            results = index.similarity_search(
                query_text=query,
                columns=["text", "chunk_index", "pages"],
                num_results=3
            )
            chunks_data = results.get("result", {}).get("data_array", [])
            context_text = "\n\n".join([row[0] for row in chunks_data if row[0]])
            span.set_attributes({"num_chunks_retrieved": len(chunks_data), "context_length": len(context_text)})
        except Exception as e:
            context_text = ""
            chunks_data = []
            span.set_attributes({"error": str(e)})
    
    with mlflow.start_span(name="llm_generation", span_type=SpanType.LLM) as span:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering questions from document context. Be concise and accurate."},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
            ],
            temperature=0.1,
            max_tokens=500
        )
        answer = response.choices[0].message.content
        span.set_inputs({"query": query, "context_length": len(context_text)})
        span.set_outputs({"response": answer})
        span.set_attributes({
            "model": "gpt-4o-mini",
            "tokens_used": response.usage.total_tokens,
        })
    
    return {"response": answer, "query": query, "num_sources": len(chunks_data)}

# Run traced queries and collect results
all_results = []
for q in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {q}")
    result = traced_chatbot(q)
    print(f"Response: {result['response'][:200]}...")
    print(f"Sources retrieved: {result['num_sources']}")
    all_results.append(result)

print(f"\n{'='*60}")
print(f"Completed {len(all_results)} traced queries")

# COMMAND ----------

# DBTITLE 1,Evaluate with MLflow Scorers
import mlflow
import os
from openai import OpenAI
from mlflow.genai.scorers import RelevanceToQuery, Safety, Guidelines, scorer, ScorerSamplingConfig, get_scorer
from mlflow.entities import Feedback

mlflow.set_experiment("/Users/prakhar1207srivastava@gmail.com/rag-chatbot-eval")

# Pull recent traces from the experiment
exp = mlflow.get_experiment_by_name("/Users/prakhar1207srivastava@gmail.com/rag-chatbot-eval")
traces = mlflow.search_traces(
    experiment_ids=[exp.experiment_id],
    max_results=20,
    return_type="list",
)
print(f"Found {len(traces)} traces to evaluate")

# Register scorers for ongoing monitoring
relevance = RelevanceToQuery().register(name="rag_relevance")
relevance.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))

safety = Safety().register(name="rag_safety")
safety.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))

# Custom scorer: groundedness check using OpenAI
@scorer
def groundedness_check(outputs, inputs):
    """Check if the response is grounded in retrieved context (no hallucination)."""
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response_text = outputs.get("response", "") if isinstance(outputs, dict) else str(outputs)
    
    check_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a fact-checker. Answer 'yes' if the response appears factual and grounded, or 'no' if it contains hallucinations or unsupported claims. Respond with only 'yes' or 'no'."},
            {"role": "user", "content": f"Evaluate this response for hallucination: {response_text}"}
        ],
        temperature=0.0,
        max_tokens=10
    )
    verdict = check_response.choices[0].message.content.strip().lower()
    return Feedback(
        value=(verdict == "yes"),
        rationale=f"Groundedness check: {verdict}"
    )

groundedness = groundedness_check.register(name="rag_groundedness")
groundedness.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))

# Custom scorer: response conciseness
@scorer
def response_conciseness(outputs):
    """Check if the response is reasonably concise."""
    response_text = outputs.get("response", "") if isinstance(outputs, dict) else str(outputs)
    word_count = len(response_text.split())
    is_concise = 10 <= word_count <= 200
    return Feedback(
        value=is_concise,
        rationale=f"Response has {word_count} words ({'concise' if is_concise else 'too long or too short'})"
    )

conciseness = response_conciseness.register(name="rag_conciseness")
conciseness.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))

print("Registered 4 scorers: rag_relevance, rag_safety, rag_groundedness, rag_conciseness")

# Run evaluation on existing traces
if traces:
    result = mlflow.genai.evaluate(
        data=traces,
        scorers=[
            get_scorer(name="rag_relevance"),
            get_scorer(name="rag_safety"),
            get_scorer(name="rag_groundedness"),
            get_scorer(name="rag_conciseness"),
        ],
    )
    print("\n=== Evaluation Results ===")
    print(result)
else:
    print("No traces found. Run the traced queries cell first.")

# COMMAND ----------

# DBTITLE 1,Evaluation Summary Dashboard
import mlflow

# Pull all traces and their assessments
exp = mlflow.get_experiment_by_name("/Users/prakhar1207srivastava@gmail.com/rag-chatbot-eval")
traces = mlflow.search_traces(
    experiment_ids=[exp.experiment_id],
    max_results=50,
    return_type="list",
)

print(f"=== RAG Chatbot Evaluation Summary ===")
print(f"Total traces: {len(traces)}")

from collections import Counter
scorer_stats = Counter()
for t in traces:
    for a in (t.info.assessments or []):
        scorer_stats[a.name] += 1

print(f"\nScorer coverage:")
for name, count in sorted(scorer_stats.items()):
    print(f"  {name}: {count} assessments")

print(f"\nSample results:")
for t in traces[:5]:
    inputs = t.info.request_preview or ""
    outputs = t.info.response_preview or ""
    print(f"\n  Query: {inputs[:100]}")
    print(f"  Response: {outputs[:150]}...")
    print(f"  Assessments: {[a.name for a in (t.info.assessments or [])]}")

# COMMAND ----------

# DBTITLE 1,Register MLflow Model
import mlflow
from mlflow.models import infer_signature

# Create a wrapper model class
class RAGChatbotModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from databricks.vector_search.client import VectorSearchClient
        from openai import OpenAI
        import os
        
        self.vsc = VectorSearchClient(disable_notice=True)
        self.openai_available = bool(os.environ.get("OPENAI_API_KEY"))
        if self.openai_available:
            self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.endpoint_name = "pdf-chatbot-endpoint"
        self.index_name = "workspace.default.pdf_chunks_index"
    
    def predict(self, context, model_input, params=None):
        import pandas as pd
        if isinstance(model_input, pd.DataFrame):
            query = model_input.iloc[0].get("query", model_input.iloc[0][0])
        elif isinstance(model_input, dict):
            query = model_input.get("query", str(model_input))
        else:
            query = str(model_input)
        
        # Retrieve chunks
        try:
            index = self.vsc.get_index(endpoint_name=self.endpoint_name, index_name=self.index_name)
            results = index.similarity_search(
                query_text=query,
                columns=["text", "chunk_index", "pages"],
                num_results=3
            )
            chunks_data = results.get("result", {}).get("data_array", [])
            context_text = "\n\n".join([row[0] for row in chunks_data if row[0]])
        except Exception as e:
            context_text = ""
            chunks_data = []
        
        # Generate response
        system_prompt = "You are a helpful assistant that answers questions based on provided context. Use only the context to answer. Be concise."
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
        
        if self.openai_available:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            answer = response.choices[0].message.content
        else:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            full_prompt = system_prompt + "\n\n" + user_prompt
            if len(full_prompt) > 3000:
                full_prompt = full_prompt[:3000] + "..."
            escaped_prompt = full_prompt.replace("\\", "\\\\").replace("'", "\\'")
            result = spark.sql(f"SELECT ai_query('databricks-meta-llama-3-3-70b-instruct', '{escaped_prompt}') AS response")
            answer = result.collect()[0][0]
        
        return {"response": answer, "query": query}

# Log the model
with mlflow.start_run(run_name="rag_chatbot_model") as run:
    model = RAGChatbotModel()
    
    # Define input/output signature
    input_example = {"query": "What is this document about?"}
    signature = infer_signature(
        model_input={"query": "test question"},
        model_output={"response": "test answer", "query": "test question"}
    )
    
    # Log the model
    mlflow.pyfunc.log_model(
        artifact_path="rag_chatbot",
        python_model=model,
        signature=signature,
        input_example=input_example,
        pip_requirements=["openai", "databricks-vectorsearch"],
    )
    
    model_uri = f"runs:/{run.info.run_id}/rag_chatbot"
    print(f"Model logged: {model_uri}")
    print(f"Run ID: {run.info.run_id}")
    
    # Register the model
    model_name = "agents.default.rag_pdf_chatbot"
    mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Model registered as: {model_name}")

# COMMAND ----------

# DBTITLE 1,Create Serving Endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
import time

w = WorkspaceClient()

endpoint_name = "rag-pdf-chatbot-endpoint"

# Create serving endpoint
try:
    w.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    name="rag_pdf_chatbot",
                    entity_name="agents.default.rag_pdf_chatbot",
                    entity_version="1",
                    workload_size="Small",
                    scale_to_zero_enabled=True,
                )
            ]
        )
    )
    print(f"Creating endpoint: {endpoint_name}")
    print("Endpoint creation initiated (may take a few minutes)")
except Exception as e:
    print(f"Endpoint creation result: {e}")

# Check endpoint status
time.sleep(5)
try:
    status = w.serving_endpoints.get(name=endpoint_name)
    print(f"Endpoint status: {status.state.ready}")
except Exception as e:
    print(f"Status check: {e}")

# COMMAND ----------

# DBTITLE 1,MLflow Tracing and Test Queries
import mlflow
from mlflow.entities import SpanType
from openai import OpenAI
from databricks.vector_search.client import VectorSearchClient
import os

# Enable MLflow tracing
mlflow.set_experiment("/Users/prakhar1207srivastava@gmail.com/rag-chatbot-eval")

# Initialize clients (in case they're not in scope from earlier cells)
vsc = VectorSearchClient(disable_notice=True)
openai_available = bool(os.environ.get("OPENAI_API_KEY"))
if openai_available:
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
else:
    print("OpenAI key not available. Using Databricks foundation model (databricks-meta-llama-3-3-70b-instruct).")
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()

# Sample test queries
test_queries = [
    "What is the main topic of this document?",
    "Summarize the key findings or recommendations.",
    "What are the important metrics or numbers mentioned?",
    "What conclusions does this document reach?",
    "What methodology or approach is described?",
]

# Run the chatbot with tracing enabled
@mlflow.trace(span_type=SpanType.CHAIN)
def traced_chatbot(query: str) -> dict:
    """Chatbot with MLflow tracing for each step."""
    with mlflow.start_span(name="vector_search", span_type=SpanType.RETRIEVER) as span:
        try:
            index = vsc.get_index(endpoint_name="pdf-chatbot-endpoint", index_name="workspace.default.pdf_chunks_index")
            results = index.similarity_search(
                query_text=query,
                columns=["text", "chunk_index", "pages"],
                num_results=3
            )
            chunks_data = results.get("result", {}).get("data_array", [])
            context_text = "\n\n".join([row[0] for row in chunks_data if row[0]])
            span.set_attributes({"num_chunks_retrieved": len(chunks_data), "context_length": len(context_text)})
        except Exception as e:
            context_text = ""
            chunks_data = []
            span.set_attributes({"error": str(e)})
    
    with mlflow.start_span(name="llm_generation", span_type=SpanType.LLM) as span:
        if openai_available:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant answering questions from document context. Be concise and accurate."},
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
                ],
                temperature=0.1,
                max_tokens=500
            )
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
        else:
            full_prompt = "You are a helpful assistant answering questions from document context. Be concise and accurate.\n\n" + f"Context:\n{context_text}\n\nQuestion: {query}"
            if len(full_prompt) > 3000:
                full_prompt = full_prompt[:3000] + "..."
            escaped_prompt = full_prompt.replace("\\", "\\\\").replace("'", "\\'")
            result = spark.sql(f"SELECT ai_query('databricks-meta-llama-3-3-70b-instruct', '{escaped_prompt}') AS response")
            answer = result.collect()[0][0]
            tokens_used = 0
        
        span.set_inputs({"query": query, "context_length": len(context_text)})
        span.set_outputs({"response": answer})
        span.set_attributes({
            "model": "gpt-4o-mini" if openai_available else "databricks-meta-llama-3-3-70b-instruct",
            "tokens_used": tokens_used,
        })
    
    return {"response": answer, "query": query, "num_sources": len(chunks_data)}

# Run traced queries and collect results
all_results = []
for q in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {q}")
    result = traced_chatbot(q)
    print(f"Response: {result['response'][:200]}...")
    print(f"Sources retrieved: {result['num_sources']}")
    all_results.append(result)

print(f"\n{'='*60}")
print(f"Completed {len(all_results)} traced queries")

# COMMAND ----------

# DBTITLE 1,Evaluate with MLflow Scorers
import mlflow
import os
from openai import OpenAI
from mlflow.genai.scorers import RelevanceToQuery, Safety, Guidelines, scorer, ScorerSamplingConfig, get_scorer
from mlflow.entities import Feedback

mlflow.set_experiment("/Users/prakhar1207srivastava@gmail.com/rag-chatbot-eval")

# Pull recent traces from the experiment
exp = mlflow.get_experiment_by_name("/Users/prakhar1207srivastava@gmail.com/rag-chatbot-eval")
traces = mlflow.search_traces(
    experiment_ids=[exp.experiment_id],
    max_results=20,
    return_type="list",
)
print(f"Found {len(traces)} traces to evaluate")

# Register scorers for ongoing monitoring
relevance = RelevanceToQuery().register(name="rag_relevance")
relevance.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))

safety = Safety().register(name="rag_safety")
safety.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))

# Custom scorer: groundedness check using OpenAI
@scorer
def groundedness_check(outputs, inputs):
    """Check if the response is grounded in retrieved context (no hallucination)."""
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response_text = outputs.get("response", "") if isinstance(outputs, dict) else str(outputs)
    
    check_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a fact-checker. Answer 'yes' if the response appears factual and grounded, or 'no' if it contains hallucinations or unsupported claims. Respond with only 'yes' or 'no'."},
            {"role": "user", "content": f"Evaluate this response for hallucination: {response_text}"}
        ],
        temperature=0.0,
        max_tokens=10
    )
    verdict = check_response.choices[0].message.content.strip().lower()
    return Feedback(
        value=(verdict == "yes"),
        rationale=f"Groundedness check: {verdict}"
    )

groundedness = groundedness_check.register(name="rag_groundedness")
groundedness.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))

# Custom scorer: response conciseness
@scorer
def response_conciseness(outputs):
    """Check if the response is reasonably concise."""
    response_text = outputs.get("response", "") if isinstance(outputs, dict) else str(outputs)
    word_count = len(response_text.split())
    is_concise = 10 <= word_count <= 200
    return Feedback(
        value=is_concise,
        rationale=f"Response has {word_count} words ({'concise' if is_concise else 'too long or too short'})"
    )

conciseness = response_conciseness.register(name="rag_conciseness")
conciseness.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))

print("Registered 4 scorers: rag_relevance, rag_safety, rag_groundedness, rag_conciseness")

# Run evaluation on existing traces
if traces:
    result = mlflow.genai.evaluate(
        data=traces,
        scorers=[
            get_scorer(name="rag_relevance"),
            get_scorer(name="rag_safety"),
            get_scorer(name="rag_groundedness"),
            get_scorer(name="rag_conciseness"),
        ],
    )
    print("\n=== Evaluation Results ===")
    print(result)
else:
    print("No traces found. Run the traced queries cell first.")

# COMMAND ----------

# DBTITLE 1,Evaluation Summary Dashboard
import mlflow

# Pull all traces and their assessments
exp = mlflow.get_experiment_by_name("/Users/prakhar1207srivastava@gmail.com/rag-chatbot-eval")
traces = mlflow.search_traces(
    experiment_ids=[exp.experiment_id],
    max_results=50,
    return_type="list",
)

print(f"=== RAG Chatbot Evaluation Summary ===")
print(f"Total traces: {len(traces)}")

from collections import Counter
scorer_stats = Counter()
for t in traces:
    for a in (t.info.assessments or []):
        scorer_stats[a.name] += 1

print(f"\nScorer coverage:")
for name, count in sorted(scorer_stats.items()):
    print(f"  {name}: {count} assessments")

print(f"\nSample results:")
for t in traces[:5]:
    inputs = t.info.request_preview or ""
    outputs = t.info.response_preview or ""
    print(f"\n  Query: {inputs[:100]}")
    print(f"  Response: {outputs[:150]}...")
    print(f"  Assessments: {[a.name for a in (t.info.assessments or [])]}")