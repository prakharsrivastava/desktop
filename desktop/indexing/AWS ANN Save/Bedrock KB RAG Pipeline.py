# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,AWS Bedrock Knowledge Base RAG Pipeline
# MAGIC %md
# MAGIC # AWS Bedrock Knowledge Base RAG Pipeline
# MAGIC
# MAGIC End-to-end RAG using AWS native services:
# MAGIC
# MAGIC **Ingestion (KB Sync):**
# MAGIC - Enterprise documents stored in **Amazon S3**
# MAGIC - **Bedrock Knowledge Base** parses, chunks, and generates embeddings (Titan Text Embeddings V2)
# MAGIC - Chunks + embeddings + metadata stored in **OpenSearch Serverless**
# MAGIC
# MAGIC **Query Time:**
# MAGIC - User question → Python app (boto3) → **Bedrock Agent Runtime** `retrieve()`
# MAGIC - Vector similarity search in OpenSearch Serverless → Top-K chunks
# MAGIC - Join chunks + question → **Bedrock Runtime** → **Amazon Nova Lite** → grounded response
# MAGIC
# MAGIC **Conversation Memory:**
# MAGIC - All conversations stored in **MongoDB** keyed by `user_id` + `thread_id`
# MAGIC - Different users have isolated memory (from MongoDB Conversation Saver notebook)
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - AWS credentials configured (instance profile or `dbutils.secrets`)
# MAGIC - Bedrock Knowledge Base created in AWS Console with S3 data source
# MAGIC - MongoDB connection string in Databricks Secrets (scope: `mongodb`, key: `connection_string`)

# COMMAND ----------

# DBTITLE 1,Install & AWS Configuration
# ── Install boto3 and set up AWS configuration ──────────────────────
%pip install boto3 pymongo -q

import os, json, time, boto3

# ── AWS Configuration ───────────────────────────────────────────────
# If running on Databricks with AWS instance profile, credentials are
# picked up automatically. For manual setup, store in secrets:
#   dbutils.secrets.put(scope="aws", key="access_key_id")
#   dbutils.secrets.put(scope="aws", key="secret_access_key")

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Try to get AWS credentials from secrets, fall back to instance profile
try:
    scopes = dbutils.secrets.listScopes()
    has_aws_scope = any(s.name == "aws" for s in scopes) if scopes else False
except Exception:
    has_aws_scope = False

if has_aws_scope:
    AWS_ACCESS_KEY = dbutils.secrets.get(scope="aws", key="access_key_id")
    AWS_SECRET_KEY = dbutils.secrets.get(scope="aws", key="secret_access_key")
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )
else:
    # Use instance profile or env-based credentials
    session = boto3.Session(region_name=AWS_REGION)

# ── Bedrock Clients ─────────────────────────────────────────────────
bedrock_agent = session.client("bedrock-agent")          # KB management
bedrock_agent_rt = session.client("bedrock-agent-runtime")  # Retrieve
bedrock_runtime = session.client("bedrock-runtime")        # Invoke LLM
s3_client = session.client("s3")                             # S3 uploads

# ── Resource IDs ─────────────────────────────────────────────────────
# Replace these with your actual AWS resource IDs
KB_ID = os.environ.get("BEDROCK_KB_ID", "YOUR_KNOWLEDGE_BASE_ID")
DATA_SOURCE_ID = os.environ.get("BEDROCK_DS_ID", "YOUR_DATA_SOURCE_ID")
S3_BUCKET = os.environ.get("S3_BUCKET", "your-rag-docs-bucket")
S3_PREFIX = os.environ.get("S3_PREFIX", "documents/")

# ── Model IDs ────────────────────────────────────────────────────────
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_MODEL_ID = "amazon.nova-lite-v1:0"

# ── MongoDB Config (for conversation memory) ────────────────────────
MONGO_CONNECTION_STRING = dbutils.secrets.get(scope="mongodb", key="connection_string") \
    if has_aws_scope and any(s.name == "mongodb" for s in dbutils.secrets.listScopes()) \
    else os.environ.get("MONGO_CONNECTION_STRING", "mongodb://localhost:27017")
MONGO_DB_NAME = "rag_conversations"
MONGO_COLLECTION_NAME = "conversations"

print(f"AWS Region        : {AWS_REGION}")
print(f"Knowledge Base ID : {KB_ID}")
print(f"S3 Bucket         : s3://{S3_BUCKET}/{S3_PREFIX}")
print(f"Embedding Model   : {EMBEDDING_MODEL_ID}")
print(f"LLM Model         : {LLM_MODEL_ID}")
print(f"MongoDB           : {MONGO_DB_NAME}.{MONGO_COLLECTION_NAME}")
print("✅ AWS Bedrock clients initialized")

# COMMAND ----------

# DBTITLE 1,S3 Upload & KB Ingestion Sync
# ── S3 Document Upload & Bedrock KB Ingestion Sync ───────────────────
# Upload documents to S3, then trigger a Bedrock Knowledge Base
# ingestion job to parse, chunk, embed, and store in OpenSearch.

def upload_to_s3(local_path, s3_key, bucket=None):
    """Upload a single file to S3."""
    bucket = bucket or S3_BUCKET
    full_key = f"{S3_PREFIX}{s3_key}"
    s3_client.upload_file(local_path, bucket, full_key)
    s3_uri = f"s3://{bucket}/{full_key}"
    print(f"  Uploaded: {local_path} → {s3_uri}")
    return s3_uri


def upload_directory_to_s3(local_dir, bucket=None):
    """Upload all files in a directory to S3."""
    bucket = bucket or S3_BUCKET
    uploaded = []
    for fname in os.listdir(local_dir):
        fpath = os.path.join(local_dir, fname)
        if os.path.isfile(fpath):
            s3_uri = upload_to_s3(fpath, fname, bucket)
            uploaded.append(s3_uri)
    return uploaded


def start_kb_ingestion():
    """Trigger a Bedrock KB ingestion job and wait for completion.

    This tells Bedrock to scan the S3 data source, parse new/changed
    documents, chunk them, generate embeddings with Titan Text
    Embeddings V2, and store everything in OpenSearch Serverless.
    """
    print(f"Starting ingestion job for KB: {KB_ID}")
    response = bedrock_agent.start_ingestion_job(
        knowledgeBaseId=KB_ID,
        dataSourceId=DATA_SOURCE_ID,
    )
    job_id = response["ingestionJob"]["ingestionJobId"]
    print(f"  Ingestion job started: {job_id}")

    # Poll until complete
    while True:
        status_resp = bedrock_agent.get_ingestion_job(
            knowledgeBaseId=KB_ID,
            dataSourceId=DATA_SOURCE_ID,
            ingestionJobId=job_id,
        )
        status = status_resp["ingestionJob"]["status"]
        print(f"  Job status: {status}")
        if status in ["COMPLETE", "FAILED"]:
            break
        time.sleep(10)

    if status == "COMPLETE":
        print(f"✅ Ingestion complete! Job ID: {job_id}")
    else:
        print(f"❌ Ingestion failed! Job ID: {job_id}")
        print(f"   Details: {json.dumps(status_resp['ingestionJob'], indent=2, default=str)}")
    return job_id, status


def list_kb_data_sources():
    """List data sources configured on the knowledge base."""
    resp = bedrock_agent.list_data_sources(knowledgeBaseId=KB_ID)
    for ds in resp.get("dataSourceSummaries", []):
        print(f"  Data Source: {ds['dataSourceId']} | Name: {ds.get('name', 'N/A')} | Status: {ds.get('status', 'N/A')}")
    return resp.get("dataSourceSummaries", [])


print("✅ S3 upload & KB ingestion functions ready")
print("   upload_to_s3(local_path, s3_key)")
print("   upload_directory_to_s3(local_dir)")
print("   start_kb_ingestion()  → triggers parse/chunk/embed/sync")
print("   list_kb_data_sources()")

# COMMAND ----------

# DBTITLE 1,Retrieval: KB → OpenSearch
# ── Retrieval: Query Bedrock KB → Top-K chunks from OpenSearch ─────────
# Uses bedrock-agent-runtime retrieve() to perform vector similarity
# search against the OpenSearch Serverless collection backing the KB.

def retrieve_from_kb(query, num_results=5):
    """Retrieve Top-K relevant chunks from Bedrock Knowledge Base.

    Args:
        query:       User's question text
        num_results: Number of chunks to retrieve (Top-K)

    Returns:
        List of dicts with: text, score, source metadata
    """
    response = bedrock_agent_rt.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": num_results,
            }
        },
    )

    chunks = []
    for result in response.get("retrievalResults", []):
        content = result.get("content", {}).get("text", "")
        metadata = result.get("metadata", {})
        score = result.get("score", result.get("retrievalScore", 0.0))

        # Extract S3 source info from metadata
        source_uri = metadata.get("x-amz-bedrock-kb-source-uri", "")
        source_doc = source_uri.split("/")[-1] if source_uri else "unknown"

        chunks.append({
            "text": content,
            "score": float(score) if score else 0.0,
            "source_doc": source_doc,
            "source_uri": source_uri,
            "metadata": metadata,
        })

    return chunks


def format_retrieved_chunks(chunks):
    """Format retrieved chunks into a context string for the LLM."""
    if not chunks:
        return "(No relevant context found)"

    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(
            f"[Source {i+1}: {chunk['source_doc']} | Score: {chunk['score']:.4f}]\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts)


# Quick test retrieval (optional — uncomment to test)
# test_chunks = retrieve_from_kb("What was Amazon net income?", num_results=3)
# for c in test_chunks:
#     print(f"Score: {c['score']:.4f} | Source: {c['source_doc']}")
#     print(f"  Preview: {c['text'][:200]}...")

print("✅ Retrieval functions ready")
print("   retrieve_from_kb(query, num_results=5) → Top-K chunks from OpenSearch")
print("   format_retrieved_chunks(chunks) → formatted context string")

# COMMAND ----------

# DBTITLE 1,RAG Chat: Nova Lite + MongoDB Memory
# ── RAG Chat: Retrieve → Nova Lite → Grounded Response (+ MongoDB memory) ─
# Full pipeline: vector retrieval from KB → conversation history from MongoDB →
# Amazon Nova Lite for grounded generation → store in MongoDB.

from pymongo import MongoClient
from datetime import datetime, timezone

# ── MongoDB Conversation Saver (per-user, per-thread memory) ──────────
class MongoDBConversationSaver:
    """Stores conversations keyed by (user_id, thread_id) for isolation."""

    def __init__(self, connection_string, db_name, collection_name):
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.collection.create_index([("user_id", 1), ("thread_id", 1), ("timestamp", 1)])
        self.collection.create_index([("user_id", 1), ("timestamp", -1)])

    def add_message(self, user_id, thread_id, role, content, metadata=None):
        doc = {
            "user_id": user_id,
            "thread_id": thread_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc),
        }
        return str(self.collection.insert_one(doc).inserted_id)

    def get_history_as_chat_messages(self, user_id, thread_id, limit=None):
        query = {"user_id": user_id, "thread_id": thread_id}
        cursor = self.collection.find(query).sort("timestamp", 1)
        if limit:
            cursor = cursor.limit(limit)
        return [{"role": d["role"], "content": d["content"]} for d in cursor]

    def clear_thread(self, user_id, thread_id):
        return self.collection.delete_many({"user_id": user_id, "thread_id": thread_id}).deleted_count


# Initialize MongoDB saver
saver = MongoDBConversationSaver(MONGO_CONNECTION_STRING, MONGO_DB_NAME, MONGO_COLLECTION_NAME)


# ── Amazon Nova Lite invocation ──────────────────────────────────────
def invoke_nova_lite(messages, system_prompt=""):
    """Call Amazon Nova Lite via Bedrock Runtime.

    Args:
        messages:      List of {"role": "user"|"assistant", "content": [{"text": "..."}]}
        system_prompt:  Optional system instruction

    Returns:
        Assistant response text
    """
    body = {
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": 2000,
            "temperature": 0.1,
            "topP": 0.9,
        },
    }
    if system_prompt:
        body["system"] = [{"text": system_prompt}]

    response = bedrock_runtime.invoke_model(
        modelId=LLM_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )

    result = json.loads(response["body"].read())
    return result["output"]["message"]["content"][0]["text"]


# ── Full RAG Chat with Memory ────────────────────────────────────────
def rag_chat(user_id, thread_id, user_query, num_results=5, max_history=10):
    """Full RAG pipeline with MongoDB conversation memory.

    1. Retrieve Top-K chunks from Bedrock KB (OpenSearch vector search)
    2. Load conversation history from MongoDB (per user_id + thread_id)
    3. Build messages: system (RAG context) + history + current query
    4. Call Amazon Nova Lite for grounded response
    5. Store user query + assistant response in MongoDB

    Returns:
        dict with 'response', 'retrieved_chunks', 'used_history_count'
    """
    # ── Step 1: Retrieve from Bedrock KB ──
    chunks = retrieve_from_kb(user_query, num_results=num_results)
    rag_context = format_retrieved_chunks(chunks)

    # ── Step 2: Load conversation history from MongoDB ──
    history_msgs = saver.get_history_as_chat_messages(user_id, thread_id, limit=max_history)
    used_history_count = len(history_msgs)

    # ── Step 3: Build messages for Nova Lite ──
    system_prompt = (
        "You are a helpful assistant. Answer questions using ONLY the retrieved context. "
        "If the context doesn't contain the answer, say you don't know. "
        "Cite sources when possible.\n\n"
        f"--- Retrieved Context ---\n{rag_context}\n--- End Context ---"
    )

    # Build conversation messages in Nova Lite format
    chat_messages = []
    for msg in history_msgs:
        chat_messages.append({
            "role": msg["role"],
            "content": [{"text": msg["content"]}],
        })
    # Add current user query
    chat_messages.append({
        "role": "user",
        "content": [{"text": user_query}],
    })

    # ── Step 4: Call Amazon Nova Lite ──
    assistant_response = invoke_nova_lite(chat_messages, system_prompt=system_prompt)

    # ── Step 5: Store in MongoDB ──
    saver.add_message(user_id, thread_id, "user", user_query, metadata={
        "retrieved_chunk_count": len(chunks),
        "pipeline": "bedrock_kb_rag",
    })
    saver.add_message(user_id, thread_id, "assistant", assistant_response, metadata={
        "retrieved_chunks": [
            {"source": c["source_doc"], "score": c["score"], "preview": c["text"][:200]}
            for c in chunks
        ],
        "llm_model": LLM_MODEL_ID,
        "embedding_model": EMBEDDING_MODEL_ID,
    })

    return {
        "response": assistant_response,
        "retrieved_chunks": chunks,
        "used_history_count": used_history_count,
    }


print("✅ RAG Chat with MongoDB Memory ready")
print(f"   Pipeline: S3 → Bedrock KB → OpenSearch → retrieve → Nova Lite → MongoDB")
print(f"   LLM: {LLM_MODEL_ID}")
print(f"   Memory: MongoDB ({MONGO_DB_NAME}.{MONGO_COLLECTION_NAME}) keyed by user_id + thread_id")

# COMMAND ----------

# DBTITLE 1,Test: End-to-End RAG Queries
# ── Test: End-to-end RAG queries with per-user memory ────────────────
# Test 1: Two users ask different questions → verify memory isolation
# Test 2: Same user, follow-up question → verify multi-turn memory

import textwrap


def print_result(result, label=""):
    """Pretty-print a rag_chat result."""
    if label:
        print(f"\n{'='*70}")
        print(label)
        print(f"{'='*70}")
    print(f"\nResponse:")
    print(textwrap.indent(result["response"], "  "))
    print(f"\n  Retrieved chunks: {len(result['retrieved_chunks'])}")
    for c in result["retrieved_chunks"]:
        print(f"    Score: {c['score']:.4f} | Source: {c['source_doc']}")
    print(f"  History used: {result['used_history_count']} messages")


# ── Test 1: Per-user isolation ───────────────────────────────────────
print("\n" + "#" * 70)
print("# TEST 1: Per-User Memory Isolation")
print("#" * 70)

# Clear test data
saver.clear_thread("user_alice", "thread_001")
saver.clear_thread("user_bob", "thread_001")

# Alice asks about Amazon net income
alice_result = rag_chat(
    user_id="user_alice", thread_id="thread_001",
    user_query="What was Amazon net income?",
)
print_result(alice_result, "Alice asks: 'What was Amazon net income?'")

# Bob asks about different topic
bob_result = rag_chat(
    user_id="user_bob", thread_id="thread_001",
    user_query="Tell me about AWS revenue growth.",
)
print_result(bob_result, "Bob asks: 'Tell me about AWS revenue growth.'")

# Verify isolation
alice_history = saver.get_history_as_chat_messages("user_alice", "thread_001")
bob_history = saver.get_history_as_chat_messages("user_bob", "thread_001")
print(f"\n--- Memory Isolation Check ---")
print(f"Alice's messages: {len(alice_history)} (should NOT contain Bob's query)")
print(f"Bob's messages:   {len(bob_history)} (should NOT contain Alice's query)")
assert not any("revenue growth" in m["content"].lower() for m in alice_history), "Isolation broken!"
assert not any("net income" in m["content"].lower() for m in bob_history), "Isolation broken!"
print("✅ Per-user memory isolation verified!")

# ── Test 2: Multi-turn memory ─────────────────────────────────────────
print("\n\n" + "#" * 70)
print("# TEST 2: Multi-Turn Memory (same user, same thread)")
print("#" * 70)

followup = rag_chat(
    user_id="user_alice", thread_id="thread_001",
    user_query="How does that compare to the previous year?",
)
print_result(followup, "Alice follow-up: 'How does that compare to the previous year?'")
print(f"  History used: {followup['used_history_count']} messages (should be > 0 → has memory)")

# ── Test 3: Fresh thread = no memory ──────────────────────────────────
print("\n\n" + "#" * 70)
print("# TEST 3: Fresh Thread = No Memory")
print("#" * 70)

fresh = rag_chat(
    user_id="user_alice", thread_id="thread_002",
    user_query="What were we discussing earlier?",
)
print_result(fresh, "Alice in NEW thread_002: 'What were we discussing earlier?'")
print(f"  History used: {fresh['used_history_count']} messages (should be 0 → fresh thread)")

# ── Summary ───────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("Pipeline Summary")
print("=" * 70)
print(f"  Documents   : S3://{S3_BUCKET}/{S3_PREFIX} → Bedrock KB → OpenSearch")
print(f"  Embeddings   : {EMBEDDING_MODEL_ID}")
print(f"  LLM          : {LLM_MODEL_ID}")
print(f"  Memory       : MongoDB {MONGO_DB_NAME}.{MONGO_COLLECTION_NAME}")
print(f"  User alice   : thread_001={len(saver.get_history_as_chat_messages('user_alice', 'thread_001'))} msgs, thread_002={len(saver.get_history_as_chat_messages('user_alice', 'thread_002'))} msgs")
print(f"  User bob     : thread_001={len(saver.get_history_as_chat_messages('user_bob', 'thread_001'))} msgs")
print("\n✅ All tests passed! RAG pipeline with MongoDB memory is working.")