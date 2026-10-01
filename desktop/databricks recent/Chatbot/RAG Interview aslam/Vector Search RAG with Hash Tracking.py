# Databricks notebook source
# DBTITLE 1,Setup and document hash tracking
from hashlib import md5
from databricks.sdk import WorkspaceClient
import json

# Example source documents
documents = [
    {"doc_id": "101", "text": "Updated refund policy for premium users."},
    {"doc_id": "102", "text": "Password reset steps for mobile users."},
    {"doc_id": "103", "text": "How to enable two-factor authentication for your account."},
    {"doc_id": "104", "text": "Databricks Vector Search enables semantic search for RAG applications."},
]

# Pretend this is stored from the previous run
old_hashes = {
    "101": "old_hash_value",
    "102": md5("Password reset steps for mobile users.".encode()).hexdigest()
}

def get_hash(text):
    """Generate MD5 hash for document text to detect changes"""
    print("text",text)
    print("text encode",text.encode())
    print("md5 : ",md5(text.encode()))
    print("hexdigest : ",md5(text.encode()).hexdigest())
    return md5(text.encode()).hexdigest()

def detect_changes(documents, old_hashes):
    """Detect which documents have changed since last sync"""
    changed_docs = []
    new_docs = []
    unchanged_docs = []
    
    for doc in documents:
        doc_id = doc["doc_id"]
        current_hash = get_hash(doc["text"])
        print("current_hash",current_hash)
        if doc_id not in old_hashes:
            new_docs.append(doc)
        elif old_hashes[doc_id] != current_hash:
            changed_docs.append(doc)
        else:
            unchanged_docs.append(doc)
    print("new_docs",new_docs)
    print("changed_docs",changed_docs)
    print("unchanged_docs",unchanged_docs)
    return {
        "new": new_docs,
        "changed": changed_docs,
        "unchanged": unchanged_docs
    }

# Detect changes
change_report = detect_changes(documents, old_hashes)

print("📊 Document Change Detection Report:", change_report)
print(f"  New documents: {len(change_report['new'])}")
print(f"  Changed documents: {len(change_report['changed'])}")
print(f"  Unchanged documents: {len(change_report['unchanged'])}")

# Display details
if change_report['new']:
    print("\n✨ New documents:")
    for doc in change_report['new']:
        print(f"  - {doc['doc_id']}: {doc['text'][:50]}...")

if change_report['changed']:
    print("\n🔄 Changed documents:")
    for doc in change_report['changed']:
        print(f"  - {doc['doc_id']}: {doc['text'][:50]}...")

# COMMAND ----------

# DBTITLE 1,Create Delta table for documents
import pandas as pd
from pyspark.sql import SparkSession

# Add hash column to documents
for doc in documents:
    doc['content_hash'] = get_hash(doc['text'])

# Convert to DataFrame
df = spark.createDataFrame([
    (doc['doc_id'], doc['text'], doc['content_hash']) 
    for doc in documents
], ["doc_id", "content", "content_hash"])

# Define table name - adjust catalog and schema as needed
catalog = "workspace"  # Change to your catalog
schema = "default"  # Change to your schema
table_name = f"{catalog}.{schema}.rag_documents"

# Create Delta table
df.write.format("delta").mode("overwrite").saveAsTable(table_name)

print(f"✅ Created table: {table_name}")
print(f"   Total documents: {df.count()}")

# Display sample data
display(spark.table(table_name).limit(5))

# COMMAND ----------

# DBTITLE 1,Enable Change Data Feed and add primary key
# MAGIC %sql
# MAGIC -- Enable Change Data Feed (required for Vector Search Delta Sync)
# MAGIC ALTER TABLE workspace.default.rag_documents 
# MAGIC SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
# MAGIC
# MAGIC -- Set doc_id column to NOT NULL (required for PRIMARY KEY)
# MAGIC ALTER TABLE workspace.default.rag_documents 
# MAGIC ALTER COLUMN doc_id SET NOT NULL;
# MAGIC
# MAGIC -- Add primary key constraint (required for Vector Search)
# MAGIC ALTER TABLE workspace.default.rag_documents 
# MAGIC ADD CONSTRAINT pk_doc_id PRIMARY KEY (doc_id);
# MAGIC
# MAGIC -- Verify table properties
# MAGIC DESCRIBE DETAIL workspace.default.rag_documents;

# COMMAND ----------

# DBTITLE 1,Create Vector Search endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointStatusState, EndpointType

w = WorkspaceClient()

# Endpoint configuration
endpoint_name = "rag-demo-endpoint"

# Check if endpoint exists
try:
    endpoint = w.vector_search_endpoints.get_endpoint(endpoint_name=endpoint_name)
    print(f"✅ Endpoint '{endpoint_name}' jalready exists")
    print(f"   Status: {endpoint.endpoint_status.state}")
    print(f"   Type: {endpoint.endpoint_type}")
except Exception as e:
    if "does not exist" in str(e).lower() or "not found" in str(e).lower():
        print(f"🔧 Creating endpoint '{endpoint_name}'...")
        
        # Create standard endpoint
        endpoint = w.vector_search_endpoints.create_endpoint(
            name=endpoint_name,
            endpoint_type=EndpointType.STANDARD
        )
        
        print(f"✅ Endpoint creation initiated")
        print(f"   Name: {endpoint_name}")
        print(f"   Type: STANDARD")
        print("   Note: Endpoint creation is asynchronous. Wait a few minutes before creating indexes.")
    else:
        raise e

# COMMAND ----------

# DBTITLE 1,Create Vector Search index with managed embeddings
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
    VectorIndexType
)

# Define table configuration
catalog = "workspace"
schema = "default"
table_name = f"{catalog}.{schema}.rag_documents"

# Index configuration
index_name = f"{catalog}.{schema}.rag_documents_index"

# Check if index exists
try:
    existing_index = w.vector_search_indexes.get_index(index_name=index_name)
    print(f"✅ Index '{index_name}' already exists")
    print(f"   Status: {existing_index.status.detailed_state}")
except Exception as e:
    if "does not exist" in str(e).lower() or "not found" in str(e).lower():
        print(f"🔧 Creating Vector Search index '{index_name}'...")
        
        # Create Delta Sync index with managed embeddings
        index = w.vector_search_indexes.create_index(
            name=index_name,
            endpoint_name=endpoint_name,
            primary_key="doc_id",
            index_type=VectorIndexType.DELTA_SYNC,
            delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                source_table=table_name,
                embedding_source_columns=[
                    EmbeddingSourceColumn(
                        name="content",  # Text column to embed
                        embedding_model_endpoint_name="databricks-gte-large-en"  # Built-in embedding model
                    )
                ],
                pipeline_type=PipelineType.TRIGGERED  # Manual sync control
            )
        )
        
        print(f"✅ Index creation initiated")
        print(f"   Index: {index_name}")
        print(f"   Source: {table_name}")
        print(f"   Embedding model: databricks-gte-large-en (1024 dimensions)")
        print(f"   Pipeline: TRIGGERED (manual sync)")
        print("   Note: Index creation is asynchronous. It may take several minutes.")
    else:
        raise e

# COMMAND ----------

# DBTITLE 1,Sync index and check status
import time

# Check if index is ready
index_info = w.vector_search_indexes.get_index(index_name=index_name)

if not index_info.status.ready:
    print(f"⚠️ Index is not ready yet")
    print(f"   Status: {index_info.status.message or 'Provisioning...'}")
    print(f"\n💡 Index creation is asynchronous and can take several minutes.")
    print(f"   Please wait for the index to be ready before syncing.")
    print(f"   You can check status by running this cell again.")
else:
    print(f"✅ Index is ready")
    
    # Trigger index sync (for TRIGGERED pipeline type)
    print(f"\n🔄 Syncing index '{index_name}'...")
    w.vector_search_indexes.sync_index(index_name=index_name)

    # Wait a bit for sync to complete
    time.sleep(5)
    
    # Get index details
    index_info = w.vector_search_indexes.get_index(index_name=index_name)

    print(f"\n✅ Index Status:")
    print(f"   Name: {index_info.name}")
    print(f"   Ready: {index_info.status.ready}")
    print(f"   Primary key: {index_info.primary_key}")
    print(f"   Endpoint: {index_info.endpoint_name}")
    
    if index_info.delta_sync_index_spec:
        print(f"   Source table: {index_info.delta_sync_index_spec.source_table}")
        if index_info.delta_sync_index_spec.embedding_source_columns:
            for col in index_info.delta_sync_index_spec.embedding_source_columns:
                print(f"   Embedding column: {col.name} -> {col.embedding_model_endpoint_name}")

# COMMAND ----------

# DBTITLE 1,Query the vector search index
# Query the index with semantic search
query_text = "How do I reset my password?"

print(f"🔍 Searching for: '{query_text}'\n")

try:
    results = w.vector_search_indexes.query_index(
        index_name=index_name,
        columns=["doc_id", "content", "content_hash"],
        query_text=query_text,
        num_results=3
    )
    print(results)
    print(f"✅ Found {len(results.result.data_array)} relevant documents:\n")
    
    for i, doc in enumerate(results.result.data_array, 1):
        doc_id = doc[0]
        content = doc[1]
        content_hash = doc[2]
        score = doc[-1]  # Similarity score is the last column
        
        print(f"{i}. Document ID: {doc_id}")
        print(f"   Similarity Score: {score:.4f}")
        print(f"   Content: {content}")
        print(f"   Hash: {content_hash[:16]}...")
        print()
        
except Exception as e:
    print(f"⚠️ Query failed: {e}")
    print("Note: If the index was just created, wait a few minutes for it to be ready.")

# COMMAND ----------

# DBTITLE 1,Incremental updates using hash detection
# Simulate document updates
new_documents = [
    {"doc_id": "101", "text": "UPDATED: New refund policy for all users, including premium."},  # Changed
    {"doc_id": "102", "text": "Password reset steps for mobile users."},  # Unchanged
    {"doc_id": "105", "text": "Guide to using Databricks notebooks for data science."},  # New
]

# Load current hashes from the table
current_hashes = {}
for row in spark.table(table_name).select("doc_id", "content_hash").collect():
    current_hashes[row.doc_id] = row.content_hash

# Detect changes
change_report = detect_changes(new_documents, current_hashes)

print("📊 Incremental Update Analysis:")
print(f"  New documents: {len(change_report['new'])}")
print(f"  Changed documents: {len(change_report['changed'])}")
print(f"  Unchanged documents: {len(change_report['unchanged'])}")
print()

# Only process documents that changed or are new
docs_to_update = change_report['new'] + change_report['changed']

if docs_to_update:
    print(f"📥 Upserting {len(docs_to_update)} documents...\n")
    
    # Add hashes
    for doc in docs_to_update:
        doc['content_hash'] = get_hash(doc['text'])
    
    # Create DataFrame for updates
    update_df = spark.createDataFrame([
        (doc['doc_id'], doc['text'], doc['content_hash']) 
        for doc in docs_to_update
    ], ["doc_id", "content", "content_hash"])
    
    update_df.show()
    # Merge into existing table
    update_df.write.format("delta").mode("append").saveAsTable(table_name)
    
    # Check if index is ready before syncing
    index_info = w.vector_search_indexes.get_index(index_name=index_name)
    
    if not index_info.status.ready:
        print(f"⚠️ Index is not ready yet. Skipping sync.")
        print(f"   Status: {index_info.status.message or 'Provisioning...'}")
        print(f"   Please wait for the index to be ready, then run this cell again.")
    else:
        # Trigger sync to update the index
        print(f"🔄 Syncing vector index...")
        w.vector_search_indexes.sync_index(index_name=index_name)
    
        print(f"✅ Update complete!")
        print(f"   Documents updated in table: {table_name}")
        print(f"   Vector index synced: {index_name}")
    
    # Show updated documents
    if change_report['changed']:
        print("\n🔄 Changed documents:")
        for doc in change_report['changed']:
            print(f"  - {doc['doc_id']}: {doc['text'][:60]}...")
    
    if change_report['new']:
        print("\n✨ New documents:")
        for doc in change_report['new']:
            print(f"  - {doc['doc_id']}: {doc['text'][:60]}...")
else:
    print("✅ No changes detected. Skip update.")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM workspace.default.rag_documents 

# COMMAND ----------

# DBTITLE 1,Check table status and contents
# MAGIC %sql
# MAGIC -- Check table properties (verify Change Data Feed is enabled)
# MAGIC SHOW TBLPROPERTIES workspace.default.rag_documents;
# MAGIC
# MAGIC -- Check table details and constraints
# MAGIC DESCRIBE TABLE EXTENDED workspace.default.rag_documents;
# MAGIC
# MAGIC -- View all documents with their hashes
# MAGIC SELECT * FROM workspace.default.rag_documents ORDER BY doc_id;
# MAGIC
# MAGIC -- Count total documents
# MAGIC SELECT COUNT(*) as total_documents FROM workspace.default.rag_documents;
# MAGIC
# MAGIC -- Check for duplicate documents (should be 0 with PRIMARY KEY)
# MAGIC SELECT doc_id, COUNT(*) as count 
# MAGIC FROM workspace.default.rag_documents 
# MAGIC GROUP BY doc_id 
# MAGIC HAVING COUNT(*) > 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ============================================
# MAGIC -- 1. VERIFY CHANGE DATA FEED IS ENABLED
# MAGIC -- ============================================
# MAGIC SHOW TBLPROPERTIES workspace.default.rag_documents;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC
# MAGIC -- ============================================
# MAGIC -- 2. VIEW TABLE HISTORY (all operations)
# MAGIC -- ============================================
# MAGIC DESCRIBE HISTORY workspace.default.rag_documents;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ============================================
# MAGIC -- 3. READ CHANGE DATA FEED - See what changed
# MAGIC -- Shows INSERT, UPDATE_PREIMAGE, UPDATE_POSTIMAGE, DELETE
# MAGIC -- ============================================
# MAGIC SELECT 
# MAGIC     _change_type,
# MAGIC     _commit_version,
# MAGIC     _commit_timestamp,
# MAGIC     doc_id,
# MAGIC     content,
# MAGIC     content_hash
# MAGIC FROM table_changes('workspace.default.rag_documents', 9)
# MAGIC ORDER BY _commit_version DESC, doc_id;

# COMMAND ----------

# DBTITLE 1,Check Change Data Feed (CDC) status and history
# MAGIC %sql
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC -- ============================================
# MAGIC -- 4. CURRENT TABLE STATE WITH DUPLICATES
# MAGIC -- ============================================
# MAGIC SELECT 
# MAGIC     doc_id, 
# MAGIC     COUNT(*) as duplicate_count,
# MAGIC     COLLECT_LIST(content) as all_versions,
# MAGIC     COLLECT_LIST(content_hash) as all_hashes
# MAGIC FROM workspace.default.rag_documents 
# MAGIC GROUP BY doc_id 
# MAGIC ORDER BY duplicate_count DESC, doc_id;