# Databricks notebook source
# MAGIC %md
# MAGIC # Production Document Chunking Pipeline
# MAGIC
# MAGIC **Distributed PDF Processing for Millions of Documents at Scale**
# MAGIC
# MAGIC This production-grade pipeline processes PDFs with images and signatures using:
# MAGIC
# MAGIC - **Auto Loader**: Continuous ingestion with checkpointing for millions of files
# MAGIC - **ai_parse_document v2.0**: Structure extraction with image/signature AI descriptions
# MAGIC - **Distributed Spark**: Pandas UDFs for parallel chunking across cluster workers
# MAGIC - **Delta Lake**: ACID storage with partitioning, time travel, and error handling
# MAGIC
# MAGIC ## Architecture
# MAGIC
# MAGIC ```
# MAGIC Cloud Storage (S3/ADLS/GCS) → Auto Loader → ai_parse_document → Pandas UDF Chunking → Delta Tables
# MAGIC                                      ↓                                    ↓
# MAGIC                              Error Handling                        Quality Metrics
# MAGIC ```
# MAGIC
# MAGIC ## Features
# MAGIC
# MAGIC - ✅ **Handles images & signatures**: AI-generated descriptions for figures
# MAGIC - ✅ **Horizontal scaling**: Processes documents in parallel across all workers
# MAGIC - ✅ **Fault tolerance**: Automatic checkpointing and retry logic
# MAGIC - ✅ **Error tracking**: Dead-letter queue for failed documents
# MAGIC - ✅ **Quality monitoring**: Token distribution and element type analytics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# === PRODUCTION CONFIGURATION ===

# Source paths - Configure for your environment
source_path_test = "idbfs:/2026-09-17/03/_f399c322-b14b-4a68-9cce-bbfa9f85187a"

# Production paths (uncomment and configure for your cloud provider)
# source_path = "s3://your-company-bucket/incoming-documents/"  # AWS
# source_path = "abfss://documents@youraccount.dfs.core.windows.net/incoming/"  # Azure
# source_path = "gs://your-company-bucket/documents/"  # GCP
# source_path = "/Volumes/main/raw_data/documents/"  # UC Volume

source_path = source_path_test  # Using test path for now

# Output configuration
catalog = "workspace"
schema = "default"

# Unity Catalog tables
parsed_docs_table = f"{catalog}.{schema}.parsed_documents"
chunks_table = f"{catalog}.{schema}.document_chunks"
error_table = f"{catalog}.{schema}.document_processing_errors"
metrics_table = f"{catalog}.{schema}.chunking_metrics"

# Checkpoint location (use cloud storage in production for durability)
checkpoint_base = f"/tmp/doc_pipeline_{catalog}_{schema}"
# checkpoint_base = "s3://your-bucket/checkpoints/doc_pipeline"  # Production

# Image/signature output volume
image_volume = f"/Volumes/{catalog}/{schema}/document_images"

# Processing mode
STREAMING_MODE = False  # Set True for continuous processing
BATCH_SIZE = 100  # Documents to process per micro-batch

# Chunking parameters (optimized for RAG retrieval)
MIN_TOKENS = 500
MAX_TOKENS = 700
OVERLAP_SENTENCES = 2

print("✅ Configuration loaded")
print(f"   Source: {source_path}")
print(f"   Mode: {'Streaming (continuous)' if STREAMING_MODE else 'Batch (one-time)'}")
print(f"   Output: {chunks_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Create Unity Catalog Assets

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create volume for extracted images/signatures
# MAGIC CREATE VOLUME IF NOT EXISTS workspace.default.document_images
# MAGIC COMMENT 'Stores extracted images and signatures from parsed PDFs for visual validation';
# MAGIC
# MAGIC -- Create error tracking table
# MAGIC CREATE TABLE IF NOT EXISTS workspace.default.document_processing_errors (
# MAGIC   path STRING,
# MAGIC   file_name STRING,
# MAGIC   file_size_bytes BIGINT,
# MAGIC   error_details VARIANT,
# MAGIC   error_message STRING,
# MAGIC   processed_at TIMESTAMP,
# MAGIC   batch_id STRING
# MAGIC ) USING DELTA
# MAGIC TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');
# MAGIC
# MAGIC SELECT 'Setup complete ✓' AS status;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Distributed Document Parsing with Auto Loader
# MAGIC
# MAGIC Auto Loader automatically:
# MAGIC - Discovers new files as they arrive
# MAGIC - Maintains processing state with checkpoints
# MAGIC - Scales to millions of files
# MAGIC - Handles schema evolution

# COMMAND ----------

from pyspark.sql.functions import col, current_timestamp, expr

# Configure source based on mode
if STREAMING_MODE:
    print("🔄 Starting streaming ingestion...")
    documents_df = (spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .option("cloudFiles.schemaLocation", f"{checkpoint_base}/schema")
        .option("cloudFiles.maxFilesPerTrigger", BATCH_SIZE)
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "*.pdf")  # Add |*.docx|*.pptx for other formats
        .load(source_path)
    )
else:
    print("📦 Running batch ingestion...")
    documents_df = (spark.read
        .format("binaryFile")
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "*.pdf")
        .load(source_path)
    )

# Apply ai_parse_document with image extraction and AI descriptions
parsed_df = documents_df.selectExpr(
    "path",
    "_metadata.file_name AS file_name",
    "_metadata.file_size AS file_size_bytes",
    "_metadata.file_modification_time AS file_modified_at",
    "content",
    f"""ai_parse_document(
        content,
        MAP(
            'version', '2.0',
            'descriptionElementTypes', '*',
            'imageOutputPath', '{image_volume}'
        )
    ) AS parsed_content""",
    "current_timestamp() AS processed_at",
    "uuid() AS batch_id"
)

# Add quality checks
quality_checked_df = parsed_df.withColumn(
    "has_error",
    expr("NOT is_variant_null(parsed_content:error_status)")
).withColumn(
    "element_count",
    expr("COALESCE(size(try_cast(parsed_content:document:elements AS ARRAY<VARIANT>)), 0)")
).withColumn(
    "page_count",
    expr("COALESCE(size(try_cast(parsed_content:document:pages AS ARRAY<VARIANT>)), 0)")
)

print(f"✅ Document parsing configured")
print(f"   - Image extraction: Enabled (→ {image_volume})")
print(f"   - Figure descriptions: AI-generated")
print(f"   - Error tracking: Enabled")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Error Handling & Dead Letter Queue
# MAGIC
# MAGIC Separates successful parses from errors for monitoring and retry logic

# COMMAND ----------

# Split success and error streams
success_df = quality_checked_df.filter("NOT has_error AND element_count > 0")
error_df = quality_checked_df.filter("has_error OR element_count = 0")

# Write errors to dead-letter table
if STREAMING_MODE:
    error_writer = (error_df
        .select(
            "path",
            "file_name",
            "file_size_bytes",
            "parsed_content:error_status AS error_details",
            expr("cast(parsed_content:error_status AS STRING) AS error_message"),
            "processed_at",
            "batch_id"
        )
        .writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", f"{checkpoint_base}/errors")
        .trigger(processingTime="30 seconds")
        .toTable(error_table)
    )
    print(f"✅ Error stream configured → {error_table}")
else:
    error_count = error_df.count()
    if error_count > 0:
        (error_df
            .select(
                "path",
                "file_name",
                "file_size_bytes",
                "parsed_content:error_status AS error_details",
                expr("cast(parsed_content:error_status AS STRING) AS error_message"),
                "processed_at",
                "batch_id"
            )
            .write
            .mode("append")
            .saveAsTable(error_table)
        )
        print(f"⚠️  {error_count} documents failed parsing → {error_table}")
    else:
        print(f"✅ All documents parsed successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Save Parsed Documents to Delta
# MAGIC
# MAGIC Preserves raw content for reprocessing and stores parsed structure

# COMMAND ----------

# Write successful parses to Delta with partitioning
if STREAMING_MODE:
    parsed_writer = (success_df
        .select(
            "path",
            "file_name",
            "file_size_bytes",
            "file_modified_at",
            "element_count",
            "page_count",
            "content",  # Keep raw for reprocessing
            "parsed_content",
            "processed_at",
            "batch_id",
            expr("date(processed_at) AS processing_date")
        )
        .writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", f"{checkpoint_base}/parsed")
        .partitionBy("processing_date")
        .trigger(processingTime="30 seconds")
        .toTable(parsed_docs_table)
    )
    print(f"✅ Parsed documents stream → {parsed_docs_table}")
else:
    (success_df
        .withColumn("processing_date", expr("date(processed_at)"))
        .select(
            "path",
            "file_name",
            "file_size_bytes",
            "file_modified_at",
            "element_count",
            "page_count",
            "content",
            "parsed_content",
            "processed_at",
            "batch_id",
            "processing_date"
        )
        .write
        .mode("overwrite")  # Use 'append' for incremental
        .partitionBy("processing_date")
        .saveAsTable(parsed_docs_table)
    )
    doc_count = success_df.count()
    print(f"✅ Saved {doc_count} documents → {parsed_docs_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Distributed Chunking with Pandas UDF
# MAGIC
# MAGIC Uses Spark's pandas UDF to parallelize chunking across workers:
# MAGIC - Each worker processes a batch of documents
# MAGIC - Chunking happens in-memory on worker nodes
# MAGIC - Scales horizontally with cluster size

# COMMAND ----------

import pandas as pd
import json
import re
from typing import Iterator
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType

# Install tiktoken
try:
    import tiktoken
except ImportError:
    %pip install -q tiktoken
    import tiktoken

# Initialize tokenizer (cl100k_base = GPT-4, GPT-3.5-turbo encoding)
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    """Count tokens using tiktoken."""
    if not text:
        return 0
    return len(encoding.encode(str(text)))

def split_into_sentences(text):
    """Split text into sentences for smart overlap."""
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    return [s.strip() for s in sentences if s.strip()]

def split_large_element(content, element_type, max_tokens, overlap_sentences):
    """
    Split elements larger than max_tokens at sentence boundaries.
    Adds overlap between chunks to preserve context.
    """
    sentences = split_into_sentences(content)
    if not sentences:
        return [{'content': content, 'type': element_type, 'tokens': count_tokens(content)}]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        # If adding this sentence exceeds max, save current chunk
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'type': element_type,
                'tokens': count_tokens(chunk_text)
            })
            
            # Keep last N sentences for overlap
            overlap_start = max(0, len(current_chunk) - overlap_sentences)
            current_chunk = current_chunk[overlap_start:]
            current_tokens = count_tokens(' '.join(current_chunk))
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Save remaining sentences
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            'content': chunk_text,
            'type': element_type,
            'tokens': count_tokens(chunk_text)
        })
    
    return chunks

def chunk_elements(elements, min_tokens, max_tokens, overlap_sentences):
    """
    Core chunking algorithm:
    1. Group small elements together
    2. Split large elements at sentence boundaries
    3. Respect natural document structure (titles stay with content)
    4. Maintain target token range
    """
    chunks = []
    current_group = []
    current_tokens = 0
    
    # Element types that should group with following content
    grouping_types = {'title', 'section_header', 'caption', 'page_header'}
    
    for i, element in enumerate(elements):
        element_type = element.get('type', 'text')
        content = element.get('content', '')
        
        if not content:
            continue
        
        element_tokens = count_tokens(content)
        
        # Handle oversized elements
        if element_tokens > max_tokens:
            # Save current group first
            if current_group:
                group_content = '\n\n'.join([e['content'] for e in current_group])
                chunks.append({
                    'content': group_content,
                    'types': [e['type'] for e in current_group],
                    'tokens': count_tokens(group_content)
                })
                current_group = []
                current_tokens = 0
            
            # Split large element
            split_chunks = split_large_element(content, element_type, max_tokens, overlap_sentences)
            for chunk in split_chunks:
                chunks.append({
                    'content': chunk['content'],
                    'types': [chunk['type']],
                    'tokens': chunk['tokens']
                })
            continue
        
        # Try adding to current group
        if current_tokens + element_tokens <= max_tokens:
            current_group.append({'content': content, 'type': element_type})
            current_tokens += element_tokens
        else:
            # Current group is full, save it
            if current_group:
                group_content = '\n\n'.join([e['content'] for e in current_group])
                chunks.append({
                    'content': group_content,
                    'types': [e['type'] for e in current_group],
                    'tokens': count_tokens(group_content)
                })
            
            # Start new group
            current_group = [{'content': content, 'type': element_type}]
            current_tokens = element_tokens
        
        # Force save if we've hit target and not a grouping type
        if (element_type not in grouping_types and 
            current_tokens >= min_tokens and 
            i < len(elements) - 1):
            next_type = elements[i + 1].get('type', '')
            if next_type not in grouping_types:
                group_content = '\n\n'.join([e['content'] for e in current_group])
                chunks.append({
                    'content': group_content,
                    'types': [e['type'] for e in current_group],
                    'tokens': count_tokens(group_content)
                })
                current_group = []
                current_tokens = 0
    
    # Save final group
    if current_group:
        group_content = '\n\n'.join([e['content'] for e in current_group])
        chunks.append({
            'content': group_content,
            'types': [e['type'] for e in current_group],
            'tokens': count_tokens(group_content)
        })
    
    return chunks

# Define output schema for chunks
chunk_schema = StructType([
    StructField("chunk_id", IntegerType(), False),
    StructField("content", StringType(), False),
    StructField("token_count", IntegerType(), False),
    StructField("element_types", ArrayType(StringType()), False)
])

@pandas_udf(ArrayType(chunk_schema))
def chunk_document_udf(parsed_content_series: pd.Series) -> pd.Series:
    """
    Pandas UDF for distributed document chunking.
    
    Processes batches of documents in parallel across Spark workers.
    Each worker handles a subset of documents independently.
    """
    results = []
    
    for parsed_content_str in parsed_content_series:
        try:
            # Parse VARIANT content
            parsed = json.loads(str(parsed_content_str))
            elements = parsed.get('document', {}).get('elements', [])
            
            if not elements:
                results.append([])
                continue
            
            # Apply chunking logic
            chunks = chunk_elements(elements, MIN_TOKENS, MAX_TOKENS, OVERLAP_SENTENCES)
            
            # Format output
            chunk_records = [
                (i + 1, 
                 chunk['content'],
                 chunk['tokens'],
                 chunk['types'])
                for i, chunk in enumerate(chunks)
            ]
            results.append(chunk_records)
            
        except Exception as e:
            # Log but don't fail entire batch
            print(f"Chunking error: {str(e)[:200]}")
            results.append([])
    
    return pd.Series(results)

print("✅ Distributed chunking UDF defined")
print(f"   - Token range: {MIN_TOKENS}-{MAX_TOKENS}")
print(f"   - Sentence overlap: {OVERLAP_SENTENCES}")
print(f"   - Encoding: cl100k_base (GPT-4 compatible)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Apply Chunking & Write to Delta
# MAGIC
# MAGIC Processes all documents and saves chunks with metadata

# COMMAND ----------

from pyspark.sql.functions import explode, col, monotonically_increasing_id, expr

# Read parsed documents
if STREAMING_MODE:
    source_df = spark.readStream.table(parsed_docs_table)
else:
    source_df = spark.read.table(parsed_docs_table)

# Apply chunking UDF (distributed across workers)
print("🔄 Applying distributed chunking...")
chunked_df = source_df.withColumn(
    "chunks",
    chunk_document_udf(col("parsed_content"))
)

# Explode chunks array to individual rows
chunks_final = (chunked_df
    .withColumn("chunk", explode("chunks"))
    .select(
        "path",
        "file_name",
        "batch_id",
        col("chunk.chunk_id"),
        col("chunk.content"),
        col("chunk.token_count"),
        col("chunk.element_types"),
        "file_size_bytes",
        "element_count",
        "page_count",
        "processed_at",
        current_timestamp().alias("chunked_at"),
        expr("date(processed_at) AS processing_date")
    )
)

# Write to Delta with partitioning
if STREAMING_MODE:
    chunks_writer = (chunks_final
        .writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", f"{checkpoint_base}/chunks")
        .partitionBy("processing_date")
        .trigger(processingTime="60 seconds")
        .toTable(chunks_table)
    )
    print(f"✅ Chunk stream started → {chunks_table}")
    print(f"\n🚀 PIPELINE RUNNING")
    print(f"\nMonitor with:")
    print(f"  SELECT COUNT(*) FROM {parsed_docs_table}")
    print(f"  SELECT COUNT(*) FROM {chunks_table}")
    print(f"  SELECT * FROM {error_table}")
else:
    (chunks_final
        .write
        .mode("overwrite")  # Use 'append' for incremental
        .partitionBy("processing_date")
        .option("overwriteSchema", "true")
        .saveAsTable(chunks_table)
    )
    total_chunks = chunks_final.count()
    print(f"✅ Saved {total_chunks:,} chunks → {chunks_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Quality Metrics & Analysis

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Overall chunking statistics
# MAGIC SELECT
# MAGIC   COUNT(*) AS total_chunks,
# MAGIC   COUNT(DISTINCT file_name) AS total_documents,
# MAGIC   ROUND(AVG(token_count), 1) AS avg_tokens,
# MAGIC   MIN(token_count) AS min_tokens,
# MAGIC   MAX(token_count) AS max_tokens,
# MAGIC   PERCENTILE(token_count, 0.5) AS median_tokens,
# MAGIC   ROUND(STDDEV(token_count), 1) AS stddev_tokens,
# MAGIC   SUM(CASE WHEN token_count BETWEEN 500 AND 700 THEN 1 ELSE 0 END) AS chunks_in_target,
# MAGIC   ROUND(100.0 * SUM(CASE WHEN token_count BETWEEN 500 AND 700 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_in_target
# MAGIC FROM workspace.default.document_chunks;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Element type distribution (shows images, tables, text, etc.)
# MAGIC SELECT
# MAGIC   explode(element_types) AS element_type,
# MAGIC   COUNT(*) AS chunk_count,
# MAGIC   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) AS percentage,
# MAGIC   ROUND(AVG(token_count), 1) AS avg_tokens
# MAGIC FROM workspace.default.document_chunks
# MAGIC GROUP BY element_type
# MAGIC ORDER BY chunk_count DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Per-document breakdown
# MAGIC SELECT
# MAGIC   file_name,
# MAGIC   COUNT(*) AS num_chunks,
# MAGIC   MAX(page_count) AS pages,
# MAGIC   MAX(element_count) AS elements,
# MAGIC   ROUND(AVG(token_count), 1) AS avg_chunk_tokens,
# MAGIC   SUM(token_count) AS total_tokens
# MAGIC FROM workspace.default.document_chunks
# MAGIC GROUP BY file_name
# MAGIC ORDER BY num_chunks DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Sample Chunks

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Show sample chunks with content preview
# MAGIC SELECT
# MAGIC   file_name,
# MAGIC   chunk_id,
# MAGIC   element_types,
# MAGIC   token_count,
# MAGIC   SUBSTRING(content, 1, 300) || '\n...' AS content_preview
# MAGIC FROM workspace.default.document_chunks
# MAGIC ORDER BY file_name, chunk_id
# MAGIC LIMIT 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Check Images & Signatures
# MAGIC
# MAGIC Verify extracted images and AI-generated descriptions

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Find chunks containing images/figures (including signatures)
# MAGIC SELECT
# MAGIC   file_name,
# MAGIC   chunk_id,
# MAGIC   element_types,
# MAGIC   token_count,
# MAGIC   content
# MAGIC FROM workspace.default.document_chunks
# MAGIC WHERE array_contains(element_types, 'figure')
# MAGIC ORDER BY file_name, chunk_id
# MAGIC LIMIT 10;

# COMMAND ----------

# List extracted image files in the volume
dbutils.fs.ls(image_volume)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production Deployment Checklist
# MAGIC
# MAGIC ### Before Going to Production:
# MAGIC
# MAGIC 1. **Configure Cloud Storage**
# MAGIC    - Update `source_path` to your S3/ADLS/GCS bucket
# MAGIC    - Set `checkpoint_base` to external cloud storage (not /tmp)
# MAGIC    - Configure Unity Catalog volumes for image storage
# MAGIC
# MAGIC 2. **Enable Streaming Mode**
# MAGIC    - Set `STREAMING_MODE = True`
# MAGIC    - Configure trigger intervals based on data arrival rate
# MAGIC    - Set up monitoring alerts
# MAGIC
# MAGIC 3. **Cluster Configuration**
# MAGIC    - Use autoscaling cluster (min 2, max 10+ workers for millions of docs)
# MAGIC    - Enable Dynamic Allocation
# MAGIC    - Consider spot/preemptible instances for cost savings
# MAGIC
# MAGIC 4. **Performance Tuning**
# MAGIC    - Adjust `BATCH_SIZE` based on file sizes
# MAGIC    - Tune `maxFilesPerTrigger` for Auto Loader
# MAGIC    - Monitor partition sizes and adjust partitioning strategy
# MAGIC
# MAGIC 5. **Monitoring & Alerts**
# MAGIC    - Set up dashboards for:
# MAGIC      - Documents processed per hour
# MAGIC      - Error rates
# MAGIC      - Processing latency
# MAGIC      - Token distribution quality
# MAGIC    - Configure alerts for:
# MAGIC      - High error rates
# MAGIC      - Processing delays
# MAGIC      - Storage growth
# MAGIC
# MAGIC 6. **Data Quality**
# MAGIC    - Enable Delta table properties:
# MAGIC      - `delta.enableChangeDataFeed = true` (for downstream CDC)
# MAGIC      - `delta.autoOptimize.optimizeWrite = true` (for small file compaction)
# MAGIC      - `delta.autoOptimize.autoCompact = true`
# MAGIC    - Set up data expectations/tests
# MAGIC
# MAGIC 7. **Cost Optimization**
# MAGIC    - Use spot instances for workers
# MAGIC    - Enable Auto Loader's schema inference caching
# MAGIC    - Implement lifecycle policies for old partitions
# MAGIC    - Consider cold storage for processed raw files
# MAGIC
# MAGIC 8. **Security**
# MAGIC    - Configure IAM roles for cloud storage access
# MAGIC    - Enable Unity Catalog governance
# MAGIC    - Set up row/column-level security if needed
# MAGIC    - Audit access patterns

# COMMAND ----------

