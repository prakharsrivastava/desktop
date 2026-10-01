# Databricks notebook source
# DBTITLE 1,Section-Aware RAG Chunking Pipeline
# MAGIC %md
# MAGIC # Section-Aware RAG Chunking & Vector Search Pipeline
# MAGIC
# MAGIC This notebook parses documents (PDF + CSV), creates structure-aware chunks with rich metadata,
# MAGIC supports incremental updates via content hashing, and stores everything in a Databricks Vector
# MAGIC Search index for fast RAG retrieval.
# MAGIC
# MAGIC **Key features:**
# MAGIC - PDF parsing via `ai_parse_document` (section headers, text, tables as separate chunks)
# MAGIC - CSV multi-page table chunking (batches of 50 rows with repeated headers)
# MAGIC - Metadata: `chunk_id`, `doc_id`, `page`, `section_id`, `chunk_text`, `chunk_type`, `content_hash`, `file_path`, `file_version`, `acl`
# MAGIC - Incremental MERGE: only changed chunks are upserted (content hash comparison)
# MAGIC - Delta Sync Vector Search index with managed embeddings

# COMMAND ----------

# DBTITLE 1,Configuration & Setup
# ── Configuration ─────────────────────────────────────────────────────
# Unity Catalog target for chunks table + vector search index
CATALOG = "workspace"
SCHEMA = "default"

# Source files (uploaded to IDBFS)
PDF_IDBFS_URL = "idbfs:/2026-09-18/05/_a47575ee-8837-4ee1-9790-0105e9ef1b72"
CSV_IDBFS_URL = "idbfs:/2026-09-18/05/_93669c32-bd7f-4d65-a127-8affc4102ee9"
PDF_FILE_NAME = "dummy_structure_aware_chunking.pdf"
CSV_FILE_NAME = "dummy_multipage_table.csv"

# Chunking parameters
TABLE_CHUNK_SIZE = 50  # rows per table chunk (CSV)

# Vector Search
VS_ENDPOINT_NAME = "rag-chunking-endpoint"
EMBEDDING_MODEL = "databricks-gte-large-en"

# Derived names
FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"
CHUNKS_TABLE = f"{FULL_SCHEMA}.rag_chunks"
VS_INDEX_NAME = f"{FULL_SCHEMA}.rag_chunks_index"

print(f"Chunks table  : {CHUNKS_TABLE}")
print(f"VS endpoint  : {VS_ENDPOINT_NAME}")
print(f"VS index     : {VS_INDEX_NAME}")
print(f"Embedding    : {EMBEDDING_MODEL}")

# COMMAND ----------

# DBTITLE 1,Parse PDF with ai_parse_document
# ── Parse PDF into structured elements via ai_parse_document ─────────
# Returns a DataFrame of elements: type (section_header, text, table, figure, ...),
# page number, content text, and section path.

pdf_parsed_sql = f"""
WITH raw_pdf AS (
  SELECT
    _metadata.file_name AS file_name,
    ai_parse_document(content, MAP('version', '2.0')) AS parsed
  FROM READ_FILES('{PDF_IDBFS_URL}', format => 'binaryFile')
),
valid_pdf AS (
  SELECT * FROM raw_pdf
  WHERE is_variant_null(parsed:error_status)
),
element_rows AS (
  SELECT
    file_name,
    idx AS element_idx,
    try_cast(elem:type AS STRING) AS element_type,
    try_cast(elem:content AS STRING) AS element_text,
    try_cast(elem:bbox[0]:page_id AS INT) + 1 AS page_number
  FROM valid_pdf,
    LATERAL posexplode(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) AS (idx, elem)
)
SELECT * FROM element_rows ORDER BY file_name, element_idx
"""

pdf_elements_df = spark.sql(pdf_parsed_sql)
print(f"PDF elements parsed: {pdf_elements_df.count()}")
display(pdf_elements_df.limit(20))

# COMMAND ----------

# DBTITLE 1,Build structure-aware PDF chunks
# ── Build structure-aware chunks from PDF elements ──────────────────────
# Each element becomes a chunk. We track section hierarchy by maintaining
# a running section_id as we encounter section_header elements.

import hashlib
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType

# Convert to Pandas for sequential section tracking (PDF is small)
pdf_pandas = pdf_elements_df.toPandas()

section_counter = 0
current_section_id = None
chunk_rows = []

for _, row in pdf_pandas.iterrows():
    el_type = row["element_type"] or "text"
    el_text = row["element_text"] or ""
    page = row["page_number"] or 1

    if el_type == "section_header":
        section_counter += 1
        current_section_id = f"sec_{section_counter:03d}"

    # Build chunk text – include section context for non-header chunks
    if el_type == "section_header":
        chunk_text = f"[Section: {el_text}]"
    elif el_type == "table":
        chunk_text = f"[Table on page {page}]\n{el_text}"
    else:
        section_prefix = f"[Section: {current_section_id}] " if current_section_id else ""
        chunk_text = f"{section_prefix}{el_text}"

    chunk_id = hashlib.sha256(
        f"{row['file_name']}|{row['element_idx']}|{chunk_text}".encode()
    ).hexdigest()[:16]

    content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()

    chunk_rows.append({
        "chunk_id": chunk_id,
        "doc_id": row["file_name"],
        "page": int(page),
        "section_id": current_section_id or "sec_root",
        "chunk_text": chunk_text,
        "chunk_type": el_type,
        "content_hash": content_hash,
        "file_path": PDF_IDBFS_URL,
        "file_name": row["file_name"],
        "file_version": 1,
        "acl": "public",
        "source_type": "pdf",
    })

pdf_chunks_df = spark.createDataFrame(chunk_rows)
print(f"PDF chunks created: {pdf_chunks_df.count()}")
display(pdf_chunks_df.limit(10))

# COMMAND ----------

# DBTITLE 1,Parse & chunk CSV (table-aware)
# ── Parse CSV and create table-aware chunks ────────────────────────────
# Batch rows into groups of TABLE_CHUNK_SIZE with the header repeated
# in each chunk so every chunk is self-contained.

import hashlib

# Read CSV via spark
csv_df = spark.read.option("header", "true").csv(CSV_IDBFS_URL)
csv_cols = csv_df.columns
header_line = ",".join(csv_cols)
total_rows = csv_df.count()
print(f"CSV columns: {csv_cols}")
print(f"Total data rows: {total_rows}")

# Collect to Pandas for batching (small file)
csv_pandas = csv_df.toPandas()

csv_chunk_rows = []
num_chunks = (len(csv_pandas) + TABLE_CHUNK_SIZE - 1) // TABLE_CHUNK_SIZE

for batch_idx in range(num_chunks):
    start = batch_idx * TABLE_CHUNK_SIZE
    end = min(start + TABLE_CHUNK_SIZE, len(csv_pandas))
    batch = csv_pandas.iloc[start:end]

    # Build chunk text: header + rows
    lines = [header_line]
    for _, r in batch.iterrows():
        lines.append(",".join(str(v) for v in r.values))
    chunk_text = f"[Table Header\n{header_line}\n\nRows {start+1}-{end}\n+ repeat header]\n\n" + "\n".join(lines)

    chunk_id = hashlib.sha256(
        f"{CSV_FILE_NAME}|batch_{batch_idx}".encode()
    ).hexdigest()[:16]
    content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()

    # Page = batch index + 1 (each batch ~ one "page" of the table)
    csv_chunk_rows.append({
        "chunk_id": chunk_id,
        "doc_id": CSV_FILE_NAME,
        "page": batch_idx + 1,
        "section_id": f"table_batch_{batch_idx + 1:03d}",
        "chunk_text": chunk_text,
        "chunk_type": "table",
        "content_hash": content_hash,
        "file_path": CSV_IDBFS_URL,
        "file_name": CSV_FILE_NAME,
        "file_version": 1,
        "acl": "public",
        "source_type": "csv",
    })

    print(f"  Chunk {batch_idx+1}/{num_chunks}: rows {start+1}-{end} ({end-start} rows)")

csv_chunks_df = spark.createDataFrame(csv_chunk_rows)
print(f"\nCSV table chunks created: {csv_chunks_df.count()}")
display(csv_chunks_df.limit(5))

# COMMAND ----------

# DBTITLE 1,Create chunks Delta table
# ── Create / ensure chunks Delta table with CDF + primary key ──────────
# This table is the source for the Vector Search Delta Sync index.

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CHUNKS_TABLE} (
    chunk_id      STRING NOT NULL,
    doc_id        STRING NOT NULL,
    page          INT,
    section_id    STRING,
    chunk_text    STRING,
    chunk_type    STRING,
    content_hash  STRING,
    file_path    STRING,
    file_name     STRING,
    file_version  INT,
    acl           STRING,
    source_type   STRING,
    updated_at    TIMESTAMP,
    CONSTRAINT pk_rag_chunks PRIMARY KEY (chunk_id)
) TBLPROPERTIES (
    delta.enableChangeDataFeed = true
)
""")

print(f"Table ready: {CHUNKS_TABLE}")

# COMMAND ----------

# DBTITLE 1,Incremental MERGE – only changed chunks
# ── Incremental MERGE: upsert only changed chunks ────────────────────
# 1. Union PDF + CSV chunks into a staging temp view
# 2. MERGE on chunk_id; only UPDATE rows whose content_hash changed
# 3. DELETE chunks that no longer exist in the new batch
# This ensures the Vector Search index only re-embeds changed chunks.

# Union all new chunks
all_new_chunks = pdf_chunks_df.unionByName(csv_chunks_df)
all_new_chunks.createOrReplaceTempView("new_chunks")

print(f"Total new chunks to merge: {all_new_chunks.count()}")

merge_sql = f"""
MERGE INTO {CHUNKS_TABLE} AS target
USING new_chunks AS source
ON target.chunk_id = source.chunk_id

WHEN MATCHED AND target.content_hash <> source.content_hash THEN
  UPDATE SET
    doc_id       = source.doc_id,
    page         = source.page,
    section_id   = source.section_id,
    chunk_text   = source.chunk_text,
    chunk_type   = source.chunk_type,
    content_hash = source.content_hash,
    file_path    = source.file_path,
    file_name    = source.file_name,
    file_version = source.file_version + 1,
    acl          = source.acl,
    source_type  = source.source_type,
    updated_at   = current_timestamp()

WHEN NOT MATCHED THEN
  INSERT (
    chunk_id, doc_id, page, section_id, chunk_text, chunk_type,
    content_hash, file_path, file_name, file_version, acl,
    source_type, updated_at
  )
  VALUES (
    source.chunk_id, source.doc_id, source.page, source.section_id,
    source.chunk_text, source.chunk_type, source.content_hash,
    source.file_path, source.file_name, source.file_version,
    source.acl, source.source_type, current_timestamp()
  )
"""

spark.sql(merge_sql)

# Delete chunks that are no longer in the source (removed documents/sections)
delete_sql = f"""
DELETE FROM {CHUNKS_TABLE}
WHERE chunk_id NOT IN (SELECT chunk_id FROM new_chunks)
"""
spark.sql(delete_sql)

# Show summary
result = spark.sql(f"SELECT chunk_type, COUNT(*) AS cnt FROM {CHUNKS_TABLE} GROUP BY chunk_type")
print("Chunks in table after MERGE:")
display(result)

# Show a few sample chunks
print("\nSample chunks:")
display(spark.sql(f"SELECT chunk_id, doc_id, page, section_id, chunk_type, LEFT(chunk_text, 120) AS preview, content_hash FROM {CHUNKS_TABLE} ORDER BY doc_id, page LIMIT 15"))

# COMMAND ----------

# DBTITLE 1,Create Vector Search endpoint & index
# ── Create Vector Search endpoint and Delta Sync index ────────────────
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointType
import time

w = WorkspaceClient()

# 1. Create endpoint (idempotent – skip if exists)
endpoints = [e.name for e in w.vector_search_endpoints.list_endpoints()]
if VS_ENDPOINT_NAME not in endpoints:
    print(f"Creating endpoint: {VS_ENDPOINT_NAME}")
    w.vector_search_endpoints.create_endpoint(
        name=VS_ENDPOINT_NAME,
        endpoint_type=EndpointType.STANDARD,
    )
else:
    print(f"Endpoint already exists: {VS_ENDPOINT_NAME}")

# Wait for endpoint to be ready
for _ in range(60):
    status = w.vector_search_endpoints.get_endpoint(endpoint_name=VS_ENDPOINT_NAME)
    state = status.endpoint_status.state
    print(f"  Endpoint state: {state}")
    if state == "ONLINE":
        break
    time.sleep(5)

print(f"Endpoint ready: {state}")

# COMMAND ----------

# DBTITLE 1,Create Delta Sync index
# 2. Create Delta Sync index with managed embeddings on chunk_text ──
# The index auto-syncs from the Delta table; embeddings are computed by Databricks.

from databricks.sdk.service.vectorsearch import (
    VectorIndexType, DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn, PipelineType
)

indexes = [i.name for i in w.vector_search_indexes.list_indexes(endpoint_name=VS_ENDPOINT_NAME)]
if VS_INDEX_NAME not in indexes:
    print(f"Creating index: {VS_INDEX_NAME}")
    w.vector_search_indexes.create_index(
        name=VS_INDEX_NAME,
        endpoint_name=VS_ENDPOINT_NAME,
        primary_key="chunk_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=CHUNKS_TABLE,
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    name="chunk_text",
                    embedding_model_endpoint_name=EMBEDDING_MODEL,
                )
            ],
            pipeline_type=PipelineType.TRIGGERED,
            columns_to_sync=[
                "chunk_id", "doc_id", "page", "section_id",
                "chunk_text", "chunk_type", "content_hash",
                "file_path", "file_name", "file_version", "acl", "source_type",
            ],
        ),
    )
    print("Index creation initiated.")
else:
    print(f"Index already exists: {VS_INDEX_NAME}")

# Wait for index to be ready before triggering sync (may take 10+ minutes)
print("Waiting for index to be ready...")
ready = False
for wait_attempt in range(120):
    idx_status = w.vector_search_indexes.get_index(index_name=VS_INDEX_NAME)
    ready = idx_status.status.ready if idx_status.status else False
    msg = idx_status.status.message if idx_status.status else "no status"
    if wait_attempt % 10 == 0 or ready:
        print(f"  Attempt {wait_attempt+1}: ready={ready}, msg={msg}")
    if ready:
        break
    time.sleep(10)

# Trigger initial sync (retry until index accepts the sync request)
if ready:
    print("Triggering sync...")
    for sync_attempt in range(30):
        try:
            w.vector_search_indexes.sync_index(index_name=VS_INDEX_NAME)
            print("Sync triggered.")
            break
        except Exception as sync_err:
            print(f"  Sync attempt {sync_attempt+1} failed: {sync_err}")
            time.sleep(10)
else:
    print("Index not ready yet - sync will need to be triggered later.")

print(f"Index ready: {ready}")

# COMMAND ----------

# DBTITLE 1,Test retrieval query
# ── Test: query the Vector Search index ───────────────────────────────
# Semantic search over all chunks (PDF sections + CSV table batches)

test_queries = [
    "What sections are in this document?",
    "Show me customer order data",
    "Analytics Pro product orders",
]

for query in test_queries:
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}")
    results = w.vector_search_indexes.query_index(
        index_name=VS_INDEX_NAME,
        columns=["chunk_id", "doc_id", "page", "section_id", "chunk_type", "chunk_text"],
        query_text=query,
        num_results=3,
    )
    if results.result and results.result.data_array:
        for row in results.result.data_array:
            # Last element is the score
            score = row[-1]
            doc_id = row[1] if len(row) > 1 else "?"
            chunk_type = row[4] if len(row) > 4 else "?"
            preview = row[5][:150] if len(row) > 5 and row[5] else ""
            print(f"  Score: {score:.4f} | Doc: {doc_id} | Type: {chunk_type}")
            print(f"  Preview: {preview}...")
    else:
        print("  No results (index may still be syncing).")

print("\n\n✅ Pipeline complete. Chunks stored in Delta table + Vector Search index ready for RAG.")