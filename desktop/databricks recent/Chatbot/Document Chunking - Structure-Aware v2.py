# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Structure-Aware Document Chunking
# MAGIC
# MAGIC This notebook demonstrates an intelligent approach to document chunking for Retrieval-Augmented Generation (RAG) applications.
# MAGIC
# MAGIC ## Chunking Philosophy
# MAGIC
# MAGIC **Structure-First Parsing**: We parse the document structure first using `ai_parse_document` v2.0, which identifies semantic elements like paragraphs, titles, tables, and figures.
# MAGIC
# MAGIC **Preserve Paragraph Boundaries**: Rather than splitting text arbitrarily, we respect natural boundaries (paragraphs, sections) to maintain semantic coherence.
# MAGIC
# MAGIC **Token Target for Retrieval Quality**: We target 500-700 tokens per chunk—large enough to provide context, small enough for precise retrieval and to fit within embedding model limits.
# MAGIC
# MAGIC **Smart Splitting**: When individual elements exceed our target, we split at sentence boundaries with 2-sentence overlap to preserve context across chunks.
# MAGIC
# MAGIC **Group Related Elements**: We combine small consecutive elements (titles with their paragraphs, captions with tables) to create meaningful, context-rich chunks.

# COMMAND ----------

# MAGIC %sql
# MAGIC select _metadata from FROM READ_FILES(
# MAGIC   'idbfs:/2026-09-17/03/_f399c322-b14b-4a68-9cce-bbfa9f85187a',
# MAGIC   format => 'binaryFile'
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Parse the PDF using ai_parse_document v2.0
# MAGIC CREATE OR REPLACE TEMPORARY VIEW parsed_document AS
# MAGIC SELECT
# MAGIC   _metadata.file_name AS file_name,
# MAGIC   ai_parse_document(content, MAP('version', '2.0')) AS parsed
# MAGIC FROM READ_FILES(
# MAGIC   'idbfs:/2026-09-17/03/_f399c322-b14b-4a68-9cce-bbfa9f85187a',
# MAGIC   format => 'binaryFile'
# MAGIC );
# MAGIC
# MAGIC SELECT * FROM parsed_document;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Analyze element types and counts from the parsed document
# MAGIC SELECT
# MAGIC   element:type::STRING AS element_type,
# MAGIC   COUNT(*) AS count
# MAGIC FROM (
# MAGIC   SELECT explode(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) AS element
# MAGIC   FROM parsed_document
# MAGIC )
# MAGIC GROUP BY element:type::STRING
# MAGIC ORDER BY count DESC;

# COMMAND ----------

# Extract elements from the parsed document for Python processing
import json

# Get the parsed document
parsed_df = spark.table("parsed_document")
parsed_row = parsed_df.collect()[0]

# Extract elements - convert VARIANT to string first
parsed_variant = parsed_row['parsed']
elements_json = json.loads(str(parsed_variant))

# Get elements array
if 'document' in elements_json and 'elements' in elements_json['document']:
    elements = elements_json['document']['elements']
    print(f"Total elements extracted: {len(elements)}")
    print(f"\nSample elements (first 3):")
    for i, elem in enumerate(elements[:3]):
        print(f"\nElement {i+1}:")
        print(f"  Type: {elem.get('type', 'unknown')}")
        print(f"  description: {elem.get('description', '')}")
        content_preview = str(elem.get('content', ''))[:100]
        print(f"  Content: {content_preview}...")
else:
    print("Error: Could not extract elements from parsed document")
    elements = []

# COMMAND ----------

import re
try:
    import tiktoken
except ImportError:
    %pip install tiktoken
    import tiktoken

# Initialize tokenizer
encoding = tiktoken.get_encoding("cl100k_base")

# Chunking parameters
MIN_TOKENS = 500
MAX_TOKENS = 700
OVERLAP_SENTENCES = 2

def count_tokens(text):
    """Count tokens in text using tiktoken."""
    if not text:
        return 0
    return len(encoding.encode(str(text)))

def split_into_sentences(text):
    """Split text into sentences."""
    # Simple sentence splitter - handles common cases
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    return [s.strip() for s in sentences if s.strip()]

def split_large_element(content, element_type):
    """Split a large element at sentence boundaries with overlap."""
    sentences = split_into_sentences(content)
    if not sentences:
        return [content]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for i, sentence in enumerate(sentences):
        sentence_tokens = count_tokens(sentence)
        
        # If adding this sentence exceeds max, save current chunk
        if current_tokens + sentence_tokens > MAX_TOKENS and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'type': element_type,
                'tokens': count_tokens(chunk_text)
            })
            
            # Start new chunk with overlap (last N sentences)
            overlap_start = max(0, len(current_chunk) - OVERLAP_SENTENCES)
            current_chunk = current_chunk[overlap_start:]
            current_tokens = count_tokens(' '.join(current_chunk))
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Add remaining sentences
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            'content': chunk_text,
            'type': element_type,
            'tokens': count_tokens(chunk_text)
        })
    
    return chunks

def chunk_elements(elements):
    """Chunk elements while preserving structure and boundaries."""
    chunks = []
    current_group = []
    current_tokens = 0
    
    # Element types that should be grouped with following content
    grouping_types = {'title', 'section_header', 'caption'}
    
    for i, element in enumerate(elements):
        element_type = element.get('type', 'text')
        content = element.get('content', '')
        
        if not content:
            continue
        
        element_tokens = count_tokens(content)
        
        # If single element is too large, split it
        if element_tokens > MAX_TOKENS:
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
            split_chunks = split_large_element(content, element_type)
            chunks.extend(split_chunks)
            continue
        
        # Try to add to current group
        if current_tokens + element_tokens <= MAX_TOKENS:
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
        
        # Force grouping for headers/titles with next element
        if element_type not in grouping_types and current_tokens >= MIN_TOKENS:
            # We have enough tokens and this isn't a grouping type, save the group
            if i < len(elements) - 1:  # Not the last element
                next_type = elements[i + 1].get('type', '')
                if next_type not in grouping_types:  # Next isn't a grouping type either
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

print("Chunking algorithm loaded successfully!")

# COMMAND ----------

# Apply chunking to the extracted elements
chunks = chunk_elements(elements)

# Calculate statistics
token_counts = [chunk.get('tokens', 0) for chunk in chunks]

print(f"\n{'='*60}")
print(f"CHUNKING STATISTICS")
print(f"{'='*60}")
print(f"Total chunks created: {len(chunks)}")
print(f"\nToken distribution:")
print(f"  Min tokens: {min(token_counts) if token_counts else 0}")
print(f"  Max tokens: {max(token_counts) if token_counts else 0}")
print(f"  Mean tokens: {sum(token_counts) / len(token_counts) if token_counts else 0:.1f}")
print(f"  Target range: {MIN_TOKENS}-{MAX_TOKENS} tokens")

# Count chunks in target range
in_range = sum(1 for t in token_counts if MIN_TOKENS <= t <= MAX_TOKENS)
print(f"\nChunks in target range: {in_range} ({in_range/len(chunks)*100:.1f}%)")

# Show token distribution
import statistics
if token_counts:
    print(f"\nDetailed statistics:")
    print(f"  Median: {statistics.median(token_counts):.1f}")
    print(f"  Std Dev: {statistics.stdev(token_counts):.1f}" if len(token_counts) > 1 else "  Std Dev: N/A")

# COMMAND ----------

# Display first 2 chunks
print(f"\n{'='*60}")
print(f"CHUNK PREVIEW")
print(f"{'='*60}\n")

for i, chunk in enumerate(chunks[:2]):
    print(f"\n{'─'*60}")
    print(f"CHUNK {i+1}")
    print(f"{'─'*60}")
    print(f"Types: {', '.join(chunk.get('types', [chunk.get('type', 'unknown')]))}")
    print(f"Tokens: {chunk.get('tokens', 0)}")
    print(f"\nContent:")
    print(chunk.get('content', '')[:500])  # Show first 500 chars
    if len(chunk.get('content', '')) > 500:
        print("\n[... content truncated ...]")

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType
from datetime import datetime

# Prepare data for Delta table
chunk_data = []
for i, chunk in enumerate(chunks):
    chunk_data.append((
        i + 1,  # chunk_id
        chunk.get('content', ''),
        chunk.get('tokens', 0),
        chunk.get('types', [chunk.get('type', 'unknown')]),
        parsed_row['file_name'],
        datetime.now()
    ))

# Define schema
schema = StructType([
    StructField("chunk_id", IntegerType(), False),
    StructField("content", StringType(), False),
    StructField("token_count", IntegerType(), False),
    StructField("element_types", ArrayType(StringType()), False),
    StructField("source_file", StringType(), False),
    StructField("created_at", StringType(), False)
])

# Convert datetime to string for schema compatibility
chunk_data_str = [
    (cid, content, tokens, types, source, created.isoformat())
    for cid, content, tokens, types, source, created in chunk_data
]

# Create DataFrame
chunks_df = spark.createDataFrame(chunk_data_str, schema)

# Save to Delta table
table_name = "document_chunks"
chunks_df.write.mode("overwrite").saveAsTable(table_name)

print(f"\n✓ Successfully saved {len(chunks)} chunks to table: {table_name}")
print(f"\nTable schema:")
chunks_df.printSchema()

# Show sample
print(f"\nSample rows:")
display(spark.table(table_name).limit(3))