# Databricks notebook source
# FIXED RAG PIPELINE — Product-level chunking with natural language conversion

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Raw Data

# COMMAND ----------

raw_document = """{
  "products": [
    {
      "id": "ELEC-001",
      "name": "Lithium-Ion Battery L100",
      "category": "Energy Storage",
      "quantitative_properties": {
        "capacity": {
          "min": 2900,
          "max": 3100,
          "unit": "mAh"
        },
        "voltage": {
          "exact": 3.7,
          "unit": "V"
        },
        "charge_cycles": {
          "min": 500,
          "max": 800,
          "unit": "cycles"
        },
        "energy_density": {
          "min": 180,
          "max": 220,
          "unit": "Wh/kg"
        }
      },
      "qualitative_properties": {
        "form_factor": "Cylindrical 18650",
        "safety_features": "Overcharge and thermal protection",
        "application": "Consumer electronics, EV packs",
        "chemistry": "Li-ion NMC"
      }
    },
    {
      "id": "ELEC-002",
      "name": "Microcontroller M200",
      "category": "Semiconductors",
      "quantitative_properties": {
        "clock_speed": {
          "min": 80,
          "max": 120,
          "unit": "MHz"
        },
        "flash_memory": {
          "min": 256,
          "max": 512,
          "unit": "KB"
        },
        "ram": {
          "min": 64,
          "max": 128,
          "unit": "KB"
        },
        "operating_voltage": {
          "min": 1.8,
          "max": 3.6,
          "unit": "V"
        }
      },
      "qualitative_properties": {
        "architecture": "ARM Cortex-M4",
        "power_consumption": "Low power",
        "interfaces": "UART, SPI, I2C",
        "application": "IoT devices, embedded systems"
      }
    },
    {
      "id": "ELEC-003",
      "name": "LED Display Panel D300",
      "category": "Displays",
      "quantitative_properties": {
        "screen_size": {
          "exact": 15.6,
          "unit": "inch"
        },
        "resolution_width": {
          "exact": 1920,
          "unit": "pixels"
        },
        "resolution_height": {
          "exact": 1080,
          "unit": "pixels"
        },
        "brightness": {
          "min": 250,
          "max": 300,
          "unit": "nits"
        },
        "refresh_rate": {
          "exact": 60,
          "unit": "Hz"
        }
      },
      "qualitative_properties": {
        "panel_type": "IPS",
        "color_quality": "High color accuracy",
        "viewing_angle": "Wide",
        "application": "Laptops, monitors"
      }
    },
    {
      "id": "ELEC-004",
      "name": "Power Supply Unit P400",
      "category": "Power Electronics",
      "quantitative_properties": {
        "output_power": {
          "exact": 500,
          "unit": "W"
        },
        "efficiency": {
          "min": 85,
          "max": 90,
          "unit": "%"
        },
        "input_voltage": {
          "min": 100,
          "max": 240,
          "unit": "V"
        },
        "output_voltage": {
          "min": 12,
          "max": 12,
          "unit": "V"
        }
      },
      "qualitative_properties": {
        "cooling": "Active fan cooling",
        "certification": "80 Plus Bronze",
        "protection": "Overvoltage, short circuit",
        "application": "Desktop computers"
      }
    },
    {
      "id": "ELEC-005",
      "name": "Temperature Sensor T500",
      "category": "Sensors",
      "quantitative_properties": {
        "temperature_range": {
          "min": -40,
          "max": 125,
          "unit": "°C"
        },
        "accuracy": {
          "min": 0.1,
          "max": 0.5,
          "unit": "°C"
        },
        "response_time": {
          "exact": 2,
          "unit": "seconds"
        },
        "supply_voltage": {
          "min": 2.7,
          "max": 5.5,
          "unit": "V"
        }
      },
      "qualitative_properties": {
        "sensor_type": "Digital",
        "interface": "I2C",
        "stability": "High long-term stability",
        "application": "Industrial and consumer electronics"
      }
    }
  ]
}"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: FIX — Convert Each Product to Natural Language (1 chunk per product)
# MAGIC
# MAGIC **OLD (WRONG):** Split raw JSON string by character count → broken fragments
# MAGIC
# MAGIC **NEW (CORRECT):** Parse JSON → convert each product to readable text → 1 chunk per product

# COMMAND ----------

import json
from datetime import datetime
import uuid

# ── FIX 1: Parse JSON properly ──────────────────────────────────────────────
data = json.loads(raw_document)

def product_to_natural_language(product: dict) -> str:
    """
    Convert a structured product JSON dict into a clean natural language
    description. This is what gets embedded — NOT raw JSON.

    WHY: Embedding models are trained on natural language.
    Feeding raw JSON gives poor semantic representations because
    keys like '"min":', '"unit":' add noise and break meaning.
    """
    lines = []
    lines.append(f"Product ID: {product['id']}")
    lines.append(f"Product Name: {product['name']}")
    lines.append(f"Category: {product['category']}")

    # Quantitative properties — convert min/max/exact to readable sentences
    quant = product.get("quantitative_properties", {})
    if quant:
        lines.append("Technical Specifications:")
        for prop_name, prop_val in quant.items():
            label = prop_name.replace("_", " ").title()
            unit  = prop_val.get("unit", "")
            if "exact" in prop_val:
                lines.append(f"  - {label}: {prop_val['exact']} {unit}")
            elif "min" in prop_val and "max" in prop_val:
                if prop_val["min"] == prop_val["max"]:
                    lines.append(f"  - {label}: {prop_val['min']} {unit}")
                else:
                    lines.append(
                        f"  - {label}: {prop_val['min']} to {prop_val['max']} {unit}"
                    )

    # Qualitative properties — plain key: value
    qual = product.get("qualitative_properties", {})
    if qual:
        lines.append("Features and Properties:")
        for prop_name, prop_val in qual.items():
            label = prop_name.replace("_", " ").title()
            lines.append(f"  - {label}: {prop_val}")

    return "\n".join(lines)


def extract_numeric_metadata(product: dict) -> dict:
    """
    Pull numeric values out of quantitative_properties and store them
    as flat metadata columns. These are used for HARD FILTERING
    (e.g. cycles >= 500, price <= 200) — separate from semantic search.

    WHY: You cannot do range filtering inside a vector similarity search.
    Numeric constraints must be applied as metadata filters BEFORE
    or AFTER the vector search step.
    """
    metadata = {}
    quant = product.get("quantitative_properties", {})
    for prop_name, prop_val in quant.items():
        unit = prop_val.get("unit", "")
        if "exact" in prop_val:
            metadata[f"{prop_name}_exact"] = float(prop_val["exact"])
            metadata[f"{prop_name}_unit"]  = unit
        elif "min" in prop_val and "max" in prop_val:
            metadata[f"{prop_name}_min"] = float(prop_val["min"])
            metadata[f"{prop_name}_max"] = float(prop_val["max"])
            metadata[f"{prop_name}_unit"] = unit
    return metadata


# ── FIX 2: Build one record per product (not per character chunk) ─────────────
chunk_records = []

for idx, product in enumerate(data["products"]):

    # Convert to clean readable text
    natural_text = product_to_natural_language(product)

    # Extract numeric fields for metadata filtering
    numeric_meta = extract_numeric_metadata(product)

    # Qualitative fields for metadata filtering
    qual_meta = {
        k: v for k, v in product.get("qualitative_properties", {}).items()
    }

    record = {
        # Core identification
        "document_id"  : "DOC_PRODUCT_001",
        "chunk_id"     : str(uuid.uuid4()),
        "product_id"   : product["id"],          # NEW: traceable to source product
        "chunk_index"  : idx,

        # The actual text that will be embedded — clean natural language
        "chunk_text"   : natural_text,

        # Metadata for filtering and display
        "product_name" : product["name"],
        "category"     : product["category"],

        # All numeric specs as flat columns — used for range filters
        **numeric_meta,

        # All qualitative specs as flat columns — used for exact filters
        **qual_meta,

        "source"       : "pastes_io",
        "created_ts"   : datetime.now(),
    }
    chunk_records.append(record)

# Verify: should be exactly 5 records (one per product)
print(f"Total chunks created: {len(chunk_records)}")
print(f"\nSample chunk text for ELEC-001:\n")
print(chunk_records[0]["chunk_text"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Bronze Layer — Store Raw Product Records

# COMMAND ----------

# ── FIX 3: Schema now has product_id, category, and numeric metadata ──────────
# We only define the core columns in the Spark schema.
# Numeric metadata columns vary per product category so we let Spark infer them.

from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, TimestampType
)

# Use spark.createDataFrame with schema inference for flexibility
# (numeric metadata columns differ across product categories)
bronze_df = spark.createDataFrame(chunk_records)

# Add row_number for ordering
from pyspark.sql.functions import row_number, col
from pyspark.sql.window import Window

windowSpec  = Window.orderBy("chunk_index")
bronze_df   = bronze_df.withColumn("row_num", row_number().over(windowSpec))

display(bronze_df.select(
    "product_id", "product_name", "category", "chunk_index", "chunk_text"
))

# COMMAND ----------

bronze_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("bronze_product_chunks")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify: 5 rows, one per product, no broken fragments
# MAGIC SELECT product_id, product_name, category, chunk_index,
# MAGIC        LEFT(chunk_text, 120) AS chunk_preview
# MAGIC FROM   bronze_product_chunks
# MAGIC ORDER  BY chunk_index

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Silver Layer — Clean and Validate

# COMMAND ----------

from pyspark.sql.functions import trim, length, col

silver_df = bronze_df.withColumn(
    "clean_chunk",
    trim(col("chunk_text"))
)

# ── FIX 4: Validate — reject empty or too-short chunks ───────────────────────
# OLD code had no validation — broken fragments like ': "Semiconductors",'
# (14 chars) passed through silently and got embedded, polluting the index.
silver_df = silver_df.filter(length(col("clean_chunk")) > 50)

print(f"Valid chunks after cleaning: {silver_df.count()}")

silver_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("silver_product_chunks")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT product_id, product_name, category,
# MAGIC        LENGTH(clean_chunk) AS text_length,
# MAGIC        LEFT(clean_chunk, 200) AS preview
# MAGIC FROM   silver_product_chunks
# MAGIC ORDER  BY chunk_index

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Generate Embeddings

# COMMAND ----------

# MAGIC %pip install sentence-transformers

# COMMAND ----------

from sentence_transformers import SentenceTransformer

# ── FIX 5: Use a better embedding model ──────────────────────────────────────
# OLD: all-MiniLM-L6-v2 (384 dims) — general purpose, adequate
# NEW: BAAI/bge-small-en-v1.5 (384 dims) — better retrieval performance
#      Same dimension so vector index config stays the same.
#      For production use bge-large-en-v1.5 (1024 dims) — update
#      embedding_dimension in the index creation cell accordingly.
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType

# ── FIX 6: Embed clean_chunk (natural language), not raw chunk_text ───────────
# OLD: embedding_udf("clean_chunk") — but clean_chunk was still raw JSON
# NEW: clean_chunk is now proper natural language — embedding quality is high

def generate_embedding(text: str):
    """
    BGE models work best with a query prefix for retrieval tasks.
    For DOCUMENTS (stored chunks) no prefix is needed.
    For QUERIES (user search) prefix with 'Represent this sentence: '
    """
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()

embedding_udf = udf(generate_embedding, ArrayType(FloatType()))

embedding_df = silver_df.withColumn(
    "embedding",
    embedding_udf("clean_chunk")
)

display(embedding_df.select(
    "product_id", "product_name", "category",
    "clean_chunk",
    col("embedding")[0].alias("embedding_dim_0_sample")
))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualise Embeddings — Now Products Will Cluster by Meaning

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

sample_pd = embedding_df.select(
    "product_name", "category", "embedding"
).toPandas()

embeddings_np = np.stack(sample_pd["embedding"].values)

pca     = PCA(n_components=2)
reduced = pca.fit_transform(embeddings_np)

# Colour by category
categories      = sample_pd["category"].values
unique_cats     = list(set(categories))
colour_map      = plt.cm.get_cmap("tab10", len(unique_cats))
category_colours = {c: colour_map(i) for i, c in enumerate(unique_cats)}

plt.figure(figsize=(10, 7))
for i, (name, cat) in enumerate(zip(sample_pd["product_name"], categories)):
    colour = category_colours[cat]
    plt.scatter(reduced[i, 0], reduced[i, 1], color=colour, s=200, zorder=3)
    plt.annotate(
        name,
        (reduced[i, 0], reduced[i, 1]),
        fontsize=9,
        xytext=(8, 4),
        textcoords="offset points"
    )

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=category_colours[c], label=c) for c in unique_cats
]
plt.legend(handles=legend_elements, loc="best")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Product Embeddings — Each Point = One Complete Product")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# With correct chunking, similar products cluster together in the plot.
# Energy Storage should be far from Displays, Sensors, etc.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Gold Layer — Store Embeddings

# COMMAND ----------

# ── FIX 7: Gold table now has product_id and category as queryable columns ────
embedding_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("gold_product_embeddings")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Enable Change Data Feed for Vector Search sync
# MAGIC ALTER TABLE workspace.default.gold_product_embeddings
# MAGIC SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Vector Search Index

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()

# Create endpoint (skip if already exists)
try:
    client.create_endpoint(
        name="product-vector-endpoint",
        endpoint_type="STANDARD"
    )
    print("Endpoint created.")
except Exception as e:
    print(f"Endpoint may already exist: {e}")

# COMMAND ----------

# Create vector index (skip if already exists)
try:
    client.create_delta_sync_index(
        endpoint_name  = "product-vector-endpoint",
        index_name     = "workspace.default.product_index_v2",   # v2 = fixed index
        source_table_name = "workspace.default.gold_product_embeddings",
        pipeline_type  = "TRIGGERED",
        primary_key    = "chunk_id",
        embedding_dimension = 384,           # matches bge-small-en-v1.5
        embedding_vector_column = "embedding"
    )
    print("Index creation started. Wait 2-3 minutes for sync.")
except Exception as e:
    print(f"Index may already exist: {e}")

# COMMAND ----------

# Wait for index to be ready
import time

index = client.get_index(
    endpoint_name = "product-vector-endpoint",
    index_name    = "workspace.default.product_index_v2"
)

for attempt in range(20):
    status = index.describe().get("status", {})
    ready  = status.get("ready", False)
    print(f"Attempt {attempt+1}/20 — Index ready: {ready}")
    if ready:
        break
    time.sleep(15)
else:
    raise RuntimeError("Index not ready after 5 minutes. Check Databricks UI.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Query — Semantic Search

# COMMAND ----------

# ── FIX 8: Add BGE query prefix for better retrieval accuracy ─────────────────
# BGE models use a prefix for query encoding (not for document encoding)
# "Represent this sentence for searching relevant passages: "

def encode_query(query_text: str) -> list:
    """Encode user query with BGE prefix for best retrieval performance."""
    prefixed = f"Represent this sentence for searching relevant passages: {query_text}"
    return model.encode(prefixed, normalize_embeddings=True).tolist()


query        = "battery with high energy density"
query_vector = encode_query(query)
print(f"Query: '{query}'")
print(f"Query vector dimension: {len(query_vector)}")

# COMMAND ----------

# ── FIX 9: Retrieve with category metadata filter ─────────────────────────────
# Semantic search finds similar products.
# Metadata filter optionally narrows to a specific category.
# For a general query like this we don't filter — let semantic search decide.

results = index.similarity_search(
    query_vector = query_vector,
    columns      = [
        "chunk_text",    # full natural language product description
        "document_id",
        "chunk_id",
        "product_id",    # NEW — traceable to source
        "product_name",  # NEW — human readable
        "category",      # NEW — for display and filtering
        "row_num"
    ],
    num_results = 3,
    # Uncomment to add hard category filter:
    # filters = {"category": "Energy Storage"}
)

print(f"\nTop {len(results['result']['data_array'])} results for: '{query}'\n")
print("-" * 60)
for i, row in enumerate(results["result"]["data_array"]):
    chunk_text   = row[0]
    product_id   = row[3]
    product_name = row[4]
    category     = row[5]
    score        = row[-1]  # similarity score — last column
    print(f"\nRank {i+1} | Score: {score:.4f}")
    print(f"Product: {product_name} ({product_id}) | Category: {category}")
    print(f"Preview: {chunk_text[:200]}...")
    print("-" * 60)

# Expected output with fixed chunking:
# Rank 1 | Score: ~0.85+ → Lithium-Ion Battery L100 (Energy Storage)
# Rank 2 | Score: ~0.65  → some other product (much lower, correctly ranked down)
# OLD result had score 0.48 — fixed chunking significantly improves scores

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Build Context and Call LLM

# COMMAND ----------

# ── FIX 10: Build clean, structured context from full product descriptions ─────
# OLD: context was broken JSON fragments — LLM could not parse them correctly
# NEW: context is clean natural language — LLM reads it accurately

data_rows = results["result"]["data_array"]

context_parts = []
for row in data_rows:
    chunk_text   = row[0]
    product_name = row[4]
    category     = row[5]
    score        = row[-1]
    context_parts.append(
        f"[Product: {product_name} | Category: {category} | Relevance: {score:.3f}]\n"
        f"{chunk_text}"
    )

context = "\n\n---\n\n".join(context_parts)

print("Context being sent to LLM:\n")
print(context)

# COMMAND ----------

prompt = f"""You are a product search assistant. Answer the user's question 
using ONLY the product information provided in the context below.

If multiple products are relevant, list them in order of relevance.
If no product matches the query, say "No matching product found."
Always mention the Product ID and specific specifications that match the query.

Context:
{context}

Question: {query}

Answer:"""

print(prompt)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

w = WorkspaceClient()

response = w.serving_endpoints.query(
    name     = "databricks-meta-llama-3-1-405b-instruct",
    messages = [
        ChatMessage(
            role    = ChatMessageRole.USER,
            content = prompt
        )
    ],
    # ── FIX 11: Set temperature=0 for factual product retrieval ──────────────
    # We want deterministic, accurate answers — not creative responses.
    # temperature=0 means the LLM always picks the most likely token.
    temperature = 0,
    max_tokens  = 512
)

print("LLM Response:\n")
print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10 (Bonus): Hard Filter Query Example
# MAGIC
# MAGIC When the user specifies numeric constraints like "battery with minimum 600 charge cycles",
# MAGIC use metadata filtering to apply the constraint BEFORE semantic ranking.

# COMMAND ----------

# Example: "Show me batteries with at least 600 charge cycles"
# Step 1: Semantic search (finds battery-related products)
# Step 2: Metadata filter (enforces the numeric constraint)

filtered_results = index.similarity_search(
    query_vector = encode_query("battery with high charge cycles"),
    columns      = [
        "chunk_text", "document_id", "chunk_id",
        "product_id", "product_name", "category",
        "charge_cycles_min", "charge_cycles_max",   # numeric metadata columns
        "row_num"
    ],
    num_results = 5,
    filters     = {
        # Hard filter: only products where charge_cycles_min >= 600
        "charge_cycles_min >=": 600
    }
)

print("Filtered results (charge_cycles_min >= 600):")
for row in filtered_results["result"]["data_array"]:
    print(f"  {row[4]} | cycles: {row[6]} to {row[7]} | score: {row[-1]:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary of All Fixes
# MAGIC
# MAGIC | # | Old (Broken) | New (Fixed) |
# MAGIC |---|---|---|
# MAGIC | 1 | `chunk_text(raw_json, 500)` — character split | `product_to_natural_language(product)` — one chunk per product |
# MAGIC | 2 | 9 broken JSON fragments for 5 products | 5 clean natural language descriptions |
# MAGIC | 3 | Products split across multiple chunks | Each chunk = one complete product |
# MAGIC | 4 | No metadata — no range filtering possible | Numeric fields extracted as flat columns |
# MAGIC | 5 | `all-MiniLM-L6-v2` — general model | `bge-small-en-v1.5` — retrieval-optimised |
# MAGIC | 6 | No query prefix | BGE query prefix for better recall |
# MAGIC | 7 | Best retrieval score: 0.48 | Expected score: 0.85+ |
# MAGIC | 8 | Wrong chunks retrieved (display panel for battery query) | Only relevant products retrieved |
# MAGIC | 9 | LLM got broken JSON context | LLM gets clean readable context |
# MAGIC | 10 | No numeric filtering | Hard filter via metadata (e.g. cycles >= 600) |
# MAGIC | 11 | No temperature set | temperature=0 for factual answers |
