# Databricks notebook source

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

def chunk_text(text, chunk_size=500, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

    return chunks

# COMMAND ----------

chunks = chunk_text(raw_document)

# COMMAND ----------

chunks

# COMMAND ----------

from datetime import datetime
import uuid

chunk_records = []

for idx, chunk in enumerate(chunks):

    chunk_records.append({

        "document_id": "DOC_PRODUCT_001",

        "chunk_id": str(uuid.uuid4()),

        "chunk_index": idx,

        "chunk_text": chunk,

        "source": "pastes_io",

        "created_ts": datetime.now()
    })

# COMMAND ----------

chunk_records 

# COMMAND ----------

from pyspark.sql.types import *

schema = StructType([

    StructField("document_id", StringType(), True),

    StructField("chunk_id", StringType(), True),

    StructField("chunk_index", IntegerType(), True),

    StructField("chunk_text", StringType(), True),

    StructField("source", StringType(), True),

    StructField("created_ts", TimestampType(), True)

])

# COMMAND ----------

df = spark.createDataFrame(
    chunk_records,
    schema=schema
)

# COMMAND ----------

display(df)

# COMMAND ----------

from pyspark.sql.functions import row_number
from pyspark.sql.window import Window
from pyspark.sql.functions import col

windowSpec = Window.orderBy("chunk_index")

df = df.withColumn(
    "row_num",
    row_number().over(windowSpec)
)

display(df)

# COMMAND ----------


batch_size = 5

df = df.withColumn(
    "batch_id",
    ((col("row_num") - 1) / batch_size).cast("integer")
)
display(df,100)

# COMMAND ----------

df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("bronze_product_chunks")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from bronze_product_chunks

# COMMAND ----------

from pyspark.sql.functions import trim

silver_df = df.withColumn(
    "clean_chunk",
    trim(col("chunk_text"))
)


# COMMAND ----------

silver_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("silver_product_chunks")

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC *
# MAGIC FROM silver_product_chunks
# MAGIC

# COMMAND ----------

# MAGIC %pip install sentence-transformers

# COMMAND ----------

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType

def generate_embedding(text):

    embedding = model.encode(text)

    return embedding.tolist()

embedding_udf = udf(
    generate_embedding,
    ArrayType(FloatType())
)

# COMMAND ----------

embedding_df = silver_df.withColumn(
    "embedding",
    embedding_udf("clean_chunk")
)

# COMMAND ----------

display(embedding_df)

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

sample_df = embedding_df.select("clean_chunk", "embedding").limit(20).toPandas()

embeddings = np.stack(sample_df["embedding"].values)
texts = sample_df["clean_chunk"].values

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 7))
plt.scatter(reduced[:, 0], reduced[:, 1])

for i, text in enumerate(texts):
    plt.annotate(text[:30], (reduced[i, 0], reduced[i, 1]), fontsize=8)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Embedding Visualization with Clean Text")
plt.show()

# COMMAND ----------

embedding_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("gold_product_embeddings")

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch

# COMMAND ----------

# DBTITLE 1,Cell 23
from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()

# COMMAND ----------

client.create_endpoint(
    name="product-vector-endpoint",
    endpoint_type="STANDARD"
)

# COMMAND ----------

# DBTITLE 1,Enable Change Data Feed
# MAGIC %sql
# MAGIC ALTER TABLE workspace.default.gold_product_embeddings SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# DBTITLE 1,Cell 26
client.create_delta_sync_index(

    endpoint_name="product-vector-endpoint",

    index_name="workspace.default.product_index",

    source_table_name="workspace.default.gold_product_embeddings",

    pipeline_type="TRIGGERED",

    primary_key="chunk_id",

    embedding_dimension=384,

    embedding_vector_column="embedding"
)

# COMMAND ----------

query = "battery with high energy density"
query_vector = model.encode(query).tolist()

# COMMAND ----------

# DBTITLE 1,Cell 29
index = client.get_index(
    endpoint_name="product-vector-endpoint",
    index_name="workspace.default.product_index"
)

# Check if index is ready
index_status = index.describe()
status_state = index_status.get('status', {}).get('ready', False)
print(f"Index ready: {status_state}")
print(f"Full status: {index_status.get('status')}")

if status_state:
    results = index.similarity_search(
        query_vector=query_vector,
        columns=["chunk_text", "document_id","chunk_id","row_num","batch_id"],
        num_results=3
    )
else:
    print("Index is not ready yet. Please wait a few moments and try again.")

# COMMAND ----------

results 

# COMMAND ----------

# DBTITLE 1,Cell 30
if 'results' in locals() and results is not None:
    # data_array contains lists: [chunk_text, document_id, score]
    context = "\n".join([
        r[0]  # chunk_text is the first element
        for r in results["result"]["data_array"]
    ])
    print(f"Retrieved {len(results['result']['data_array'])} relevant chunks")
else:
    print("Results not available. Please run the previous cell and ensure the index is ready.")

# COMMAND ----------

# DBTITLE 1,Cell 31
if 'context' in locals() and context is not None and 'query' in locals() and query is not None:
    prompt = f"""

Answer the question based on the context.

Context:
{context}

Question:
{query}

"""
else:
    print("Context or query not available. Please ensure the index is ready and run the previous cells first.")

# COMMAND ----------

# DBTITLE 1,Cell 32
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

if 'prompt' in locals() and prompt is not None:
    w = WorkspaceClient()

    response = w.serving_endpoints.query(
        name="databricks-meta-llama-3.1-405b-instruct",
        messages=[
            ChatMessage(
                role=ChatMessageRole.USER,
                content=prompt
            )
        ]
    )

    print(response.choices[0].message.content)
else:
    print("Prompt not available. Please run the previous cells to generate the prompt first.")

# COMMAND ----------

# MAGIC %md
# MAGIC Correct Code
# MAGIC

# COMMAND ----------

