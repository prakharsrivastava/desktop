# Databricks notebook source
# MAGIC %md
# MAGIC ### A. Text similarity using AI Query function in SQL

# COMMAND ----------

# MAGIC %md
# MAGIC #### A.1 Generate embeddings using AI function in SQL

# COMMAND ----------

# DBTITLE 1,Use ai_query function
embeddingsDF = spark.sql("""  SELECT  
                                       -- Get embeddings for first prompt                            
                                       ai_query(
                                                    "databricks-gte-large-en",              -- Model name
                                                    "XBox is an excellent gaming console"   -- Prompt

                                               ) AS embedding1,

                                       -- Get embeddings for second prompt
                                       ai_query(
                                                    "databricks-gte-large-en",              -- Model name
                                                    "I can spend all day with PlayStation"  -- Prompt

                                               ) AS embedding2
                        """)

display(embeddingsDF)


# COMMAND ----------

# MAGIC %md
# MAGIC #### A.2 Create 'Cosine Similarity' method

# COMMAND ----------

# DBTITLE 1,Define method
import numpy as np

def cosine_similarity(embedding1, embedding2):

    # Convert lists to numpy arrays for vector operations
    arr1 = np.array(embedding1)
    arr2 = np.array(embedding2)

    # Perform cosine similarity
    cosineSimilarityScore = np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))

    return cosineSimilarityScore

# COMMAND ----------

# MAGIC %md
# MAGIC #### A.3 Use cosine similarity to generate similarity score

# COMMAND ----------

# DBTITLE 1,Compare embeddings
# Get first row from embeddingsDF DataFrame
embeddingsRow = embeddingsDF.first()

# Extract values of embedding1 and embedding2 columns
embedding1 = embeddingsRow['embedding1']
embedding2 = embeddingsRow['embedding2']

# Perform cosine similarity
score = cosine_similarity(embedding1, embedding2)

# Print similarity score
print(f"Cosine similarity score: {score}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### B. Text similarity by querying serving endpoint in Python

# COMMAND ----------

# DBTITLE 1,Use serving_endpoints.query function
from databricks.sdk import WorkspaceClient

# Create client to workspace
client = WorkspaceClient()

# Generate embedding 1
response1 = client.serving_endpoints.query(
                                              name="databricks-gte-large-en",                               #Model
                                              input="How many days are covered under pre-hospitalization?"  # Prompt
                                          )
embedding1 = response1.data[0].embedding

# Generate embedding 2
response2 = client.serving_endpoints.query(
                                              name="databricks-gte-large-en",
                                              input="We will not be liable to pay Pre-hospitalization Medical Expenses for more than 90 days immediately preceding the Insured Person's admission for Inpatient Care/ Day Care Treatment/ Domiciliary Hospitalization / Modern Treatments or such expenses incurred prior to inception of the First Policy with Us."
                                          )
embedding2 = response2.data[0].embedding

# Perform cosine similarity
score = cosine_similarity(embedding1, embedding2)

# Print similarity score
print(f"Cosine similarity score: {score}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### C. Text similarity using AI Similarity function in SQL

# COMMAND ----------

# DBTITLE 1,Use ai_similarity function
# MAGIC %sql
# MAGIC
# MAGIC SELECT ai_similarity(
# MAGIC                         'I can spend all day with PlayStation',
# MAGIC
# MAGIC                         'XBox is an excellent gaming console'
# MAGIC                     )

# COMMAND ----------

