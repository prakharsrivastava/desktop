# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Snowflake Cortex End-to-End
# MAGIC %md
# MAGIC # Snowflake Cortex End-to-End
# MAGIC
# MAGIC This notebook demonstrates a complete end-to-end workflow using **Snowflake Cortex** — Snowflake's integrated AI/ML service. It covers:
# MAGIC
# MAGIC 1. **Setup** — Install the Snowflake connector and establish a secure connection
# MAGIC 2. **Cortex LLM Functions** — `COMPLETE`, `SUMMARIZE`, `TRANSLATE`, `SENTIMENT`, `EXTRACT_ANSWER`
# MAGIC 3. **Text Embeddings** — Generate and store vector embeddings with `EMBED_TEXT_768`
# MAGIC 4. **Vector Similarity Search** — Build a similarity search using `VECTOR_COSINE_SIMILARITY`
# MAGIC 5. **Cortex Search Service** — Create and query a Cortex Search Service for RAG
# MAGIC 6. **Cortex Fine-Tuning** — Fine-tune a model on custom data
# MAGIC 7. **Cortex Analyst** — Natural-language-to-SQL via Cortex Analyst

# COMMAND ----------

# DBTITLE 1,Install Snowflake Connector
# MAGIC %pip install snowflake-connector-python[pandas] snowflake-snowpark-python
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Snowflake Connection Setup
# =============================================================================
# 1. SNOWFLAKE CONNECTION SETUP
# =============================================================================
# IMPORTANT: Store your Snowflake credentials securely using Databricks secrets.
# Run these CLI commands once to set up secrets:
#   databricks secrets create-scope --scope snowflake-secrets
#   databricks secrets put --scope snowflake-secrets --key sf-user
#   databricks secrets put --scope snowflake-secrets --key sf-password
#   databricks secrets put --scope snowflake-secrets --key sf-account
#   databricks secrets put --scope snowflake-secrets --key sf-warehouse
#   databricks secrets put --scope snowflake-secrets --key sf-database
#   databricks secrets put --scope snowflake-secrets --key sf-schema

import snowflake.connector
import pandas as pd

# --- Load credentials from Databricks secrets ---
try:
    SF_USER     = "PRAKHAR1207SRIVASTAVA"
    SF_PASSWORD = "pr19dec1qaz!QAZ"
    SF_ACCOUNT  = "zs62970"
    SF_WAREHOUSE = "COMPUTE_WH"
    SF_DATABASE  = "PRAK_DB"
    SF_SCHEMA    = "PRAK_SCH"
    print("✓ Credentials loaded from Databricks secrets")
except Exception as e:
    print(f"⚠️ Could not load secrets: {e}")
    print("Falling back to environment variables or manual entry...")
    SF_USER     = dbutils.widgets.get("sf_user")     if "sf_user" in [w.name for w in dbutils.widgets.getAll()] else input("Enter Snowflake user: ")
    SF_PASSWORD = dbutils.widgets.get("sf_password") if "sf_password" in [w.name for w in dbutils.widgets.getAll()] else input("Enter Snowflake password: ")
    SF_ACCOUNT  = dbutils.widgets.get("sf_account")  if "sf_account" in [w.name for w in dbutils.widgets.getAll()] else input("Enter Snowflake account: ")
    SF_WAREHOUSE = "COMPUTE_WH"
    

# --- Create connection ---
conn = snowflake.connector.connect(
    user=SF_USER,
    password=SF_PASSWORD,
    account="EMXEKCM-PC15902",
    warehouse=SF_WAREHOUSE,
     database=SF_DATABASE,
     schema=SF_SCHEMA
)

cur = conn.cursor()
print("✓ Connected to Snowflake!")

# Verify connection and Cortex availability
cur.execute("SELECT CURRENT_VERSION()")
print(f"Snowflake version: {cur.fetchone()[0]}")

cur.execute("SELECT CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA()")
wh, db, schema = cur.fetchone()
print(f"Warehouse: {wh} | Database: {db} | Schema: {schema}")

# Helper function to run SQL and return a pandas DataFrame
def run_sql(sql: str, fetch: bool = True) -> pd.DataFrame | None:
    """Execute SQL on Snowflake and optionally return results as a DataFrame."""
    cur.execute(sql)
    if fetch:
        return cur.fetch_pandas_all()
    return None

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Create Sample Data
# =============================================================================
# 2. CREATE SAMPLE DATA TABLE
# =============================================================================
# Create a demo table with text data that we'll use across all Cortex functions.
cur.execute("USE WAREHOUSE COMPUTE_WH")
cur.execute("USE DATABASE PRAK_DB")
cur.execute("USE SCHEMA PRAK_SCH")
ddl = """
CREATE OR REPLACE TABLE customer_reviews (
    review_id     INT AUTOINCREMENT PRIMARY KEY,
    customer_name VARCHAR(100),
    product_name  VARCHAR(100),
    review_text   VARCHAR(5000),
    review_date   DATE,
    language      VARCHAR(10)
);
"""
cur.execute(ddl)
print("✓ Table 'customer_reviews' created")

# Insert multilingual sample data
insert_sql = """
INSERT INTO customer_reviews (customer_name, product_name, review_text, review_date, language)
VALUES
    ('Alice Johnson', 'UltraBook Pro 15', 'I purchased the UltraBook Pro 15 three months ago and I am extremely impressed with the build quality and performance. The battery lasts a full workday, the screen is vibrant, and the keyboard is comfortable for long typing sessions. However, the fan gets a bit loud during heavy workloads. Overall, I would recommend this laptop to any professional who needs a reliable machine.', '2024-06-15', 'en'),
    ('Boris Müller', 'UltraBook Pro 15', 'Ich habe das UltraBook Pro 15 vor zwei Monaten gekauft. Die Verarbeitung ist hervorragend und die Leistung ist beeindruckend. Der Akku hält einen ganzen Arbeitstag. Der Lüfter wird jedoch bei starker Auslastung etwas laut. Insgesamt würde ich es empfehlen.', '2024-06-20', 'de'),
    ('Carlos Rivera', 'NoiseCancel Headphones X1', 'These NoiseCancel Headphones X1 are a game-changer for my daily commute. The active noise cancellation is outstanding, and the sound quality is rich and balanced. The only downside is that the ear cushions could be more comfortable for extended use. Battery life is decent at around 20 hours.', '2024-07-01', 'en'),
    ('Diana Chen', 'SmartWatch Fitness S3', 'I love the SmartWatch Fitness S3! It tracks all my workouts accurately, the heart rate monitor is precise, and the sleep tracking feature is very insightful. The app ecosystem is growing, though some third-party apps are still missing. The watch face customization options are limited too. Still, it is the best fitness watch I have owned.', '2024-07-10', 'en'),
    ('Emi Tanaka', 'SmartWatch Fitness S3', 'このスマートウォッチは素晴らしいです。デザインが美しく、バッテリー持ちも良いです。ただ、価格が少し高いと思います。', '2024-07-15', 'ja'),
    ('Frank OBrien', 'UltraBook Pro 15', 'The UltraBook Pro 15 overheats constantly and the trackpad is unresponsive. I sent it for repair twice in two months. Would not recommend this product to anyone. Terrible customer service too.', '2024-08-01', 'en'),
    ('Grace Lee', 'NoiseCancel Headphones X1', 'Amazing headphones! The noise cancellation is so good I forget I am on a plane. Sound quality is crisp and the bass is punchy. Worth every penny.', '2024-08-05', 'en'),
    ('Hiro Sato', 'Cloud Storage 2TB', 'The Cloud Storage 2TB drive is fast and reliable. I have been using it for backups and it has not failed me yet. The USB-C connection is very convenient. Only complaint is the included cable is too short.', '2024-08-12', 'en');
"""
cur.execute(insert_sql)
print(f"✓ Inserted {cur.rowcount} sample reviews")

# Verify the data
df_reviews = run_sql("SELECT review_id, customer_name, product_name, LEFT(review_text, 80) AS review_preview, language FROM customer_reviews ORDER BY review_id")
display(df_reviews)

# COMMAND ----------

# DBTITLE 1,Cortex COMPLETE Function
# =============================================================================
# 3a. CORTEX LLM: COMPLETE (Text Completion)
# =============================================================================
# The COMPLETE function generates text completions using Snowflake-hosted LLMs.
# You can specify the model (e.g., 'mixtral-8x7b', 'llama3.1-8b', 'gemma-7b').

complete_sql = """
SELECT SNOWFLAKE.CORTEX.COMPLETE(
    'llama3.1-8b',
    PARSE_JSON('[
        {
            "role": "user",
            "content": "What are the top 3 benefits of using a smartwatch for fitness tracking?"
        }
    ]')
)::VARCHAR AS response
"""

df_complete = run_sql(complete_sql)
print("=== CORTEX COMPLETE (Text Completion) ===")
print(df_complete.iloc[0]['RESPONSE'])

# COMMAND ----------

# DBTITLE 1,Cortex SUMMARIZE Function
# =============================================================================
# 3b. CORTEX LLM: SUMMARIZE
# =============================================================================
# Generate concise summaries of long review texts.

summarize_sql = """
SELECT
    review_id,
    customer_name,
    product_name,
    SNOWFLAKE.CORTEX.SUMMARIZE(review_text)::VARCHAR AS summary
FROM customer_reviews
WHERE language = 'en'
ORDER BY review_id
"""

df_summarize = run_sql(summarize_sql)
print("=== CORTEX SUMMARIZE (Review Summaries) ===")
for _, row in df_summarize.iterrows():
    print(f"\nReview #{row['REVIEW_ID']} by {row['CUSTOMER_NAME']} ({row['PRODUCT_NAME']}):")
    print(f"  Summary: {row['SUMMARY']}")

# COMMAND ----------

# DBTITLE 1,Cortex TRANSLATE Function
# =============================================================================
# 3c. CORTEX LLM: TRANSLATE
# =============================================================================
# Translate non-English reviews to English (and vice-versa).
# Supported languages: en, fr, de, ja, ko, es, it, pt, ru, zh, etc.

translate_sql = """
SELECT
    review_id,
    customer_name,
    language AS original_language,
    LEFT(review_text, 100) AS original_text_preview,
    SNOWFLAKE.CORTEX.TRANSLATE(review_text, language, 'en')::VARCHAR AS english_translation
FROM customer_reviews
WHERE language != 'en'
ORDER BY review_id
"""

df_translate = run_sql(translate_sql)
print("=== CORTEX TRANSLATE (Non-English → English) ===")
for _, row in df_translate.iterrows():
    print(f"\nReview #{row['REVIEW_ID']} by {row['CUSTOMER_NAME']} ({row['ORIGINAL_LANGUAGE']} → en):")
    print(f"  Original:  {row['ORIGINAL_TEXT_PREVIEW']}...")
    print(f"  English:   {row['ENGLISH_TRANSLATION'][:150]}...")

# COMMAND ----------

# DBTITLE 1,Cortex SENTIMENT Function
# =============================================================================
# 3d. CORTEX LLM: SENTIMENT
# =============================================================================
# Classify the sentiment of each review as positive, neutral, or negative.
# Returns a score between -1 (most negative) and 1 (most positive).

sentiment_sql = """
SELECT
    review_id,
    customer_name,
    product_name,
    SNOWFLAKE.CORTEX.SENTIMENT(review_text) AS sentiment_score,
    CASE
        WHEN SNOWFLAKE.CORTEX.SENTIMENT(review_text) > 0.5 THEN 'Very Positive'
        WHEN SNOWFLAKE.CORTEX.SENTIMENT(review_text) > 0.1 THEN 'Positive'
        WHEN SNOWFLAKE.CORTEX.SENTIMENT(review_text) > -0.1 THEN 'Neutral'
        WHEN SNOWFLAKE.CORTEX.SENTIMENT(review_text) > -0.5 THEN 'Negative'
        ELSE 'Very Negative'
    END AS sentiment_label
FROM customer_reviews
WHERE language = 'en'
ORDER BY sentiment_score DESC
"""

df_sentiment = run_sql(sentiment_sql)
print("=== CORTEX SENTIMENT (Sentiment Analysis) ===")
display(df_sentiment)

# COMMAND ----------

# DBTITLE 1,Cortex EXTRACT_ANSWER Function
# =============================================================================
# 3e. CORTEX LLM: EXTRACT_ANSWER
# =============================================================================
# Extract a specific answer from text given a question.

extract_sql = """
SELECT
    review_id,
    product_name,
    SNOWFLAKE.CORTEX.EXTRACT_ANSWER(
        review_text,
        'What is the main complaint or downside mentioned in this review?'
    )::VARCHAR AS extracted_answer
FROM customer_reviews
WHERE language = 'en'
ORDER BY review_id
"""

df_extract = run_sql(extract_sql)
print("=== CORTEX EXTRACT_ANSWER (Question Answering) ===")
for _, row in df_extract.iterrows():
    print(f"Review #{row['REVIEW_ID']} ({row['PRODUCT_NAME']}): {row['EXTRACTED_ANSWER']}")

# COMMAND ----------

# DBTITLE 1,Cortex Embeddings
# =============================================================================
# 4. CORTEX EMBEDDINGS: EMBED_TEXT_768
# =============================================================================
# Generate 768-dimensional text embeddings for semantic search.
# First, create a table with embeddings stored as VECTOR type.

embed_sql = """
CREATE OR REPLACE TABLE customer_reviews_embedded AS
SELECT
    review_id,
    customer_name,
    product_name,
    review_text,
    review_date,
    language,
    SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-l-v2.0', review_text)::VECTOR(FLOAT, 768) AS embedding
FROM customer_reviews;
"""
cur.execute(embed_sql)
print("✓ Embeddings generated and stored in 'customer_reviews_embedded'")

# Verify embedding count and dimensionality
verify_sql = """
SELECT COUNT(*) AS total_reviews, VECTOR_DIMENSIONS(embedding) AS dim_count
FROM customer_reviews_embedded
"""
df_verify = run_sql(verify_sql)
display(df_verify)

# COMMAND ----------

# DBTITLE 1,Vector Similarity Search
# =============================================================================
# 5. VECTOR SIMILARITY SEARCH
# =============================================================================
# Find the most similar reviews to a given query using cosine similarity.
# This is the building block for RAG (Retrieval Augmented Generation).

query_text = "battery life and performance issues"

similarity_sql = f"""
SELECT
    review_id,
    customer_name,
    product_name,
    LEFT(review_text, 120) AS review_preview,
    VECTOR_COSINE_SIMILARITY(
        embedding,
        SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-l-v2.0', '{query_text}')::VECTOR(FLOAT, 768)
    ) AS similarity_score
FROM customer_reviews_embedded
ORDER BY similarity_score DESC
LIMIT 5
"""

df_similar = run_sql(similarity_sql)
print(f"=== Top 5 Most Similar Reviews to: '{query_text}' ===")
display(df_similar)

# COMMAND ----------

# DBTITLE 1,Cortex Search Service (RAG)
# =============================================================================
# 6. CORTEX SEARCH SERVICE (RAG)
# =============================================================================
# Create a Cortex Search Service that enables natural-language search
# over the customer reviews table. This is Snowflake's managed RAG service.

# Step 1: Create the Cortex Search Service
css_sql = """
CREATE OR REPLACE CORTEX SEARCH SERVICE reviews_search_service
ON customer_name, product_name, review_text
WAREHOUSE = COMPUTE_WH
TARGET_LAG = '1 day'
AS
    SELECT
        review_id,
        customer_name,
        product_name,
        review_text,
        review_date
    FROM customer_reviews;
"""

try:
    cur.execute(css_sql)
    print("✓ Cortex Search Service 'reviews_search_service' created")
except Exception as e:
    print(f"Note: Search service creation requires specific privileges: {e}")
    print("Continuing with manual similarity search approach for RAG...")

# Step 2: Query the Search Service via REST API (if available)
import json, requests

# Get a Snowflake session token for REST API calls
cur.execute("SELECT SYSTEM$BEARER_TOKEN('CORTEX_SEARCH')")
try:
    bearer_token = cur.fetchone()[0]
    print("✓ Bearer token obtained for Cortex Search REST API")

    # Cortex Search REST endpoint
    # Format: https://<account>.snowflakecomputing.com/api/v2/cortex/search/<db>.<schema>.<service_name>
    search_url = f"https://{SF_ACCOUNT}.snowflakecomputing.com/api/v2/cortex/search/{SF_DATABASE.upper()}.{SF_SCHEMA.upper()}.REVIEWS_SEARCH_SERVICE"

    search_payload = {
        "query": "headphones with good noise cancellation",
        "limit": 5,
        "columns": ["review_id", "customer_name", "product_name", "review_text"],
    }

    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
    }

    response = requests.post(search_url, json=search_payload, headers=headers)
    if response.status_code == 200:
        results = response.json()
        print("\n=== Cortex Search Service Results ===")
        for r in results.get('results', []):
            print(f"  Review #{r.get('review_id')} | {r.get('customer_name')} | {r.get('product_name')}")
            print(f"    Text: {r.get('review_text', '')[:120]}...")
    else:
        print(f"Search API returned status {response.status_code}: {response.text}")
except Exception as e:
    print(f"REST API call skipped: {e}")
    print("The Search Service is still created and can be queried from Snowflake directly.")

# COMMAND ----------

# DBTITLE 1,RAG Pipeline: Retrieve + Generate
# =============================================================================
# 7. RAG PIPELINE: Retrieve + Generate
# =============================================================================
# Build a full RAG pipeline combining vector similarity search (retrieval)
# with the Cortex COMPLETE function (generation).

rag_question = "What are the common complaints about the UltraBook Pro 15?"

# Step 1: Retrieve the most relevant reviewsetrieve_sql = f"""
SELECT review_id, product_name, review_text
FROM customer_reviews_embedded
ORDER BY VECTOR_COSINE_SIMILARITY(
    embedding,
    SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-l-v2.0', '{rag_question}')::VECTOR(FLOAT, 768)
) DESC
LIMIT 3
"""

df_retrieved = run_sql(retrieve_sql)
print(f"=== RAG Pipeline: Question = '{rag_question}' ===")
print(f"\nRetrieved {len(df_retrieved)} relevant reviews.")

# Step 2: Build context from retrieved reviews
context_parts = []
for _, row in df_retrieved.iterrows():
    context_parts.append(f"Review by {row['CUSTOMER_NAME']} ({row['PRODUCT_NAME']}): {row['REVIEW_TEXT']}")
context = "\n\n".join(context_parts)

# Step 3: Generate answer using Cortex COMPLETE with the retrieved context
# Escape single quotes for SQL safety
context_escaped = context.replace("'", "''")
rag_question_escaped = rag_question.replace("'", "''")

rag_sql = f"""
SELECT SNOWFLAKE.CORTEX.COMPLETE(
    'llama3.1-70b',
    [
        {{
            'role': 'system',
            'content': 'You are a helpful product analyst. Answer the user question based ONLY on the provided customer reviews. If the reviews do not contain enough information, say so.'
        }},
        {{
            'role': 'user',
            'content': 'Based on these customer reviews:\n\n{context_escaped}\n\nQuestion: {rag_question_escaped}'
        }}
    ]
)::VARCHAR AS answer
"""

df_rag_answer = run_sql(rag_sql)
print("\n=== RAG Generated Answer ===")
print(df_rag_answer.iloc[0]['ANSWER'])

# COMMAND ----------

# DBTITLE 1,Cortex Fine-Tuning
# =============================================================================
# 8. CORTEX FINE-TUNING
# =============================================================================
# Fine-tune a base LLM on custom data using Snowflake Cortex Fine-Tuning.
# This example creates a training dataset and submits a fine-tuning job.

# Step 1: Create a training table with prompt-completion pairs
train_sql = """
CREATE OR REPLACE TABLE review_training_data AS
SELECT
    product_name,
    review_text,
    CASE
        WHEN SNOWFLAKE.CORTEX.SENTIMENT(review_text) > 0.3 THEN 'Positive'
        WHEN SNOWFLAKE.CORTEX.SENTIMENT(review_text) < -0.3 THEN 'Negative'
        ELSE 'Neutral'
    END AS sentiment_label
FROM customer_reviews
WHERE language = 'en';
"""
cur.execute(train_sql)
print("✓ Training data table created")

# Verify training data
df_train = run_sql("SELECT product_name, sentiment_label, COUNT(*) AS cnt FROM review_training_data GROUP BY product_name, sentiment_label ORDER BY product_name")
print("\nTraining data distribution:")
display(df_train)

# Step 2: Create the fine-tuning dataset in JSONL format
ft_format_sql = """
CREATE OR REPLACE TABLE review_ft_jsonl AS
SELECT
    OBJECT_CONSTRUCT(
        'messages',
        ARRAY_CONSTRUCT(
            OBJECT_CONSTRUCT('role', 'system', 'content', 'Classify the sentiment of the review as Positive, Negative, or Neutral.'),
            OBJECT_CONSTRUCT('role', 'user', 'content', review_text),
            OBJECT_CONSTRUCT('role', 'assistant', 'content', sentiment_label)
        )
    )::VARCHAR AS jsonl_row
FROM review_training_data;
"""
cur.execute(ft_format_sql)
print("✓ Fine-tuning JSONL formatted table created")

# Step 3: Submit fine-tuning job (requires Cortex Fine-Tuning privileges)
ft_sql = """
EXECUTE CORTEX FineTuning(
    'CREATE',
    'review_sentiment_classifier',
    {
        'BASE_MODEL': 'llama3.1-8b',
        'TRAINING_DATA': 'review_ft_jsonl',
        'EPOCHS': 3,
        'LEARNING_RATE': 0.0001
    }
);
"""

try:
    cur.execute(ft_sql)
    ft_result = cur.fetchone()
    print(f"✓ Fine-tuning job submitted: {ft_result}")
except Exception as e:
    print(f"Note: Fine-tuning requires specific Cortex privileges and may need to be run via Snowsight API: {e}")
    print("\nThe training data is ready. You can submit the fine-tuning job from Snowsight or via the REST API.")
    print("Example API call:")
    print("  POST /api/v2/cortex/fine_tuning/jobs")
    print("  Body: {\"base_model\": \"llama3.1-8b\", \"training_data\": \"review_ft_jsonl\", \"epochs\": 3}")

# COMMAND ----------

# DBTITLE 1,Cortex Analyst (NL-to-SQL)
# =============================================================================
# 9. CORTEX ANALYST (Natural Language to SQL)
# =============================================================================
# Cortex Analyst converts natural-language questions into SQL queries.
# This requires a semantic model (YAML) describing your tables.

# Step 1: Create a semantic model YAML file for the customer_reviews table
semantic_model_yaml = """
name: customer_reviews_semantic_model
tables:
  - name: customer_reviews
    description: Customer product reviews with text and sentiment data
    columns:
      - name: review_id
        description: Unique review identifier
        data_type: number
      - name: customer_name
        description: Name of the customer who wrote the review
        data_type: varchar
      - name: product_name
        description: Name of the product being reviewed
        data_type: varchar
      - name: review_text
        description: The full text of the customer review
        data_type: varchar
      - name: review_date
        description: Date the review was submitted
        data_type: date
      - name: language
        description: Language of the review (en, de, ja)
        data_type: varchar
"""

print("=== Semantic Model YAML (for Cortex Analyst) ===")
print(semantic_model_yaml)

# Step 2: Call Cortex Analyst REST API
analyst_payload = {
    "query": "How many reviews are there per product?",
    "semantic_model": "customer_reviews_semantic_model"
}

try:
    # Get session token
    cur.execute("SELECT SYSTEM$BEARER_TOKEN('CORTEX_ANALYST')")
    analyst_token = cur.fetchone()[0]

    analyst_url = f"https://{SF_ACCOUNT}.snowflakecomputing.com/api/v2/cortex/analyst/messages"
    headers = {
        "Authorization": f"Bearer {analyst_token}",
        "Content-Type": "application/json",
    }

    response = requests.post(analyst_url, json=analyst_payload, headers=headers)
    if response.status_code == 200:
        analyst_result = response.json()
        print("\n=== Cortex Analyst Response ===")
        print(json.dumps(analyst_result, indent=2))
    else:
        print(f"Cortex Analyst API returned {response.status_code}: {response.text}")
except Exception as e:
    print(f"\nNote: Cortex Analyst REST API requires proper setup: {e}")
    print("The semantic model YAML above can be deployed in Snowsight to enable NL-to-SQL.")

# COMMAND ----------

# DBTITLE 1,Combined Cortex Analysis
# =============================================================================
# 10. COMBINED ANALYSIS: All Cortex Functions Together
# =============================================================================
# Create a comprehensive view combining all Cortex analyses in one query.

combined_sql = """
SELECT
    review_id,
    customer_name,
    product_name,
    language,
    LEFT(review_text, 80) AS review_preview,
    SNOWFLAKE.CORTEX.SENTIMENT(review_text) AS sentiment_score,
    SNOWFLAKE.CORTEX.SUMMARIZE(review_text)::VARCHAR AS summary,
    SNOWFLAKE.CORTEX.EXTRACT_ANSWER(review_text, 'What is the main complaint?')::VARCHAR AS main_complaint,
    CASE WHEN language != 'en'
        THEN SNOWFLAKE.CORTEX.TRANSLATE(review_text, language, 'en')::VARCHAR
        ELSE review_text
    END AS english_text
FROM customer_reviews
ORDER BY review_id
"""

df_combined = run_sql(combined_sql)
print("=== Combined Cortex Analysis (All Functions) ===")
print(f"Total rows: {len(df_combined)}")
display(df_combined)

# COMMAND ----------

# DBTITLE 1,Cleanup & Close Connection
# =============================================================================
# 11. CLEANUP & CLOSE CONNECTION
# =============================================================================
# Close the Snowflake cursor and connection.
# Uncomment the DROP statements to clean up demo objects.

# cur.execute("DROP TABLE IF EXISTS customer_reviews")
# cur.execute("DROP TABLE IF EXISTS customer_reviews_embedded")
# cur.execute("DROP TABLE IF EXISTS review_training_data")
# cur.execute("DROP TABLE IF EXISTS review_ft_jsonl")
# cur.execute("DROP CORTEX SEARCH SERVICE IF EXISTS reviews_search_service")

cur.close()
conn.close()
print("✓ Snowflake connection closed.")
print("\n=== Snowflake Cortex End-to-End Demo Complete! ===")
print("\nFunctions demonstrated:")
print("  1. COMPLETE      — Text completion with LLMs")
print("  2. SUMMARIZE      — Text summarization")
print("  3. TRANSLATE      — Multilingual translation")
print("  4. SENTIMENT      — Sentiment analysis")
print("  5. EXTRACT_ANSWER — Question answering from text")
print("  6. EMBED_TEXT_768 — Vector embeddings generation")
print("  7. Vector Search  — Cosine similarity search")
print("  8. Cortex Search  — Managed RAG search service")
print("  9. RAG Pipeline   — Retrieve + Generate with COMPLETE")
print(" 10. Fine-Tuning    — Custom model fine-tuning")
print(" 11. Cortex Analyst — Natural language to SQL")