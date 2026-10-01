# Databricks notebook source


# COMMAND ----------

# DBTITLE 1,Cortex Analyst API — Copay Semantic View
# MAGIC %md
# MAGIC # Cortex Analyst API — Copay Semantic View
# MAGIC
# MAGIC This notebook calls the **Snowflake Cortex Analyst** REST API to run natural-language queries against the `COPAY_SEMANTIC_VIEW` semantic view deployed at `PRAK_DB.PRAK_SCH.COPAY_SEMANTIC_VIEW`.

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install snowflake-connector-python[pandas] snowflake-snowpark-python requests

# COMMAND ----------

# MAGIC %skip
# MAGIC # =============================================================================
# MAGIC # 1. SNOWFLAKE CONNECTION + CORTEX ANALYST API SETUP
# MAGIC # =============================================================================
# MAGIC import snowflake.connector
# MAGIC import pandas as pd
# MAGIC import requests
# MAGIC import json
# MAGIC
# MAGIC # --- Snowflake connection parameters (same as Cortex End-to-End notebook) ---
# MAGIC SF_USER      = "PRAKHAR1207SRIVASTAVA"
# MAGIC SF_PASSWORD  = "pr19dec1qaz!QAZ"
# MAGIC SF_ACCOUNT   = "EMXEKCM-PC15902"   # org-account identifier (used in connection)
# MAGIC SF_WAREHOUSE = "COMPUTE_WH"
# MAGIC SF_DATABASE  = "PRAK_DB"
# MAGIC SF_SCHEMA    = "PRAK_SCH"
# MAGIC

# COMMAND ----------

# DBTITLE 1,Connection + Cortex Analyst API Setup

# Semantic view fully qualified name
SEMANTIC_VIEW = f"{SF_DATABASE}.{SF_SCHEMA}.COPAY_SEMANTIC_VIEW"

# --- Create Snowflake connection ---
conn = snowflake.connector.connect(
    user=SF_USER,
    password=SF_PASSWORD,
    account=SF_ACCOUNT,
    warehouse=SF_WAREHOUSE,
    database=SF_DATABASE,
    schema=SF_SCHEMA,
)
cur = conn.cursor()
print("✓ Connected to Snowflake!")

# --- Helper: run SQL on Snowflake and return pandas DataFrame ---
def run_sql(sql: str) -> pd.DataFrame:
    cur.execute(sql)
    return cur.fetch_pandas_all()

# --- Cortex Analyst REST API caller ---
ANALYST_URL = f"https://{SF_ACCOUNT}.snowflakecomputing.com/api/v2/cortex/analyst/messages"

def ask_cortex_analyst(question: str, run_generated_sql: bool = True) -> dict:
    """
    Send a natural-language question to Cortex Analyst via REST API.

    Returns a dict with:
      - question:   the original NL question
        - sql:        generated SQL (if returned)
        - explanation: analyst's text response
        - results:    DataFrame of executed SQL (if run_generated_sql=True)
    """
    # 1. Build the request body
    payload = {
        "messages": [
            {"role": "user", "content": question}
        ],
        "semantic_view": SEMANTIC_VIEW,
    }
    body_str = json.dumps(payload)

    # 2. POST to Cortex Analyst via the connector's REST client
    #    conn.rest handles authentication (session token + cookies) automatically,
    #    avoiding the need for SYSTEM$BEARER_TOKEN which is unavailable here.
    api_path = "/api/v2/cortex/analyst/messages"
    response = conn.rest.request(
        url=api_path,
        body=body_str,
        method="post",
        timeout=120,
    )

    # 3. Read and parse the response
    raw = response.read() if hasattr(response, "read") else response
    status = response.status if hasattr(response, "status") else 200

    if status != 200:
        raise RuntimeError(
            f"Cortex Analyst API returned {status}:\n{raw}"
        )

    result = json.loads(raw) if isinstance(raw, (str, bytes)) else raw

    # 4. Parse the response — extract SQL and text from message content
    generated_sql = None
    explanation = ""
    message_content = result.get("message", {}).get("content", [])

    for block in message_content:
        if block.get("type") == "sql":
            generated_sql = block.get("statement", "")
        elif block.get("type") == "text":
            explanation += block.get("text_value", "")

    print(f"\n{'='*70}")
    print(f"❓ Question: {question}")
    if generated_sql:
        print(f"\n📝 Generated SQL:\n{generated_sql}")
    if explanation:
        print(f"\n💬 Analyst says:\n{explanation}")

    # 5. Optionally execute the generated SQL
    df_results = None
    if run_generated_sql and generated_sql:
        try:
            df_results = run_sql(generated_sql)
            print(f"\n📊 Results ({len(df_results)} rows):")
            display(df_results)
        except Exception as e:
            print(f"\n⚠️ Could not execute generated SQL: {e}")

    return {
        "question": question,
        "sql": generated_sql,
        "explanation": explanation,
        "results": df_results,
        "raw_response": result,
    }

print("✓ Cortex Analyst API helper ready!")
print(f"   Semantic View : {SEMANTIC_VIEW}")
print(f"   API Endpoint  : {ANALYST_URL}")

# COMMAND ----------

# DBTITLE 1,Diagnostic: Account Info & API Host
# =============================================================================
# DIAGNOSTIC: Check conn.rest API surface (no external calls)
# =============================================================================
import inspect

cur.execute("SELECT CURRENT_ACCOUNT(), CURRENT_REGION()")
account_info = cur.fetchone()
print(f"CURRENT_ACCOUNT():  {account_info[0]}")
print(f"CURRENT_REGION():   {account_info[1]}")
print(f"conn.rest host:      {conn.rest._host}")

# Check conn.rest.request signature
print(f"\nconn.rest.request signature: {inspect.signature(conn.rest.request)}")

# List all public methods on conn.rest
rest_methods = [m for m in dir(conn.rest) if not m.startswith('_') and callable(getattr(conn.rest, m))]
print(f"\nconn.rest public methods: {rest_methods}")

# Check if there's a 'token' attribute and what type it is
print(f"\nconn.rest.token type: {type(conn.rest.token)}")
print(f"conn.rest.token preview: {str(conn.rest.token)[:30]}...")

# --- Check what Cortex / Analyst functions exist ---
print("\n=== Checking available Cortex functions ===")
try:
    cur.execute("SHOW FUNCTIONS LIKE '%CORTEX%'")
    funcs = cur.fetchall()
    print(f"Functions matching CORTEX: {len(funcs)}")
    for f in funcs[:20]:
        print(f"  {f[0] if isinstance(f, tuple) else f}")
except Exception as e:
    print(f"SHOW FUNCTIONS error: {e}")

try:
    cur.execute("SHOW FUNCTIONS LIKE '%ANALYST%'")
    funcs = cur.fetchall()
    print(f"\nFunctions matching ANALYST: {len(funcs)}")
    for f in funcs[:20]:
        print(f"  {f[0] if isinstance(f, tuple) else f}")
except Exception as e:
    print(f"SHOW FUNCTIONS ANALYST error: {e}")

# --- Check if Cortex Analyst SQL function works ---
print("\n=== Testing Cortex Analyst SQL function ===")
try:
    cur.execute("""
        SELECT SNOWFLAKE.CORTEX.ANALYST(
            'What is the total Copay spend?',
            'PRAK_DB.PRAK_SCH.COPAY_SEMANTIC_VIEW'
        )
    """)
    result = cur.fetchone()
    print(f"SQL function result: {result}")
except Exception as e:
    print(f"SNOWFLAKE.CORTEX.ANALYST error: {e}")

# --- Check if the semantic view object exists and underlying tables ---
print("\n=== Checking semantic view and table objects ===")

# Check if COPAY_SEMANTIC_VIEW exists as a view/table
try:
    cur.execute("SELECT * FROM PRAK_DB.PRAK_SCH.COPAY_SEMANTIC_VIEW LIMIT 1")
    cols = [desc[0] for desc in cur.description]
    print(f"COPAY_SEMANTIC_VIEW exists! Columns: {cols}")
except Exception as e:
    print(f"COPAY_SEMANTIC_VIEW query error: {e}")

# Check if COPAY_CLAIMS table exists
try:
    cur.execute("SELECT * FROM PRAK_DB.PRAK_SCH.COPAY_CLAIMS LIMIT 5")
    df_sample = cur.fetch_pandas_all()
    print(f"\nCOPAY_CLAIMS table exists! Shape: {df_sample.shape}")
    print(f"Columns: {list(df_sample.columns)}")
    display(df_sample)
except Exception as e:
    print(f"COPAY_CLAIMS query error: {e}")

# List all objects in PRAK_SCH schema
try:
    cur.execute("SHOW TABLES IN PRAK_DB.PRAK_SCH")
    tables = cur.fetchall()
    print(f"\nTables in PRAK_SCH ({len(tables)}):")
    for t in tables[:20]:
        print(f"  {t[1] if isinstance(t, tuple) else t}")
except Exception as e:
    print(f"SHOW TABLES error: {e}")

try:
    cur.execute("SHOW VIEWS IN PRAK_DB.PRAK_SCH")
    views = cur.fetchall()
    print(f"\nViews in PRAK_SCH ({len(views)}):")
    for v in views[:20]:
        print(f"  {v[1] if isinstance(v, tuple) else v}")
except Exception as e:
    print(f"SHOW VIEWS error: {e}")

# Check if any Cortex Analyst projects are defined
try:
    cur.execute("SHOW CORTEX PROJECTS")
    projects = cur.fetchall()
    print(f"\nCortex Projects ({len(projects)}):")
    for p in projects[:10]:
        print(f"  {p}")
except Exception as e:
    print(f"\nSHOW CORTEX PROJECTS error: {e}")

# Try Snowflake SQL REST API to check if REST infra works
try:
    resp = conn.rest.request(url="/api/v2/statements", body="{}", method="post", timeout=30)
    raw = resp.read() if hasattr(resp, "read") else str(resp)
    status = resp.status if hasattr(resp, "status") else "?"
    print(f"\n/api/v2/statements → {status}: {str(raw)[:200]}")
except Exception as e:
    print(f"\n/api/v2/statements → ERROR: {str(e)[:200]}")

# COMMAND ----------

# DBTITLE 1,Queries: Copay Spend, by Brand, Unique Patients
# =============================================================================
# 2. QUERIES — Direct SQL against COPAY_CLAIMS (semantic view + API notes)
# =============================================================================
# NOTE: The Cortex Analyst REST API (/api/v2/cortex/analyst/messages) returns
# 404 on this Snowflake account — it's not available on trial accounts.
# Instead, we run the three NL questions as direct SQL against COPAY_CLAIMS,
# using the same metrics/dimutions defined in the COPAY_SEMANTIC_VIEW.

# --- Q1: What is the total Copay spend? ---
print("=" * 70)
print("❓ Q1: What is the total Copay spend?")
q1_sql = """
SELECT SUM(COPAY_AMOUNT) AS TOTAL_COPAY_SPEND
FROM PRAK_DB.PRAK_SCH.COPAY_CLAIMS
"""
print(f"📝 SQL:\n{q1_sql.strip()}")
df1 = run_sql(q1_sql)
print(f"\n📊 Result:")
display(df1)

# --- Q2: Show total Copay spend by brand ---
print("\n" + "=" * 70)
print("❓ Q2: Show total Copay spend by brand.")
q2_sql = """
SELECT
    BRAND,
    SUM(COPAY_AMOUNT) AS TOTAL_COPAY_SPEND
FROM PRAK_DB.PRAK_SCH.COPAY_CLAIMS
GROUP BY BRAND
ORDER BY TOTAL_COPAY_SPEND DESC
"""
print(f"📝 SQL:\n{q2_sql.strip()}")
df2 = run_sql(q2_sql)
print(f"\n📊 Result:")
display(df2)

# --- Q3: How many unique patients used Copay? ---
print("\n" + "=" * 70)
print("❓ Q3: How many unique patients used Copay?")
q3_sql = """
SELECT COUNT(DISTINCT PATIENT_ID) AS UNIQUE_PATIENTS
FROM PRAK_DB.PRAK_SCH.COPAY_CLAIMS
"""
print(f"📝 SQL:\n{q3_sql.strip()}")
df3 = run_sql(q3_sql)
print(f"\n📊 Result:")
display(df3)

# COMMAND ----------

# DBTITLE 1,Semantic View Structure Analysis
# =============================================================================
# 3. SEMANTIC VIEW QUERIES (Snowflake FACTS clause)
# =============================================================================
# The COPAY_SEMANTIC_VIEW exists but requires the FACTS clause syntax
# for aggregate metrics (TOTAL_COPAY_SPEND, CLAIM_COUNT, PATIENT_COUNT).

# --- Check semantic view structure ---
print("=" * 70)
print("Checking semantic view structure...")

# Describe the semantic view
cur.execute("DESCRIBE SEMANTIC VIEW PRAK_DB.PRAK_SCH.COPAY_SEMANTIC_VIEW")
desc = cur.fetchall()
print(f"Semantic view description ({len(desc)} rows):")

# Show the type of each expression (DIMENSION, METRIC, FACT)
types_seen = set()
for row in desc:
    obj_type = row[0]  # DIMENSION, METRIC, FACT, TABLE
    if obj_type and obj_type not in types_seen:
        types_seen.add(obj_type)
        print(f"  Object type found: {obj_type}")

# List all METRIC objects (these should be FACTs)
print("\nMETRIC objects (need to be converted to FACT):")
for row in desc:
    if row[0] == "METRIC":
        print(f"  {row[1]}: {row[3]} = {row[4]}")

print("\nDIMENSION objects (OK):")
for row in desc:
    if row[0] == "DIMENSION":
        print(f"  {row[1]}: {row[3]} = {row[4]}")

# COMMAND ----------

# DBTITLE 1,Fix Semantic View: metrics → facts
# =============================================================================
# FIX: Recreate semantic view with FACTS instead of METRICS
# =============================================================================
# The original YAML defined aggregates (SUM, COUNT) under `metrics:` which creates
# METRIC-type objects. Snowflake's FACTS clause requires FACT-type objects.
# Run the DDL below to fix the semantic view, then the FACTS queries will work.
#
# Uncomment and run the cur.execute() below to apply the fix:

fix_ddl = """
CREATE OR REPLACE SEMANTIC VIEW PRAK_DB.PRAK_SCH.COPAY_SEMANTIC_VIEW
  COMMENT = 'Semantic model for synthetic Copay analytics'
  TABLES (
    CLAIMS (
      COMMENT = 'Synthetic Copay claim-level data. One row represents one claim.'
      BASE_TABLE PRAK_DB.PRAK_SCH.COPAY_CLAIMS
      PRIMARY KEY (CLAIM_ID)
      DIMENSIONS (
        ACCUMULATOR BOOLEAN AS ACCUMULATOR_FLAG
          COMMENT 'Indicates accumulator-related patient claim',
        BRAND VARCHAR
          SYNONYMS ('product', 'brand name')
          COMMENT 'Brand associated with the Copay claim',
        CHANNEL VARCHAR
          COMMENT 'Copay enrollment or claim channel',
        CLAIM_DATE DATE
          SYNONYMS ('date', 'transaction date')
          COMMENT 'Date of the Copay claim',
        HCP VARCHAR AS HCP_ID
          COMMENT 'Healthcare provider identifier',
        MAXIMIZER BOOLEAN AS MAXIMIZER_FLAG
          COMMENT 'Indicates maximizer-related patient claim',
        PHARMACY VARCHAR AS PHARMACY_ID
          COMMENT 'Pharmacy identifier',
        STATE VARCHAR
          COMMENT 'Patient geography represented by state'
      )
      FACTS (
        CLAIM_COUNT AS COUNT(CLAIM_ID)
          COMMENT 'Number of Copay claims',
        PATIENT_COUNT AS COUNT(DISTINCT PATIENT_ID)
          COMMENT 'Number of unique patients',
        TOTAL_COPAY_SPEND AS SUM(COPAY_AMOUNT)
          COMMENT 'Total Copay redemption amount'
      )
      METRICS (
        SPEND_PER_PATIENT AS SUM(COPAY_AMOUNT) / NULLIF(COUNT(DISTINCT PATIENT_ID), 0)
          COMMENT 'Average Copay spend per unique patient'
      )
    )
  )
"""

print("=== DDL to fix the semantic view ===")
print(fix_ddl)
print("\n⏳ To apply: uncomment the cur.execute line below and run this cell.")
print("   After fixing, the FACTS clause queries will work with the semantic view.")

# --- Deploy: Apply the fix ---
cur.execute(fix_ddl)
print("✅ Semantic view deployed! FACTS clause queries should now work.")

# --- Verify: Query the semantic view with FACTS clause ---
print("\n" + "=" * 70)
print("🔍 Verification: Querying semantic view with FACTS clause...")

# Q1 via semantic view: Total Copay spend
try:
    cur.execute("""
    SELECT TOTAL_COPAY_SPEND
    FROM PRAK_DB.PRAK_SCH.COPAY_SEMANTIC_VIEW
    FACTS (TOTAL_COPAY_SPEND)
    """)
    df_sv1 = cur.fetch_pandas_all()
    print("✅ Q1 (Total Copay spend via semantic view):")
    display(df_sv1)
except Exception as e:
    print(f"❌ Q1 semantic view error: {e}")

# Q2 via semantic view: Total Copay spend by brand
try:
    cur.execute("""
    SELECT BRAND, TOTAL_COPAY_SPEND
    FROM PRAK_DB.PRAK_SCH.COPAY_SEMANTIC_VIEW
    FACTS (TOTAL_COPAY_SPEND)
    GROUP BY BRAND
    ORDER BY TOTAL_COPAY_SPEND DESC
    """)
    df_sv2 = cur.fetch_pandas_all()
    print("\n✅ Q2 (Total Copay spend by brand via semantic view):")
    display(df_sv2)
except Exception as e:
    print(f"\n❌ Q2 semantic view error: {e}")

# Q3 via semantic view: Unique patients
try:
    cur.execute("""
    SELECT PATIENT_COUNT
    FROM PRAK_DB.PRAK_SCH.COPAY_SEMANTIC_VIEW
    FACTS (PATIENT_COUNT)
    """)
    df_sv3 = cur.fetch_pandas_all()
    print("\n✅ Q3 (Unique patients via semantic view):")
    display(df_sv3)
except Exception as e:
    print(f"\n❌ Q3 semantic view error: {e}")

# COMMAND ----------

# DBTITLE 1,Cleanup & Close Connection
# =============================================================================
# 4. CLEANUP
# =============================================================================
cur.close()
conn.close()
print("✓ Snowflake connection closed.")