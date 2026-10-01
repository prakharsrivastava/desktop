# =============================================================================
# Cortex Analyst API — FastAPI Web Service
# =============================================================================
# Wraps the Snowflake Cortex Analyst REST API as a deployable web service.
# Deploy via Docker to AWS (ECS Fargate, App Runner, or EKS).
#
# Endpoints:
#   GET  /health       — health check
#   POST /query        — natural language → SQL → results
#   POST /sql           — direct SQL against COPAY_CLAIMS
#   GET  /schema       — list tables and columns
# =============================================================================

import os
import json
import time
import logging
import threading
import snowflake.connector
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Logging (production-grade)
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("cortex-analyst")

# Simple TTL cache for pre-built queries (avoids redundant Snowflake calls)
_CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", "60"))
_cache: dict = {}
_cache_lock = threading.Lock()

def cache_get(key):
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (time.time() - entry[0]) < _CACHE_TTL:
            logger.info(f"Cache HIT: {key}")
            return entry[1]
        logger.info(f"Cache MISS: {key}")
        return None

def cache_set(key, value):
    with _cache_lock:
        _cache[key] = (time.time(), value)

# ---------------------------------------------------------------------------
# Configuration (from environment variables — never hardcode in Docker)
# ---------------------------------------------------------------------------
SF_USER      = os.environ.get("SF_USER", "")
SF_PASSWORD  = os.environ.get("SF_PASSWORD", "")
SF_ACCOUNT   = os.environ.get("SF_ACCOUNT", "")
SF_WAREHOUSE = os.environ.get("SF_WAREHOUSE", "COMPUTE_WH")
SF_DATABASE  = os.environ.get("SF_DATABASE", "PRAK_DB")
SF_SCHEMA    = os.environ.get("SF_SCHEMA", "PRAK_SCH")

SEMANTIC_VIEW = f"{SF_DATABASE}.{SF_SCHEMA}.COPAY_SEMANTIC_VIEW"
CLAIMS_TABLE  = f"{SF_DATABASE}.{SF_SCHEMA}.COPAY_CLAIMS"

app = FastAPI(
    title="Cortex Analyst API - Copay",
    description="Natural-language-to-SQL via Snowflake Cortex Analyst",
    version="1.1.0",
)

# ---------------------------------------------------------------------------
# Snowflake connection (thread-safe, with timeouts and session keep-alive)
# ---------------------------------------------------------------------------
_conn = None
_conn_lock = threading.Lock()
SF_QUERY_TIMEOUT = int(os.environ.get("SF_QUERY_TIMEOUT", "30"))

def get_connection():
    global _conn
    with _conn_lock:
        if _conn is not None:
            try:
                cur = _conn.cursor()
                cur.execute("SELECT 1", timeout=5)
                cur.close()
                return _conn, _conn.cursor()
            except Exception:
                logger.warning("Connection lost, reconnecting...")
                try:
                    _conn.close()
                except Exception:
                    pass
                _conn = None

        logger.info(f"Connecting to Snowflake {SF_ACCOUNT}...")
        _conn = snowflake.connector.connect(
            user=SF_USER,
            password=SF_PASSWORD,
            account=SF_ACCOUNT,
            warehouse=SF_WAREHOUSE,
            database=SF_DATABASE,
            schema=SF_SCHEMA,
            client_session_keep_alive=True,
            connect_timeout=10,
            network_timeout=30,
        )
        logger.info("Snowflake connected.")
        return _conn, _conn.cursor()

def run_sql(sql: str, timeout: int = None) -> list[dict]:
    """Execute SQL and return rows as a list of dicts."""
    t = timeout or SF_QUERY_TIMEOUT
    _, cur = get_connection()
    cur.execute(sql, timeout=t)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]

# ---------------------------------------------------------------------------
# Cortex Analyst API caller
# ---------------------------------------------------------------------------
ANALYST_URL = f"https://{SF_ACCOUNT}.snowflakecomputing.com/api/v2/cortex/analyst/messages"

def ask_cortex_analyst(question: str) -> dict:
    """
    Send a natural-language question to Cortex Analyst via REST API.
    Returns generated SQL, explanation text, and query results.
    """
    try:
        conn, _ = get_connection()
    except Exception as e:
        return {"question": question, "error": f"Snowflake connection failed: {e}"}

    payload = {
        "messages": [{"role": "user", "content": question}],
        "semantic_view": SEMANTIC_VIEW,
    }
    body_str = json.dumps(payload)

    api_path = "/api/v2/cortex/analyst/messages"
    try:
        response = conn.rest.request(url=api_path, body=body_str, method="post", timeout=120)
    except Exception as e:
        return {"question": question, "error": f"Cortex Analyst API request failed: {e}"}

    raw = response.read() if hasattr(response, "read") else response
    status = response.status if hasattr(response, "status") else 200

    if status != 200:
        return {
            "question": question,
            "error": f"Cortex Analyst API returned {status}",
            "details": str(raw)[:500],
        }

    result = json.loads(raw) if isinstance(raw, (str, bytes)) else raw

    generated_sql = None
    explanation = ""
    for block in result.get("message", {}).get("content", []):
        if block.get("type") == "sql":
            generated_sql = block.get("statement", "")
        elif block.get("type") == "text":
            explanation += block.get("text_value", "")

    results = None
    if generated_sql:
        try:
            results = run_sql(generated_sql)
        except Exception as e:
            results = {"error": str(e)}

    return {
        "question": question,
        "sql": generated_sql,
        "explanation": explanation,
        "results": results,
        "raw_response": result,
    }

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str

class SqlRequest(BaseModel):
    sql: str

class QueryResponse(BaseModel):
    question: str
    sql: Optional[str] = None
    explanation: Optional[str] = None
    results: Optional[list | dict] = None
    error: Optional[str] = None

class SqlResponse(BaseModel):
    sql: str
    results: list[dict]
    row_count: int

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Health check — verifies Snowflake connectivity."""
    try:
        conn, cur = get_connection()
        cur.execute("SELECT CURRENT_VERSION(), CURRENT_ACCOUNT()")
        ver, acct = cur.fetchone()
        return {"status": "healthy", "snowflake_version": ver, "account": acct}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Snowflake connection failed: {e}")

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Natural-language → SQL → results via Cortex Analyst."""
    result = ask_cortex_analyst(req.question)
    return QueryResponse(
        question=result.get("question", req.question),
        sql=result.get("sql"),
        explanation=result.get("explanation"),
        results=result.get("results"),
        error=result.get("error"),
    )

@app.post("/sql", response_model=SqlResponse)
def direct_sql(req: SqlRequest):
    """Execute direct SQL against COPAY_CLAIMS (fallback when Cortex Analyst API is unavailable)."""
    try:
        rows = run_sql(req.sql)
        return SqlResponse(sql=req.sql, results=rows, row_count=len(rows))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL execution failed: {e}")

@app.get("/schema")
def schema():
    """List tables and columns in the configured Snowflake schema."""
    try:
        _, cur = get_connection()
        cur.execute(f"""
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
            FROM {SF_DATABASE}.INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{SF_SCHEMA}'
            ORDER BY TABLE_NAME, ORDINAL_POSITION
        """)
        rows = [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]
        return {"schema": SF_SCHEMA, "columns": rows}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Schema query failed: {e}")

# ---------------------------------------------------------------------------
# Pre-built queries (no Cortex Analyst API needed)
# ---------------------------------------------------------------------------
@app.get("/total-spend")
def total_spend():
    """Total Copay spend (cached for CACHE_TTL_SECONDS)."""
    cached = cache_get("total-spend")
    if cached is not None:
        return cached
    try:
        rows = run_sql(f"SELECT SUM(COPAY_AMOUNT) AS TOTAL_COPAY_SPEND FROM {CLAIMS_TABLE}")
        result = {"total_copay_spend": rows[0]["TOTAL_COPAY_SPEND"]}
        cache_set("total-spend", result)
        return result
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Query failed: {e}")

@app.get("/spend-by-brand")
def spend_by_brand():
    """Total Copay spend by brand."""
    try:
        rows = run_sql(f"""
            SELECT BRAND, SUM(COPAY_AMOUNT) AS TOTAL_COPAY_SPEND
            FROM {CLAIMS_TABLE}
            GROUP BY BRAND
            ORDER BY TOTAL_COPAY_SPEND DESC
        """)
        return {"results": rows}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Query failed: {e}")

@app.get("/patient-count")
def patient_count():
    """Count of unique patients."""
    try:
        rows = run_sql(f"SELECT COUNT(DISTINCT PATIENT_ID) AS UNIQUE_PATIENTS FROM {CLAIMS_TABLE}")
        return {"unique_patients": rows[0]["UNIQUE_PATIENTS"]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Query failed: {e}")

@app.get("/")
def root():
    return {
        "service": "Cortex Analyst API — Copay",
        "endpoints": [
            "GET  /health",
            "POST /query   (body: {\"question\": \"...\"})",
            "POST /sql     (body: {\"sql\": \"SELECT ...\"})",
            "GET  /schema",
            "GET  /total-spend",
            "GET  /spend-by-brand",
            "GET  /patient-count",
        ],
    }

# ---------------------------------------------------------------------------
# Run with: uvicorn app:app --host 0.0.0.0 --port 8080
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)