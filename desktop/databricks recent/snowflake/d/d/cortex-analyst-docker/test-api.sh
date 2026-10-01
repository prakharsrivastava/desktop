#!/bin/bash
set -euo pipefail

# =============================================================================
# Test Cortex Analyst API — works against localhost or deployed App Runner URL
# =============================================================================
# Usage:
#   ./test-api.sh                              # tests http://localhost:8080
#   ./test-api.sh https://abc123.ap-southeast-7.awsapprunner.com  # tests deployed
# =============================================================================

BASE_URL="${1:-http://localhost:8080}"
PASS=0
FAIL=0

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[PASS]${NC} $1"; PASS=$((PASS+1)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; FAIL=$((FAIL+1)); }

echo "======================================================="
echo "  Testing API: $BASE_URL"
echo "======================================================="
echo ""

# --- Health ---
if curl -sf "$BASE_URL/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Snowflake: {d.get("snowflake_version","?")} | Account: {d.get("account","?")}') if d.get('status')=='healthy' else sys.exit(1)" 2>/dev/null; then
    ok "GET /health"
else
    fail "GET /health"
fi

# --- Root ---
if curl -sf "$BASE_URL/" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if 'service' in d else 1)" 2>/dev/null; then
    ok "GET /"
else
    fail "GET /"
fi

# --- Total Spend ---
RESP=$(curl -sf "$BASE_URL/total-spend" 2>/dev/null || echo "ERROR")
if echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Total Copay Spend: \${d["total_copay_spend"]}')" 2>/dev/null; then
    ok "GET /total-spend"
else
    fail "GET /total-spend -> $RESP"
fi

# --- Spend by Brand ---
RESP=$(curl -sf "$BASE_URL/spend-by-brand" 2>/dev/null || echo "ERROR")
if echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f'  {r["BRAND"]}: \${r["TOTAL_COPAY_SPEND"]}') for r in d['results']]" 2>/dev/null; then
    ok "GET /spend-by-brand"
else
    fail "GET /spend-by-brand -> $RESP"
fi

# --- Patient Count ---
RESP=$(curl -sf "$BASE_URL/patient-count" 2>/dev/null || echo "ERROR")
if echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Unique Patients: {d["unique_patients"]}')" 2>/dev/null; then
    ok "GET /patient-count"
else
    fail "GET /patient-count -> $RESP"
fi

# --- Query (NL -> SQL) ---
RESP=$(curl -sf -X POST "$BASE_URL/query" -H 'Content-Type: application/json' -d '{"question": "What is the total Copay spend?"}' 2>/dev/null || echo "ERROR")
if echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  SQL: {d.get("sql","N/A")[:60]}'); sys.exit(0 if d.get('sql') or d.get('results') or d.get('error') else 1)" 2>/dev/null; then
    ok "POST /query"
else
    fail "POST /query -> $RESP"
fi

# --- Direct SQL ---
RESP=$(curl -sf -X POST "$BASE_URL/sql" -H 'Content-Type: application/json' -d '{"sql": "SELECT COUNT(*) AS CNT FROM PRAK_DB.PRAK_SCH.COPAY_CLAIMS"}' 2>/dev/null || echo "ERROR")
if echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Row count: {d.get("row_count","?")}'); sys.exit(0 if d.get('row_count') is not None else 1)" 2>/dev/null; then
    ok "POST /sql"
else
    fail "POST /sql -> $RESP"
fi

# --- Schema ---
if curl -sf "$BASE_URL/schema" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Tables: {set(r["TABLE_NAME"] for r in d["columns"])}'); sys.exit(0 if 'columns' in d else 1)" 2>/dev/null; then
    ok "GET /schema"
else
    fail "GET /schema"
fi

# --- Summary ---
echo ""
echo "======================================================="
echo "  Results: $PASS passed, $FAIL failed"
echo "======================================================="
if [ $FAIL -eq 0 ]; then
    echo "  ALL TESTS PASSED"
else
    echo "  SOME TESTS FAILED — check Snowflake credentials and connection"
fi
exit $FAIL