# ============================================================
# 1018 — COMPLETE SH REFERENCE  (1 snippets)
# Every on-slide snippet across all modules. Ctrl+F by keyword or title.
# ============================================================



# ########################################################
# MODULE 09
# ########################################################

# --- Calling Analyst Directly with curl ---

curl -X POST https://<account>.snowflakecomputing.com/api/v2/cortex/analyst/message \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "semantic_model_file": "@cpg_db.analytics.semantic_models/cpg_sales_model.yaml",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "Which 5 stores had the lowest sell-through last quarter?"}]}]
  }'

