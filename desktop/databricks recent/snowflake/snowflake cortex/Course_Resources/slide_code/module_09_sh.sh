# ============================================================
# Course 1018 — Module 09
# On-slide SH reference  (1 snippets)
# Authentic code as shown on the course slides. Copy, run, experiment.
# ============================================================


# ----------------------------------------------------------
# Calling Analyst Directly with curl
# ----------------------------------------------------------

curl -X POST https://<account>.snowflakecomputing.com/api/v2/cortex/analyst/message \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "semantic_model_file": "@cpg_db.analytics.semantic_models/cpg_sales_model.yaml",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "Which 5 stores had the lowest sell-through last quarter?"}]}]
  }'

