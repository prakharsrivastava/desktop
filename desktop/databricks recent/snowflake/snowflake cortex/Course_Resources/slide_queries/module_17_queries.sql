-- ============================================================
-- Course 1018 — Module 17
-- On-slide QUERIES reference  (10 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- Anti-Pattern First — Unbounded History Concatenation
-- ----------------------------------------------------------

-- ANTI-PATTERN — unbounded history concatenation, no cap
SELECT AI_COMPLETE(
    'llama3.1-70b',
    ARRAY_TO_STRING(ARRAY_AGG(msg), '\n') || current_question
)
FROM chat_history
WHERE session_id = ?;
-- No LIMIT, no windowing, no truncation


-- ----------------------------------------------------------
-- The Fix — Window, Summarize, Then Complete
-- ----------------------------------------------------------

SELECT AI_COMPLETE(
    'llama3.1-70b',
    AI_SUMMARIZE_AGG(msg) || current_question
)
FROM (
    SELECT msg
    FROM chat_history
    WHERE session_id = ?
    ORDER BY created_at DESC
    LIMIT 10
);


-- ----------------------------------------------------------
-- Running the Checks
-- ----------------------------------------------------------

SHOW GRANTS TO ROLE prod_service_role;

SELECT CURRENT_ROLE(), CURRENT_WAREHOUSE();

GRANT DATABASE ROLE snowflake.cortex_user
  TO ROLE prod_service_role;


-- ----------------------------------------------------------
-- The Fix — Update the Model and Add a Drift Check
-- ----------------------------------------------------------

-- Update the semantic model column reference
-- (gross_revenue -> net_revenue), then add a
-- scheduled drift-check task:
CREATE OR REPLACE TASK check_semantic_model_drift
  WAREHOUSE = cpg_wh
  SCHEDULE = 'USING CRON 0 6 * * 1 UTC'
AS
  SELECT column_name
  FROM information_schema.columns
  WHERE table_name = 'FINANCE_SUMMARY'
    AND column_name = 'GROSS_REVENUE';
-- Zero rows returned = the model's stale reference is confirmed


-- ----------------------------------------------------------
-- The Fix — Tighten the Lag Budget
-- ----------------------------------------------------------

ALTER CORTEX SEARCH SERVICE product_search
  SET target_lag = '1 hour';

SELECT *
FROM snowflake.account_usage.cortex_search_daily_usage_history
WHERE service_name = 'product_search'
ORDER BY usage_date DESC;


-- ----------------------------------------------------------
-- The Fix — Retrain With Seasonal Coverage
-- ----------------------------------------------------------

-- Retrain on a window spanning a full seasonal cycle
CALL anomaly_model!DETECT_ANOMALIES(
  INPUT_DATA => TABLE(sales_view),
  SERIES_COLNAME => 'sku',
  TIMESTAMP_COLNAME => 'sale_date',
  TARGET_COLNAME => 'units_sold',
  CONFIG_OBJECT => {'prediction_interval': 0.95}
);
-- Add a quarterly retrain schedule — not a one-time train-and-forget


-- ----------------------------------------------------------
-- Finding the Spike
-- ----------------------------------------------------------

SELECT
    DATE(usage_time) AS usage_date,
    function_name,
    SUM(token_credits)
FROM snowflake.account_usage.cortex_aisql_usage_history
WHERE usage_time >= DATEADD(day, -30, CURRENT_DATE())
GROUP BY 1, 2
ORDER BY 1;


-- ----------------------------------------------------------
-- The Fix — Watch Before the Invoice Does
-- ----------------------------------------------------------

-- Right-size the model for the batch job's actual accuracy need,
-- then schedule a daily spend-threshold alert:
CREATE OR REPLACE TASK cortex_spend_alert
  WAREHOUSE = cpg_wh
  SCHEDULE = 'USING CRON 0 7 * * * UTC'
AS
  SELECT SUM(token_credits) AS daily_credits
  FROM snowflake.account_usage.cortex_aisql_usage_history
  WHERE DATE(usage_time) = DATEADD(day, -1, CURRENT_DATE())
  HAVING daily_credits > 100;
-- Require a cost-estimate note on any PR changing model size


-- ----------------------------------------------------------
-- Anti-Pattern First — One Bad Row Kills the Whole Batch
-- ----------------------------------------------------------

-- ANTI-PATTERN — no error isolation across the batch
SELECT
    product_id,
    AI_COMPLETE('llama3.1-70b',
      'Summarize this customer review: ' || review_text
    ) AS summary
FROM customer_reviews
WHERE batch_date = CURRENT_DATE();
-- Row 40,812 has garbled encoding from a scraped source --
-- the call throws, the WHOLE statement fails, zero rows
-- get a summary that night


-- ----------------------------------------------------------
-- The Fix — TRY_COMPLETE, NULLs, and the Blast Radius
-- ----------------------------------------------------------

SELECT product_id,
    TRY_COMPLETE('llama3.1-70b',
      'Summarize this customer review: ' || review_text
    ) AS summary
FROM customer_reviews
WHERE batch_date = CURRENT_DATE();

-- Downstream check: requeue only the failed rows
SELECT COUNT(*) FROM batch_output WHERE summary IS NULL;

-- Before running at scale, estimate the blast radius
SELECT SUM(COUNT_TOKENS('llama3.1-70b', review_text))
FROM customer_reviews
WHERE batch_date = CURRENT_DATE();

