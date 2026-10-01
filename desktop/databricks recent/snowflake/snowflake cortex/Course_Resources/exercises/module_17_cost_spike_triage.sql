-- ==========================================================
-- Exercise 17.2 — Find the Cortex Cost Spike Before the Invoice Does
-- Course: 1018
-- Module 17
-- ==========================================================
--
-- Finance flags a Cortex line item 3.2x higher than last month, with no
-- incident logged and no error thrown — the workload just got quietly
-- more expensive. The current, non-deprecated source of truth for this
-- is SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AISQL_USAGE_HISTORY (billed in the
-- TOKEN_CREDITS column) — NOT the older CORTEX_FUNCTIONS_USAGE_HISTORY
-- view, which this course's own fact-audit flagged as deprecated. You'll
-- also make one row-killing batch job resilient with TRY_COMPLETE so one
-- bad review doesn't zero out a whole night's summaries.
--
-- Your task:
-- 1. Query CORTEX_AISQL_USAGE_HISTORY for a 30-day daily trend by function,
--    ordered so the biggest spike is easy to spot.
-- 2. Rewrite the risky nightly batch (AI_COMPLETE over every row, one bad
--    row aborts everything) to use TRY_COMPLETE so failures degrade to
--    NULL instead of killing the whole run.
-- 3. Add a pre-flight token-estimate query so you know the blast radius
--    before running the batch at full scale.
-- ==========================================================

-- YOUR CODE:

-- Step 1: 30-day daily credit trend by function
SELECT
    DATE(usage_time) AS usage_date,
    function_name,
    SUM(___) AS daily_credits
FROM snowflake.account_usage.___
WHERE usage_time >= DATEADD(day, -30, CURRENT_DATE())
GROUP BY 1, 2
ORDER BY 1;

-- Step 2: batch summary that survives one garbled row
SELECT product_id,
    ___('llama3.1-70b',
      'Summarize this customer review: ' || review_text
    ) AS summary
FROM customer_reviews
WHERE batch_date = CURRENT_DATE();

-- Downstream check: requeue only the rows that came back NULL
SELECT COUNT(*) FROM batch_output WHERE summary IS ___;

-- Step 3: estimate the blast radius before running at scale
SELECT SUM(___('llama3.1-70b', review_text)) AS total_tokens_estimate
FROM customer_reviews
WHERE batch_date = CURRENT_DATE();


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
-- Step 1: 30-day daily credit trend by function — the current view, not
-- the deprecated CORTEX_FUNCTIONS_USAGE_HISTORY
SELECT
    DATE(usage_time) AS usage_date,
    function_name,
    SUM(token_credits) AS daily_credits
FROM snowflake.account_usage.cortex_aisql_usage_history
WHERE usage_time >= DATEADD(day, -30, CURRENT_DATE())
GROUP BY 1, 2
ORDER BY 1;

-- Step 2: TRY_COMPLETE returns NULL for the row that fails instead of
-- raising — the other 49,999 rows still get summarized that night
SELECT product_id,
    TRY_COMPLETE('llama3.1-70b',
      'Summarize this customer review: ' || review_text
    ) AS summary
FROM customer_reviews
WHERE batch_date = CURRENT_DATE();

SELECT COUNT(*) FROM batch_output WHERE summary IS NULL;
-- requeue only the failed rows, don't silently drop them

-- Step 3: blast-radius estimate — a token is roughly 4 characters and the
-- count is model-specific; re-run count_tokens() if you change models
SELECT SUM(COUNT_TOKENS('llama3.1-70b', review_text)) AS total_tokens_estimate
FROM customer_reviews
WHERE batch_date = CURRENT_DATE();

-- Optional: a daily scheduled spend-threshold alert built on the same view
CREATE OR REPLACE TASK cortex_spend_alert
  WAREHOUSE = cpg_wh
  SCHEDULE = 'USING CRON 0 7 * * * UTC'
AS
  SELECT SUM(token_credits) AS daily_credits
  FROM snowflake.account_usage.cortex_aisql_usage_history
  WHERE DATE(usage_time) = DATEADD(day, -1, CURRENT_DATE())
  HAVING daily_credits > 100;
*/
