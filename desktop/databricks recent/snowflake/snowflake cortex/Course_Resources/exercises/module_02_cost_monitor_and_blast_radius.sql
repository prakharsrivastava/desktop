-- ==========================================================
-- Exercise 2.2 — Build a Cost Monitor and Cap the Blast Radius
-- Course: 1018
-- Module 02
-- ==========================================================
--
-- Every Cortex call is metered in SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AISQL_USAGE_HISTORY,
-- and the most common way to blow a budget is running a task function over an
-- entire raw table with no WHERE clause and no row limit. This exercise builds
-- the monitoring query a real CPG analytics team would bookmark, then applies
-- the same discipline (filter before you summarize) to a runaway query.
--
-- Your task:
-- 1. Write a query that shows, for the last 7 days, credits and call count
--    grouped by function and model, ordered by highest cost first.
-- 2. Take the unfiltered anti-pattern below and fix it so it only summarizes
--    the last month of reviews instead of the whole table.
-- ==========================================================

-- YOUR CODE:

-- 1. 7-day cost monitor
SELECT function_name, model_name,
       SUM(___) AS credits,
       COUNT(*) AS calls
FROM SNOWFLAKE.ACCOUNT_USAGE.___
WHERE usage_time >= DATEADD(day, -___, CURRENT_TIMESTAMP())
GROUP BY function_name, model_name
ORDER BY ___ DESC;

-- ANTI-PATTERN (do not ship this):
-- SELECT SNOWFLAKE.CORTEX.SUMMARIZE(review_text)
-- FROM all_reviews;

-- 2. FIX: filter to the real analysis window first
SELECT SNOWFLAKE.CORTEX.SUMMARIZE(review_text)
FROM all_reviews
WHERE review_date >= DATEADD(___, -1, ___());


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
-- 1. 7-day cost monitor
SELECT function_name, model_name,
       SUM(token_credits) AS credits,
       COUNT(*) AS calls
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AISQL_USAGE_HISTORY
WHERE usage_time >= DATEADD(day, -7, CURRENT_TIMESTAMP())
GROUP BY function_name, model_name
ORDER BY credits DESC;

-- 2. FIX: filter to the real analysis window first
SELECT SNOWFLAKE.CORTEX.SUMMARIZE(review_text)
FROM all_reviews
WHERE review_date >= DATEADD(month, -1, CURRENT_DATE());
*/
