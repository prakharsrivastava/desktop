-- ============================================================
-- Course 1018 — Module 02
-- On-slide QUERIES reference  (8 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- Anti-Pattern — Assuming ACCOUNTADMIN Means You're Covered
-- ----------------------------------------------------------

SELECT AI_COMPLETE('claude-sonnet-4-5', 'summarize this feedback');
-- works out of the box on a fresh account, any role


-- ----------------------------------------------------------
-- Granting the Cortex Usage Privilege to a Role
-- ----------------------------------------------------------

-- 1. Check whether the default PUBLIC grant is still present
SHOW GRANTS TO ROLE analytics_role;

-- 2. If governance revoked it, re-grant the Cortex usage privilege
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE analytics_role;

-- 3. If account-level USE AI FUNCTIONS was also revoked, re-grant it too
GRANT USE AI FUNCTIONS ON ACCOUNT TO ROLE analytics_role;


-- ----------------------------------------------------------
-- Checking and Enabling Cross-Region Inference
-- ----------------------------------------------------------

SHOW PARAMETERS LIKE 'CORTEX_ENABLED_CROSS_REGION' IN ACCOUNT;

ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'AWS_US';


-- ----------------------------------------------------------
-- Estimating Cost and Token Volume by Model and Warehouse
-- ----------------------------------------------------------

-- compare estimated cost and token volume by model and warehouse
SELECT model_name, warehouse_id,
       AVG(token_credits) AS avg_credits,
       SUM(tokens) AS total_tokens
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AISQL_USAGE_HISTORY
GROUP BY model_name, warehouse_id;


-- ----------------------------------------------------------
-- Anti-Pattern — Summarizing Raw, Unfiltered Tables at Full Volume
-- ----------------------------------------------------------

-- ANTI-PATTERN: no WHERE clause, no row limit
SELECT SNOWFLAKE.CORTEX.SUMMARIZE(review_text)
FROM all_reviews;

-- FIX: filter to the real analysis window first
SELECT SNOWFLAKE.CORTEX.SUMMARIZE(review_text)
FROM all_reviews
WHERE review_date >= DATEADD(month, -1, CURRENT_DATE());


-- ----------------------------------------------------------
-- Reference Card — A Monitoring Query for Ongoing Cost Control
-- ----------------------------------------------------------

SELECT function_name, model_name,
       SUM(token_credits) AS credits,
       COUNT(*) AS calls
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AISQL_USAGE_HISTORY
WHERE usage_time >= DATEADD(day, -7, CURRENT_TIMESTAMP())
GROUP BY function_name, model_name
ORDER BY credits DESC;


-- ----------------------------------------------------------
-- Anti-Pattern — Reaching for AI_COMPLETE for Everything
-- ----------------------------------------------------------

-- ANTI-PATTERN: reaching for AI_COMPLETE for everything
SELECT AI_COMPLETE('claude-sonnet-4-5',
  'What is the sentiment of: ' || review_text)
FROM reviews; -- 2M rows

-- FIX: purpose-built task function — cheaper, structured output
SELECT AI_SENTIMENT(review_text)
FROM reviews;


-- ----------------------------------------------------------
-- Case Study — Mapping a CPG Use Case to the Matrix
-- ----------------------------------------------------------

-- Q1: 'which SKUs underperformed in the Midwest last quarter?'
-- structured tables, plain English -> Cortex Analyst

-- Q2: 'find the clause on returns in these 4,000 retailer compliance PDFs'
-- unstructured docs, retrieval -> Cortex Search

-- Combined: reasoning across both Q1 and Q2 in one conversation
-- wrap both calls in Cortex Agents

