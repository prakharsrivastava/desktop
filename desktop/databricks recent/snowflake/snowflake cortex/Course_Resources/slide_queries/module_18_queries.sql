-- ============================================================
-- Course 1018 — Module 18
-- On-slide QUERIES reference  (9 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- Prove the Boundary Before You Ship It
-- ----------------------------------------------------------

USE ROLE cpg_analyst_role;
SELECT * FROM cpg_db.retail.customer_pii LIMIT 1;
-- Expect: insufficient privileges error
SHOW GRANTS TO ROLE cpg_analyst_role;


-- ----------------------------------------------------------
-- Debug — A Team Lost Cortex Access After a Role Change
-- ----------------------------------------------------------

-- Symptom: category managers hit 'insufficient privileges' on AI_CLASSIFY, no code changed
SHOW GRANTS TO ROLE cpg_category_mgr_role;
-- USAGE ON SNOWFLAKE.CORTEX_USER is listed, looks fine

-- Diagnosis: the direct grant is intact, but this role no longer
-- INHERITS it via a parent role that was cleaned up earlier that week
SHOW GRANTS OF ROLE cortex_platform_role;
-- cpg_category_mgr_role is missing from the grantee list

-- Fix: re-grant the role membership, then verify
GRANT ROLE cortex_platform_role TO ROLE cpg_category_mgr_role;
USE ROLE cpg_category_mgr_role;
SELECT AI_CLASSIFY('test row', ['a','b']); -- confirms access restored


-- ----------------------------------------------------------
-- The Grant Checklist for a Brand-New Team Member
-- ----------------------------------------------------------

-- Tier 1: everyday Cortex access (AI_COMPLETE, AI_CLASSIFY, AI_REDACT)
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE cpg_analyst_role;
-- some accounts instead expose this as SNOWFLAKE.AI_FUNCTIONS_USER --
-- confirm the current exact grant in Snowflake's docs before running in prod

-- Tier 2: fine-tuning needs TWO MORE grants on top of Tier 1
GRANT USAGE ON DATABASE cpg_training_db TO ROLE cpg_ml_role;
GRANT CREATE MODEL ON SCHEMA cpg_training_db.models TO ROLE cpg_ml_role;
-- OWNERSHIP on that schema works too, if the role fully manages its models


-- ----------------------------------------------------------
-- Find Out Who's Spending What
-- ----------------------------------------------------------

SELECT function_name,
       DATE_TRUNC('day', start_time) AS day,
       SUM(credits) AS cortex_credits
FROM snowflake.account_usage.cortex_ai_functions_usage_history
WHERE start_time >= DATEADD(day, -30, CURRENT_DATE())
GROUP BY 1, 2
ORDER BY cortex_credits DESC;


-- ----------------------------------------------------------
-- The Warehouse-Level Backstop — and Its Limit
-- ----------------------------------------------------------

CREATE RESOURCE MONITOR cortex_guard
  WITH CREDIT_QUOTA = 500
  FREQUENCY = MONTHLY
  START_TIMESTAMP = IMMEDIATELY
  TRIGGERS
    ON 75 PERCENT DO NOTIFY
    ON 100 PERCENT DO SUSPEND;
ALTER WAREHOUSE cortex_wh SET RESOURCE_MONITOR = cortex_guard;


-- ----------------------------------------------------------
-- AI_REDACT in One Query
-- ----------------------------------------------------------

SELECT
    order_id,
    AI_REDACT(customer_name) AS masked_name,
    AI_REDACT(loyalty_id) AS masked_loyalty_id,
    AI_COMPLETE('llama3.1-70b',
      'Summarize this order for a support ticket: ' || review_text
    ) AS ticket_summary
FROM cpg_db.retail.orders
LIMIT 100;


-- ----------------------------------------------------------
-- Wiring the Masking Policy to the Table
-- ----------------------------------------------------------

CREATE MASKING POLICY mask_customer_name AS (val STRING) RETURNS STRING ->
  CASE
    WHEN CURRENT_ROLE() IN ('CPG_ANALYST_ROLE') THEN AI_REDACT(val)
    ELSE val
  END;

ALTER TABLE cpg_db.retail.orders
  MODIFY COLUMN customer_name
  SET MASKING POLICY mask_customer_name;


-- ----------------------------------------------------------
-- Guardrails on the Way Out, Not Just the Way In
-- ----------------------------------------------------------

-- Illustrative only — parameter name/availability is version-dependent,
-- confirm current syntax in docs
SELECT AI_COMPLETE(
    'llama3.1-70b',
    'Summarize this support ticket for the customer-facing portal: ' || review_text,
    {'guardrails': true}
) AS safe_summary
FROM cpg_db.retail.support_tickets;


-- ----------------------------------------------------------
-- One Query to Prove the Checklist Is Live
-- ----------------------------------------------------------

SHOW RESOURCE MONITORS;
SHOW MASKING POLICIES IN DATABASE cpg_db;
SELECT role_name, granted_on, privilege
FROM snowflake.account_usage.grants_to_roles
WHERE role_name ILIKE '%cortex%'
  AND deleted_on IS NULL;

