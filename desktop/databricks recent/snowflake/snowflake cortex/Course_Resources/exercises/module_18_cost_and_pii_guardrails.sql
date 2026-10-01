-- ==========================================================
-- Exercise 18.2 — Cost Guardrail, PII Redaction, and the Pre-Launch Checklist
-- Course: 1018
-- Module 18
-- ==========================================================
--
-- A misconfigured retry loop once burned roughly 4,000 credits over one
-- unattended weekend, and a well-meaning analyst once piped a raw
-- customer_loyalty_id straight into AI_COMPLETE, where it echoed back
-- inside a shareable summary export. Production Cortex needs three
-- controls working together: a resource-monitor backstop on warehouse
-- compute, AI_REDACT + a masking policy so PII never reaches a model
-- unmasked, and a one-query check that proves both are actually live
-- before you hand Cortex to the rest of the company.
--
-- Your task:
-- 1. Create a resource monitor with a 500-credit monthly quota that
--    notifies at 75% and suspends the warehouse at 100%, then attach it.
-- 2. Redact the sensitive columns BEFORE they ever reach AI_COMPLETE in
--    a single query, and wire a masking policy onto the table itself so
--    the redaction happens automatically for a scoped role.
-- 3. Write the one-query pre-launch checklist that confirms a resource
--    monitor and a masking policy both actually exist before go-live.
-- ==========================================================

-- YOUR CODE:

-- Step 1: warehouse-level cost backstop
CREATE RESOURCE MONITOR cortex_guard
  WITH CREDIT_QUOTA = ___
  FREQUENCY = MONTHLY
  START_TIMESTAMP = IMMEDIATELY
  TRIGGERS
    ON ___ PERCENT DO NOTIFY
    ON ___ PERCENT DO SUSPEND;
ALTER WAREHOUSE cortex_wh SET RESOURCE_MONITOR = ___;

-- Step 2a: redact first, generate second — never the other order
SELECT
    order_id,
    ___(customer_name) AS masked_name,
    ___(loyalty_id) AS masked_loyalty_id,
    AI_COMPLETE('llama3.1-70b',
      'Summarize this order for a support ticket: ' || review_text
    ) AS ticket_summary
FROM cpg_db.retail.orders
LIMIT 100;

-- Step 2b: make it automatic for the scoped analyst role — no analyst
-- has to remember to call AI_REDACT themselves
CREATE MASKING POLICY mask_customer_name AS (val STRING) RETURNS STRING ->
  CASE
    WHEN CURRENT_ROLE() IN (___) THEN AI_REDACT(val)
    ELSE val
  END;

ALTER TABLE cpg_db.retail.orders
  MODIFY COLUMN customer_name
  SET MASKING POLICY ___;

-- Step 3: prove the checklist is live before go-live
SHOW ___ MONITORS;
SHOW MASKING POLICIES IN DATABASE cpg_db;
SELECT role_name, granted_on, privilege
FROM snowflake.account_usage.grants_to_roles
WHERE role_name ILIKE '%cortex%'
  AND deleted_on IS NULL;


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
-- Step 1: a monitor without a suspend trigger is just an expensive
-- notification — SUSPEND stops new queries but lets running ones finish
CREATE RESOURCE MONITOR cortex_guard
  WITH CREDIT_QUOTA = 500
  FREQUENCY = MONTHLY
  START_TIMESTAMP = IMMEDIATELY
  TRIGGERS
    ON 75 PERCENT DO NOTIFY
    ON 100 PERCENT DO SUSPEND;
ALTER WAREHOUSE cortex_wh SET RESOURCE_MONITOR = cortex_guard;
-- Note: this caps warehouse COMPUTE credits. It is an indirect lever on
-- AI spend — the direct cap on AI Function TOKEN credits themselves is
-- a dedicated AI resource budget (tag-based), not a warehouse monitor.

-- Step 2a: AI_REDACT runs as a transformation step, before the model
-- ever sees the row — the model only ever receives the redacted value
SELECT
    order_id,
    AI_REDACT(customer_name) AS masked_name,
    AI_REDACT(loyalty_id) AS masked_loyalty_id,
    AI_COMPLETE('llama3.1-70b',
      'Summarize this order for a support ticket: ' || review_text
    ) AS ticket_summary
FROM cpg_db.retail.orders
LIMIT 100;

-- Step 2b: a masking policy applies automatically, Cortex call or not
CREATE MASKING POLICY mask_customer_name AS (val STRING) RETURNS STRING ->
  CASE
    WHEN CURRENT_ROLE() IN ('CPG_ANALYST_ROLE') THEN AI_REDACT(val)
    ELSE val
  END;

ALTER TABLE cpg_db.retail.orders
  MODIFY COLUMN customer_name
  SET MASKING POLICY mask_customer_name;
-- Test the policy with two roles side by side before trusting it in prod

-- Step 3: one query to prove the checklist is live
SHOW RESOURCE MONITORS;
SHOW MASKING POLICIES IN DATABASE cpg_db;
SELECT role_name, granted_on, privilege
FROM snowflake.account_usage.grants_to_roles
WHERE role_name ILIKE '%cortex%'
  AND deleted_on IS NULL;
-- No resource-monitor row for a Cortex warehouse = the cost guardrail
-- isn't live yet. No masking-policy row for customer_name = the PII gap
-- is still open.

-- Optional, output-side layer (illustrative only — parameter name and
-- availability are version-dependent; confirm current syntax in docs
-- before relying on it): a completion-call guardrail screens the MODEL'S
-- OWN response before it reaches a user, on top of the input-side
-- AI_REDACT/masking above.
-- SELECT AI_COMPLETE(
--     'llama3.1-70b',
--     'Summarize this support ticket for the customer-facing portal: ' || review_text,
--     {'guardrails': true}
-- ) AS safe_summary
-- FROM cpg_db.retail.support_tickets;
*/
