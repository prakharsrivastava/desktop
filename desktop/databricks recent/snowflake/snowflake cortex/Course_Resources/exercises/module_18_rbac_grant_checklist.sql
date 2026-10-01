-- ==========================================================
-- Exercise 18.1 — Scope a New Team's Cortex Access and Prove the Boundary
-- Course: 1018
-- Module 18
-- ==========================================================
--
-- A CPG analyst asked for "Cortex access" and got full warehouse admin by
-- accident — six weeks later an ad-hoc DROP TABLE wiped a category-manager
-- dashboard. Role-per-job beats user-per-grant: map job function to
-- privilege scope, grant the CORTEX_USER database role at the role level,
-- and never trust a GRANT on faith — test it as the actual role before it
-- ships. This exercise builds the onboarding checklist for a brand-new
-- analyst role and proves the boundary holds.
--
-- Your task:
-- 1. Grant the everyday-Cortex database role to a new analyst role (the
--    Tier 1 grant every AI_COMPLETE/AI_CLASSIFY/AI_REDACT caller needs).
-- 2. Prove the boundary: switch to that role and confirm it CANNOT read
--    a raw PII table, then confirm what it actually has via SHOW GRANTS.
-- 3. Grant the two ADDITIONAL Tier 2 privileges a separate ML role needs
--    to fine-tune a model — Tier 1 alone does not cover this.
-- ==========================================================

-- YOUR CODE:

-- Step 1: Tier 1 — everyday Cortex access for the new analyst role
GRANT DATABASE ROLE ___.CORTEX_USER TO ROLE cpg_analyst_role;

-- Step 2: prove the boundary — this role should NOT see raw PII
USE ROLE ___;
SELECT * FROM cpg_db.retail.customer_pii LIMIT 1;
-- Expect: insufficient privileges error
SHOW GRANTS TO ROLE ___;

-- Step 3: Tier 2 — fine-tuning needs TWO MORE grants on top of Tier 1
GRANT USAGE ON DATABASE cpg_training_db TO ROLE ___;
GRANT ___ MODEL ON SCHEMA cpg_training_db.models TO ROLE cpg_ml_role;


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
-- Step 1: Tier 1 — everyday Cortex access (AI_COMPLETE, AI_CLASSIFY, AI_REDACT)
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE cpg_analyst_role;
-- Some accounts instead expose this as SNOWFLAKE.AI_FUNCTIONS_USER —
-- confirm the current exact grant in Snowflake's docs before running in prod

-- Step 2: never trust a GRANT statement on faith — test it as the role
USE ROLE cpg_analyst_role;
SELECT * FROM cpg_db.retail.customer_pii LIMIT 1;
-- Expect: insufficient privileges error — an error here is a PASS
SHOW GRANTS TO ROLE cpg_analyst_role;

-- Step 3: Tier 2 — fine-tuning needs two MORE grants; the general Cortex
-- role alone will not let a role create a tuned model
GRANT USAGE ON DATABASE cpg_training_db TO ROLE cpg_ml_role;
GRANT CREATE MODEL ON SCHEMA cpg_training_db.models TO ROLE cpg_ml_role;
-- OWNERSHIP on that schema works too, if the role should fully manage
-- the models it creates
*/
