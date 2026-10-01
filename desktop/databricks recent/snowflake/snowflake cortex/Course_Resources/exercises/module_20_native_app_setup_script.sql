-- ==========================================================
-- Exercise 20.1 — Package the Forecast Logic as a Native App, Not a Grant
-- Course: 1018
-- Module 20
-- ==========================================================
--
-- Foodservice and international BU leads wanted the retail-ops Cortex
-- forecast, so retail-ops took the fast shortcut: grant their roles
-- direct SELECT on the raw sales tables. That hands every BU the full
-- schema, every column, every historical row — far more than the
-- forecast logic ever touches, with no versioning and no audit trail.
-- The Native App Framework fixes this: package the forecast logic once,
-- let each BU install a governed app instance that only touches what
-- they explicitly grant it, and never inherits blanket account privileges.
--
-- Your task:
-- 1. Write the setup script that runs once per install: an application
--    role, a versioned schema, and a procedure that calls a Cortex
--    function against the consumer's own granted table.
-- 2. Grant USAGE on that procedure to the app's own application role
--    (not to the consumer's account-level roles).
-- 3. List the three things the consumer account must do BEFORE the app
--    can touch any data (per the packaging checklist) — as a comment.
-- ==========================================================

-- YOUR CODE:

-- Step 1: setup script — runs once per install in the consumer account
CREATE APPLICATION ROLE IF NOT EXISTS ___;
CREATE OR ALTER VERSIONED SCHEMA ___;

CREATE OR REPLACE PROCEDURE core.run_forecast(input_table STRING)
  RETURNS STRING
  LANGUAGE SQL
  AS
  $$
    -- calls a Cortex function against the consumer's own granted table
    SELECT AI_COMPLETE('___', 'Summarize demand trend for ' || :input_table);
  $$;

-- Step 2: scope the procedure to the app's own role, not a blanket grant
GRANT ___ ON PROCEDURE core.run_forecast(STRING) TO APPLICATION ROLE app_public;

-- Step 3: (comment) what the CONSUMER must do before the app can touch data
-- 1. ___
-- 2. ___
-- 3. ___


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
-- Step 1: a setup script is plain SQL executed once at install time —
-- it defines exactly what objects get created on install, nothing implicit
CREATE APPLICATION ROLE IF NOT EXISTS app_public;
CREATE OR ALTER VERSIONED SCHEMA core;

CREATE OR REPLACE PROCEDURE core.run_forecast(input_table STRING)
  RETURNS STRING
  LANGUAGE SQL
  AS
  $$
    -- calls a Cortex function against the consumer's own granted table
    SELECT AI_COMPLETE('llama3-70b', 'Summarize demand trend for ' || :input_table);
  $$;

-- Step 2: the app declares its own roles, schema, and grants — it never
-- inherits blanket account-level privileges automatically
GRANT USAGE ON PROCEDURE core.run_forecast(STRING) TO APPLICATION ROLE app_public;
-- Exact grant and versioning syntax should be verified against current
-- docs before shipping.

-- Step 3: what the consumer account must do before the app touches data
-- 1. Install the application package as a private, running app instance
--    in their own account (no data physically moves; the app's logic
--    runs inside the consumer's account).
-- 2. Explicitly GRANT the app instance access to their own local data —
--    the app never inherits blanket account privileges automatically.
-- 3. Confirm the app's requested privileges (via its manifest) match what
--    was actually granted — same least-privilege discipline as module_04,
--    applied at app-install time.
*/
