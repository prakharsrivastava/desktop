-- ==========================================================
-- Capstone Layer 5 — Orchestrating Agent + Governance Guardrails
-- Meridian Foods: one front door for Search, Analyst, Forecast, Anomaly
-- Pattern from: modules 15-16 (trade-promotion agent), module 18 (governance)
-- ==========================================================
--
-- The trade-promotion agent from modules 15-16 already proved the pattern:
-- one agent, several tools, one conversation. Meridian's agent gets five
-- tools instead of trade promotion's three: Search, Analyst, Forecast,
-- Anomaly, and a summarizer. Given a question, the agent decides which
-- tool answers it, calls it, and folds the result back into one response.
-- This is orchestration, not a bigger model — the underlying LLM calls
-- stay small and task-specific.
--
-- Illustrative only — exact object/DDL syntax is version-dependent, verify
-- current docs (current Cortex Agents are defined FROM SPECIFICATION
-- $$ <yaml> $$, not this simplified TOOLS=/INSTRUCTIONS= form used below
-- for teaching purposes).

-- TODO 1: register each tool once — search from Layer 2, the semantic
-- view from Layer 3, forecast + anomaly from Layer 4. Write INSTRUCTIONS
-- that route document/review questions to search, structured sales
-- questions to analyst, trend questions to forecast, and risk questions
-- to anomaly detection.
CREATE OR REPLACE AGENT meridian_intelligence_agent
    TOOLS = (
        CORTEX_SEARCH_SERVICE '___',    -- TODO: 'meridian_field_search'
        CORTEX_ANALYST_SERVICE '___',   -- TODO: 'meridian_sales_model'
        ML_FORECAST '___',              -- TODO: 'meridian_sku_forecast'
        ML_ANOMALY_DETECTION '___'      -- TODO: 'meridian_shipment_anomaly'
    )
    INSTRUCTIONS = '___';   -- TODO: routing string — document/review -> search,
                             --   structured sales -> analyst, trend -> forecast,
                             --   risk -> anomaly

-- ----------------------------------------------------------
-- Governance guardrails (module 18) — applied here because lesson_86
-- pairs the agent/app layer with governance: every table and function
-- this capstone touches gets the same checklist, not just one.
-- Roles: field reps see search and catalog data only; category managers
-- additionally see the semantic model and forecasts. Loyalty member PII
-- stays masked for every role except a narrow compliance role. A resource
-- monitor caps the warehouse compute credits behind a runaway job — the
-- direct cap on AI Function token spend itself is a separate, dedicated
-- AI resource budget, not this monitor.
-- ----------------------------------------------------------

-- TODO 2: mask loyalty_members.email for every role except the
-- compliance role.
CREATE MASKING POLICY mask_member_pii AS (val STRING) RETURNS STRING ->
    CASE WHEN CURRENT_ROLE() = '___' THEN val   -- TODO: 'COMPLIANCE_ANALYST'
         ELSE '___' END;                         -- TODO: masked placeholder, e.g. '***MASKED***'

ALTER TABLE loyalty_members
    MODIFY COLUMN email SET MASKING POLICY mask_member_pii;

-- TODO 3: cap the warehouse behind the agent's queries. NOTIFY at 90%,
-- SUSPEND (fail safe, stop submitting new queries) at 100%.
CREATE RESOURCE MONITOR meridian_wh_guard
    WITH CREDIT_QUOTA = ___        -- TODO: 500
    TRIGGERS ON ___ PERCENT DO NOTIFY    -- TODO: 90
             ON ___ PERCENT DO SUSPEND;  -- TODO: 100

ALTER WAREHOUSE meridian_wh SET RESOURCE_MONITOR = meridian_wh_guard;

-- Checkpoint 5 (README rubric): the agent registers all four tools with
-- INSTRUCTIONS that route by question type.
-- Checkpoint 6 (README rubric): a masking policy hides loyalty_members.email
-- from every role except compliance, and a resource monitor caps warehouse
-- spend with a NOTIFY/SUSPEND threshold pair.
