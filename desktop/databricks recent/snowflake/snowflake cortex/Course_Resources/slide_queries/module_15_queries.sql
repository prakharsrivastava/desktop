-- ============================================================
-- Course 1018 — Module 15
-- On-slide QUERIES reference  (3 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- Illustrating an Agent Definition
-- ----------------------------------------------------------

-- Illustrative only — exact object/DDL syntax is version-dependent, verify current docs
CREATE AGENT sales_ops_agent
  TOOLS = (cortex_search_returns_docs, cortex_analyst_sales_model, custom_pricing_proc)
  ORCHESTRATION_MODEL = 'default';


-- ----------------------------------------------------------
-- Illustrating a Container Runtime Footprint
-- ----------------------------------------------------------

-- Illustrative only — confirm exact deployment/runtime settings in current docs
CREATE COMPUTE POOL agent_app_pool
  INSTANCE_FAMILY = CPU_X64_S
  MIN_NODES = 1
  MAX_NODES = 2;
-- A Streamlit-in-Snowflake app service referencing this pool runs under
-- container runtime, so it can call the Cortex Agents APIs.
-- (A plain external app calling the Agents REST API directly does not need this pool.)


-- ----------------------------------------------------------
-- Illustrating the Account-Level Guardrail Toggle
-- ----------------------------------------------------------

-- AI_SETTINGS takes a nested YAML block, not a flat key
ALTER ACCOUNT SET AI_SETTINGS = $$
guardrails:
  advanced_prompt_injection:
    - enabled: true
$$;
-- Guardrails apply account-wide, screening every CoWork agent,
-- Cortex Agent, and Cortex Code response

