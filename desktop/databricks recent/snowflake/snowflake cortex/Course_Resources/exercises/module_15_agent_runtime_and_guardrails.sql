-- ==========================================================
-- Exercise 15.2 — Define an Agent, Plan Its Runtime, Turn on Guardrails
-- Course: 1018
-- Module 15
-- ==========================================================
--
-- Priya's team built Cortex Search, Analyst, and a custom pricing tool
-- separately — business users didn't know which chatbox to open for which
-- question. Snowflake CoWork (rebranded from Snowflake Intelligence in
-- mid-2026) adds one orchestration layer that registers already-built tools
-- rather than re-implementing them. Separately: her agent worked fine in the
-- CoWork chat UI, but the SAME agent called from a Streamlit-in-Snowflake app
-- threw a runtime error — a Streamlit-in-Snowflake app calling the Cortex
-- Agents APIs needs container runtime, not standard warehouse runtime.
-- NOTE: every DDL/config snippet below is illustrative — exact object and
-- parameter syntax is version-dependent; verify against current Snowflake
-- docs before running against a real account.
--
-- Your task:
-- 1. Define an agent object that references 3 already-registered tools.
-- 2. Provision a compute pool sized for a Streamlit-in-Snowflake app that
--    will call the Cortex Agents APIs (container runtime, not a warehouse).
-- 3. Turn on account-level response guardrails so every CoWork agent, Cortex
--    Agent, and Cortex Code response gets screened before delivery.
-- ==========================================================

-- YOUR CODE:

-- Illustrative only — exact object/DDL syntax is version-dependent, verify current docs
CREATE AGENT ___
  TOOLS = (___, ___, ___)
  ORCHESTRATION_MODEL = '___';

-- Illustrative only — confirm exact deployment/runtime settings in current docs
CREATE COMPUTE POOL ___
  INSTANCE_FAMILY = ___
  MIN_NODES = ___
  MAX_NODES = ___;

-- AI_SETTINGS takes a nested YAML block, not a flat key
ALTER ACCOUNT SET AI_SETTINGS = $$
guardrails:
  ___:
    - enabled: ___
$$;


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
-- Illustrative only — exact object/DDL syntax is version-dependent, verify current docs
CREATE AGENT sales_ops_agent
  TOOLS = (cortex_search_returns_docs, cortex_analyst_sales_model, custom_pricing_proc)
  ORCHESTRATION_MODEL = 'default';
-- An agent object references a list of already-registered tools —
-- registration, not re-implementation. Each tool keeps its own logic.

-- Illustrative only — confirm exact deployment/runtime settings in current docs
CREATE COMPUTE POOL agent_app_pool
  INSTANCE_FAMILY = CPU_X64_S
  MIN_NODES = 1
  MAX_NODES = 2;
-- A Streamlit-in-Snowflake app service referencing this pool runs under
-- container runtime, so it can call the Cortex Agents APIs — separate from
-- the warehouse used for Search/Analyst calls. (A plain external app calling
-- the Agents REST API directly over HTTPS has no such runtime requirement.)

-- AI_SETTINGS takes a nested YAML block, not a flat key
ALTER ACCOUNT SET AI_SETTINGS = $$
guardrails:
  advanced_prompt_injection:
    - enabled: true
$$;
-- Guardrails are configured ONCE at the account level, not toggled per
-- individual agent — once enabled, every agent response across the account
-- is screened before delivery. Cortex AI Guardrails protects against prompt
-- injection, jailbreak attempts, and zero-day attacks; a blocked response
-- returns a safe fallback message, not silence or a crash. This is distinct
-- from the older COMPLETE(..., guardrails=>TRUE) completion-call filter,
-- which screens generated text for harmful-content categories at the
-- individual completion level.
*/
