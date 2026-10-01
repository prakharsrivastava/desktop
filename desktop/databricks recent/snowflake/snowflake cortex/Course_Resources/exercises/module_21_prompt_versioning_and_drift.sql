-- ==========================================================
-- Exercise 21.1 — Version Every Prompt, Then Trend the Agreement Rate
-- Course: 1018
-- Module 21
-- ==========================================================
--
-- Module 21 opened with a team whose AI_SENTIMENT accuracy was 94% at
-- launch and nobody checked it again for 11 weeks. Without a prompt_version
-- column, a drift dip can never be pinned to a prompt change vs. a genuine
-- model shift. Here you'll stand up the prompt_registry table and adapt
-- the week-by-prompt_version agreement query to a different quality log —
-- a product-description QA table instead of the module's sentiment_qa_log.
--
-- Your task:
-- 1. Create `prompt_registry` with columns prompt_version, prompt_text,
--    created_at, and is_active — same shape as the module's table.
-- 2. Write the weekly trend query against `desc_qa_log` (columns:
--    checked_at, prompt_version, ai_label, human_label), grouping by
--    week and prompt_version, computing agreement_rate as the average of
--    a CASE WHEN ai_label = human_label THEN 1 ELSE 0 END.
-- 3. Order the result by week ascending.
-- ==========================================================

-- YOUR CODE:

CREATE TABLE prompt_registry (
    prompt_version ___,
    prompt_text ___,
    created_at ___,
    is_active ___
);

SELECT
    DATE_TRUNC('___', checked_at) AS wk,
    prompt_version,
    AVG(CASE WHEN ___ = ___ THEN 1 ELSE 0 END) AS agreement_rate
FROM ___
GROUP BY ___, ___
ORDER BY ___;

-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
CREATE TABLE prompt_registry (
    prompt_version STRING,
    prompt_text STRING,
    created_at TIMESTAMP_NTZ,
    is_active BOOLEAN
);

SELECT
    DATE_TRUNC('week', checked_at) AS wk,
    prompt_version,
    AVG(CASE WHEN ai_label = human_label THEN 1 ELSE 0 END) AS agreement_rate
FROM desc_qa_log
GROUP BY wk, prompt_version
ORDER BY wk;
*/


-- ==========================================================
-- Exercise 21.2 — Fix the Instant-Rollout Anti-Pattern
-- Course: 1018
-- Module 21
-- ==========================================================
--
-- The module's anti-pattern slide showed the tempting move: flip
-- `is_active` to TRUE for a new prompt_version with a straight UPDATE —
-- 100% of traffic, no rollback window, no comparison data. The fix is a
-- deterministic hash split so only a slice of traffic samples the
-- candidate first. Rewrite the anti-pattern into the safe routing query,
-- this time starting the candidate at a 10% slice instead of the module's
-- 5%, and log the agent_decision_log columns that must travel with every
-- routed row.
--
-- Your task:
-- 1. In a comment, name the anti-pattern statement (an UPDATE) and why it
--    fails.
-- 2. Write the HASH-based routing SELECT against `incoming_requests`,
--    routing 10% of customer_id values to 'variant_b' and the rest to
--    'variant_a'.
-- 3. Write the CREATE TABLE for `agent_decision_log` with all 8 columns
--    from the module: session_id, step_number, tool_called, tool_input,
--    tool_output, model_version, prompt_version, logged_at.
-- ==========================================================

-- YOUR CODE:

-- Anti-pattern (do NOT do this): ___

SELECT
    CASE WHEN MOD(ABS(HASH(customer_id)), 100) < ___
         THEN '___' ELSE '___' END AS variant,
    customer_id
FROM ___;

CREATE TABLE agent_decision_log (
    session_id ___,
    step_number ___,
    tool_called ___,
    tool_input ___,
    tool_output ___,
    model_version ___,
    prompt_version ___,
    logged_at ___
);

-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
-- Anti-pattern (do NOT do this):
-- UPDATE prompt_registry SET is_active = TRUE WHERE prompt_version = 'v2';
-- -- flips 100% of traffic instantly, no rollback window, no comparison data.

SELECT
    CASE WHEN MOD(ABS(HASH(customer_id)), 100) < 10
         THEN 'variant_b' ELSE 'variant_a' END AS variant,
    customer_id
FROM incoming_requests;

CREATE TABLE agent_decision_log (
    session_id STRING,
    step_number INT,
    tool_called STRING,
    tool_input VARIANT,
    tool_output VARIANT,
    model_version STRING,
    prompt_version STRING,
    logged_at TIMESTAMP_NTZ
);
*/
