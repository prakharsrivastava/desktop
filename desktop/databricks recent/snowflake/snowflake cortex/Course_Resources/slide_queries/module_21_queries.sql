-- ============================================================
-- Course 1018 — Module 21
-- On-slide QUERIES reference  (5 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- Step 1 — Give Every Prompt a Version
-- ----------------------------------------------------------

CREATE TABLE prompt_registry (
    prompt_version STRING,
    prompt_text STRING,
    created_at TIMESTAMP_NTZ,
    is_active BOOLEAN
);

-- Every AI_COMPLETE / AI_SENTIMENT call logs
-- which prompt_version produced each row


-- ----------------------------------------------------------
-- Step 2 — Trend the Agreement Rate by Version
-- ----------------------------------------------------------

SELECT
    DATE_TRUNC('week', checked_at) AS wk,
    prompt_version,
    AVG(CASE WHEN ai_label = human_label THEN 1 ELSE 0 END) AS agreement_rate
FROM sentiment_qa_log
GROUP BY wk, prompt_version
ORDER BY wk;


-- ----------------------------------------------------------
-- The Decision Log Table
-- ----------------------------------------------------------

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


-- ----------------------------------------------------------
-- Anti-Pattern First — Rolling Out to 100% at Once
-- ----------------------------------------------------------

-- WRONG — flips 100% of traffic instantly, no rollback window
UPDATE prompt_registry
SET is_active = TRUE
WHERE prompt_version = 'v2';

-- RIGHT — route by a deterministic hash, sample a slice first
SELECT
    CASE WHEN MOD(ABS(HASH(customer_id)), 100) < 5
         THEN 'variant_b' ELSE 'variant_a' END AS variant,
    customer_id
FROM incoming_requests;


-- ----------------------------------------------------------
-- Pinpointing the Exact Day
-- ----------------------------------------------------------

SELECT
    DATE_TRUNC('day', checked_at) AS dy,
    AVG(CASE WHEN ai_label = human_label THEN 1 ELSE 0 END) AS agreement_rate
FROM sentiment_qa_log
GROUP BY dy
ORDER BY dy;

