-- ==========================================================
-- Exercise 17.1 — Cap Runaway Chat History Before It Breaks AI_COMPLETE
-- Course: 1018
-- Module 17
-- ==========================================================
--
-- A support-bot prompt built from unbounded chat history worked fine
-- through turn 5, then started throwing max-tokens/context-length
-- exceptions around turn 40. The root cause was never the model — it
-- was an ARRAY_AGG with no LIMIT quietly growing every prior message
-- into every new prompt. This exercise rebuilds the fix: window the
-- history to the last N turns, summarize anything older, and stop
-- shipping a 210,000-character prompt to a model that never asked for one.
--
-- Your task:
-- 1. Write the ANTI-PATTERN query first (as it originally shipped) so you
--    can see exactly what it does wrong.
-- 2. Rewrite it to cap raw history to the last 10 turns using ORDER BY + LIMIT.
-- 3. Fold anything outside that window through AI_SUMMARIZE_AGG instead of
--    dropping it, then concatenate the current question last.
-- ==========================================================

-- YOUR CODE:

-- Step 1: reproduce the anti-pattern (unbounded, for comparison only)
SELECT AI_COMPLETE(
    'llama3.1-70b',
    ARRAY_TO_STRING(ARRAY_AGG(___), '\n') || current_question
)
FROM chat_history
WHERE session_id = ?;
-- No LIMIT, no windowing, no truncation — this is the broken version

-- Step 2 + 3: the fix — window to the last 10 turns, summarize, then complete
SELECT AI_COMPLETE(
    'llama3.1-70b',
    ___(msg) || current_question
)
FROM (
    SELECT msg
    FROM chat_history
    WHERE session_id = ?
    ORDER BY created_at ___
    LIMIT ___
);


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
-- Step 1: the anti-pattern, for reference (do not ship this)
SELECT AI_COMPLETE(
    'llama3.1-70b',
    ARRAY_TO_STRING(ARRAY_AGG(msg), '\n') || current_question
)
FROM chat_history
WHERE session_id = ?;
-- No LIMIT, no windowing, no truncation — every prior message rides along
-- forever; a 12,000-character prompt on turn 3 becomes 210,000 on turn 40

-- Step 2 + 3: the fix
SELECT AI_COMPLETE(
    'llama3.1-70b',
    AI_SUMMARIZE_AGG(msg) || current_question
)
FROM (
    SELECT msg
    FROM chat_history
    WHERE session_id = ?
    ORDER BY created_at DESC
    LIMIT 10
);
-- Cap raw history to the last 10 turns with LIMIT + ORDER BY, summarize
-- anything older instead of dropping it silently. Add a pre-flight token
-- estimate check before every call, not after the failure.
*/
