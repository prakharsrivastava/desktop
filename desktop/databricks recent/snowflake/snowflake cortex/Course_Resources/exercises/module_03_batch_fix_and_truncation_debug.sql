-- ==========================================================
-- Exercise 3.2 — Replace the Cursor Loop, Then Fix the Truncated Output
-- Course: 1018
-- Module 03
-- ==========================================================
--
-- The most expensive AI_COMPLETE mistake is calling it once per row inside a
-- cursor loop — a stored procedure that UPDATEs one row at a time pays one
-- LLM round trip per row and quietly multiplies cost and latency. The fix is
-- always the same: one set-based UPDATE over the whole table. Separately,
-- once a batch call is running, output can still come back cut off if
-- max_tokens is set too low for what you asked the model to write.
--
-- Your task:
-- 1. Rewrite the slow, cursor-based classify_complaints_slow() logic below
--    as a single UPDATE statement over the whole cpg_complaints table.
-- 2. Diagnose why the AI_COMPLETE call on cpg_feedback returns a summary cut
--    off mid-sentence, and fix it by right-sizing max_tokens.
-- ==========================================================

-- ANTI-PATTERN (do not ship this — one AI_COMPLETE call per row):
-- CREATE OR REPLACE PROCEDURE classify_complaints_slow()
-- RETURNS STRING
-- LANGUAGE SQL
-- AS
-- $$
--   DECLARE
--     c1 CURSOR FOR
--       SELECT complaint_id, complaint_text
--       FROM cpg_complaints;
--   BEGIN
--     FOR row IN c1 DO
--       UPDATE cpg_complaints
--       SET complaint_category = AI_COMPLETE(
--         'llama3.1-8b',
--         'Classify: ' || :row.complaint_text
--       )
--       WHERE complaint_id = :row.complaint_id;
--     END FOR;
--     RETURN 'done';
--   END;
-- $$;

-- YOUR CODE:

-- 1. The fix — one statement, whole table
UPDATE ___
SET complaint_category = AI_COMPLETE(
  '___',
  'Classify this complaint into exactly one category: Quality, Shipping, Billing, or Other. Complaint: ' || ___
);

-- BUG (returns a summary cut off mid-sentence):
-- SELECT
--   AI_COMPLETE(
--     'llama3.1-70b',
--     'Write a detailed 5-sentence summary of this customer feedback: ' || feedback_text,
--     {'max_tokens': 15}
--   ) AS summary
-- FROM cpg_feedback
-- LIMIT 1;

-- 2. THE FIX — right-size max_tokens for a 5-sentence summary
SELECT
  AI_COMPLETE(
    'llama3.1-70b',
    'Write a detailed 5-sentence summary of this customer feedback: ' || feedback_text,
    {'max_tokens': ___}
  ) AS summary
FROM cpg_feedback
LIMIT 1;


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
-- 1. The fix — one statement, whole table
UPDATE cpg_complaints
SET complaint_category = AI_COMPLETE(
  'llama3.1-8b',
  'Classify this complaint into exactly one category: Quality, Shipping, Billing, or Other. Complaint: ' || complaint_text
);

-- 2. THE FIX — right-size max_tokens for a 5-sentence summary
SELECT
  AI_COMPLETE(
    'llama3.1-70b',
    'Write a detailed 5-sentence summary of this customer feedback: ' || feedback_text,
    {'max_tokens': 200}
  ) AS summary
FROM cpg_feedback
LIMIT 1;
*/
