-- ==========================================================
-- Exercise 1.2 — Batch the Work, Serve the Read
-- Course: 1018
-- Module 01
-- ==========================================================
--
-- A common first mistake is calling AI_COMPLETE once per user click or once
-- per support ticket lookup, with no caching and no batching — every click
-- pays a fresh LLM round trip. The fix is the same idea that closes out
-- module 01: pre-compute the AI output for the WHOLE table on a schedule,
-- then let interactive reads simply SELECT the already-computed column.
--
-- Your task:
-- 1. Identify the anti-pattern below (one AI_COMPLETE call per ticket_id,
--    triggered live on every support-desk page load).
-- 2. Replace it with a batch job that pre-computes a summary for every row
--    in support_tickets, stored in a new table.
-- 3. Confirm the interactive path becomes a plain SELECT with no Cortex
--    call at read time.
-- ==========================================================

-- ANTI-PATTERN (do not ship this):
-- SELECT AI_COMPLETE('llama3.1-70b', 'Answer this ticket: ' || ticket_text)
-- FROM support_tickets
-- WHERE ticket_id = ?;

-- YOUR CODE:

CREATE OR REPLACE TABLE ___ AS
SELECT ticket_id, SNOWFLAKE.CORTEX.___(ticket_text) AS summary
FROM ___;

-- interactive read path (no Cortex call at request time):
SELECT summary
FROM ticket_summaries
WHERE ticket_id = ___;


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
CREATE OR REPLACE TABLE ticket_summaries AS
SELECT ticket_id, SNOWFLAKE.CORTEX.SUMMARIZE(ticket_text) AS summary
FROM support_tickets;

-- interactive read path (no Cortex call at request time):
SELECT summary
FROM ticket_summaries
WHERE ticket_id = 'TCK-10482';
*/
