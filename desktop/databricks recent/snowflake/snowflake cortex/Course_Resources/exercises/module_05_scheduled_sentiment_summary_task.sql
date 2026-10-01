-- ==========================================================
-- Exercise 5.1 — Build a Scheduled Sentiment + Summary Pipeline
-- Course: 1018
-- Module 05
-- ==========================================================
--
-- The module's case study scores incoming product reviews with AI_SENTIMENT
-- and SNOWFLAKE.CORTEX.SUMMARIZE, then wraps the whole INSERT in a Snowflake
-- TASK so a CPG brand's review queue is triaged automatically instead of by
-- a human reading 10,000 reviews a week. Here you'll stand up the same
-- pattern for a different raw table: incoming customer-support tickets
-- instead of reviews, so the pipeline reads new tickets, scores sentiment,
-- summarizes the ticket body, and runs on a schedule without manual re-runs.
--
-- Your task:
-- 1. Create a results table `ticket_scores` with columns for the ticket id,
--    the raw ticket text, the sentiment result (VARIANT), a summary, and a
--    scored_at timestamp.
-- 2. Write the INSERT ... SELECT that scores AI_SENTIMENT and summarizes
--    with SNOWFLAKE.CORTEX.SUMMARIZE in one pass, skipping tickets already
--    scored.
-- 3. Wrap that INSERT in a CREATE OR REPLACE TASK that runs hourly on an
--    XS warehouse.
-- ==========================================================

-- YOUR CODE:

CREATE TABLE ticket_scores (
  ticket_id INT,
  ticket_text STRING,
  sentiment_result ___,
  summary_text STRING,
  scored_at ___
);

CREATE OR REPLACE TASK triage_tickets_task
  WAREHOUSE = ___
  SCHEDULE = 'USING CRON 0 * * * * UTC'
AS
INSERT INTO ticket_scores (ticket_id, ticket_text, sentiment_result, summary_text, scored_at)
SELECT
  ticket_id,
  ticket_text,
  ___(ticket_text),
  ___.___(ticket_text),
  CURRENT_TIMESTAMP()
FROM raw_tickets
WHERE ticket_id NOT IN (SELECT ticket_id FROM ___);


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
CREATE TABLE ticket_scores (
  ticket_id INT,
  ticket_text STRING,
  sentiment_result VARIANT,
  summary_text STRING,
  scored_at TIMESTAMP_NTZ
);

CREATE OR REPLACE TASK triage_tickets_task
  WAREHOUSE = xs_wh
  SCHEDULE = 'USING CRON 0 * * * * UTC'
AS
INSERT INTO ticket_scores (ticket_id, ticket_text, sentiment_result, summary_text, scored_at)
SELECT
  ticket_id,
  ticket_text,
  AI_SENTIMENT(ticket_text),
  SNOWFLAKE.CORTEX.SUMMARIZE(ticket_text),
  CURRENT_TIMESTAMP()
FROM raw_tickets
WHERE ticket_id NOT IN (SELECT ticket_id FROM ticket_scores);
*/
