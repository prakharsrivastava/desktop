-- ============================================================
-- Course 1018 — Module 01
-- On-slide QUERIES reference  (6 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- One Query, Not Five Dashboards
-- ----------------------------------------------------------

SELECT product_id, AI_SENTIMENT(review_text) AS sentiment
FROM customer_reviews
WHERE product_id = 'SKU-48213';


-- ----------------------------------------------------------
-- Task Functions — Narrow and Fast
-- ----------------------------------------------------------

SELECT review_id,
       AI_SENTIMENT(review_text) AS sentiment,
       AI_TRANSLATE(review_text, '', 'en') AS review_en,
       SNOWFLAKE.CORTEX.SUMMARIZE(review_text) AS summary
FROM product_reviews;


-- ----------------------------------------------------------
-- AI_COMPLETE — When the Question Isn't Standard
-- ----------------------------------------------------------

SELECT AI_COMPLETE(
  'llama3.1-70b',
  'Summarize why negative reviews spiked for SKU-48213 in the last 30 days: ' || review_text
)
FROM product_reviews
WHERE product_id = 'SKU-48213';


-- ----------------------------------------------------------
-- Debug Slide — Why Did My Nightly Job Time Out?
-- ----------------------------------------------------------

-- FIX: one batch task-function pass replaces per-row Cortex Search calls
SELECT review_id, AI_SENTIMENT(review_text) AS sentiment
FROM product_reviews;
-- A single SELECT touches every row — no loop, no per-row round trip


-- ----------------------------------------------------------
-- ANTI-PATTERN — Calling AI_COMPLETE Per Click
-- ----------------------------------------------------------

-- ANTI-PATTERN: one AI_COMPLETE call per user click, no caching, no batching
SELECT AI_COMPLETE('llama3.1-70b', 'Answer this ticket: ' || ticket_text)
FROM support_tickets
WHERE ticket_id = ?;


-- ----------------------------------------------------------
-- THE FIX — Batch the Work, Serve the Read
-- ----------------------------------------------------------

-- BATCH: pre-compute summaries over the whole table on a schedule
CREATE OR REPLACE TABLE ticket_summaries AS
SELECT ticket_id, SNOWFLAKE.CORTEX.SUMMARIZE(ticket_text) AS summary
FROM support_tickets;

