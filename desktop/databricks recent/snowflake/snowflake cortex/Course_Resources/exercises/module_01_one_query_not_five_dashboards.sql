-- ==========================================================
-- Exercise 1.1 — One Query, Not Five Dashboards
-- Course: 1018
-- Module 01
-- ==========================================================
--
-- The opening story of this course is a CPG stockout that stayed unexplained
-- for a week because the answer was scattered across five disconnected
-- systems and dashboards. Cortex task functions collapse that fan-out: you
-- can score sentiment, normalize language, and condense long text in ONE
-- SELECT over ONE table, instead of exporting to five tools.
--
-- Your task:
-- 1. Query the product_reviews table for a single SKU under investigation.
-- 2. In one SELECT, add a sentiment score, an English-normalized review
--    (reviews may arrive in any language), and a one-line summary.
-- 3. Keep it to a single statement — no separate dashboard exports.
-- ==========================================================

-- YOUR CODE:

SELECT review_id,
       ___(review_text) AS sentiment,
       ___(review_text, '', '___') AS review_en,
       SNOWFLAKE.CORTEX.___(review_text) AS summary
FROM ___
WHERE product_id = 'SKU-48213';


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
SELECT review_id,
       AI_SENTIMENT(review_text) AS sentiment,
       AI_TRANSLATE(review_text, '', 'en') AS review_en,
       SNOWFLAKE.CORTEX.SUMMARIZE(review_text) AS summary
FROM product_reviews
WHERE product_id = 'SKU-48213';
*/
