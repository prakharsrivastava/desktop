-- ==========================================================
-- Exercise 4.1 — Task-Function Triage on a Product Review
-- Course: 1018
-- Module 04
-- ==========================================================
--
-- The three most-used Cortex task functions — AI_SENTIMENT, AI_TRANSLATE,
-- and SNOWFLAKE.CORTEX.SUMMARIZE — each answer one narrow, well-defined
-- question. AI_SENTIMENT can even score sentiment against specific aspects
-- (like "packaging" vs "product quality") instead of one blended score,
-- which matters when a review is mixed: good product, bad box.
--
-- Your task:
-- 1. Score the sentiment of a mixed review against two named aspects:
--    packaging and product quality.
-- 2. Translate a distributor's message that arrived in Spanish into English.
-- 3. Summarize a long incoming vendor email down to its key point.
-- ==========================================================

-- YOUR CODE:

-- 1. Aspect-based sentiment on a mixed review
SELECT AI_SENTIMENT(
  'the packaging arrived crushed but the product inside is great',
  [___, ___]
);

-- 2. Translate a distributor message (source language left blank = auto-detect)
SELECT AI_TRANSLATE(
  'el pedido llego incompleto, faltan tres cajas',
  '___',
  '___'
);

-- 3. Summarize a long vendor email
SELECT SNOWFLAKE.CORTEX.___(vendor_email_body)
FROM ___;


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
-- 1. Aspect-based sentiment on a mixed review
SELECT AI_SENTIMENT(
  'the packaging arrived crushed but the product inside is great',
  ['packaging', 'product quality']
);

-- 2. Translate a distributor message (source language left blank = auto-detect)
SELECT AI_TRANSLATE(
  'el pedido llego incompleto, faltan tres cajas',
  '',
  'en'
);

-- 3. Summarize a long vendor email
SELECT SNOWFLAKE.CORTEX.SUMMARIZE(vendor_email_body)
FROM incoming_vendor_emails;
*/
