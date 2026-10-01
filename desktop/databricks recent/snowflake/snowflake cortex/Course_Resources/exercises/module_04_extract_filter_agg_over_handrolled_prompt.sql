-- ==========================================================
-- Exercise 4.2 — Replace a Hand-Rolled Prompt With Task Functions
-- Course: 1018
-- Module 04
-- ==========================================================
--
-- Before AI_EXTRACT, AI_FILTER, and AI_AGG existed as task functions, teams
-- hand-rolled the same jobs in AI_COMPLETE with a long prompt asking the
-- model to behave like a classifier, a WHERE clause, or a SUM. That works,
-- but it's slower to write, harder to keep consistent across rows, and more
-- expensive at scale than the purpose-built function. This exercise pulls
-- three fields out of a ticket in one call, filters reviews by meaning
-- instead of keyword, and aggregates a "what are people complaining about"
-- answer — three jobs a hand-rolled AI_COMPLETE prompt would otherwise do.
--
-- Your task:
-- 1. Use AI_EXTRACT to pull three answers out of one ticket_text in a
--    single call: complaint category, whether a refund was requested, and
--    urgency.
-- 2. Use AI_FILTER with a natural-language predicate to find reviews that
--    express frustration about shipping delays.
-- 3. Use AI_AGG to summarize, in one sentence, what customers are most
--    commonly complaining about this month.
-- ==========================================================

-- ANTI-PATTERN (do not ship this — hand-rolled classifier prompt):
-- SELECT AI_COMPLETE(
--   'llama3.1-70b',
--   'You are a sentiment classifier. Read the review and respond with EXACTLY one word: positive, negative, or neutral. Do not add punctuation or explanation. Review: ' || review_text
-- )
-- FROM product_reviews;

-- YOUR CODE:

-- 1. One AI_EXTRACT call, three questions
SELECT AI_EXTRACT(
  ticket_text,
  [___, ___, ___]
);

-- 2. AI_FILTER: natural-language predicate in a WHERE clause
SELECT review_id, review_text
FROM product_reviews
WHERE AI_FILTER(
  PROMPT('___', review_text)
);

-- 3. AI_AGG: natural-language aggregate, like a smarter SUM
SELECT AI_AGG(
  review_text,
  '___'
) AS complaint_summary
FROM product_reviews
WHERE review_month = '___';


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
-- 1. One AI_EXTRACT call, three questions
SELECT AI_EXTRACT(
  ticket_text,
  ['what is the complaint category?', 'is a refund requested?', 'how urgent is this?']
);

-- 2. AI_FILTER: natural-language predicate in a WHERE clause
SELECT review_id, review_text
FROM product_reviews
WHERE AI_FILTER(
  PROMPT('Does this review express frustration about shipping delays? Review: {0}', review_text)
);

-- 3. AI_AGG: natural-language aggregate, like a smarter SUM
SELECT AI_AGG(
  review_text,
  'What are customers most commonly complaining about this month?'
) AS complaint_summary
FROM product_reviews
WHERE review_month = '2026-07';
*/
