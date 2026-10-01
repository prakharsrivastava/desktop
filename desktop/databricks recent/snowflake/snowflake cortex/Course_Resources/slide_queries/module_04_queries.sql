-- ============================================================
-- Course 1018 — Module 04
-- On-slide QUERIES reference  (9 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- AI_SENTIMENT — Score a Product Review
-- ----------------------------------------------------------

SELECT AI_SENTIMENT(
  'the packaging arrived crushed but the product inside is great',
  ['packaging', 'product quality']
);


-- ----------------------------------------------------------
-- AI_TRANSLATE — A Distributor Message in Spanish
-- ----------------------------------------------------------

SELECT AI_TRANSLATE(
  'el pedido llego incompleto, faltan tres cajas',
  '',
  'en'
);


-- ----------------------------------------------------------
-- SNOWFLAKE.CORTEX.SUMMARIZE — Condense a Vendor Email
-- ----------------------------------------------------------

SELECT SNOWFLAKE.CORTEX.SUMMARIZE(vendor_email_body)
FROM incoming_vendor_emails;


-- ----------------------------------------------------------
-- AI_EXTRACT — Pull the Complaint Category
-- ----------------------------------------------------------

SELECT AI_EXTRACT(
  'the bottle cap was loose and product leaked into the box.',
  {'category': 'what is the complaint category?'}
);


-- ----------------------------------------------------------
-- One Call, Multiple Questions
-- ----------------------------------------------------------

SELECT AI_EXTRACT(
  ticket_text,
  ['what is the complaint category?', 'is a refund requested?', 'how urgent is this?']
);


-- ----------------------------------------------------------
-- Anti-Pattern — The Hand-Rolled Sentiment Prompt
-- ----------------------------------------------------------

SELECT AI_COMPLETE(
  'llama3.1-70b',
  'You are a sentiment classifier. Read the review and respond with EXACTLY one word: positive, negative, or neutral. Do not add punctuation or explanation. Review: ' || review_text
)
FROM product_reviews;


-- ----------------------------------------------------------
-- The One-Line Fix
-- ----------------------------------------------------------

SELECT AI_SENTIMENT(review_text)
FROM product_reviews;


-- ----------------------------------------------------------
-- AI_FILTER and AI_AGG — SQL That Filters and Aggregates on Meaning
-- ----------------------------------------------------------

-- AI_FILTER: natural-language predicate in a WHERE clause
SELECT review_id, review_text
FROM product_reviews
WHERE AI_FILTER(
  PROMPT('Does this review express frustration about shipping delays? Review: {0}', review_text)
);

-- AI_AGG: natural-language aggregate, like a smarter SUM
SELECT AI_AGG(
  review_text,
  'What are customers most commonly complaining about this month?'
) AS complaint_summary
FROM product_reviews
WHERE review_month = '2026-07';


-- ----------------------------------------------------------
-- AI_SIMILARITY and AI_TRANSCRIBE — Matching Meaning, Turning Audio Into Text
-- ----------------------------------------------------------

-- AI_SIMILARITY: score how semantically close two texts are
SELECT a.review_id, b.review_id,
       AI_SIMILARITY(a.review_text, b.review_text) AS similarity_score
FROM product_reviews a
JOIN product_reviews b ON a.review_id < b.review_id
WHERE AI_SIMILARITY(a.review_text, b.review_text) > 0.85;

-- AI_TRANSCRIBE: turn a staged audio file into text
SELECT relative_path AS recording_name,
       AI_TRANSCRIBE(TO_FILE('@call_recordings_stage', relative_path)) AS transcript_text
FROM DIRECTORY(@call_recordings_stage);

