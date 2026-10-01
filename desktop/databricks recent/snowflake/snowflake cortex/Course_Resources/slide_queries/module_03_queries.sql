-- ============================================================
-- Course 1018 — Module 03
-- On-slide QUERIES reference  (6 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- Your First Production-Ready AI_COMPLETE Call
-- ----------------------------------------------------------

SELECT complaint_id, complaint_text,
  AI_COMPLETE(
    'llama3.1-8b',
    'Classify this customer complaint into exactly one category: Quality, Shipping, Billing, or Other. Reply with only the category word. Complaint: ' || complaint_text
  ) AS complaint_category
FROM cpg_complaints;


-- ----------------------------------------------------------
-- Setting Temperature and Max Tokens in SQL
-- ----------------------------------------------------------

SELECT complaint_id,
  AI_COMPLETE(
    'llama3.1-8b',
    'You are a strict classifier. Reply with exactly one word: Quality, Shipping, Billing, or Other. Text: ' || complaint_text,
    {'temperature': 0, 'max_tokens': 10}
  ) AS complaint_category
FROM cpg_complaints;


-- ----------------------------------------------------------
-- The Anti-Pattern in Code — Don't Ship This
-- ----------------------------------------------------------

CREATE OR REPLACE PROCEDURE classify_complaints_slow()
RETURNS STRING
LANGUAGE SQL
AS
$$
  DECLARE
    c1 CURSOR FOR
      SELECT complaint_id, complaint_text
      FROM cpg_complaints;
  BEGIN
    FOR row IN c1 DO
      UPDATE cpg_complaints
      SET complaint_category = AI_COMPLETE(
        'llama3.1-8b',
        'Classify: ' || :row.complaint_text
      )
      WHERE complaint_id = :row.complaint_id;
    END FOR;
    RETURN 'done';
  END;
$$;


-- ----------------------------------------------------------
-- The Fix — One Statement, Whole Table
-- ----------------------------------------------------------

UPDATE cpg_complaints
SET complaint_category = AI_COMPLETE(
  'llama3.1-8b',
  'Classify this complaint into exactly one category: Quality, Shipping, Billing, or Other. Complaint: ' || complaint_text
);


-- ----------------------------------------------------------
-- Reproducing the Bug
-- ----------------------------------------------------------

SELECT
  AI_COMPLETE(
    'llama3.1-70b',
    'Write a detailed 5-sentence summary of this customer feedback: ' || feedback_text,
    {'max_tokens': 15}
  ) AS summary
FROM cpg_feedback
LIMIT 1;


-- ----------------------------------------------------------
-- The Fix — Right-Sizing max_tokens and the Prompt
-- ----------------------------------------------------------

SELECT
  AI_COMPLETE(
    'llama3.1-70b',
    'Write a detailed 5-sentence summary of this customer feedback: ' || feedback_text,
    {'max_tokens': 200}
  ) AS summary
FROM cpg_feedback
LIMIT 1;

