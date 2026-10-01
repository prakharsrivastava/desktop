-- ==========================================================
-- Exercise 3.1 — Your First Production AI_COMPLETE Call
-- Course: 1018
-- Module 03
-- ==========================================================
--
-- AI_COMPLETE gives you a raw LLM call inside SQL, but "raw" means you own
-- the prompt structure and the output control. A classifier prompt with no
-- constraints will ramble; setting temperature to 0 and capping max_tokens
-- keeps the output short, consistent, and safe to parse downstream.
--
-- Your task:
-- 1. Write an AI_COMPLETE call that classifies each cpg_complaints row into
--    exactly one of: Quality, Shipping, Billing, Other.
-- 2. Instruct the model, in the prompt itself, to reply with only the
--    category word.
-- 3. Pass temperature and max_tokens so the output is deterministic and short.
-- ==========================================================

-- YOUR CODE:

SELECT complaint_id,
  AI_COMPLETE(
    '___',
    'You are a strict classifier. Reply with exactly one word: Quality, Shipping, Billing, or Other. Text: ' || ___,
    {'temperature': ___, 'max_tokens': ___}
  ) AS complaint_category
FROM ___;


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
SELECT complaint_id,
  AI_COMPLETE(
    'llama3.1-8b',
    'You are a strict classifier. Reply with exactly one word: Quality, Shipping, Billing, or Other. Text: ' || complaint_text,
    {'temperature': 0, 'max_tokens': 10}
  ) AS complaint_category
FROM cpg_complaints;
*/
