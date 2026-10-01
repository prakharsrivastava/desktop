-- ==========================================================
-- Exercise 14.1 — Fine-Tune a Model to Parse Brand-Specific Promo Codes
-- Course: 1018
-- Module 14
-- ==========================================================
--
-- A beverage brand's promo codes look like PRM-2024-CHOC-12PK-WMT — retailer,
-- pack size, and product baked into one proprietary string. A generic Cortex
-- model reads it as a random SKU and drops the retailer segment: prompting
-- alone got segments 1-3 right but returned "unknown" for the retailer on 38%
-- of the validation set. FINETUNE doesn't add facts — it reshapes how the
-- model reads YOUR format, which is exactly the fix here.
--
-- Your task:
-- 1. Kick off a SNOWFLAKE.CORTEX.FINETUNE('CREATE', ...) job named
--    promo_code_parser_v1, based on the llama3.1-8b base model, pointing at a
--    training query and a validation query over labeled prompt/completion pairs.
-- 2. Check the job's status with FINETUNE('DESCRIBE', ...) before assuming the
--    model is ready to call.
-- 3. Call the finished model exactly like a base model with AI_COMPLETE, just
--    passing your fine-tuned model's name instead of a base model name.
-- ==========================================================

-- YOUR CODE:

SELECT SNOWFLAKE.CORTEX.FINETUNE(
  '___',
  '___',
  '___',
  'SELECT prompt, completion FROM ___',
  'SELECT prompt, completion FROM ___'
);

SELECT SNOWFLAKE.CORTEX.FINETUNE('___', 'promo_code_parser_v1');

SELECT AI_COMPLETE('___', 'Parse this code: PRM-2024-CHOC-12PK-WMT');


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
SELECT SNOWFLAKE.CORTEX.FINETUNE(
  'CREATE',
  'promo_code_parser_v1',
  'llama3.1-8b',
  'SELECT prompt, completion FROM promo_training_labeled',
  'SELECT prompt, completion FROM promo_validation_labeled'
);

SELECT SNOWFLAKE.CORTEX.FINETUNE('DESCRIBE', 'promo_code_parser_v1');

SELECT AI_COMPLETE('promo_code_parser_v1', 'Parse this code: PRM-2024-CHOC-12PK-WMT');

-- Real numbers from the case study: retailer-field accuracy rose from 62%
-- baseline to 97% after fine-tuning on 800 human-verified examples, with
-- every one of the 60+ retailer abbreviations appearing at least ten times
-- in the training set. The team ran the fine-tuned model in shadow mode for
-- two weeks, comparing its output against the manual process without acting
-- on it, before letting it take over reconciliation automatically.
*/
