-- ============================================================
-- Course 1018 — Module 14
-- On-slide QUERIES reference  (7 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- Kicking Off a FINETUNE Job
-- ----------------------------------------------------------

SELECT SNOWFLAKE.CORTEX.FINETUNE(
  'CREATE',
  'sku_parser_v1',
  'llama3.1-8b',
  'SELECT prompt, completion FROM cpg_db.ml.sku_training_set',
  'SELECT prompt, completion FROM cpg_db.ml.sku_validation_set'
);


-- ----------------------------------------------------------
-- Using and Checking the Fine-Tuned Model
-- ----------------------------------------------------------

SELECT SNOWFLAKE.CORTEX.FINETUNE('DESCRIBE', 'sku_parser_v1');

SELECT AI_COMPLETE('sku_parser_v1', 'Parse this code: PRM-2024-CHOC-12PK-WMT');


-- ----------------------------------------------------------
-- Baseline Attempt: Prompting Alone Falls Short
-- ----------------------------------------------------------

SELECT AI_COMPLETE(
  'llama3.1-8b',
  'Extract promo_type, year, flavor, pack_size, retailer from: PRM-2024-CHOC-12PK-WMT'
);
-- returns retailer: "unknown" on 38% of the validation set


-- ----------------------------------------------------------
-- Fine-Tuning on 800 Labeled Promo Codes
-- ----------------------------------------------------------

SELECT SNOWFLAKE.CORTEX.FINETUNE(
  'CREATE',
  'promo_code_parser_v1',
  'llama3.1-8b',
  'SELECT prompt, completion FROM promo_training_labeled',
  'SELECT prompt, completion FROM promo_validation_labeled'
);


-- ----------------------------------------------------------
-- What a Better Prompt Solved Instead
-- ----------------------------------------------------------

SELECT AI_CLASSIFY(
  customer_message,
  ['billing_question', 'product_question', 'complaint'],
  {'task_description': 'Classify CPG customer message. See examples.'}
);


-- ----------------------------------------------------------
-- The Guardrail Check Before You Fine-Tune
-- ----------------------------------------------------------

SELECT
  COUNT_IF(predicted_label = actual_label) / COUNT(*) AS classify_accuracy
FROM validation_results;
-- if classify_accuracy >= target_threshold, skip fine-tuning entirely


-- ----------------------------------------------------------
-- The Self-Labeling Lab — Let a Big Model Write Your Training Data
-- ----------------------------------------------------------

SELECT
  raw_code,
  AI_COMPLETE(
    'llama3.1-70b',
    'Parse this CPG promo code into promo_type, year, flavor, pack_size, retailer as JSON: ' || raw_code
  ) AS draft_label
FROM cpg_db.ml.unlabeled_promo_codes;

