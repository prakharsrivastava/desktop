-- ==========================================================
-- Exercise 14.2 — The Escalation-Ladder Gate: Prove You Need FINETUNE First
-- Course: 1018
-- Module 14
-- ==========================================================
--
-- A CPG chatbot team reached for FINETUNE in their very first sprint, before
-- testing prompts. Three weeks curating labeled data later, accuracy barely
-- beat a five-minute prompt rewrite — and the labeled dataset went stale
-- within a month as the product catalog changed. Fine-tuning is the LAST
-- lever in the Cortex toolkit (prompting -> AI_EXTRACT/AI_CLASSIFY -> RAG ->
-- fine-tuning), not the first. This exercise builds the gate check that
-- stops you from skipping straight to it.
--
-- Your task:
-- 1. First try the zero-training-data option: classify a customer message
--    into one of three categories with AI_CLASSIFY.
-- 2. Then write the guardrail query that measures whether AI_CLASSIFY (or
--    AI_EXTRACT) already clears your accuracy bar against a validation table
--    of predicted vs. actual labels.
-- 3. Only if that guardrail query comes back BELOW your target threshold does
--    a FINETUNE job get justified — comment where that decision point sits.
-- ==========================================================

-- YOUR CODE:

SELECT AI_CLASSIFY(
  customer_message,
  ['billing_question', 'product_question', '___'],
  {'task_description': '___'}
);

SELECT
  COUNT_IF(___ = ___) / COUNT(*) AS classify_accuracy
FROM ___;
-- if classify_accuracy >= target_threshold, skip fine-tuning entirely


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
SELECT AI_CLASSIFY(
  customer_message,
  ['billing_question', 'product_question', 'complaint'],
  {'task_description': 'Classify CPG customer message. See examples.'}
);
-- Real result: AI_CLASSIFY covered the routing task at 91% accuracy with
-- zero training data — most of the chatbot's failures were actually fixed
-- just by adding 3 few-shot examples to the prompt, before touching FINETUNE.

SELECT
  COUNT_IF(predicted_label = actual_label) / COUNT(*) AS classify_accuracy
FROM validation_results;
-- if classify_accuracy >= target_threshold, skip fine-tuning entirely

-- Escalation ladder checklist (bookmark before your next Cortex build):
-- [ ] Task repeats at high volume (thousands+ calls/month) with a stable,
--     narrow format? If no, stop at prompting.
-- [ ] Do you have 300+ clean, human-verified labeled examples covering every
--     edge case? If no, don't fine-tune yet.
-- [ ] Does the model need your live documents/facts, not just format or
--     vocabulary? Use RAG/Cortex Search instead.
-- [ ] Does AI_EXTRACT or AI_CLASSIFY already clear your accuracy bar with
--     zero training data? If yes, skip fine-tuning.
-- [ ] All four boxes checked and cost still justified? Only then run
--     FINETUNE('CREATE', ...).

-- Bonus — the real bottleneck is often hand-labeling itself. Self-labeling
-- distillation flips it: call a large model to DRAFT labels, a human just
-- approves/fixes them, then fine-tune a smaller/cheaper model on that
-- distilled set:
-- SELECT
--   raw_code,
--   AI_COMPLETE(
--     'llama3.1-70b',
--     'Parse this CPG promo code into promo_type, year, flavor, pack_size, retailer as JSON: ' || raw_code
--   ) AS draft_label
-- FROM cpg_db.ml.unlabeled_promo_codes;
-- Net effect: 70B-quality labeling speed feeding an 8B-cost production model.
*/
