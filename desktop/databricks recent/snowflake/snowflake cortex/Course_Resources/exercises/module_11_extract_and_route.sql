-- ==========================================================
-- Exercise 11.2 — Extract Named Fields With Confidence Scores and Route
-- Course: 1018
-- Module 11
-- ==========================================================
--
-- A single vague "extract everything about this invoice" prompt to
-- AI_EXTRACT gives you a blob you still have to parse by hand. The fix is
-- named fields with a proper JSON schema — one row per line item, with
-- scores => TRUE turned on so every extracted value carries a confidence
-- score. Low-confidence extractions (e.g. a smudged unit price) should
-- never go straight into a downstream table; they should route to a human
-- review queue instead.
--
-- Your task:
-- 1. Call AI_EXTRACT with a JSON schema requesting vendor_name,
--    invoice_number, and arrays for sku / quantity / unit_price, with
--    scores => TRUE.
-- 2. Insert rows into invoice_line_items only where the unit_price score
--    is 0.85 or higher.
-- 3. Insert the rest into review_queue, tagged with a reason column.
-- ==========================================================

-- YOUR CODE:

SELECT relative_path,
  AI_EXTRACT(
    file => TO_FILE('@cpg_docs_stage', relative_path),
    responseFormat => {
      'type': 'json',
      'schema': {
        'type': 'object',
        'properties': {
          'vendor_name':    {'type': 'string'},
          'invoice_number': {'type': 'string'},
          'sku':          {'type': 'array', 'items': {'type': 'string'}},
          'quantity':     {'type': 'array', 'items': {'type': '___'}},
          'unit_price':   {'type': 'array', 'items': {'type': 'number'}}
        }
      }
    },
    scores => ___
  ) AS extracted
FROM DIRECTORY(@cpg_docs_stage);

INSERT INTO invoice_line_items
SELECT * FROM extracted
WHERE extracted:scoring:scores:unit_price:score::FLOAT >= ___;

INSERT INTO review_queue
SELECT *, '___' AS reason FROM extracted
WHERE extracted:scoring:scores:unit_price:score::FLOAT < 0.85;


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
SELECT relative_path,
  AI_EXTRACT(
    file => TO_FILE('@cpg_docs_stage', relative_path),
    responseFormat => {
      'type': 'json',
      'schema': {
        'type': 'object',
        'properties': {
          'vendor_name':    {'type': 'string'},
          'invoice_number': {'type': 'string'},
          'sku':          {'type': 'array', 'items': {'type': 'string'}},
          'quantity':     {'type': 'array', 'items': {'type': 'number'}},
          'unit_price':   {'type': 'array', 'items': {'type': 'number'}}
        }
      }
    },
    scores => TRUE
  ) AS extracted
FROM DIRECTORY(@cpg_docs_stage);

INSERT INTO invoice_line_items
SELECT * FROM extracted
WHERE extracted:scoring:scores:unit_price:score::FLOAT >= 0.85;

INSERT INTO review_queue
SELECT *, 'low_confidence_extraction' AS reason FROM extracted
WHERE extracted:scoring:scores:unit_price:score::FLOAT < 0.85;
*/
