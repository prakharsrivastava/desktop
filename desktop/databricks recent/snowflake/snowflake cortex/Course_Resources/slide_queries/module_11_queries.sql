-- ============================================================
-- Course 1018 — Module 11
-- On-slide QUERIES reference  (7 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- Anti-Pattern: Regex Against a Raw Text Dump
-- ----------------------------------------------------------

SELECT invoice_no, REGEXP_SUBSTR(raw_text, 'Total: \$([0-9.,]+)', 1, 1, 'e') AS total_amt FROM raw_invoice_text; -- breaks the moment a vendor reorders fields


-- ----------------------------------------------------------
-- The Real Pattern: Four Statements, No Glue Code
-- ----------------------------------------------------------

PUT file://invoice_0091.pdf @cpg_docs_stage;
CREATE OR REPLACE TABLE parsed_docs AS
SELECT relative_path, AI_PARSE_DOCUMENT(TO_FILE('@cpg_docs_stage', relative_path), {'mode':'LAYOUT'}) AS layout_json
FROM DIRECTORY(@cpg_docs_stage);
SELECT relative_path, AI_EXTRACT(file => TO_FILE('@cpg_docs_stage', relative_path), responseFormat => [['vendor_name','What is the vendor name?'],['invoice_number','What is the invoice number?'],['total_amount','What is the total amount?']]) AS extracted
FROM DIRECTORY(@cpg_docs_stage);


-- ----------------------------------------------------------
-- Calling AI_PARSE_DOCUMENT
-- ----------------------------------------------------------

SELECT relative_path, AI_PARSE_DOCUMENT(TO_FILE('@cpg_docs_stage', relative_path), {'mode':'LAYOUT'}):content::STRING AS layout_text FROM DIRECTORY(@cpg_docs_stage) WHERE relative_path LIKE '%.pdf';


-- ----------------------------------------------------------
-- Anti-Pattern: Trusting the Unstated Default
-- ----------------------------------------------------------

-- anti-pattern: no mode specified, table structure not guaranteed
SELECT AI_PARSE_DOCUMENT(TO_FILE('@cpg_docs_stage','po_4471.pdf')) FROM DIRECTORY(@cpg_docs_stage);
-- fix: pin LAYOUT explicitly
SELECT AI_PARSE_DOCUMENT(TO_FILE('@cpg_docs_stage','po_4471.pdf'), {'mode':'LAYOUT'}) FROM DIRECTORY(@cpg_docs_stage);


-- ----------------------------------------------------------
-- Anti-Pattern: One Giant 'Extract Everything' Prompt
-- ----------------------------------------------------------

-- anti-pattern: one vague catch-all field
SELECT AI_EXTRACT(file => TO_FILE('@cpg_docs_stage','inv_2291.pdf'), responseFormat => ['everything about this invoice']) FROM DIRECTORY(@cpg_docs_stage);


-- ----------------------------------------------------------
-- The Fix: Named Fields, One Row Per Line Item
-- ----------------------------------------------------------

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
          'description':  {'type': 'array', 'items': {'type': 'string'}},
          'quantity':     {'type': 'array', 'items': {'type': 'number'}},
          'unit_price':   {'type': 'array', 'items': {'type': 'number'}}
        }
      }
    },
    scores => TRUE
  ) AS extracted
FROM DIRECTORY(@cpg_docs_stage);


-- ----------------------------------------------------------
-- Routing in SQL
-- ----------------------------------------------------------

SELECT relative_path,
  AI_EXTRACT(
    file => TO_FILE('@cpg_docs_stage', relative_path),
    responseFormat => [['sku','What is the SKU?'],
                        ['quantity','What is the quantity?'],
                        ['unit_price','What is the unit price?']],
    scores => TRUE
  ) AS extracted
FROM DIRECTORY(@cpg_docs_stage);

INSERT INTO invoice_line_items
SELECT * FROM extracted
WHERE extracted:scoring:scores:unit_price:score::FLOAT >= 0.85;

INSERT INTO review_queue
SELECT *, 'low_confidence_extraction' AS reason FROM extracted
WHERE extracted:scoring:scores:unit_price:score::FLOAT < 0.85;

