-- ==========================================================
-- Exercise 11.1 — Stage a Vendor Invoice and Parse It With AI_PARSE_DOCUMENT
-- Course: 1018
-- Module 11
-- ==========================================================
--
-- Most CPG AI projects stall on the unglamorous first step: turning a PDF
-- invoice into a queryable row. Regex against a raw text dump breaks the
-- moment a vendor reorders fields on the page. AI_PARSE_DOCUMENT reads the
-- file layout-aware instead of as a flat string — but you must explicitly
-- pin {'mode':'LAYOUT'}; leaving the mode unspecified means the table
-- structure of the output isn't guaranteed.
--
-- Your task:
-- 1. Stage a local PDF into cpg_docs_stage.
-- 2. Create parsed_docs by calling AI_PARSE_DOCUMENT over every file in the
--    stage directory, explicitly in LAYOUT mode.
-- 3. Write a second query that pulls just the layout text back out as a
--    STRING for a quick eyeball check, filtered to only .pdf files.
-- ==========================================================

-- YOUR CODE:

PUT file://invoice_0091.pdf @cpg_docs_stage;

CREATE OR REPLACE TABLE parsed_docs AS
SELECT relative_path,
  AI_PARSE_DOCUMENT(TO_FILE('@cpg_docs_stage', relative_path), {'mode':'___'}) AS layout_json
FROM DIRECTORY(@cpg_docs_stage);

SELECT relative_path,
  AI_PARSE_DOCUMENT(TO_FILE('@cpg_docs_stage', relative_path), {'mode':'LAYOUT'}):___::STRING AS layout_text
FROM DIRECTORY(@cpg_docs_stage)
WHERE relative_path LIKE '___';


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
PUT file://invoice_0091.pdf @cpg_docs_stage;

CREATE OR REPLACE TABLE parsed_docs AS
SELECT relative_path,
  AI_PARSE_DOCUMENT(TO_FILE('@cpg_docs_stage', relative_path), {'mode':'LAYOUT'}) AS layout_json
FROM DIRECTORY(@cpg_docs_stage);

SELECT relative_path,
  AI_PARSE_DOCUMENT(TO_FILE('@cpg_docs_stage', relative_path), {'mode':'LAYOUT'}):content::STRING AS layout_text
FROM DIRECTORY(@cpg_docs_stage)
WHERE relative_path LIKE '%.pdf';
*/
