-- ============================================================
-- Course 1018 — Module 08
-- On-slide QUERIES reference  (4 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- Loading Raw Documents Into a Snowflake Stage
-- ----------------------------------------------------------

-- Create the internal stage for raw CPG source documents
CREATE STAGE IF NOT EXISTS cpg_docs_stage
  DIRECTORY = (ENABLE = TRUE)
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Upload from a local/cloud path (PUT run per file or via SnowSQL script)
-- PUT file:///local/product_specs/*.pdf @cpg_docs_stage/product_specs/;

-- Confirm the files landed and are visible to the account
SELECT relative_path, size_bytes, last_modified
FROM DIRECTORY(@cpg_docs_stage)
ORDER BY last_modified DESC;


-- ----------------------------------------------------------
-- Creating the Cortex Search Service Over Chunked Specs
-- ----------------------------------------------------------

-- Chunked table already populated: cpg_doc_chunks(chunk_id, sku, doc_type,
-- region, effective_date, page_number, chunk_text)
CREATE OR REPLACE CORTEX SEARCH SERVICE cpg_product_spec_search
  ON chunk_text
  PRIMARY KEY (chunk_id)
  ATTRIBUTES sku, doc_type, region, effective_date, page_number
  WAREHOUSE = cortex_search_wh
  TARGET_LAG = '1 hour'
  AS (
    SELECT chunk_id, chunk_text, sku, doc_type, region, effective_date, page_number
    FROM cpg_doc_chunks
  );


-- ----------------------------------------------------------
-- Generating a Grounded Answer With AI_COMPLETE
-- ----------------------------------------------------------

SELECT AI_COMPLETE(
  'claude-sonnet-4-5',
  CONCAT(
    'Answer using ONLY the context below. Cite the source page.\n\n',
    'Context:\n', context_text, '\n\n',
    'Question: does the granola bar SKU contain tree nuts, per the current spec?'
  )
) AS grounded_answer
FROM (
  SELECT LISTAGG(chunk_text || ' [p.' || page_number || ']', '\n---\n') AS context_text
  FROM retrieved_chunks
);


-- ----------------------------------------------------------
-- Prompting AI_COMPLETE to Decline Outside the Corpus
-- ----------------------------------------------------------

SELECT AI_COMPLETE(
  'claude-sonnet-4-5',
  CONCAT(
    'You are a CPG product-spec assistant. Answer ONLY using the context below. ',
    'If the context does not contain enough information to answer confidently, ',
    'respond exactly with: "This is not covered in the indexed product-spec corpus." ',
    'Do not guess or use outside knowledge.\n\n',
    'Context:\n', context_text, '\n\n',
    'Question: ', :buyer_question
  )
) AS grounded_or_decline;

