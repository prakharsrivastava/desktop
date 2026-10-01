-- ==========================================================
-- Exercise 8.1 — Index Chunked Docs and Retrieve Top-k Passages
-- Course: 1018
-- Module 08
-- ==========================================================
--
-- The field-rep assistant's whole value depends on indexing the RIGHT
-- documents with the RIGHT metadata: a rep asking "does this SKU contain
-- tree nuts in the EU?" needs the search to filter by sku AND region, not
-- just match on text. Here you'll create the Cortex Search Service over a
-- different chunked table — planogram_chunks — carrying the metadata
-- columns a rep would actually filter on, then retrieve the top passages
-- for a live question using the same SEARCH_PREVIEW pattern module 07
-- taught you.
--
-- Your task:
-- 1. Write CREATE OR REPLACE CORTEX SEARCH SERVICE `planogram_search` over
--    chunk_text from a table planogram_chunks(chunk_id, sku, store_format,
--    region, effective_date, chunk_text). PRIMARY KEY = chunk_id,
--    ATTRIBUTES = sku, store_format, region, effective_date. WAREHOUSE =
--    cortex_search_wh, TARGET_LAG = '1 hour'.
-- 2. Write a SEARCH_PREVIEW query retrieving the top 5 passages for the
--    question "where does the granola bar SKU sit in a small-format
--    store planogram", returning chunk_id and sku columns.
-- ==========================================================

-- YOUR CODE:

CREATE OR REPLACE CORTEX SEARCH SERVICE planogram_search
  ON ___
  PRIMARY KEY (___)
  ATTRIBUTES ___, ___, ___, ___
  WAREHOUSE = ___
  TARGET_LAG = '___'
  AS (
    SELECT chunk_id, chunk_text, sku, store_format, region, effective_date
    FROM planogram_chunks
  );

SELECT PARSE_JSON(
  SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
    '___',
    '{"query":"___","columns":["chunk_id","sku"],"limit":5}'
  )
) AS results;


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
CREATE OR REPLACE CORTEX SEARCH SERVICE planogram_search
  ON chunk_text
  PRIMARY KEY (chunk_id)
  ATTRIBUTES sku, store_format, region, effective_date
  WAREHOUSE = cortex_search_wh
  TARGET_LAG = '1 hour'
  AS (
    SELECT chunk_id, chunk_text, sku, store_format, region, effective_date
    FROM planogram_chunks
  );

SELECT PARSE_JSON(
  SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
    'planogram_search',
    '{"query":"where does the granola bar SKU sit in a small-format store planogram","columns":["chunk_id","sku"],"limit":5}'
  )
) AS results;
*/
