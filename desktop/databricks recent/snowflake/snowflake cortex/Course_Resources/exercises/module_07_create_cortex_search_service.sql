-- ==========================================================
-- Exercise 7.1 — Stand Up a Managed CORTEX SEARCH SERVICE
-- Course: 1018
-- Module 07
-- ==========================================================
--
-- Module 06 showed you the pain of hand-rolled vector search: your own
-- embedding column, your own cosine-similarity ranking, your own scaling
-- problems. CORTEX SEARCH SERVICE replaces all of that with one managed
-- CREATE statement. Here you'll create the service over a different CPG
-- source table — product_recall_notices — instead of the module's
-- support_tickets, and query it exactly like the module's SEARCH_PREVIEW
-- example.
--
-- Your task:
-- 1. Write CREATE CORTEX SEARCH SERVICE `product_recall_search` over the
--    notice_text column of product_recall_notices. Use notice_id as the
--    PRIMARY KEY, carry sku and region as ATTRIBUTES, run on
--    `search_wh`, and set TARGET_LAG to '1 hour'.
--    Remember the required clause order: ON -> PRIMARY KEY -> ATTRIBUTES
--    -> WAREHOUSE -> TARGET_LAG.
-- 2. Query the service with SNOWFLAKE.CORTEX.SEARCH_PREVIEW for the
--    phrase "undeclared allergen recall", limiting to 5 results.
-- ==========================================================

-- YOUR CODE:

CREATE CORTEX SEARCH SERVICE product_recall_search
  ON ___
  PRIMARY KEY (___)
  ATTRIBUTES ___, ___
  WAREHOUSE = ___
  TARGET_LAG = '___'
AS
  SELECT notice_id, sku, region, notice_text
  FROM product_recall_notices;

SELECT PARSE_JSON(
  ___.___.___(
    'product_recall_search',
    '{"query":"undeclared allergen recall","columns":["notice_id","sku","region"],"limit":5}'
  )
) AS results;


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
CREATE CORTEX SEARCH SERVICE product_recall_search
  ON notice_text
  PRIMARY KEY (notice_id)
  ATTRIBUTES sku, region
  WAREHOUSE = search_wh
  TARGET_LAG = '1 hour'
AS
  SELECT notice_id, sku, region, notice_text
  FROM product_recall_notices;

SELECT PARSE_JSON(
  SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
    'product_recall_search',
    '{"query":"undeclared allergen recall","columns":["notice_id","sku","region"],"limit":5}'
  )
) AS results;
*/
