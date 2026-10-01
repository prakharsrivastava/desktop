-- ==========================================================
-- Capstone Layer 2 — Search Index
-- Meridian Foods: one search box across vendor terms, catalog, reviews
-- Pattern from: module 7 (CORTEX SEARCH SERVICE), module 8 (RAG assistant)
-- ==========================================================
--
-- Module 8's field-rep RAG assistant searched product reviews and spec
-- sheets. Same job here, wider scope: Meridian's field reps need one
-- search box across vendor terms (Layer 1), catalog descriptions, and
-- review text. Cortex Search handles retrieval; the assistant on top is
-- the same low-latency pattern from module 8.
--
-- TODO 1: define meridian_unified_text as a view UNION-ing the three text
-- sources — vendor_terms (Layer 1), catalog descriptions, and
-- product_reviews — each aliased to a common (sku_id, doc_type,
-- search_text) shape. This view is what Layer 2's search service reads.
-- CREATE OR REPLACE VIEW meridian_unified_text AS
--     SELECT sku_id, 'vendor_terms' AS doc_type, terms::STRING AS search_text
--     FROM vendor_terms /* TODO: join to a sku mapping if vendor_terms
--                          doesn't carry sku_id directly */
--     UNION ALL
--     SELECT sku_id, 'catalog' AS doc_type, description AS search_text
--     FROM product_catalog
--     UNION ALL
--     SELECT sku_id, 'review' AS doc_type, review_text AS search_text
--     FROM product_reviews;

-- TODO 2: stand up the search service itself. Same service definition
-- shape as module 7's build — just three sources instead of one, via the
-- unified view above. TARGET_LAG keeps the index current without a live
-- re-index on every write.
CREATE OR REPLACE CORTEX SEARCH SERVICE meridian_field_search
    ON ___              -- TODO: which column carries the searchable text?
    ATTRIBUTES ___, ___  -- TODO: doc_type, sku_id
    WAREHOUSE = ___      -- TODO: meridian_wh
    TARGET_LAG = '___'   -- TODO: '1 hour'
AS
SELECT sku_id, doc_type, search_text
FROM meridian_unified_text;

-- Checkpoint 2 (README rubric): meridian_field_search is created ON
-- search_text with ATTRIBUTES doc_type, sku_id, a named WAREHOUSE, and
-- TARGET_LAG set (not left at default).
