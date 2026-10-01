-- ============================================================
-- Course 1018 — Module 06
-- On-slide QUERIES reference  (5 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- Generating and Storing an Embedding with AI_EMBED
-- ----------------------------------------------------------

SELECT AI_EMBED('snowflake-arctic-embed-m',
    'Milk smelled off after two days, very disappointed'
) AS review_vector;


-- ----------------------------------------------------------
-- VECTOR_COSINE_SIMILARITY in a Single Query
-- ----------------------------------------------------------

SELECT review_id, review_text,
    VECTOR_COSINE_SIMILARITY(
        review_vector,
        AI_EMBED('snowflake-arctic-embed-m', 'packaging arrived damaged')
    ) AS score
FROM product_reviews
ORDER BY score DESC
LIMIT 5;


-- ----------------------------------------------------------
-- Ranking an Entire Product Catalog by Similarity
-- ----------------------------------------------------------

WITH query_vec AS (
    SELECT AI_EMBED('snowflake-arctic-embed-m', 'milk smells bad') AS v
)
SELECT r.review_id, r.review_text,
    VECTOR_COSINE_SIMILARITY(r.review_vector, q.v) AS score
FROM product_reviews r, query_vec q
ORDER BY score DESC
LIMIT 10;


-- ----------------------------------------------------------
-- Anti-Pattern: All-Pairs Cross Join Similarity Scan
-- ----------------------------------------------------------

SELECT a.review_id, b.review_id,
    VECTOR_COSINE_SIMILARITY(a.review_vector, b.review_vector) AS score
FROM product_reviews a, product_reviews b
WHERE a.review_id != b.review_id
ORDER BY score DESC
LIMIT 100;


-- ----------------------------------------------------------
-- Measuring the Blowup: Query Time at Three Table Sizes
-- ----------------------------------------------------------

SELECT COUNT(*) FROM product_reviews;
-- run the SAME cross-join query at 500, 50,000, and 5,000,000 rows,
-- timing each run and checking Query History for compute + spillage

