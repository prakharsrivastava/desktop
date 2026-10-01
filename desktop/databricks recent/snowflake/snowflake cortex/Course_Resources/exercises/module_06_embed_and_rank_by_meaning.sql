-- ==========================================================
-- Exercise 6.1 — Embed and Rank a Catalog by Meaning, Not Keywords
-- Course: 1018
-- Module 06
-- ==========================================================
--
-- Keyword search misses a customer who writes "the yogurt went sour early"
-- when your catalog note says "spoilage." AI_EMBED converts text into a
-- VECTOR your query can compare with VECTOR_COSINE_SIMILARITY, so meaning
-- matches even when the words don't. Here you'll generate an embedding for
-- a new complaint phrase and rank an entire product_reviews table by how
-- semantically close each review is to it.
--
-- Your task:
-- 1. Generate an embedding for the phrase "the yogurt went sour early"
--    using the same embedding model the module uses.
-- 2. Write a WITH clause that computes that query embedding once, then
--    joins it against product_reviews to rank all reviews by
--    VECTOR_COSINE_SIMILARITY, descending, top 10.
-- ==========================================================

-- YOUR CODE:

SELECT ___('snowflake-arctic-embed-m',
    'the yogurt went sour early'
) AS review_vector;

WITH query_vec AS (
    SELECT AI_EMBED('___', 'the yogurt went sour early') AS v
)
SELECT r.review_id, r.review_text,
    ___(r.review_vector, q.v) AS score
FROM product_reviews r, query_vec q
ORDER BY score ___
LIMIT ___;


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
SELECT AI_EMBED('snowflake-arctic-embed-m',
    'the yogurt went sour early'
) AS review_vector;

WITH query_vec AS (
    SELECT AI_EMBED('snowflake-arctic-embed-m', 'the yogurt went sour early') AS v
)
SELECT r.review_id, r.review_text,
    VECTOR_COSINE_SIMILARITY(r.review_vector, q.v) AS score
FROM product_reviews r, query_vec q
ORDER BY score DESC
LIMIT 10;
*/
