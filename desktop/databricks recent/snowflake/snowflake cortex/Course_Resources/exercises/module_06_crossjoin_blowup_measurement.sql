-- ==========================================================
-- Exercise 6.2 — Prove Why the Cross-Join Anti-Pattern Falls Over
-- Course: 1018
-- Module 06
-- ==========================================================
--
-- The module's anti-pattern slide shows an all-pairs cross join comparing
-- every review to every other review with VECTOR_COSINE_SIMILARITY — it
-- works on a small table and grinds to a halt as row count grows, because
-- the comparison count grows O(n^2). Before you ever reach for Cortex
-- Search (module 07), you should FEEL this blowup yourself: write the
-- brute-force query, then measure it at three table sizes exactly as the
-- module's "Measuring the Blowup" slide describes.
--
-- Your task:
-- 1. Write the all-pairs self-join similarity query over product_reviews
--    (exclude a row matching itself), ordered by score descending, top 100.
-- 2. Write the COUNT(*) query you'd run before each timing pass to confirm
--    the current row count of product_reviews.
-- 3. In a comment, note what you would check in Query History after each
--    run (the two things the module calls out).
-- ==========================================================

-- YOUR CODE:

SELECT a.review_id, b.review_id,
    ___(a.review_vector, b.review_vector) AS score
FROM product_reviews a, product_reviews b
WHERE a.review_id ___ b.review_id
ORDER BY score DESC
LIMIT 100;

SELECT ___(*) FROM product_reviews;
-- run the SAME cross-join query at 500, 50,000, and 5,000,000 rows,
-- timing each run and checking Query History for: ___ and ___


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
SELECT a.review_id, b.review_id,
    VECTOR_COSINE_SIMILARITY(a.review_vector, b.review_vector) AS score
FROM product_reviews a, product_reviews b
WHERE a.review_id != b.review_id
ORDER BY score DESC
LIMIT 100;

SELECT COUNT(*) FROM product_reviews;
-- run the SAME cross-join query at 500, 50,000, and 5,000,000 rows,
-- timing each run and checking Query History for: compute time and spillage
*/
