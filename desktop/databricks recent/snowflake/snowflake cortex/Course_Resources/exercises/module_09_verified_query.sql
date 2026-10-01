-- ==========================================================
-- Exercise 9.2 — Lock In a Verified Query (Without Locking In a Bug)
-- Course: 1018
-- Module 09
-- ==========================================================
--
-- A verified query pins the exact SQL Cortex Analyst returns for a
-- must-always-be-right question, so onboarding users get a guaranteed-
-- correct answer instead of a fresh (and possibly wrong) NL-to-SQL
-- generation every time. The catch: a verified query is only as good as
-- the SQL you locked into it — a bug baked into a verified query never
-- gets a second chance to be caught, since Cortex Analyst will always
-- return it verbatim for a matching question.
--
-- Your task:
-- 1. Write a verified_queries entry named quarterly_sellthrough_by_category
--    that answers "What was last quarter's sell-through by category?"
-- 2. Join fact_sales to dim_product on product_id AND to dim_time on
--    date_id (a missing time join is the classic verified-query bug —
--    without it, "last quarter" can't be filtered at all).
-- 3. Filter on t.quarter using DATEADD(quarter, -1, CURRENT_DATE()).
-- 4. Group by p.category and mark it usable as an onboarding question.
-- ==========================================================

-- YOUR CODE:

verified_queries:
  - name: quarterly_sellthrough_by_category
    question: "What was last quarter's sell-through by category?"
    sql: |
      SELECT p.category,
             SUM(f.units_sold) AS total_units
      FROM fact_sales f
      JOIN dim_product p ON f.product_id = p.___
      JOIN dim_time t ON f.date_id = t.___
      WHERE t.quarter = DATEADD(___, -1, CURRENT_DATE())
      GROUP BY p.___
    use_as_onboarding_question: ___


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
verified_queries:
  - name: quarterly_sellthrough_by_category
    question: "What was last quarter's sell-through by category?"
    sql: |
      SELECT p.category,
             SUM(f.units_sold) AS total_units
      FROM fact_sales f
      JOIN dim_product p ON f.product_id = p.product_id
      JOIN dim_time t ON f.date_id = t.date_id
      WHERE t.quarter = DATEADD(quarter, -1, CURRENT_DATE())
      GROUP BY p.category
    use_as_onboarding_question: true
*/
