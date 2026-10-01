-- ============================================================
-- Course 1018 — Module 09
-- On-slide QUERIES reference  (2 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- Anti-Pattern: A Verified Query That Locks In a Bug
-- ----------------------------------------------------------

sql: |
  SELECT p.category, SUM(f.units_sold) AS total_units
  FROM fact_sales f
  JOIN dim_product p ON f.product_id = p.product_id
  WHERE f.sale_date >= DATEADD(day, -90, CURRENT_DATE())


-- ----------------------------------------------------------
-- Two Ways to Author: YAML File vs. Native Semantic View
-- ----------------------------------------------------------

CREATE SEMANTIC VIEW cpg_analyst_sv
  TABLES (
    fact_sales AS cpg_db.analytics.fact_sales PRIMARY KEY (sale_id),
    dim_product AS cpg_db.analytics.dim_product PRIMARY KEY (product_id)
  )
  RELATIONSHIPS (
    sales_to_product AS fact_sales (product_id) REFERENCES dim_product (product_id)
  )
  FACTS ( fact_sales.units_sold AS units_sold )
  DIMENSIONS ( dim_product.category AS product_category )
  METRICS ( fact_sales.net_sales AS SUM(fact_sales.net_sales_amount) );

