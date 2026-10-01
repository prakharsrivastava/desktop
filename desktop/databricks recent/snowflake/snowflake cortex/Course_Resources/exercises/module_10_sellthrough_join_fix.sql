-- ==========================================================
-- Exercise 10.2 — Pre-Aggregate Before Joining Fact Tables
-- Course: 1018
-- Module 10
-- ==========================================================
--
-- Cortex Analyst generated a flawed sell-through query that joins
-- pos_sales_daily directly to shipments on sku + date. Both tables have
-- multiple rows per sku/date (multiple stores, multiple shipment lines),
-- so the raw join fans out rows before the SUM ever runs — units_sold gets
-- double- and triple-counted. The fix is a pattern, not a one-off: always
-- pre-aggregate each side to the grain you actually want (sku + date)
-- BEFORE joining, so the join is one-row-to-one-row and the SUM is correct.
--
-- Your task:
-- 1. Build pos_agg: sku, sale_date, SUM(units_sold) from pos_sales_daily,
--    grouped by sku, sale_date.
-- 2. Build ship_agg: sku, ship_date, SUM(cases_shipped) from shipments,
--    grouped by sku, ship_date.
-- 3. Join pos_agg to ship_agg on sku + date, then join to product_master
--    on sku, and group the final result by pack_size.
-- ==========================================================

-- YOUR CODE:

WITH pos_agg AS (
  SELECT sku, sale_date, SUM(___) AS units_sold
  FROM pos_sales_daily
  GROUP BY sku, ___
),
ship_agg AS (
  SELECT sku, ship_date, SUM(cases_shipped) AS cases_shipped
  FROM shipments
  GROUP BY ___, ship_date
)
SELECT
  pm.pack_size,
  SUM(pos_agg.units_sold) AS units_sold
FROM pos_agg
JOIN ship_agg ON ship_agg.sku = pos_agg.sku AND ship_agg.___ = pos_agg.sale_date
JOIN product_master pm ON pm.sku = ___.sku
GROUP BY pm.___;


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
WITH pos_agg AS (
  SELECT sku, sale_date, SUM(units_sold) AS units_sold
  FROM pos_sales_daily
  GROUP BY sku, sale_date
),
ship_agg AS (
  SELECT sku, ship_date, SUM(cases_shipped) AS cases_shipped
  FROM shipments
  GROUP BY sku, ship_date
)
SELECT
  pm.pack_size,
  SUM(pos_agg.units_sold) AS units_sold
FROM pos_agg
JOIN ship_agg ON ship_agg.sku = pos_agg.sku AND ship_agg.ship_date = pos_agg.sale_date
JOIN product_master pm ON pm.sku = pos_agg.sku
GROUP BY pm.pack_size;
*/
