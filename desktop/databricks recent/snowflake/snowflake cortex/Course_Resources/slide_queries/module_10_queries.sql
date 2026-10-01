-- ============================================================
-- Course 1018 — Module 10
-- On-slide QUERIES reference  (4 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- Declaring the Verified Query for Sell-Through Rate
-- ----------------------------------------------------------

verified_queries:
  - name: sell_through_rate_by_pack_size
    question: "sell-through rate by pack size for a category and region, this quarter vs last"
    sql: |
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
        SUM(pos_agg.units_sold) AS units_sold,
        SUM(ship_agg.cases_shipped) * 12 AS units_shipped,
        SUM(pos_agg.units_sold) / NULLIF(SUM(ship_agg.cases_shipped) * 12, 0) AS sell_through_rate
      FROM pos_agg
      JOIN ship_agg ON ship_agg.sku = pos_agg.sku AND ship_agg.ship_date = pos_agg.sale_date
      JOIN product_master pm ON pm.sku = pos_agg.sku
      WHERE pm.category = 'salty_snacks'
      GROUP BY pm.pack_size


-- ----------------------------------------------------------
-- Registering the Semantic Model With Cortex Analyst
-- ----------------------------------------------------------

PUT file:///local/cpg_sell_through_model.yaml @cpg_semantic_stage AUTO_COMPRESS=FALSE;

-- Cortex Analyst reads the model directly from the staged YAML file
-- when a request specifies semantic_model_file pointing at this stage path


-- ----------------------------------------------------------
-- The Flawed Join Cortex Analyst Generated
-- ----------------------------------------------------------

-- FLAWED: joins on sku + date only, no store/DC grain control
SELECT
  pm.pack_size,
  SUM(p.units_sold) AS units_sold_flawed
FROM pos_sales_daily p
JOIN shipments s ON s.sku = p.sku AND s.ship_date = p.sale_date
JOIN product_master pm ON pm.sku = p.sku
GROUP BY pm.pack_size;


-- ----------------------------------------------------------
-- The Fix: Pre-Aggregate Each Side Before Joining
-- ----------------------------------------------------------

-- FIXED: pre-aggregate each side to sku+date grain first
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

