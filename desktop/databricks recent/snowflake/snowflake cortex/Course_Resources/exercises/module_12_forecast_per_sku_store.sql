-- ==========================================================
-- Exercise 12.1 — Train a Per-SKU/Store Forecast, Not One Global Model
-- Course: 1018
-- Module 12
-- ==========================================================
--
-- The anti-pattern is blending all 50,000 SKUs into ONE series_id ("ALL_SKUS")
-- and training a single global model — it forecasts total demand fine but is
-- useless for reordering any individual SKU at any individual store.
-- SNOWFLAKE.ML.FORECAST needs exactly 3 columns in its training view: a
-- series id, a timestamp, and a numeric target — one CREATE MODEL statement
-- then covers every series_id present in that view as its own series.
--
-- Your task:
-- 1. Build demand_training_view with series_id = sku_id concatenated with
--    store_id, ts = sale_date, y = SUM(quantity), filtered to history
--    before '2026-01-01'.
-- 2. Train sku_store_forecast_model with SNOWFLAKE.ML.FORECAST pointing at
--    that view via SYSTEM$REFERENCE.
-- 3. Call the model to forecast 14 periods ahead.
-- ==========================================================

-- YOUR CODE:

CREATE OR REPLACE VIEW demand_training_view AS
SELECT
  sku_id || '_' || store_id AS ___,
  sale_date AS ts,
  SUM(___) AS y
FROM daily_sku_store_sales
WHERE sale_date < '2026-01-01'
GROUP BY series_id, ts;

CREATE OR REPLACE SNOWFLAKE.ML.FORECAST sku_store_forecast_model(
  INPUT_DATA => SYSTEM$REFERENCE('___', 'demand_training_view'),
  SERIES_COLNAME => '___',
  TIMESTAMP_COLNAME => 'ts',
  TARGET_COLNAME => '___'
);

CALL sku_store_forecast_model!FORECAST(FORECASTING_PERIODS => ___);


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
CREATE OR REPLACE VIEW demand_training_view AS
SELECT
  sku_id || '_' || store_id AS series_id,
  sale_date AS ts,
  SUM(quantity) AS y
FROM daily_sku_store_sales
WHERE sale_date < '2026-01-01'
GROUP BY series_id, ts;

CREATE OR REPLACE SNOWFLAKE.ML.FORECAST sku_store_forecast_model(
  INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'demand_training_view'),
  SERIES_COLNAME => 'series_id',
  TIMESTAMP_COLNAME => 'ts',
  TARGET_COLNAME => 'y'
);

CALL sku_store_forecast_model!FORECAST(FORECASTING_PERIODS => 14);
*/
