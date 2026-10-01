-- ============================================================
-- Course 1018 — Module 12
-- On-slide QUERIES reference  (7 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- Building the Training View
-- ----------------------------------------------------------

-- SNOWFLAKE.ML.FORECAST needs 3 columns: series id, timestamp, numeric target
CREATE OR REPLACE VIEW demand_training_view AS
SELECT
  sku_id || '_' || store_id AS series_id,
  sale_date AS ts,
  SUM(quantity) AS y
FROM daily_sku_store_sales
WHERE sale_date < '2026-01-01'
GROUP BY series_id, ts;


-- ----------------------------------------------------------
-- Training and Calling the Forecast Model
-- ----------------------------------------------------------

-- One statement trains one model object covering every series_id in the view
CREATE OR REPLACE SNOWFLAKE.ML.FORECAST sku_store_forecast_model(
  INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'demand_training_view'),
  SERIES_COLNAME => 'series_id',
  TIMESTAMP_COLNAME => 'ts',
  TARGET_COLNAME => 'y'
);

CALL sku_store_forecast_model!FORECAST(FORECASTING_PERIODS => 14);


-- ----------------------------------------------------------
-- Adding Exogenous Columns to the Training View
-- ----------------------------------------------------------

-- Extend the training view with promo/holiday exogenous columns
CREATE OR REPLACE VIEW demand_training_view_v2 AS
SELECT
  s.sku_id || '_' || s.store_id AS series_id,
  s.sale_date AS ts,
  SUM(s.quantity) AS y,
  MAX(p.is_promo_day) AS promo_flag,
  MAX(h.is_holiday) AS holiday_flag
FROM daily_sku_store_sales s
LEFT JOIN promo_calendar p
  ON p.calendar_date = s.sale_date AND p.sku_id = s.sku_id
LEFT JOIN holiday_calendar h
  ON h.calendar_date = s.sale_date
GROUP BY series_id, ts;
-- CREATE SNOWFLAKE.ML.FORECAST has no separate exogenous-variables
-- parameter: extra columns beyond series_id/ts/target (promo_flag,
-- holiday_flag here) are automatically picked up as exogenous regressors


-- ----------------------------------------------------------
-- Forecasting With Future Exogenous Data
-- ----------------------------------------------------------

CREATE OR REPLACE SNOWFLAKE.ML.FORECAST sku_store_forecast_model_v2(
  INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'demand_training_view_v2'),
  SERIES_COLNAME => 'series_id',
  TIMESTAMP_COLNAME => 'ts',
  TARGET_COLNAME => 'y'
);

-- Second input: future exogenous columns, no target (that's predicted)
CALL sku_store_forecast_model_v2!FORECAST(
  INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'future_promo_calendar_view'),
  SERIES_COLNAME => 'series_id',
  TIMESTAMP_COLNAME => 'ts'
);


-- ----------------------------------------------------------
-- Anti-Pattern: One Series, Every SKU Blended
-- ----------------------------------------------------------

-- The naive view collapses every SKU/store into ONE series
CREATE OR REPLACE VIEW naive_training_view AS
SELECT
  'ALL_SKUS' AS series_id,
  sale_date AS ts,
  SUM(quantity) AS y
FROM daily_sku_store_sales
GROUP BY ts;

CREATE OR REPLACE SNOWFLAKE.ML.FORECAST global_demand_model(
  INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'naive_training_view'),
  SERIES_COLNAME => 'series_id',
  TIMESTAMP_COLNAME => 'ts',
  TARGET_COLNAME => 'y'
);


-- ----------------------------------------------------------
-- The Fix — Return to the Per-SKU/Store Grain
-- ----------------------------------------------------------

-- Reuse demand_training_view from lesson_44 — series_id per SKU/store pair
CREATE OR REPLACE SNOWFLAKE.ML.FORECAST sku_store_forecast_model(
  INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'demand_training_view'),
  SERIES_COLNAME => 'series_id',
  TIMESTAMP_COLNAME => 'ts',
  TARGET_COLNAME => 'y'
);


-- ----------------------------------------------------------
-- Computing Reorder Points from Forecast Output
-- ----------------------------------------------------------

-- Join persisted forecast output to lead_time_reference, per series
CREATE OR REPLACE VIEW forecast_reorder_points AS
SELECT
  f.series_id,
  f.ts,
  f.forecast,
  f.upper_bound - f.lower_bound AS uncertainty_width,
  l.lead_time_days,
  (f.forecast * l.lead_time_days)
    + (1.65 * (f.upper_bound - f.lower_bound)) AS reorder_point
FROM forecast_output f
JOIN lead_time_reference l ON l.series_id = f.series_id;

