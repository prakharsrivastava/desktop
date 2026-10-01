-- ==========================================================
-- Exercise 12.2 — Add Promo/Holiday Exogenous Columns and Compute Reorder Points
-- Course: 1018
-- Module 12
-- ==========================================================
--
-- Promotions and holidays swing CPG demand hard, and a forecast that
-- ignores them is a forecast that's wrong every time a promo runs.
-- SNOWFLAKE.ML.FORECAST has no separate "exogenous variables" parameter —
-- any extra column beyond series_id/ts/target in the training view is
-- automatically picked up as an exogenous regressor. The business payoff is
-- turning the forecast's uncertainty band into a safety-stock number: the
-- wider the band, the more buffer stock you carry.
--
-- Your task:
-- 1. Extend demand_training_view into demand_training_view_v2, LEFT JOINing
--    promo_calendar and holiday_calendar to add promo_flag and
--    holiday_flag (MAX() per group, since a day can have multiple rows).
-- 2. Train sku_store_forecast_model_v2 against the v2 view.
-- 3. Build forecast_reorder_points: join persisted forecast output to
--    lead_time_reference and compute reorder_point as
--    (forecast * lead_time_days) + a 1.65 safety multiplier on the
--    uncertainty width (upper_bound - lower_bound).
-- ==========================================================

-- YOUR CODE:

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
LEFT JOIN ___ h
  ON h.calendar_date = s.sale_date
GROUP BY series_id, ts;

CREATE OR REPLACE SNOWFLAKE.ML.FORECAST sku_store_forecast_model_v2(
  INPUT_DATA => SYSTEM$REFERENCE('VIEW', '___'),
  SERIES_COLNAME => 'series_id',
  TIMESTAMP_COLNAME => 'ts',
  TARGET_COLNAME => 'y'
);

CREATE OR REPLACE VIEW forecast_reorder_points AS
SELECT
  f.series_id,
  f.ts,
  f.forecast,
  f.upper_bound - f.___ AS uncertainty_width,
  l.lead_time_days,
  (f.forecast * l.lead_time_days)
    + (___ * (f.upper_bound - f.lower_bound)) AS reorder_point
FROM forecast_output f
JOIN lead_time_reference l ON l.series_id = f.___;


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
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

CREATE OR REPLACE SNOWFLAKE.ML.FORECAST sku_store_forecast_model_v2(
  INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'demand_training_view_v2'),
  SERIES_COLNAME => 'series_id',
  TIMESTAMP_COLNAME => 'ts',
  TARGET_COLNAME => 'y'
);

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
*/
