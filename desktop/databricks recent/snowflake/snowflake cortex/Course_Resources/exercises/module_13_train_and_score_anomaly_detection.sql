-- ==========================================================
-- Exercise 13.1 — Train and Score a Per-DC ANOMALY_DETECTION Model
-- Course: 1018
-- Module 13
-- ==========================================================
--
-- A distribution ops lead trusted a FORECAST-based weekly report and missed a
-- DC's shipment volume dropping 40% for three days straight — the forecast
-- predicts the trend, it doesn't flag that today is already off. ANOMALY_DETECTION
-- closes that gap: train it once on a labeled history view, then call it
-- repeatedly to score new rows same-day instead of waiting for the weekly rollup.
--
-- Your task:
-- 1. Train a SNOWFLAKE.ML.ANOMALY_DETECTION model named dc_shipment_anomaly_model
--    over the dc_shipment_history view, modeling ship_date against shipment_volume.
-- 2. Since there is no pre-labeled anomaly set yet, pass the required
--    LABEL_COLNAME argument as an empty string.
-- 3. Score the last 7 days of volume against the trained model with
--    DETECT_ANOMALIES, using the same TIMESTAMP_COLNAME / TARGET_COLNAME pair.
-- 4. Note which output columns tell you not just THAT a row is anomalous, but
--    HOW confident the model is (the lower/upper bound gap).
-- ==========================================================

-- YOUR CODE:

CREATE OR REPLACE SNOWFLAKE.ML.ANOMALY_DETECTION ___(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', '___'),
    TIMESTAMP_COLNAME => '___',
    TARGET_COLNAME => '___',
    LABEL_COLNAME => '___'
);

CALL dc_shipment_anomaly_model!___(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'dc_shipment_last_7_days'),
    TIMESTAMP_COLNAME => 'ship_date',
    TARGET_COLNAME => '___'
);

-- Bonus: order the flagged rows by how far shipment_volume sits from the
-- model's expected value, worst offenders first (use the result columns
-- from DETECT_ANOMALIES — is_anomaly, forecast/expected value, bounds).


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
CREATE OR REPLACE SNOWFLAKE.ML.ANOMALY_DETECTION dc_shipment_anomaly_model(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'dc_shipment_history'),
    TIMESTAMP_COLNAME => 'ship_date',
    TARGET_COLNAME => 'shipment_volume',
    LABEL_COLNAME => ''
);

CALL dc_shipment_anomaly_model!DETECT_ANOMALIES(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'dc_shipment_last_7_days'),
    TIMESTAMP_COLNAME => 'ship_date',
    TARGET_COLNAME => 'shipment_volume'
);

-- Anti-pattern to avoid (from the same module): don't pool every DC into one
-- model. A pooled model's "normal range" is so wide a 40% drop at one DC gets
-- averaged away by 200 other DCs shipping normally. Train one model per series
-- that actually needs its own seasonal pattern (per DC, or per DC+category).
*/
