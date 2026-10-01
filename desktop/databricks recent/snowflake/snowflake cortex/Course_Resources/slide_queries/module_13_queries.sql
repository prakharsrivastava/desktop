-- ============================================================
-- Course 1018 — Module 13
-- On-slide QUERIES reference  (8 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- Step 1: Create the Model Over Shipment History
-- ----------------------------------------------------------

CREATE OR REPLACE SNOWFLAKE.ML.ANOMALY_DETECTION dc_shipment_anomaly_model(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'dc_shipment_history'),
    TIMESTAMP_COLNAME => 'ship_date',
    TARGET_COLNAME => 'shipment_volume',
    LABEL_COLNAME => ''
);


-- ----------------------------------------------------------
-- Step 2: Score New Volume with DETECT_ANOMALIES
-- ----------------------------------------------------------

CALL dc_shipment_anomaly_model!DETECT_ANOMALIES(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'dc_shipment_last_7_days'),
    TIMESTAMP_COLNAME => 'ship_date',
    TARGET_COLNAME => 'shipment_volume'
);


-- ----------------------------------------------------------
-- Running the Decomposition
-- ----------------------------------------------------------

CREATE SNOWFLAKE.ML.TOP_INSIGHTS pos_swing_insights();

CALL pos_swing_insights!GET_DRIVERS(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'pos_weekly_by_dimension'),
    LABEL_COLNAME => 'is_this_week',
    METRIC_COLNAME => 'revenue'
);


-- ----------------------------------------------------------
-- Diagnosis: The Model Wasn't Given Enough History to Learn Seasonality
-- ----------------------------------------------------------

-- confirm season coverage in the training data
SELECT MIN(ship_date), MAX(ship_date)
FROM dc_shipment_history;


-- ----------------------------------------------------------
-- Fix Step 1: Retrain on 2+ Years of History
-- ----------------------------------------------------------

CREATE OR REPLACE SNOWFLAKE.ML.ANOMALY_DETECTION dc_shipment_anomaly_model(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'dc_shipment_history_2yr'),
    TIMESTAMP_COLNAME => 'ship_date',
    TARGET_COLNAME => 'shipment_volume',
    LABEL_COLNAME => ''
);


-- ----------------------------------------------------------
-- Verify the Fix Before Trusting It Live
-- ----------------------------------------------------------

-- backtest against last year's known-normal spike
CALL dc_shipment_anomaly_model!DETECT_ANOMALIES(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'dc_shipment_last_thanksgiving'),
    TIMESTAMP_COLNAME => 'ship_date',
    TARGET_COLNAME => 'shipment_volume'
);


-- ----------------------------------------------------------
-- Signal 1: ANOMALY_DETECTION Flags the Divergence Same-Day
-- ----------------------------------------------------------

CALL inventory_variance_model!DETECT_ANOMALIES(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'daily_inventory_variance'),
    TIMESTAMP_COLNAME => 'count_date',
    TARGET_COLNAME => 'variance_units'
);


-- ----------------------------------------------------------
-- Signal 2: Contribution Explorer Narrows It to One Cause
-- ----------------------------------------------------------

CREATE SNOWFLAKE.ML.TOP_INSIGHTS dc0044_insights();

CALL dc0044_insights!GET_DRIVERS(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'dc0044_variance_by_dimension'),
    LABEL_COLNAME => 'is_flagged_week',
    METRIC_COLNAME => 'variance_units'
);

