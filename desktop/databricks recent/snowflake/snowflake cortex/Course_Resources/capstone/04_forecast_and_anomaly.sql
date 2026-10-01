-- ==========================================================
-- Capstone Layer 4 — Forecast + Anomaly Detection
-- Meridian Foods: sell-through forecast + vendor lead-time anomalies
-- Pattern from: module 12 (SNOWFLAKE.ML.FORECAST),
--               module 13 (SNOWFLAKE.ML.ANOMALY_DETECTION)
-- ==========================================================
--
-- Two silos from lesson_01 remain open after Layers 1-3: inventory/ERP
-- forward-looking risk, and cross-signal support tickets. This layer
-- closes the first one. Both models train on the SAME pos_sales and
-- vendor lead-time data already unified in Layers 1-3 — no new data
-- source is introduced here, only new questions asked of data already
-- in place.

-- TODO 1: one forecast model, all SKUs. SERIES_COLNAME splits the
-- forecast per SKU without 4,000 separate model calls. Point INPUT_DATA
-- at the daily POS view, TIMESTAMP_COLNAME at the sale date column,
-- TARGET_COLNAME at the units column, SERIES_COLNAME at the SKU column.
CREATE SNOWFLAKE.ML.FORECAST meridian_sku_forecast(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', '___'),   -- TODO: 'pos_sales_daily'
    TIMESTAMP_COLNAME => '___',                       -- TODO: the sale-date column
    TARGET_COLNAME => '___',                          -- TODO: the units column
    SERIES_COLNAME => '___'                           -- TODO: the SKU column
);

-- TODO 2: the anomaly model. vendor_terms.lead_time_days (Layer 1) is
-- exactly the input this model expects — a shipment logged several days
-- late for a vendor with historically tight variance is a real signal.
CREATE SNOWFLAKE.ML.ANOMALY_DETECTION meridian_shipment_anomaly(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', '___'),    -- TODO: 'vendor_shipment_events'
    TIMESTAMP_COLNAME => '___',                       -- TODO: the shipment-date column
    TARGET_COLNAME => '___',                          -- TODO: the lead-time column
    LABEL_COLNAME => '',
    SERIES_COLNAME => '___'                           -- TODO: the vendor column
);

-- TODO 3: DETECT_ANOMALIES scores NEW shipment events against the model
-- trained above — point this at the *_new view, not the training view.
CALL meridian_shipment_anomaly!DETECT_ANOMALIES(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', '___'),    -- TODO: 'vendor_shipment_events_new'
    TIMESTAMP_COLNAME => '___',
    TARGET_COLNAME => '___',
    SERIES_COLNAME => '___'
);

-- Checkpoint 4 (README rubric): both models created with a SERIES_COLNAME,
-- and DETECT_ANOMALIES called against a NEW events view, not the training
-- view. Flagged rows should land in a table the agent (Layer 5) can query
-- directly, no dashboard round-trip.
