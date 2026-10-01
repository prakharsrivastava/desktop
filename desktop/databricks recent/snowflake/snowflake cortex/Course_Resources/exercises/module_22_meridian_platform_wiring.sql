-- ==========================================================
-- Exercise 22.1 — Wire Documents Into Search, Structured Data Into a Semantic View
-- Course: 1018
-- Module 22 (Capstone)
-- ==========================================================
--
-- The capstone's first half (lesson_84) makes Meridian Foods' unstructured
-- documents and structured tables both queryable in plain language. Here
-- you'll repeat the exact same wiring the capstone used — AI_PARSE_DOCUMENT
-- + AI_EXTRACT to structure vendor documents, a CORTEX SEARCH SERVICE over
-- the unified text, and a SEMANTIC VIEW over the sales tables — but pointed
-- at a second Meridian stage: promotional flyers instead of vendor
-- contracts, alongside the same pos_sales / product_catalog / loyalty_members
-- tables from the capstone's semantic model.
--
-- Your task:
-- 1. Parse every file in `@meridian_promo_stage` with AI_PARSE_DOCUMENT in
--    LAYOUT mode into `promo_flyers_parsed`, then AI_EXTRACT the fields
--    ['sku_list','discount_pct','promo_start','promo_end'] into
--    `promo_terms`.
-- 2. Create `meridian_promo_search`, a CORTEX SEARCH SERVICE ON search_text
--    with ATTRIBUTES doc_type, sku_id, over `meridian_unified_text`,
--    running on `meridian_wh` with a 1 hour TARGET_LAG.
-- 3. Create `meridian_sales_model` as a SEMANTIC VIEW over pos_sales,
--    product_catalog, and loyalty_members, joining pos_sales(sku_id) to
--    product_catalog(sku_id) and pos_sales(member_id) to
--    loyalty_members(member_id), with metrics sell_through_units (SUM of
--    units_sold) and shelf_revenue (SUM of units_sold * unit_price).
-- ==========================================================

-- YOUR CODE:

CREATE OR REPLACE TABLE promo_flyers_parsed AS
SELECT
    relative_path AS doc_name,
    AI_PARSE_DOCUMENT(
        TO_FILE('@___', relative_path),
        {'mode':'___'}
    ) AS parsed_doc
FROM DIRECTORY(@___);

CREATE OR REPLACE TABLE promo_terms AS
SELECT
    doc_name,
    AI_EXTRACT(
        parsed_doc:content::STRING,
        [___, ___, ___, ___]
    ) AS terms
FROM ___;

CREATE OR REPLACE CORTEX SEARCH SERVICE ___
    ON ___
    ATTRIBUTES ___, ___
    WAREHOUSE = ___
    TARGET_LAG = '___'
AS
SELECT sku_id, doc_type, search_text
FROM meridian_unified_text;

CREATE OR REPLACE SEMANTIC VIEW meridian_sales_model
    TABLES (___, ___, ___)
    RELATIONSHIPS (
        pos_sales(___) REFERENCES product_catalog(___),
        pos_sales(___) REFERENCES loyalty_members(___)
    )
    METRICS (
        pos_sales.sell_through_units AS ___(units_sold),
        pos_sales.shelf_revenue AS SUM(___ * ___)
    );

-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
CREATE OR REPLACE TABLE promo_flyers_parsed AS
SELECT
    relative_path AS doc_name,
    AI_PARSE_DOCUMENT(
        TO_FILE('@meridian_promo_stage', relative_path),
        {'mode':'LAYOUT'}
    ) AS parsed_doc
FROM DIRECTORY(@meridian_promo_stage);

CREATE OR REPLACE TABLE promo_terms AS
SELECT
    doc_name,
    AI_EXTRACT(
        parsed_doc:content::STRING,
        ['sku_list','discount_pct','promo_start','promo_end']
    ) AS terms
FROM promo_flyers_parsed;

CREATE OR REPLACE CORTEX SEARCH SERVICE meridian_promo_search
    ON search_text
    ATTRIBUTES doc_type, sku_id
    WAREHOUSE = meridian_wh
    TARGET_LAG = '1 hour'
AS
SELECT sku_id, doc_type, search_text
FROM meridian_unified_text;

CREATE OR REPLACE SEMANTIC VIEW meridian_sales_model
    TABLES (pos_sales, product_catalog, loyalty_members)
    RELATIONSHIPS (
        pos_sales(sku_id) REFERENCES product_catalog(sku_id),
        pos_sales(member_id) REFERENCES loyalty_members(member_id)
    )
    METRICS (
        pos_sales.sell_through_units AS SUM(units_sold),
        pos_sales.shelf_revenue AS SUM(units_sold * unit_price)
    );
*/


-- ==========================================================
-- Exercise 22.2 — Forecast, Detect, and Route Through One Agent
-- Course: 1018
-- Module 22 (Capstone)
-- ==========================================================
--
-- The capstone's second half (lesson_85) closes the two remaining silos —
-- forward-looking risk and cross-signal support — with one forecast model,
-- one anomaly model, and one orchestrating agent that decides which tool a
-- question needs. Here you'll stand up the exact same two ML objects and
-- agent registration the capstone used, then close the loop with the
-- masking policy and resource monitor from lesson_86 so every layer stays
-- governed.
--
-- Your task:
-- 1. Create `meridian_sku_forecast` with SNOWFLAKE.ML.FORECAST over the
--    `pos_sales_daily` view, using sale_date as the timestamp column,
--    units_sold as the target, and sku_id as the series column.
-- 2. Create `meridian_shipment_anomaly` with SNOWFLAKE.ML.ANOMALY_DETECTION
--    over `vendor_shipment_events`, using shipment_date as the timestamp
--    column, lead_time_days as the target, and vendor_id as the series
--    column, then CALL its DETECT_ANOMALIES against
--    `vendor_shipment_events_new`.
-- 3. Register `meridian_intelligence_agent` with TOOLS for the search
--    service, the analyst semantic view, ML_FORECAST, and
--    ML_ANOMALY_DETECTION, and an INSTRUCTIONS string routing document
--    questions to search, sales questions to analyst, trend questions to
--    forecast, and risk questions to anomaly detection.
-- 4. Add a masking policy `mask_member_pii` on loyalty_members.email that
--    unmasks only for CURRENT_ROLE() = 'COMPLIANCE_ANALYST', and a
--    RESOURCE MONITOR `meridian_wh_guard` with a 500-credit quota that
--    notifies at 90% and suspends at 100%, attached to `meridian_wh`.
-- ==========================================================

-- YOUR CODE:

CREATE SNOWFLAKE.ML.FORECAST meridian_sku_forecast(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', '___'),
    TIMESTAMP_COLNAME => '___',
    TARGET_COLNAME => '___',
    SERIES_COLNAME => '___'
);

CREATE SNOWFLAKE.ML.ANOMALY_DETECTION meridian_shipment_anomaly(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', '___'),
    TIMESTAMP_COLNAME => '___',
    TARGET_COLNAME => '___',
    LABEL_COLNAME => '',
    SERIES_COLNAME => '___'
);

CALL meridian_shipment_anomaly!DETECT_ANOMALIES(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', '___'),
    TIMESTAMP_COLNAME => '___',
    TARGET_COLNAME => '___',
    SERIES_COLNAME => '___'
);

-- Illustrative only — exact object/DDL syntax is version-dependent, verify
-- current docs (current Cortex Agents are defined FROM SPECIFICATION $$ <yaml> $$,
-- not this simplified TOOLS=/INSTRUCTIONS= form)
CREATE OR REPLACE AGENT meridian_intelligence_agent
    TOOLS = (
        CORTEX_SEARCH_SERVICE '___',
        CORTEX_ANALYST_SERVICE '___',
        ML_FORECAST '___',
        ML_ANOMALY_DETECTION '___'
    )
    INSTRUCTIONS = '___';

CREATE MASKING POLICY mask_member_pii AS (val STRING) RETURNS STRING ->
    CASE WHEN CURRENT_ROLE() = '___' THEN val
         ELSE '___' END;

ALTER TABLE loyalty_members
    MODIFY COLUMN email SET MASKING POLICY ___;

CREATE RESOURCE MONITOR meridian_wh_guard
    WITH CREDIT_QUOTA = ___
    TRIGGERS ON ___ PERCENT DO NOTIFY
             ON ___ PERCENT DO SUSPEND;

ALTER WAREHOUSE meridian_wh SET RESOURCE_MONITOR = ___;

-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
CREATE SNOWFLAKE.ML.FORECAST meridian_sku_forecast(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'pos_sales_daily'),
    TIMESTAMP_COLNAME => 'sale_date',
    TARGET_COLNAME => 'units_sold',
    SERIES_COLNAME => 'sku_id'
);

CREATE SNOWFLAKE.ML.ANOMALY_DETECTION meridian_shipment_anomaly(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'vendor_shipment_events'),
    TIMESTAMP_COLNAME => 'shipment_date',
    TARGET_COLNAME => 'lead_time_days',
    LABEL_COLNAME => '',
    SERIES_COLNAME => 'vendor_id'
);

CALL meridian_shipment_anomaly!DETECT_ANOMALIES(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'vendor_shipment_events_new'),
    TIMESTAMP_COLNAME => 'shipment_date',
    TARGET_COLNAME => 'lead_time_days',
    SERIES_COLNAME => 'vendor_id'
);

-- Illustrative only — exact object/DDL syntax is version-dependent, verify
-- current docs (current Cortex Agents are defined FROM SPECIFICATION $$ <yaml> $$,
-- not this simplified TOOLS=/INSTRUCTIONS= form)
CREATE OR REPLACE AGENT meridian_intelligence_agent
    TOOLS = (
        CORTEX_SEARCH_SERVICE 'meridian_field_search',
        CORTEX_ANALYST_SERVICE 'meridian_sales_model',
        ML_FORECAST 'meridian_sku_forecast',
        ML_ANOMALY_DETECTION 'meridian_shipment_anomaly'
    )
    INSTRUCTIONS = 'Route document/review questions to search,
        structured sales questions to analyst, trend questions
        to forecast, and risk questions to anomaly detection.';

CREATE MASKING POLICY mask_member_pii AS (val STRING) RETURNS STRING ->
    CASE WHEN CURRENT_ROLE() = 'COMPLIANCE_ANALYST' THEN val
         ELSE '***MASKED***' END;

ALTER TABLE loyalty_members
    MODIFY COLUMN email SET MASKING POLICY mask_member_pii;

CREATE RESOURCE MONITOR meridian_wh_guard
    WITH CREDIT_QUOTA = 500
    TRIGGERS ON 90 PERCENT DO NOTIFY
             ON 100 PERCENT DO SUSPEND;

ALTER WAREHOUSE meridian_wh SET RESOURCE_MONITOR = meridian_wh_guard;
*/
