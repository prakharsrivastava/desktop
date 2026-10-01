-- ============================================================
-- Course 1018 — Module 22
-- On-slide QUERIES reference  (7 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- Vendor Contracts to Structured Rows
-- ----------------------------------------------------------

CREATE OR REPLACE TABLE vendor_contracts_parsed AS
SELECT
    relative_path AS doc_name,
    AI_PARSE_DOCUMENT(
        TO_FILE('@meridian_vendor_stage', relative_path),
        {'mode':'LAYOUT'}
    ) AS parsed_doc
FROM DIRECTORY(@meridian_vendor_stage);

CREATE OR REPLACE TABLE vendor_terms AS
SELECT
    doc_name,
    AI_EXTRACT(
        parsed_doc:content::STRING,
        ['vendor_name','lead_time_days','payment_terms','sku_list']
    ) AS terms
FROM vendor_contracts_parsed;


-- ----------------------------------------------------------
-- Standing Up the Search Service
-- ----------------------------------------------------------

CREATE OR REPLACE CORTEX SEARCH SERVICE meridian_field_search
    ON search_text
    ATTRIBUTES doc_type, sku_id
    WAREHOUSE = meridian_wh
    TARGET_LAG = '1 hour'
AS
SELECT sku_id, doc_type, search_text
FROM meridian_unified_text;


-- ----------------------------------------------------------
-- Defining Meridian's Semantic View
-- ----------------------------------------------------------

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


-- ----------------------------------------------------------
-- Forecasting Sell-Through, Per SKU
-- ----------------------------------------------------------

CREATE SNOWFLAKE.ML.FORECAST meridian_sku_forecast(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'pos_sales_daily'),
    TIMESTAMP_COLNAME => 'sale_date',
    TARGET_COLNAME => 'units_sold',
    SERIES_COLNAME => 'sku_id'
);


-- ----------------------------------------------------------
-- Wiring the Anomaly Model
-- ----------------------------------------------------------

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


-- ----------------------------------------------------------
-- Registering the Agent's Tools
-- ----------------------------------------------------------

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


-- ----------------------------------------------------------
-- Masking and Guardrails, Applied
-- ----------------------------------------------------------

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

