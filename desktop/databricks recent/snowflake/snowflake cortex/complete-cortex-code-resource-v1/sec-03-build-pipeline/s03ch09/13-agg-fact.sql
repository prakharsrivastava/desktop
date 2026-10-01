-- =====================================================
-- Gold Layer - Aggregated Fact Tables
-- Pre-computed summaries for dashboard performance
-- =====================================================

-- =====================================================
-- FACT_SALES_DAILY - Daily Grain Aggregation
-- Use for: Daily trends, day-of-week analysis
-- =====================================================
CREATE OR REPLACE DYNAMIC TABLE SALES_DEV.GOLD.FACT_SALES_DAILY
    TARGET_LAG = '5 minutes'
    WAREHOUSE = COMPUTE_WH
    REFRESH_MODE = INCREMENTAL
    INITIALIZE = ON_CREATE
    COMMENT = 'Daily aggregated sales fact for executive dashboards and trend analysis'
AS
SELECT
    -- Date Attributes
    d.DATE_DIM_KEY,
    d.DATE_KEY,
    d.YEAR_NUMBER,
    d.QUARTER_NUMBER,
    d.MONTH_NUMBER,
    d.WEEK_NUMBER,
    d.YEAR_MONTH,
    d.MONTH_NAME,
    d.DAY_NAME,
    d.IS_WEEKEND,
    
    -- Geography
    cn.COUNTRY_DIM_KEY,
    cn.COUNTRY_CODE,
    cn.COUNTRY_NAME,
    cn.REGION_CODE,
    cn.REGION_NAME,
    
    -- Store
    s.STORE_DIM_KEY,
    s.STORE_CODE,
    s.STORE_NAME,
    s.FORMAT_CODE,
    
    -- Channel
    fh.CHANNEL_ID,
    fh.CURRENCY,
    
    -- Measures
    COUNT(*) AS TRANSACTION_COUNT,
    SUM(fh.GROSS_AMOUNT) AS TOTAL_GROSS_AMOUNT,
    SUM(fh.TOTAL_DISCOUNT) AS TOTAL_DISCOUNT,
    SUM(fh.TOTAL_TAX) AS TOTAL_TAX,
    SUM(fh.NET_TOTAL) AS TOTAL_NET_SALES,
    AVG(fh.NET_TOTAL) AS AVG_TRANSACTION_VALUE,
    COUNT(DISTINCT fh.CUSTOMER_DIM_KEY) AS UNIQUE_CUSTOMERS

FROM SALES_DEV.GOLD.FACT_SALES_HEADER fh
JOIN SALES_DEV.GOLD.DIM_DATE d ON fh.DATE_DIM_KEY = d.DATE_DIM_KEY
JOIN SALES_DEV.GOLD.DIM_STORE s ON fh.STORE_DIM_KEY = s.STORE_DIM_KEY
JOIN SALES_DEV.GOLD.DIM_COUNTRY cn ON fh.COUNTRY_DIM_KEY = cn.COUNTRY_DIM_KEY
GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21;

-- =====================================================
-- FACT_SALES_WEEKLY - Weekly Grain Aggregation
-- Use for: Week-over-week comparisons, weekly reports
-- =====================================================
CREATE OR REPLACE DYNAMIC TABLE SALES_DEV.GOLD.FACT_SALES_WEEKLY
    TARGET_LAG = '7 days'
    WAREHOUSE = COMPUTE_WH
    REFRESH_MODE = INCREMENTAL
    INITIALIZE = ON_CREATE
    COMMENT = 'Weekly aggregated sales fact for trend analysis and WoW comparisons'
AS
SELECT
    -- Time Period
    d.YEAR_NUMBER,
    d.WEEK_NUMBER,
    MIN(d.DATE_KEY) AS WEEK_START_DATE,
    MAX(d.DATE_KEY) AS WEEK_END_DATE,
    
    -- Geography
    cn.COUNTRY_DIM_KEY,
    cn.COUNTRY_CODE,
    cn.COUNTRY_NAME,
    cn.REGION_CODE,
    cn.REGION_NAME,
    
    -- Channel
    fh.CHANNEL_ID,
    fh.CURRENCY,
    
    -- Measures
    COUNT(*) AS TRANSACTION_COUNT,
    SUM(fh.GROSS_AMOUNT) AS TOTAL_GROSS_AMOUNT,
    SUM(fh.TOTAL_DISCOUNT) AS TOTAL_DISCOUNT,
    SUM(fh.TOTAL_TAX) AS TOTAL_TAX,
    SUM(fh.NET_TOTAL) AS TOTAL_NET_SALES,
    AVG(fh.NET_TOTAL) AS AVG_TRANSACTION_VALUE,
    COUNT(DISTINCT fh.CUSTOMER_DIM_KEY) AS UNIQUE_CUSTOMERS,
    COUNT(DISTINCT fh.STORE_DIM_KEY) AS ACTIVE_STORES,
    COUNT(DISTINCT d.DATE_KEY) AS SELLING_DAYS

FROM SALES_DEV.GOLD.FACT_SALES_HEADER fh
JOIN SALES_DEV.GOLD.DIM_DATE d ON fh.DATE_DIM_KEY = d.DATE_DIM_KEY
JOIN SALES_DEV.GOLD.DIM_COUNTRY cn ON fh.COUNTRY_DIM_KEY = cn.COUNTRY_DIM_KEY
GROUP BY 1,2,5,6,7,8,9,10,11;

-- =====================================================
-- FACT_SALES_MONTHLY - Monthly Grain Aggregation
-- Use for: Executive reporting, MoM/YoY analysis
-- =====================================================
CREATE OR REPLACE DYNAMIC TABLE SALES_DEV.GOLD.FACT_SALES_MONTHLY
    TARGET_LAG = '30 days'
    WAREHOUSE = COMPUTE_WH
    REFRESH_MODE = INCREMENTAL
    INITIALIZE = ON_CREATE
    COMMENT = 'Monthly aggregated sales fact for executive reporting and MoM/YoY analysis'
AS
SELECT
    -- Calendar Time
    d.YEAR_NUMBER,
    d.MONTH_NUMBER,
    d.YEAR_MONTH,
    d.MONTH_NAME,
    d.QUARTER_NUMBER,
    d.YEAR_QUARTER,
    
    -- Fiscal Time
    d.FISCAL_YEAR,
    d.FISCAL_QUARTER,
    
    -- Period Dates
    MIN(d.DATE_KEY) AS MONTH_START_DATE,
    MAX(d.DATE_KEY) AS MONTH_END_DATE,
    
    -- Geography
    cn.COUNTRY_DIM_KEY,
    cn.COUNTRY_CODE,
    cn.COUNTRY_NAME,
    cn.REGION_CODE,
    cn.REGION_NAME,
    
    -- Channel
    fh.CHANNEL_ID,
    fh.CURRENCY,
    
    -- Measures
    COUNT(*) AS TRANSACTION_COUNT,
    SUM(fh.GROSS_AMOUNT) AS TOTAL_GROSS_AMOUNT,
    SUM(fh.TOTAL_DISCOUNT) AS TOTAL_DISCOUNT,
    SUM(fh.TOTAL_TAX) AS TOTAL_TAX,
    SUM(fh.NET_TOTAL) AS TOTAL_NET_SALES,
    AVG(fh.NET_TOTAL) AS AVG_TRANSACTION_VALUE,
    COUNT(DISTINCT fh.CUSTOMER_DIM_KEY) AS UNIQUE_CUSTOMERS,
    COUNT(DISTINCT fh.STORE_DIM_KEY) AS ACTIVE_STORES,
    COUNT(DISTINCT d.DATE_KEY) AS SELLING_DAYS

FROM SALES_DEV.GOLD.FACT_SALES_HEADER fh
JOIN SALES_DEV.GOLD.DIM_DATE d ON fh.DATE_DIM_KEY = d.DATE_DIM_KEY
JOIN SALES_DEV.GOLD.DIM_COUNTRY cn ON fh.COUNTRY_DIM_KEY = cn.COUNTRY_DIM_KEY
GROUP BY 1,2,3,4,5,6,7,8,11,12,13,14,15,16,17;
