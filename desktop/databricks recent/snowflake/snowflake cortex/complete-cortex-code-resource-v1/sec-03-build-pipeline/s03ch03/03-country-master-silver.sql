-- =====================================================
-- Silver Layer - REGION_MASTER Dynamic Table
-- Deduplication, data quality checks, incremental refresh
-- =====================================================

CREATE OR REPLACE DYNAMIC TABLE SALES_DEV.SILVER.REGION_MASTER
    TARGET_LAG = DOWNSTREAM
    WAREHOUSE = COMPUTE_WH
    REFRESH_MODE = INCREMENTAL
    INITIALIZE = ON_CREATE
    COMMENT = 'Silver layer region master with deduplication and data quality validation'
AS
SELECT
    REGION_CODE,
    REGION_NAME,
    IS_ACTIVE,
    EFFECTIVE_START_DATE,
    EFFECTIVE_END_DATE,
    CREATED_AT,
    SOURCE_SYSTEM,
    
    -- Data Quality Flags
    CASE 
        WHEN REGION_CODE IS NULL THEN FALSE
        WHEN REGION_NAME IS NULL OR TRIM(REGION_NAME) = '' THEN FALSE
        WHEN IS_ACTIVE NOT IN ('Y', 'N') THEN FALSE
        WHEN EFFECTIVE_START_DATE IS NULL THEN FALSE
        WHEN EFFECTIVE_END_DATE < EFFECTIVE_START_DATE THEN FALSE
        ELSE TRUE
    END AS IS_VALID_RECORD,
    
    -- Audit Columns
    __FILE_NAME,
    __ROW_NUMBER AS __SOURCE_ROW_NUMBER,
    __LOAD_TS AS __BRONZE_LOAD_TS

FROM SALES_DEV.BRONZE.REGION_MASTER

-- Deduplication: Keep latest record per REGION_CODE based on load timestamp
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY REGION_CODE 
    ORDER BY __LOAD_TS DESC, __ROW_NUMBER DESC
) = 1;

-- =====================================================
-- Silver Layer - COUNTRY_MASTER Dynamic Table
-- Deduplication, data quality checks, incremental refresh
-- =====================================================

CREATE OR REPLACE DYNAMIC TABLE SALES_DEV.SILVER.COUNTRY_MASTER
    TARGET_LAG = DOWNSTREAM
    WAREHOUSE = COMPUTE_WH
    REFRESH_MODE = INCREMENTAL
    INITIALIZE = ON_CREATE
    COMMENT = 'Silver layer country master with deduplication and data quality validation'
AS
SELECT
    COUNTRY_CODE,
    COUNTRY_NAME,
    REGION_CODE,
    CURRENCY_CODE,
    TAX_CODE,
    PRIMARY_LANGUAGE,
    TIMEZONE,
    ECOMMERCE_SUPPORTED,
    RETAIL_STORE_SUPPORTED,
    MARKET_TIER,
    
    -- Data Quality Flag
    CASE 
        WHEN COUNTRY_CODE IS NULL THEN FALSE
        WHEN COUNTRY_NAME IS NULL OR TRIM(COUNTRY_NAME) = '' THEN FALSE
        WHEN REGION_CODE IS NULL THEN FALSE
        WHEN CURRENCY_CODE IS NULL THEN FALSE
        WHEN ECOMMERCE_SUPPORTED NOT IN ('Y', 'N') THEN FALSE
        WHEN RETAIL_STORE_SUPPORTED NOT IN ('Y', 'N') THEN FALSE
        ELSE TRUE
    END AS IS_VALID_RECORD,
    
    -- Audit Columns
    __FILE_NAME,
    __ROW_NUMBER AS __SOURCE_ROW_NUMBER,
    __LOAD_TS AS __BRONZE_LOAD_TS

FROM SALES_DEV.BRONZE.COUNTRY_MASTER

-- Deduplication: Keep latest record per COUNTRY_CODE
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY COUNTRY_CODE 
    ORDER BY __LOAD_TS DESC, __ROW_NUMBER DESC
) = 1;

-- =====================================================
-- Silver Layer - CURRENCY_MASTER Dynamic Table
-- Deduplication, data quality checks, incremental refresh
-- =====================================================

CREATE OR REPLACE DYNAMIC TABLE SALES_DEV.SILVER.CURRENCY_MASTER
    TARGET_LAG = DOWNSTREAM
    WAREHOUSE = COMPUTE_WH
    REFRESH_MODE = INCREMENTAL
    INITIALIZE = ON_CREATE
    COMMENT = 'Silver layer currency master with deduplication and data quality validation'
AS
SELECT
    CURRENCY_CODE,
    CURRENCY_NAME,
    CURRENCY_SYMBOL,
    MINOR_UNIT,
    IS_ACTIVE,
    EFFECTIVE_START_DATE,
    EFFECTIVE_END_DATE,
    CREATED_AT,
    SOURCE_SYSTEM,
    
    -- Data Quality Flag
    CASE 
        WHEN CURRENCY_CODE IS NULL THEN FALSE
        WHEN CURRENCY_NAME IS NULL OR TRIM(CURRENCY_NAME) = '' THEN FALSE
        WHEN IS_ACTIVE NOT IN ('Y', 'N') THEN FALSE
        WHEN EFFECTIVE_START_DATE IS NULL THEN FALSE
        WHEN EFFECTIVE_END_DATE < EFFECTIVE_START_DATE THEN FALSE
        ELSE TRUE
    END AS IS_VALID_RECORD,
    
    -- Audit Columns
    __FILE_NAME,
    __ROW_NUMBER AS __SOURCE_ROW_NUMBER,
    __LOAD_TS AS __BRONZE_LOAD_TS

FROM SALES_DEV.BRONZE.CURRENCY_MASTER

-- Deduplication: Keep latest record per CURRENCY_CODE
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY CURRENCY_CODE 
    ORDER BY __LOAD_TS DESC, __ROW_NUMBER DESC
) = 1;

-- =====================================================
-- Silver Layer - TAX_MASTER Dynamic Table
-- Deduplication, data quality checks, incremental refresh
-- =====================================================

CREATE OR REPLACE DYNAMIC TABLE SALES_DEV.SILVER.TAX_MASTER
    TARGET_LAG = DOWNSTREAM
    WAREHOUSE = COMPUTE_WH
    REFRESH_MODE = INCREMENTAL
    INITIALIZE = ON_CREATE
    COMMENT = 'Silver layer tax master with deduplication and data quality validation'
AS
SELECT
    TAX_CODE,
    TAX_TYPE,
    TAX_RATE,
    TAX_INCLUSIVE_FLAG,
    EFFECTIVE_START_DATE,
    EFFECTIVE_END_DATE,
    IS_ACTIVE,
    CREATED_AT,
    SOURCE_SYSTEM,
    
    -- Data Quality Flag
    CASE 
        WHEN TAX_CODE IS NULL THEN FALSE
        WHEN TAX_TYPE IS NULL OR TRIM(TAX_TYPE) = '' THEN FALSE
        WHEN TAX_RATE IS NULL OR TAX_RATE < 0 OR TAX_RATE > 1 THEN FALSE
        WHEN TAX_INCLUSIVE_FLAG NOT IN ('Y', 'N') THEN FALSE
        WHEN IS_ACTIVE NOT IN ('Y', 'N') THEN FALSE
        WHEN EFFECTIVE_START_DATE IS NULL THEN FALSE
        WHEN EFFECTIVE_END_DATE < EFFECTIVE_START_DATE THEN FALSE
        ELSE TRUE
    END AS IS_VALID_RECORD,
    
    -- Audit Columns
    __FILE_NAME,
    __ROW_NUMBER AS __SOURCE_ROW_NUMBER,
    __LOAD_TS AS __BRONZE_LOAD_TS

FROM SALES_DEV.BRONZE.TAX_MASTER

-- Deduplication: Keep latest record per TAX_CODE
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TAX_CODE 
    ORDER BY __LOAD_TS DESC, __ROW_NUMBER DESC
) = 1;
