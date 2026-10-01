-- =====================================================
-- Silver Layer - Sales Transaction Dynamic Tables
-- Deduplication, data quality checks, incremental refresh
-- =====================================================

-- =====================================================
-- SALES_HEADER
-- =====================================================
CREATE OR REPLACE DYNAMIC TABLE SALES_DEV.SILVER.SALES_HEADER
    TARGET_LAG = DOWNSTREAM
    WAREHOUSE = COMPUTE_WH
    REFRESH_MODE = INCREMENTAL
    INITIALIZE = ON_CREATE
    COMMENT = 'Silver layer sales header with deduplication and quality validation'
AS
SELECT
    TRANSACTION_ID,
    TRANSACTION_NUMBER,
    TRANSACTION_TIMESTAMP,
    CUSTOMER_ID,
    STORE_ID,
    CHANNEL_ID,
    PAYMENT_METHOD,
    CURRENCY,
    GROSS_AMOUNT,
    TOTAL_DISCOUNT,
    TOTAL_TAX,
    NET_TOTAL,
    CREATED_AT,
    
    -- Data Quality Flag
    CASE 
        WHEN TRANSACTION_ID IS NULL THEN FALSE
        WHEN TRANSACTION_NUMBER IS NULL THEN FALSE
        WHEN CUSTOMER_ID IS NULL THEN FALSE
        WHEN STORE_ID IS NULL THEN FALSE
        WHEN CURRENCY IS NULL THEN FALSE
        WHEN GROSS_AMOUNT < 0 THEN FALSE
        WHEN TOTAL_DISCOUNT < 0 THEN FALSE
        WHEN TOTAL_TAX < 0 THEN FALSE
        WHEN NET_TOTAL < 0 THEN FALSE
        ELSE TRUE
    END AS IS_VALID_RECORD,
    
    -- Audit Columns
    __FILE_NAME,
    __ROW_NUMBER AS __SOURCE_ROW_NUMBER,
    __LOAD_TS AS __BRONZE_LOAD_TS

FROM SALES_DEV.BRONZE.SALES_HEADER

-- Deduplication: Keep latest record per TRANSACTION_ID
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TRANSACTION_ID 
    ORDER BY __LOAD_TS DESC, __ROW_NUMBER DESC
) = 1;

-- =====================================================
-- SALES_ITEM
-- =====================================================
CREATE OR REPLACE DYNAMIC TABLE SALES_DEV.SILVER.SALES_ITEM
    TARGET_LAG = DOWNSTREAM
    WAREHOUSE = COMPUTE_WH
    REFRESH_MODE = INCREMENTAL
    INITIALIZE = ON_CREATE
    COMMENT = 'Silver layer sales item with deduplication and quality validation'
AS
SELECT
    TRANSACTION_LINE_ID,
    TRANSACTION_ID,
    SKU_CODE,
    QUANTITY,
    UNIT_PRICE,
    DISCOUNT_AMOUNT,
    TAX_AMOUNT,
    LINE_TOTAL,
    CREATED_AT,
    
    -- Data Quality Flag
    CASE 
        WHEN TRANSACTION_LINE_ID IS NULL THEN FALSE
        WHEN TRANSACTION_ID IS NULL THEN FALSE
        WHEN SKU_CODE IS NULL THEN FALSE
        WHEN QUANTITY IS NULL OR QUANTITY <= 0 THEN FALSE
        WHEN UNIT_PRICE < 0 THEN FALSE
        WHEN DISCOUNT_AMOUNT < 0 THEN FALSE
        WHEN TAX_AMOUNT < 0 THEN FALSE
        WHEN LINE_TOTAL < 0 THEN FALSE
        ELSE TRUE
    END AS IS_VALID_RECORD,
    
    -- Audit Columns
    __FILE_NAME,
    __ROW_NUMBER AS __SOURCE_ROW_NUMBER,
    __LOAD_TS AS __BRONZE_LOAD_TS

FROM SALES_DEV.BRONZE.SALES_ITEM

-- Deduplication: Keep latest record per TRANSACTION_LINE_ID
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TRANSACTION_LINE_ID 
    ORDER BY __LOAD_TS DESC, __ROW_NUMBER DESC
) = 1;
