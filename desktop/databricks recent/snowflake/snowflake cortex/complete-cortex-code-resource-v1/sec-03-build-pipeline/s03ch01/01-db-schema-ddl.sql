-- =====================================================
-- Sales Cloud Data Platform - Database DDL
-- DEV & QA: Transient databases (no fail-safe, 0-1 day time travel)
-- PROD: Permanent database with 7 days time travel
-- =====================================================

-- DEV Database (Transient)
CREATE OR REPLACE TRANSIENT DATABASE SALES_DEV
    DATA_RETENTION_TIME_IN_DAYS = 1
    COMMENT = 'Development environment for Sales - Transient database';

-- QA Database (Transient)
CREATE OR REPLACE TRANSIENT DATABASE SALES_QA
    DATA_RETENTION_TIME_IN_DAYS = 1
    COMMENT = 'QA/Testing environment for Sales - Transient database';

-- PROD Database (Permanent with 7 days time travel)
CREATE OR REPLACE DATABASE SALES_PROD
    DATA_RETENTION_TIME_IN_DAYS = 7
    COMMENT = 'Production environment for Sales - 7 days time travel enabled';

-- =====================================================
-- DEV Schemas
-- =====================================================
CREATE OR REPLACE SCHEMA SALES_DEV.BRONZE
    COMMENT = 'Landing zone - raw data ingestion with minimal transformation';

CREATE OR REPLACE SCHEMA SALES_DEV.SILVER
    COMMENT = 'Cleansed & curated data with deduplication and standardization';

CREATE OR REPLACE SCHEMA SALES_DEV.GOLD
    COMMENT = 'Business-ready aggregated data for reporting and analytics';

CREATE OR REPLACE SCHEMA SALES_DEV.COMMON
    COMMENT = 'Shared utilities, functions, and stored procedures';

-- =====================================================
-- QA Schemas
-- =====================================================
CREATE OR REPLACE SCHEMA SALES_QA.BRONZE
    COMMENT = 'Landing zone - raw data ingestion with minimal transformation';

CREATE OR REPLACE SCHEMA SALES_QA.SILVER
    COMMENT = 'Cleansed & curated data with deduplication and standardization';

CREATE OR REPLACE SCHEMA SALES_QA.GOLD
    COMMENT = 'Business-ready aggregated data for reporting and analytics';

CREATE OR REPLACE SCHEMA SALES_QA.COMMON
    COMMENT = 'Shared utilities, functions, and stored procedures';

-- =====================================================
-- PROD Schemas
-- =====================================================
CREATE OR REPLACE SCHEMA SALES_PROD.BRONZE
    COMMENT = 'Landing zone - raw data ingestion with minimal transformation';

CREATE OR REPLACE SCHEMA SALES_PROD.SILVER
    COMMENT = 'Cleansed & curated data with deduplication and standardization';

CREATE OR REPLACE SCHEMA SALES_PROD.GOLD
    COMMENT = 'Business-ready aggregated data for reporting and analytics';

CREATE OR REPLACE SCHEMA SALES_PROD.COMMON
    COMMENT = 'Shared utilities, functions, and stored procedures';
