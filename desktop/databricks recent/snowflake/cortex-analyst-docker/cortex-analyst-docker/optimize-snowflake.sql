-- =============================================================================
-- Snowflake Warehouse Optimization (BUDGET MODE)
-- =============================================================================
-- Run this in Snowflake worksheet or via snowsql:
--   snowsql -a EMXEKCM-PC15902 -u PRAKHAR1207SRIVASTAVA -f optimize-snowflake.sql
-- =============================================================================

-- Budget config: XSMALL warehouse, auto-suspend after 60s, auto-resume on query
-- Estimated cost: ~$2-5/month (only bills for active query time)

ALTER WAREHOUSE COMPUTE_WH SET
    WAREHOUSE_SIZE = XSMALL
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE;

-- Verify settings
SHOW WAREHOUSES LIKE 'COMPUTE_WH';

-- Note: If XSMALL is too slow for concurrent users, upgrade to SMALL:
-- ALTER WAREHOUSE COMPUTE_WH SET WAREHOUSE_SIZE = SMALL AUTO_SUSPEND = 60;

-- Optional: Set query timeout at warehouse level (30s)
ALTER WAREHOUSE COMPUTE_WH SET STATEMENT_TIMEOUT_IN_SECONDS = 30;