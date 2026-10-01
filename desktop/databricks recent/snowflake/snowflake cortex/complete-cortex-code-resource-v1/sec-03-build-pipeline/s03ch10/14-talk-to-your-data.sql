-- =====================================================
-- Cortex Analyst Setup - Talk to Your Sales Data
-- Semantic Model for Natural Language Queries
-- =====================================================

-- Create stage for semantic model YAML
CREATE OR REPLACE STAGE SALES_DEV.GOLD.SEMANTIC_MODELS
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for Cortex Analyst semantic model files';

-- Remove old semantic model (if exists)
REMOVE @SALES_DEV.GOLD.SEMANTIC_MODELS/sales_semantic_model.yaml;

-- Upload the semantic model YAML file to the stage
COPY FILES INTO @SALES_DEV.GOLD.SEMANTIC_MODELS
FROM 'snow://workspace/USER$.PUBLIC.DEFAULT$/versions/live/'
FILES=('sales_semantic_model.yaml');

-- Verify the file is uploaded
LIST @SALES_DEV.GOLD.SEMANTIC_MODELS;

-- =====================================================
-- Semantic Model Features
-- =====================================================
-- The semantic model includes:
-- 
-- TABLES:
--   - dim_customer: Customer demographics, loyalty tiers
--   - dim_product: Product hierarchy (category > family > model > SKU)
--   - dim_store: Store locations, formats, geography
--   - dim_country: Countries with regions, currencies, tax info
--   - dim_date: Calendar dimension with fiscal periods
--   - fact_sales_header: Transaction-level sales
--   - fact_sales_item: Line item-level sales
--
-- RELATIONSHIPS (10 defined):
--   - fact_sales_header -> dim_customer, dim_store, dim_country, dim_date
--   - fact_sales_item -> dim_product, dim_customer, dim_store, dim_country, dim_date
--   - fact_sales_item -> fact_sales_header
--
-- METRICS:
--   - total_net_sales, total_transactions, average_transaction_value
--   - total_units_sold, total_line_revenue, average_unit_price
--   - customer_count, product_count, store_count
--
-- VERIFIED QUERIES (8 examples):
--   - Total sales by region
--   - Top 10 products by revenue
--   - Monthly sales trend
--   - Sales by loyalty tier
--   - Top performing stores
--   - Sales by product category
--   - POS vs Web channel comparison
--   - Quarterly sales
--
-- =====================================================
-- Using Cortex Analyst
-- =====================================================
-- 1. Via Snowflake UI: Go to AI & ML > Cortex Analyst
--    Select: @SALES_DEV.GOLD.SEMANTIC_MODELS/sales_semantic_model.yaml
--
-- 2. Via REST API:
--    POST https://<account>.snowflakecomputing.com/api/v2/cortex/analyst/message
--    {
--      "messages": [{"role": "user", "content": [{"type": "text", "text": "Your question"}]}],
--      "semantic_model_file": "@SALES_DEV.GOLD.SEMANTIC_MODELS/sales_semantic_model.yaml"
--    }
--
-- Sample Questions to Ask:
-- - "What are the total sales by region?"
-- - "Show me top 10 products by revenue"
-- - "What is the monthly sales trend?"
-- - "Which stores have the highest sales?"
-- - "What are sales by customer loyalty tier?"
-- - "Compare sales between Q1 and Q2"
-- - "Which product categories are growing fastest?"
-- - "What is the average transaction value by channel?"
-- - "Show me sales for iPhone products"
-- - "Who are the top customers by total spend?"
-- =====================================================
