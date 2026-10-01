-- ============================================================
-- 1018 — COMPLETE QUERIES REFERENCE  (125 snippets)
-- Every on-slide snippet across all modules. Ctrl+F by keyword or title.
-- ============================================================



-- ########################################################
-- MODULE 01
-- ########################################################

-- --- One Query, Not Five Dashboards ---

SELECT product_id, AI_SENTIMENT(review_text) AS sentiment
FROM customer_reviews
WHERE product_id = 'SKU-48213';


-- --- Task Functions — Narrow and Fast ---

SELECT review_id,
       AI_SENTIMENT(review_text) AS sentiment,
       AI_TRANSLATE(review_text, '', 'en') AS review_en,
       SNOWFLAKE.CORTEX.SUMMARIZE(review_text) AS summary
FROM product_reviews;


-- --- AI_COMPLETE — When the Question Isn't Standard ---

SELECT AI_COMPLETE(
  'llama3.1-70b',
  'Summarize why negative reviews spiked for SKU-48213 in the last 30 days: ' || review_text
)
FROM product_reviews
WHERE product_id = 'SKU-48213';


-- --- Debug Slide — Why Did My Nightly Job Time Out? ---

-- FIX: one batch task-function pass replaces per-row Cortex Search calls
SELECT review_id, AI_SENTIMENT(review_text) AS sentiment
FROM product_reviews;
-- A single SELECT touches every row — no loop, no per-row round trip


-- --- ANTI-PATTERN — Calling AI_COMPLETE Per Click ---

-- ANTI-PATTERN: one AI_COMPLETE call per user click, no caching, no batching
SELECT AI_COMPLETE('llama3.1-70b', 'Answer this ticket: ' || ticket_text)
FROM support_tickets
WHERE ticket_id = ?;


-- --- THE FIX — Batch the Work, Serve the Read ---

-- BATCH: pre-compute summaries over the whole table on a schedule
CREATE OR REPLACE TABLE ticket_summaries AS
SELECT ticket_id, SNOWFLAKE.CORTEX.SUMMARIZE(ticket_text) AS summary
FROM support_tickets;



-- ########################################################
-- MODULE 02
-- ########################################################

-- --- Anti-Pattern — Assuming ACCOUNTADMIN Means You're Covered ---

SELECT AI_COMPLETE('claude-sonnet-4-5', 'summarize this feedback');
-- works out of the box on a fresh account, any role


-- --- Granting the Cortex Usage Privilege to a Role ---

-- 1. Check whether the default PUBLIC grant is still present
SHOW GRANTS TO ROLE analytics_role;

-- 2. If governance revoked it, re-grant the Cortex usage privilege
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE analytics_role;

-- 3. If account-level USE AI FUNCTIONS was also revoked, re-grant it too
GRANT USE AI FUNCTIONS ON ACCOUNT TO ROLE analytics_role;


-- --- Checking and Enabling Cross-Region Inference ---

SHOW PARAMETERS LIKE 'CORTEX_ENABLED_CROSS_REGION' IN ACCOUNT;

ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'AWS_US';


-- --- Estimating Cost and Token Volume by Model and Warehouse ---

-- compare estimated cost and token volume by model and warehouse
SELECT model_name, warehouse_id,
       AVG(token_credits) AS avg_credits,
       SUM(tokens) AS total_tokens
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AISQL_USAGE_HISTORY
GROUP BY model_name, warehouse_id;


-- --- Anti-Pattern — Summarizing Raw, Unfiltered Tables at Full Volume ---

-- ANTI-PATTERN: no WHERE clause, no row limit
SELECT SNOWFLAKE.CORTEX.SUMMARIZE(review_text)
FROM all_reviews;

-- FIX: filter to the real analysis window first
SELECT SNOWFLAKE.CORTEX.SUMMARIZE(review_text)
FROM all_reviews
WHERE review_date >= DATEADD(month, -1, CURRENT_DATE());


-- --- Reference Card — A Monitoring Query for Ongoing Cost Control ---

SELECT function_name, model_name,
       SUM(token_credits) AS credits,
       COUNT(*) AS calls
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AISQL_USAGE_HISTORY
WHERE usage_time >= DATEADD(day, -7, CURRENT_TIMESTAMP())
GROUP BY function_name, model_name
ORDER BY credits DESC;


-- --- Anti-Pattern — Reaching for AI_COMPLETE for Everything ---

-- ANTI-PATTERN: reaching for AI_COMPLETE for everything
SELECT AI_COMPLETE('claude-sonnet-4-5',
  'What is the sentiment of: ' || review_text)
FROM reviews; -- 2M rows

-- FIX: purpose-built task function — cheaper, structured output
SELECT AI_SENTIMENT(review_text)
FROM reviews;


-- --- Case Study — Mapping a CPG Use Case to the Matrix ---

-- Q1: 'which SKUs underperformed in the Midwest last quarter?'
-- structured tables, plain English -> Cortex Analyst

-- Q2: 'find the clause on returns in these 4,000 retailer compliance PDFs'
-- unstructured docs, retrieval -> Cortex Search

-- Combined: reasoning across both Q1 and Q2 in one conversation
-- wrap both calls in Cortex Agents



-- ########################################################
-- MODULE 03
-- ########################################################

-- --- Your First Production-Ready AI_COMPLETE Call ---

SELECT complaint_id, complaint_text,
  AI_COMPLETE(
    'llama3.1-8b',
    'Classify this customer complaint into exactly one category: Quality, Shipping, Billing, or Other. Reply with only the category word. Complaint: ' || complaint_text
  ) AS complaint_category
FROM cpg_complaints;


-- --- Setting Temperature and Max Tokens in SQL ---

SELECT complaint_id,
  AI_COMPLETE(
    'llama3.1-8b',
    'You are a strict classifier. Reply with exactly one word: Quality, Shipping, Billing, or Other. Text: ' || complaint_text,
    {'temperature': 0, 'max_tokens': 10}
  ) AS complaint_category
FROM cpg_complaints;


-- --- The Anti-Pattern in Code — Don't Ship This ---

CREATE OR REPLACE PROCEDURE classify_complaints_slow()
RETURNS STRING
LANGUAGE SQL
AS
$$
  DECLARE
    c1 CURSOR FOR
      SELECT complaint_id, complaint_text
      FROM cpg_complaints;
  BEGIN
    FOR row IN c1 DO
      UPDATE cpg_complaints
      SET complaint_category = AI_COMPLETE(
        'llama3.1-8b',
        'Classify: ' || :row.complaint_text
      )
      WHERE complaint_id = :row.complaint_id;
    END FOR;
    RETURN 'done';
  END;
$$;


-- --- The Fix — One Statement, Whole Table ---

UPDATE cpg_complaints
SET complaint_category = AI_COMPLETE(
  'llama3.1-8b',
  'Classify this complaint into exactly one category: Quality, Shipping, Billing, or Other. Complaint: ' || complaint_text
);


-- --- Reproducing the Bug ---

SELECT
  AI_COMPLETE(
    'llama3.1-70b',
    'Write a detailed 5-sentence summary of this customer feedback: ' || feedback_text,
    {'max_tokens': 15}
  ) AS summary
FROM cpg_feedback
LIMIT 1;


-- --- The Fix — Right-Sizing max_tokens and the Prompt ---

SELECT
  AI_COMPLETE(
    'llama3.1-70b',
    'Write a detailed 5-sentence summary of this customer feedback: ' || feedback_text,
    {'max_tokens': 200}
  ) AS summary
FROM cpg_feedback
LIMIT 1;



-- ########################################################
-- MODULE 04
-- ########################################################

-- --- AI_SENTIMENT — Score a Product Review ---

SELECT AI_SENTIMENT(
  'the packaging arrived crushed but the product inside is great',
  ['packaging', 'product quality']
);


-- --- AI_TRANSLATE — A Distributor Message in Spanish ---

SELECT AI_TRANSLATE(
  'el pedido llego incompleto, faltan tres cajas',
  '',
  'en'
);


-- --- SNOWFLAKE.CORTEX.SUMMARIZE — Condense a Vendor Email ---

SELECT SNOWFLAKE.CORTEX.SUMMARIZE(vendor_email_body)
FROM incoming_vendor_emails;


-- --- AI_EXTRACT — Pull the Complaint Category ---

SELECT AI_EXTRACT(
  'the bottle cap was loose and product leaked into the box.',
  {'category': 'what is the complaint category?'}
);


-- --- One Call, Multiple Questions ---

SELECT AI_EXTRACT(
  ticket_text,
  ['what is the complaint category?', 'is a refund requested?', 'how urgent is this?']
);


-- --- Anti-Pattern — The Hand-Rolled Sentiment Prompt ---

SELECT AI_COMPLETE(
  'llama3.1-70b',
  'You are a sentiment classifier. Read the review and respond with EXACTLY one word: positive, negative, or neutral. Do not add punctuation or explanation. Review: ' || review_text
)
FROM product_reviews;


-- --- The One-Line Fix ---

SELECT AI_SENTIMENT(review_text)
FROM product_reviews;


-- --- AI_FILTER and AI_AGG — SQL That Filters and Aggregates on Meaning ---

-- AI_FILTER: natural-language predicate in a WHERE clause
SELECT review_id, review_text
FROM product_reviews
WHERE AI_FILTER(
  PROMPT('Does this review express frustration about shipping delays? Review: {0}', review_text)
);

-- AI_AGG: natural-language aggregate, like a smarter SUM
SELECT AI_AGG(
  review_text,
  'What are customers most commonly complaining about this month?'
) AS complaint_summary
FROM product_reviews
WHERE review_month = '2026-07';


-- --- AI_SIMILARITY and AI_TRANSCRIBE — Matching Meaning, Turning Audio Into Text ---

-- AI_SIMILARITY: score how semantically close two texts are
SELECT a.review_id, b.review_id,
       AI_SIMILARITY(a.review_text, b.review_text) AS similarity_score
FROM product_reviews a
JOIN product_reviews b ON a.review_id < b.review_id
WHERE AI_SIMILARITY(a.review_text, b.review_text) > 0.85;

-- AI_TRANSCRIBE: turn a staged audio file into text
SELECT relative_path AS recording_name,
       AI_TRANSCRIBE(TO_FILE('@call_recordings_stage', relative_path)) AS transcript_text
FROM DIRECTORY(@call_recordings_stage);



-- ########################################################
-- MODULE 05
-- ########################################################

-- --- Step 1 — The Results Table ---

CREATE TABLE review_scores (
  review_id INT,
  review_text STRING,
  sentiment_result VARIANT,
  summary_text STRING,
  scored_at TIMESTAMP_NTZ
);


-- --- Step 2 — Score and Summarize in One Pass ---

INSERT INTO review_scores (review_id, review_text, sentiment_result, summary_text, scored_at)
SELECT
  review_id,
  review_text,
  AI_SENTIMENT(review_text),
  SNOWFLAKE.CORTEX.SUMMARIZE(review_text),
  CURRENT_TIMESTAMP()
FROM raw_reviews
WHERE review_id NOT IN (SELECT review_id FROM review_scores);


-- --- Step 3 — Wrap It in a Snowflake Task ---

CREATE OR REPLACE TASK triage_reviews_task
  WAREHOUSE = xs_wh
  SCHEDULE = 'USING CRON 0 * * * * UTC'
AS
INSERT INTO review_scores (review_id, review_text, sentiment_result, summary_text, scored_at)
SELECT
  review_id,
  review_text,
  AI_SENTIMENT(review_text),
  SNOWFLAKE.CORTEX.SUMMARIZE(review_text),
  CURRENT_TIMESTAMP()
FROM raw_reviews
WHERE review_id NOT IN (SELECT review_id FROM review_scores);


-- --- Debug Slide — The Scheduled Task Silently Stopped Running ---

-- Debug: review_scores hasn't gained a row in 6 hours, no error message
SHOW TASKS LIKE 'triage_reviews_task';  -- check the STATE column

SELECT *
FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
  TASK_NAME => 'TRIAGE_REVIEWS_TASK'))
ORDER BY scheduled_time DESC LIMIT 10;

-- Root cause A: warehouse resized/dropped -> task auto-suspended
ALTER TASK triage_reviews_task RESUME;

-- Root cause B: raw_reviews column mismatch -> TASK_HISTORY STATE='FAILED'
-- fix: correct the INSERT column list, then RESUME


-- --- Tracking Table and the Sampling Query ---

CREATE TABLE sentiment_qa_log (
  review_id INT,
  ai_label STRING,
  human_label STRING,
  checked_at TIMESTAMP_NTZ
);

SELECT *
FROM review_scores
SAMPLE (5)
WHERE scored_at >= DATEADD(day, -1, CURRENT_TIMESTAMP());


-- --- Detecting the Spike ---

SELECT COUNT(*) AS negative_count
FROM review_scores,
     LATERAL FLATTEN(input => sentiment_result:categories) c
WHERE c.value:name::string = 'overall'
  AND c.value:sentiment::string = 'negative'
  AND scored_at >= DATEADD(hour, -1, CURRENT_TIMESTAMP());


-- --- Writing the Alert Row ---

INSERT INTO alerts (alert_type, detail, negative_count, triggered_at)
SELECT
  'negative_sentiment_spike',
  'Packaging complaints trending up',
  negative_count,
  CURRENT_TIMESTAMP()
FROM (
  SELECT COUNT(*) AS negative_count
  FROM review_scores,
       LATERAL FLATTEN(input => sentiment_result:categories) c
  WHERE c.value:name::string = 'overall'
    AND c.value:sentiment::string = 'negative'
    AND scored_at >= DATEADD(hour, -1, CURRENT_TIMESTAMP())
)
WHERE negative_count > 25;



-- ########################################################
-- MODULE 06
-- ########################################################

-- --- Generating and Storing an Embedding with AI_EMBED ---

SELECT AI_EMBED('snowflake-arctic-embed-m',
    'Milk smelled off after two days, very disappointed'
) AS review_vector;


-- --- VECTOR_COSINE_SIMILARITY in a Single Query ---

SELECT review_id, review_text,
    VECTOR_COSINE_SIMILARITY(
        review_vector,
        AI_EMBED('snowflake-arctic-embed-m', 'packaging arrived damaged')
    ) AS score
FROM product_reviews
ORDER BY score DESC
LIMIT 5;


-- --- Ranking an Entire Product Catalog by Similarity ---

WITH query_vec AS (
    SELECT AI_EMBED('snowflake-arctic-embed-m', 'milk smells bad') AS v
)
SELECT r.review_id, r.review_text,
    VECTOR_COSINE_SIMILARITY(r.review_vector, q.v) AS score
FROM product_reviews r, query_vec q
ORDER BY score DESC
LIMIT 10;


-- --- Anti-Pattern: All-Pairs Cross Join Similarity Scan ---

SELECT a.review_id, b.review_id,
    VECTOR_COSINE_SIMILARITY(a.review_vector, b.review_vector) AS score
FROM product_reviews a, product_reviews b
WHERE a.review_id != b.review_id
ORDER BY score DESC
LIMIT 100;


-- --- Measuring the Blowup: Query Time at Three Table Sizes ---

SELECT COUNT(*) FROM product_reviews;
-- run the SAME cross-join query at 500, 50,000, and 5,000,000 rows,
-- timing each run and checking Query History for compute + spillage



-- ########################################################
-- MODULE 07
-- ########################################################

-- --- Anti-Pattern: Building Your Own Index When One Exists ---

-- BAD: 200+ lines of Python, scheduled nightly on a warehouse
-- 1. SELECT new/changed rows from support_tickets
-- 2. Call an embedding model row-by-row
-- 3. Write vectors into a hand-maintained VECTOR column
-- 4. Re-run cosine-similarity ranking logic by hand

-- GOOD: one managed CREATE statement instead
CREATE CORTEX SEARCH SERVICE support_ticket_search
  ON ticket_text
  PRIMARY KEY (ticket_id)
  ATTRIBUTES ticket_id, product_line
  WAREHOUSE = search_wh
  TARGET_LAG = '1 hour'
AS
  SELECT ticket_id, product_line, ticket_text
  FROM support_tickets;


-- --- Creating the Service Over a Table ---

CREATE CORTEX SEARCH SERVICE support_ticket_search
  ON ticket_text
  PRIMARY KEY (ticket_id)
  ATTRIBUTES ticket_id, product_line
  WAREHOUSE = search_wh
  TARGET_LAG = '1 hour'
AS
  SELECT ticket_id, product_line, ticket_text
  FROM support_tickets;


-- --- Querying the Service ---

SELECT PARSE_JSON(
  SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
    'support_ticket_search',
    '{"query":"refund not processed","columns":["ticket_id","product_line"],"limit":5}'
  )
) AS results;


-- --- Anti-Pattern: Bolting Keyword Search on Yourself ---

-- BAD: patching a vector-only pipeline with a substring filter
SELECT ticket_id, ticket_text, vector_rank
FROM vector_search_results
WHERE ticket_text LIKE '%' || :sku || '%'
ORDER BY vector_rank;
-- only catches exact substring matches, not partial or reordered mentions

-- GOOD: let Cortex Search's native hybrid ranking handle both signals
SELECT PARSE_JSON(
  SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
    'support_ticket_search',
    '{"query":"SKU 48219 recall notice","limit":5}'
  )
) AS results;


-- --- Checking Index Freshness ---

DESCRIBE CORTEX SEARCH SERVICE support_ticket_search;


-- --- Debug Slide — My Search Service Returns Stale Results ---

-- Diagnosis step 1: check last-refresh vs the source's last write
DESCRIBE CORTEX SEARCH SERVICE support_ticket_search;

-- Diagnosis step 2: compare that gap to TARGET_LAG
--   gap smaller than TARGET_LAG -> expected behavior, not a bug
--   gap larger  than TARGET_LAG -> refresh is falling behind

-- Root cause A: TARGET_LAG set too wide for the workload
-- Fix A: tighten the SLA to match how often the source changes
ALTER CORTEX SEARCH SERVICE support_ticket_search
  SET TARGET_LAG = '15 minutes';

-- Root cause B: backing warehouse suspended or under-sized
-- Fix B: confirm WAREHOUSE = search_wh is running and sized for
--        the refresh volume, then re-check DESCRIBE for caught-up state

-- Do NOT drop and recreate to force freshness — that pays a full
-- re-embed cost for a problem TARGET_LAG/warehouse sizing already fixes


-- --- Anti-Pattern: Dropping and Recreating on Every Change ---

-- BAD: DROP + CREATE loop triggered on every catalog batch load
DROP CORTEX SEARCH SERVICE IF EXISTS support_ticket_search;

CREATE CORTEX SEARCH SERVICE support_ticket_search
  ON ticket_text
  PRIMARY KEY (ticket_id)
  ATTRIBUTES ticket_id, product_line
  WAREHOUSE = search_wh
  TARGET_LAG = '1 hour'
AS
  SELECT ticket_id, product_line, ticket_text
  FROM support_tickets;

-- GOOD: leave the service running, just tune TARGET_LAG once
ALTER CORTEX SEARCH SERVICE support_ticket_search
  SET TARGET_LAG = '30 minutes';


-- --- When Hand-Rolled Still Wins ---

-- Rare case: calling a custom embedding endpoint before writing
-- to your own vector column
SELECT ticket_id,
       CALL custom_embed_udf(ticket_text) AS embedding
FROM support_tickets;

-- Only reach for the hand-rolled path when:
--  1. You need a domain-specific embedding model Cortex Search doesn't offer
--  2. Chunking logic depends on custom document structure (e.g. nested BOM trees)
--  3. You must control exact vector storage for a system outside Snowflake



-- ########################################################
-- MODULE 08
-- ########################################################

-- --- Loading Raw Documents Into a Snowflake Stage ---

-- Create the internal stage for raw CPG source documents
CREATE STAGE IF NOT EXISTS cpg_docs_stage
  DIRECTORY = (ENABLE = TRUE)
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Upload from a local/cloud path (PUT run per file or via SnowSQL script)
-- PUT file:///local/product_specs/*.pdf @cpg_docs_stage/product_specs/;

-- Confirm the files landed and are visible to the account
SELECT relative_path, size_bytes, last_modified
FROM DIRECTORY(@cpg_docs_stage)
ORDER BY last_modified DESC;


-- --- Creating the Cortex Search Service Over Chunked Specs ---

-- Chunked table already populated: cpg_doc_chunks(chunk_id, sku, doc_type,
-- region, effective_date, page_number, chunk_text)
CREATE OR REPLACE CORTEX SEARCH SERVICE cpg_product_spec_search
  ON chunk_text
  PRIMARY KEY (chunk_id)
  ATTRIBUTES sku, doc_type, region, effective_date, page_number
  WAREHOUSE = cortex_search_wh
  TARGET_LAG = '1 hour'
  AS (
    SELECT chunk_id, chunk_text, sku, doc_type, region, effective_date, page_number
    FROM cpg_doc_chunks
  );


-- --- Generating a Grounded Answer With AI_COMPLETE ---

SELECT AI_COMPLETE(
  'claude-sonnet-4-5',
  CONCAT(
    'Answer using ONLY the context below. Cite the source page.\n\n',
    'Context:\n', context_text, '\n\n',
    'Question: does the granola bar SKU contain tree nuts, per the current spec?'
  )
) AS grounded_answer
FROM (
  SELECT LISTAGG(chunk_text || ' [p.' || page_number || ']', '\n---\n') AS context_text
  FROM retrieved_chunks
);


-- --- Prompting AI_COMPLETE to Decline Outside the Corpus ---

SELECT AI_COMPLETE(
  'claude-sonnet-4-5',
  CONCAT(
    'You are a CPG product-spec assistant. Answer ONLY using the context below. ',
    'If the context does not contain enough information to answer confidently, ',
    'respond exactly with: "This is not covered in the indexed product-spec corpus." ',
    'Do not guess or use outside knowledge.\n\n',
    'Context:\n', context_text, '\n\n',
    'Question: ', :buyer_question
  )
) AS grounded_or_decline;



-- ########################################################
-- MODULE 09
-- ########################################################

-- --- Anti-Pattern: A Verified Query That Locks In a Bug ---

sql: |
  SELECT p.category, SUM(f.units_sold) AS total_units
  FROM fact_sales f
  JOIN dim_product p ON f.product_id = p.product_id
  WHERE f.sale_date >= DATEADD(day, -90, CURRENT_DATE())


-- --- Two Ways to Author: YAML File vs. Native Semantic View ---

CREATE SEMANTIC VIEW cpg_analyst_sv
  TABLES (
    fact_sales AS cpg_db.analytics.fact_sales PRIMARY KEY (sale_id),
    dim_product AS cpg_db.analytics.dim_product PRIMARY KEY (product_id)
  )
  RELATIONSHIPS (
    sales_to_product AS fact_sales (product_id) REFERENCES dim_product (product_id)
  )
  FACTS ( fact_sales.units_sold AS units_sold )
  DIMENSIONS ( dim_product.category AS product_category )
  METRICS ( fact_sales.net_sales AS SUM(fact_sales.net_sales_amount) );



-- ########################################################
-- MODULE 10
-- ########################################################

-- --- Declaring the Verified Query for Sell-Through Rate ---

verified_queries:
  - name: sell_through_rate_by_pack_size
    question: "sell-through rate by pack size for a category and region, this quarter vs last"
    sql: |
      WITH pos_agg AS (
        SELECT sku, sale_date, SUM(units_sold) AS units_sold
        FROM pos_sales_daily
        GROUP BY sku, sale_date
      ),
      ship_agg AS (
        SELECT sku, ship_date, SUM(cases_shipped) AS cases_shipped
        FROM shipments
        GROUP BY sku, ship_date
      )
      SELECT
        pm.pack_size,
        SUM(pos_agg.units_sold) AS units_sold,
        SUM(ship_agg.cases_shipped) * 12 AS units_shipped,
        SUM(pos_agg.units_sold) / NULLIF(SUM(ship_agg.cases_shipped) * 12, 0) AS sell_through_rate
      FROM pos_agg
      JOIN ship_agg ON ship_agg.sku = pos_agg.sku AND ship_agg.ship_date = pos_agg.sale_date
      JOIN product_master pm ON pm.sku = pos_agg.sku
      WHERE pm.category = 'salty_snacks'
      GROUP BY pm.pack_size


-- --- Registering the Semantic Model With Cortex Analyst ---

PUT file:///local/cpg_sell_through_model.yaml @cpg_semantic_stage AUTO_COMPRESS=FALSE;

-- Cortex Analyst reads the model directly from the staged YAML file
-- when a request specifies semantic_model_file pointing at this stage path


-- --- The Flawed Join Cortex Analyst Generated ---

-- FLAWED: joins on sku + date only, no store/DC grain control
SELECT
  pm.pack_size,
  SUM(p.units_sold) AS units_sold_flawed
FROM pos_sales_daily p
JOIN shipments s ON s.sku = p.sku AND s.ship_date = p.sale_date
JOIN product_master pm ON pm.sku = p.sku
GROUP BY pm.pack_size;


-- --- The Fix: Pre-Aggregate Each Side Before Joining ---

-- FIXED: pre-aggregate each side to sku+date grain first
WITH pos_agg AS (
  SELECT sku, sale_date, SUM(units_sold) AS units_sold
  FROM pos_sales_daily
  GROUP BY sku, sale_date
),
ship_agg AS (
  SELECT sku, ship_date, SUM(cases_shipped) AS cases_shipped
  FROM shipments
  GROUP BY sku, ship_date
)
SELECT
  pm.pack_size,
  SUM(pos_agg.units_sold) AS units_sold
FROM pos_agg
JOIN ship_agg ON ship_agg.sku = pos_agg.sku AND ship_agg.ship_date = pos_agg.sale_date
JOIN product_master pm ON pm.sku = pos_agg.sku
GROUP BY pm.pack_size;



-- ########################################################
-- MODULE 11
-- ########################################################

-- --- Anti-Pattern: Regex Against a Raw Text Dump ---

SELECT invoice_no, REGEXP_SUBSTR(raw_text, 'Total: \$([0-9.,]+)', 1, 1, 'e') AS total_amt FROM raw_invoice_text; -- breaks the moment a vendor reorders fields


-- --- The Real Pattern: Four Statements, No Glue Code ---

PUT file://invoice_0091.pdf @cpg_docs_stage;
CREATE OR REPLACE TABLE parsed_docs AS
SELECT relative_path, AI_PARSE_DOCUMENT(TO_FILE('@cpg_docs_stage', relative_path), {'mode':'LAYOUT'}) AS layout_json
FROM DIRECTORY(@cpg_docs_stage);
SELECT relative_path, AI_EXTRACT(file => TO_FILE('@cpg_docs_stage', relative_path), responseFormat => [['vendor_name','What is the vendor name?'],['invoice_number','What is the invoice number?'],['total_amount','What is the total amount?']]) AS extracted
FROM DIRECTORY(@cpg_docs_stage);


-- --- Calling AI_PARSE_DOCUMENT ---

SELECT relative_path, AI_PARSE_DOCUMENT(TO_FILE('@cpg_docs_stage', relative_path), {'mode':'LAYOUT'}):content::STRING AS layout_text FROM DIRECTORY(@cpg_docs_stage) WHERE relative_path LIKE '%.pdf';


-- --- Anti-Pattern: Trusting the Unstated Default ---

-- anti-pattern: no mode specified, table structure not guaranteed
SELECT AI_PARSE_DOCUMENT(TO_FILE('@cpg_docs_stage','po_4471.pdf')) FROM DIRECTORY(@cpg_docs_stage);
-- fix: pin LAYOUT explicitly
SELECT AI_PARSE_DOCUMENT(TO_FILE('@cpg_docs_stage','po_4471.pdf'), {'mode':'LAYOUT'}) FROM DIRECTORY(@cpg_docs_stage);


-- --- Anti-Pattern: One Giant 'Extract Everything' Prompt ---

-- anti-pattern: one vague catch-all field
SELECT AI_EXTRACT(file => TO_FILE('@cpg_docs_stage','inv_2291.pdf'), responseFormat => ['everything about this invoice']) FROM DIRECTORY(@cpg_docs_stage);


-- --- The Fix: Named Fields, One Row Per Line Item ---

SELECT relative_path,
  AI_EXTRACT(
    file => TO_FILE('@cpg_docs_stage', relative_path),
    responseFormat => {
      'type': 'json',
      'schema': {
        'type': 'object',
        'properties': {
          'vendor_name':    {'type': 'string'},
          'invoice_number': {'type': 'string'},
          'sku':          {'type': 'array', 'items': {'type': 'string'}},
          'description':  {'type': 'array', 'items': {'type': 'string'}},
          'quantity':     {'type': 'array', 'items': {'type': 'number'}},
          'unit_price':   {'type': 'array', 'items': {'type': 'number'}}
        }
      }
    },
    scores => TRUE
  ) AS extracted
FROM DIRECTORY(@cpg_docs_stage);


-- --- Routing in SQL ---

SELECT relative_path,
  AI_EXTRACT(
    file => TO_FILE('@cpg_docs_stage', relative_path),
    responseFormat => [['sku','What is the SKU?'],
                        ['quantity','What is the quantity?'],
                        ['unit_price','What is the unit price?']],
    scores => TRUE
  ) AS extracted
FROM DIRECTORY(@cpg_docs_stage);

INSERT INTO invoice_line_items
SELECT * FROM extracted
WHERE extracted:scoring:scores:unit_price:score::FLOAT >= 0.85;

INSERT INTO review_queue
SELECT *, 'low_confidence_extraction' AS reason FROM extracted
WHERE extracted:scoring:scores:unit_price:score::FLOAT < 0.85;



-- ########################################################
-- MODULE 12
-- ########################################################

-- --- Building the Training View ---

-- SNOWFLAKE.ML.FORECAST needs 3 columns: series id, timestamp, numeric target
CREATE OR REPLACE VIEW demand_training_view AS
SELECT
  sku_id || '_' || store_id AS series_id,
  sale_date AS ts,
  SUM(quantity) AS y
FROM daily_sku_store_sales
WHERE sale_date < '2026-01-01'
GROUP BY series_id, ts;


-- --- Training and Calling the Forecast Model ---

-- One statement trains one model object covering every series_id in the view
CREATE OR REPLACE SNOWFLAKE.ML.FORECAST sku_store_forecast_model(
  INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'demand_training_view'),
  SERIES_COLNAME => 'series_id',
  TIMESTAMP_COLNAME => 'ts',
  TARGET_COLNAME => 'y'
);

CALL sku_store_forecast_model!FORECAST(FORECASTING_PERIODS => 14);


-- --- Adding Exogenous Columns to the Training View ---

-- Extend the training view with promo/holiday exogenous columns
CREATE OR REPLACE VIEW demand_training_view_v2 AS
SELECT
  s.sku_id || '_' || s.store_id AS series_id,
  s.sale_date AS ts,
  SUM(s.quantity) AS y,
  MAX(p.is_promo_day) AS promo_flag,
  MAX(h.is_holiday) AS holiday_flag
FROM daily_sku_store_sales s
LEFT JOIN promo_calendar p
  ON p.calendar_date = s.sale_date AND p.sku_id = s.sku_id
LEFT JOIN holiday_calendar h
  ON h.calendar_date = s.sale_date
GROUP BY series_id, ts;
-- CREATE SNOWFLAKE.ML.FORECAST has no separate exogenous-variables
-- parameter: extra columns beyond series_id/ts/target (promo_flag,
-- holiday_flag here) are automatically picked up as exogenous regressors


-- --- Forecasting With Future Exogenous Data ---

CREATE OR REPLACE SNOWFLAKE.ML.FORECAST sku_store_forecast_model_v2(
  INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'demand_training_view_v2'),
  SERIES_COLNAME => 'series_id',
  TIMESTAMP_COLNAME => 'ts',
  TARGET_COLNAME => 'y'
);

-- Second input: future exogenous columns, no target (that's predicted)
CALL sku_store_forecast_model_v2!FORECAST(
  INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'future_promo_calendar_view'),
  SERIES_COLNAME => 'series_id',
  TIMESTAMP_COLNAME => 'ts'
);


-- --- Anti-Pattern: One Series, Every SKU Blended ---

-- The naive view collapses every SKU/store into ONE series
CREATE OR REPLACE VIEW naive_training_view AS
SELECT
  'ALL_SKUS' AS series_id,
  sale_date AS ts,
  SUM(quantity) AS y
FROM daily_sku_store_sales
GROUP BY ts;

CREATE OR REPLACE SNOWFLAKE.ML.FORECAST global_demand_model(
  INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'naive_training_view'),
  SERIES_COLNAME => 'series_id',
  TIMESTAMP_COLNAME => 'ts',
  TARGET_COLNAME => 'y'
);


-- --- The Fix — Return to the Per-SKU/Store Grain ---

-- Reuse demand_training_view from lesson_44 — series_id per SKU/store pair
CREATE OR REPLACE SNOWFLAKE.ML.FORECAST sku_store_forecast_model(
  INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'demand_training_view'),
  SERIES_COLNAME => 'series_id',
  TIMESTAMP_COLNAME => 'ts',
  TARGET_COLNAME => 'y'
);


-- --- Computing Reorder Points from Forecast Output ---

-- Join persisted forecast output to lead_time_reference, per series
CREATE OR REPLACE VIEW forecast_reorder_points AS
SELECT
  f.series_id,
  f.ts,
  f.forecast,
  f.upper_bound - f.lower_bound AS uncertainty_width,
  l.lead_time_days,
  (f.forecast * l.lead_time_days)
    + (1.65 * (f.upper_bound - f.lower_bound)) AS reorder_point
FROM forecast_output f
JOIN lead_time_reference l ON l.series_id = f.series_id;



-- ########################################################
-- MODULE 13
-- ########################################################

-- --- Step 1: Create the Model Over Shipment History ---

CREATE OR REPLACE SNOWFLAKE.ML.ANOMALY_DETECTION dc_shipment_anomaly_model(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'dc_shipment_history'),
    TIMESTAMP_COLNAME => 'ship_date',
    TARGET_COLNAME => 'shipment_volume',
    LABEL_COLNAME => ''
);


-- --- Step 2: Score New Volume with DETECT_ANOMALIES ---

CALL dc_shipment_anomaly_model!DETECT_ANOMALIES(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'dc_shipment_last_7_days'),
    TIMESTAMP_COLNAME => 'ship_date',
    TARGET_COLNAME => 'shipment_volume'
);


-- --- Running the Decomposition ---

CREATE SNOWFLAKE.ML.TOP_INSIGHTS pos_swing_insights();

CALL pos_swing_insights!GET_DRIVERS(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'pos_weekly_by_dimension'),
    LABEL_COLNAME => 'is_this_week',
    METRIC_COLNAME => 'revenue'
);


-- --- Diagnosis: The Model Wasn't Given Enough History to Learn Seasonality ---

-- confirm season coverage in the training data
SELECT MIN(ship_date), MAX(ship_date)
FROM dc_shipment_history;


-- --- Fix Step 1: Retrain on 2+ Years of History ---

CREATE OR REPLACE SNOWFLAKE.ML.ANOMALY_DETECTION dc_shipment_anomaly_model(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'dc_shipment_history_2yr'),
    TIMESTAMP_COLNAME => 'ship_date',
    TARGET_COLNAME => 'shipment_volume',
    LABEL_COLNAME => ''
);


-- --- Verify the Fix Before Trusting It Live ---

-- backtest against last year's known-normal spike
CALL dc_shipment_anomaly_model!DETECT_ANOMALIES(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'dc_shipment_last_thanksgiving'),
    TIMESTAMP_COLNAME => 'ship_date',
    TARGET_COLNAME => 'shipment_volume'
);


-- --- Signal 1: ANOMALY_DETECTION Flags the Divergence Same-Day ---

CALL inventory_variance_model!DETECT_ANOMALIES(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'daily_inventory_variance'),
    TIMESTAMP_COLNAME => 'count_date',
    TARGET_COLNAME => 'variance_units'
);


-- --- Signal 2: Contribution Explorer Narrows It to One Cause ---

CREATE SNOWFLAKE.ML.TOP_INSIGHTS dc0044_insights();

CALL dc0044_insights!GET_DRIVERS(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'dc0044_variance_by_dimension'),
    LABEL_COLNAME => 'is_flagged_week',
    METRIC_COLNAME => 'variance_units'
);



-- ########################################################
-- MODULE 14
-- ########################################################

-- --- Kicking Off a FINETUNE Job ---

SELECT SNOWFLAKE.CORTEX.FINETUNE(
  'CREATE',
  'sku_parser_v1',
  'llama3.1-8b',
  'SELECT prompt, completion FROM cpg_db.ml.sku_training_set',
  'SELECT prompt, completion FROM cpg_db.ml.sku_validation_set'
);


-- --- Using and Checking the Fine-Tuned Model ---

SELECT SNOWFLAKE.CORTEX.FINETUNE('DESCRIBE', 'sku_parser_v1');

SELECT AI_COMPLETE('sku_parser_v1', 'Parse this code: PRM-2024-CHOC-12PK-WMT');


-- --- Baseline Attempt: Prompting Alone Falls Short ---

SELECT AI_COMPLETE(
  'llama3.1-8b',
  'Extract promo_type, year, flavor, pack_size, retailer from: PRM-2024-CHOC-12PK-WMT'
);
-- returns retailer: "unknown" on 38% of the validation set


-- --- Fine-Tuning on 800 Labeled Promo Codes ---

SELECT SNOWFLAKE.CORTEX.FINETUNE(
  'CREATE',
  'promo_code_parser_v1',
  'llama3.1-8b',
  'SELECT prompt, completion FROM promo_training_labeled',
  'SELECT prompt, completion FROM promo_validation_labeled'
);


-- --- What a Better Prompt Solved Instead ---

SELECT AI_CLASSIFY(
  customer_message,
  ['billing_question', 'product_question', 'complaint'],
  {'task_description': 'Classify CPG customer message. See examples.'}
);


-- --- The Guardrail Check Before You Fine-Tune ---

SELECT
  COUNT_IF(predicted_label = actual_label) / COUNT(*) AS classify_accuracy
FROM validation_results;
-- if classify_accuracy >= target_threshold, skip fine-tuning entirely


-- --- The Self-Labeling Lab — Let a Big Model Write Your Training Data ---

SELECT
  raw_code,
  AI_COMPLETE(
    'llama3.1-70b',
    'Parse this CPG promo code into promo_type, year, flavor, pack_size, retailer as JSON: ' || raw_code
  ) AS draft_label
FROM cpg_db.ml.unlabeled_promo_codes;



-- ########################################################
-- MODULE 15
-- ########################################################

-- --- Illustrating an Agent Definition ---

-- Illustrative only — exact object/DDL syntax is version-dependent, verify current docs
CREATE AGENT sales_ops_agent
  TOOLS = (cortex_search_returns_docs, cortex_analyst_sales_model, custom_pricing_proc)
  ORCHESTRATION_MODEL = 'default';


-- --- Illustrating a Container Runtime Footprint ---

-- Illustrative only — confirm exact deployment/runtime settings in current docs
CREATE COMPUTE POOL agent_app_pool
  INSTANCE_FAMILY = CPU_X64_S
  MIN_NODES = 1
  MAX_NODES = 2;
-- A Streamlit-in-Snowflake app service referencing this pool runs under
-- container runtime, so it can call the Cortex Agents APIs.
-- (A plain external app calling the Agents REST API directly does not need this pool.)


-- --- Illustrating the Account-Level Guardrail Toggle ---

-- AI_SETTINGS takes a nested YAML block, not a flat key
ALTER ACCOUNT SET AI_SETTINGS = $$
guardrails:
  advanced_prompt_injection:
    - enabled: true
$$;
-- Guardrails apply account-wide, screening every CoWork agent,
-- Cortex Agent, and Cortex Code response



-- ########################################################
-- MODULE 16
-- ########################################################

-- --- Code: Querying the Promo Search Service ---

SELECT PARSE_JSON(
  SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
    'CPG_DOCS.PROMO_SEARCH_SVC',
    '{"query": "20 percent discount promo yogurt cannibalization lessons",
      "columns": ["snippet", "source_doc"],
      "filter": {"@eq": {"category": "DAIRY_YOGURT"}},
      "limit": 5}'
  )
)['results'];



-- ########################################################
-- MODULE 17
-- ########################################################

-- --- Anti-Pattern First — Unbounded History Concatenation ---

-- ANTI-PATTERN — unbounded history concatenation, no cap
SELECT AI_COMPLETE(
    'llama3.1-70b',
    ARRAY_TO_STRING(ARRAY_AGG(msg), '\n') || current_question
)
FROM chat_history
WHERE session_id = ?;
-- No LIMIT, no windowing, no truncation


-- --- The Fix — Window, Summarize, Then Complete ---

SELECT AI_COMPLETE(
    'llama3.1-70b',
    AI_SUMMARIZE_AGG(msg) || current_question
)
FROM (
    SELECT msg
    FROM chat_history
    WHERE session_id = ?
    ORDER BY created_at DESC
    LIMIT 10
);


-- --- Running the Checks ---

SHOW GRANTS TO ROLE prod_service_role;

SELECT CURRENT_ROLE(), CURRENT_WAREHOUSE();

GRANT DATABASE ROLE snowflake.cortex_user
  TO ROLE prod_service_role;


-- --- The Fix — Update the Model and Add a Drift Check ---

-- Update the semantic model column reference
-- (gross_revenue -> net_revenue), then add a
-- scheduled drift-check task:
CREATE OR REPLACE TASK check_semantic_model_drift
  WAREHOUSE = cpg_wh
  SCHEDULE = 'USING CRON 0 6 * * 1 UTC'
AS
  SELECT column_name
  FROM information_schema.columns
  WHERE table_name = 'FINANCE_SUMMARY'
    AND column_name = 'GROSS_REVENUE';
-- Zero rows returned = the model's stale reference is confirmed


-- --- The Fix — Tighten the Lag Budget ---

ALTER CORTEX SEARCH SERVICE product_search
  SET target_lag = '1 hour';

SELECT *
FROM snowflake.account_usage.cortex_search_daily_usage_history
WHERE service_name = 'product_search'
ORDER BY usage_date DESC;


-- --- The Fix — Retrain With Seasonal Coverage ---

-- Retrain on a window spanning a full seasonal cycle
CALL anomaly_model!DETECT_ANOMALIES(
  INPUT_DATA => TABLE(sales_view),
  SERIES_COLNAME => 'sku',
  TIMESTAMP_COLNAME => 'sale_date',
  TARGET_COLNAME => 'units_sold',
  CONFIG_OBJECT => {'prediction_interval': 0.95}
);
-- Add a quarterly retrain schedule — not a one-time train-and-forget


-- --- Finding the Spike ---

SELECT
    DATE(usage_time) AS usage_date,
    function_name,
    SUM(token_credits)
FROM snowflake.account_usage.cortex_aisql_usage_history
WHERE usage_time >= DATEADD(day, -30, CURRENT_DATE())
GROUP BY 1, 2
ORDER BY 1;


-- --- The Fix — Watch Before the Invoice Does ---

-- Right-size the model for the batch job's actual accuracy need,
-- then schedule a daily spend-threshold alert:
CREATE OR REPLACE TASK cortex_spend_alert
  WAREHOUSE = cpg_wh
  SCHEDULE = 'USING CRON 0 7 * * * UTC'
AS
  SELECT SUM(token_credits) AS daily_credits
  FROM snowflake.account_usage.cortex_aisql_usage_history
  WHERE DATE(usage_time) = DATEADD(day, -1, CURRENT_DATE())
  HAVING daily_credits > 100;
-- Require a cost-estimate note on any PR changing model size


-- --- Anti-Pattern First — One Bad Row Kills the Whole Batch ---

-- ANTI-PATTERN — no error isolation across the batch
SELECT
    product_id,
    AI_COMPLETE('llama3.1-70b',
      'Summarize this customer review: ' || review_text
    ) AS summary
FROM customer_reviews
WHERE batch_date = CURRENT_DATE();
-- Row 40,812 has garbled encoding from a scraped source --
-- the call throws, the WHOLE statement fails, zero rows
-- get a summary that night


-- --- The Fix — TRY_COMPLETE, NULLs, and the Blast Radius ---

SELECT product_id,
    TRY_COMPLETE('llama3.1-70b',
      'Summarize this customer review: ' || review_text
    ) AS summary
FROM customer_reviews
WHERE batch_date = CURRENT_DATE();

-- Downstream check: requeue only the failed rows
SELECT COUNT(*) FROM batch_output WHERE summary IS NULL;

-- Before running at scale, estimate the blast radius
SELECT SUM(COUNT_TOKENS('llama3.1-70b', review_text))
FROM customer_reviews
WHERE batch_date = CURRENT_DATE();



-- ########################################################
-- MODULE 18
-- ########################################################

-- --- Prove the Boundary Before You Ship It ---

USE ROLE cpg_analyst_role;
SELECT * FROM cpg_db.retail.customer_pii LIMIT 1;
-- Expect: insufficient privileges error
SHOW GRANTS TO ROLE cpg_analyst_role;


-- --- Debug — A Team Lost Cortex Access After a Role Change ---

-- Symptom: category managers hit 'insufficient privileges' on AI_CLASSIFY, no code changed
SHOW GRANTS TO ROLE cpg_category_mgr_role;
-- USAGE ON SNOWFLAKE.CORTEX_USER is listed, looks fine

-- Diagnosis: the direct grant is intact, but this role no longer
-- INHERITS it via a parent role that was cleaned up earlier that week
SHOW GRANTS OF ROLE cortex_platform_role;
-- cpg_category_mgr_role is missing from the grantee list

-- Fix: re-grant the role membership, then verify
GRANT ROLE cortex_platform_role TO ROLE cpg_category_mgr_role;
USE ROLE cpg_category_mgr_role;
SELECT AI_CLASSIFY('test row', ['a','b']); -- confirms access restored


-- --- The Grant Checklist for a Brand-New Team Member ---

-- Tier 1: everyday Cortex access (AI_COMPLETE, AI_CLASSIFY, AI_REDACT)
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE cpg_analyst_role;
-- some accounts instead expose this as SNOWFLAKE.AI_FUNCTIONS_USER --
-- confirm the current exact grant in Snowflake's docs before running in prod

-- Tier 2: fine-tuning needs TWO MORE grants on top of Tier 1
GRANT USAGE ON DATABASE cpg_training_db TO ROLE cpg_ml_role;
GRANT CREATE MODEL ON SCHEMA cpg_training_db.models TO ROLE cpg_ml_role;
-- OWNERSHIP on that schema works too, if the role fully manages its models


-- --- Find Out Who's Spending What ---

SELECT function_name,
       DATE_TRUNC('day', start_time) AS day,
       SUM(credits) AS cortex_credits
FROM snowflake.account_usage.cortex_ai_functions_usage_history
WHERE start_time >= DATEADD(day, -30, CURRENT_DATE())
GROUP BY 1, 2
ORDER BY cortex_credits DESC;


-- --- The Warehouse-Level Backstop — and Its Limit ---

CREATE RESOURCE MONITOR cortex_guard
  WITH CREDIT_QUOTA = 500
  FREQUENCY = MONTHLY
  START_TIMESTAMP = IMMEDIATELY
  TRIGGERS
    ON 75 PERCENT DO NOTIFY
    ON 100 PERCENT DO SUSPEND;
ALTER WAREHOUSE cortex_wh SET RESOURCE_MONITOR = cortex_guard;


-- --- AI_REDACT in One Query ---

SELECT
    order_id,
    AI_REDACT(customer_name) AS masked_name,
    AI_REDACT(loyalty_id) AS masked_loyalty_id,
    AI_COMPLETE('llama3.1-70b',
      'Summarize this order for a support ticket: ' || review_text
    ) AS ticket_summary
FROM cpg_db.retail.orders
LIMIT 100;


-- --- Wiring the Masking Policy to the Table ---

CREATE MASKING POLICY mask_customer_name AS (val STRING) RETURNS STRING ->
  CASE
    WHEN CURRENT_ROLE() IN ('CPG_ANALYST_ROLE') THEN AI_REDACT(val)
    ELSE val
  END;

ALTER TABLE cpg_db.retail.orders
  MODIFY COLUMN customer_name
  SET MASKING POLICY mask_customer_name;


-- --- Guardrails on the Way Out, Not Just the Way In ---

-- Illustrative only — parameter name/availability is version-dependent,
-- confirm current syntax in docs
SELECT AI_COMPLETE(
    'llama3.1-70b',
    'Summarize this support ticket for the customer-facing portal: ' || review_text,
    {'guardrails': true}
) AS safe_summary
FROM cpg_db.retail.support_tickets;


-- --- One Query to Prove the Checklist Is Live ---

SHOW RESOURCE MONITORS;
SHOW MASKING POLICIES IN DATABASE cpg_db;
SELECT role_name, granted_on, privilege
FROM snowflake.account_usage.grants_to_roles
WHERE role_name ILIKE '%cortex%'
  AND deleted_on IS NULL;



-- ########################################################
-- MODULE 20
-- ########################################################

-- --- A Minimal Setup Script, Conceptually ---

-- setup script, runs once per install in the consumer account
CREATE APPLICATION ROLE IF NOT EXISTS app_public;
CREATE OR ALTER VERSIONED SCHEMA core;

CREATE OR REPLACE PROCEDURE core.run_forecast(input_table STRING)
  RETURNS STRING
  LANGUAGE SQL
  AS
  $$
    -- calls a Cortex function against the consumer's own granted table
    SELECT AI_COMPLETE('llama3-70b', 'Summarize demand trend for ' || :input_table);
  $$;

GRANT USAGE ON PROCEDURE core.run_forecast(STRING) TO APPLICATION ROLE app_public;


-- --- Row Access Policy, Conceptually ---

-- conceptual pattern, verify exact clause syntax against current docs
CREATE OR REPLACE ROW ACCESS POLICY bu_scope_policy
  AS (bu_code STRING) RETURNS BOOLEAN ->
  EXISTS (
    SELECT 1 FROM bu_role_map
    WHERE bu_role_map.role_name = CURRENT_ROLE()
      AND bu_role_map.bu_code = bu_code
  );

ALTER TABLE sales_shared ADD ROW ACCESS POLICY bu_scope_policy ON (bu_code);



-- ########################################################
-- MODULE 21
-- ########################################################

-- --- Step 1 — Give Every Prompt a Version ---

CREATE TABLE prompt_registry (
    prompt_version STRING,
    prompt_text STRING,
    created_at TIMESTAMP_NTZ,
    is_active BOOLEAN
);

-- Every AI_COMPLETE / AI_SENTIMENT call logs
-- which prompt_version produced each row


-- --- Step 2 — Trend the Agreement Rate by Version ---

SELECT
    DATE_TRUNC('week', checked_at) AS wk,
    prompt_version,
    AVG(CASE WHEN ai_label = human_label THEN 1 ELSE 0 END) AS agreement_rate
FROM sentiment_qa_log
GROUP BY wk, prompt_version
ORDER BY wk;


-- --- The Decision Log Table ---

CREATE TABLE agent_decision_log (
    session_id STRING,
    step_number INT,
    tool_called STRING,
    tool_input VARIANT,
    tool_output VARIANT,
    model_version STRING,
    prompt_version STRING,
    logged_at TIMESTAMP_NTZ
);


-- --- Anti-Pattern First — Rolling Out to 100% at Once ---

-- WRONG — flips 100% of traffic instantly, no rollback window
UPDATE prompt_registry
SET is_active = TRUE
WHERE prompt_version = 'v2';

-- RIGHT — route by a deterministic hash, sample a slice first
SELECT
    CASE WHEN MOD(ABS(HASH(customer_id)), 100) < 5
         THEN 'variant_b' ELSE 'variant_a' END AS variant,
    customer_id
FROM incoming_requests;


-- --- Pinpointing the Exact Day ---

SELECT
    DATE_TRUNC('day', checked_at) AS dy,
    AVG(CASE WHEN ai_label = human_label THEN 1 ELSE 0 END) AS agreement_rate
FROM sentiment_qa_log
GROUP BY dy
ORDER BY dy;



-- ########################################################
-- MODULE 22
-- ########################################################

-- --- Vendor Contracts to Structured Rows ---

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


-- --- Standing Up the Search Service ---

CREATE OR REPLACE CORTEX SEARCH SERVICE meridian_field_search
    ON search_text
    ATTRIBUTES doc_type, sku_id
    WAREHOUSE = meridian_wh
    TARGET_LAG = '1 hour'
AS
SELECT sku_id, doc_type, search_text
FROM meridian_unified_text;


-- --- Defining Meridian's Semantic View ---

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


-- --- Forecasting Sell-Through, Per SKU ---

CREATE SNOWFLAKE.ML.FORECAST meridian_sku_forecast(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'pos_sales_daily'),
    TIMESTAMP_COLNAME => 'sale_date',
    TARGET_COLNAME => 'units_sold',
    SERIES_COLNAME => 'sku_id'
);


-- --- Wiring the Anomaly Model ---

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


-- --- Registering the Agent's Tools ---

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


-- --- Masking and Guardrails, Applied ---

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

