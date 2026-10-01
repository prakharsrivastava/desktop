-- ============================================================
-- Course 1018 — Module 07
-- On-slide QUERIES reference  (8 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- Anti-Pattern: Building Your Own Index When One Exists
-- ----------------------------------------------------------

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


-- ----------------------------------------------------------
-- Creating the Service Over a Table
-- ----------------------------------------------------------

CREATE CORTEX SEARCH SERVICE support_ticket_search
  ON ticket_text
  PRIMARY KEY (ticket_id)
  ATTRIBUTES ticket_id, product_line
  WAREHOUSE = search_wh
  TARGET_LAG = '1 hour'
AS
  SELECT ticket_id, product_line, ticket_text
  FROM support_tickets;


-- ----------------------------------------------------------
-- Querying the Service
-- ----------------------------------------------------------

SELECT PARSE_JSON(
  SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
    'support_ticket_search',
    '{"query":"refund not processed","columns":["ticket_id","product_line"],"limit":5}'
  )
) AS results;


-- ----------------------------------------------------------
-- Anti-Pattern: Bolting Keyword Search on Yourself
-- ----------------------------------------------------------

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


-- ----------------------------------------------------------
-- Checking Index Freshness
-- ----------------------------------------------------------

DESCRIBE CORTEX SEARCH SERVICE support_ticket_search;


-- ----------------------------------------------------------
-- Debug Slide — My Search Service Returns Stale Results
-- ----------------------------------------------------------

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


-- ----------------------------------------------------------
-- Anti-Pattern: Dropping and Recreating on Every Change
-- ----------------------------------------------------------

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


-- ----------------------------------------------------------
-- When Hand-Rolled Still Wins
-- ----------------------------------------------------------

-- Rare case: calling a custom embedding endpoint before writing
-- to your own vector column
SELECT ticket_id,
       CALL custom_embed_udf(ticket_text) AS embedding
FROM support_tickets;

-- Only reach for the hand-rolled path when:
--  1. You need a domain-specific embedding model Cortex Search doesn't offer
--  2. Chunking logic depends on custom document structure (e.g. nested BOM trees)
--  3. You must control exact vector storage for a system outside Snowflake

