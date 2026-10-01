# Environment Setup — Snowflake Cortex AI Masterclass

This guide gets you a working Snowflake environment that matches every exercise and slide
demo in this course. Budget 20-30 minutes.

## 1. Get access

You need a Snowflake account. Two options:

- **Free 30-day trial** (recommended if you don't already have an account):
  https://signup.snowflake.com — pick any cloud/region on Standard edition or higher.
  Cortex AI functions (`AI_COMPLETE`, `AI_SENTIMENT`, `AI_EXTRACT`, Cortex Search, Cortex
  Analyst) are available on every edition — no Enterprise upgrade needed for this course.
- **Existing account** — you don't need `ACCOUNTADMIN` to use Cortex functions day-to-day.
  `USE AI FUNCTIONS` (account-level) and the `SNOWFLAKE.CORTEX_USER` database role are
  granted to `PUBLIC` **by default** on every account — confirmed directly against
  Snowflake's own docs during this course's fact-audit. If your organization has
  deliberately revoked that default grant (some do, for governance reasons), module 2 shows
  exactly how to re-grant it — see Step 2 below.

## 2. Confirm Cortex access (module 2's own content, verified working)

Run this first — it's the exact anti-pattern/fix pair taught in module 2:

```sql
-- This should just work on a fresh account, any role — no special grant needed
SELECT AI_COMPLETE('claude-sonnet-4-5', 'summarize this feedback');
```

If it fails with an access-denied error, someone revoked the default grant. Fix it:

```sql
-- 1. Check whether the default PUBLIC grant is still present
SHOW GRANTS TO ROLE analytics_role;   -- swap in your actual role name

-- 2. If governance revoked it, re-grant the Cortex usage privilege
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE analytics_role;

-- 3. If account-level USE AI FUNCTIONS was also revoked, re-grant it too
GRANT USE AI FUNCTIONS ON ACCOUNT TO ROLE analytics_role;
```

## 3. Check your region and model availability

Not every Cortex model is available in every Snowflake region. Before you rely on a specific
model name in an exercise, check:

```sql
SHOW PARAMETERS LIKE 'CORTEX_ENABLED_CROSS_REGION' IN ACCOUNT;

-- If the model you need isn't hosted in your account's region, enable cross-region
-- inference (adds a small latency/cost delta — module 2 covers estimating it):
ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'AWS_US';
```

## 4. Provision the core objects

Run this once, as a role with `CREATE DATABASE`/`CREATE WAREHOUSE` privileges:

```sql
-- A small, cheap warehouse for the whole course (X-Small, auto-suspend fast)
CREATE WAREHOUSE IF NOT EXISTS cortex_wh
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE;

-- The database and schemas every exercise in this course assumes — modeled on the
-- course's running CPG scenario (product reviews, POS/shipment data, product docs)
CREATE DATABASE IF NOT EXISTS CPG_INTEL;
CREATE SCHEMA IF NOT EXISTS CPG_INTEL.RAW;
CREATE SCHEMA IF NOT EXISTS CPG_INTEL.ANALYTICS;

USE DATABASE CPG_INTEL;
USE WAREHOUSE cortex_wh;
```

Some individual modules introduce their own named warehouse or object for a specific demo
(e.g. a dedicated `search_wh` in module 7/8 for Cortex Search, or a `CORTEX SEARCH SERVICE`
in module 8) — create those ad hoc, matching the exact name shown on screen, when you reach
that lesson. `cortex_wh` above covers the majority of exercises.

## 5. Load sample data

This course teaches on a running CPG scenario — customer reviews, POS/shipment tables, and
product documentation (spec sheets, planogram PDFs). Minimal starter stubs to unblock the
first several modules:

```sql
CREATE TABLE IF NOT EXISTS CPG_INTEL.RAW.CUSTOMER_REVIEWS (
  review_id NUMBER, product_id STRING, review_text STRING, review_date DATE
);
INSERT INTO CPG_INTEL.RAW.CUSTOMER_REVIEWS VALUES
  (1, 'SKU-48213', 'Packaging arrived damaged, product was fine.', CURRENT_DATE()),
  (2, 'SKU-48213', 'Great taste, will buy again.', CURRENT_DATE());

CREATE TABLE IF NOT EXISTS CPG_INTEL.RAW.POS_SALES_DAILY (
  sku STRING, sale_date DATE, store_id STRING, units_sold NUMBER
);
CREATE TABLE IF NOT EXISTS CPG_INTEL.RAW.SHIPMENTS (
  sku STRING, ship_date DATE, dc_id STRING, units_shipped NUMBER
);
```

Each exercise file's SCENARIO section names exactly which table(s) it needs — stub any
additional table with a few representative rows as you reach that exercise, rather than
building the full layer up front.

## 6. Connect your tools

- **Snowsight (web UI)** — no setup needed, log in at `https://app.snowflake.com`. Good
  enough for every exercise in this course, including Cortex Analyst's chat interface.
- **Python connector / REST API** — only needed for module 10's Cortex Analyst REST-API
  exercise and module 19's Streamlit app-layer exercises. Instructions are inline in those
  exercise files.

## 7. Troubleshooting

- **Access-denied on a Cortex function call** — see Step 2 above; almost always a revoked
  default grant, not a real permissions gap.
- **A model name in an exercise isn't available** — check Step 3's region/cross-region
  settings; not every model ships in every region on day one.
- **The bill doesn't match your mental model** — every Cortex call is metered by tokens
  processed x model tier, not by "number of calls." A single `AI_COMPLETE` over a huge
  unfiltered table costs far more than the same call scoped with a `WHERE` clause first
  (module 2's anti-pattern/fix pair demonstrates this directly). Use the monitoring query
  below before assuming something's wrong.
- **Invalid model id** — use real, current model ids only (e.g. `claude-sonnet-4-5`,
  `llama3.1-70b`, `mistral-large2`). A model id copied from an old blog post or a made-up
  name will fail immediately.

## 8. Cost hygiene

- Keep `cortex_wh` X-Small unless a lesson specifically asks for larger.
- Always scope AI function calls with a `WHERE` clause / row limit while developing —
  module 2's "summarizing raw, unfiltered tables at full volume" anti-pattern is the single
  most common way to burn credits by accident.
- Monitor actual spend with the real, current usage view (module 2's reference card —
  note this is `CORTEX_AISQL_USAGE_HISTORY`, not the older `CORTEX_FUNCTIONS_USAGE_HISTORY`
  view, which is deprecated):

```sql
SELECT function_name, model_name,
       SUM(token_credits) AS credits,
       COUNT(*) AS calls
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AISQL_USAGE_HISTORY
WHERE usage_time >= DATEADD(day, -7, CURRENT_TIMESTAMP())
GROUP BY function_name, model_name
ORDER BY credits DESC;
```

You're ready. Start with `exercises/module_01_*` and work through in order — later modules
assume earlier ones' objects (semantic models, search services, tables) already exist.
