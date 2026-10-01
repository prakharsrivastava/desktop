-- ==========================================================
-- Exercise 7.2 — Diagnose and Fix a Stale Search Service
-- Course: 1018
-- Module 07
-- ==========================================================
--
-- The module's debug slide walks through a search service returning stale
-- results: check DESCRIBE for the last-refresh gap, compare it to
-- TARGET_LAG, and fix by tightening the SLA — never by dropping and
-- recreating (that pays a full re-embed cost for a problem TARGET_LAG
-- already fixes). Apply the same diagnosis-and-fix sequence to the
-- `product_recall_search` service you created in Exercise 7.1, this time
-- tightening it to a 15-minute lag because recall notices need to be
-- searchable almost immediately after they're filed.
--
-- Your task:
-- 1. Write the DESCRIBE statement to check the service's current refresh
--    state.
-- 2. Write the ALTER CORTEX SEARCH SERVICE statement that tightens
--    TARGET_LAG to '15 minutes' — the correct fix.
-- 3. In a comment, state the anti-pattern fix you should NOT use instead.
-- ==========================================================

-- YOUR CODE:

___ CORTEX SEARCH SERVICE product_recall_search;

ALTER CORTEX SEARCH SERVICE product_recall_search
  SET ___ = '___';

-- Anti-pattern to avoid: ___


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
DESCRIBE CORTEX SEARCH SERVICE product_recall_search;

ALTER CORTEX SEARCH SERVICE product_recall_search
  SET TARGET_LAG = '15 minutes';

-- Anti-pattern to avoid: DROP CORTEX SEARCH SERVICE + CREATE CORTEX SEARCH
-- SERVICE on every catalog batch load — that re-triggers a full re-embed
-- instead of just tuning TARGET_LAG or confirming the backing warehouse
-- (search_wh) is running and sized for the refresh volume.
*/
