-- ============================================================
-- Course 1018 — Module 16
-- On-slide QUERIES reference  (1 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- Code: Querying the Promo Search Service
-- ----------------------------------------------------------

SELECT PARSE_JSON(
  SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
    'CPG_DOCS.PROMO_SEARCH_SVC',
    '{"query": "20 percent discount promo yogurt cannibalization lessons",
      "columns": ["snippet", "source_doc"],
      "filter": {"@eq": {"category": "DAIRY_YOGURT"}},
      "limit": 5}'
  )
)['results'];

