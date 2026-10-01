-- ============================================================
-- Course 1018 — Module 20
-- On-slide QUERIES reference  (2 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- A Minimal Setup Script, Conceptually
-- ----------------------------------------------------------

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


-- ----------------------------------------------------------
-- Row Access Policy, Conceptually
-- ----------------------------------------------------------

-- conceptual pattern, verify exact clause syntax against current docs
CREATE OR REPLACE ROW ACCESS POLICY bu_scope_policy
  AS (bu_code STRING) RETURNS BOOLEAN ->
  EXISTS (
    SELECT 1 FROM bu_role_map
    WHERE bu_role_map.role_name = CURRENT_ROLE()
      AND bu_role_map.bu_code = bu_code
  );

ALTER TABLE sales_shared ADD ROW ACCESS POLICY bu_scope_policy ON (bu_code);

