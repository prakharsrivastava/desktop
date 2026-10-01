-- ==========================================================
-- Exercise 13.2 — Find the Dimension Driving a Metric Swing with TOP_INSIGHTS
-- Course: 1018
-- Module 13
-- ==========================================================
--
-- ANOMALY_DETECTION flagged total weekly POS revenue down 12% region-wide, and
-- the category manager's first move was 6 phone calls guessing which store was
-- the problem. SNOWFLAKE.ML.TOP_INSIGHTS decomposes a metric swing across every
-- dimension in a view and ranks which one actually explains it — turning a
-- 2-day guessing hunt into a single ranked query.
--
-- Your task:
-- 1. Instantiate a SNOWFLAKE.ML.TOP_INSIGHTS object (it must be CREATEd once
--    before you CALL it — same instantiate-then-call pattern as other Cortex
--    ML classes).
-- 2. Call GET_DRIVERS against pos_weekly_by_dimension, using is_this_week as
--    the boolean LABEL_COLNAME (FALSE = baseline period, TRUE = comparison
--    period) and revenue as the METRIC_COLNAME.
-- 3. Remember: dimensions (store_id, sku, region) are auto-inferred from every
--    other column in the view — there is no dimension parameter to hand-pick
--    them.
-- 4. Rank the returned rows by RELATIVE_CONTRIBUTION, largest swing-driver first.
-- ==========================================================

-- YOUR CODE:

CREATE SNOWFLAKE.ML.TOP_INSIGHTS ___();

CALL pos_swing_insights!___(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', '___'),
    LABEL_COLNAME => '___',
    METRIC_COLNAME => '___'
);

-- then, conceptually: ORDER BY RELATIVE_CONTRIBUTION (largest swing first)


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
CREATE SNOWFLAKE.ML.TOP_INSIGHTS pos_swing_insights();

CALL pos_swing_insights!GET_DRIVERS(
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'pos_weekly_by_dimension'),
    LABEL_COLNAME => 'is_this_week',
    METRIC_COLNAME => 'revenue'
);

-- Real result from the module's case study: CONTRIBUTOR = sku_47821,
-- RELATIVE_CONTRIBUTION = -75% — one SKU pulled from shelf at 40 stores
-- explained three-quarters of the entire regional swing. The remaining 25%
-- was spread thin across dozens of other CONTRIBUTOR rows: noise, not a story.

-- Reminder: TOP_INSIGHTS tells you WHICH dimension moved the metric, not WHY.
-- Treat the ranked output as a triage list — pair it with the source system
-- (inventory, pricing) to confirm the actual cause before acting on it.
*/
