-- ============================================================
-- Course 1018 — Module 05
-- On-slide QUERIES reference  (7 snippets)
-- Authentic code as shown on the course slides. Copy, run, experiment.
-- ============================================================


-- ----------------------------------------------------------
-- Step 1 — The Results Table
-- ----------------------------------------------------------

CREATE TABLE review_scores (
  review_id INT,
  review_text STRING,
  sentiment_result VARIANT,
  summary_text STRING,
  scored_at TIMESTAMP_NTZ
);


-- ----------------------------------------------------------
-- Step 2 — Score and Summarize in One Pass
-- ----------------------------------------------------------

INSERT INTO review_scores (review_id, review_text, sentiment_result, summary_text, scored_at)
SELECT
  review_id,
  review_text,
  AI_SENTIMENT(review_text),
  SNOWFLAKE.CORTEX.SUMMARIZE(review_text),
  CURRENT_TIMESTAMP()
FROM raw_reviews
WHERE review_id NOT IN (SELECT review_id FROM review_scores);


-- ----------------------------------------------------------
-- Step 3 — Wrap It in a Snowflake Task
-- ----------------------------------------------------------

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


-- ----------------------------------------------------------
-- Debug Slide — The Scheduled Task Silently Stopped Running
-- ----------------------------------------------------------

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


-- ----------------------------------------------------------
-- Tracking Table and the Sampling Query
-- ----------------------------------------------------------

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


-- ----------------------------------------------------------
-- Detecting the Spike
-- ----------------------------------------------------------

SELECT COUNT(*) AS negative_count
FROM review_scores,
     LATERAL FLATTEN(input => sentiment_result:categories) c
WHERE c.value:name::string = 'overall'
  AND c.value:sentiment::string = 'negative'
  AND scored_at >= DATEADD(hour, -1, CURRENT_TIMESTAMP());


-- ----------------------------------------------------------
-- Writing the Alert Row
-- ----------------------------------------------------------

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

