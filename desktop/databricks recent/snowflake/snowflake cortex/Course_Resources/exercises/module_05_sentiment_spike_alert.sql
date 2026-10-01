-- ==========================================================
-- Exercise 5.2 — Detect a Sentiment Spike and Write an Alert
-- Course: 1018
-- Module 05
-- ==========================================================
--
-- Automation without a watchdog just moves the problem from "nobody is
-- reading reviews" to "nobody is watching the automation." The module's
-- alert step flattens the AI_SENTIMENT VARIANT result to count negative
-- categories in the last hour, then writes an alert row if the count
-- crosses a threshold. Apply the same pattern to flag a spike in negative
-- sentiment specifically about a "shipping_damage" category, over the
-- last 4 hours, using a lower threshold since this signal is rarer.
--
-- Your task:
-- 1. Write a query that flattens sentiment_result:categories from
--    ticket_scores, filters for category name = 'shipping_damage' AND
--    sentiment = 'negative', within the last 4 hours.
-- 2. Wrap it as a subquery and INSERT a row into `alerts` (alert_type,
--    detail, negative_count, triggered_at) only when the count exceeds 10.
-- ==========================================================

-- YOUR CODE:

SELECT COUNT(*) AS negative_count
FROM ticket_scores,
     LATERAL FLATTEN(input => ___:categories) c
WHERE c.value:name::string = '___'
  AND c.value:sentiment::string = '___'
  AND scored_at >= DATEADD(___, -4, CURRENT_TIMESTAMP());

INSERT INTO alerts (alert_type, detail, negative_count, triggered_at)
SELECT
  'shipping_damage_spike',
  'Shipping-damage complaints trending up',
  negative_count,
  CURRENT_TIMESTAMP()
FROM (
  SELECT COUNT(*) AS negative_count
  FROM ticket_scores,
       LATERAL FLATTEN(input => sentiment_result:categories) c
  WHERE c.value:name::string = 'shipping_damage'
    AND c.value:sentiment::string = 'negative'
    AND scored_at >= DATEADD(hour, -4, CURRENT_TIMESTAMP())
)
WHERE negative_count > ___;


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
SELECT COUNT(*) AS negative_count
FROM ticket_scores,
     LATERAL FLATTEN(input => sentiment_result:categories) c
WHERE c.value:name::string = 'shipping_damage'
  AND c.value:sentiment::string = 'negative'
  AND scored_at >= DATEADD(hour, -4, CURRENT_TIMESTAMP());

INSERT INTO alerts (alert_type, detail, negative_count, triggered_at)
SELECT
  'shipping_damage_spike',
  'Shipping-damage complaints trending up',
  negative_count,
  CURRENT_TIMESTAMP()
FROM (
  SELECT COUNT(*) AS negative_count
  FROM ticket_scores,
       LATERAL FLATTEN(input => sentiment_result:categories) c
  WHERE c.value:name::string = 'shipping_damage'
    AND c.value:sentiment::string = 'negative'
    AND scored_at >= DATEADD(hour, -4, CURRENT_TIMESTAMP())
)
WHERE negative_count > 10;
*/
