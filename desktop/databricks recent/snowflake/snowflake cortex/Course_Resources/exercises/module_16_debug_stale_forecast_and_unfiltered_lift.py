# ==========================================================
# Exercise 16.2 — Debug: The Agent Approved a Promo That Lost Margin
# Course: 1018
# Module 16
# ==========================================================
#
# The trade-promotion agent approved a 20 percent discount, but the promo
# closed with negative net margin instead of the promised lift. This wasn't a
# reasoning failure in the agent itself — it was two quiet data problems
# feeding it: stale seasonality in Forecast, and an un-filtered cannibalization
# signal in Analyst. Same lesson as module_13's anomaly false-positives: fix
# the data feeding the tool, don't distrust the orchestration layer.
#
# Your task:
# 1. Diagnose Cortex ML Forecast first: what's wrong with training it on an
#    18-month-old seasonality profile that predates a category reset?
# 2. Diagnose Cortex Analyst second: what's wrong with a historical-lift query
#    that doesn't exclude a concurrent competitor promo running the same week?
# 3. Write the two fixes: a retrain call using a rolling, shorter seasonality
#    window, and a WHERE-clause note on the semantic model excluding
#    overlapping competitor promos from the lift baseline.
# ==========================================================

# YOUR CODE:

# DIAGNOSIS STEP 1 — Forecast
# symptom: promo closed at negative net margin despite a +9% forecast
# question: how old was the seasonality profile Forecast was trained on, and
#           did a category reset happen inside that window?
stale_window_months = ___   # the training window that caused the miss

# DIAGNOSIS STEP 2 — Analyst
# question: did the historical-lift query filter out concurrent competitor
#           promos running the same week, or did they inflate the baseline?
# (conceptual WHERE clause on the semantic model's lift query)
# WHERE NOT EXISTS (SELECT 1 FROM competitor_promos cp
#                    WHERE cp.___ = promo_fact.sku_id
#                      AND cp.promo_week = promo_fact.___)

# FIX 1 — retrain the underlying model on a rolling 6-month seasonality
# window instead of the stale 18-month one (the retrain itself is the same
# CREATE OR REPLACE pattern used throughout this course — the call below is
# the unchanged scoring call once that retrain is done)
forecast_df = session.call(
    'CPG_MODELS.PROMO_DEMAND_FORECAST',
    sku_id, discount_pct, horizon_weeks=___,
)
# FIX note: retrain on a rolling ___-month window, not the stale 18-month one

# FIX 2 — note the semantic model change needed (excluding overlapping
# competitor promos from the lift baseline) — describe it as a comment,
# the actual edit lives in promo_semantic_model.yaml
# FIX: add a WHERE clause excluding overlapping competitor promos from ___


# ==========================================================
# SOLUTION (attempt first before scrolling!)
# ==========================================================
"""
# DIAGNOSIS STEP 1 — Forecast
# ROOT CAUSE: Forecast's demand curve was trained on an 18-month-old
# seasonality profile — from BEFORE a category reset. A stale seasonality
# window means the model is projecting demand against a pattern that no
# longer reflects how this category actually behaves today.
stale_window_months = 18   # the training window that caused the miss

# DIAGNOSIS STEP 2 — Analyst
# ROOT CAUSE: the historical-lift query didn't exclude a concurrent
# competitor promo that ran the same week, inflating the baseline lift it
# reported. The reported +12% lift / 4% cannibalization baseline was
# contaminated by a promo effect that wasn't this SKU's own discount.
# WHERE NOT EXISTS (SELECT 1 FROM competitor_promos cp
#                    WHERE cp.sku_id = promo_fact.sku_id
#                      AND cp.promo_week = promo_fact.promo_week)

# FIX 1 — retrain the underlying model on a rolling 6-month seasonality
# window instead of the stale 18-month one, then re-score with the same
# unchanged call signature used throughout this course:
forecast_df = session.call(
    'CPG_MODELS.PROMO_DEMAND_FORECAST',
    sku_id, discount_pct, horizon_weeks=4,
)
# FIX note: retrain on a rolling 6-month window, not the stale 18-month one

# FIX 2 — the semantic model's lift query needs a WHERE clause excluding
# overlapping competitor promos from the cannibalization/lift baseline before
# Analyst answers the next "what was the historical lift" question.
# FIX: add a WHERE clause excluding overlapping competitor promos from the
# Analyst semantic model's lift baseline query.

# TAKEAWAY: two quiet data problems, not a reasoning failure in the agent
# itself. This is exactly why you re-verify the ground truth beneath each
# tool whenever it moves — same discipline as module_13's "check the
# training window first" debug slide, applied one layer up at the agent
# orchestration level.
"""
