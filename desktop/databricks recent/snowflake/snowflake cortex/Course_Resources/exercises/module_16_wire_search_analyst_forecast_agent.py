# ==========================================================
# Exercise 16.1 — Wire Search, Analyst, and Forecast Into One Promo Agent
# Course: 1018
# Module 16
# ==========================================================
#
# "Should we run 20% off Brand X yogurt next month?" used to trigger three
# separate manual investigations (category management, trade finance, supply
# chain) that never talked to each other — a 6-8 week cycle that often closed
# after the retail promo calendar had already locked. One orchestrator agent
# wiring three Cortex tools together collapses that same rigor into minutes:
# search grounds context first, analyst quantifies history, forecast projects
# forward. Call order matters.
#
# Your task:
# 1. Query Cortex Search (via SEARCH_PREVIEW) scoped to the SKU's category for
#    prior promo post-mortems and category reports on similar discount depths.
# 2. Call Cortex Analyst with the plain-language lift question over the promo
#    semantic model — no hand-written SQL, Analyst generates it.
# 3. Call the Snowpark stored procedure that projects demand under this exact
#    discount scenario.
# 4. Preserve the call order: search first, analyst second, forecast last.
# ==========================================================

# YOUR CODE:

import requests

# Step 1 — ground context: what happened last time a similar promo ran?
search_result = session.sql("""
    SELECT PARSE_JSON(
      SNOWFLAKE.CORTEX.___(
        '___',
        '{"query": "20 percent discount promo yogurt cannibalization lessons",
          "columns": ["snippet", "source_doc"],
          "filter": {"@eq": {"category": "___"}},
          "limit": ___}'
      )
    )['results']
""").collect()

# Step 2 — quantify history: what was the actual lift/cannibalization?
lift_question = "What was the average lift and cannibalization for a 20 percent yogurt discount?"
response = requests.___(
    analyst_url,
    json={'semantic_model_file': '___', 'messages': [{'role': 'user', 'content': ___}]},
)

# Step 3 — project forward: what will demand do under THIS discount?
forecast_df = session.call('___', sku_id, discount_pct, horizon_weeks=___)

# Step 4 — the orchestrator merges all three signals into one recommendation
# (search context + analyst lift/cannibalization + forecast projection)


# ==========================================================
# SOLUTION (attempt first before scrolling!)
# ==========================================================
"""
import requests

# Step 1 — ground context: what happened last time a similar promo ran?
search_result = session.sql('''
    SELECT PARSE_JSON(
      SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
        'CPG_DOCS.PROMO_SEARCH_SVC',
        '{"query": "20 percent discount promo yogurt cannibalization lessons",
          "columns": ["snippet", "source_doc"],
          "filter": {"@eq": {"category": "DAIRY_YOGURT"}},
          "limit": 5}'
      )
    )['results']
''').collect()
# SEARCH_PREVIEW is a quick ad hoc query helper; production agent code
# typically calls the Cortex Search Python or REST API instead of raw SQL.
# Limiting to 5 results keeps the tool call fast and the context window lean.

# Step 2 — quantify history: what was the actual lift/cannibalization?
lift_question = "What was the average lift and cannibalization for a 20 percent yogurt discount?"
response = requests.post(
    analyst_url,
    json={
        'semantic_model_file': 'promo_semantic_model.yaml',
        'messages': [{'role': 'user', 'content': lift_question}],
    },
)
# Cortex Analyst returns the generated SQL alongside the executed result set.
# Real result from the case study: average lift 12%, cannibalization 4%.

# Step 3 — project forward: what will demand do under THIS discount?
forecast_df = session.call(
    'CPG_MODELS.PROMO_DEMAND_FORECAST',
    sku_id, discount_pct, horizon_weeks=4,
)
# Real result from the case study: units projected up 9% for this discount depth.

# Step 4 — the orchestrator agent calls all three tools, then merges search
# context, lift stats, and the forecast into one recommendation. Tool-calling
# order matters: search grounds context first, analyst quantifies history,
# forecast projects forward. In the walked case study, the top retrieved
# search snippet wasn't a number — it was a warning that a similar 20 percent
# promo caused a 3-day stockout the prior Q3. The agent carried that caution
# forward as a supply-chain flag attached to its final recommendation, instead
# of discarding it because it didn't fit a spreadsheet cell.
"""
