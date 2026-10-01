# ==========================================================
# Exercise 19.2 — Cache Repeated Questions, and Pick the Right Runtime Lane
# Course: 1018
# Module 19
# ==========================================================
#
# A pilot team's chat assistant got popular fast — reps kept re-asking
# near-identical questions, and every keystroke variant of "what's the Q2
# promo lift" triggered a brand-new AI_COMPLETE call. A cache lookup
# keyed on the normalized question plus region turns a repeated question
# from a billed call into a free dictionary lookup. Separately, a
# supply-planning lead once asked for an app that checks inventory,
# drafts a reorder, and waits for a human email approval — that's a
# multi-step, long-running, human-in-the-loop workload that does NOT fit
# inside a single Streamlit-in-Snowflake button click.
#
# Your task:
# 1. Build a session-level answer cache keyed on (region, normalized
#    question) that short-circuits AI_COMPLETE on a repeat question.
# 2. Write the anti-pattern button handler that tries to run a multi-tool,
#    human-in-the-loop agent synchronously inside a Streamlit click.
# 3. Fix it: the button should only SUBMIT the request to an
#    already-deployed container-runtime agent service, not run it inline.
# ==========================================================

# YOUR CODE:

import streamlit as st
from snowflake.snowpark.context import get_active_session

session = get_active_session()
region = "Midwest"
question = "Did the Q2 end-cap promo lift units?"
prompt = f"Region: {region}\nQuestion: {question}"

# Step 1: session-level answer cache
if "___" not in st.session_state:
    st.session_state.answer_cache = {}

cache_key = (region, question.strip().___())

if cache_key in st.session_state.answer_cache:
    answer = st.session_state.answer_cache[___]
else:
    answer = session.sql(
        "SELECT AI_COMPLETE('claude-4-sonnet', ?) AS a", params=[prompt]
    ).to_pandas()["A"][0]
    st.session_state.answer_cache[___] = answer

# Step 2: ANTI-PATTERN — do not do this inside Streamlit in Snowflake
sku, store = "SKU-1", "Store-412"
if st.button("Run Reorder Agent"):
    plan = call_agent_orchestrator(___, ___)  # checks inventory -> drafts PO -> waits for a human email approval
    st.write(plan)  # never returns in time

# Step 3: RIGHT — Streamlit only submits the request
if st.button("Submit Reorder Request"):
    session.sql(
        "CALL cpg_agent_pool.___(?, ?)", params=[sku, store]
    ).collect()
    st.success("Request submitted — track status in the Agent Ops dashboard")


# ==========================================================
# SOLUTION (attempt first before scrolling!)
# ==========================================================
"""
import streamlit as st
from snowflake.snowpark.context import get_active_session

session = get_active_session()
region = "Midwest"
question = "Did the Q2 end-cap promo lift units?"
prompt = f"Region: {region}\\nQuestion: {question}"

# Step 1: cache_key normalizes case and whitespace so near-duplicate
# phrasing still hits the cache. Cache hit = zero new model tokens spent.
if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}

cache_key = (region, question.strip().lower())

if cache_key in st.session_state.answer_cache:
    answer = st.session_state.answer_cache[cache_key]
else:
    answer = session.sql(
        "SELECT AI_COMPLETE('claude-4-sonnet', ?) AS a", params=[prompt]
    ).to_pandas()["A"][0]
    st.session_state.answer_cache[cache_key] = answer
# A pilot's 200-question day, uncached: ~200 AI_COMPLETE calls. Cached:
# closer to 60-80 once repeats and near-duplicates collapse.

# Step 2: ANTI-PATTERN — a Streamlit-in-Snowflake button handler trying to
# run a full multi-tool reorder agent synchronously, inside the warehouse
# turn. Streamlit in Snowflake is a single request-response turn under the
# warehouse's statement timeout — it cannot hold a process open waiting on
# a person to reply to an email.
sku, store = "SKU-1", "Store-412"
if st.button("Run Reorder Agent"):
    plan = call_agent_orchestrator(sku, store)  # checks inventory -> drafts PO -> waits for a human email approval
    st.write(plan)  # never returns in time — the human approval step alone can take hours

# Step 3: RIGHT — Streamlit submits the request; the agent loop runs in
# container runtime (SPCS). Same button, same click for the category
# manager — the orchestration loop just lives in the lane built to hold it.
if st.button("Submit Reorder Request"):
    session.sql(
        "CALL cpg_agent_pool.submit_reorder_request(?, ?)", params=[sku, store]
    ).collect()
    st.success("Request submitted — track status in the Agent Ops dashboard")

# Decision test before writing a line of app code: does it need more than
# one tool call chained with re-planning? Does it need to run longer than
# a single page load? Does it need to wait on something outside Snowflake?
# Any "yes" points to container runtime / external app (SPCS); all "no"
# keeps you in the simple Streamlit-in-Snowflake lane.
"""
