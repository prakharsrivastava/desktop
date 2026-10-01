# ==========================================================
# Exercise 19.1 — Fix the Chat App That Re-Answers Every Turn on Rerun
# Course: 1018
# Module 19
# ==========================================================
#
# Field reps reported the Cortex Search chat app hangs or times out after
# 4-5 turns, even though the first question or two was fast. The bug:
# Streamlit reruns the ENTIRE script top-to-bottom on every widget
# interaction, so a loop over st.session_state.messages re-triggers Search
# + AI_COMPLETE for turns that were already answered — not just the new
# one. Call count grows every rerun until cumulative latency trips the
# warehouse statement timeout. The fix: only the newest user turn calls
# Search + AI_COMPLETE; prior turns render from session_state for free.
#
# Your task:
# 1. Build the chat scaffold: session_state.messages list, replay prior
#    turns with st.chat_message, and capture the new turn with st.chat_input.
# 2. Write the BUGGY handler that loops over ALL messages on every rerun
#    (so you can see exactly what breaks).
# 3. Write the FIXED handler that processes only the new user_msg —
#    remember SEARCH_PREVIEW's second argument must be a literal JSON
#    string, not an OBJECT_CONSTRUCT() VARIANT.
# ==========================================================

# YOUR CODE:

import streamlit as st
from snowflake.snowpark.context import get_active_session

session = get_active_session()

# Step 1: chat scaffold
if "___" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["___"]):
        st.write(msg["content"])

user_msg = st.chat_input("Ask about this store's planogram or last visit")
if user_msg:
    st.session_state.messages.append({"role": "___", "content": user_msg})

# Step 2: BUGGY — do NOT ship this (loops over every past turn, every rerun)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        results = session.sql(
            """SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
                 'store_docs_search_svc',
                 '{"query": "' || ? || '", "limit": 3}'
               ) AS hits""",
            params=[___],
        ).___()

# Step 3: FIX — only the newest user turn calls Search + AI_COMPLETE
if ___:
    results = session.sql(
        """SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
             'store_docs_search_svc',
             '{"query": "' || ? || '", "limit": 3}'
           ) AS hits""",
        params=[___],
    ).collect()
    # prior turns are read straight from session_state for display —
    # never re-sent to Search or AI_COMPLETE


# ==========================================================
# SOLUTION (attempt first before scrolling!)
# ==========================================================
"""
import streamlit as st
from snowflake.snowpark.context import get_active_session

session = get_active_session()

# Step 1: chat scaffold
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_msg = st.chat_input("Ask about this store's planogram or last visit")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})

# Step 2: BUGGY handler — this loop runs again on EVERY Streamlit rerun,
# re-triggering Search + AI_COMPLETE for turns already answered
for msg in st.session_state.messages:
    if msg["role"] == "user":
        results = session.sql(
            \"\"\"SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
                 'store_docs_search_svc',
                 '{"query": "' || ? || '", "limit": 3}'
               ) AS hits\"\"\",
            params=[msg["content"]],
        ).collect()
        # ...then re-answers every prior question, on every single rerun

# Step 3: FIX — process only the new turn each rerun; history renders
# from session_state for free, so turn count no longer multiplies model
# calls. SEARCH_PREVIEW's second argument must be a literal JSON string,
# not an OBJECT_CONSTRUCT() VARIANT.
if user_msg:
    results = session.sql(
        \"\"\"SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
             'store_docs_search_svc',
             '{"query": "' || ? || '", "limit": 3}'
           ) AS hits\"\"\",
        params=[user_msg],
    ).collect()
"""
