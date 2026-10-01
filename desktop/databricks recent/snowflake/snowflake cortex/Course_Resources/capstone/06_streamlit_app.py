# ==========================================================
# Capstone Layer 6 — Streamlit Front End
# Meridian Foods: the app people actually adopt
# Pattern from: module 19 (Streamlit in Snowflake app-layer pattern)
# ==========================================================
#
# The agent (Layer 5) works from a SQL console; Meridian's category
# managers and field reps won't touch one. This app's ONLY job is to pass
# user text to the agent and render the response — no business logic gets
# duplicated here. A chat-style input box, one text response area, and a
# source-citation panel are all the UI needs.
#
# get_active_session reuses the same Snowpark session pattern from module
# 19's app build — no separate API key, no external hosting, the app runs
# inside the same governed account as the data.

import streamlit as st
from snowflake.snowpark.context import get_active_session

session = get_active_session()

# TODO 1: prompt for a chat-style question, matching the capstone's
# framing (Meridian sales, vendors, or risk).
question = st.chat_input("___")  # TODO: replace with the prompt text

if question:
    # TODO 2: call the agent registered in Layer 5 (meridian_intelligence_agent)
    # via its RESPOND method, passing the user's question through.
    result = session.sql(
        f"CALL ___!RESPOND('{question}')"  # TODO: agent name before !RESPOND
    ).collect()

    # TODO 3: render the agent's RESPONSE column as an assistant chat
    # message.
    st.chat_message("assistant").write(result[0]["___"])  # TODO: column name

# TODO 4 (your extension, README's "done" criteria): add a small
# source-citation panel below the chat message — e.g. an st.expander
# showing which tool (search / analyst / forecast / anomaly) the agent
# routed to for this turn, if the agent's response payload carries that
# metadata.

# Checkpoint 5 (README rubric, app half): the app's only logic is passing
# text to the agent and rendering RESPONSE — no duplicated business logic
# in the UI layer.
