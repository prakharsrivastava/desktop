# ============================================================
# Course 1018 — Module 22
# On-slide PY reference  (1 snippets)
# Authentic code as shown on the course slides. Copy, run, experiment.
# ============================================================


# ----------------------------------------------------------
# The Streamlit Front End, Calling the Agent
# ----------------------------------------------------------

import streamlit as st
from snowflake.snowpark.context import get_active_session

session = get_active_session()
question = st.chat_input('Ask about Meridian sales, vendors, or risk')

if question:
    result = session.sql(
        f"CALL meridian_intelligence_agent!RESPOND('{question}')"
    ).collect()
    st.chat_message('assistant').write(result[0]['RESPONSE'])

