# ============================================================
# Course 1018 — Module 19
# On-slide PY reference  (6 snippets)
# Authentic code as shown on the course slides. Copy, run, experiment.
# ============================================================


# ----------------------------------------------------------
# The App Skeleton
# ----------------------------------------------------------

import streamlit as st
from snowflake.snowpark.context import get_active_session

session = get_active_session()
st.title("CPG Category Insight Assistant")

region = st.selectbox("Region", ["Northeast", "Midwest", "South", "West"])
question = st.text_area("Ask a question about this region's promo performance")

if st.button("Get Insight"):
    st.session_state["last_question"] = question


# ----------------------------------------------------------
# Wiring the Button to AI_COMPLETE
# ----------------------------------------------------------

if st.session_state.get("last_question"):
    prompt = f"""Region: {region}
    Question: {st.session_state['last_question']}
    Answer using only promo_performance_summary for this region."""

    df = session.sql(
        "SELECT AI_COMPLETE('claude-4-sonnet', ?) AS answer", params=[prompt]
    ).to_pandas()

    st.write(df["ANSWER"][0])


# ----------------------------------------------------------
# Chat History in Session State
# ----------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_msg = st.chat_input("Ask about this store's planogram or last visit")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})


# ----------------------------------------------------------
# Debug — My Streamlit App Times Out on Every Cortex Call
# ----------------------------------------------------------

# BUGGY — this loop runs again on EVERY Streamlit rerun
for msg in st.session_state.messages:
    if msg["role"] == "user":
        results = session.sql(
            """SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
                 'store_docs_search_svc',
                 '{"query": "' || ? || '", "limit": 3}'
               ) AS hits""",
            params=[msg["content"]],
        ).collect()
        # ...re-answers every prior question, on every rerun

# FIX — only the newest user turn calls Search + AI_COMPLETE
if user_msg:
    results = session.sql(
        """SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
             'store_docs_search_svc',
             '{"query": "' || ? || '", "limit": 3}'
           ) AS hits""",
        params=[user_msg],
    ).collect()


# ----------------------------------------------------------
# Anti-Pattern — Calling Agents APIs from Streamlit
# ----------------------------------------------------------

# ANTI-PATTERN — do not do this inside Streamlit in Snowflake
if st.button("Run Reorder Agent"):
    plan = call_agent_orchestrator(sku, store)
    # checks inventory -> drafts PO -> waits for a human email approval
    st.write(plan)  # never returns in time — approval alone can take hours

# RIGHT — Streamlit submits the request; the agent loop
# runs in container runtime
if st.button("Submit Reorder Request"):
    session.sql(
        "CALL cpg_agent_pool.submit_reorder_request(?, ?)",
        params=[sku, store]
    ).collect()
    st.success("Request submitted — track status in Agent Ops dashboard")


# ----------------------------------------------------------
# Caching the Call in Session State
# ----------------------------------------------------------

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

