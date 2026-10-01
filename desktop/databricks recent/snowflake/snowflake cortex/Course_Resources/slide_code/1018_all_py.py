# ============================================================
# 1018 — COMPLETE PY REFERENCE  (9 snippets)
# Every on-slide snippet across all modules. Ctrl+F by keyword or title.
# ============================================================



# ########################################################
# MODULE 08
# ########################################################

# --- Retrieving Top-k Passages for a Live Question ---

# Production path: call the Cortex Search REST/Python API via snowflake.core
from snowflake.core import Root

root = Root(session)
search_service = (
    root.databases["cpg_db"].schemas["public"]
    .cortex_search_services["cpg_product_spec_search"]
)
response = search_service.search(
    query="does the granola bar SKU contain tree nuts",
    columns=["chunk_id", "chunk_text", "sku", "doc_type", "page_number", "effective_date"],
    filter={"@eq": {"doc_type": "product_spec"}},
    limit=3,
)
retrieved_chunks = response.results



# ########################################################
# MODULE 15
# ########################################################

# --- Anti-Pattern First — A Hand-Rolled If/Else Tool Router ---

# WRONG — Priya's first instinct: bypass the agent's own routing, hand-code it
def pick_tool(question: str):
    q = question.lower()
    if "how many" in q or "total" in q or "$" in q:
        return "cortex_analyst_sales_model"
    elif "why" in q or "policy" in q:
        return "cortex_search_returns_docs"
    else:
        return "custom_pricing_proc"  # default guess — zero evidence

# RIGHT — agent-level routing instructions (plain-language tie-breakers)
When the user asks about NUMBERS, TOTALS, or TRENDS in sales/returns,
route to cortex_analyst_sales_model.
When the user asks WHY, POLICY, or references a DOCUMENT/NOTICE,
route to cortex_search_returns_docs.



# ########################################################
# MODULE 19
# ########################################################

# --- The App Skeleton ---

import streamlit as st
from snowflake.snowpark.context import get_active_session

session = get_active_session()
st.title("CPG Category Insight Assistant")

region = st.selectbox("Region", ["Northeast", "Midwest", "South", "West"])
question = st.text_area("Ask a question about this region's promo performance")

if st.button("Get Insight"):
    st.session_state["last_question"] = question


# --- Wiring the Button to AI_COMPLETE ---

if st.session_state.get("last_question"):
    prompt = f"""Region: {region}
    Question: {st.session_state['last_question']}
    Answer using only promo_performance_summary for this region."""

    df = session.sql(
        "SELECT AI_COMPLETE('claude-4-sonnet', ?) AS answer", params=[prompt]
    ).to_pandas()

    st.write(df["ANSWER"][0])


# --- Chat History in Session State ---

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_msg = st.chat_input("Ask about this store's planogram or last visit")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})


# --- Debug — My Streamlit App Times Out on Every Cortex Call ---

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


# --- Anti-Pattern — Calling Agents APIs from Streamlit ---

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


# --- Caching the Call in Session State ---

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



# ########################################################
# MODULE 22
# ########################################################

# --- The Streamlit Front End, Calling the Agent ---

import streamlit as st
from snowflake.snowpark.context import get_active_session

session = get_active_session()
question = st.chat_input('Ask about Meridian sales, vendors, or risk')

if question:
    result = session.sql(
        f"CALL meridian_intelligence_agent!RESPOND('{question}')"
    ).collect()
    st.chat_message('assistant').write(result[0]['RESPONSE'])

