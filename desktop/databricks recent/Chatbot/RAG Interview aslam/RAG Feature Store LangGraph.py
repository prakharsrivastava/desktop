# Databricks notebook source
# DBTITLE 1,RAG + Feature Store LangGraph Pipeline
# MAGIC %md
# MAGIC # RAG + Feature Store LangGraph Pipeline
# MAGIC
# MAGIC LangGraph-based orchestrator that routes queries to a RAG vector store, a feature store, or both in parallel — then merges results for final LLM generation.

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install "langgraph<1.0" "langchain-core<1.0" langgraph-checkpoint-mongodb pymongo

# COMMAND ----------

# DBTITLE 1,Setup MongoDB Secret
# ------------------------------------------------------------------
# MONGODB SECRET SETUP (run once)
# ------------------------------------------------------------------
# Step 1: Create a Databricks secret scope
# databricks secrets create-scope mongodb_scope
#
# Step 2: Store your MongoDB connection string as a secret
# databricks secrets put-secret mongodb_scope mongodb_uri \
#   --string-value "mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority"
#
# Step 3: Verify the secret is retrievable (run the cell below)
# ------------------------------------------------------------------

# Test that the secret is accessible
try:
    _uri = dbutils.secrets.get(scope="mongodb_scope", key="mongodb_uri")
    print(f"Secret found! URI starts with: {_uri[:20]}...")
except Exception as e:
    print(f"Secret not found: {e}")
    print("Run the CLI commands above to set up the MongoDB secret.")
    print("Or replace dbutils.secrets.get() with a direct string for local testing.")

# COMMAND ----------

# DBTITLE 1,LangGraph Pipeline
from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pydantic import BaseModel
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver


# -------------------------
# 0. MONGODB SETUP
# -------------------------
# MongoDB URI stored in Databricks Secrets for security.
# To set it up, run the "Setup MongoDB Secret" cell below (Cell 3) first.
MONGODB_URI = dbutils.secrets.get(scope="mongodb_scope", key="mongodb_uri")

try:
    mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    mongo_client.admin.command("ping")  # verify connection
    checkpointer = MongoDBSaver(mongo_client)
    print("MongoDB connected -- conversation persistence enabled")
except Exception as e:
    print(f"MongoDB not reachable ({e}). Falling back to MemorySaver.")
    print("Ensure the MongoDB secret is set up (run the Setup cell) and the URI is valid.")
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()


# -------------------------
# 1. STATE
# -------------------------
class GraphState(TypedDict, total=False):
    query: str
    user_id: str
    route: Literal["rag", "feature", "both"]

    rag_context: list
    feature_context: dict

    summary: str            # running conversation summary (persisted via MongoDB)
    messages: list          # conversation history [{role, content}]

    final_answer: str


# -------------------------
# 2. PLANNER OUTPUT
# -------------------------
class Plan(BaseModel):
    route: Literal["rag", "feature", "both"]


# -------------------------
# 3. PLANNER NODE (with summary context)
# -------------------------
def planner_node(state: GraphState):

    planner_llm = llm.with_structured_output(Plan)

    prev_summary = state.get("summary", "")
    summary_block = (
        f"\n\nPrevious conversation summary:\n{prev_summary}"
        if prev_summary else ""
    )

    plan = planner_llm.invoke(
        f"""
        Decide which source is required.

        rag = document/knowledge question
        feature = student performance/data question
        both = needs knowledge + student data

        Query: {state['query']}
        {summary_block}
        """
    )

    return {"route": plan.route}


# -------------------------
# 4. RAG NODE
# -------------------------
def rag_node(state: GraphState):

    docs = vector_store.similarity_search(
        state["query"],
        k=5
    )

    return {
        "rag_context": docs
    }


# -------------------------
# 5. FEATURE STORE NODE
# -------------------------
def feature_node(state: GraphState):

    features = get_student_features(
        state["query"]
    )

    return {
        "feature_context": features
    }


# -------------------------
# 6. ROUTER
# -------------------------
def router(state: GraphState):

    if state["route"] == "rag":
        return [Send("rag", state)]

    elif state["route"] == "feature":
        return [Send("feature", state)]

    else:
        # BOTH → parallel fan-out
        return [
            Send("rag", state),
            Send("feature", state)
        ]


# -------------------------
# 7. FINAL GENERATION (with summary context)
# -------------------------
def generate_node(state: GraphState):

    prev_summary = state.get("summary", "")
    summary_block = (
        f"\n\nPrevious conversation summary:\n{prev_summary}"
        if prev_summary else ""
    )

    response = llm.invoke(
        f"""
        Question:
        {state['query']}
        {summary_block}

        RAG Context:
        {state.get('rag_context', [])}

        Feature Context:
        {state.get('feature_context', {})}

        Generate the final grounded answer.
        """
    )

    # Append to conversation history
    messages = state.get("messages", [])
    messages = messages + [
        {"role": "user", "content": state["query"]},
        {"role": "assistant", "content": response.content},
    ]

    return {
        "final_answer": response.content,
        "messages": messages,
    }


# -------------------------
# 8. SUMMARIZE NODE
# -------------------------
def summarize_node(state: GraphState):

    prev_summary = state.get("summary", "")

    summary_response = llm.invoke(
        f"""
        You are a conversation summarizer.
        Create a concise running summary of the conversation so far.
        This summary persists in MongoDB and provides context for future
        questions from the same user — even after an app restart.

        Previous summary (if any):
        {prev_summary}

        Latest exchange:
        User: {state['query']}
        Assistant: {state['final_answer']}

        Provide an updated summary that captures all key context.
        """
    )

    return {"summary": summary_response.content}


# -------------------------
# 9. BUILD LANGGRAPH (with MongoDB checkpointer)
# -------------------------

graph_builder = StateGraph(GraphState)

graph_builder.add_node("planner", planner_node)
graph_builder.add_node("rag", rag_node)
graph_builder.add_node("feature", feature_node)
graph_builder.add_node("generate", generate_node)
graph_builder.add_node("summarize", summarize_node)


# START → PLANNER
graph_builder.add_edge(
    START,
    "planner"
)


# PLANNER → ROUTER → RAG / FEATURE / BOTH
graph_builder.add_conditional_edges(
    "planner",
    router,
    ["rag", "feature"]
)


# Both branches converge → generate
graph_builder.add_edge(
    "rag",
    "generate"
)

graph_builder.add_edge(
    "feature",
    "generate"
)


# generate → summarize → END
graph_builder.add_edge(
    "generate",
    "summarize"
)

graph_builder.add_edge(
    "summarize",
    END
)


# Compile with MongoDB checkpointer for conversation persistence
graph = graph_builder.compile(checkpointer=checkpointer)

# COMMAND ----------

# DBTITLE 1,Graph Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 11)
ax.axis('off')
ax.set_title('RAG + Feature Store LangGraph Pipeline', fontsize=16, fontweight='bold', pad=20)

nodes = {
    'START':    (5.0, 10.0, '#bfb6fc'),
    'planner':  (5.0, 8.3, '#f2f0ff'),
    'rag':      (2.0, 5.8, '#f2f0ff'),
    'feature':  (8.0, 5.8, '#f2f0ff'),
    'generate': (5.0, 3.8, '#f2f0ff'),
    'summarize':(5.0, 2.2, '#d4f4dd'),
    'END':      (5.0, 0.8, '#bfb6fc'),
}

for name, (x, y, color) in nodes.items():
    style = 'round' if name not in ('START', 'END') else 'round'
    box = mpatches.FancyBboxPatch((x - 1.0, y - 0.35), 2.0, 0.7,
                                  boxstyle=style + ',pad=0.15',
                                  facecolor=color, edgecolor='#6b5fb0', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, name, ha='center', va='center', fontsize=11, fontweight='bold', color='#333')

def draw_arrow(x1, y1, x2, y2, dashed=False, label=''):
    style = '->'
    ls = '--' if dashed else '-'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color='#6b5fb0', lw=1.8, ls=ls,
                                connectionstyle='arc3,rad=0'))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.25, label, ha='center', va='center', fontsize=8,
                fontstyle='italic', color='#888')

draw_arrow(5.0, 9.65, 5.0, 8.65, label='')
draw_arrow(4.3, 7.95, 2.5, 6.15, dashed=True, label='route=rag')
draw_arrow(5.7, 7.95, 7.5, 6.15, dashed=True, label='route=feature')
draw_arrow(4.0, 7.95, 2.3, 6.15, dashed=True, label='')
draw_arrow(6.0, 7.95, 7.7, 6.15, dashed=True, label='')
draw_arrow(2.5, 5.45, 4.3, 4.15, label='merge')
draw_arrow(7.5, 5.45, 5.7, 4.15, label='merge')
draw_arrow(5.0, 3.45, 5.0, 2.55, label='save')
draw_arrow(5.0, 1.85, 5.0, 1.15, label='')

ax.text(5.0, 7.05, 'route = both =>\nparallel fan-out', ha='center', va='center',
        fontsize=8, fontstyle='italic', color='#999',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
ax.text(7.2, 2.2, 'MongoDB\ncheckpoint', ha='center', va='center',
        fontsize=7, fontstyle='italic', color='#4a9d5f',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Usage: Thread ID + Restart Recovery
# ------------------------------------------------------------------
# USAGE: Invoke with thread_id + user_id and retrieve prev summary
# ------------------------------------------------------------------
#
# thread_id = f"{user_id}_{session_id}"
# MongoDB checkpointer auto-saves state after every super-step.
# On app restart, invoke with the SAME thread_id to load prev summary.
#

USER_ID = "student_001"
SESSION_ID = "chat_session_1"
THREAD_ID = f"{USER_ID}_{SESSION_ID}"

# Config passed to every invoke — checkpointer uses thread_id to
# load/save conversation state (including summary) from MongoDB.
config = {"configurable": {"thread_id": THREAD_ID}}

# --- Turn 1 ---
result = graph.invoke(
    {"query": "What is my attendance this semester?", "user_id": USER_ID},
    config=config
)
print("Turn 1 Answer:", result["final_answer"])

# --- Turn 2 (same thread — summary from Turn 1 is available) ---
result = graph.invoke(
    {"query": "How does that compare to last semester?", "user_id": USER_ID},
    config=config
)
print("Turn 2 Answer:", result["final_answer"])

# --- Retrieve current conversation summary (persisted in MongoDB) ---
state = graph.get_state(config=config)
print("\n--- Conversation Summary (persisted) ---")
print(state.values.get("summary", "No summary yet"))

# --- App restart scenario ---
# After restart, re-invoke with the SAME thread_id.
# MongoDB checkpointer loads the saved state, including the
# running summary, so the LLM has full conversation context.
#
# result = graph.invoke(
#     {"query": "What did we discuss earlier?", "user_id": USER_ID},
#     config=config  # same thread_id -> loads prev summary from MongoDB
# )
# print(result["final_answer"])