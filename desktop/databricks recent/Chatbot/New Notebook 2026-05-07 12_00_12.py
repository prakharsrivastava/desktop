# Databricks notebook source
# DBTITLE 1,Cell 1
# MAGIC %pip install \
# MAGIC     "numpy>=1.21.6,<2.0" \
# MAGIC     "pandas>=1.0.5,<3.0" \
# MAGIC     langchain \
# MAGIC     langchain-community \
# MAGIC     langgraph \
# MAGIC     langchain-databricks \
# MAGIC     langchain-huggingface \
# MAGIC     sentence-transformers \
# MAGIC     --quiet
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from langchain_databricks import ChatDatabricks
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

import requests
import numpy as np
import pandas as pd

# COMMAND ----------


w = WorkspaceClient()


llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3.1-405b-instruct",
    temperature=0.1,
    max_tokens=200
)



# COMMAND ----------


VS_ENDPOINT  = "rag_endpoint"
VS_INDEX     = "agents.default.rag_index"
TEXT_COLUMN  = "content"

from langchain_databricks import DatabricksVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings

# Create embedding model for query vectorization (384 dimensions to match index)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = DatabricksVectorSearch(
    endpoint=VS_ENDPOINT,
    index_name=VS_INDEX,
    text_column=TEXT_COLUMN,
    embedding=embedding_model,
)


from langchain.tools import tool

dummy_events = {
    "Dallas": [
        {
            "name": "Monster Jam",
            "date": "2026-06-12",
            "venue": "AT&T Stadium"
        },
        {
            "name": "Monster Truck Nitro Tour",
            "date": "2026-06-18",
            "venue": "Dallas Arena"
        }
    ],

    "Houston": [
        {
            "name": "Monster Truck Wars",
            "date": "2026-07-02",
            "venue": "NRG Stadium"
        }
    ],

    "Austin": [
        {
            "name": "Texas Monster Bash",
            "date": "2026-08-10",
            "venue": "Austin Speedway"
        }
    ]
}

# COMMAND ----------


@tool
def lookup_event_dates(city: str) -> str:
    """
    Look up monster truck event dates for a given city.
    Uses dummy in-memory data.
    """

    try:
        city = city.strip().title()

        if city not in dummy_events:
            return f"No upcoming monster truck events found in {city}"

        events = dummy_events[city]

        results = []

        for event in events:
            results.append(
                f"{event['name']} — "
                f"{event['date']} at "
                f"{event['venue']}"
            )

        return "\n".join(results)

    except Exception as e:
        return f"API Error: {str(e)}"

# COMMAND ----------


print(lookup_event_dates.invoke("Dallas"))


# COMMAND ----------



@tool
def query_team_standings(team_name: str) -> str:
    """Query the current standings for a given monster truck team."""
    try:
        # FIX 7: Use parameterised query via spark.sql + format to avoid
        #         SQL injection. The original replace("'","") is fragile.
        safe_name = team_name.strip()

        result = spark.sql("""
            SELECT team, wins, losses, points
            FROM monster_truck_standings
            WHERE team = '{name}'
        """.format(name=safe_name.replace("'", "''")))   # double-quote escape

        pdf = result.toPandas()

        if pdf.empty:
            return f"No standings found for team: {safe_name}"

        return pdf.to_string(index=False)

    except Exception as e:
        return f"SQL Error: {str(e)}"

# COMMAND ----------


@tool
def answer_team_questions(question: str) -> str:
    """Answer general questions about the monster truck team using RAG."""
    try:
        docs = vector_store.similarity_search(question, k=3)
        print(docs)
        if not docs:
            return "No relevant information found."

        return "\n\n".join([d.page_content for d in docs])

    except Exception as e:
        return f"Vector Search Error: {str(e)}"

# COMMAND ----------


tools = [
    lookup_event_dates,
    query_team_standings,
    answer_team_questions,
]

agent_executor = create_react_agent(
    model=llm,
    tools=tools,
)


# COMMAND ----------



def get_answer(response: dict) -> str:
    """Extract the final text reply from a LangGraph agent response."""
    messages = response.get("messages", [])
    for msg in reversed(messages):
        # Last AIMessage that is NOT a tool-call is the final answer
        if hasattr(msg, "content") and msg.content:
            return msg.content
    return str(response)

# COMMAND ----------

print("\n===== QUERY 1: Event dates =====")
response1 = agent_executor.invoke({
    "messages": [{"role": "user", "content": "When is the next event in Dallas?"}]
})
print(get_answer(response1))

# COMMAND ----------

response2 = agent_executor.invoke({
    "messages": [{"role": "user", "content": "What is Team Titan's current ranking?"}]
})
print(get_answer(response2))

# COMMAND ----------

# DBTITLE 1,Cell 4

response3 = agent_executor.invoke({
    "messages": [{"role": "user", "content": "Who drives the Grave Digger truck?"}]
})
print(get_answer(response3))