# Databricks notebook source
# MAGIC %md
# MAGIC ### A. Setup environment

# COMMAND ----------

# DBTITLE 1,Install dependencies - MLflow and vector search
# MAGIC %pip install databricks-vectorsearch mlflow
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### B. Define variables and create clients

# COMMAND ----------

# DBTITLE 1,Define VS endpoint, index, models, workspace endpoint, token
import mlflow
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole


# VS endpoint & index
vectorSearchEndpointName = "insurancevectorsearch"
indexName                = "insurance.rag.insuranceindex"

# Model
llmModelName             = "databricks-gpt-oss-120b"

# Databricks endpoint & token
workspaceUrl = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

accessToken = "***ACCESS TOKEN***"

# COMMAND ----------

# DBTITLE 1,Create clients & get reference to index
# Get vector search client
client = VectorSearchClient(
                                workspace_url          =  workspaceUrl,
                                personal_access_token  =  accessToken,
                                disable_notice         =  True
                           )

# Get index reference
index = client.get_index(
                            endpoint_name =  vectorSearchEndpointName,
                            index_name    =  indexName
                        )

# client = VectorSearchClient() 
# # You can use client without parameters when inside Databricks notebook
# # workspace_url and personal_access_token are auto-detected when inside Databricks notebook
# # Parameters must be defined when running outside Databricks notebook or while deploying model

# COMMAND ----------

# MAGIC %md
# MAGIC ### C. Retrieval
# MAGIC
# MAGIC Define methods to retrieve chunks and prepare context

# COMMAND ----------

# DBTITLE 1,Retrieve chunks from vector search
def retrieveChunks (query):

    results = index.similarity_search(
                            query_text  = query,                             # Query text to search

                            query_type  = "HYBRID",                          # FULL_TEXT: For keyword search
                                                                             # ANN: For semantic search (default)
                                                                             # HYBRID: For keyword and semantic search

                            columns     = ["Text", "Source", "PolicyName"],  # Columns to retrieve from index

                            num_results = 3,                                 # top-k retrieval strategy

                            score_threshold = 0.8,                           # Minimum score stategy

                            filters     = {"PolicyName LIKE": "Contoso"}     # Apply SQL-like filters
                        )    
    return results

# COMMAND ----------

# DBTITLE 1,Combine chunks to prepare context
def prepareContext (chunks):

    docs = ([
                "[Source: " + r[1] + "] \n[Content]: " + r[0]          #Format: [Source: <doc path>] chunk

                    for r in chunks["result"]["data_array"]
        ])    

    chunksContext = "\n\n".join(docs)

    return chunksContext

# COMMAND ----------

# DBTITLE 1,Test retrieval step
# User query
input = {"messages": [{"role": "user", "content": "How many days are covered under pre-hospitalization?"}]}

# Extract actual query
query = input["messages"][0]["content"]

# Call vector search and retrieve chunks
chunks = retrieveChunks(query)

# Combine chunks to prepare context
chunksContext = prepareContext(chunks)

# Print chunks
print(chunksContext)

# COMMAND ----------

# MAGIC %md
# MAGIC ### D. Augmentation
# MAGIC
# MAGIC Define methods to combine user query, chunks and system instructions to prepare prompt

# COMMAND ----------

# DBTITLE 1,Define system message and prompt template
# System message
systemMessage = "You are a health insurance assistant answering questions using the provided context."

# Prompt template
PROMPT_TEMPLATE = """

Context:
{context}

Question:
{question}

Instructions:
- Answer using only the context above
- If answer is not contained in the context, say "Information not available"
- Provide citations for sources in [Source - Content] format

"""

# COMMAND ----------

# DBTITLE 1,Build prompt by replacing context and query in template
def buildPrompt(query, chunksContext):    

    prompt = PROMPT_TEMPLATE.format(
                                       context  = chunksContext,
                                       question = query
                                   )

    return prompt


# Test the prompt
prompt = buildPrompt(query, chunks)
print(prompt)

# COMMAND ----------

# MAGIC %md
# MAGIC ### E. Generation
# MAGIC
# MAGIC Pass prompt (with context and query) to LLM model to generate response on your own data

# COMMAND ----------

# DBTITLE 1,Create method to generate LLM response
# Create workspace client
workspace = WorkspaceClient(
                                host   =  workspaceUrl,
                                token  =  accessToken
                           )

# Define method to generate LLM response
def generateAnswer(systemMessage, prompt):

    # Calling pre-deployed endpoint
    response = workspace.serving_endpoints.query(
        
        # Specify model to use
        name = llmModelName,                                            # Using databricks-gpt-oss-120b

        # Specify inputs
        messages=[
                    ChatMessage(
                                role    = ChatMessageRole.SYSTEM,
                                content = systemMessage,
                            ),
                    ChatMessage(
                                role    = ChatMessageRole.USER,
                                content = prompt,
                            )
            ]
    )

    # Extract answer from LLM output
    answer = response.choices[0].message.content[1]["text"]

    return answer


# Test RAG pipeline
answer = generateAnswer(systemMessage, prompt)
print(answer)

# COMMAND ----------

