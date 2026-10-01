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

# DBTITLE 1,Define VS endpoint, index, models, workspace endpoint, token, model parameters
import mlflow
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
import pandas as pd

# VS endpoint & index
vectorSearchEndpointName = "insurancevectorsearch"
indexName                = "insurance.rag.insuranceindex"

# Models
llmModelName             = "databricks-gpt-oss-120b"

# Databricks endpoint & token
workspaceUrl = "***WORKSPACE URL***"   # Example: https://adb-************.***.azuredatabricks.net/

accessToken = "***ACCESS TOKEN***"

# Model parameters
numOfResults   = 3
scoreThreshold = 0.8

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

# Get workspace client
workspace = WorkspaceClient(
                                host   =  workspaceUrl,
                                token  =  accessToken
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

                            num_results = numOfResults,                      # top-k retrieval strategy

                            score_threshold = scoreThreshold,                # Minimum score stategy

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

# MAGIC %md
# MAGIC ### D. Augmentation
# MAGIC
# MAGIC Define methods to combine user query, chunks and system instructions to prepare prompt

# COMMAND ----------

# DBTITLE 1,Define system message and prompt template
systemMessage = "You are a health insurance assistant answering questions using the provided context."

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### E. Generation
# MAGIC
# MAGIC Pass prompt (with context and query) to LLM model to generate response on your own data

# COMMAND ----------

# DBTITLE 1,Generate LLM response
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

# COMMAND ----------

# MAGIC %md
# MAGIC ### F. Define complete RAG pipeline

# COMMAND ----------

# DBTITLE 1,Define method for calling RAG pipeline
def ragPipeline(query):

    chunks = retrieveChunks(query)

    chunksContext = prepareContext(chunks)

    prompt = buildPrompt(query, chunksContext)

    answer = generateAnswer(systemMessage, prompt)

    answer = {"predictions": [{"content": answer}]}

    return answer

# COMMAND ----------

# MAGIC %md
# MAGIC ### G. Register RAG pipeline as a Model

# COMMAND ----------

# DBTITLE 1,Define Python Model class for RAG
# Class inherited from 'PythonModel'
class InsuranceRagModel(mlflow.pyfunc.PythonModel):

    # Define predict method
    def predict(self, context, model_input):

        # When deployed as a model, input is received as a pandas dataframe
        if isinstance(model_input, pd.DataFrame):
            query = model_input["messages"].iloc[0][0]["content"]            
        else:
            query = model_input["messages"][0]["content"]

        # Call RAG pipeline
        answer = ragPipeline(query)                

        # Return response
        return answer
    

# COMMAND ----------

# DBTITLE 1,Set model as current/active
mlflow.models.set_model( model = InsuranceRagModel() )