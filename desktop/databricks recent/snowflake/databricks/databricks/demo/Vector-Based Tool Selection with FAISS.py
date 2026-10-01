# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # 🎯 Vector-Based Tool Selection with FAISS
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook demonstrates **intelligent tool selection** using vector embeddings and similarity search.
# MAGIC
# MAGIC ### The Problem:
# MAGIC
# MAGIC When an agent has many tools available, how does it choose the right one?
# MAGIC
# MAGIC ### The Solution:
# MAGIC
# MAGIC 1. **Embed tool descriptions** into vectors using OpenAI embeddings
# MAGIC 2. **Store in FAISS** (Facebook AI Similarity Search) vector database
# MAGIC 3. **Query with user input** - find most similar tool
# MAGIC 4. **Use LLM to extract parameters** for the selected tool
# MAGIC 5. **Invoke the tool** with correct arguments
# MAGIC
# MAGIC ### Architecture:
# MAGIC
# MAGIC ```
# MAGIC User Query
# MAGIC     ↓
# MAGIC Embed Query → Vector
# MAGIC     ↓
# MAGIC FAISS Search → Find Most Similar Tool
# MAGIC     ↓
# MAGIC LLM → Extract Parameters
# MAGIC     ↓
# MAGIC Invoke Tool → Get Result
# MAGIC ```
# MAGIC
# MAGIC ### Benefits:
# MAGIC
# MAGIC ✅ **Fast** - Vector search is O(log n)  
# MAGIC ✅ **Scalable** - Works with hundreds of tools  
# MAGIC ✅ **Semantic** - Matches meaning, not just keywords  
# MAGIC ✅ **Flexible** - Easy to add new tools

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install langchain langchain-groq langchain-huggingface faiss-cpu numpy requests sentence-transformers -q

# COMMAND ----------

# DBTITLE 1,Import Libraries
import os
import requests
import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# Simple tool decorator to replace langchain's @tool
def tool(func):
    func.name = func.__name__
    func.description = func.__doc__ or ""
    func.invoke = lambda args: func(**args)
    return func

print("✅ All libraries imported successfully")

# COMMAND ----------

# DBTITLE 1,Define Tools
@tool
def query_wolfram_alpha(expression: str) -> str:
    """Use Wolfram Alpha to compute mathematical expressions or retrieve information."""
    # Mock implementation for demo
    # In production: Use Wolfram Alpha API
    
    # Simple equation solver for demo
    if "2x + 3 = 7" in expression:
        return "Solution: x = 2"
    elif "x" in expression and "=" in expression:
        return f"Wolfram Alpha would solve: {expression}"
    else:
        return f"Wolfram Alpha result for: {expression}"

@tool
def trigger_zapier_webhook(zap_id: str, payload: dict) -> str:
    """Trigger a Zapier webhook to execute predefined automated workflows."""
    # Mock implementation
    return f"✅ Triggered Zapier webhook {zap_id} with payload: {payload}"

@tool
def send_slack_message(channel: str, message: str) -> str:
    """Send messages to specific Slack channels to communicate with team members."""
    # Mock implementation
    return f"✅ Sent to Slack {channel}: {message}"

print("✅ Tools defined:")
print(f"   1. {query_wolfram_alpha.name}")
print(f"   2. {trigger_zapier_webhook.name}")
print(f"   3. {send_slack_message.name}")

# COMMAND ----------

# DBTITLE 1,Setup OpenAI
# Set your Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Initialize SentenceTransformer embeddings (free, local)
embeddings_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Create a simple embeddings interface
class Embeddings:
    def __init__(self, model):
        self.model = model
    
    def embed_query(self, text):
        return self.model.encode(text).tolist()

embeddings = Embeddings(embeddings_model)

# Initialize Groq client for parameter extraction (fast inference!)
client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.3-70b-versatile"

print("✅ Groq AI initialized")
print(f"   Embedding model: all-MiniLM-L6-v2 (SentenceTransformer)")
print(f"   LLM model: {MODEL} (Groq)")

# COMMAND ----------

# DBTITLE 1,Create Tool Embeddings and FAISS Index
print("🔧 Creating tool embeddings and FAISS index...\n")

# Tool descriptions
tool_descriptions = {
    "query_wolfram_alpha": "Use Wolfram Alpha to compute mathematical expressions or retrieve information.",
    "trigger_zapier_webhook": "Trigger a Zapier webhook to execute predefined automated workflows.",
    "send_slack_message": "Send messages to specific Slack channels to communicate with team members."
}

# Create embeddings for each tool description
tool_embeddings = []
tool_names = []

for tool_name, description in tool_descriptions.items():
    print(f"  Embedding: {tool_name}")
    embedding = embeddings.embed_query(description)
    tool_embeddings.append(embedding)
    tool_names.append(tool_name)

# Initialize FAISS vector store
dimension = len(tool_embeddings[0])
index = faiss.IndexFlatL2(dimension)

print(f"\n✅ FAISS index created (dimension: {dimension})")

# Convert to FAISS-compatible format and normalize for cosine similarity
tool_embeddings_np = np.array(tool_embeddings).astype('float32')
faiss.normalize_L2(tool_embeddings_np)

# Add to index
index.add(tool_embeddings_np)

print(f"✅ Added {len(tool_embeddings)} tools to FAISS index")

# Map index to tool functions
index_to_tool = {
    0: query_wolfram_alpha,
    1: trigger_zapier_webhook,
    2: send_slack_message
}

print("\n📊 Index Mapping:")
for idx, tool_name in enumerate(tool_names):
    print(f"   {idx} → {tool_name}")

# COMMAND ----------

# DBTITLE 1,Tool Selection Function
def select_tool(query: str, top_k: int = 1) -> list:
    """
    Select the most relevant tool(s) based on the user's query using
    vector-based retrieval.
    
    Args:
        query (str): The user's input query.
        top_k (int): Number of top tools to retrieve.
    
    Returns:
        list: List of selected tool functions.
    """
    # Embed the query
    query_embedding = embeddings.embed_query(query)
    query_embedding_np = np.array([query_embedding]).astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding_np)
    
    # Search FAISS index
    distances, indices = index.search(query_embedding_np, top_k)
    
    # Get selected tools
    selected_tools = []
    for i, idx in enumerate(indices[0]):
        if idx in index_to_tool:
            tool_func = index_to_tool[idx]
            tool_name = tool_names[idx]
            distance = distances[0][i]
            selected_tools.append((tool_func, tool_name, distance))
    
    return selected_tools

print("✅ Tool selection function defined")
print("   Uses: Vector similarity search with FAISS")
print("   Returns: Most relevant tool(s) for user query")

# COMMAND ----------

# DBTITLE 1,Parameter Determination Function
def determine_parameters(query: str, tool_name: str) -> dict:
    """
    Use the LLM to analyze the query and determine the parameters for the tool
    to be invoked.
    
    Args:
        query (str): The user's input query.
        tool_name (str): The selected tool name.
    
    Returns:
        dict: Parameters for the tool.
    """
    # Build prompt for LLM
    prompt = f"""Based on the user's query: '{query}', extract the parameters needed for the tool '{tool_name}'.
    
    Return ONLY a JSON object with the parameters. No other text.
    
    For query_wolfram_alpha: {{'expression': '<math_expression>'}}
    For trigger_zapier_webhook: {{'zap_id': '<id>', 'payload': {{'data': '<value>'}}}}
    For send_slack_message: {{'channel': '<channel_name>', 'message': '<message_text>'}}
    """
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    # Parse response
    import json
    try:
        # Extract JSON from response
        content = response.choices[0].message.content
        # Clean up if wrapped in markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        parameters = json.loads(content)
        return parameters
    except Exception as e:
        print(f"⚠️ Error parsing LLM response: {e}")
        # Fallback defaults
        if tool_name == "query_wolfram_alpha":
            return {"expression": query}
        elif tool_name == "trigger_zapier_webhook":
            return {"zap_id": "123456", "payload": {"data": query}}
        elif tool_name == "send_slack_message":
            return {"channel": "#general", "message": query}
        return {}

print("✅ Parameter determination function defined")
print("   Uses: LLM (GPT-4) to extract parameters from user query")
print("   Returns: Structured parameters for tool invocation")

# COMMAND ----------

# DBTITLE 1,Execute Example Queries
# Test queries for different tools
test_queries = [
    "Solve this equation: 2x + 3 = 7",
    "Send a message to the team about today's meeting",
    "Automate my weekly report workflow"
]

print("🧪 TESTING VECTOR-BASED TOOL SELECTION\n")
print("="*70)

for i, user_query in enumerate(test_queries, 1):
    print(f"\n\n{'='*70}")
    print(f"Test {i}: {user_query}")
    print("="*70)
    
    # Step 1: Select the top tool
    print("\n🔍 Step 1: Vector search for best tool...")
    selected_tools = select_tool(user_query, top_k=1)
    
    if selected_tools:
        tool_func, tool_name, distance = selected_tools[0]
        print(f"   ✅ Selected: {tool_name}")
        print(f"   📊 Similarity score: {1 / (1 + distance):.4f}")
        
        # Step 2: Use LLM to determine the parameters
        print("\n🤖 Step 2: LLM extracting parameters...")
        args = determine_parameters(user_query, tool_name)
        print(f"   ✅ Parameters: {args}")
        
        # Step 3: Invoke the selected tool
        print("\n🔧 Step 3: Invoking tool...")
        try:
            tool_result = tool_func.invoke(args)
            print(f"\n✅ RESULT: {tool_result}")
        except Exception as e:
            print(f"\n❌ Error invoking tool: {e}")
    else:
        print("\n❌ No tool was selected.")

print("\n\n" + "="*70)
print("✅ All tests complete!")
print("="*70)

# COMMAND ----------

# DBTITLE 1,Summary and Key Concepts
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Summary
# MAGIC
# MAGIC ### What We Built:
# MAGIC
# MAGIC A **semantic tool selection system** that uses vector embeddings to intelligently choose the right tool based on natural language queries.
# MAGIC
# MAGIC ### How It Works:
# MAGIC
# MAGIC 1. **Tool Descriptions → Vectors**
# MAGIC    - Each tool has a text description
# MAGIC    - OpenAI embeddings convert text to 1536-dimensional vectors
# MAGIC    - Vectors capture semantic meaning
# MAGIC
# MAGIC 2. **FAISS Vector Store**
# MAGIC    - Fast similarity search (Facebook AI library)
# MAGIC    - Uses L2 distance with normalization (cosine similarity)
# MAGIC    - O(log n) search time - scales to thousands of tools
# MAGIC
# MAGIC 3. **Query Processing**
# MAGIC    - User query is embedded the same way
# MAGIC    - FAISS finds nearest vector (most similar tool)
# MAGIC    - Returns top-k matches
# MAGIC
# MAGIC 4. **Parameter Extraction**
# MAGIC    - LLM (GPT-4) reads the query and tool name
# MAGIC    - Extracts structured parameters
# MAGIC    - Returns JSON for tool invocation
# MAGIC
# MAGIC 5. **Tool Invocation**
# MAGIC    - Call the selected tool with extracted parameters
# MAGIC    - Return result to user
# MAGIC
# MAGIC ### Key Benefits:
# MAGIC
# MAGIC | Feature | Benefit |
# MAGIC |---------|----------|
# MAGIC | **Semantic Search** | Matches meaning, not keywords |
# MAGIC | **Scalable** | Works with 100s or 1000s of tools |
# MAGIC | **Fast** | Vector search is O(log n) |
# MAGIC | **Flexible** | Easy to add new tools |
# MAGIC | **LLM-Powered** | Intelligent parameter extraction |
# MAGIC
# MAGIC ### Production Enhancements:
# MAGIC
# MAGIC 1. **Add more tools** - System scales linearly
# MAGIC 2. **Use real APIs** - Replace mock implementations
# MAGIC 3. **Error handling** - Retry logic, fallbacks
# MAGIC 4. **Monitoring** - Log tool usage and performance
# MAGIC 5. **Caching** - Cache embeddings to save API calls
# MAGIC 6. **Multi-tool** - Execute multiple tools in sequence
# MAGIC 7. **Confidence threshold** - Only use tool if similarity > threshold
# MAGIC
# MAGIC ### Alternative Approaches:
# MAGIC
# MAGIC **Rule-Based:**
# MAGIC ```python
# MAGIC if "calculate" in query or "solve" in query:
# MAGIC     use_wolfram()
# MAGIC ```
# MAGIC ❌ Brittle, doesn't scale
# MAGIC
# MAGIC **LLM Direct:**
# MAGIC ```python
# MAGIC llm.invoke(f"Which tool to use: {query}?")
# MAGIC ```
# MAGIC ❌ Slow, expensive for every query
# MAGIC
# MAGIC **Vector Search (This Approach):**
# MAGIC ```python
# MAGIC faiss.search(embed(query))
# MAGIC ```
# MAGIC ✅ Fast, semantic, scalable
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 Next Steps
# MAGIC
# MAGIC 1. **Add real API keys** for Wolfram Alpha, Zapier, Slack
# MAGIC 2. **Expand tool catalog** with 10-100 more tools
# MAGIC 3. **Add confidence scoring** - only invoke if match is good
# MAGIC 4. **Implement caching** - store embeddings in persistent storage
# MAGIC 5. **Add logging** - track which tools are used most
# MAGIC 6. **Build agent loop** - chain multiple tool calls together
# MAGIC
# MAGIC This pattern is used in production by **LangChain**, **AutoGPT**, and other agent frameworks! 🎉

# COMMAND ----------

