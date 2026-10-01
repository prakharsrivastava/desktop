# Databricks notebook source
# ============================================================
# Large Language Model (LLM) Configuration Example
# ============================================================

# Different AI models have different strengths:
# - Context window = how much text the model can remember at once
# - Open source = whether you can run it yourself
# - Parameters = rough size/power of the model

# ------------------------------------------------------------
# Example Models
# ------------------------------------------------------------

models = [
    {
        "name": "MPT-30B",
        "company": "MosaicML",
        "parameters": "30 Billion",
        "context_window": 8192,
        "open_source": True,
        "best_for": [
            "Long documents",
            "Chatbots",
            "Research",
            "Code generation"
        ]
    },

    {
        "name": "Llama2-70B",
        "company": "Meta",
        "parameters": "70 Billion",
        "context_window": 4096,
        "open_source": True,
        "best_for": [
            "Reasoning",
            "Chat applications",
            "General AI tasks"
        ]
    },

    {
        "name": "DistilBERT",
        "company": "Hugging Face",
        "parameters": "66 Million",
        "context_window": 512,
        "open_source": True,
        "best_for": [
            "Fast inference",
            "Text classification",
            "Low-memory environments"
        ]
    },

    {
        "name": "DBRX",
        "company": "Databricks",
        "parameters": "132 Billion (MoE)",
        "context_window": 32768,
        "open_source": True,
        "best_for": [
            "Enterprise AI",
            "Long context",
            "High-performance inference"
        ]
    },

    {
        "name": "Mixtral-8x7B",
        "company": "Mistral AI",
        "parameters": "46.7 Billion (MoE)",
        "context_window": 32768,
        "open_source": True,
        "best_for": [
            "Efficient inference",
            "Coding",
            "Reasoning"
        ]
    },

    {
        "name": "Gemma-7B",
        "company": "Google",
        "parameters": "7 Billion",
        "context_window": 8192,
        "open_source": True,
        "best_for": [
            "Lightweight deployment",
            "Research",
            "Local AI"
        ]
    }
]

# ------------------------------------------------------------
# Print all model information
# ------------------------------------------------------------

for model in models:
    print("=" * 60)
    print(f"Model Name      : {model['name']}")
    print(f"Company         : {model['company']}")
    print(f"Parameters      : {model['parameters']}")
    print(f"Context Window  : {model['context_window']} tokens")
    print(f"Open Source     : {model['open_source']}")
    print("Best For        :")

    for use_case in model["best_for"]:
        print(f"  - {use_case}")

# ------------------------------------------------------------
# Example: Selecting a model based on requirement
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("MODEL SELECTION EXAMPLE")
print("=" * 60)

required_context = 8000

for model in models:
    if model["context_window"] >= required_context:
        print(
            f"{model['name']} supports "
            f"{model['context_window']} tokens "
            f"(Suitable)"
        )
    else:
        print(
            f"{model['name']} supports "
            f"{model['context_window']} tokens "
            f"(Too Small)"
        )

# ------------------------------------------------------------
# Example Output Logic
# ------------------------------------------------------------

selected_model = None

for model in models:
    if (
        model["open_source"] and
        model["context_window"] >= 8000
    ):
        selected_model = model
        break

print("\n")
print("=" * 60)
print("FINAL SELECTED MODEL")
print("=" * 60)

if selected_model:
    print(f"Name            : {selected_model['name']}")
    print(f"Company         : {selected_model['company']}")
    print(f"Context Window  : {selected_model['context_window']}")
else:
    print("No suitable model found.")

# COMMAND ----------

# DBTITLE 1,Cell 2
# ============================================================
# Goal:
# Generate a poetic / haiku-style summary
# ============================================================

document = """
The city experienced heavy rainfall yesterday.
People rushed through crowded streets while trains were delayed.
Many citizens described the atmosphere as gloomy but calming.
"""

# ============================================================
# ✅ METHODS THAT HELP OUTPUT STYLE
# ============================================================

# 1. Explicit prompt instruction
prompt = f"""
Write a poetic haiku-style summary
with emotional and reflective language.

Document:
{document}
"""

# 2. Few-shot prompting
few_shot_prompt = """
Example Input:
A forest burned during summer drought.

Example Output:
Ash drifts through the pines,
Silent winds mourn fading green,
Summer weeps alone.

Now summarize this document poetically:
"""

# 3. Fine-tuning on poetry dataset
# (Conceptual example)
model_config = {
    "fine_tuned_for": "poetic_summaries"
}

# ============================================================
# ❌ METHOD THAT DOES NOT HELP
# ============================================================

def neutralizer(doc):
    """
    Removes emotional tone from INPUT text.
    This does NOT teach the model to write poetically.
    """
    # Conceptual demonstration - function not implemented
    return doc

neutral_doc = neutralizer(document)

print("Neutralized Input:")
print(neutral_doc)

# Problem:
# The LLM still needs style instructions.
# Neutralizing input only strips emotion/context.

# COMMAND ----------

# MAGIC %pip install langchain-text-splitters
# MAGIC

# COMMAND ----------

# MAGIC %pip install langchain

# COMMAND ----------

# DBTITLE 1,Cell 5
from langchain_text_splitters import RecursiveCharacterTextSplitter



docs = [
    """
    Account Setup
    Users can create an account using email verification.

    Password Reset
    Password reset links expire after 24 hours.

    Billing Support
    Refunds are processed within 5-7 business days.
    """
]



splitter = RecursiveCharacterTextSplitter(
    chunk_size=5,      # Smaller chunk size
    chunk_overlap=2
)

chunks = splitter.create_documents(docs)

# COMMAND ----------

chunks

# COMMAND ----------

def extract_header(text):
    """
    Simple example:
    Extract first line as section header
    """
    return text.strip().split("\n")[0]

# COMMAND ----------

for i, chunk in enumerate(chunks):
    print(chunk)
    chunk.metadata["section"] = (
        f"Section {i+1}: {extract_header(chunk.page_content)}"
    )

# COMMAND ----------

for chunk in chunks:
    print("=" * 60)
    print("SECTION:", chunk.metadata["section"])
    print(chunk.page_content)

# COMMAND ----------

prompt = """
You are a restaurant booking assistant.

Given this customer chat:
{chat_log}

Return JSON button options the user can click.
"""

response_format = {
    "options": [
        "Book a table",
        "Check availability",
        "Modify booking",
        "Cancel reservation"
    ]
}

print(response_format)

# COMMAND ----------

# MAGIC %pip install unstructured[pdf]

# COMMAND ----------

# MAGIC %pip install -U unstructured
# MAGIC %pip install -U pdfminer.six

# COMMAND ----------

# DBTITLE 1,Cell 13
# Install PyPDF2 - lightweight PDF reader compatible with serverless
%pip install PyPDF2

import PyPDF2

# Read and extract text from PDF
pdf_path = "/Workspace/Users/prakhar1207srivastava@gmail.com/Drafts/InstructionsCreatePDFofE-VerifyManual.pdf"

with open(pdf_path, 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    
    # Extract text from all pages
    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        print(f"--- Page {page_num + 1} ---")
        print(text)
        print()

# COMMAND ----------

import re

# ------------------------------------------------------------
# Mock token counter
# ------------------------------------------------------------

def count_tokens(text):
    return len(text.split())

# ------------------------------------------------------------
# Split oversized sections further
# ------------------------------------------------------------

def split_further(section, max_tokens):

    words = section.split()

    chunks = []

    for i in range(0, len(words), max_tokens):

        chunk = " ".join(
            words[i:i + max_tokens]
        )

        chunks.append(chunk)

    return chunks

# ------------------------------------------------------------
# Main legal chunking strategy
# ------------------------------------------------------------

def chunk_legal_text(text, max_tokens=1):

    # Split by logical legal headings
    sections = re.split(
        r'\n(?=Section \d+|Article \d+|CHAPTER)',
        text
    )

    chunks = []

    for section in sections:

        # Keep full logical section if small enough
        if count_tokens(section) <= max_tokens:

            chunks.append(section)

        else:
            # Split large sections carefully
            chunks.extend(
                split_further(section, max_tokens)
            )

    return chunks

# ------------------------------------------------------------
# Example legal text
# ------------------------------------------------------------

legal_text = """
CHAPTER 1

Section 1
This agreement defines the legal obligations of both parties.

Section 2
The user must comply with all applicable regulations.

Article 3
Violation of terms may result in termination.
"""

# ------------------------------------------------------------
# Run chunking
# ------------------------------------------------------------

chunks = chunk_legal_text(legal_text)

for i, chunk in enumerate(chunks):

    print("=" * 60)
    print(f"Chunk {i+1}")
    print(chunk)

# COMMAND ----------

# ============================================================
# Customer Support Chatbot Evaluation
# ============================================================

def check_relevance(prompt, user_query):

    """
    Mock relevance check:
    Does the prompt align with user intent?
    """

    return user_query.lower() in prompt.lower()

# ------------------------------------------------------------

def check_completeness(response, user_query):

    """
    Mock completeness check:
    Does response address all parts of query?
    """

    required_keywords = [
        "refund",
        "timeline",
        "support"
    ]

    covered = 0

    for word in required_keywords:

        if word.lower() in response.lower():
            covered += 1

    return covered / len(required_keywords)

# ------------------------------------------------------------

def evaluate_chatbot_response(
    prompt,
    response,
    user_query
):

    scores = {}

    # ✅ Prompt relevance
    scores["relevance"] = check_relevance(
        prompt,
        user_query
    )

    # ✅ Response completeness
    scores["completeness"] = check_completeness(
        response,
        user_query
    )

    # ❌ NOT token efficiency
    # ❌ NOT verbosity
    # ❌ NOT random sampling

    return scores

# ============================================================
# Example
# ============================================================

user_query = """
I want a refund and need to know
how long it takes.
"""

prompt = """
Customer is asking about refund process
and refund timeline.
"""

response = """
Refund requests are processed by support.
The refund timeline is usually 5-7 business days.
Please contact support for help.
"""

results = evaluate_chatbot_response(
    prompt,
    response,
    user_query
)

print(results)

# COMMAND ----------

# DBTITLE 1,Install packages
# MAGIC %pip install langgraph langchain-community chromadb sentence-transformers
# MAGIC dbutils.library.restartPython()

# COMMAND ----------


"""
PROJECT OBJECTIVE
-----------------
Build a graph-based RAG pipeline from COBOL code
to modernize legacy systems into Python/Java.

Generate:
- User stories
- Architecture documents
- Dependency analysis
- Technical artifacts

Track everything using MLflow:
- Parameters
- Retrieval metrics
- Generated outputs
- Execution performance
"""

# ============================================================
# IMPORTS
# ============================================================

from typing import TypedDict
import time

import mlflow

from langgraph.graph import StateGraph, END

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.documents import Document

# ============================================================
# SAMPLE COBOL CODE
# ============================================================

# COMMAND ----------

cobol_code = """
IDENTIFICATION DIVISION.
PROGRAM-ID. CUSTOMER-UPDATE.

DATA DIVISION.

WORKING-STORAGE SECTION.

01 CUSTOMER-BALANCE PIC 9(5).

PROCEDURE DIVISION.

READ CUSTOMER-FILE.

IF CUSTOMER-BALANCE > 1000
    PERFORM APPLY-DISCOUNT.

WRITE CUSTOMER-REPORT.
"""

# COMMAND ----------


class GraphState(TypedDict):

    cobol_code: str

    chunks: list

    retrieved_docs: list

    generated_output: str


# COMMAND ----------

def chunk_cobol_code(state):

    start = time.time()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    docs = [
        Document(page_content=state["cobol_code"])
    ]

    chunks = splitter.split_documents(docs)

    # MLflow Tracking
    mlflow.log_param("chunk_size", 300)
    mlflow.log_param("chunk_overlap", 50)

    mlflow.log_metric(
        "num_chunks",
        len(chunks)
    )

    mlflow.log_metric(
        "chunking_time_sec",
        round(time.time() - start, 2)
    )

    print("\nCHUNKS CREATED")
    print("=" * 60)

    for chunk in chunks:
        print(chunk.page_content)

    return {
        "chunks": chunks
    }

# COMMAND ----------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = None

def build_vector_store(state):

    global vector_db

    start = time.time()

    vector_db = Chroma.from_documents(
        documents=state["chunks"],
        embedding=embedding_model
    )

    mlflow.log_param(
        "embedding_model",
        "all-MiniLM-L6-v2"
    )

    mlflow.log_metric(
        "vector_db_build_time_sec",
        round(time.time() - start, 2)
    )

    print("\nVECTOR STORE CREATED")
    print("=" * 60)

    return state

# COMMAND ----------


def retrieve_context(state):

    start = time.time()

    query = """
    customer discount business rules
    """

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 2}
    )

    docs = retriever.invoke(query)
    print(docs)
    retrieval_count = len(docs)

    # Mock retrieval score
    retrieval_precision = 0.92

    mlflow.log_param("retrieval_top_k", 2)

    mlflow.log_metric(
        "retrieved_documents",
        retrieval_count
    )

    mlflow.log_metric(
        "retrieval_precision",
        retrieval_precision
    )

    mlflow.log_metric(
        "retrieval_time_sec",
        round(time.time() - start, 2)
    )

    print("\nRETRIEVED DOCUMENTS")
    print("=" * 60)

    for doc in docs:
        print(doc.page_content)

    return {
        "retrieved_docs": docs
    }

# COMMAND ----------


def generate_user_story(state):

    start = time.time()

    context = "\n".join([
        doc.page_content
        for doc in state["retrieved_docs"]
    ])

    generated_story = f"""
USER STORY
===========

As a banking customer,
I want discount processing to happen automatically
when account balance exceeds threshold,
so that loyalty benefits are applied correctly.

SOURCE CONTEXT
===============
{context}

MODERNIZATION TARGET
=====================
- Python FastAPI microservice
- Java Spring Boot service
"""

    # Mock evaluation metrics
    factual_accuracy = 0.95
    completeness = 0.91

    mlflow.log_metric(
        "factual_accuracy",
        factual_accuracy
    )

    mlflow.log_metric(
        "response_completeness",
        completeness
    )

    mlflow.log_metric(
        "generation_time_sec",
        round(time.time() - start, 2)
    )

    # Save generated artifact
    with open("generated_story.txt", "w") as f:
        f.write(generated_story)

    mlflow.log_artifact(
        "generated_story.txt"
    )

    print("\nGENERATED USER STORY")
    print("=" * 60)

    print(generated_story)

    return {
        "generated_output": generated_story
    }

# COMMAND ----------


workflow = StateGraph(GraphState)

workflow.add_node(
    "chunk_cobol",
    chunk_cobol_code
)

workflow.add_node(
    "build_vector_store",
     build_vector_store
)

workflow.add_node(
    "retrieve_context",
    retrieve_context
)

workflow.add_node(
    "generate_story",
    generate_user_story
)

workflow.set_entry_point(
    "chunk_cobol"
)

workflow.add_edge(
    "chunk_cobol",
    "build_vector_store"
)

workflow.add_edge(
    "build_vector_store",
    "retrieve_context"
)

workflow.add_edge(
    "retrieve_context",
    "generate_story"
)

workflow.add_edge(
    "generate_story",
    END
)


app = workflow.compile()


# COMMAND ----------

mlflow.set_experiment(
    "/Users/prakhar1207srivastava@gmail.com/COBOL_Modernization_RAG"
)

# COMMAND ----------


with mlflow.start_run():

    mlflow.log_param(
        "project_type",
        "Graph-Based RAG"
    )

    mlflow.log_param(
        "source_language",
        "COBOL"
    )

    mlflow.log_param(
        "target_languages",
        "Python, Java"
    )

    result = app.invoke({

        "cobol_code": cobol_code

    })

    print("\nFINAL OUTPUT")
    print("=" * 60)

    print(result["generated_output"])

# COMMAND ----------

# DBTITLE 1,Cell 16


# ============================================================
# WHAT MLFLOW TRACKS
# ============================================================

"""
MLflow Tracks:
---------------
1. Parameters
   - chunk_size
   - overlap
   - embedding model
   - retrieval top-k

2. Metrics
   - retrieval precision
   - factual accuracy
   - completeness
   - execution time

3. Artifacts
   - generated user stories
   - architecture docs
   - migration outputs

4. Experiments
   - compare chunking strategies
   - compare embedding models
   - compare retrieval quality
"""

# COMMAND ----------

# ============================================================
# MULTI-AGENT LLM SYSTEM
# ============================================================

"""
Q3. What LLM model are you using and why multiple models?

Answer:
--------
We use a multi-agent architecture.

- Nova Pro:
    Main orchestration agent for complex reasoning,
    workflow coordination, RAG analysis,
    architecture generation, and modernization tasks.

- Claude Haiku:
    Lightweight agents for simpler operations like:
    authentication,
    summarization,
    validation,
    routing,
    quick responses.

Model selection depends on:
- task complexity
- latency requirements
- cost optimization
- reasoning depth
"""

# ============================================================
# MOCK LLM CLIENTS
# ============================================================

class NovaProLLM:

    def invoke(self, task):

        return f"""
[NOVA PRO]

Complex reasoning completed for:
{task}

Generated:
- architecture analysis
- dependency mapping
- modernization workflow
"""

# ------------------------------------------------------------

class ClaudeHaikuLLM:

    def invoke(self, task):

        return f"""
[CLAUDE HAIKU]

Fast lightweight response for:
{task}
"""

# ============================================================
# MULTI-AGENT ROUTER
# ============================================================

class ModelRouter:

    def __init__(self):

        self.nova_pro = NovaProLLM()

        self.claude_haiku = ClaudeHaikuLLM()

    # --------------------------------------------------------

    def route_task(self, task_type, task):

        """
        Select model based on task complexity.
        """

        # Complex reasoning tasks
        if task_type in [

            "rag_analysis",
            "architecture_generation",
            "dependency_analysis",
            "cobol_modernization",
            "workflow_orchestration"

        ]:

            print("\nUsing NOVA PRO")
            print("=" * 60)

            return self.nova_pro.invoke(task)

        # Lightweight fast tasks
        elif task_type in [

            "authentication",
            "session_validation",
            "quick_summary",
            "classification",
            "routing"

        ]:

            print("\nUsing CLAUDE HAIKU")
            print("=" * 60)

            return self.claude_haiku.invoke(task)

        else:

            return "No suitable model found"

# ============================================================
# INITIALIZE ROUTER
# ============================================================

router = ModelRouter()

# ============================================================
# EXAMPLE 1 — COMPLEX TASK
# ============================================================

complex_result = router.route_task(

    task_type="cobol_modernization",

    task="""
    Analyze COBOL dependency graph and
    generate modernization plan for Java migration
    """

)

print(complex_result)

# ============================================================
# EXAMPLE 2 — SIMPLE TASK
# ============================================================

simple_result = router.route_task(

    task_type="authentication",

    task="""
    Validate user JWT token
    """

)

print(simple_result)

# ============================================================
# WHY MULTIPLE MODELS?
# ============================================================

"""
Benefits of Multi-Model Architecture
------------------------------------

1. Cost Optimization
--------------------
Use expensive models only when needed.

2. Lower Latency
----------------
Small models respond faster.

3. Better Scalability
---------------------
Simple tasks avoid overloading large models.

4. Specialized Agents
---------------------
Different agents optimized for different tasks.

5. Improved Performance
-----------------------
Complex reasoning handled by stronger LLMs.
"""

# ============================================================
# REAL-WORLD FLOW
# ============================================================

"""
USER REQUEST
      ↓

Task Classifier Agent
      ↓

+-------------------------------+
| COMPLEX TASK ?                |
+-------------------------------+

YES ----------------> NOVA PRO
                      - reasoning
                      - RAG
                      - architecture
                      - orchestration

NO -----------------> CLAUDE HAIKU
                      - auth
                      - validation
                      - summaries
                      - routing
"""

# ============================================================
# OTHER ALTERNATIVE MODELS
# ============================================================

alternative_models = {

    "Complex Reasoning Models": [

        "GPT-4.1",
        "Claude Sonnet",
        "Claude Opus",
        "Gemini 1.5 Pro",
        "Llama 3 70B",
        "Mistral Large"

    ],

    "Fast Lightweight Models": [

        "Claude Haiku",
        "GPT-4o-mini",
        "Gemini Flash",
        "Llama 3 8B",
        "Mistral Small",
        "Phi-3 Mini"

    ]
}

print("\n")
print("=" * 60)
print("ALTERNATIVE MODELS")
print("=" * 60)

for category, models in alternative_models.items():

    print(f"\n{category}")

    for model in models:

        print(f" - {model}")

# ============================================================
# INTERVIEW SUMMARY
# ============================================================

"""
Interview Answer (Short Version)
--------------------------------

We use a multi-agent architecture.

Nova Pro handles:
- orchestration
- RAG reasoning
- architecture generation
- COBOL modernization

Claude Haiku handles:
- authentication
- lightweight agents
- fast validation tasks

We select models dynamically based on:
- task complexity
- latency
- cost
- reasoning requirements
"""

# COMMAND ----------

# DBTITLE 1,Install sentence-transformers
# MAGIC %pip install sentence-transformers

# COMMAND ----------

# ============================================================
# HALLUCINATION REDUCTION TECHNIQUES
# ============================================================

"""
Q7. How do you tackle hallucination?

Answer:
--------
1. Ground responses using RAG
2. Use groundedness similarity checks
3. Reduce temperature for deterministic output
4. Use LLM-as-a-judge evaluation
5. Retry generation when hallucination detected
"""

from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

import numpy as np



# COMMAND ----------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# COMMAND ----------


user_query = """
What happens when customer balance exceeds 1000?
"""

retrieved_chunk = """
If CUSTOMER-BALANCE > 1000
PERFORM APPLY-DISCOUNT
"""

generated_answer = """
The system applies a discount when
customer balance exceeds 1000.
"""

# COMMAND ----------


def groundedness_check(context, answer):

    context_embedding = embedding_model.encode([context])

    answer_embedding = embedding_model.encode([answer])

    similarity = cosine_similarity(

        context_embedding,
        answer_embedding

    )[0][0]

    print("\n")
    print("=" * 60)
    print("GROUNDEDNESS CHECK")
    print("=" * 60)

    print(f"Cosine Similarity: {similarity:.2f}")

    threshold = 0.70

    if similarity < threshold:

        print("\nPotential hallucination detected")

        return False

    print("\nAnswer grounded in retrieved context")

    return True

# COMMAND ----------


class MockLLM:

    def __init__(self, temperature):

        self.temperature = temperature

    def invoke(self, query):

        return f"""
LLM Response generated with
temperature = {self.temperature}
"""


# COMMAND ----------


# Lower temperature → less creativity → less hallucination
llm = MockLLM(

    temperature=0.0

)

print("\n")
print("=" * 60)
print("LOW TEMPERATURE SETTING")
print("=" * 60)

print(llm.invoke(user_query))

# COMMAND ----------


class JudgeLLM:

    def evaluate(

        self,
        question,
        context,
        answer

    ):

        print("\n")
        print("=" * 60)
        print("LLM AS JUDGE")
        print("=" * 60)

        print(f"Question: {question}")

        print(f"\nContext:\n{context}")

        print(f"\nAnswer:\n{answer}")

        # Mock evaluation logic
        hallucination_detected = False

        if "discount" not in context.lower():

            hallucination_detected = True

        if hallucination_detected:

            return {

                "status": "FAILED",

                "reason": "Answer not supported by context"

            }

        return {

            "status": "PASSED",

            "reason": "Answer grounded correctly"

        }

judge_llm = JudgeLLM()

judge_result = judge_llm.evaluate(

    question=user_query,

    context=retrieved_chunk,

    answer=generated_answer

)

print("\nJudge Result:")
print(judge_result)


# COMMAND ----------

def rag_pipeline(query):

    """
    Retrieval-Augmented Generation

    Hallucination reduced because
    answer is generated using
    retrieved enterprise documents.
    """

    retrieved_docs = [

        """
        CUSTOMER BALANCE > 1000
        APPLY DISCOUNT
        """

    ]

    context = "\n".join(retrieved_docs)

    answer = f"""
Based on enterprise documents:

{context}

Generated Answer:
Customer receives discount when balance exceeds 1000.
"""

    return answer

print("\n")
print("=" * 60)
print("RAG GROUNDED RESPONSE")
print("=" * 60)

print(rag_pipeline(user_query))

# COMMAND ----------


# ============================================================
# RETRY MECHANISM
# ============================================================

def generate_with_retry(context, answer):

    is_grounded = groundedness_check(

        context,
        answer

    )

    if not is_grounded:

        print("\nRetrying generation...")

        regenerated_answer = """
        Based on retrieved records,
        discount is applied only when
        balance exceeds 1000.
        """

        return regenerated_answer

    return answer

final_answer = generate_with_retry(

    retrieved_chunk,
    generated_answer

)

print("\n")
print("=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(final_answer)

# ============================================================
# BEST PRACTICES
# ============================================================

"""
BEST PRACTICES TO REDUCE HALLUCINATION
--------------------------------------

1. Use RAG
-----------
Ground answers in enterprise documents.

2. Lower Temperature
---------------------
temperature = 0.0

3. Similarity Validation
-------------------------
Check generated answer against retrieved chunks.

4. LLM-as-a-Judge
------------------
Secondary validation model verifies consistency.

5. Prompt Constraints
----------------------
Tell model:
"Answer only from provided context."

6. Confidence Scoring
----------------------
Reject low-confidence responses.

7. Human-in-the-Loop
---------------------
Critical workflows require human approval.
"""

# ============================================================
# INTERVIEW SHORT ANSWER
# ============================================================

"""
Interview Answer
----------------

We reduce hallucination using multiple strategies:

- RAG grounding using enterprise documents
- groundedness similarity checks
- low temperature settings for deterministic output
- LLM-as-a-judge validation
- retry mechanisms when inconsistencies are detected

This ensures generated responses remain factual,
context-aware, and aligned with retrieved data.
"""

# COMMAND ----------


"""
Question:
---------
What is retrieval precision and recall?
If recall is low, what do you improve?

Answer:
--------
Precision:
Of retrieved chunks,
how many are relevant?

Recall:
Of all relevant chunks available,
how many did we retrieve?
"""

# COMMAND ----------


all_documents = [

    "Customer discount rule",                 # Relevant
    "Customer balance processing",            # Relevant
    "Login authentication system",            # Not Relevant
    "Discount approval workflow",             # Relevant
    "Employee payroll details",               # Not Relevant
    "Customer loyalty benefits",              # Relevant
    "Discount manager approval required",     # Relevant
    "Frontend UI settings"                    # Not Relevant

]

# COMMAND ----------

query = "customer discount rules"

# COMMAND ----------

retrieved_chunks = [

    "Customer discount rule",              # Relevant
    "Customer balance processing",         # Relevant
    "Login authentication system",         # Not Relevant
    "Discount approval workflow",          # Relevant
    "Employee payroll details"             # Not Relevant

]

# COMMAND ----------

total_relevant_chunks = [
    "Customer discount rule",
    "Customer balance processing",
    "Discount approval workflow",
    "Customer loyalty benefits",
    "Discount manager approval required"

]

# COMMAND ----------

relevant_retrieved = 0

for chunk in retrieved_chunks:

    if chunk in total_relevant_chunks:

        relevant_retrieved += 1

# COMMAND ----------

relevant_retrieved

# COMMAND ----------

"""
Precision =
Relevant Retrieved / Total Retrieved
"""

total_retrieved = len(retrieved_chunks)

precision = relevant_retrieved / total_retrieved

# COMMAND ----------

"""
Recall =
Relevant Retrieved / Total Relevant
"""

total_relevant = len(total_relevant_chunks)

recall = relevant_retrieved / total_relevant

# COMMAND ----------


print("\n")
print("=" * 60)
print("RETRIEVAL RESULTS")
print("=" * 60)

print(f"Query: {query}")

print("\nRetrieved Chunks:")

for chunk in retrieved_chunks:

    print(f"- {chunk}")

print("\n")
print("=" * 60)
print("METRICS")
print("=" * 60)

print(f"Relevant Retrieved : {relevant_retrieved}")

print(f"Total Retrieved    : {total_retrieved}")

print(f"Total Relevant     : {total_relevant}")

print("\n")
print(f"Precision = {precision:.2f}")

print(f"Recall    = {recall:.2f}")

# COMMAND ----------

print("\n")
print("=" * 60)
print("INTERPRETATION")
print("=" * 60)

print("""
Precision:
Out of retrieved chunks,
how many were useful?

Recall:
Out of all useful chunks available,
how many did we successfully retrieve?
""")

# COMMAND ----------


if recall < 0.80:

    print("\n")
    print("=" * 60)
    print("LOW RECALL DETECTED")
    print("=" * 60)

    print("""
System is missing important chunks.

This may cause:
- incomplete answers
- hallucinations
- missing business logic
""")

# COMMAND ----------


print("\n")
print("=" * 60)
print("WAYS TO IMPROVE RECALL")
print("=" * 60)

# ------------------------------------------------------------
# 1. Increase TOP-K
# ------------------------------------------------------------

print("""
1. Increase TOP-K Retrieval
---------------------------
Retrieve more chunks from vector DB
""")

top_k = 10

print(f"Example: top_k = {top_k}")

# ------------------------------------------------------------
# 2. Better Embedding Model
# ------------------------------------------------------------

print("""
2. Improve Embedding Model
--------------------------
Use stronger embeddings:
- BGE
- E5
- OpenAI embeddings
- Instructor XL
""")

# ------------------------------------------------------------
# 3. Hybrid Search
# ------------------------------------------------------------

print("""
3. Hybrid Search
----------------
Combine:
- BM25 keyword search
- Semantic vector search

This improves recall significantly.
""")

# ------------------------------------------------------------
# 4. Better Chunking
# ------------------------------------------------------------

print("""
4. Better Chunking
------------------
Split documents logically:
- headings
- sections
- paragraphs

Avoid random fixed splitting.
""")

# COMMAND ----------


print("\n")
print("=" * 60)
print("VISUAL EXPLANATION")
print("=" * 60)

print("""
All Relevant Chunks in DB:
------------------------------------------------
1. Discount rule
2. Customer balance
3. Approval workflow
4. Loyalty benefits
5. Manager approval

Retrieved:
------------------------------------------------
1. Discount rule
2. Customer balance
3. Approval workflow

Retrieved 3 out of 5 relevant chunks.

Recall = 3/5 = 60%
""")

# ============================================================
# INTERVIEW SHORT ANSWER
# ============================================================

"""
Interview Answer
----------------

Precision measures how many retrieved chunks
are actually relevant.

Recall measures how many of the total relevant
chunks were successfully retrieved.

Low recall means the retriever is missing
important information.

To improve recall:
- increase top-k retrieval,
- improve embeddings,
- use hybrid search,
- and optimize chunking strategy.
"""

# COMMAND ----------



# COMMAND ----------

# MAGIC %pip install aiohttp

# COMMAND ----------



"""
PROBLEM
-------
Process billions of records using an LLM API
within a strict 12-hour SLA.

MAIN BOTTLENECK
---------------
LLM API latency + network I/O.

BEST STRATEGIES
---------------
1. Async API calls
2. Batching
3. Spark partitioning
4. Parallel preprocessing
5. Multiprocessing for CPU-heavy tasks
6. Autoscaling SageMaker endpoints
7. Caching duplicate requests
8. Retry + rate limit handling
"""





import asyncio

import aiohttp

import json

import redis

import hashlib

from concurrent.futures import ThreadPoolExecutor

from multiprocessing import Pool

from pyspark.sql import SparkSession

# COMMAND ----------

# MAGIC %pip install  redis

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------


LLM_API_URL = "https://llm-api.company.com/generate"

MAX_CONCURRENT_REQUESTS = 100

BATCH_SIZE = 50

# COMMAND ----------

import redis
"""
Avoid duplicate LLM calls.

Huge cost optimization in enterprise systems.
"""

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)


# COMMAND ----------

from pyspark.sql import SparkSession
spark = (SparkSession.builder
    .appName("Billions_Record_LLM_Pipeline")
    .getOrCreate())


df = spark.read.parquet(
    "/Workspace/Users/prakhar1207srivastava@gmail.com/Drafts/InstructionsCreatePDFofE-VerifyManual.pdf"
)


# COMMAND ----------



def preprocess(record):
    """
    CPU-light cleanup
    """
    text = record.strip().lower()
    return text

# COMMAND ----------

def parallel_preprocess(records):
    with ThreadPoolExecutor(max_workers=16) as executor:
        return list(executor.map(preprocess, records))

# COMMAND ----------

def embedding_generation(text):
    """
    Simulated CPU-heavy operation
    """
    return hashlib.md5(text.encode()).hexdigest()

# COMMAND ----------

def multiprocessing_embeddings(records):
    with Pool(processes=8) as pool:
        return pool.map(embedding_generation, records)

# COMMAND ----------

def get_cache_key(text):
    return hashlib.md5(text.encode()).hexdigest()

# COMMAND ----------

async def process_record(session, semaphore, record):
    async with semaphore:
        cache_key = get_cache_key(record)
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)

        payload = {"input": record}

        async with session.post(LLM_API_URL, json=payload) as response:
            result = await response.json()

            # Save cache
            redis_client.setex(cache_key, 3600, json.dumps(result))
            return result

# COMMAND ----------

async def process_batch(batch):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_record(session, semaphore, record)
            for record in batch
        ]
        return await asyncio.gather(*tasks)


# COMMAND ----------



async def main_pipeline(records):

    print("Preprocessing records...")
    preprocessed = parallel_preprocess(records)
    print("Generating embeddings...")
    embeddings = multiprocessing_embeddings(preprocessed)
    batches = [
        preprocessed[i:i + BATCH_SIZE]
        for i in range(0, len(preprocessed), BATCH_SIZE)
    ]
    print(f"Total Batches: {len(batches)}")
    all_results = []
    for idx, batch in enumerate(batches):
        print(f"Processing Batch {idx}")
        results = await process_batch(batch)
        all_results.extend(results)

    return all_results

# COMMAND ----------

spark = (SparkSession.builder
    .appName("Billions_Record_LLM_Pipeline")
    .getOrCreate())

df = spark.createDataFrame(
    [(f"Sample text record {i}",) for i in range(1000)],
    ["text"]
)

records = [row["text"] for row in df.collect()]

# COMMAND ----------

# Import required modules
import asyncio
import aiohttp
import json
import redis
import hashlib
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pyspark.sql import SparkSession


results = await main_pipeline(records)


# COMMAND ----------


output_df = spark.createDataFrame(results)
output_df.write.mode("overwrite").saveAsTable("default.llm_pipeline_output")


# COMMAND ----------

# DBTITLE 1,Cell 54


# COMMAND ----------



# COMMAND ----------

# MAGIC %pip install redis aiohttp