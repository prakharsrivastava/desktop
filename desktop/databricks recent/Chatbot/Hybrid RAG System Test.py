# Databricks notebook source
# DBTITLE 1,Install Dependencies
# MAGIC %pip install sentence-transformers rank-bm25
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup Models
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np
from databricks.sdk import WorkspaceClient

# Load models
print("Loading embedding model...")
embed_model = SentenceTransformer("BAAI/bge-large-en")
print("Loading reranker model...")
reranker = CrossEncoder("BAAI/bge-reranker-large")
print("Models loaded successfully!")

# COMMAND ----------

# Sample documents
documents = [
    {"content": "CI/CD automates build, test, and deployment."},
    {"content": "Docker ensures consistent runtime environments using containers."},
    {"content": "Kubernetes orchestrates container deployment and scaling."}
]

corpus = [doc["content"] for doc in documents]
corpus

# COMMAND ----------

# DBTITLE 1,Prepare Document Corpus

tokenized_corpus = [doc.split() for doc in corpus]

# Initialize BM25
bm25 = BM25Okapi(tokenized_corpus)

# Pre-compute document embeddings for vector search
print("Computing document embeddings...")
doc_embeddings = embed_model.encode(corpus)
print(f"Created embeddings for {len(documents)} documents")

# COMMAND ----------

# DBTITLE 1,Define Hybrid RAG Function
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

def hybrid_rag(question, doc_embeddings, corpus, bm25):
    """
    Hybrid RAG combining BM25, vector search, and reranking.
    """
    
    # ---- BM25 Search ----
    tokenized_query = question.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:5]
    bm25_results = [(corpus[i], bm25_scores[i]) for i in bm25_top_idx]
    
    # ---- Vector Search (using cosine similarity) ----
    question_embedding = embed_model.encode(question)
    
    # Compute cosine similarity
    similarities = np.dot(doc_embeddings, question_embedding) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    
    vector_top_idx = np.argsort(similarities)[::-1][:5]
    vector_results = [(corpus[i], similarities[i]) for i in vector_top_idx]
    
    # ---- Combine Results ----
    combined = bm25_results + vector_results
    
    # ---- Reranking ----
    pairs = [(question, doc[0]) for doc in combined]
    scores = reranker.predict(pairs)
    
    reranked = sorted(zip(combined, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc[0][0] for doc in reranked[:3]]
    
    # ---- Build Context ----
    context = "\n".join(top_docs)
    
    prompt = f"""Answer only from the given context.

Context:
{context}

Question:
{question}
"""
    
    # ---- LLM Call ----
    w = WorkspaceClient()
    response = w.serving_endpoints.query(
        name="databricks-meta-llama-3.1-405b-instruct",
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)]
    )
    
    return {
        "answer": response.choices[0].message.content,
        "top_docs": top_docs,
        "bm25_scores": [(corpus[i], bm25_scores[i]) for i in bm25_top_idx[:3]],
        "vector_scores": [(corpus[i], similarities[i]) for i in vector_top_idx[:3]]
    }

print("Hybrid RAG function defined successfully!")

# COMMAND ----------

# DBTITLE 1,Test Hybrid RAG System
# Test the hybrid RAG system
test_question = "What is Kubernetes used for?"

print(f"Question: {test_question}\n")
print("Running hybrid RAG...\n")

result = hybrid_rag(test_question, doc_embeddings, corpus, bm25)

print("=" * 60)
print("BM25 Top Results:")
for doc, score in result["bm25_scores"]:
    print(f"  - {doc} (score: {score:.4f})")

print("\n" + "=" * 60)
print("Vector Search Top Results:")
for doc, score in result["vector_scores"]:
    print(f"  - {doc} (score: {score:.4f})")

print("\n" + "=" * 60)
print("Top 3 Reranked Documents:")
for i, doc in enumerate(result["top_docs"], 1):
    print(f"  {i}. {doc}")

print("\n" + "=" * 60)
print("Final Answer:")
print(result["answer"])