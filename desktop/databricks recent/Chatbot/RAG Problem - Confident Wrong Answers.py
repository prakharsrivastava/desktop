# Databricks notebook source
# DBTITLE 1,Problem Statement
# MAGIC %md
# MAGIC # RAG Problem: Confident but Wrong Answers
# MAGIC
# MAGIC ## 🚨 Scenario
# MAGIC Aapka **RAG chatbot** ek sawal ka jawab deta hai jo:
# MAGIC - **Confident lagta hai** (LLM ne confidently answer diya)
# MAGIC - **Par galat hai** (answer sahi nahi hai)
# MAGIC
# MAGIC Jab aap **check karte ho**:
# MAGIC - ❌ Retrieved chunks mein **sahi information thi hi nahi** (galat chunks aaye)
# MAGIC - ✅ Source documents mein **answer maujood hai** (data mein hai, par retrieve nahi hua)
# MAGIC
# MAGIC ## ⚠️ Yeh Khatrnak Scenario Hai
# MAGIC Yeh sabse dangerous RAG failure hai kyunki:
# MAGIC 1. **Retrieval fail hua** → Sahi chunk retrieve nahi hua
# MAGIC 2. **LLM ne hallucinate kiya** → Jo chunks mile, unse confident galat jawab bana diya
# MAGIC 3. **User ko pata nahi chala** → Answer confident tha, isliye user ne maan liya

# COMMAND ----------

# DBTITLE 1,Answer (a) - Which Layer Failed
# MAGIC %md
# MAGIC ## (a) Kis Layer Ki Failure Hai?
# MAGIC
# MAGIC ### ✅ Primary Failure: RETRIEVAL Layer
# MAGIC
# MAGIC **Reason**: Retrieved chunks mein sahi information nahi thi → **Retrieval ne relevant chunk laana hi miss kar diya**
# MAGIC
# MAGIC **Yeh Retrieval failure hai, NOT chunking failure:**
# MAGIC - **Chunking failure**: Answer chunk ke beech mein cut gaya ho (split across boundaries)
# MAGIC - **Retrieval failure**: Answer documents mein hai, chunks bhi honge, par **search/embedding ne relevant chunk rank nahi kiya**
# MAGIC
# MAGIC Is case mein: **Galat chunks retrieve hue** → Retrieval layer ne relevant document/chunk ko top-k mein nahi laaya.
# MAGIC
# MAGIC ### ⚠️ Secondary Issue: Generation Layer (Hallucination)
# MAGIC
# MAGIC **Confident wrong answer** iska signal hai:
# MAGIC - LLM ko **irrelevant context mila**
# MAGIC - LLM ne **hallucinate karke** plausible answer banaya
# MAGIC - **Grounding/faithfulness check** nahi tha prompt mein
# MAGIC
# MAGIC **Toh dono layers involved hain:**
# MAGIC 1. **Primary**: Retrieval failure (sahi chunk nahi aaya)
# MAGIC 2. **Secondary**: Generation ne hallucinate kiya (guardrails nahi the)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 3-Line Model Answer
# MAGIC
# MAGIC > **"(a) Yeh RETRIEVAL failure hai — relevant chunk retrieve hi nahi hua. Secondary issue: LLM ne hallucinate karke confident galat jawab diya kyunki prompt mein grounding guardrail nahi tha."**

# COMMAND ----------

# DBTITLE 1,Answer (b) - Metrics to Measure
# MAGIC %md
# MAGIC ## (b) Kaunse Metrics Se Measure Karoge?
# MAGIC
# MAGIC ### For Retrieval Failure:
# MAGIC
# MAGIC **1. Recall@k** (Primary metric)
# MAGIC - **Definition**: Top-k retrieved chunks mein se kitne relevant chunks aaye?
# MAGIC - **Formula**: (Relevant chunks in top-k) / (Total relevant chunks in corpus)
# MAGIC - **Is case mein**: Recall@k = 0 (kyunki sahi chunk retrieve hi nahi hua)
# MAGIC - **Target**: High Recall@k (>0.8) means most relevant chunks are being retrieved
# MAGIC
# MAGIC **2. Context Precision** (Ragas metric)
# MAGIC - **Definition**: Retrieved chunks mein se kitne actually relevant the?
# MAGIC - **Measures**: Noise level in retrieved context
# MAGIC - **Is case mein**: Low context precision (irrelevant chunks retrieve hue)
# MAGIC - **Why important**: High noise → LLM ko galat info milti hai
# MAGIC
# MAGIC **3. MRR (Mean Reciprocal Rank)**
# MAGIC - **Definition**: Pehla relevant chunk kaunsi position par aaya?
# MAGIC - **Formula**: 1 / (rank of first relevant chunk)
# MAGIC - **Is case mein**: MRR = 0 (relevant chunk top-k mein hi nahi)
# MAGIC - **Good MRR**: 0.8+ (relevant chunks top positions par hain)
# MAGIC
# MAGIC ### For Hallucination (Generation Issue):
# MAGIC
# MAGIC **4. Faithfulness / Groundedness** (Ragas metric)
# MAGIC - **Definition**: Answer context ke andar grounded hai ya bahar se info add hua?
# MAGIC - **Measures**: Hallucination detection
# MAGIC - **Is case mein**: Low faithfulness (LLM ne context se bahar jaake answer diya)
# MAGIC - **How to check**: Har statement ko context ke saath verify karo
# MAGIC
# MAGIC **5. Answer Relevance**
# MAGIC - **Definition**: Answer question ko address karta hai?
# MAGIC - **Measures**: Whether LLM stayed on topic
# MAGIC - **Complementary metric**: Answer relevant ho sakta hai par faithful nahi (hallucinated)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 3-Line Model Answer
# MAGIC
# MAGIC > **"(b) Retrieval measure karo: Recall@k (sahi chunk top-k mein aaya ya nahi) aur Context Precision (kitna noise tha). Generation measure karo: Faithfulness via Ragas (answer grounded hai ya hallucinated)."**

# COMMAND ----------

# DBTITLE 1,Answer (c) - 2 Concrete Fixes
# MAGIC %md
# MAGIC ## (c) 2 Concrete Fixes
# MAGIC
# MAGIC ### Fix 1: Hybrid Search + Cross-Encoder Re-ranker
# MAGIC
# MAGIC **Problem**: Pure vector search ne relevant chunk miss kar diya
# MAGIC
# MAGIC **Solution**: 2-stage retrieval
# MAGIC
# MAGIC **Stage 1 - Hybrid Search:**
# MAGIC - **Dense (Vector) Search**: Semantic similarity (embeddings)
# MAGIC - **Sparse (BM25) Search**: Exact keyword matching
# MAGIC - **Combine**: Weighted fusion (e.g., 70% dense + 30% sparse)
# MAGIC
# MAGIC **Why it helps**:
# MAGIC - Vector search semantic understanding deta hai
# MAGIC - BM25 exact keywords nahi miss karta
# MAGIC - Together: Better recall
# MAGIC
# MAGIC **Stage 2 - Cross-Encoder Re-ranker:**
# MAGIC - Top-k candidates (say 20) retrieve karo
# MAGIC - Cross-encoder model se query-chunk pairs ko re-score karo
# MAGIC - Top 3-5 best chunks select karo
# MAGIC
# MAGIC **Why it helps**:
# MAGIC - Cross-encoder zyada accurate hai (query aur chunk dono saath process karta hai)
# MAGIC - Relevant chunks ko top par laata hai
# MAGIC - Precision improve hota hai
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Fix 2: Prompt Guardrail + Better Chunking
# MAGIC
# MAGIC **Problem 1**: LLM hallucinate kar raha hai jab irrelevant context milta hai
# MAGIC
# MAGIC **Solution - Grounding Guardrail:**
# MAGIC ```
# MAGIC IMPORTANT: Answer ONLY using the context provided.
# MAGIC If the answer is NOT in the context, respond:
# MAGIC "I don't have enough information to answer this question."
# MAGIC Do NOT make up information.
# MAGIC ```
# MAGIC
# MAGIC **Why it helps**:
# MAGIC - LLM ko explicitly instruction milta hai
# MAGIC - Hallucination reduce hota hai
# MAGIC - User ko pata chal jata hai ki answer nahi mila
# MAGIC
# MAGIC **Problem 2**: Chunking strategy weak hai (answer retrieve nahi ho raha)
# MAGIC
# MAGIC **Solution - Better Chunking:**
# MAGIC - **Chunk size**: 700-1000 tokens (experiment karo)
# MAGIC - **Overlap**: 100-150 tokens (10-20% of chunk size)
# MAGIC - **Semantic splitting**: Sentence/paragraph boundaries par split karo
# MAGIC - **Re-index**: Naye chunking strategy ke baad re-index karo
# MAGIC
# MAGIC **Why it helps**:
# MAGIC - Overlap se boundary issues solve hote hain
# MAGIC - Context loss nahi hota
# MAGIC - Semantic splitting se coherent chunks bante hain
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 3-Line Model Answer
# MAGIC
# MAGIC > **"(c) Fix-1: Hybrid search (dense + BM25) + cross-encoder re-ranker taaki relevant chunk top-k mein aaye. Fix-2: Prompt guardrail 'only answer from context, else say I don't know' + better chunking (size + overlap) aur re-index."**

# COMMAND ----------

# DBTITLE 1,Code - Hybrid Search Implementation
# Fix 1: Hybrid Search + Re-ranker Implementation

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

def hybrid_search_with_reranker(query, documents, vector_index, top_k=20, final_k=5):
    """
    Implement hybrid search with cross-encoder re-ranking.
    
    Args:
        query: User's question
        documents: List of document chunks
        vector_index: Your vector search index
        top_k: Number of candidates to retrieve
        final_k: Final number of chunks after re-ranking
    
    Returns:
        Top final_k most relevant chunks
    """
    
    # --- Stage 1: Hybrid Search ---
    
    # 1. Dense (Vector) Search
    query_embedding = vector_index.embed(query)
    vector_results = vector_index.query(query_embedding, top_k=top_k)
    vector_scores = {doc['id']: doc['score'] for doc in vector_results}
    
    # 2. Sparse (BM25) Search
    tokenized_docs = [doc['text'].split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = query.split()
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Normalize BM25 scores to [0, 1]
    bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
    bm25_results = {documents[i]['id']: score for i, score in enumerate(bm25_scores_norm)}
    
    # 3. Combine with weights (70% dense, 30% sparse)
    combined_scores = {}
    all_doc_ids = set(vector_scores.keys()) | set(bm25_results.keys())
    
    for doc_id in all_doc_ids:
        vector_score = vector_scores.get(doc_id, 0)
        bm25_score = bm25_results.get(doc_id, 0)
        combined_scores[doc_id] = 0.7 * vector_score + 0.3 * bm25_score
    
    # Get top-k candidates
    top_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    candidate_ids = [doc_id for doc_id, _ in top_candidates]
    
    # --- Stage 2: Cross-Encoder Re-ranking ---
    
    # Get document texts for candidates
    candidate_docs = [doc for doc in documents if doc['id'] in candidate_ids]
    
    # Re-rank using cross-encoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[query, doc['text']] for doc in candidate_docs]
    rerank_scores = reranker.predict(pairs)
    
    # Sort by re-ranked scores
    reranked = sorted(zip(candidate_docs, rerank_scores), key=lambda x: x[1], reverse=True)
    
    # Return top final_k
    final_chunks = [doc for doc, score in reranked[:final_k]]
    
    return final_chunks

# Example usage:
# results = hybrid_search_with_reranker(
#     query="What is the refund policy?",
#     documents=your_document_chunks,
#     vector_index=your_vector_index,
#     top_k=20,
#     final_k=5
# )

print("✅ Hybrid search + re-ranker implementation ready!")

# COMMAND ----------

# DBTITLE 1,Code - Prompt Guardrail
# Fix 2: Prompt Guardrail to Prevent Hallucination

def generate_with_guardrail(query, retrieved_chunks, llm):
    """
    Generate answer with grounding guardrail to prevent hallucination.
    
    Args:
        query: User's question
        retrieved_chunks: List of retrieved chunks
        llm: Your LLM instance
    
    Returns:
        Answer with hallucination prevention
    """
    
    # Build context from chunks
    context = "\n\n---\n\n".join([f"[Chunk {i+1}]\n{chunk['text']}" for i, chunk in enumerate(retrieved_chunks)])
    
    # Prompt with strong grounding instructions
    prompt = f"""
You are a helpful assistant that answers questions based ONLY on the provided context.

**CRITICAL RULES:**
1. Answer ONLY using information from the context below.
2. If the answer is NOT clearly stated in the context, respond EXACTLY:
   "I don't have enough information to answer this question."
3. Do NOT make assumptions or add information from outside the context.
4. Cite which chunk(s) you used (e.g., "According to Chunk 2...").
5. If the context is partially relevant but incomplete, say:
   "Based on the available context, [partial answer], but I don't have complete information."

**CONTEXT:**
{context}

**QUESTION:**
{query}

**ANSWER (follow the rules above):**
"""
    
    # Generate response
    response = llm.generate(prompt)
    
    return {
        "answer": response,
        "chunks_used": retrieved_chunks,
        "guardrail_active": True
    }

# Example usage:
# result = generate_with_guardrail(
#     query="What is the refund policy?",
#     retrieved_chunks=chunks,
#     llm=your_llm_instance
# )
# print(result['answer'])

print("✅ Prompt guardrail implementation ready!")

# COMMAND ----------

# DBTITLE 1,Code - Better Chunking Strategy
# Fix 2 (Part 2): Better Chunking with Overlap

from langchain.text_splitter import RecursiveCharacterTextSplitter

def improved_chunking(documents, chunk_size=800, overlap=100):
    """
    Implement better chunking strategy with overlap and semantic boundaries.
    
    Args:
        documents: List of documents to chunk
        chunk_size: Target chunk size in tokens (700-1000 recommended)
        overlap: Overlap between chunks in tokens (10-20% of chunk_size)
    
    Returns:
        List of chunks with metadata
    """
    
    # Configure text splitter with semantic boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=[
            "\n\n",    # Paragraph breaks (highest priority)
            "\n",      # Line breaks
            ". ",      # Sentence endings
            "! ",      # Exclamation sentences
            "? ",      # Question sentences
            "; ",      # Semi-colon breaks
            ", ",      # Comma breaks
            " ",       # Word breaks
            ""         # Character-level (last resort)
        ],
        is_separator_regex=False
    )
    
    chunks = []
    
    for doc_id, doc in enumerate(documents):
        # Split document into chunks
        doc_chunks = text_splitter.split_text(doc['text'])
        
        # Add metadata to each chunk
        for chunk_idx, chunk_text in enumerate(doc_chunks):
            chunks.append({
                'id': f"doc_{doc_id}_chunk_{chunk_idx}",
                'text': chunk_text,
                'source_doc': doc_id,
                'chunk_index': chunk_idx,
                'metadata': doc.get('metadata', {})
            })
    
    return chunks

# Example usage:
"""
documents = [
    {'text': 'Your long document text here...', 'metadata': {'source': 'policy.pdf'}},
    {'text': 'Another document...', 'metadata': {'source': 'faq.pdf'}}
]

chunks = improved_chunking(
    documents=documents,
    chunk_size=800,   # Experiment: 700-1000
    overlap=100       # 10-20% of chunk_size
)

print(f"Created {len(chunks)} chunks with overlap")
print(f"Sample chunk: {chunks[0]['text'][:200]}...")

# After chunking, RE-INDEX your vector database
# vector_index.index(chunks)  # Re-build embeddings
"""

print("✅ Improved chunking strategy ready!")
print("⚠️ Important: After changing chunking strategy, RE-INDEX your vector database!")

# COMMAND ----------

# DBTITLE 1,Summary - Complete Solution
# MAGIC %md
# MAGIC ## 📋 Complete Solution Summary
# MAGIC
# MAGIC ### Problem Recap
# MAGIC - **Issue**: Confident but wrong answers
# MAGIC - **Root cause**: Retrieval failure + hallucination
# MAGIC - **Data**: Answer IS in documents, but not retrieved
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Solution Framework
# MAGIC
# MAGIC #### (a) Layer Analysis
# MAGIC 1. **Primary**: RETRIEVAL failure (relevant chunks not retrieved)
# MAGIC 2. **Secondary**: GENERATION hallucination (no grounding guardrail)
# MAGIC
# MAGIC #### (b) Metrics
# MAGIC **Retrieval:**
# MAGIC - Recall@k → Are relevant chunks in top-k?
# MAGIC - Context Precision → How much noise in retrieved chunks?
# MAGIC - MRR → Where is the first relevant chunk ranked?
# MAGIC
# MAGIC **Generation:**
# MAGIC - Faithfulness (Ragas) → Is answer grounded in context?
# MAGIC - Answer Relevance → Does answer address the question?
# MAGIC
# MAGIC #### (c) Fixes
# MAGIC
# MAGIC **Fix 1: Hybrid Search + Re-ranker**
# MAGIC - Stage 1: Combine dense (vector) + sparse (BM25) search
# MAGIC - Stage 2: Cross-encoder re-ranking for top candidates
# MAGIC - Result: Better recall, relevant chunks rank higher
# MAGIC
# MAGIC **Fix 2: Prompt Guardrail + Better Chunking**
# MAGIC - Add explicit grounding instruction in prompt
# MAGIC - "Answer ONLY from context, else say 'I don't know'"
# MAGIC - Improve chunking: size (700-1000) + overlap (100-150)
# MAGIC - Re-index after chunking changes
# MAGIC - Result: Less hallucination, better retrieval coverage
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Interview Model Answer (3 Lines)
# MAGIC
# MAGIC **(a)** Yeh RETRIEVAL failure hai — relevant chunk retrieve hi nahi hua, aur LLM ne hallucinate karke confident galat jawab diya.
# MAGIC
# MAGIC **(b)** Measure: Recall@k & Context Precision (retrieval), Faithfulness via Ragas (answer grounded hai ya nahi).
# MAGIC
# MAGIC **(c)** Fix-1: Hybrid search (dense + BM25) + cross-encoder re-ranker; Fix-2: Prompt guardrail 'only answer from context, else say I don't know' + better chunking (size + overlap) + re-index.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔄 Implementation Checklist
# MAGIC
# MAGIC - [ ] Implement hybrid search (vector + BM25)
# MAGIC - [ ] Add cross-encoder re-ranker
# MAGIC - [ ] Update chunking strategy with overlap
# MAGIC - [ ] Re-index vector database
# MAGIC - [ ] Add prompt guardrail for grounding
# MAGIC - [ ] Implement Ragas metrics for monitoring
# MAGIC - [ ] Set up Recall@k and Context Precision tracking
# MAGIC - [ ] A/B test changes in production
# MAGIC - [ ] Monitor faithfulness scores

# COMMAND ----------

