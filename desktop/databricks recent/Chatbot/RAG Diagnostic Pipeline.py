# Databricks notebook source
# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # RAG System Debugging Guide: Production Issues
# MAGIC
# MAGIC ## Problem Statement
# MAGIC Aapne ek **RAG-based system** banaya hai jo company documents par questions ke answers deta hai. Production mein users complain kar rahe hain ki:
# MAGIC - Answers **galat** hote hain
# MAGIC - Answers **adhure** (incomplete) hote hain
# MAGIC - **Sahi answer documents mein hai** lekin system wo nahi de pa raha
# MAGIC
# MAGIC ## Goal
# MAGIC Is notebook mein hum **end-to-end debugging approach** dekhenge:
# MAGIC 1. **Retrieval vs Generation** ko isolate karna
# MAGIC 2. RAG pipeline ke har layer ko debug karna
# MAGIC 3. Concrete metrics aur fixes apply karna

# COMMAND ----------

# DBTITLE 1,Step 0: Isolation Test - Critical First Step
# MAGIC %md
# MAGIC ## 🎯 Step 0: Isolation Test - Retrieval vs Generation
# MAGIC
# MAGIC **SABSE PEHLE YEH TEST KARO** — diagnosis ka sahi order:
# MAGIC
# MAGIC ### Test Logic
# MAGIC 1. **Manual Gold Chunk Test**: Jo chunk mein answer hai, use manually LLM ko feed karo
# MAGIC 2. **Interpret Result**:
# MAGIC    - ✅ **Tab sahi jawab aaya** → Problem **RETRIEVAL** mein hai
# MAGIC    - ❌ **Tab bhi galat/adhure** → Problem **GENERATION/PROMPT** mein hai
# MAGIC
# MAGIC ### Why This Matters
# MAGIC Yeh test pipeline ko **split** karta hai aur batata hai ki focus kahan lagana hai:
# MAGIC - Retrieval fix: Chunking, embedding, search algorithm
# MAGIC - Generation fix: Prompt design, context handling, groundedness
# MAGIC
# MAGIC **Yeh test sabse pehle karna zaroori hai** — baaki debugging iske baad directed hogi.

# COMMAND ----------

# DBTITLE 1,Isolation Test - Code Example
# Isolation Test: Manually feed the correct chunk to LLM


# COMMAND ----------

# Try the isolation test with OpenAI LLM instead of Groq

from openai import OpenAI

openai_client = OpenAI(api_key="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")

from openai import OpenAI
import os
client = OpenAI(
    api_key="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",
    base_url="https://api.groq.com/openai/v1",
)
# Create LLM object
groq_llm = GroqLLM(client)


# COMMAND ----------

# DBTITLE 1,Cell 2

class OpenAILLM:
    def __init__(self, client, model="gpt-4o-mini"):
        self.client = client
        self.model = model

    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content

openai_llm = OpenAILLM(openai_client)

result_openai = isolation_test(
    question="What is the refund policy?",
    gold_chunk="Company refund policy: Full refunds available within 60 days of purchase...",
    llm=openai_llm
)
print(result_openai)

# COMMAND ----------


def isolation_test(question, gold_chunk, llm):
    """
    Test whether the issue is in retrieval or generation.
    
    Args:
        gold_chunk: The chunk that contains the correct answer
        llm: Your LLM instance
    
    Returns:
        Dict with diagnosis
    """
    prompt = f"""
You are a helpful assistant. Answer the question based ONLY on the context provided.

Context: {gold_chunk}

Question: {question}

Answer:"""
    
    response = llm.generate(prompt)
    
    return {
        "question": question,
        "gold_chunk_preview": gold_chunk[:200] + "...",
        "llm_response": response,
        "diagnosis": "Check if response is correct. If YES → Retrieval issue. If NO → Generation issue."
    }


# COMMAND ----------

# DBTITLE 1,Cell 5

class GroqLLM:
    def __init__(self, client, model="openai/gpt-oss-120b"):
        self.client = client
        self.model = model

    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        return response.choices[0].message.content

groq_llm = GroqLLM(client)

result = isolation_test(
     question="What is the refund policy?",
     gold_chunk="Company refund policy: Full refunds available within 30 days of purchase...",
     llm=groq_llm
)


# COMMAND ----------

# COMMAND ----------

result

# COMMAND ----------

# DBTITLE 1,Concrete Fixes - Action Items
# MAGIC %md
# MAGIC ## 🔧 Concrete Fixes - Action Items
# MAGIC
# MAGIC ### For Retrieval Issues:
# MAGIC
# MAGIC **1. Chunking Strategy**
# MAGIC ```python
# MAGIC # Better chunking with overlap
# MAGIC from langchain.text_splitter import RecursiveCharacterTextSplitter
# MAGIC
# MAGIC text_splitter = RecursiveCharacterTextSplitter(
# MAGIC     chunk_size=800,          # Experiment: 500-1000
# MAGIC     chunk_overlap=100,       # 10-20% of chunk_size
# MAGIC     separators=["\n\n", "\n", ". ", " ", ""]  # Semantic boundaries
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC **2. Hybrid Search (Dense + Sparse)**
# MAGIC ```python
# MAGIC # Combine vector search with BM25
# MAGIC from rank_bm25 import BM25Okapi
# MAGIC
# MAGIC # Vector search results
# MAGIC vector_results = vector_index.query(query_embedding, top_k=20)
# MAGIC
# MAGIC # BM25 keyword search
# MAGIC bm25_results = bm25.get_top_n(query_tokens, corpus, n=20)
# MAGIC
# MAGIC # Combine with weights
# MAGIC final_results = combine_scores(
# MAGIC     vector_results, 
# MAGIC     bm25_results, 
# MAGIC     vector_weight=0.7, 
# MAGIC     bm25_weight=0.3
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC **3. Re-ranker (Cross-Encoder)**
# MAGIC ```python
# MAGIC from sentence_transformers import CrossEncoder
# MAGIC
# MAGIC # After initial retrieval, re-rank top candidates
# MAGIC reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# MAGIC
# MAGIC # Score query-chunk pairs
# MAGIC pairs = [[query, chunk['text']] for chunk in context_retrival]
# MAGIC scores = reranker.predict(pairs)
# MAGIC
# MAGIC # Sort by new scores
# MAGIC reranked = sorted(zip(context_retrival, scores), key=lambda x: x[1], reverse=True)
# MAGIC top_chunks = [chunk for chunk, score in reranked[:5]]
# MAGIC ```
# MAGIC
# MAGIC ### For Generation Issues:
# MAGIC
# MAGIC **1. Better Prompt Design**
# MAGIC ```python
# MAGIC prompt_template = """
# MAGIC You are a helpful assistant. Answer the question using ONLY the context provided below.
# MAGIC
# MAGIC IMPORTANT RULES:
# MAGIC 1. If the answer is not in the context, say "I don't have enough information to answer this."
# MAGIC 2. Do not add information from outside the context.
# MAGIC 3. Cite which part of the context you used (e.g., "According to section 2...").
# MAGIC
# MAGIC Context:
# MAGIC {context}
# MAGIC
# MAGIC Question: {question}
# MAGIC
# MAGIC Answer (be concise and cite sources):
# MAGIC """
# MAGIC ```
# MAGIC
# MAGIC **2. Context Window Management**
# MAGIC ```python
# MAGIC # Limit to top 3-5 BEST chunks (after re-ranking)
# MAGIC max_chunks = 5
# MAGIC max_context_length = 3000  # tokens
# MAGIC
# MAGIC selected_chunks = []
# MAGIC total_length = 0
# MAGIC
# MAGIC for chunk in reranked_chunks:
# MAGIC     chunk_length = len(chunk['text'].split())
# MAGIC     if len(selected_chunks) >= max_chunks or total_length + chunk_length > max_context_length:
# MAGIC         break
# MAGIC     selected_chunks.append(chunk)
# MAGIC     total_length += chunk_length
# MAGIC
# MAGIC # Place best chunks at START and END (avoid "lost in middle")
# MAGIC if len(selected_chunks) > 2:
# MAGIC     ordered = [selected_chunks[0]] + selected_chunks[2:-1] + [selected_chunks[1]]
# MAGIC else:
# MAGIC     ordered = selected_chunks
# MAGIC ```
# MAGIC
# MAGIC **3. Faithfulness Checking**
# MAGIC ```python
# MAGIC # Post-generation validation
# MAGIC def check_faithfulness(answer, context_chunks, llm):
# MAGIC     check_prompt = f"""
# MAGIC Verify if the following answer is faithful to the context.
# MAGIC Answer ONLY 'yes' or 'no' and provide reason.
# MAGIC
# MAGIC Context: {' '.join([c['text'][:500] for c in context_chunks])}
# MAGIC
# MAGIC Answer: {answer}
# MAGIC
# MAGIC Is this answer faithful to the context? (yes/no):
# MAGIC """
# MAGIC     
# MAGIC     result = llm.generate(check_prompt)
# MAGIC     return 'yes' in result.lower()
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Complete Diagnostic Flow - Summary
# MAGIC %md
# MAGIC ## 🏁 Complete End-to-End Diagnostic Flow
# MAGIC
# MAGIC ### Quick Reference Framework
# MAGIC
# MAGIC **Pehle main retrieval vs generation isolate karunga** (gold chunk manually feed karke).
# MAGIC
# MAGIC **Agar Retrieval Issue →**
# MAGIC - **Chunking/Embedding/Hybrid+Re-ranker** test karo
# MAGIC - **Measured by**: Recall@k, Hit Rate@k, MRR
# MAGIC - **Fixes**: Chunk size + overlap, hybrid search, cross-encoder re-ranker
# MAGIC
# MAGIC **Agar Generation Issue →**
# MAGIC - **Prompt/Context Size/Faithfulness** test karo
# MAGIC - **Measured by**: Faithfulness, Answer Relevance, Context Precision (via Ragas)
# MAGIC - **Fixes**: Better prompt grounding, limit chunks (3-5), faithfulness validation
# MAGIC
# MAGIC ### Step-by-Step Process:
# MAGIC
# MAGIC 1. ✅ **Isolation Test** (Step 0)
# MAGIC    - Manual gold chunk test
# MAGIC    - Determines retrieval vs generation
# MAGIC
# MAGIC 2. 🔍 **Layer-by-Layer Debug**
# MAGIC    - Retrieval: Chunking → Embedding → Search
# MAGIC    - Generation: Prompt → Context → Faithfulness
# MAGIC
# MAGIC 3. 📊 **Measure with Metrics**
# MAGIC    - Retrieval: Recall@k, MRR, NDCG
# MAGIC    - Generation: Ragas (faithfulness, relevance)
# MAGIC
# MAGIC 4. 🔧 **Apply Concrete Fixes**
# MAGIC    - Retrieval: Hybrid search + re-ranker
# MAGIC    - Generation: Prompt engineering + context limiting
# MAGIC
# MAGIC 5. ♻️ **Iterate and Monitor**
# MAGIC    - A/B test changes
# MAGIC    - Track metrics in production
# MAGIC    - User feedback loop

# COMMAND ----------

# DBTITLE 1,Complete Example - Full Diagnostic Pipeline
# Complete End-to-End Diagnostic Example


# COMMAND ----------

# DBTITLE 1,Metrics to Track - Concrete RAG Evaluation
# MAGIC %md
# MAGIC ## 📊 Metrics to Track - Concrete RAG Evaluation
# MAGIC
# MAGIC ### Retrieval Metrics
# MAGIC **Yeh metrics batate hain ki retrieval kitna effective hai:**
# MAGIC
# MAGIC 1. **Recall@k**: Top-k retrieved chunks mein se kitne relevant hain?
# MAGIC    - Formula: (Relevant chunks in top-k) / (Total relevant chunks)
# MAGIC    - Example: Agar 5 relevant chunks hain aur top-10 mein se 3 mile → Recall@10 = 3/5 = 0.6
# MAGIC
# MAGIC 2. **Hit Rate@k**: Kam se kam 1 relevant chunk top-k mein hai ya nahi?
# MAGIC    - Binary metric: 0 or 1
# MAGIC    - Production mein useful: "Kya answer mil sakta hai ya nahi"
# MAGIC
# MAGIC 3. **MRR (Mean Reciprocal Rank)**: Pehla relevant chunk kaunsi position par hai?
# MAGIC    - Formula: 1 / rank_of_first_relevant_chunk
# MAGIC    - Example: Pehla relevant chunk rank 3 par → MRR = 1/3 = 0.33
# MAGIC    - Higher is better (ideal = 1.0 jab rank 1 par ho)
# MAGIC
# MAGIC 4. **NDCG@k (Normalized Discounted Cumulative Gain)**: Ranking quality measure
# MAGIC    - Considers both relevance and position
# MAGIC    - Range: 0 to 1 (1 = perfect ranking)
# MAGIC
# MAGIC ### Generation Metrics
# MAGIC **Yeh metrics batate hain ki LLM ka output kitna accurate aur grounded hai:**
# MAGIC
# MAGIC 1. **Faithfulness**: Answer context ke andar hai ya bahar se info add hua?
# MAGIC    - Measures hallucination
# MAGIC    - Check: Har statement context mein verify ho sakti hai?
# MAGIC
# MAGIC 2. **Answer Relevance**: Answer question ko address karta hai?
# MAGIC    - Semantic similarity between question and answer
# MAGIC    - Checks if LLM stayed on topic
# MAGIC
# MAGIC 3. **Context Precision**: Retrieved chunks kitne relevant the?
# MAGIC    - Measures retrieval quality from generation perspective
# MAGIC    - Lower noise in context = better generation
# MAGIC
# MAGIC 4. **Context Recall**: Answer ke liye zaroori saari info context mein thi?
# MAGIC    - Measures if enough context was provided
# MAGIC
# MAGIC ### 🔧 Ragas Framework
# MAGIC **Industry-standard RAG evaluation framework** jo in saare metrics ko implement karta hai:
# MAGIC - pip install ragas
# MAGIC - Automatic evaluation of faithfulness, relevance, context metrics
# MAGIC - Production monitoring ke liye ready
# MAGIC - Integrates with LangChain, LlamaIndex

# COMMAND ----------

# DBTITLE 1,Metrics Implementation - Code Example
# Concrete Metrics Implementation

# COMMAND ----------

def calculate_retrieval_metrics(context_retrival, relevant_chunk_ids, k=10):
    """
    Calculate key retrieval metrics.
    
    Args:
        context_retrival: List of retrieved chunk IDs (in order)
        relevant_chunk_ids: Set of actually relevant chunk IDs
        k: Top-k to consider
    
    Returns:
        Dict of metrics
    """
    top_k = context_retrival[:k]
    
    # Recall@k
    relevant_in_topk = len(set(top_k) & relevant_chunk_ids)
    recall_at_k = relevant_in_topk / len(relevant_chunk_ids) if relevant_chunk_ids else 0
    
    # Hit Rate@k
    hit_rate = 1 if relevant_in_topk > 0 else 0
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0
    for idx, chunk_id in enumerate(top_k, start=1):
        if chunk_id in relevant_chunk_ids:
            mrr = 1 / idx
            break
    
    return {
        "recall@k": recall_at_k,
        "hit_rate@k": hit_rate,
        "mrr": mrr,
        "relevant_found": relevant_in_topk,
        "total_relevant": len(relevant_chunk_ids)
    }

# COMMAND ----------

# Install: 
%pip install ragas

# COMMAND ----------

# DBTITLE 1,Cell 9


# Example: Using Ragas for Generation Metrics

# Compatibility shim: langchain-community 0.4+ removed chat_models.vertexai
# but ragas 0.4.3 still imports it. ChatVertexAI is only used for isinstance
# checks in MULTIPLE_COMPLETION_SUPPORTED, so a dummy class suffices.
import sys
import types
_vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI:
    pass
_vertexai_module.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_module

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# Prepare your dataset
from datasets import Dataset

data = {
    "question": ["What is the refund policy?"],
    "answer": ["Full refunds within 30 days"],
    "contexts": [["Company policy: Full refunds available within 30 days..."]],
    "ground_truth": ["Full refunds available within 30 days of purchase"]
}

dataset = Dataset.from_dict(data)

# Evaluate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.fake import FakeEmbeddings

eval_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_client.api_key,
)

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=eval_llm,
    embeddings=FakeEmbeddings(size=1536)
)

print(result)
# Output: {'faithfulness': 0.95, 'answer_relevancy': 0.92, ...}


print("Metrics code ready. Install ragas for generation metrics: pip install ragas")


# COMMAND ----------

def diagnose_generation(question, context_retrival, llm_response, llm):
    """
    Diagnose generation issues.
    
    Args:
        question: User question
        context_retrival: List of chunks fed to LLM
        llm_response: LLM's generated response
        llm: LLM instance
    
    Returns:
        Diagnostic report
    """
    report = {
        "context_length": sum(len(chunk['text']) for chunk in context_retrival),
        "num_chunks": len(context_retrival),
        "response_length": len(llm_response),
        "diagnosis": []
    }
    
    # Check context window overflow
    if report["context_length"] > 4000:  # Adjust based on your model
        report["diagnosis"].append("⚠️ Context too long - may cause 'lost in middle' issue")
        report["diagnosis"].append("→ Try: Reduce chunks, use re-ranker, context compression")
    
    # Check faithfulness (simple heuristic)
    faithfulness_prompt = f"""
Check if the Answer is faithful to the Context. Answer ONLY 'yes' or 'no'.

Context: {' '.join([chunk['text'][:200] for chunk in context_retrival])}

Answer: {llm_response}

Is the answer faithful? (yes/no):"""
    
    faithfulness_check = llm.generate(faithfulness_prompt).strip().lower()
    
    if faithfulness_check == 'no':
        report["diagnosis"].append("❌ Faithfulness issue detected - LLM may be hallucinating")
        report["diagnosis"].append("→ Try: Stronger grounding prompt, citation requirement")
    
    # Check for incomplete answer indicators
    incomplete_signals = ['partial', 'some information', 'not complete', 'additional details']
    if any(signal in llm_response.lower() for signal in incomplete_signals):
        report["diagnosis"].append("⚠️ Answer may be incomplete")
        report["diagnosis"].append("→ Try: Increase k, check if answer spans multiple chunks")
    
    if not report["diagnosis"]:
        report["diagnosis"].append("✅ Generation seems okay - verify prompt design")
    
    return report

# Example usage
# report = diagnose_generation(
#     question="What is the refund policy?",
#     context_retrival=chunks,
#     llm_response=response,
#     llm=your_llm_instance
# )
# print(report)

# COMMAND ----------

# DBTITLE 1,Layer 1: Retrieval Debugging
# MAGIC %md
# MAGIC ## 🔍 Layer 1: Retrieval Debugging
# MAGIC
# MAGIC **Agar isolation test se pata chala ki problem RETRIEVAL mein hai**, toh yeh layers check karo:
# MAGIC
# MAGIC ### 1.1 Chunking Issues
# MAGIC **Problem**: Context loss ya fragmented answers
# MAGIC - **Chunk size too small** → Important context break ho jata hai
# MAGIC - **No overlap** → Answer jo 2 chunks ke boundary par hai wo miss ho jata
# MAGIC - **Poor splitting logic** → Sentences/paragraphs beech mein cut ho jate hain
# MAGIC
# MAGIC **Fix**:
# MAGIC - Chunk size: 500-1000 tokens (experiment karo)
# MAGIC - Overlap: 50-100 tokens
# MAGIC - Semantic splitting (sentence boundaries, paragraph breaks)
# MAGIC
# MAGIC ### 1.2 Embedding Issues
# MAGIC **Problem**: Semantic mismatch between query and documents
# MAGIC - **Embedding model outdated** → Domain-specific queries fail
# MAGIC - **Embeddings stale** → New documents add hue lekin re-index nahi hua
# MAGIC - **Query-document asymmetry** → Question style vs document style mismatch
# MAGIC
# MAGIC **Fix**:
# MAGIC - Domain-specific embedding model use karo
# MAGIC - Regular re-indexing after doc updates
# MAGIC - Query reformulation or expansion
# MAGIC
# MAGIC ### 1.3 Search Algorithm Issues
# MAGIC **Problem**: Pure vector search limitations
# MAGIC - **Semantic search alone** → Misses exact keyword matches
# MAGIC - **Low top-k** → Answer hai lekin top results mein nahi
# MAGIC - **No re-ranking** → Relevant chunks retrieve hue lekin rank low hai
# MAGIC
# MAGIC **Fix**:
# MAGIC - **Hybrid search**: Dense (embedding) + Sparse (BM25/keyword)
# MAGIC - Increase k value (retrieve more candidates)
# MAGIC - Add **cross-encoder re-ranker** for final ranking

# COMMAND ----------

# DBTITLE 1,Retrieval Debugging - Diagnostic Code
# Retrieval Layer Diagnostics

# COMMAND ----------

# DBTITLE 1,Cell 6


# COMMAND ----------


# COMMAND ----------

# DBTITLE 1,Layer 2: Generation Debugging
# MAGIC %md
# MAGIC ## ⚡ Layer 2: Generation Debugging
# MAGIC
# MAGIC **Agar isolation test se pata chala ki problem GENERATION mein hai**, toh yeh check karo:
# MAGIC
# MAGIC ### 2.1 Prompt Design Issues
# MAGIC **Problem**: LLM instructions unclear ya insufficient
# MAGIC - **No grounding instruction** → LLM hallucinates beyond context
# MAGIC - **Vague answering style** → LLM adds unnecessary info
# MAGIC - **No example in prompt** → Output format inconsistent
# MAGIC
# MAGIC **Fix**:
# MAGIC - Clear instruction: "Answer ONLY using the context. If answer not found, say 'I don't know'"
# MAGIC - Add few-shot examples
# MAGIC - Specify output format (bullet points, concise answer, etc.)
# MAGIC
# MAGIC ### 2.2 Context Window Issues
# MAGIC **Problem**: "Lost in the Middle" phenomenon
# MAGIC - **Too many chunks** → Important info buried in middle
# MAGIC - **Long context** → LLM attention degrades for middle portions
# MAGIC - **Irrelevant chunks dilute signal** → Wrong info confuses LLM
# MAGIC
# MAGIC **Fix**:
# MAGIC - Limit number of chunks (3-5 most relevant)
# MAGIC - Use re-ranker to pick BEST chunks
# MAGIC - Place most relevant chunks at start and end
# MAGIC - Consider context compression techniques
# MAGIC
# MAGIC ### 2.3 Faithfulness/Groundedness Issues
# MAGIC **Problem**: LLM generates plausible but incorrect info
# MAGIC - **Hallucination** → LLM invents facts not in context
# MAGIC - **Over-generalization** → Combines info from multiple chunks incorrectly
# MAGIC - **Citation missing** → Can't verify which chunk was used
# MAGIC
# MAGIC **Fix**:
# MAGIC - Add citation requirement in prompt
# MAGIC - Use faithfulness checking (compare answer with retrieved chunks)
# MAGIC - Implement answer validation layer
# MAGIC - Consider using smaller, more controllable models
# MAGIC
# MAGIC ### 2.4 Incomplete Answers - Specific Case
# MAGIC **Problem**: Answer multiple chunks mein split hai
# MAGIC - Retrieval ne sirf 1-2 chunks laaye
# MAGIC - Answer fragments different contexts mein hain
# MAGIC
# MAGIC **Fix**:
# MAGIC - Increase k (retrieve more chunks)
# MAGIC - Better chunking with overlap
# MAGIC - Multi-hop retrieval for complex questions

# COMMAND ----------

# DBTITLE 1,Generation Debugging - Code Example
# Generation Layer Diagnostics

# COMMAND ----------





def diagnose_retrieval(question, expected_chunk_id, retrieval_system, top_k=10):
    """
    Diagnose retrieval issues.
    
    Args:
        question: User question
        expected_chunk_id: ID of chunk that contains correct answer
        retrieval_system: Your retrieval system
        top_k: Number of chunks to retrieve
    
    Returns:
        Diagnostic report
    """
    # Retrieve chunks
    context_retrival = retrieval_system.retrieve(question, top_k=top_k)
    
    # Check if expected chunk was retrieved
    retrieved_ids = [chunk['id'] for chunk in context_retrival]
    expected_found = expected_chunk_id in retrieved_ids
    expected_rank = retrieved_ids.index(expected_chunk_id) + 1 if expected_found else None
    
    # Calculate similarity scores
    scores = [chunk['score'] for chunk in context_retrival]
    
    report = {
        "expected_chunk_retrieved": expected_found,
        "expected_chunk_rank": expected_rank,
        "top_k_scores": scores,
        "avg_score": sum(scores) / len(scores),
        "diagnosis": []
    }
    
    # Diagnosis logic
    if not expected_found:
        report["diagnosis"].append("❌ Expected chunk NOT retrieved at all")
        report["diagnosis"].append("→ Try: Increase k, improve chunking, use hybrid search")
    elif expected_rank and expected_rank > 5:
        report["diagnosis"].append(f"⚠️ Expected chunk retrieved but rank is low: {expected_rank}")
        report["diagnosis"].append("→ Try: Add re-ranker (cross-encoder)")
    else:
        report["diagnosis"].append("✅ Retrieval working - check generation layer")
    
    return report

# Example usage
# report = diagnose_retrieval(
#     question="What is the refund policy?",
#     expected_chunk_id="chunk_123",
#     retrieval_system=your_retrieval_system
# )
# print(report)

def full_rag_diagnostic(question, gold_chunk_id, rag_system):
    """
    Complete diagnostic pipeline for RAG system.
    
    Args:
        question: User's question
        gold_chunk_id: ID of chunk with correct answer
        rag_system: Your RAG system instance
    
    Returns:
        Comprehensive diagnostic report
    """
    print("=" * 60)
    print("RAG SYSTEM DIAGNOSTIC REPORT")
    print("=" * 60)
    
    # Step 0: Isolation Test
    print("\n[STEP 0] Isolation Test - Retrieval vs Generation")
    print("-" * 60)
    
    gold_chunk = rag_system.get_chunk_by_id(gold_chunk_id)
    isolation_result = isolation_test(question, gold_chunk['text'], rag_system.llm)
    
    print(f"Gold Chunk Preview: {gold_chunk['text'][:100]}...")
    print(f"LLM Response: {isolation_result['llm_response'][:100]}...")
    
    # Determine if retrieval or generation issue
    is_correct = "no"  # input() not supported in Databricks notebooks
    
    if is_correct:
        print("✅ Generation works with gold chunk → RETRIEVAL ISSUE")
        focus = "generation"
    else:
        print("❌ Generation fails even with gold chunk → GENERATION ISSUE")
        focus = "generation"
    
    # Step 1: Layer-Specific Debugging
    print(f"\n[STEP 1] Debugging {focus.upper()} Layer")
    print("-" * 60)
    
    if focus == "retrieval":
        # Retrieval diagnostics
        retrieved = rag_system.retrieve(question, top_k=10)
        retrieval_report = diagnose_retrieval(
            question, 
            gold_chunk_id, 
            rag_system.retrieval_system
        )
        
        print(f"Expected chunk retrieved: {retrieval_report['expected_chunk_retrieved']}")
        print(f"Rank (if found): {retrieval_report['expected_chunk_rank']}")
        print(f"\nDiagnosis:")
        for d in retrieval_report['diagnosis']:
            print(f"  {d}")
        
        # Calculate metrics
        metrics = calculate_retrieval_metrics(
            [c['id'] for c in retrieved],
            {gold_chunk_id},
            k=10
        )
        print(f"\nRetrieval Metrics:")
        print(f"  Recall@10: {metrics['recall@k']:.2f}")
        print(f"  Hit Rate@10: {metrics['hit_rate@k']}")
        print(f"  MRR: {metrics['mrr']:.2f}")
        
        print(f"\n🔧 Recommended Fixes:")
        print("  1. Implement hybrid search (dense + BM25)")
        print("  2. Add cross-encoder re-ranker")
        print("  3. Tune chunk size (current: ?, try: 700-800)")
        print("  4. Add chunk overlap (try: 100 tokens)")
        
    else:
        # Generation diagnostics
        retrieved = rag_system.retrieve(question, top_k=5)
        response = rag_system.generate(question, retrieved)
        
        generation_report = diagnose_generation(
            question,
            retrieved,
            response,
            rag_system.llm
        )
        
        print(f"Context length: {generation_report['context_length']} chars")
        print(f"Number of chunks: {generation_report['num_chunks']}")
        print(f"\nDiagnosis:")
        for d in generation_report['diagnosis']:
            print(f"  {d}")
        
        print(f"\n🔧 Recommended Fixes:")
        print("  1. Strengthen grounding instruction in prompt")
        print("  2. Limit context to 3-5 best chunks")
        print("  3. Add citation requirement")
        print("  4. Implement faithfulness validation")
        print("  5. Consider using Ragas for ongoing monitoring")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    
    return {
        "focus": focus,
        "isolation_test": isolation_result,
        "layer_report": retrieval_report if focus == "retrieval" else generation_report
    }

# Usage:
class MockRAGSystem:
    """Simple mock RAG system for testing the diagnostic pipeline."""
    def __init__(self, llm, chunks):
        self.llm = llm
        self.chunks = chunks
        self.retrieval_system = self

    def get_chunk_by_id(self, chunk_id):
        for chunk in self.chunks:
            if chunk['id'] == chunk_id:
                return chunk
        return None

    def retrieve(self, question, top_k=10):
        return self.chunks[:top_k]

    def generate(self, question, retrieved_chunks):
        context = ' '.join([c['text'] for c in retrieved_chunks])
        return self.llm.generate(f"Context: {context}\nQuestion: {question}\nAnswer:")

your_rag_system = MockRAGSystem(
    llm=groq_llm,
    chunks=[
        {"id": "chunk_42", "text": "Company refund policy: Full refunds available within 30 days of purchase.", "score": 0.95},
        {"id": "chunk_43", "text": "For returns after 30 days, store credit may be issued at the company's discretion.", "score": 0.85},
    ]
)

report = full_rag_diagnostic(
     question="What is the company refund policy?",
     gold_chunk_id="chunk_42",
     rag_system=your_rag_system
)

print("✅ Complete diagnostic pipeline ready!")
print("Use the functions above to debug your RAG system systematically.")