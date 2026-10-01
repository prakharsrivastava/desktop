# Databricks notebook source
# DBTITLE 1,Problem Statement
# MAGIC %md
# MAGIC # RAG Problem: High Recall but Wrong Answer
# MAGIC
# MAGIC ## 🎯 Scenario
# MAGIC Aapka RAG system ka **Recall@5 = 95%** hai:
# MAGIC - ✅ **Sahi chunk top-5 mein aa raha hai** (Retrieval working!)
# MAGIC - ❌ **Par jawab phir bhi galat hai** (Answer still wrong)
# MAGIC
# MAGIC ## 🔍 Key Insight
# MAGIC Jab **retrieval metrics acche hain** (Recall@k high hai) lekin **answer galat** hai, toh:
# MAGIC
# MAGIC **Problem RETRIEVAL mein NAHI hai → Problem GENERATION layer mein hai**
# MAGIC
# MAGIC ### Why?
# MAGIC - Sahi context mil raha hai (chunk retrieved hai)
# MAGIC - Lekin **LLM us context ko theek se use/ground nahi kar raha**
# MAGIC - Ya **LLM apni training knowledge se answer de raha**, context ignore karke
# MAGIC
# MAGIC ## 📊 Yeh Previous Problem Se Alag Hai
# MAGIC
# MAGIC | Aspect | Previous Problem | This Problem |
# MAGIC |--------|------------------|-------------|
# MAGIC | **Retrieval** | Failed (wrong chunks) | ✅ Working (Recall@5 = 95%) |
# MAGIC | **Generation** | Hallucinated | ❌ Not grounding in context |
# MAGIC | **Primary Issue** | Retrieval layer | Generation layer |
# MAGIC | **Fix Focus** | Hybrid search, re-ranker | Better prompt, context handling |

# COMMAND ----------

# DBTITLE 1,Layer Analysis - Generation Failure
# MAGIC %md
# MAGIC ## Layer: GENERATION Failure
# MAGIC
# MAGIC ### ✅ Retrieval is Working
# MAGIC **Evidence**: Recall@5 = 95%
# MAGIC - Sahi chunk top-5 mein aa hi raha hai
# MAGIC - Search/embedding/ranking sab theek kaam kar rahe hain
# MAGIC - Relevant context LLM ko mil raha hai
# MAGIC
# MAGIC ### ❌ Generation is Failing
# MAGIC **Evidence**: Answer phir bhi galat hai
# MAGIC - LLM ko **sahi context mila par use nahi kiya**
# MAGIC - Ya context ko **misinterpret** kiya
# MAGIC - Ya apni **training knowledge se answer diya** (context ignore karke)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Clear Answer
# MAGIC
# MAGIC > **"Jab Recall@5 = 95% hai (sahi chunk top-5 mein hai) par jawab galat hai, toh problem RETRIEVAL mein nahi, GENERATION layer mein hai. Sahi context mil raha hai, lekin LLM us context ko theek se use/ground nahi kar raha."**

# COMMAND ----------

# DBTITLE 1,Metric - How to Confirm
# MAGIC %md
# MAGIC ## Metric: Confirm/Measure Kaise Kare?
# MAGIC
# MAGIC ### Primary Metric: Faithfulness (Groundedness)
# MAGIC
# MAGIC **Definition**: Answer retrieved context par based hai ya LLM ne hallucinate/ignore kiya?
# MAGIC
# MAGIC **How it works**:
# MAGIC - Har statement ko context ke saath verify karo
# MAGIC - Check: Kya yeh statement context mein hai?
# MAGIC - Score: 0 (completely unfaithful) to 1 (fully grounded)
# MAGIC
# MAGIC **Tool**: **Ragas framework** - Industry-standard RAG evaluation
# MAGIC ```python
# MAGIC from ragas.metrics import faithfulness
# MAGIC
# MAGIC # Ragas automatically checks:
# MAGIC # - Har claim ko context se verify karta hai
# MAGIC # - Hallucination detect karta hai
# MAGIC # - 0-1 score deta hai
# MAGIC ```
# MAGIC
# MAGIC **Is case mein**:
# MAGIC - **Low faithfulness score** → LLM context ignore kar raha hai
# MAGIC - **High recall but low faithfulness** → Clear generation issue
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Secondary Metric: Answer Relevance
# MAGIC
# MAGIC **Definition**: Answer question ko address karta hai ya off-topic hai?
# MAGIC
# MAGIC **Why check this**:
# MAGIC - Answer faithful ho sakta hai (context-based) par question se unrelated
# MAGIC - Or answer relevant lagta hai par actually hallucinated
# MAGIC
# MAGIC **Tool**: Ragas `answer_relevancy` metric
# MAGIC ```python
# MAGIC from ragas.metrics import answer_relevancy
# MAGIC
# MAGIC # Semantic similarity between:
# MAGIC # - Question
# MAGIC # - Generated answer
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Complete Measurement Strategy
# MAGIC
# MAGIC **Step 1**: Check Recall@k → ✅ High (95%)
# MAGIC **Step 2**: Check Faithfulness → ❌ Low (problem identified)
# MAGIC **Step 3**: Check Answer Relevance → Helps understand if LLM went off-topic
# MAGIC
# MAGIC **Diagnosis**:
# MAGIC - **High Recall + Low Faithfulness** = Generation not grounding in context
# MAGIC - **High Recall + High Faithfulness + Wrong answer** = Context interpretation issue
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Model Answer
# MAGIC
# MAGIC > **"Faithfulness (groundedness) metric via Ragas. Yeh batata hai ki answer retrieved context par based hai ya LLM ne hallucinate/ignore kiya. Saath mein answer-relevance bhi dekh sakte ho."**

# COMMAND ----------

# DBTITLE 1,Causes - Why Generation Fails
# MAGIC %md
# MAGIC ## Causes: Generation Kyun Fail Hoti Hai?
# MAGIC
# MAGIC ### Cause 1: Weak Prompt Design
# MAGIC
# MAGIC **Problem**: LLM ko explicitly nahi bataya ki context USE karna hai
# MAGIC
# MAGIC **Symptoms**:
# MAGIC - LLM apni training knowledge se answer deta hai
# MAGIC - Context ko reference nahi karta
# MAGIC - "I know this from training" type answers
# MAGIC
# MAGIC **Example of weak prompt**:
# MAGIC ```
# MAGIC Context: {context}
# MAGIC Question: {question}
# MAGIC Answer:
# MAGIC ```
# MAGIC ↑ Koi grounding instruction nahi hai!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Cause 2: Context Window Overflow / "Lost in the Middle"
# MAGIC
# MAGIC **Problem**: Sahi chunk retrieve hua par **LLM ne dekha nahi**
# MAGIC
# MAGIC **Why it happens**:
# MAGIC - Top-5 chunks mein se sahi chunk **middle mein** hai (position 2-4)
# MAGIC - LLMs have **attention bias** - start aur end pe zyada focus
# MAGIC - Middle content often gets "lost"
# MAGIC - Result: LLM ne pehla/last chunk use kiya jo incomplete/irrelevant tha
# MAGIC
# MAGIC **Research**: "Lost in the Middle" (Liu et al., 2023)
# MAGIC - LLMs perform best with info at start or end
# MAGIC - Middle positions have significantly lower attention
# MAGIC
# MAGIC **Example**:
# MAGIC ```
# MAGIC Chunk 1: Partial info (❌ LLM uses this)
# MAGIC Chunk 2: Wrong info
# MAGIC Chunk 3: ✅ CORRECT ANSWER (ignored!)
# MAGIC Chunk 4: Related but incomplete
# MAGIC Chunk 5: Unrelated (❌ Or LLM uses this)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Cause 3: LLM Using Training Knowledge
# MAGIC
# MAGIC **Problem**: LLM ki training mein similar knowledge hai
# MAGIC
# MAGIC **What happens**:
# MAGIC - LLM apne parameters se answer generate karta hai
# MAGIC - Context ko "suggestion" ki tarah treat karta hai
# MAGIC - Actual context ko override kar deta hai
# MAGIC
# MAGIC **Why it's dangerous**:
# MAGIC - Training data outdated ho sakta hai
# MAGIC - Company-specific info training mein nahi hoga
# MAGIC - Answer confident lagta hai par wrong
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Cause 4: Too Many Chunks (Context Dilution)
# MAGIC
# MAGIC **Problem**: 5 chunks bahut zyada hain
# MAGIC
# MAGIC **Why it causes issues**:
# MAGIC - Signal-to-noise ratio low
# MAGIC - Relevant info dilute ho jata hai
# MAGIC - LLM confused ho jata hai multiple sources se
# MAGIC
# MAGIC **Better approach**: Top 3 BEST chunks (after re-ranking)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Summary of Causes
# MAGIC
# MAGIC 1. **Weak prompt** → No grounding instruction
# MAGIC 2. **Lost in middle** → Right chunk buried in position 2-4
# MAGIC 3. **Training override** → LLM using parametric knowledge
# MAGIC 4. **Too many chunks** → Context dilution, signal loss

# COMMAND ----------

# DBTITLE 1,Fixes - Concrete Solutions
# MAGIC %md
# MAGIC ## Fixes: Behtar Generation Ke Liye
# MAGIC
# MAGIC ### Fix 1: Stronger Grounding Prompt
# MAGIC
# MAGIC **Goal**: Force LLM to use ONLY context
# MAGIC
# MAGIC **Implementation**:
# MAGIC ```python
# MAGIC prompt_template = """
# MAGIC You are a helpful assistant. Answer based ONLY on the context provided.
# MAGIC
# MAGIC **CRITICAL INSTRUCTIONS:**
# MAGIC 1. Use ONLY the information in the context below to answer.
# MAGIC 2. Do NOT use your training knowledge.
# MAGIC 3. If the context doesn't contain the answer, say:
# MAGIC    "I don't have enough information in the provided context."
# MAGIC 4. Cite which part of context you used (e.g., "According to the context...")
# MAGIC 5. If you're unsure, admit it rather than guessing.
# MAGIC
# MAGIC **CONTEXT:**
# MAGIC {context}
# MAGIC
# MAGIC **QUESTION:**
# MAGIC {question}
# MAGIC
# MAGIC **ANSWER (remember: ONLY from context above):**
# MAGIC """
# MAGIC ```
# MAGIC
# MAGIC **Why it works**:
# MAGIC - Explicit instruction to ignore training
# MAGIC - Citation requirement forces reference
# MAGIC - "Admit uncertainty" prevents confident wrong answers
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Fix 2: Context Order Optimization (Fix "Lost in Middle")
# MAGIC
# MAGIC **Goal**: Place most relevant chunks at START and END positions
# MAGIC
# MAGIC **Strategy**: After retrieval, reorder chunks
# MAGIC
# MAGIC **Implementation**:
# MAGIC ```python
# MAGIC def optimize_context_order(chunks, relevance_scores):
# MAGIC     """
# MAGIC     Place most relevant chunks at start and end.
# MAGIC     Avoid burying important info in the middle.
# MAGIC     """
# MAGIC     # Sort by relevance
# MAGIC     sorted_chunks = sorted(
# MAGIC         zip(chunks, relevance_scores), 
# MAGIC         key=lambda x: x[1], 
# MAGIC         reverse=True
# MAGIC     )
# MAGIC     
# MAGIC     if len(sorted_chunks) <= 2:
# MAGIC         return [c for c, _ in sorted_chunks]
# MAGIC     
# MAGIC     # Reorder: best at start, second-best at end, rest in middle
# MAGIC     best = sorted_chunks[0][0]
# MAGIC     second_best = sorted_chunks[1][0]
# MAGIC     middle = [c for c, _ in sorted_chunks[2:]]
# MAGIC     
# MAGIC     return [best] + middle + [second_best]
# MAGIC ```
# MAGIC
# MAGIC **Why it works**:
# MAGIC - LLM attention peaks at start and end
# MAGIC - Most relevant info now in high-attention zones
# MAGIC - Middle can have less important supporting context
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Fix 3: Reduce Number of Chunks
# MAGIC
# MAGIC **Goal**: Better signal-to-noise ratio
# MAGIC
# MAGIC **Strategy**: Top-3 instead of top-5
# MAGIC
# MAGIC **Implementation**:
# MAGIC ```python
# MAGIC # After retrieval + re-ranking:
# MAGIC top_chunks = reranked_results[:3]  # Not [:5]
# MAGIC
# MAGIC # Why 3?
# MAGIC # - Less noise
# MAGIC # - Easier for LLM to focus
# MAGIC # - Reduces "lost in middle" effect
# MAGIC # - Faster processing
# MAGIC ```
# MAGIC
# MAGIC **Trade-off**:
# MAGIC - ✅ Better precision
# MAGIC - ⚠️ Slightly lower recall (but if Recall@3 is still 85%+, worth it)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Fix 4: Better Model or Finetuning
# MAGIC
# MAGIC **Option A**: Use a better base model
# MAGIC - Models with longer context windows
# MAGIC - Models trained specifically for RAG (e.g., Anthropic Claude with "extended context")
# MAGIC - Open-source: Mixtral, Llama-3 with RAG tuning
# MAGIC
# MAGIC **Option B**: Finetune on your domain
# MAGIC ```python
# MAGIC # Finetune dataset format:
# MAGIC {
# MAGIC     "context": "<retrieved chunks>",
# MAGIC     "question": "<user question>",
# MAGIC     "answer": "<ground truth answer>",
# MAGIC     "instruction": "Answer ONLY from context"
# MAGIC }
# MAGIC
# MAGIC # Finetune to teach:
# MAGIC # - Grounding behavior
# MAGIC # - Citation patterns
# MAGIC # - Uncertainty admission
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Priority Order
# MAGIC
# MAGIC 1. **Immediate fix**: Stronger prompt (Fix 1) - Zero cost, instant impact
# MAGIC 2. **High impact**: Context order optimization (Fix 2) - Low effort, big improvement
# MAGIC 3. **Refinement**: Reduce chunks to 3 (Fix 3) - Test and validate
# MAGIC 4. **Long-term**: Better model or finetuning (Fix 4) - Requires resources
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Model Answer
# MAGIC
# MAGIC > **"Fix: Behtar prompt ('answer ONLY from context'), context order/size tune karna (top chunks ko start/end pe rakho, middle se hatao), aur agar zaroori ho toh behtar model ya finetuning."**

# COMMAND ----------

# MAGIC     %pip install ragas

# COMMAND ----------

# Measure Faithfulness using Ragas
# Note: ragas has dependency conflicts on Python 3.12
# Manual faithfulness check is available as fallback

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from datasets import Dataset
    RAGAS_AVAILABLE = True
    print("✅ Ragas loaded successfully!")
except Exception as e:
    print(f"⚠️ Ragas not available on Python 3.12: {type(e).__name__}")
    print("\nSolution: Use Python 3.13+ runtime OR use the manual faithfulness check.")
    print("Manual check is available in the measure_generation_quality() function below.")
    RAGAS_AVAILABLE = False
    
    # Import Dataset separately for creating datasets
    try:
        from datasets import Dataset
    except:
        pass

# COMMAND ----------

from datasets import Dataset
# Example data for faithfulness evaluation
example_questions = [
    "What is the company's refund policy?"
]

example_contexts = [
    [
        "Our customer service team is available 24/7 to help you.",
        "We accept all major credit cards and PayPal for payments.",
        "Full refunds are available within 30 days of purchase with proof of receipt. After 30 days, store credit only.",  # <- CORRECT INFO
        "Shipping typically takes 3-5 business days.",
        "Products come with a 1-year manufacturer warranty."
    ]
]

example_answers = [
    "The company offers full refunds within 90 days of purchase."  # <- WRONG! (Not grounded in context)
]

example_ground_truths = [
    "Full refunds within 30 days with receipt, store credit after 30 days."
]

data = {
            "question": example_questions,
            "answer": example_answers,
            "contexts": example_contexts,
            "ground_truth": example_ground_truths
        }
        
dataset = Dataset.from_dict(data)
display(dataset[0])

# COMMAND ----------

dataset[0]['answer']

# COMMAND ----------

dataset[0]['contexts']

# COMMAND ----------

# Use the manual evaluation from measure_generation_quality function
print("📝 Manual Faithfulness Evaluation\n")
print(f"Question: {dataset[0]['question']}")
print(f"Generated Answer: {dataset[0]['answer']}")
print(f"Ground Truth: {dataset[0]['ground_truth']}")
print("\nAnalysis:")

# Check if answer keywords appear in context
answer = dataset[0]['answer']
contexts = dataset[0]['contexts']
answer_words = set(answer.lower().split())
context_text = " ".join(contexts).lower()

found_in_context = sum(1 for word in answer_words if word in context_text)
coverage = found_in_context / len(answer_words) if answer_words else 0

print(f"  • Word coverage in context: {coverage:.1%}")

# Check for the specific wrong information (90 days vs 30 days)
if "90" in answer and "30" in context_text:
    print("  • ⚠️ HALLUCINATION DETECTED: Answer says '90 days' but context says '30 days'")
    print("  • 🚨 Faithfulness Issue: LLM is NOT grounding in context!")
elif coverage < 0.5:
    print("  • ⚠️ Low overlap - possible hallucination")
else:
    print("  • ✅ Good overlap with context")

print("\n💡 This demonstrates the generation problem: retrieval succeeded, but LLM")
print("   failed to ground its answer in the provided context.")

# COMMAND ----------


result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy
    ]
)
print("Evaluation Results (Ragas):")
print(f"Faithfulness: {result['faithfulness']:.2f}")
print(f"Answer Relevancy: {result['answer_relevancy']:.2f}")

# COMMAND ----------

# MAGIC %pip uninstall ragas langchain pydantic -y
# MAGIC %pip install ragas

# COMMAND ----------

# MAGIC %pip install -U ragas langchain langchain-community langchain-core
# MAGIC %pip uninstall ragas langchain langchain-community langchain-core -y
# MAGIC
# MAGIC %pip install ragas

# COMMAND ----------

# MAGIC %pip install --upgrade ragas

# COMMAND ----------

# MAGIC %pip install -U ragas

# COMMAND ----------

# MAGIC %pip uninstall ragas langchain langchain-community langchain-core -y
# MAGIC
# MAGIC %pip install ragas

# COMMAND ----------

import ragas
print("ragas:", ragas.__version__)


# COMMAND ----------

import sys
print(sys.version)

try:
    import ragas
    print("ragas:", ragas.__version__)

    import langchain
    print("langchain:", langchain.__version__)

    import langchain_community
    print("langchain_community:", langchain_community.__version__)
except ModuleNotFoundError as e:
    print(f"Missing module: {e}")
    print("\nInstall missing dependencies:")
    print("  %pip install langchain-google-vertexai")
    print("Then re-run this cell.")

# COMMAND ----------



def measure_generation_quality(questions, contexts, answers, ground_truths):
    """
    Measure if generation is grounding in context.
    Uses Ragas if available, otherwise provides manual analysis.
    
    Args:
        questions: List of user questions
        contexts: List of retrieved context (list of chunks per question)
        answers: List of generated answers
        ground_truths: List of correct answers (for reference)
    
    Returns:
        Faithfulness and relevance scores (or manual analysis)
    """
    
    if RAGAS_AVAILABLE:
        # Use Ragas for automated evaluation
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        
        dataset = Dataset.from_dict(data)
        
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy
            ]
        )
        
        print("Evaluation Results (Ragas):")
        print(f"Faithfulness: {result['faithfulness']:.2f}")
        print(f"Answer Relevancy: {result['answer_relevancy']:.2f}")
        
        if result['faithfulness'] < 0.7:
            print("\n⚠️ LOW FAITHFULNESS - LLM not grounding in context!")
            print("Fix: Strengthen prompt with grounding instructions")
        
        if result['answer_relevancy'] < 0.8:
            print("\n⚠️ LOW RELEVANCE - LLM going off-topic!")
            print("Fix: Better prompt design, question understanding")
        
        return result
    
    else:
        # Manual faithfulness check
        print("📝 Manual Faithfulness Analysis (Ragas not available):\n")
        
        for i, (question, context_list, answer, ground_truth) in enumerate(
            zip(questions, contexts, answers, ground_truths), 1
        ):
            print(f"Question {i}: {question}")
            print(f"Generated Answer: {answer}")
            print(f"Ground Truth: {ground_truth}")
            print(f"\nContext Analysis:")
            
            # Check if answer keywords appear in context
            answer_words = set(answer.lower().split())
            context_text = " ".join(context_list).lower()
            
            found_in_context = sum(1 for word in answer_words if word in context_text)
            coverage = found_in_context / len(answer_words) if answer_words else 0
            
            print(f"  - Answer word coverage in context: {coverage:.1%}")
            
            # Simple heuristic faithfulness
            if coverage < 0.5:
                print("  - ⚠️ LOW overlap - possible hallucination or external knowledge")
            else:
                print("  - ✅ Good overlap with context")
            
            print("\n" + "="*60 + "\n")
        
        print("\n💡 Recommendation: Install Ragas for proper faithfulness scoring")
        print("   Note: Currently has dependency conflicts on this runtime")
        
        return {"manual_analysis": True, "ragas_available": False}

# COMMAND ----------

# DBTITLE 1,Example - Measure Faithfulness with Dummy Data
# Example: Measure faithfulness with dummy data
# Scenario: High recall (correct chunk retrieved) but LLM gives wrong answer

# Dummy example data
example_questions = [
    "What is the company's refund policy?"
]

example_contexts = [
    [
        "Our customer service team is available 24/7 to help you.",
        "We accept all major credit cards and PayPal for payments.",
        "Full refunds are available within 30 days of purchase with proof of receipt. After 30 days, store credit only.",  # <- CORRECT INFO
        "Shipping typically takes 3-5 business days.",
        "Products come with a 1-year manufacturer warranty."
    ]
]

example_answers = [
    "The company offers full refunds within 90 days of purchase."  # <- WRONG! (Not grounded in context)
]

example_ground_truths = [
    "Full refunds within 30 days with receipt, store credit after 30 days."
]

print("📊 Running faithfulness evaluation...\n")
print("Question:", example_questions[0])
print("\nGenerated Answer:", example_answers[0])
print("\nGround Truth:", example_ground_truths[0])
print("\n" + "="*60)

# Measure generation quality
result = measure_generation_quality(
    questions=example_questions,
    contexts=example_contexts,
    answers=example_answers,
    ground_truths=example_ground_truths
)

print("\n" + "="*60)
print("\n💡 Interpretation:")
print("This demonstrates the problem: Context had correct info (30 days)")
print("but LLM generated wrong answer (90 days) - not grounded in context!")

# COMMAND ----------

# DBTITLE 1,Code - Faithfulness Check with Ragas


# Example usage:
"""
result = measure_generation_quality(
    questions=["What is the refund policy?"],
    contexts=[["Chunk 1: ...", "Chunk 2: ...", "Chunk 3: correct info"]],
    answers=["Our refund policy is..."],
    ground_truths=["Full refunds within 30 days"]
)
"""

print("✅ Ragas faithfulness measurement ready!")
print("Install: pip install ragas")

# COMMAND ----------

# DBTITLE 1,Code - Stronger Grounding Prompt
# Fix 1: Stronger Grounding Prompt Implementation

def generate_with_strong_grounding(query, retrieved_chunks, llm):
    """
    Generate answer with strong grounding instructions.
    Force LLM to use ONLY context and admit when uncertain.
    
    Args:
        query: User's question
        retrieved_chunks: List of retrieved chunks
        llm: Your LLM instance
    
    Returns:
        Generated answer with grounding
    """
    
    # Build context with chunk numbering
    context = "\n\n".join([
        f"[Context {i+1}]\n{chunk['text']}" 
        for i, chunk in enumerate(retrieved_chunks)
    ])
    
    # Strong grounding prompt
    prompt = f"""
You are a helpful assistant. Answer based ONLY on the context provided below.

**CRITICAL INSTRUCTIONS:**
1. Use ONLY the information in the context below to answer.
2. Do NOT use your training knowledge or external information.
3. If the context doesn't contain enough information to answer, respond EXACTLY:
   "I don't have enough information in the provided context to answer this question."
4. Cite which context section you used (e.g., "According to Context 2, ...")
5. If you're unsure or the information is ambiguous, admit it rather than guessing.
6. Quote relevant parts of the context when possible.

**CONTEXT:**
{context}

**QUESTION:**
{query}

**ANSWER (remember: ONLY from context above, with citations):**
"""
    
    # Generate response
    response = llm.generate(prompt)
    
    return {
        "answer": response,
        "prompt_used": "strong_grounding",
        "num_chunks": len(retrieved_chunks)
    }

# Example usage:
"""
result = generate_with_strong_grounding(
    query="What is the refund policy?",
    retrieved_chunks=[
        {'text': 'Company policy document...'},
        {'text': 'Refund guidelines...'},
        {'text': 'Customer service info...'}
    ],
    llm=your_llm_instance
)

print(result['answer'])
"""

print("✅ Strong grounding prompt ready!")

# COMMAND ----------

# Dummy LLM for demonstration (replace with OpenAI / HF model)
class DummyLLM:
    def generate(self, prompt):
        return (
            "According to Context 3, full refunds are available within 30 days of purchase with proof of receipt. "
            "After 30 days, store credit is provided."
        )

llm = DummyLLM()

# Example retrieved chunks
retrieved_chunks = [
    {"text": "Our customer service team is available 24/7."},
    {"text": "We accept all major payment methods."},
    {"text": "Full refunds are available within 30 days of purchase with proof of receipt. After 30 days, store credit only."}
]

# Call your function
result = generate_with_strong_grounding(
    query="What is the company's refund policy?",
    retrieved_chunks=retrieved_chunks,
    llm=llm
)

# Print output
print("Generated Answer:\n")
print(result["answer"])

print("\nMetadata:")
print(result)

# COMMAND ----------

# DBTITLE 1,Code - Context Order Optimization
# Fix 2: Context Order Optimization (Avoid "Lost in Middle")

def optimize_context_order(chunks, relevance_scores):
    """
    Place most relevant chunks at START and END positions.
    Avoid burying important information in the middle.
    
    Research: "Lost in the Middle" (Liu et al., 2023)
    LLMs have attention bias towards start and end positions.
    
    Args:
        chunks: List of retrieved chunks
        relevance_scores: List of relevance scores (from re-ranker)
    
    Returns:
        Reordered chunks with best at start/end
    """
    
    # Sort chunks by relevance score
    sorted_items = sorted(
        zip(chunks, relevance_scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Edge cases
    if len(sorted_items) <= 2:
        return [chunk for chunk, _ in sorted_items]
    
    # Reorder strategy:
    # - Best chunk: Position 1 (start)
    # - Second-best: Last position (end)
    # - Rest: Middle positions (lower priority)
    
    best_chunk = sorted_items[0][0]
    second_best_chunk = sorted_items[1][0]
    middle_chunks = [chunk for chunk, _ in sorted_items[2:]]
    
    # Final order: [BEST] + [middle chunks] + [SECOND-BEST]
    optimized_order = [best_chunk] + middle_chunks + [second_best_chunk]
    
    return optimized_order

# Example usage:
"""
chunks = [
    {'id': 1, 'text': 'Chunk A'},
    {'id': 2, 'text': 'Chunk B - MOST RELEVANT'},
    {'id': 3, 'text': 'Chunk C - SECOND MOST RELEVANT'},
    {'id': 4, 'text': 'Chunk D'},
    {'id': 5, 'text': 'Chunk E'}
]

relevance_scores = [0.5, 0.95, 0.85, 0.6, 0.4]

optimized = optimize_context_order(chunks, relevance_scores)

# Result order:
# Position 1: Chunk B (best) - HIGH ATTENTION
# Position 2-4: Chunks A, D, E (middle)
# Position 5: Chunk C (second-best) - HIGH ATTENTION

print("Optimized chunk order:")
for i, chunk in enumerate(optimized, 1):
    print(f"  Position {i}: {chunk['text']}")
"""

print("✅ Context order optimization ready!")
print("Tip: Use this AFTER retrieval and re-ranking, BEFORE sending to LLM")

# COMMAND ----------

# DBTITLE 1,Code - Complete Generation Pipeline
# Complete Generation Pipeline with All Fixes

def generate_with_optimizations(query, retrieved_chunks, relevance_scores, llm, final_k=3):
    """
    Complete generation pipeline with:
    1. Reduce to top-k chunks
    2. Optimize context order
    3. Strong grounding prompt
    
    Args:
        query: User's question
        retrieved_chunks: List of retrieved chunks (e.g., top-5 from retrieval)
        relevance_scores: Relevance scores from re-ranker
        llm: Your LLM instance
        final_k: Final number of chunks to use (default: 3)
    
    Returns:
        Generated answer with all optimizations applied
    """
    
    # Step 1: Reduce to top-k (less noise)
    print(f"Input: {len(retrieved_chunks)} chunks")
    sorted_items = sorted(
        zip(retrieved_chunks, relevance_scores),
        key=lambda x: x[1],
        reverse=True
    )
    top_k_items = sorted_items[:final_k]
    top_k_chunks = [chunk for chunk, _ in top_k_items]
    top_k_scores = [score for _, score in top_k_items]
    print(f"After reduction: {len(top_k_chunks)} chunks")
    
    # Step 2: Optimize context order (fix "lost in middle")
    optimized_chunks = optimize_context_order(top_k_chunks, top_k_scores)
    print("Context order optimized")
    
    # Step 3: Build context with chunk markers
    context = "\n\n".join([
        f"[Context {i+1}]\n{chunk['text']}"
        for i, chunk in enumerate(optimized_chunks)
    ])
    
    # Step 4: Strong grounding prompt
    prompt = f"""
You are a helpful assistant. Answer based ONLY on the context provided below.

**CRITICAL INSTRUCTIONS:**
1. Use ONLY the information in the context below to answer.
2. Do NOT use your training knowledge or make assumptions.
3. If the context doesn't contain the answer, say: "I don't have enough information."
4. Cite which context section you used (e.g., "According to Context 2, ...")
5. If unsure, admit it rather than guessing.

**CONTEXT:**
{context}

**QUESTION:**
{query}

**ANSWER (ONLY from context, with citations):**
"""
    
    # Step 5: Generate
    print("Generating answer...")
    response = llm.generate(prompt)
    
    return {
        "answer": response,
        "num_input_chunks": len(retrieved_chunks),
        "num_used_chunks": len(optimized_chunks),
        "optimizations_applied": [
            "chunk_reduction",
            "context_order_optimization",
            "strong_grounding_prompt"
        ]
    }

# Example usage:
"""
result = generate_with_optimizations(
    query="What is the refund policy?",
    retrieved_chunks=top_5_chunks_from_retrieval,
    relevance_scores=reranker_scores,
    llm=your_llm_instance,
    final_k=3
)

print("Answer:", result['answer'])
print(f"Used {result['num_used_chunks']} out of {result['num_input_chunks']} chunks")
"""

print("✅ Complete optimized generation pipeline ready!")
print("This combines all fixes: reduction, ordering, and strong grounding.")

# COMMAND ----------

# DBTITLE 1,Summary - When to Use This Solution
# MAGIC %md
# MAGIC ## 📋 Complete Solution Summary
# MAGIC
# MAGIC ### Problem Recap
# MAGIC - **Symptom**: High Recall@5 (95%) but wrong answers
# MAGIC - **Root cause**: Generation not grounding in retrieved context
# MAGIC - **Layer**: GENERATION failure (retrieval is working fine)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Diagnostic Approach
# MAGIC
# MAGIC **Step 1**: Measure Recall@k
# MAGIC - ✅ High (95%) → Retrieval working
# MAGIC
# MAGIC **Step 2**: Measure Faithfulness (Ragas)
# MAGIC - ❌ Low → Generation not grounding in context
# MAGIC - This confirms: Generation issue, not retrieval
# MAGIC
# MAGIC **Step 3**: Identify Root Cause
# MAGIC - Weak prompt design?
# MAGIC - Lost in the middle?
# MAGIC - Training knowledge override?
# MAGIC - Too many chunks?
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Solution Framework
# MAGIC
# MAGIC #### Fix Priority (Implement in Order)
# MAGIC
# MAGIC **1. Immediate (Zero Cost)**
# MAGIC - ✅ Stronger grounding prompt
# MAGIC - Add explicit "ONLY from context" instruction
# MAGIC - Add citation requirement
# MAGIC - Add uncertainty admission
# MAGIC
# MAGIC **2. High Impact (Low Effort)**
# MAGIC - ✅ Context order optimization
# MAGIC - Place best chunks at start/end
# MAGIC - Avoid "lost in middle" effect
# MAGIC
# MAGIC **3. Refinement**
# MAGIC - ✅ Reduce chunks (5 → 3)
# MAGIC - Better signal-to-noise ratio
# MAGIC - Test impact on Recall@3
# MAGIC
# MAGIC **4. Long-term (Resource Intensive)**
# MAGIC - Consider better base model
# MAGIC - Or finetune on domain data
# MAGIC - Teach grounding behavior
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Key Metrics to Monitor
# MAGIC
# MAGIC | Metric | What It Measures | Target |
# MAGIC |--------|------------------|--------|
# MAGIC | Recall@k | Retrieval quality | >85% |
# MAGIC | Faithfulness | Grounding in context | >0.8 |
# MAGIC | Answer Relevance | On-topic answers | >0.85 |
# MAGIC | Context Precision | Noise in retrieval | >0.7 |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 When to Use This Solution
# MAGIC
# MAGIC **Use this approach when:**
# MAGIC - ✅ Recall@k is HIGH (>85%)
# MAGIC - ❌ But answers are still wrong
# MAGIC - ❌ Faithfulness score is LOW (<0.7)
# MAGIC
# MAGIC **Don't use this when:**
# MAGIC - ❌ Recall@k is LOW (<70%) → Fix retrieval first (see other notebook)
# MAGIC - ✅ Both Recall and Faithfulness are high → Problem might be in ground truth or evaluation
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Interview Model Answer
# MAGIC
# MAGIC **Layer**: 
# MAGIC > Jab Recall@5 = 95% hai par jawab galat hai, toh problem GENERATION mein hai, RETRIEVAL mein nahi. Sahi context mil raha hai, lekin LLM theek se ground nahi kar raha.
# MAGIC
# MAGIC **Metric**:
# MAGIC > Faithfulness (groundedness) via Ragas. Yeh batata hai ki answer context par based hai ya LLM ne ignore/hallucinate kiya.
# MAGIC
# MAGIC **Causes**:
# MAGIC > (1) Weak prompt - no grounding instruction, (2) Lost in middle - sahi chunk middle position pe buried, (3) Training knowledge override, (4) Too many chunks causing dilution.
# MAGIC
# MAGIC **Fix**:
# MAGIC > Behtar prompt ('answer ONLY from context' + citations), context order tune karna (best chunks start/end pe), chunks reduce karna (3-5), aur agar zaroori ho toh better model.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔄 Implementation Checklist
# MAGIC
# MAGIC - [ ] Measure baseline Recall@k and Faithfulness
# MAGIC - [ ] Implement strong grounding prompt
# MAGIC - [ ] Add context order optimization
# MAGIC - [ ] Test with reduced chunk count (3 instead of 5)
# MAGIC - [ ] Measure post-fix Faithfulness score
# MAGIC - [ ] A/B test in production
# MAGIC - [ ] Monitor ongoing metrics (Ragas)
# MAGIC - [ ] Consider model upgrade if fixes insufficient