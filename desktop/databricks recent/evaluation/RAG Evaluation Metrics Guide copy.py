# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # RAG Evaluation Metrics: Complete Guide
# MAGIC
# MAGIC A comprehensive guide to evaluating Retrieval-Augmented Generation (RAG) systems using multiple metrics.

# COMMAND ----------

# DBTITLE 1,Overview
# MAGIC %md
# MAGIC ## Why Evaluation Metrics Matter for RAG Systems
# MAGIC
# MAGIC Evaluating RAG systems is crucial because they combine retrieval and generation components. Traditional NLG metrics may not capture:
# MAGIC * **Faithfulness**: Does the answer stay true to retrieved context?
# MAGIC * **Relevance**: Is the answer relevant to the question?
# MAGIC * **Context Quality**: Are the retrieved documents appropriate?
# MAGIC * **Semantic Similarity**: Does the answer convey the same meaning?
# MAGIC
# MAGIC ## Metrics Covered
# MAGIC
# MAGIC This notebook demonstrates five evaluation approaches:
# MAGIC
# MAGIC 1. **RAGAS** - RAG-specific metrics (faithfulness, answer relevancy, context recall/precision)
# MAGIC 2. **ROUGE** - N-gram overlap scores (ROUGE-1, ROUGE-2, ROUGE-L)
# MAGIC 3. **BLEU** - Modified precision for n-grams
# MAGIC 4. **BERTScore** - Semantic similarity using contextual embeddings
# MAGIC 5. **METEOR** - Unigram matching with synonyms and paraphrases
# MAGIC
# MAGIC Each metric has specific use cases, strengths, and limitations.

# COMMAND ----------

# DBTITLE 1,Install Required Packages
# Install compatible versions
%pip install -U -q ragas rouge-score nltk bert-score evaluate datasets transformers langchain langchain-core langchain-community langchain-openai openai
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Libraries and Setup
# Core libraries
import pandas as pd
import numpy as np
from typing import List, Dict

# RAGAS
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
    from datasets import Dataset
except ImportError:
    print("⚠️ RAGAS not available. Run the installation cell first.")

# ROUGE
from rouge_score import rouge_scorer

# BLEU
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# BERTScore
import bert_score

# METEOR
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
from nltk.translate.meteor_score import meteor_score as nltk_meteor_score
from nltk.tokenize import word_tokenize

print("✅ All libraries imported successfully!")

# COMMAND ----------

# DBTITLE 1,Create Sample Data
# Sample data for RAG evaluation
# In a real scenario, these would come from your RAG system

questions = [
    "What is machine learning?",
    "How does a neural network work?",
    "What is the difference between supervised and unsupervised learning?",
    "Explain gradient descent algorithm"
]

# Ground truth answers (reference/expected answers)
ground_truth = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.",
    "A neural network works by processing information through interconnected layers of nodes (neurons). Each connection has a weight that gets adjusted during training. The network learns by adjusting these weights to minimize the difference between predicted and actual outputs.",
    "Supervised learning uses labeled data where the correct output is known, and the model learns to map inputs to outputs. Unsupervised learning works with unlabeled data and finds patterns or structures without predefined labels.",
    "Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss function. It calculates the gradient (derivative) of the loss with respect to parameters and moves in the opposite direction to reduce error."
]

# RAG system generated answers (what your system actually produced)
generated_answers = [
    "Machine learning is a branch of AI that allows computers to learn from data without explicit programming. It uses algorithms to identify patterns and make decisions based on data.",
    "Neural networks consist of layers of interconnected neurons. Each neuron receives input, applies weights and an activation function, then passes output to the next layer. The network learns by adjusting weights through backpropagation.",
    "Supervised learning requires labeled training data with known outputs, while unsupervised learning discovers patterns in unlabeled data without predefined categories or outcomes.",
    "Gradient descent minimizes a cost function by computing gradients and updating parameters in the direction that reduces the error. It uses the learning rate to control step size during optimization."
]

# Retrieved contexts (documents retrieved by the RAG system)
contexts = [
    ["Machine learning (ML) is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so."],
    ["Artificial neural networks are computing systems inspired by biological neural networks. They consist of interconnected groups of artificial neurons and process information using a connectionist approach. Modern neural networks use backpropagation algorithm to adjust the weights between neurons based on the error of the output."],
    ["In supervised learning, the algorithm learns from labeled training data and makes predictions based on that learning. Unsupervised learning algorithms work with unlabeled data, trying to find hidden patterns or intrinsic structures in the input data without guidance."],
    ["Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. To find a local minimum, the algorithm takes steps proportional to the negative of the gradient of the function at the current point."]
]

print(f"✅ Created {len(questions)} sample examples")
print(f"   - Questions: {len(questions)}")
print(f"   - Ground truth answers: {len(ground_truth)}")
print(f"   - Generated answers: {len(generated_answers)}")
print(f"   - Retrieved contexts: {len(contexts)}")

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,RAGAS Dataset Preparation
# RAGAS: RAG Assessment Scores
# Specialized metrics designed specifically for RAG systems

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
    from datasets import Dataset
    
    # Prepare data in RAGAS format
    ragas_data = {
        "question": questions,
        "answer": generated_answers,
        "contexts": contexts,
        "ground_truth": ground_truth
    }
    
    ragas_dataset = Dataset.from_dict(ragas_data)
    
    print("✅ RAGAS dataset created successfully!")
    print(f"\nDataset contains {len(ragas_dataset)} examples")
    print(f"\nDataset features: {ragas_dataset.features}")
    print("\n📋 Sample data:")
    print(ragas_dataset)
    
except Exception as e:
    print(f"⚠️ Failed to create RAGAS dataset: {e}")
    print("\nMake sure RAGAS is installed by running Cell 3.")
    print("You may need to restart Python after installation.")

# COMMAND ----------

# DBTITLE 1,RAGAS Evaluation

    
    # Evaluate with RAGAS metrics
    print("📡 Running RAGAS evaluation (this may take a few minutes)...")
    print("Note: RAGAS requires OpenAI API access. Set OPENAI_API_KEY environment variable.\n")
    
    # Note: This requires OpenAI API key to be set
    # import os
    # os.environ["OPENAI_API_KEY"] = "your-api-key"
    
    result = evaluate(
        ragas_dataset,
        metrics=[
            faithfulness,        # How factually accurate is the answer given the context?
            answer_relevancy,    # How relevant is the answer to the question?
            context_recall,      # How much of the ground truth is captured in the contexts?
            context_precision,   # How relevant are the retrieved contexts?
        ]
    )
    
    # Display results
    ragas_df = result.to_pandas()
    
    print("\n📋 RAGAS Evaluation Results:")
    print("=" * 80)
    
    # Show per-question scores
    display(ragas_df[['question', 'faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']].head())
    
    print("\n📊 Average Scores:")
    print("-" * 80)
    for metric in ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']:
        if metric in ragas_df.columns:
            avg_score = ragas_df[metric].mean()
            print(f"{metric:20s}: {avg_score:.3f}")
    
    print("\n📖 Metric Definitions:")
    print("-" * 80)
    print("  • Faithfulness: Measures if the answer is factually consistent with the context (0-1)")
    print("  • Answer Relevancy: Measures how well the answer addresses the question (0-1)")
    print("  • Context Recall: Measures how much of ground truth is present in contexts (0-1)")
    print("  • Context Precision: Measures relevance of retrieved contexts (0-1)")
    
except Exception as e:
    print(f"⚠️ RAGAS evaluation failed: {e}")
    print("\nCommon reasons:")
    print("  1. OPENAI_API_KEY not set")
    print("  2. Network connectivity issues")
    print("  3. RAGAS version compatibility")
    print("\nTo run RAGAS, set your OpenAI API key:")
    print("  import os")
    print("  os.environ['OPENAI_API_KEY'] = 'your-key-here'")

# COMMAND ----------

# DBTITLE 1,RAGAS Explanation
# MAGIC %md
# MAGIC ## RAGAS (RAG Assessment Scores)
# MAGIC
# MAGIC ### 🎯 When to Use
# MAGIC * **RAG-specific evaluation**: Designed specifically for retrieval-augmented generation systems
# MAGIC * **End-to-end assessment**: Evaluates both retrieval and generation quality
# MAGIC * **Production monitoring**: Track RAG system performance over time
# MAGIC * **Component-level debugging**: Identify whether issues are in retrieval or generation
# MAGIC
# MAGIC ### 📊 What It Measures
# MAGIC
# MAGIC **Faithfulness (Generation Quality)**
# MAGIC * Checks if generated answer is factually grounded in the retrieved context
# MAGIC * Uses LLM to verify consistency between answer and context
# MAGIC * Scores range from 0 (completely unfaithful) to 1 (fully faithful)
# MAGIC
# MAGIC **Answer Relevancy (Answer Quality)**
# MAGIC * Measures how well the answer addresses the original question
# MAGIC * Penalizes answers with incomplete or irrelevant information
# MAGIC * Higher scores mean better question-answer alignment
# MAGIC
# MAGIC **Context Recall (Retrieval Recall)**
# MAGIC * Measures how much of the ground truth answer can be derived from retrieved contexts
# MAGIC * Evaluates if retrieval captured all necessary information
# MAGIC * Low scores indicate retrieval is missing important documents
# MAGIC
# MAGIC **Context Precision (Retrieval Precision)**
# MAGIC * Measures how relevant the retrieved contexts are
# MAGIC * Penalizes irrelevant or noisy documents in the context
# MAGIC * Higher scores mean better retrieval quality
# MAGIC
# MAGIC ### ⚠️ Drawbacks
# MAGIC
# MAGIC 1. **Requires Ground Truth**: Needs reference answers for context recall
# MAGIC 2. **Computationally Expensive**: Uses LLM calls for evaluation (time + cost)
# MAGIC 3. **Requires LLM Access**: Needs OpenAI API or similar (additional dependency)
# MAGIC 4. **Non-Deterministic**: LLM-based evaluation can vary between runs
# MAGIC 5. **Latency**: Much slower than traditional metrics due to LLM inference
# MAGIC 6. **Cost**: Can be expensive for large-scale evaluations
# MAGIC 7. **Black Box**: Hard to debug why a specific score was given
# MAGIC
# MAGIC ### ✅ Best For
# MAGIC * RAG system evaluation and monitoring
# MAGIC * A/B testing different retrieval strategies
# MAGIC * Debugging retrieval vs generation issues
# MAGIC * Production quality assurance

# COMMAND ----------

# DBTITLE 1,ROUGE Score Evaluation
# ROUGE: Recall-Oriented Understudy for Gisting Evaluation
# Measures n-gram overlap between generated and reference text

from rouge_score import rouge_scorer

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

print("📋 ROUGE Score Evaluation")
print("=" * 80)
print("\nROUGE measures n-gram overlap between generated and reference answers.")
print("  • ROUGE-1: Unigram (single word) overlap")
print("  • ROUGE-2: Bigram (two consecutive words) overlap")
print("  • ROUGE-L: Longest Common Subsequence\n")

rouge_results = []

for i, (gen, ref) in enumerate(zip(generated_answers, ground_truth)):
    scores = scorer.score(ref, gen)
    
    rouge_results.append({
        'Question_ID': i + 1,
        'ROUGE-1_P': scores['rouge1'].precision,
        'ROUGE-1_R': scores['rouge1'].recall,
        'ROUGE-1_F1': scores['rouge1'].fmeasure,
        'ROUGE-2_P': scores['rouge2'].precision,
        'ROUGE-2_R': scores['rouge2'].recall,
        'ROUGE-2_F1': scores['rouge2'].fmeasure,
        'ROUGE-L_P': scores['rougeL'].precision,
        'ROUGE-L_R': scores['rougeL'].recall,
        'ROUGE-L_F1': scores['rougeL'].fmeasure,
    })

rouge_df = pd.DataFrame(rouge_results)

print("\n📊 Per-Question ROUGE Scores:")
print("-" * 80)
display(rouge_df)

print("\n📊 Average ROUGE Scores:")
print("-" * 80)
avg_scores = rouge_df.mean()
print(f"ROUGE-1 - Precision: {avg_scores['ROUGE-1_P']:.3f}, Recall: {avg_scores['ROUGE-1_R']:.3f}, F1: {avg_scores['ROUGE-1_F1']:.3f}")
print(f"ROUGE-2 - Precision: {avg_scores['ROUGE-2_P']:.3f}, Recall: {avg_scores['ROUGE-2_R']:.3f}, F1: {avg_scores['ROUGE-2_F1']:.3f}")
print(f"ROUGE-L - Precision: {avg_scores['ROUGE-L_P']:.3f}, Recall: {avg_scores['ROUGE-L_R']:.3f}, F1: {avg_scores['ROUGE-L_F1']:.3f}")

print("\n📖 Score Interpretation:")
print("-" * 80)
print("  • Precision: How many generated n-grams appear in reference?")
print("  • Recall: How many reference n-grams appear in generated answer?")
print("  • F1: Harmonic mean of precision and recall")
print("  • Higher is better (max = 1.0)")

# COMMAND ----------

# DBTITLE 1,ROUGE Explanation
# MAGIC %md
# MAGIC ## ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
# MAGIC
# MAGIC ### 🎯 When to Use
# MAGIC * **Text summarization tasks**: ROUGE was designed for evaluating summaries
# MAGIC * **Content overlap assessment**: When you want to measure how much content is shared
# MAGIC * **Quick baseline evaluation**: Fast to compute, good for initial assessment
# MAGIC * **Multi-reference evaluation**: Works well when you have multiple valid reference answers
# MAGIC
# MAGIC ### 📊 What It Measures
# MAGIC
# MAGIC **ROUGE-1 (Unigram Overlap)**
# MAGIC * Counts individual word matches between generated and reference text
# MAGIC * Good for measuring overall content coverage
# MAGIC * Example: "machine learning" vs "learning machine" → 100% ROUGE-1
# MAGIC
# MAGIC **ROUGE-2 (Bigram Overlap)**
# MAGIC * Counts two-word phrase matches
# MAGIC * More strict than ROUGE-1, captures word order better
# MAGIC * Example: "machine learning" vs "learning machine" → 0% ROUGE-2
# MAGIC
# MAGIC **ROUGE-L (Longest Common Subsequence)**
# MAGIC * Finds the longest sequence of words that appear in both texts (not necessarily consecutive)
# MAGIC * Balances flexibility and structure
# MAGIC * Captures sentence-level structure similarity
# MAGIC
# MAGIC **Precision vs Recall vs F1**
# MAGIC * **Precision**: How many generated words are relevant? (penalizes verbosity)
# MAGIC * **Recall**: How many reference words are covered? (penalizes brevity)
# MAGIC * **F1**: Balanced combination of precision and recall
# MAGIC
# MAGIC ### ⚠️ Drawbacks
# MAGIC
# MAGIC 1. **No Semantic Understanding**: Only measures surface-level word overlap
# MAGIC    * "big" and "large" are treated as completely different
# MAGIC    * "The cat sat on the mat" vs "The mat was sat on by the cat" → low score despite same meaning
# MAGIC
# MAGIC 2. **Sensitive to Phrasing**: Small paraphrases cause large score drops
# MAGIC    * "Machine learning is AI" vs "AI includes machine learning" → low score
# MAGIC
# MAGIC 3. **Doesn't Capture Meaning**: High ROUGE doesn't guarantee semantic correctness
# MAGIC    * Can score well by copying words without understanding
# MAGIC
# MAGIC 4. **Word Order Limitations**: ROUGE-1 ignores word order completely
# MAGIC
# MAGIC 5. **Requires Reference Text**: Needs ground truth answers to compare against
# MAGIC
# MAGIC 6. **Not Ideal for Creative Tasks**: Penalizes valid alternative phrasings
# MAGIC
# MAGIC ### ✅ Best For
# MAGIC * Text summarization evaluation
# MAGIC * Content coverage assessment  
# MAGIC * Fast baseline metrics
# MAGIC * Extractive QA (where answer is copied from text)
# MAGIC
# MAGIC ### ❌ Not Good For
# MAGIC * Semantic similarity
# MAGIC * Paraphrased answers
# MAGIC * Creative generation tasks
# MAGIC * Conversational AI

# COMMAND ----------

# DBTITLE 1,BLEU Score Evaluation
# BLEU: Bilingual Evaluation Understudy
# Originally designed for machine translation, measures n-gram precision

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Initialize smoothing function (helps with short sentences)
smoothing = SmoothingFunction().method1

print("📋 BLEU Score Evaluation")
print("=" * 80)
print("\nBLEU measures n-gram precision with brevity penalty.")
print("Originally designed for machine translation evaluation.\n")

bleu_results = []

for i, (gen, ref) in enumerate(zip(generated_answers, ground_truth)):
    # Tokenize
    reference = [word_tokenize(ref.lower())]
    candidate = word_tokenize(gen.lower())
    
    # Calculate BLEU with different n-gram weights
    bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu_2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu_3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    bleu_results.append({
        'Question_ID': i + 1,
        'BLEU-1': bleu_1,
        'BLEU-2': bleu_2,
        'BLEU-3': bleu_3,
        'BLEU-4': bleu_4
    })

bleu_df = pd.DataFrame(bleu_results)

print("\n📊 Per-Question BLEU Scores:")
print("-" * 80)
display(bleu_df)

print("\n📊 Average BLEU Scores:")
print("-" * 80)
avg_bleu = bleu_df.mean()
for n in [1, 2, 3, 4]:
    print(f"BLEU-{n}: {avg_bleu[f'BLEU-{n}']:.3f}")

print("\n📖 Score Interpretation:")
print("-" * 80)
print("  • BLEU-1: Considers only unigram matches")
print("  • BLEU-2: Considers unigram + bigram matches")
print("  • BLEU-3: Considers up to trigram matches")
print("  • BLEU-4: Considers up to 4-gram matches (standard)")
print("  • Includes brevity penalty (penalizes too-short outputs)")
print("  • Range: 0.0 (worst) to 1.0 (perfect match)")
print("  • Good scores: >0.3 decent, >0.5 good, >0.7 excellent")

# COMMAND ----------

# DBTITLE 1,BLEU Explanation
# MAGIC %md
# MAGIC ## BLEU (Bilingual Evaluation Understudy)
# MAGIC
# MAGIC ### 🎯 When to Use
# MAGIC * **Machine translation**: Originally designed for MT, still widely used
# MAGIC * **Exact match scenarios**: When precise wording matters
# MAGIC * **Template-based generation**: When output should follow specific patterns
# MAGIC * **Comparing to multiple references**: Works well with multiple valid translations
# MAGIC
# MAGIC ### 📊 What It Measures
# MAGIC
# MAGIC **Modified N-gram Precision**
# MAGIC * Counts how many n-grams in generated text appear in reference
# MAGIC * "Modified" means each reference n-gram can only be matched once (prevents gaming)
# MAGIC * Uses geometric mean of n-gram precisions (typically n=1,2,3,4)
# MAGIC
# MAGIC **Brevity Penalty**
# MAGIC * Penalizes outputs that are too short
# MAGIC * Prevents systems from gaming the metric by generating very short, safe outputs
# MAGIC * BP = 1 if output ≥ reference length, otherwise exponentially decreases
# MAGIC
# MAGIC **Calculation**
# MAGIC ```
# MAGIC BLEU = BP × exp(Σ(wn × log(pn)))
# MAGIC where:
# MAGIC   BP = brevity penalty
# MAGIC   wn = weight for n-gram (usually 0.25 for n=1,2,3,4)
# MAGIC   pn = precision for n-gram
# MAGIC ```
# MAGIC
# MAGIC ### ⚠️ Drawbacks
# MAGIC
# MAGIC 1. **Favors Brevity Over Completeness**: Even with brevity penalty, tends to favor shorter outputs
# MAGIC    * Systems learn to be conservative rather than comprehensive
# MAGIC
# MAGIC 2. **No Semantic Understanding**: Like ROUGE, only measures surface form
# MAGIC    * "not bad" vs "good" → low BLEU despite similar meaning
# MAGIC    * Treats synonyms as completely different words
# MAGIC
# MAGIC 3. **Poor for Single Reference**: BLEU was designed for multiple reference translations
# MAGIC    * With one reference, valid paraphrases are heavily penalized
# MAGIC    * Translation has many valid outputs, but QA often has one reference answer
# MAGIC
# MAGIC 4. **Word Order Matters Too Much**: Strictly counts n-gram matches
# MAGIC    * "The dog bit the man" vs "The man bit the dog" → different meanings, similar BLEU
# MAGIC
# MAGIC 5. **Not Reliable for Short Texts**: Struggles with sentence-level evaluation
# MAGIC    * Was designed for document-level translation
# MAGIC    * Smoothing functions help but don't fully solve this
# MAGIC
# MAGIC 6. **Doesn't Capture Fluency**: A word salad with correct n-grams can score well
# MAGIC
# MAGIC 7. **Insensitive to Serious Errors**: Small changes can have big meaning differences
# MAGIC    * Adding "not" changes meaning completely but barely affects BLEU
# MAGIC
# MAGIC ### ✅ Best For
# MAGIC * Machine translation evaluation
# MAGIC * Multiple reference scenarios
# MAGIC * Template-based or constrained generation
# MAGIC * Comparing similar-style outputs
# MAGIC
# MAGIC ### ❌ Not Good For
# MAGIC * Abstractive summarization
# MAGIC * Creative generation
# MAGIC * Paraphrased answers
# MAGIC * Semantic equivalence evaluation
# MAGIC * RAG systems (better to use RAG-specific metrics)

# COMMAND ----------

# DBTITLE 1,BERTScore Evaluation
# BERTScore: Leveraging contextual embeddings for semantic similarity
# Uses BERT model to compute similarity between generated and reference text

import bert_score
import warnings
warnings.filterwarnings('ignore')

print("📋 BERTScore Evaluation")
print("=" * 80)
print("\nBERTScore uses BERT embeddings to measure semantic similarity.")
print("Computing token-level similarities using contextual embeddings...\n")

# Calculate BERTScore
# model_type: can be 'bert-base-uncased', 'roberta-large', etc.
P, R, F1 = bert_score.score(
    generated_answers,
    ground_truth,
    lang="en",
    model_type="microsoft/deberta-xlarge-mnli",  # High-quality model
    verbose=False
)

# Convert to numpy for easier handling
P_scores = P.numpy()
R_scores = R.numpy()
F1_scores = F1.numpy()

bertscore_results = []
for i in range(len(questions)):
    bertscore_results.append({
        'Question_ID': i + 1,
        'Precision': P_scores[i],
        'Recall': R_scores[i],
        'F1': F1_scores[i]
    })

bertscore_df = pd.DataFrame(bertscore_results)

print("\n📊 Per-Question BERTScore:")
print("-" * 80)
display(bertscore_df)

print("\n📊 Average BERTScore:")
print("-" * 80)
print(f"Precision: {P_scores.mean():.3f}")
print(f"Recall:    {R_scores.mean():.3f}")
print(f"F1 Score:  {F1_scores.mean():.3f}")

print("\n📖 Score Interpretation:")
print("-" * 80)
print("  • Precision: How semantically similar are generated tokens to reference?")
print("  • Recall: How much of the reference semantics is captured?")
print("  • F1: Harmonic mean of precision and recall")
print("  • Range: -1.0 to 1.0 (typically 0.8-1.0 for good matches)")
print("  • Uses contextualized embeddings (understands 'big' ≈ 'large')")
print("  • Model: microsoft/deberta-xlarge-mnli (high quality but slower)")

# COMMAND ----------

# DBTITLE 1,BERTScore Explanation
# MAGIC %md
# MAGIC ## BERTScore (BERT-based Semantic Similarity)
# MAGIC
# MAGIC ### 🎯 When to Use
# MAGIC * **Semantic similarity**: When you care about meaning more than exact wording
# MAGIC * **Paraphrased content**: Evaluating outputs that express the same idea differently
# MAGIC * **Abstractive summarization**: Where rewording is expected
# MAGIC * **Open-ended generation**: Creative tasks where many phrasings are valid
# MAGIC * **RAG systems**: Better than ROUGE/BLEU for evaluating generated answers
# MAGIC
# MAGIC ### 📊 What It Measures
# MAGIC
# MAGIC **Contextual Embedding Similarity**
# MAGIC * Uses BERT (or similar) to create contextual embeddings for each token
# MAGIC * Computes cosine similarity between token embeddings
# MAGIC * Matches tokens based on semantic meaning, not exact string match
# MAGIC
# MAGIC **How It Works**
# MAGIC 1. Generate contextual embeddings for both texts using BERT
# MAGIC 2. For each token in generated text, find most similar token in reference
# MAGIC 3. Compute precision: average similarity of generated tokens
# MAGIC 4. Compute recall: average similarity of reference tokens
# MAGIC 5. Compute F1: harmonic mean of precision and recall
# MAGIC
# MAGIC **Key Advantages Over N-gram Metrics**
# MAGIC * **Synonym awareness**: "big" and "large" have high similarity
# MAGIC * **Context understanding**: "bank" (financial) vs "bank" (river) are distinguished
# MAGIC * **Paraphrase friendly**: "not bad" and "good" have reasonable similarity
# MAGIC * **Semantic focus**: Measures meaning rather than surface form
# MAGIC
# MAGIC ### ⚠️ Drawbacks
# MAGIC
# MAGIC 1. **Computationally Expensive**: Requires running BERT forward passes
# MAGIC    * Much slower than ROUGE/BLEU (seconds vs milliseconds)
# MAGIC    * Needs GPU for reasonable speed on large datasets
# MAGIC
# MAGIC 2. **Model Dependent**: Results vary based on BERT model used
# MAGIC    * `bert-base-uncased`: Fast but less accurate
# MAGIC    * `roberta-large`: Better but slower
# MAGIC    * `microsoft/deberta-xlarge-mnli`: Best but slowest
# MAGIC    * Different models = different scores (not directly comparable)
# MAGIC
# MAGIC 3. **Can Be Too Lenient**: May give high scores to semantically related but factually incorrect text
# MAGIC    * "Paris is the capital of France" vs "Paris is in France" → high BERTScore
# MAGIC    * Captures semantic similarity but not factual accuracy
# MAGIC
# MAGIC 4. **Requires GPU for Speed**: CPU inference is very slow for large evaluations
# MAGIC
# MAGIC 5. **Token-Level Matching**: Doesn't consider sentence structure or global coherence
# MAGIC    * Can miss issues with logical flow or overall structure
# MAGIC
# MAGIC 6. **Language Specific**: Requires language-specific BERT models
# MAGIC    * Pre-trained models available for major languages only
# MAGIC
# MAGIC 7. **Score Interpretation**: Unlike ROUGE (0-1), BERTScore range depends on model
# MAGIC    * Typical good scores: 0.85-0.95
# MAGIC    * Hard to set universal thresholds
# MAGIC
# MAGIC 8. **Reference Dependency**: Like other metrics, requires reference text
# MAGIC
# MAGIC ### ✅ Best For
# MAGIC * RAG system evaluation (better than ROUGE for semantic matching)
# MAGIC * Abstractive summarization
# MAGIC * Paraphrase detection
# MAGIC * Any task where semantic equivalence matters
# MAGIC * Evaluating creative or open-ended generation
# MAGIC
# MAGIC ### ❌ Not Good For
# MAGIC * Real-time evaluation (too slow)
# MAGIC * Exact match requirements (MT, code generation)
# MAGIC * Very large scale evaluation (cost/time)
# MAGIC * When no GPU available
# MAGIC * Factual accuracy verification (can miss subtle errors)

# COMMAND ----------

# DBTITLE 1,METEOR Score Evaluation
# METEOR: Metric for Evaluation of Translation with Explicit ORdering
# Balances precision/recall with synonym matching and word order

from nltk.translate.meteor_score import meteor_score as nltk_meteor_score
from nltk.tokenize import word_tokenize

print("📋 METEOR Score Evaluation")
print("=" * 80)
print("\nMETEOR measures alignment between generated and reference text.")
print("Features: unigram matching, synonym matching, stemming, word order.\n")

meteor_results = []

for i, (gen, ref) in enumerate(zip(generated_answers, ground_truth)):
    # Tokenize
    reference = word_tokenize(ref.lower())
    candidate = word_tokenize(gen.lower())
    
    # Calculate METEOR score
    # METEOR considers: exact matches, stem matches, synonym matches, and paraphrase matches
    score = nltk_meteor_score([reference], candidate)
    
    meteor_results.append({
        'Question_ID': i + 1,
        'METEOR': score
    })

meteor_df = pd.DataFrame(meteor_results)

print("\n📊 Per-Question METEOR Scores:")
print("-" * 80)
display(meteor_df)

print("\n📊 Average METEOR Score:")
print("-" * 80)
avg_meteor = meteor_df['METEOR'].mean()
print(f"METEOR: {avg_meteor:.3f}")

print("\n📖 Score Interpretation:")
print("-" * 80)
print("  • Range: 0.0 (no match) to 1.0 (perfect match)")
print("  • Considers exact matches, stems, and synonyms via WordNet")
print("  • Includes penalty for word order differences (fragmentation)")
print("  • Balances precision and recall (favors recall slightly)")
print("  • Good scores: >0.3 decent, >0.5 good, >0.7 excellent")
print("  • More lenient than BLEU (synonyms count as matches)")
print("  • Better correlation with human judgments than BLEU")

print("\nℹ️ METEOR Components:")
print("-" * 80)
print("  1. Unigram Precision and Recall")
print("  2. Synonym matching using WordNet")
print("  3. Stemming (e.g., 'running' matches 'run')")
print("  4. Fragmentation penalty for word order differences")

# COMMAND ----------

# DBTITLE 1,METEOR Explanation
# MAGIC %md
# MAGIC ## METEOR (Metric for Evaluation of Translation with Explicit ORdering)
# MAGIC
# MAGIC ### 🎯 When to Use
# MAGIC * **Balanced evaluation**: Want precision + recall + linguistic features
# MAGIC * **Synonym-aware evaluation**: When paraphrases should be recognized
# MAGIC * **Machine translation**: Designed for MT, works better than BLEU
# MAGIC * **Natural language generation**: Any task where linguistic variation is expected
# MAGIC * **When BLEU is too strict**: METEOR is more lenient and flexible
# MAGIC
# MAGIC ### 📊 What It Measures
# MAGIC
# MAGIC **Four Types of Matches**
# MAGIC 1. **Exact Match**: Identical words
# MAGIC    * "machine" matches "machine"
# MAGIC
# MAGIC 2. **Stem Match**: Same root after stemming
# MAGIC    * "running" matches "run"
# MAGIC    * "learned" matches "learning"
# MAGIC
# MAGIC 3. **Synonym Match**: Synonyms via WordNet
# MAGIC    * "big" matches "large"
# MAGIC    * "quick" matches "fast"
# MAGIC
# MAGIC 4. **Paraphrase Match**: Paraphrases from paraphrase tables
# MAGIC    * "was born in" matches "is a native of"
# MAGIC    * (Note: NLTK implementation may not include all paraphrase tables)
# MAGIC
# MAGIC **Fragmentation Penalty**
# MAGIC * Penalizes differences in word order
# MAGIC * Measures how "chunked" or fragmented the alignment is
# MAGIC * Lower penalty if matched words are in similar order
# MAGIC
# MAGIC **Calculation**
# MAGIC ```
# MAGIC METEOR = Fmean × (1 - Penalty)
# MAGIC where:
# MAGIC   Fmean = harmonic mean of precision and recall
# MAGIC   Penalty = 0.5 × (chunks / unigrams_matched)^3
# MAGIC ```
# MAGIC
# MAGIC **Precision/Recall Balance**
# MAGIC * Unlike BLEU (precision-focused), METEOR balances both
# MAGIC * Slightly favors recall over precision (α = 0.9 by default)
# MAGIC * Better for generation tasks where completeness matters
# MAGIC
# MAGIC ### ⚠️ Drawbacks
# MAGIC
# MAGIC 1. **Requires WordNet**: Needs language-specific lexical database
# MAGIC    * Available for English, but limited for other languages
# MAGIC    * WordNet coverage may not include domain-specific terms
# MAGIC    * Medical, legal, technical terms may not have synonym mappings
# MAGIC
# MAGIC 2. **Language Specific**: Each language needs its own resources
# MAGIC    * English is well-supported
# MAGIC    * Other languages have varying levels of support
# MAGIC    * Cross-lingual evaluation is difficult
# MAGIC
# MAGIC 3. **Still Surface-Level**: Despite synonyms, doesn't understand deep semantics
# MAGIC    * "The company gained profit" vs "The company was profitable" → may score low
# MAGIC    * Can't capture complex paraphrases or semantic equivalence
# MAGIC
# MAGIC 4. **Slower Than ROUGE/BLEU**: Synonym lookup adds computational cost
# MAGIC    * Still much faster than BERTScore though
# MAGIC    * Acceptable for most use cases
# MAGIC
# MAGIC 5. **Stemming Can Be Aggressive**: May match words that shouldn't be matched
# MAGIC    * "organization" and "organ" share stem
# MAGIC    * Can cause false positives
# MAGIC
# MAGIC 6. **WordNet Limitations**: Synonym detection is not perfect
# MAGIC    * May miss context-dependent synonyms
# MAGIC    * May include synonyms that don't fit the context
# MAGIC    * "bank" (financial) and "bank" (river) are treated same
# MAGIC
# MAGIC 7. **Fragmentation Penalty Can Be Too Harsh**: Penalizes valid reorderings
# MAGIC    * "The cat sat" vs "Sat the cat" (poetic order) → penalty
# MAGIC
# MAGIC ### ✅ Best For
# MAGIC * Machine translation evaluation (better than BLEU)
# MAGIC * Text generation with expected paraphrasing
# MAGIC * When you want balance between ROUGE (recall) and BLEU (precision)
# MAGIC * Domains where synonym awareness helps
# MAGIC * Natural language generation evaluation
# MAGIC
# MAGIC ### ❌ Not Good For
# MAGIC * Deep semantic understanding (use BERTScore)
# MAGIC * Real-time evaluation at massive scale
# MAGIC * Non-English languages with poor WordNet support
# MAGIC * Tasks requiring exact wording (code, structured data)
# MAGIC * Cross-lingual evaluation
# MAGIC
# MAGIC ### 🆚 Comparison with BLEU
# MAGIC * **METEOR pros**: Synonyms, recall balance, better human correlation
# MAGIC * **METEOR cons**: Slower, needs WordNet
# MAGIC * **General consensus**: METEOR > BLEU for most NLG tasks

# COMMAND ----------

# DBTITLE 1,Comparison Summary and Recommendations
# MAGIC %md
# MAGIC ## 📊 Metric Comparison Summary
# MAGIC
# MAGIC ### Quick Reference Table
# MAGIC
# MAGIC | Metric | Speed | Semantic Awareness | Use Case | Score Range | Requires LLM/GPU |
# MAGIC |--------|-------|-------------------|----------|-------------|------------------|
# MAGIC | **RAGAS** | ❌ Slow | ✅✅✅ Excellent | RAG systems | 0-1 | ✅ Yes (LLM) |
# MAGIC | **BERTScore** | ❌ Slow | ✅✅ Good | Semantic similarity | 0.8-1.0 | ✅ Yes (GPU) |
# MAGIC | **METEOR** | ⚠️ Medium | ✅ Limited | Balanced NLG | 0-1 | ❌ No |
# MAGIC | **ROUGE** | ✅ Fast | ❌ None | Summarization | 0-1 | ❌ No |
# MAGIC | **BLEU** | ✅ Fast | ❌ None | Translation | 0-1 | ❌ No |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 When to Use Each Metric
# MAGIC
# MAGIC ### Use RAGAS When:
# MAGIC * ✅ Evaluating RAG systems specifically
# MAGIC * ✅ You need component-level insights (retrieval vs generation)
# MAGIC * ✅ You have ground truth answers
# MAGIC * ✅ You have access to an LLM API (OpenAI, etc.)
# MAGIC * ✅ Cost and latency are acceptable
# MAGIC * ❌ **Avoid if**: No LLM access, need real-time evaluation, large-scale evaluation
# MAGIC
# MAGIC ### Use BERTScore When:
# MAGIC * ✅ Semantic similarity is more important than exact wording
# MAGIC * ✅ Evaluating paraphrased or abstractive content
# MAGIC * ✅ You have GPU resources
# MAGIC * ✅ Quality matters more than speed
# MAGIC * ❌ **Avoid if**: Need real-time scoring, no GPU, exact matches required
# MAGIC
# MAGIC ### Use METEOR When:
# MAGIC * ✅ You want synonym awareness without deep learning
# MAGIC * ✅ Balanced precision/recall is important
# MAGIC * ✅ Working in English or languages with good WordNet support
# MAGIC * ✅ Faster than BERTScore but better than ROUGE
# MAGIC * ❌ **Avoid if**: Non-English with poor WordNet, need deep semantics
# MAGIC
# MAGIC ### Use ROUGE When:
# MAGIC * ✅ Evaluating extractive summarization
# MAGIC * ✅ Need fast, lightweight metric
# MAGIC * ✅ Content overlap is the primary concern
# MAGIC * ✅ Running large-scale evaluations
# MAGIC * ❌ **Avoid if**: Paraphrasing expected, semantic similarity matters
# MAGIC
# MAGIC ### Use BLEU When:
# MAGIC * ✅ Evaluating machine translation
# MAGIC * ✅ Multiple reference translations available
# MAGIC * ✅ Exact wording is important
# MAGIC * ✅ Need very fast computation
# MAGIC * ❌ **Avoid if**: Single reference, abstractive tasks, semantic equivalence needed
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🛠️ Best Practices
# MAGIC
# MAGIC ### 1. 🎯 **Use Multiple Metrics Together**
# MAGIC    * No single metric is perfect
# MAGIC    * Combine fast metrics (ROUGE) with semantic metrics (BERTScore)
# MAGIC    * Example: ROUGE for baseline + BERTScore for final evaluation
# MAGIC
# MAGIC ### 2. 📈 **Consider Your Task Type**
# MAGIC    * **RAG systems**: RAGAS (if possible) + BERTScore
# MAGIC    * **Summarization**: ROUGE + BERTScore
# MAGIC    * **Translation**: BLEU + METEOR
# MAGIC    * **Open-ended QA**: BERTScore + METEOR
# MAGIC    * **Extractive QA**: ROUGE + exact match
# MAGIC
# MAGIC ### 3. ⏱️ **Speed vs Quality Tradeoff**
# MAGIC    * **Development**: Use fast metrics (ROUGE, BLEU) for quick iteration
# MAGIC    * **Evaluation**: Add slower metrics (BERTScore, RAGAS) for thorough assessment
# MAGIC    * **Production**: Monitor with fast metrics, sample with slow metrics
# MAGIC
# MAGIC ### 4. 📊 **Interpretation Guidelines**
# MAGIC    * Don't rely on absolute scores alone
# MAGIC    * Compare scores across different models/versions
# MAGIC    * Combine metrics with human evaluation
# MAGIC    * Understand metric limitations for your specific task
# MAGIC
# MAGIC ### 5. 🧩 **Human Evaluation is Still King**
# MAGIC    * Metrics are proxies, not ground truth
# MAGIC    * Always validate automated metrics with human judgments
# MAGIC    * Use metrics to prioritize what humans should review
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📝 Recommended Combinations
# MAGIC
# MAGIC ### For RAG Systems
# MAGIC ```python
# MAGIC # Fast baseline
# MAGIC ROUGE + Exact Match
# MAGIC
# MAGIC # Comprehensive evaluation  
# MAGIC RAGAS (if LLM available) + BERTScore + ROUGE
# MAGIC
# MAGIC # Production monitoring
# MAGIC ROUGE (continuous) + BERTScore (sampled) + Human review (sampled)
# MAGIC ```
# MAGIC
# MAGIC ### For Summarization
# MAGIC ```python
# MAGIC # Extractive
# MAGIC ROUGE-1, ROUGE-2, ROUGE-L
# MAGIC
# MAGIC # Abstractive
# MAGIC ROUGE + BERTScore + Human evaluation
# MAGIC ```
# MAGIC
# MAGIC ### For Question Answering
# MAGIC ```python
# MAGIC # Extractive QA
# MAGIC Exact Match + ROUGE + F1
# MAGIC
# MAGIC # Abstractive QA
# MAGIC BERTScore + METEOR + Human evaluation
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚠️ Common Pitfalls to Avoid
# MAGIC
# MAGIC 1. **Don't optimize for metrics alone** - They're imperfect proxies
# MAGIC 2. **Don't ignore context** - What works for MT may not work for RAG
# MAGIC 3. **Don't trust single metrics** - Use multiple perspectives
# MAGIC 4. **Don't skip human evaluation** - Metrics can miss important issues
# MAGIC 5. **Don't compare scores across different BERTScore models** - They're not directly comparable
# MAGIC 6. **Don't use BLEU for single-reference tasks** - It's designed for multiple references
# MAGIC 7. **Don't expect high ROUGE on paraphrased content** - It measures surface overlap
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔗 Additional Resources
# MAGIC
# MAGIC * **RAGAS**: https://docs.ragas.io/
# MAGIC * **BERTScore**: https://github.com/Tiiiger/bert_score
# MAGIC * **ROUGE**: https://github.com/google-research/google-research/tree/master/rouge
# MAGIC * **BLEU**: Original paper by Papineni et al., 2002
# MAGIC * **METEOR**: Original paper by Banerjee and Lavie, 2005
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Summary
# MAGIC
# MAGIC * **For RAG**: RAGAS > BERTScore > METEOR > ROUGE > BLEU
# MAGIC * **For Speed**: BLEU = ROUGE > METEOR > BERTScore > RAGAS
# MAGIC * **For Semantics**: RAGAS > BERTScore > METEOR > ROUGE = BLEU
# MAGIC * **For Paraphrases**: BERTScore > METEOR > ROUGE > BLEU
# MAGIC
# MAGIC **Bottom Line**: Choose metrics based on your task, resources, and what you're trying to measure. When in doubt, use multiple metrics and validate with human evaluation.