# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # Self-Attention Mechanism - From Scratch
# MAGIC
# MAGIC ## The Problem: Understanding "It"
# MAGIC
# MAGIC Consider this sentence: **"The animal didn't cross the street because it was too tired."**
# MAGIC
# MAGIC What does **"it"** refer to? The street or the animal?
# MAGIC
# MAGIC We humans immediately know "it" refers to the animal. But how? **We look at other words in the sentence.**
# MAGIC
# MAGIC This is what **self-attention** does - it lets every word look at every other word to understand the full context.
# MAGIC
# MAGIC ## Why Self-Attention?
# MAGIC
# MAGIC Old RNN models read words one by one, left to right. By the time they reached "it", they often forgot about "animal". This is called the **long-term dependency problem**.
# MAGIC
# MAGIC **Self-attention solves this** by processing all words simultaneously in parallel. Every word can directly attend to every other word.

# COMMAND ----------

# DBTITLE 1,Setup and Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("✓ Libraries imported successfully")

# COMMAND ----------

# DBTITLE 1,Step 1 - Word Embeddings
# MAGIC %md
# MAGIC ## Step 1: Convert Words to Numbers (Embeddings)
# MAGIC
# MAGIC Neural networks can't read text - they only understand numbers.
# MAGIC
# MAGIC We convert each word into a vector of numbers called an **embedding**.
# MAGIC
# MAGIC For our example sentence: **"The cat sat"**
# MAGIC - Each word becomes a vector of 4 dimensions
# MAGIC - These vectors capture the "meaning" of each word

# COMMAND ----------

# DBTITLE 1,Create Word Embeddings
# Our sentence: "The cat sat"
tokens = ["The", "cat", "sat"]
num_tokens = len(tokens)
d_model = 4  # Embedding dimension (keeping it small for visualization)

# Create random embeddings for each word
# In real transformers, these are learned during training
X = np.random.randn(num_tokens, d_model)

print(f"Input Matrix X shape: {X.shape}")
print(f"\nEmbeddings for each word:")
for i, token in enumerate(tokens):
    print(f"{token:5s}: {X[i]}")

# Visualize embeddings
plt.figure(figsize=(8, 4))
sns.heatmap(X, annot=True, fmt=".2f", cmap="coolwarm", 
            xticklabels=[f"Dim{i}" for i in range(d_model)],
            yticklabels=tokens, cbar_kws={'label': 'Value'})
plt.title("Word Embeddings Matrix X\n(3 tokens × 4 dimensions)")
plt.xlabel("Embedding Dimensions")
plt.ylabel("Tokens")
plt.tight_layout()
plt.show()

print("\n✓ Each row represents one word as a 4-dimensional vector")

# COMMAND ----------

# DBTITLE 1,Step 2 - Query Key Value
# MAGIC %md
# MAGIC ## Step 2: Create Query, Key, and Value Matrices
# MAGIC
# MAGIC Self-attention uses three different "views" of each word:
# MAGIC
# MAGIC 1. **Query (Q)**: What am I looking for?
# MAGIC 2. **Key (K)**: What information do I contain?
# MAGIC 3. **Value (V)**: What actual information should I pass?
# MAGIC
# MAGIC We create these by multiplying our embeddings X with learned weight matrices:
# MAGIC - Q = X @ W_Q
# MAGIC - K = X @ W_K  
# MAGIC - V = X @ W_V
# MAGIC
# MAGIC These weight matrices are learned during training.

# COMMAND ----------

# Create random weight matrices (in real transformers, these are learned)
# Each weight matrix projects from d_model dimensions to d_k dimensions
d_k = d_model  # For simplicity, keeping same dimension

W_Q = np.random.randn(d_model, d_k) * 0.1  # Query weight matrix
W_K = np.random.randn(d_model, d_k) * 0.1  # Key weight matrix
W_V = np.random.randn(d_model, d_k) * 0.1  # Value weight matrix
print(np.random.randn(d_model, d_k))
print(np.random.randn(d_model, d_k)*0.1)

# COMMAND ----------


# Compute Q, K, V by matrix multiplication
Q = X @ W_Q  # Query: What am I looking for?
K = X @ W_K  # Key: What do I contain?
V = X @ W_V  # Value: What information to pass?

print(f"\nQ shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")

# COMMAND ----------

# DBTITLE 1,Compute Q, K, V

print(f"Weight matrix shapes: {W_Q.shape}")


print("\n" + "="*50)
print("Query Matrix (what each word is looking for):")
print(Q)
print("\n" + "="*50)
print("Key Matrix (what each word offers):")
print(K)
print("\n" + "="*50)
print("Value Matrix (actual content to pass):")
print(V)
print("="*50)

# COMMAND ----------

# DBTITLE 1,Step 3 - Attention Scores
# MAGIC %md
# MAGIC ## Step 3: Calculate Attention Scores
# MAGIC
# MAGIC Now we compute how much each word should attend to every other word.
# MAGIC
# MAGIC **Formula**: Attention_Scores = Q @ K^T
# MAGIC
# MAGIC This creates a matrix where:
# MAGIC - Each row represents a word asking "who should I pay attention to?"
# MAGIC - Each column represents a word being evaluated
# MAGIC - Higher values mean stronger relationships

# COMMAND ----------

# Compute attention scores: Q multiplied by K transpose
attention_scores = Q @ K.T

print(f"Attention Scores shape: {attention_scores.shape}")
print(f"\nRaw Attention Scores:")
print(attention_scores)


# COMMAND ----------

# DBTITLE 1,Calculate Raw Attention Scores

# Visualize attention scores
plt.figure(figsize=(7, 6))
sns.heatmap(attention_scores, annot=True, fmt=".3f", cmap="YlOrRd",
            xticklabels=tokens, yticklabels=tokens, 
            cbar_kws={'label': 'Score'})
plt.title("Raw Attention Scores (Q @ K^T)\n\nEach cell shows how much a word (row) attends to another (column)")
plt.xlabel("Key (being evaluated)")
plt.ylabel("Query (asking)")
plt.tight_layout()
plt.show()

print("\n📝 Notice:")
print("- Diagonal values show self-attention (word attends to itself)")
print("- Off-diagonal shows relationships between different words")

# COMMAND ----------

# DBTITLE 1,Step 4 - Scaling
# MAGIC %md
# MAGIC ## Step 4: Scale the Scores
# MAGIC
# MAGIC When dimensions are large, attention scores can become very big numbers.
# MAGIC
# MAGIC This causes problems with softmax (gradients vanish).
# MAGIC
# MAGIC **Solution**: Divide by √(d_k)
# MAGIC
# MAGIC This is called **Scaled Dot-Product Attention**.

# COMMAND ----------

# Scale attention scores by square root of key dimension
scaled_scores = attention_scores / np.sqrt(d_k)

print(f"Scaling factor: √{d_k} = {np.sqrt(d_k):.3f}")
print(f"\nScaled Attention Scores:")
print(scaled_scores)

# COMMAND ----------

# DBTITLE 1,Apply Scaling


# Compare before and after scaling
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before scaling
sns.heatmap(attention_scores, annot=True, fmt=".3f", cmap="YlOrRd",
            xticklabels=tokens, yticklabels=tokens, ax=axes[0],
            cbar_kws={'label': 'Score'})
axes[0].set_title("Before Scaling")
axes[0].set_xlabel("Key")
axes[0].set_ylabel("Query")

# After scaling
sns.heatmap(scaled_scores, annot=True, fmt=".3f", cmap="YlOrRd",
            xticklabels=tokens, yticklabels=tokens, ax=axes[1],
            cbar_kws={'label': 'Score'})
axes[1].set_title("After Scaling (÷ √d_k)")
axes[1].set_xlabel("Key")
axes[1].set_ylabel("Query")

plt.tight_layout()
plt.show()

print("\n✓ Scaling brings values to a stable range for softmax")

# COMMAND ----------

# DBTITLE 1,Step 5 - Softmax
# MAGIC %md
# MAGIC ## Step 5: Apply Softmax
# MAGIC
# MAGIC Softmax converts raw scores into **probabilities** that sum to 1.
# MAGIC
# MAGIC For each word (each row):
# MAGIC 1. Exponentiate all values (amplifies differences)
# MAGIC 2. Normalize so they sum to exactly 1.0
# MAGIC
# MAGIC Now we have **attention weights** - valid probability distributions!

# COMMAND ----------

def softmax(x):
    """Apply softmax to each row"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Apply softmax to get attention weights
attention_weights = softmax(scaled_scores)

print(f"Attention Weights shape: {attention_weights.shape}")
print(f"\nAttention Weights (after softmax):")
print(attention_weights)

# COMMAND ----------



print("\n🔍 Verification:")
for i, token in enumerate(tokens):
    row_sum = attention_weights[i].sum()
    print(f"{token:5s} row sums to: {row_sum:.6f} ✓")

# COMMAND ----------

# DBTITLE 1,Compute Attention Weights


# Visualize attention weights
plt.figure(figsize=(8, 6))
sns.heatmap(attention_weights, annot=True, fmt=".3f", cmap="Blues",
            xticklabels=tokens, yticklabels=tokens,
            vmin=0, vmax=1, cbar_kws={'label': 'Attention Weight'})
plt.title("Attention Weights Matrix\n\nEach row shows how much a word attends to others (sums to 1.0)")
plt.xlabel("Attending to (Key)")
plt.ylabel("Attention from (Query)")
plt.tight_layout()
plt.show()

print("\n📊 Interpretation:")
for i, token in enumerate(tokens):
    max_idx = attention_weights[i].argmax()
    print(f"{token:5s} attends most to: {tokens[max_idx]:5s} ({attention_weights[i, max_idx]:.1%})")

# COMMAND ----------

# DBTITLE 1,Step 6 - Weighted Sum
# MAGIC %md
# MAGIC ## Step 6: Compute Final Output
# MAGIC
# MAGIC Now we use attention weights to create context-aware representations.
# MAGIC
# MAGIC **Formula**: Output = Attention_Weights @ V
# MAGIC
# MAGIC Each output word is a **weighted sum** of all value vectors:
# MAGIC - Words with high attention weight contribute more
# MAGIC - Words with low attention contribute less
# MAGIC
# MAGIC The result: **every word now contains information from the entire sentence!**

# COMMAND ----------

# Compute weighted sum of values
output = attention_weights @ V

print(f"Output shape: {output.shape}")
print(f"\nSelf-Attention Output:")
print(output)


# COMMAND ----------

print("\n" + "="*60)
print("COMPARISON: Input vs Output")
print("="*60)
for i, token in enumerate(tokens):
    print(f"\n{token}:")
    print(f"  Input (original embedding):  {X[i]}")
    print(f"  Output (context-enriched):   {output[i]}")
    print(f"  → Output is now aware of other words in the sentence!")


# COMMAND ----------

# DBTITLE 1,Calculate Final Output


# Visualize side-by-side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Input embeddings
sns.heatmap(X, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=[f"D{i}" for i in range(d_model)],
            yticklabels=tokens, ax=axes[0], cbar_kws={'label': 'Value'})
axes[0].set_title("Input Embeddings X\n(No context)")
axes[0].set_ylabel("Tokens")

# Output after self-attention
sns.heatmap(output, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=[f"D{i}" for i in range(d_model)],
            yticklabels=tokens, ax=axes[1], cbar_kws={'label': 'Value'})
axes[1].set_title("Output after Self-Attention\n(Context-aware)")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()

print("\n✨ Self-attention complete! Each word now understands the full sentence.")

# COMMAND ----------

# DBTITLE 1,Complete Function
# MAGIC %md
# MAGIC ## Complete Self-Attention Function
# MAGIC
# MAGIC Let's put everything together in one clean function:

# COMMAND ----------

# DBTITLE 1,Self-Attention Implementation
def self_attention(X, W_Q, W_K, W_V):
    """
    Complete self-attention mechanism
    
    Args:
        X: Input embeddings (num_tokens, d_model)
        W_Q, W_K, W_V: Weight matrices (d_model, d_k)
    
    Returns:
        output: Context-enriched representations (num_tokens, d_k)
        attention_weights: Attention probability matrix
    """
    # Step 1: Create Q, K, V
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    
    # Step 2: Calculate attention scores
    d_k = Q.shape[-1]
    scores = Q @ K.T
    
    # Step 3: Scale scores
    scaled_scores = scores / np.sqrt(d_k)
    
    # Step 4: Apply softmax
    attention_weights = softmax(scaled_scores)
    
    # Step 5: Weighted sum of values
    output = attention_weights @ V
    
    return output, attention_weights

# Test the function
output_test, weights_test = self_attention(X, W_Q, W_K, W_V)

print("✓ Self-attention function working correctly!")
print(f"\nOutput shape: {output_test.shape}")
print(f"Attention weights shape: {weights_test.shape}")

# Verify outputs match
assert np.allclose(output_test, output), "Outputs don't match!"
assert np.allclose(weights_test, attention_weights), "Weights don't match!"

print("\n✅ All verification passed!")

# COMMAND ----------

# DBTITLE 1,Summary
# MAGIC %md
# MAGIC ## Summary: What We Learned
# MAGIC
# MAGIC ### Self-Attention in 6 Steps:
# MAGIC
# MAGIC 1. **Word Embeddings**: Convert words to numerical vectors
# MAGIC 2. **Q, K, V Projection**: Create three different views (Query, Key, Value)
# MAGIC 3. **Attention Scores**: Calculate Q @ K^T to measure relationships
# MAGIC 4. **Scaling**: Divide by √d_k to stabilize values
# MAGIC 5. **Softmax**: Convert to probability distribution (sums to 1)
# MAGIC 6. **Weighted Sum**: Multiply attention weights by values
# MAGIC
# MAGIC ### Key Formula:
# MAGIC
# MAGIC ```
# MAGIC Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
# MAGIC ```
# MAGIC
# MAGIC ### Why It Works:
# MAGIC
# MAGIC - **Parallel Processing**: All words processed simultaneously (not sequential like RNNs)
# MAGIC - **Direct Connections**: Every word can directly attend to every other word
# MAGIC - **Context-Aware**: Output vectors contain information from entire sentence
# MAGIC - **No Long-Term Dependency Problem**: Distance doesn't matter - word 1 can directly attend to word 100
# MAGIC
# MAGIC ### What's Next?
# MAGIC
# MAGIC **Multi-Head Attention**: Instead of one attention mechanism, run several in parallel to capture different types of relationships simultaneously!