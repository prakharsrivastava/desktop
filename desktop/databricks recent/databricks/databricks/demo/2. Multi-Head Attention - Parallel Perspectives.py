# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # Multi-Head Attention - Multiple Perspectives
# MAGIC
# MAGIC ## The Ambiguity Problem
# MAGIC
# MAGIC Consider: **"The detective followed the spy with binoculars."**
# MAGIC
# MAGIC **Question**: Who has the binoculars?
# MAGIC
# MAGIC ### Two valid interpretations:
# MAGIC 1. **Detective has binoculars** - using them to follow the spy
# MAGIC 2. **Spy has binoculars** - just carrying them
# MAGIC
# MAGIC ### The Problem with Single-Head Attention:
# MAGIC - One attention mechanism can only capture **one relationship**
# MAGIC - It might focus on detective-binoculars (90% attention)
# MAGIC - But completely miss spy-binoculars (only 20% attention)
# MAGIC - **Second interpretation is lost!**
# MAGIC
# MAGIC ## The Solution: Multi-Head Attention
# MAGIC
# MAGIC Instead of one attention mechanism, run **multiple in parallel**!
# MAGIC
# MAGIC Each "head" learns to capture different types of relationships:
# MAGIC - Head 1: Grammar and syntax
# MAGIC - Head 2: Long-distance dependencies
# MAGIC - Head 3: Subject-verb connections
# MAGIC - Head 4: Positional patterns
# MAGIC
# MAGIC Think of it like a **team of detectives** - each specializes in different clues!

# COMMAND ----------

# DBTITLE 1,Setup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

np.random.seed(42)

print("✓ Setup complete")

# COMMAND ----------

# DBTITLE 1,Step 1 - Input Preparation
# MAGIC %md
# MAGIC ## Step 1: Input Embeddings
# MAGIC
# MAGIC Let's use a simple 3-token sentence with 512-dimensional embeddings (standard transformer size).

# COMMAND ----------

# DBTITLE 1,Create Input
# Sentence with 3 tokens
tokens = ["The", "cat", "sat"]
num_tokens = 3
d_model = 512  # Standard embedding dimension from original transformer paper

# Create input embeddings
X = np.random.randn(num_tokens, d_model) * 0.1

print(f"Input shape: {X.shape}")
print(f"  - {num_tokens} tokens")
print(f"  - {d_model} dimensions each")
print(f"\nInput matrix X: {num_tokens} × {d_model}")
print(f"\nFirst token embedding (first 10 dims): {X[0, :10]}")

# COMMAND ----------

# DBTITLE 1,Step 2 - Multi-Head Configuration
# MAGIC %md
# MAGIC ## Step 2: Split Into Multiple Heads
# MAGIC
# MAGIC ### Key Insight: Computational Efficiency
# MAGIC
# MAGIC Instead of running one massive 512-dimensional attention:
# MAGIC - **Split into 4 heads**
# MAGIC - Each head works on **512 ÷ 4 = 128 dimensions**
# MAGIC - All 4 run **in parallel**
# MAGIC - **Same computational cost** as one large head!
# MAGIC
# MAGIC ### Why shrink dimensions?
# MAGIC - Running 4 heads at full 512 dims = 4× cost
# MAGIC - Running 4 heads at 128 dims each = 1× cost
# MAGIC - **Get 4 perspectives for the price of 1!**

# COMMAND ----------

# DBTITLE 1,Configure Heads
# Multi-head attention configuration
num_heads = 4
d_k = d_model // num_heads  # Dimension per head

print(f"Multi-Head Configuration:")
print(f"  - Number of heads: {num_heads}")
print(f"  - Dimensions per head (d_k): {d_k}")
print(f"  - Total dimension: {num_heads} × {d_k} = {num_heads * d_k}")
print(f"\n✓ Each head processes {d_k} dimensions independently")

# COMMAND ----------

# DBTITLE 1,Step 3 - Create Weight Matrices
# MAGIC %md
# MAGIC ## Step 3: Create Q, K, V Weights
# MAGIC
# MAGIC In **standard self-attention**: 1 set of weights (W_Q, W_K, W_V)
# MAGIC
# MAGIC In **multi-head attention**: Each head has its **own independent weights**
# MAGIC
# MAGIC - Head 1: W_Q1, W_K1, W_V1
# MAGIC - Head 2: W_Q2, W_K2, W_V2
# MAGIC - Head 3: W_Q3, W_K3, W_V3
# MAGIC - Head 4: W_Q4, W_K4, W_V4
# MAGIC
# MAGIC Each head learns to focus on **different patterns** because weights are different!

# COMMAND ----------

d_model

# COMMAND ----------

d_k

# COMMAND ----------

np.random.randn(d_model, d_k) # r c

# COMMAND ----------

# DBTITLE 1,Initialize Weights
# Create separate weight matrices for each head
W_Q = [np.random.randn(d_model, d_k) * 0.1 for _ in range(num_heads)]
W_K = [np.random.randn(d_model, d_k) * 0.1 for _ in range(num_heads)]
W_V = [np.random.randn(d_model, d_k) * 0.1 for _ in range(num_heads)]

print(f"Created weight matrices for {num_heads} heads:")
print(f"\nEach W_Q shape: {W_Q[0].shape} (projects {d_model} → {d_k})")
print(f"Each W_K shape: {W_K[0].shape}")
print(f"Each W_V shape: {W_V[0].shape}")
print(f"\nTotal weight matrices: {3 * num_heads} (Q, K, V for each head)")

print("\n" + "="*50)
for i in range(num_heads):
    print(f"Head {i+1}: W_Q{i+1}({W_Q[i].shape}), W_K{i+1}({W_K[i].shape}), W_V{i+1}({W_V[i].shape})")
print("="*50)

# COMMAND ----------

# DBTITLE 1,Step 4 - Scaled Dot-Product Attention
# MAGIC %md
# MAGIC ## Step 4: Scaled Dot-Product Attention Function
# MAGIC
# MAGIC Same attention mechanism as before, but we'll apply it separately for each head.

# COMMAND ----------

    Q = X @ W_Q[0]
    K = X @ W_K[0]
    V = X @ W_V[0]
    
    print(f"  Q shape: {Q.shape}")
    print(f"  K shape: {K.shape}")
    print(f"  V shape: {V.shape}")
    
    # Run attention
    #output, attn_weights = scaled_dot_product_attention(Q, K, V)

# COMMAND ----------

Q.shape[-1]

# COMMAND ----------

# DBTITLE 1,Attention Function
def scaled_dot_product_attention(Q, K, V):
    """
    Single-head scaled dot-product attention
    
    Args:
        Q: Query matrix (num_tokens, d_k)
        K: Key matrix (num_tokens, d_k)
        V: Value matrix (num_tokens, d_k)
    
    Returns:
        output: Attention output (num_tokens, d_k)
        attention_weights: Attention probability matrix (num_tokens, num_tokens)
    """
    
    d_k = Q.shape[-1]
    
    # Calculate attention scores
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Apply softmax
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
    
    # Weighted sum of values
    output = attention_weights @ V
    
    return output, attention_weights

print("✓ Scaled dot-product attention function defined")

# COMMAND ----------

# DBTITLE 1,Step 5 - Run Each Head
# MAGIC %md
# MAGIC ## Step 5: Run Attention for Each Head Independently
# MAGIC
# MAGIC Now we:
# MAGIC 1. Project X into Q, K, V for each head
# MAGIC 2. Run attention for each head separately (in parallel)
# MAGIC 3. Each head produces its own output
# MAGIC
# MAGIC Think: **4 detectives analyzing the same sentence from different angles**

# COMMAND ----------

num_heads

# COMMAND ----------

# DBTITLE 1,Execute All Heads
# Store outputs and attention weights from each head
head_outputs = []
head_attention_weights = []

print("Running attention for each head...\n")

for i in range(num_heads):
    print(f"Head {i+1}:")
    
    # Project input to Q, K, V for this head
    Q = X @ W_Q[i]
    K = X @ W_K[i]
    V = X @ W_V[i]
    
    print(f"  Q shape: {Q.shape}")
    print(f"  K shape: {K.shape}")
    print(f"  V shape: {V.shape}")
    
    # Run attention
    output, attn_weights = scaled_dot_product_attention(Q, K, V)
    
    head_outputs.append(output)
    head_attention_weights.append(attn_weights)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")
    print(attn_weights)

print(f"✓ All {num_heads} heads computed successfully!")
print(f"\nEach head output: {num_tokens} tokens × {d_k} dimensions")


# COMMAND ----------

# DBTITLE 1,Step 6 - Visualize Attention
# MAGIC %md
# MAGIC ## Step 6: Visualize Different Attention Patterns
# MAGIC
# MAGIC Each head learns to focus on **different relationships**!
# MAGIC
# MAGIC Let's see what each head is "looking at":

# COMMAND ----------

# DBTITLE 1,Visualize Head Attention
# Visualize attention weights from all heads
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i in range(num_heads):
    sns.heatmap(head_attention_weights[i], 
                annot=True, fmt=".3f", 
                cmap="Blues",
                xticklabels=tokens, 
                yticklabels=tokens,
                vmin=0, vmax=1,
                ax=axes[i],
                cbar_kws={'label': 'Attention Weight'})
    
    axes[i].set_title(f"Head {i+1} Attention Pattern\n(Each row sums to 1.0)")
    axes[i].set_xlabel("Attending to (Key)")
    axes[i].set_ylabel("Attention from (Query)")

plt.suptitle("Multi-Head Attention: 4 Different Perspectives", fontsize=16, y=1.0)
plt.tight_layout()
plt.show()

print("\n🔍 Observations:")
print("  - Each head shows a DIFFERENT attention pattern")
print("  - Some heads focus on self-attention (diagonal)")
print("  - Others focus on specific word pairs")
print("  - This diversity captures multiple types of relationships!")

# COMMAND ----------

# DBTITLE 1,Step 7 - Concatenation
# MAGIC %md
# MAGIC ## Step 7: Concatenate Head Outputs
# MAGIC
# MAGIC Now we have 4 separate outputs:
# MAGIC - Head 1 output: 3 × 128
# MAGIC - Head 2 output: 3 × 128
# MAGIC - Head 3 output: 3 × 128
# MAGIC - Head 4 output: 3 × 128
# MAGIC
# MAGIC **Concatenate them side-by-side**:
# MAGIC - Result: 3 × 512 (back to original dimension!)

# COMMAND ----------

concat_output = np.concatenate(head_outputs, axis=-1)

# COMMAND ----------

concat_output

# COMMAND ----------

# Concatenate all head outputs along the feature dimension
concat_output = np.concatenate(head_outputs, axis=-1)

print(f"Individual head outputs: {num_heads} × ({num_tokens}, {d_k})")
print(f"\nAfter concatenation: {concat_output.shape}")
print(f"  - {concat_output.shape[0]} tokens")
print(f"  - {concat_output.shape[1]} dimensions ({num_heads} heads × {d_k} dims)")

# COMMAND ----------



# Verify dimension
assert concat_output.shape == (num_tokens, d_model), "Shape mismatch!"

print(f"\n✓ Successfully restored to original {d_model} dimensions!")


# COMMAND ----------

# DBTITLE 1,Concatenate Outputs

# Visualize concatenation
fig, axes = plt.subplots(1, 5, figsize=(18, 4))

# Show each head output
for i in range(num_heads):
    axes[i].imshow(head_outputs[i][:, :20], aspect='auto', cmap='coolwarm')
    axes[i].set_title(f"Head {i+1}\n{head_outputs[i].shape}")
    axes[i].set_ylabel("Tokens")
    axes[i].set_xlabel(f"Dims 0-19\n(of {d_k})")
    axes[i].set_yticks(range(num_tokens))
    axes[i].set_yticklabels(tokens)

# Show concatenated result
axes[4].imshow(concat_output[:, :80], aspect='auto', cmap='coolwarm')
axes[4].set_title(f"Concatenated\n{concat_output.shape}")
axes[4].set_ylabel("Tokens")
axes[4].set_xlabel(f"First 80 dims\n(of {d_model})")
axes[4].set_yticks(range(num_tokens))
axes[4].set_yticklabels(tokens)

plt.suptitle("Concatenating Head Outputs", fontsize=14)
plt.tight_layout()
plt.show()

print("\nConcatenation: [Head1 | Head2 | Head3 | Head4] = Full output")

# COMMAND ----------

# DBTITLE 1,Step 8 - Final Linear Layer
# MAGIC %md
# MAGIC ## Step 8: Final Linear Projection
# MAGIC
# MAGIC Right now, concatenated output is just **4 perspectives stacked side-by-side**.
# MAGIC
# MAGIC We need to **blend them together** with a final linear layer:
# MAGIC
# MAGIC **W_O**: Output weight matrix (512 × 512)
# MAGIC
# MAGIC This acts as a **judge** that:
# MAGIC - Mixes information from all heads
# MAGIC - Balances which perspectives matter most
# MAGIC - Produces final unified representation

# COMMAND ----------

d_model

# COMMAND ----------

# DBTITLE 1,Apply Output Projection
# Final output projection matrix
W_O = np.random.randn(d_model, d_model) * 0.1

print(f"Output projection W_O shape: {W_O.shape}")

# Apply final linear transformation
final_output = concat_output @ W_O

print(f"\nFinal output shape: {final_output.shape}")
print(f"\n✓ Multi-head attention complete!")

# Compare input and output
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Input
axes[0].imshow(X[:, :50], aspect='auto', cmap='coolwarm')
axes[0].set_title(f"Input Embeddings X\n{X.shape}")
axes[0].set_ylabel("Tokens")
axes[0].set_xlabel("First 50 dimensions")
axes[0].set_yticks(range(num_tokens))
axes[0].set_yticklabels(tokens)

# Output
axes[1].imshow(final_output[:, :50], aspect='auto', cmap='coolwarm')
axes[1].set_title(f"Multi-Head Attention Output\n{final_output.shape}")
axes[1].set_ylabel("")
axes[1].set_xlabel("First 50 dimensions")
axes[1].set_yticks(range(num_tokens))
axes[1].set_yticklabels(tokens)

plt.suptitle("Input vs Output: Same Shape, Richer Representation", fontsize=14)
plt.tight_layout()
plt.show()

print("\n✨ Output now contains insights from 4 different attention perspectives!")

# COMMAND ----------

# DBTITLE 1,Complete Function
# MAGIC %md
# MAGIC ## Complete Multi-Head Attention Function
# MAGIC
# MAGIC Let's package everything into one clean implementation:

# COMMAND ----------

# DBTITLE 1,Multi-Head Attention Implementation
def multi_head_attention(X, num_heads, W_Q_list, W_K_list, W_V_list, W_O):
    """
    Complete multi-head attention mechanism
    
    Args:
        X: Input embeddings (num_tokens, d_model)
        num_heads: Number of attention heads
        W_Q_list: List of query weight matrices, one per head
        W_K_list: List of key weight matrices, one per head
        W_V_list: List of value weight matrices, one per head
        W_O: Output projection matrix (d_model, d_model)
    
    Returns:
        output: Final context-enriched representations (num_tokens, d_model)
        all_attention_weights: List of attention matrices from each head
    """
    head_outputs = []
    all_attention_weights = []
    
    # Process each head independently
    for i in range(num_heads):
        # Project to Q, K, V
        Q = X @ W_Q_list[i]
        K = X @ W_K_list[i]
        V = X @ W_V_list[i]
        
        # Run scaled dot-product attention
        head_out, attn_weights = scaled_dot_product_attention(Q, K, V)
        
        head_outputs.append(head_out)
        all_attention_weights.append(attn_weights)
    
    # Concatenate all head outputs
    concat_output = np.concatenate(head_outputs, axis=-1)
    
    # Apply final linear projection
    output = concat_output @ W_O
    
    return output, all_attention_weights

# Test the complete function
output_test, weights_test = multi_head_attention(X, num_heads, W_Q, W_K, W_V, W_O)

print("✅ Multi-head attention function working correctly!")
print(f"\nOutput shape: {output_test.shape}")
print(f"Number of attention weight matrices: {len(weights_test)}")

# Verify
assert np.allclose(output_test, final_output), "Output mismatch!"
print("\n✓ All verifications passed!")

# COMMAND ----------

# DBTITLE 1,Architecture Diagram
# MAGIC %md
# MAGIC ## Multi-Head Attention Architecture
# MAGIC
# MAGIC ```
# MAGIC Input X (3 × 512)
# MAGIC        |
# MAGIC        |
# MAGIC    [Split into 4 heads]
# MAGIC        |
# MAGIC        +-- Head 1 (3 × 128) --> Q1, K1, V1 --> Attention 1 --> Output 1 (3 × 128)
# MAGIC        |
# MAGIC        +-- Head 2 (3 × 128) --> Q2, K2, V2 --> Attention 2 --> Output 2 (3 × 128)
# MAGIC        |
# MAGIC        +-- Head 3 (3 × 128) --> Q3, K3, V3 --> Attention 3 --> Output 3 (3 × 128)
# MAGIC        |
# MAGIC        +-- Head 4 (3 × 128) --> Q4, K4, V4 --> Attention 4 --> Output 4 (3 × 128)
# MAGIC        |
# MAGIC        v
# MAGIC   [Concatenate: 3 × 512]
# MAGIC        |
# MAGIC        v
# MAGIC   [Linear W_O: 512 × 512]
# MAGIC        |
# MAGIC        v
# MAGIC   Final Output (3 × 512)
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Summary
# MAGIC %md
# MAGIC ## Summary: Multi-Head Attention
# MAGIC
# MAGIC ### Key Concepts:
# MAGIC
# MAGIC 1. **Multiple Perspectives**: Instead of one attention mechanism, run several in parallel
# MAGIC
# MAGIC 2. **Dimension Splitting**: 
# MAGIC    - Original: 512 dimensions
# MAGIC    - Split into 4 heads of 128 dimensions each
# MAGIC    - Same computational cost!
# MAGIC
# MAGIC 3. **Independent Learning**: Each head has its own weights, learns different patterns:
# MAGIC    - Grammar and syntax
# MAGIC    - Long-distance dependencies
# MAGIC    - Subject-verb relationships
# MAGIC    - Positional patterns
# MAGIC
# MAGIC 4. **Concatenation + Projection**: 
# MAGIC    - Concatenate all head outputs
# MAGIC    - Apply final linear layer to blend perspectives
# MAGIC
# MAGIC ### The Pipeline:
# MAGIC
# MAGIC ```
# MAGIC Input (3 × 512)
# MAGIC    ↓
# MAGIC Split → 4 heads of (3 × 128) each
# MAGIC    ↓
# MAGIC Parallel attention (each head independent)
# MAGIC    ↓
# MAGIC Concatenate → (3 × 512)
# MAGIC    ↓
# MAGIC Linear projection W_O
# MAGIC    ↓
# MAGIC Output (3 × 512) with 4 perspectives combined
# MAGIC ```
# MAGIC
# MAGIC ### Why It Works:
# MAGIC
# MAGIC - **Captures Ambiguity**: Different heads can capture different interpretations simultaneously
# MAGIC - **Efficient**: 4 perspectives for the cost of 1 (due to dimension splitting)
# MAGIC - **Specialized**: Each head specializes in different linguistic patterns
# MAGIC - **Robust**: If one head misses something, another head might catch it
# MAGIC
# MAGIC ### Original Transformer Paper:
# MAGIC - Used **8 heads**
# MAGIC - Model dimension: **512**
# MAGIC - Each head: **64 dimensions** (512 ÷ 8)
# MAGIC
# MAGIC ### What's Next?
# MAGIC
# MAGIC **Positional Encoding**: Multi-head attention has one problem - it's still **order-blind**! It can't tell the difference between "dog bites man" and "man bites dog". Next, we'll add positional information!