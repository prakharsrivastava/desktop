# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # Transformer Encoder - Complete Architecture
# MAGIC
# MAGIC ## Bringing It All Together
# MAGIC
# MAGIC We've built the individual pieces:
# MAGIC 1. **Self-Attention** - words understanding context from other words
# MAGIC 2. **Multi-Head Attention** - multiple perspectives in parallel
# MAGIC 3. **Positional Encoding** - adding position information
# MAGIC
# MAGIC Now we'll assemble the **complete encoder**!
# MAGIC
# MAGIC ## The Encoder Block
# MAGIC
# MAGIC Each encoder block has **4 main operations**:
# MAGIC
# MAGIC 1. **Multi-Head Self-Attention** - words attend to each other
# MAGIC 2. **Add & Norm** - residual connection + layer normalization
# MAGIC 3. **Feed-Forward Network** - independent transformation of each word
# MAGIC 4. **Add & Norm** - another residual connection + layer normalization
# MAGIC
# MAGIC ## Stacking Blocks
# MAGIC
# MAGIC The original Transformer paper stacks **6 identical encoder blocks**:
# MAGIC - Block 1 output → Block 2 input
# MAGIC - Block 2 output → Block 3 input
# MAGIC - ...
# MAGIC - Block 6 output = Final encoder output
# MAGIC
# MAGIC Each block refines the representation!

# COMMAND ----------

# DBTITLE 1,Setup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

np.random.seed(42)

print("✓ Setup complete")

# COMMAND ----------

# DBTITLE 1,Input Preparation
# MAGIC %md
# MAGIC ## Input Preparation
# MAGIC
# MAGIC Let's start with our example: **"Apple released new phones"**
# MAGIC
# MAGIC ### Steps:
# MAGIC 1. Tokenize into words
# MAGIC 2. Create word embeddings
# MAGIC 3. Add positional encoding

# COMMAND ----------

# DBTITLE 1,Tokenization and Embeddings
# Our sentence
tokens = ["Apple", "released", "new", "phones"]
num_tokens = len(tokens)
d_model = 512  # Standard embedding dimension

print(f"Sentence: {' '.join(tokens)}")
print(f"Number of tokens: {num_tokens}")
print(f"Embedding dimension: {d_model}")

# Create word embeddings
word_embeddings = np.random.randn(num_tokens, d_model) * 0.1

print(f"\nWord embeddings shape: {word_embeddings.shape}")
print(f"  - {num_tokens} tokens")
print(f"  - {d_model} dimensions each")

# COMMAND ----------

# DBTITLE 1,Positional Encoding
def get_positional_encoding(max_seq_len, d_model):
    """Generate sinusoidal positional encodings"""
    pos_encoding = np.zeros((max_seq_len, d_model))
    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    return pos_encoding

# Generate positional encoding
pos_encoding = get_positional_encoding(num_tokens, d_model)

# Add positional encoding to embeddings
X = word_embeddings + pos_encoding

print(f"Positional encoding shape: {pos_encoding.shape}")
print(f"\nInput matrix X (embeddings + position):")
print(f"  Shape: {X.shape}")
print(f"  ✓ Each token now has both meaning AND position information!")

# COMMAND ----------

# DBTITLE 1,Multi-Head Attention
# MAGIC %md
# MAGIC ## Component 1: Multi-Head Self-Attention
# MAGIC
# MAGIC First major component of the encoder block.

# COMMAND ----------

# DBTITLE 1,Multi-Head Attention Implementation
def scaled_dot_product_attention(Q, K, V):
    """Single-head scaled dot-product attention"""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
    output = attention_weights @ V
    return output, attention_weights

def multi_head_attention(X, num_heads, d_model):
    """
    Multi-head attention mechanism
    
    Args:
        X: Input matrix (num_tokens, d_model)
        num_heads: Number of attention heads
        d_model: Model dimension
    
    Returns:
        output: Attention output (num_tokens, d_model)
        all_attention_weights: List of attention matrices
    """
    d_k = d_model // num_heads
    
    # Create weight matrices for each head
    W_Q = [np.random.randn(d_model, d_k) * 0.1 for _ in range(num_heads)]
    W_K = [np.random.randn(d_model, d_k) * 0.1 for _ in range(num_heads)]
    W_V = [np.random.randn(d_model, d_k) * 0.1 for _ in range(num_heads)]
    
    head_outputs = []
    all_attention_weights = []
    
    # Process each head
    for i in range(num_heads):
        Q = X @ W_Q[i]
        K = X @ W_K[i]
        V = X @ W_V[i]
        
        head_out, attn_weights = scaled_dot_product_attention(Q, K, V)
        head_outputs.append(head_out)
        all_attention_weights.append(attn_weights)
    
    # Concatenate heads
    concat_output = np.concatenate(head_outputs, axis=-1)
    
    # Final projection
    W_O = np.random.randn(d_model, d_model) * 0.1
    output = concat_output @ W_O
    
    return output, all_attention_weights

print("✓ Multi-head attention implemented")

# COMMAND ----------

# DBTITLE 1,Apply Multi-Head Attention
# Configure multi-head attention
num_heads = 8  # Original transformer uses 8 heads

print(f"Running multi-head attention with {num_heads} heads...")

# Apply multi-head attention
Z, attention_weights = multi_head_attention(X, num_heads, d_model)

print(f"\nAttention output Z shape: {Z.shape}")
print(f"Number of attention weight matrices: {len(attention_weights)}")
print(f"\n✓ Multi-head attention complete!")
print(f"  - Input X: {X.shape}")
print(f"  - Output Z: {Z.shape}")
print(f"  - Shape preserved: ✓")

# COMMAND ----------

# DBTITLE 1,Residual Connection
# MAGIC %md
# MAGIC ## Component 2: Residual Connection (Skip Connection)
# MAGIC
# MAGIC Residual connections are crucial for training deep networks.
# MAGIC
# MAGIC **Formula**: `output = X + Attention(X)`
# MAGIC
# MAGIC ### Why?
# MAGIC - **Gradient flow**: Gradients can skip back through the shortcut
# MAGIC - **Stability**: Original input never gets lost
# MAGIC - **Depth**: Enables stacking many layers (even 100+)
# MAGIC - **Preservation**: Positional encoding signal rides along through all blocks

# COMMAND ----------

# DBTITLE 1,Add Residual Connection
# Add residual connection
residual_1 = X + Z

print(f"Before residual: X shape = {X.shape}, Z shape = {Z.shape}")
print(f"After residual: residual_1 shape = {residual_1.shape}")
print(f"\n✓ Residual connection: output = X + Attention(X)")
print(f"\nKey benefit: Even if attention 'messes up', original input X is preserved!")

# COMMAND ----------

# DBTITLE 1,Layer Normalization
# MAGIC %md
# MAGIC ## Component 3: Layer Normalization
# MAGIC
# MAGIC After residual addition, values can be all over the place. Layer norm stabilizes them.
# MAGIC
# MAGIC **Process**:
# MAGIC 1. For each token (each row), calculate mean and standard deviation
# MAGIC 2. Normalize: subtract mean, divide by std
# MAGIC 3. Result: mean = 0, std = 1 for each token
# MAGIC
# MAGIC ### Why Layer Norm (not Batch Norm)?
# MAGIC - **Language sequences** have variable lengths with padding
# MAGIC - Batch norm mixes tokens across sequences (padding corrupts statistics)
# MAGIC - **Layer norm** normalizes each sequence independently

# COMMAND ----------

# DBTITLE 1,Layer Norm Implementation
def layer_norm(x, epsilon=1e-6):
    """
    Layer normalization: normalize each row independently
    
    Args:
        x: Input matrix (num_tokens, d_model)
        epsilon: Small value for numerical stability
    
    Returns:
        Normalized matrix
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)

print("✓ Layer normalization implemented")

# Apply layer norm
H = layer_norm(residual_1)

print(f"\nAfter layer normalization:")
print(f"  Shape: {H.shape}")

# Verify normalization
for i, token in enumerate(tokens):
    row_mean = np.mean(H[i])
    row_std = np.std(H[i])
    print(f"  {token:8s}: mean = {row_mean:.6f}, std = {row_std:.6f}")

print(f"\n✓ Each token normalized independently (mean ≈ 0, std ≈ 1)")

# COMMAND ----------

# DBTITLE 1,Feed-Forward Network
# MAGIC %md
# MAGIC ## Component 4: Feed-Forward Network (FFN)
# MAGIC
# MAGIC ### What It Does:
# MAGIC - Attention **mixes** information between words
# MAGIC - FFN actually **transforms** the information
# MAGIC - Provides non-linearity and deeper computation
# MAGIC
# MAGIC ### Architecture:
# MAGIC ```
# MAGIC Input (d_model=512) 
# MAGIC    ↓
# MAGIC Linear layer 1: expand to 2048 (4× expansion)
# MAGIC    ↓
# MAGIC ReLU activation
# MAGIC    ↓
# MAGIC Linear layer 2: compress back to 512
# MAGIC    ↓
# MAGIC Output (d_model=512)
# MAGIC ```
# MAGIC
# MAGIC ### Key Points:
# MAGIC - **4× expansion** is standard (512 → 2048 → 512)
# MAGIC - Applied to **each token independently** (but in parallel)
# MAGIC - Contains ~2/3 of all transformer parameters!

# COMMAND ----------

# DBTITLE 1,Feed-Forward Implementation
def feed_forward(x, d_model, d_ff=2048):
    """
    Position-wise feed-forward network
    
    Args:
        x: Input (num_tokens, d_model)
        d_model: Model dimension
        d_ff: Hidden layer dimension (usually 4 * d_model)
    
    Returns:
        Output (num_tokens, d_model)
    """
    # First linear layer (expansion)
    W1 = np.random.randn(d_model, d_ff) * 0.1
    b1 = np.zeros(d_ff)
    
    hidden = x @ W1 + b1
    
    # ReLU activation
    hidden = np.maximum(0, hidden)
    
    # Second linear layer (compression)
    W2 = np.random.randn(d_ff, d_model) * 0.1
    b2 = np.zeros(d_model)
    
    output = hidden @ W2 + b2
    
    return output

print("✓ Feed-forward network implemented")

# Apply FFN
d_ff = 2048  # 4x expansion
F = feed_forward(H, d_model, d_ff)

print(f"\nFeed-forward network:")
print(f"  Input shape: {H.shape}")
print(f"  Expansion: {d_model} → {d_ff} ({d_ff//d_model}×)")
print(f"  Compression: {d_ff} → {d_model}")
print(f"  Output shape: {F.shape}")
print(f"\n✓ Each token transformed independently through FFN")

# COMMAND ----------

# DBTITLE 1,Second Add and Norm
# MAGIC %md
# MAGIC ## Component 5: Second Residual + Layer Norm
# MAGIC
# MAGIC Same as before:
# MAGIC 1. Add residual connection: `H + FFN(H)`
# MAGIC 2. Apply layer normalization

# COMMAND ----------

# DBTITLE 1,Second Add and Norm
# Second residual connection
residual_2 = H + F

print(f"Second residual connection: {H.shape} + {F.shape} = {residual_2.shape}")

# Second layer norm
output_block = layer_norm(residual_2)

print(f"\nAfter second layer norm:")
print(f"  Output shape: {output_block.shape}")

# Verify normalization
for i, token in enumerate(tokens):
    row_mean = np.mean(output_block[i])
    row_std = np.std(output_block[i])
    print(f"  {token:8s}: mean = {row_mean:.6f}, std = {row_std:.6f}")

print(f"\n✓ One encoder block complete!")
print(f"  Input: {X.shape}")
print(f"  Output: {output_block.shape}")
print(f"  Shape preserved: ✓")

# COMMAND ----------

# DBTITLE 1,Complete Encoder Block
# MAGIC %md
# MAGIC ## Complete Encoder Block Function
# MAGIC
# MAGIC Let's package everything into one function:

# COMMAND ----------

# DBTITLE 1,Encoder Block Implementation
def encoder_block(X, num_heads=8, d_model=512, d_ff=2048):
    """
    Complete transformer encoder block
    
    Args:
        X: Input (num_tokens, d_model)
        num_heads: Number of attention heads
        d_model: Model dimension
        d_ff: Feed-forward hidden dimension
    
    Returns:
        Output (num_tokens, d_model)
    """
    # 1. Multi-head self-attention
    Z, _ = multi_head_attention(X, num_heads, d_model)
    
    # 2. Add & Norm (first)
    X = layer_norm(X + Z)
    
    # 3. Feed-forward network
    F = feed_forward(X, d_model, d_ff)
    
    # 4. Add & Norm (second)
    X = layer_norm(X + F)
    
    return X

print("✓ Complete encoder block function defined")

# Test it
test_output = encoder_block(X, num_heads=8, d_model=d_model, d_ff=2048)

print(f"\nEncoder block test:")
print(f"  Input shape: {X.shape}")
print(f"  Output shape: {test_output.shape}")
print(f"  ✓ Single encoder block working correctly!")

# COMMAND ----------

# DBTITLE 1,Stacking Encoder Blocks
# MAGIC %md
# MAGIC ## Stacking Multiple Encoder Blocks
# MAGIC
# MAGIC The original Transformer uses **6 identical encoder blocks** stacked sequentially.
# MAGIC
# MAGIC ### Why Stack?
# MAGIC - **Progressive refinement**: Each layer captures increasingly abstract features
# MAGIC - **Specialization**: Early layers learn simple patterns, deeper layers learn complex relationships
# MAGIC - **Depth = Capacity**: More layers = more learning capacity
# MAGIC
# MAGIC ### The Flow:
# MAGIC ```
# MAGIC Input (with positional encoding)
# MAGIC    ↓
# MAGIC Encoder Block 1
# MAGIC    ↓
# MAGIC Encoder Block 2
# MAGIC    ↓
# MAGIC Encoder Block 3
# MAGIC    ↓
# MAGIC Encoder Block 4
# MAGIC    ↓
# MAGIC Encoder Block 5
# MAGIC    ↓
# MAGIC Encoder Block 6
# MAGIC    ↓
# MAGIC Final encoder output
# MAGIC ```
# MAGIC
# MAGIC Each block has **same structure**, but **different learned weights**.

# COMMAND ----------

# DBTITLE 1,Stack 6 Encoder Blocks
def transformer_encoder(X, num_blocks=6, num_heads=8, d_model=512, d_ff=2048):
    """
    Complete transformer encoder (stack of encoder blocks)
    
    Args:
        X: Input embeddings with positional encoding (num_tokens, d_model)
        num_blocks: Number of encoder blocks to stack
        num_heads: Number of attention heads per block
        d_model: Model dimension
        d_ff: Feed-forward hidden dimension
    
    Returns:
        Final encoder output (num_tokens, d_model)
    """
    block_outputs = [X]  # Store outputs from each block
    
    for block_num in range(num_blocks):
        X = encoder_block(X, num_heads, d_model, d_ff)
        block_outputs.append(X)
        print(f"Block {block_num + 1} complete: output shape {X.shape}")
    
    return X, block_outputs

print("Running complete transformer encoder (6 blocks)...")
print()

final_output, all_block_outputs = transformer_encoder(
    X, 
    num_blocks=6, 
    num_heads=8, 
    d_model=d_model, 
    d_ff=2048
)

print(f"\n✓ Complete encoder finished!")
print(f"  Input shape: {X.shape}")
print(f"  Final output shape: {final_output.shape}")
print(f"  Number of blocks: 6")
print(f"  Total outputs captured: {len(all_block_outputs)} (input + 6 blocks)")

# COMMAND ----------

# DBTITLE 1,Visualize Block Evolution
# MAGIC %md
# MAGIC ## Visualize How Representations Evolve
# MAGIC
# MAGIC Let's see how the encodings change through the 6 layers:

# COMMAND ----------

# DBTITLE 1,Evolution Visualization
# Visualize first 20 dimensions for each block
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx in range(7):  # 0 (input) + 6 blocks
    ax = axes[idx]
    data = all_block_outputs[idx][:, :20]  # First 20 dimensions
    
    im = ax.imshow(data, aspect='auto', cmap='coolwarm')
    
    if idx == 0:
        ax.set_title(f"Input\n(with pos encoding)", fontsize=10)
    else:
        ax.set_title(f"After Block {idx}", fontsize=10)
    
    ax.set_ylabel("Tokens")
    ax.set_yticks(range(num_tokens))
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("Dims 0-19")
    
    plt.colorbar(im, ax=ax)

# Hide the 8th subplot
axes[7].axis('off')

plt.suptitle("Representation Evolution Through 6 Encoder Blocks\n(First 20 dimensions)", 
             fontsize=14)
plt.tight_layout()
plt.show()

print("\n📊 Each block refines the representation!")
print("  - Early blocks: Basic patterns")
print("  - Middle blocks: Intermediate features")
print("  - Deep blocks: Abstract, context-rich representations")

# COMMAND ----------

# DBTITLE 1,Architecture Summary
# MAGIC %md
# MAGIC ## Complete Encoder Architecture Summary
# MAGIC
# MAGIC ### Input Processing:
# MAGIC ```
# MAGIC Tokens → Word Embeddings (d_model) → + Positional Encoding → Input Matrix X
# MAGIC ```
# MAGIC
# MAGIC ### Single Encoder Block:
# MAGIC ```
# MAGIC 1. Multi-Head Self-Attention (8 heads)
# MAGIC    Input: X (num_tokens, 512)
# MAGIC    Output: Z (num_tokens, 512)
# MAGIC    
# MAGIC 2. Add & Norm #1
# MAGIC    Output: LayerNorm(X + Z)
# MAGIC    
# MAGIC 3. Feed-Forward Network
# MAGIC    512 → 2048 → 512
# MAGIC    ReLU activation
# MAGIC    
# MAGIC 4. Add & Norm #2
# MAGIC    Output: LayerNorm(X + FFN(X))
# MAGIC ```
# MAGIC
# MAGIC ### Full Encoder:
# MAGIC ```
# MAGIC Stack 6 identical blocks (same structure, different weights)
# MAGIC Block 1 → Block 2 → ... → Block 6 → Final Output
# MAGIC ```
# MAGIC
# MAGIC ### Key Properties:
# MAGIC
# MAGIC ✅ **Parallel Processing**: All tokens processed simultaneously
# MAGIC ✅ **Residual Connections**: Enable deep networks (6+ layers)
# MAGIC ✅ **Layer Normalization**: Stabilize training
# MAGIC ✅ **Multi-Head Attention**: Multiple perspectives
# MAGIC ✅ **Position-Aware**: Positional encoding throughout
# MAGIC ✅ **Context-Rich**: Each token aware of full sentence

# COMMAND ----------

# DBTITLE 1,Final Statistics
# Calculate some statistics about the final output
print("Final Encoder Output Statistics:")
print(f"  Shape: {final_output.shape}")
print(f"  - {final_output.shape[0]} tokens")
print(f"  - {final_output.shape[1]} dimensions")
print()

print("Per-token statistics:")
for i, token in enumerate(tokens):
    token_vec = final_output[i]
    print(f"  {token:8s}:")
    print(f"    Mean: {np.mean(token_vec):8.4f}")
    print(f"    Std:  {np.std(token_vec):8.4f}")
    print(f"    Min:  {np.min(token_vec):8.4f}")
    print(f"    Max:  {np.max(token_vec):8.4f}")
    print()

print("✓ Each token now contains:")
print("  - Original word meaning")
print("  - Position information")
print("  - Context from all other words")
print("  - Refined through 6 layers of processing")

# COMMAND ----------

# DBTITLE 1,Original Transformer Configuration
# MAGIC %md
# MAGIC ## Original Transformer Paper Configuration
# MAGIC
# MAGIC ### From "Attention Is All You Need" (2017)
# MAGIC
# MAGIC **Model Dimensions:**
# MAGIC - `d_model` = 512 (embedding dimension)
# MAGIC - `num_heads` = 8 (attention heads)
# MAGIC - `d_k` = 64 (dimension per head = 512 / 8)
# MAGIC - `d_ff` = 2048 (feed-forward hidden dimension = 4 × 512)
# MAGIC
# MAGIC **Architecture:**
# MAGIC - **6 encoder blocks** (identical structure, different weights)
# MAGIC - **6 decoder blocks** (not covered here)
# MAGIC
# MAGIC **Other Hyperparameters:**
# MAGIC - Dropout: 0.1
# MAGIC - Label smoothing: 0.1
# MAGIC - Optimizer: Adam
# MAGIC - Learning rate: Custom schedule with warmup
# MAGIC
# MAGIC **Parameter Count:**
# MAGIC - Encoder parameters: ~30 million
# MAGIC - Full model (encoder + decoder): ~65 million
# MAGIC
# MAGIC ### Modern Variants:
# MAGIC
# MAGIC **BERT** (2018):
# MAGIC - BERT-Base: 12 encoder blocks, 768 dimensions, 110M params
# MAGIC - BERT-Large: 24 encoder blocks, 1024 dimensions, 340M params
# MAGIC
# MAGIC **GPT-2** (2019):
# MAGIC - GPT-2: Decoder-only, 12-48 layers, 117M-1.5B params
# MAGIC
# MAGIC **GPT-3** (2020):
# MAGIC - 96 layers, 12,288 dimensions, 175B params!

# COMMAND ----------

# DBTITLE 1,Applications
# MAGIC %md
# MAGIC ## What Can You Do With Encoder Outputs?
# MAGIC
# MAGIC ### 1. Text Classification
# MAGIC - Add special `[CLS]` token at start
# MAGIC - After encoding, use `[CLS]` token's final vector
# MAGIC - Feed through linear layer → softmax
# MAGIC - **Use cases**: Sentiment analysis, spam detection, topic classification
# MAGIC
# MAGIC ### 2. Named Entity Recognition (NER)
# MAGIC - Use final vector for **each token**
# MAGIC - Classify each word independently
# MAGIC - **Use cases**: Extract person names, locations, organizations, dates
# MAGIC
# MAGIC ### 3. Semantic Search
# MAGIC - Encode query and documents
# MAGIC - Average token vectors to get sentence embeddings
# MAGIC - Compare using cosine similarity
# MAGIC - **Use cases**: Search engines, document retrieval, question answering
# MAGIC
# MAGIC ### 4. Question Answering
# MAGIC - Encode question and paragraph together
# MAGIC - Predict start and end positions of answer span
# MAGIC - **Use cases**: Reading comprehension, FAQ systems
# MAGIC
# MAGIC ### 5. Masked Language Modeling (Training)
# MAGIC - Mask random words during training
# MAGIC - Encoder predicts masked words from context
# MAGIC - **Use cases**: Pre-training models like BERT
# MAGIC
# MAGIC ### Key Advantage of Encoders:
# MAGIC
# MAGIC **Bidirectional Context**: Encoders see both left AND right context simultaneously!
# MAGIC
# MAGIC Compare to decoder-only models (GPT):
# MAGIC - Decoders: Only see left context (past words)
# MAGIC - Encoders: See full sentence (past + future)
# MAGIC
# MAGIC This makes encoders especially powerful for **understanding** tasks!

# COMMAND ----------

# DBTITLE 1,Summary
# MAGIC %md
# MAGIC ## Complete Transformer Encoder - Summary
# MAGIC
# MAGIC ### What We Built:
# MAGIC
# MAGIC A complete transformer encoder from scratch that:
# MAGIC 1. Takes word tokens as input
# MAGIC 2. Converts them to embeddings
# MAGIC 3. Adds positional information
# MAGIC 4. Processes through 6 encoder blocks
# MAGIC 5. Outputs context-rich representations
# MAGIC
# MAGIC ### Each Encoder Block Contains:
# MAGIC
# MAGIC ```
# MAGIC 1. Multi-Head Self-Attention (8 heads)
# MAGIC    - Each head: 64 dimensions
# MAGIC    - Captures different relationship types
# MAGIC    
# MAGIC 2. Residual Connection + Layer Norm
# MAGIC    - Enables deep networks
# MAGIC    - Stabilizes training
# MAGIC    
# MAGIC 3. Feed-Forward Network (512 → 2048 → 512)
# MAGIC    - ReLU activation
# MAGIC    - Transforms information
# MAGIC    
# MAGIC 4. Residual Connection + Layer Norm
# MAGIC    - Second stabilization point
# MAGIC ```
# MAGIC
# MAGIC ### Key Innovations:
# MAGIC
# MAGIC ✅ **Self-Attention**: Words understand each other in context
# MAGIC ✅ **Multi-Head**: Multiple perspectives simultaneously
# MAGIC ✅ **Positional Encoding**: Order information preserved
# MAGIC ✅ **Residual Connections**: Enable 6+ layer depth
# MAGIC ✅ **Layer Normalization**: Stable training
# MAGIC ✅ **Parallel Processing**: All tokens processed at once
# MAGIC
# MAGIC ### Why It Works:
# MAGIC
# MAGIC 1. **No Sequential Bottleneck**: Unlike RNNs, processes all tokens in parallel
# MAGIC 2. **Direct Connections**: Every word can attend directly to every other word
# MAGIC 3. **Scalable**: Can stack many layers without gradient problems
# MAGIC 4. **Flexible**: Same architecture works for many NLP tasks
# MAGIC
# MAGIC ### Original Paper:
# MAGIC
# MAGIC **"Attention Is All You Need"** (Vaswani et al., 2017)
# MAGIC - Introduced the transformer architecture
# MAGIC - Revolutionized NLP
# MAGIC - Foundation for BERT, GPT, and all modern LLMs
# MAGIC
# MAGIC ### What We Didn't Cover (Future Topics):
# MAGIC
# MAGIC - **Decoder**: For generation tasks (translation, text generation)
# MAGIC - **Cross-Attention**: How decoder attends to encoder output
# MAGIC - **Training**: Loss functions, optimization, schedules
# MAGIC - **Inference**: Beam search, sampling strategies
# MAGIC - **Fine-tuning**: Adapting pre-trained models to specific tasks
# MAGIC
# MAGIC ### Congratulations! 🎉
# MAGIC
# MAGIC You now understand the transformer encoder from the ground up:
# MAGIC - Mathematical foundations
# MAGIC - Implementation details  
# MAGIC - Architecture design choices
# MAGIC - Real-world applications
# MAGIC
# MAGIC This is the foundation of modern NLP and the backbone of models like BERT, RoBERTa, ALBERT, and the encoder part of T5!

# COMMAND ----------

# DBTITLE 1,Practical Use Cases Guide
# MAGIC %md
# MAGIC # 🎯 How to Use These Notebooks - Practical Applications
# MAGIC
# MAGIC ## 💡 TL;DR - Quick Answer
# MAGIC
# MAGIC **For 99% of real projects**: Don't rebuild transformers from scratch! Use **HuggingFace Transformers** library with pre-trained models.
# MAGIC
# MAGIC **These notebooks teach you**:
# MAGIC - ✅ How transformers work internally (foundation knowledge)
# MAGIC - ✅ Why architecture choices matter
# MAGIC - ✅ How to debug and optimize models
# MAGIC
# MAGIC **For production**: Use existing battle-tested implementations!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 Path 1: Using Pre-Trained Models (Recommended)
# MAGIC
# MAGIC ### Install HuggingFace
# MAGIC ```python
# MAGIC %pip install transformers torch
# MAGIC ```
# MAGIC
# MAGIC ### Example 1: Sentiment Analysis
# MAGIC ```python
# MAGIC from transformers import pipeline
# MAGIC
# MAGIC # One line to load pre-trained model!
# MAGIC classifier = pipeline("sentiment-analysis")
# MAGIC
# MAGIC # Use it
# MAGIC result = classifier("I absolutely love this product!")
# MAGIC print(result)
# MAGIC # Output: [{'label': 'POSITIVE', 'score': 0.9998}]
# MAGIC ```
# MAGIC
# MAGIC ### Example 2: Named Entity Recognition
# MAGIC ```python
# MAGIC ner = pipeline("ner", grouped_entities=True)
# MAGIC
# MAGIC text = "Elon Musk founded SpaceX in 2002"
# MAGIC entities = ner(text)
# MAGIC
# MAGIC for entity in entities:
# MAGIC     print(f"{entity['word']}: {entity['entity_group']}")
# MAGIC # Output:
# MAGIC # Elon Musk: PER (Person)
# MAGIC # SpaceX: ORG (Organization)  
# MAGIC # 2002: DATE
# MAGIC ```
# MAGIC
# MAGIC ### Example 3: Semantic Search
# MAGIC ```python
# MAGIC from transformers import AutoTokenizer, AutoModel
# MAGIC import torch
# MAGIC
# MAGIC model_name = "sentence-transformers/all-MiniLM-L6-v2"
# MAGIC tokenizer = AutoTokenizer.from_pretrained(model_name)
# MAGIC model = AutoModel.from_pretrained(model_name)
# MAGIC
# MAGIC def get_embedding(text):
# MAGIC     inputs = tokenizer(text, return_tensors="pt", padding=True)
# MAGIC     with torch.no_grad():
# MAGIC         outputs = model(**inputs)
# MAGIC     return outputs.last_hidden_state.mean(dim=1)  # Mean pooling
# MAGIC
# MAGIC # Get embeddings
# MAGIC query_emb = get_embedding("cat on mat")
# MAGIC doc1_emb = get_embedding("feline on carpet")
# MAGIC doc2_emb = get_embedding("machine learning")
# MAGIC
# MAGIC # Compute similarity
# MAGIC from torch.nn.functional import cosine_similarity
# MAGIC sim1 = cosine_similarity(query_emb, doc1_emb)
# MAGIC sim2 = cosine_similarity(query_emb, doc2_emb)
# MAGIC
# MAGIC print(f"Query vs Doc1: {sim1.item():.3f}")  # High - semantic match!
# MAGIC print(f"Query vs Doc2: {sim2.item():.3f}")  # Low - different topic
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🏗️ Path 2: Fine-Tuning for Custom Tasks
# MAGIC
# MAGIC ### When to fine-tune:
# MAGIC - ✅ You have labeled data for your specific task
# MAGIC - ✅ Domain-specific language (medical, legal, technical)
# MAGIC - ✅ Need better accuracy than general models
# MAGIC
# MAGIC ### Example: Custom Text Classifier
# MAGIC ```python
# MAGIC from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# MAGIC
# MAGIC # Load pre-trained model + add task-specific head
# MAGIC model = AutoModelForSequenceClassification.from_pretrained(
# MAGIC     "bert-base-uncased",
# MAGIC     num_labels=3  # Your number of classes
# MAGIC )
# MAGIC tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# MAGIC
# MAGIC # Prepare your data
# MAGIC train_texts = ["text 1", "text 2", ...]
# MAGIC train_labels = [0, 1, ...]  # Your labels
# MAGIC
# MAGIC # Tokenize
# MAGIC train_encodings = tokenizer(train_texts, truncation=True, padding=True)
# MAGIC
# MAGIC # Create dataset
# MAGIC import torch
# MAGIC class CustomDataset(torch.utils.data.Dataset):
# MAGIC     def __init__(self, encodings, labels):
# MAGIC         self.encodings = encodings
# MAGIC         self.labels = labels
# MAGIC     
# MAGIC     def __getitem__(self, idx):
# MAGIC         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
# MAGIC         item['labels'] = torch.tensor(self.labels[idx])
# MAGIC         return item
# MAGIC     
# MAGIC     def __len__(self):
# MAGIC         return len(self.labels)
# MAGIC
# MAGIC train_dataset = CustomDataset(train_encodings, train_labels)
# MAGIC
# MAGIC # Training configuration
# MAGIC training_args = TrainingArguments(
# MAGIC     output_dir='./results',
# MAGIC     num_train_epochs=3,
# MAGIC     per_device_train_batch_size=16,
# MAGIC     learning_rate=2e-5,
# MAGIC     evaluation_strategy="epoch",
# MAGIC )
# MAGIC
# MAGIC # Train!
# MAGIC trainer = Trainer(
# MAGIC     model=model,
# MAGIC     args=training_args,
# MAGIC     train_dataset=train_dataset,
# MAGIC )
# MAGIC
# MAGIC trainer.train()
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 Real-World Use Cases
# MAGIC
# MAGIC ### Use Case 1: Customer Support Ticket Classifier
# MAGIC
# MAGIC **Goal**: Automatically route support tickets to correct department
# MAGIC
# MAGIC ```python
# MAGIC from transformers import pipeline
# MAGIC
# MAGIC # Fine-tuned classifier on your ticket data
# MAGIC classifier = pipeline(
# MAGIC     "text-classification",
# MAGIC     model="your-org/support-ticket-classifier"  # Your fine-tuned model
# MAGIC )
# MAGIC
# MAGIC # Classify incoming ticket
# MAGIC ticket = "My credit card payment failed and I can't access my account"
# MAGIC result = classifier(ticket)
# MAGIC
# MAGIC print(f"Route to: {result[0]['label']}")  
# MAGIC # Output: "billing_department"
# MAGIC print(f"Confidence: {result[0]['score']:.2%}")  
# MAGIC # Output: "94.5%"
# MAGIC ```
# MAGIC
# MAGIC ### Use Case 2: Document Similarity Engine
# MAGIC
# MAGIC **Goal**: Find similar documents for search/recommendations
# MAGIC
# MAGIC ```python
# MAGIC import numpy as np
# MAGIC from transformers import AutoTokenizer, AutoModel
# MAGIC import torch
# MAGIC
# MAGIC class DocumentSimilarityEngine:
# MAGIC     def __init__(self):
# MAGIC         self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# MAGIC         self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# MAGIC         self.documents = []
# MAGIC         self.embeddings = []
# MAGIC     
# MAGIC     def add_document(self, doc_id, text):
# MAGIC         self.documents.append({' id': doc_id, 'text': text})
# MAGIC         embedding = self._get_embedding(text)
# MAGIC         self.embeddings.append(embedding)
# MAGIC     
# MAGIC     def _get_embedding(self, text):
# MAGIC         inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
# MAGIC         with torch.no_grad():
# MAGIC             outputs = self.model(**inputs)
# MAGIC         return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
# MAGIC     
# MAGIC     def search(self, query, top_k=5):
# MAGIC         query_emb = self._get_embedding(query)
# MAGIC         
# MAGIC         # Compute similarities
# MAGIC         similarities = []
# MAGIC         for doc, emb in zip(self.documents, self.embeddings):
# MAGIC             sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
# MAGIC             similarities.append((doc, sim))
# MAGIC         
# MAGIC         # Sort and return top k
# MAGIC         similarities.sort(key=lambda x: x[1], reverse=True)
# MAGIC         return similarities[:top_k]
# MAGIC
# MAGIC # Usage
# MAGIC engine = DocumentSimilarityEngine()
# MAGIC
# MAGIC # Index documents
# MAGIC engine.add_document("doc1", "Python is a programming language")
# MAGIC engine.add_document("doc2", "Machine learning requires data")
# MAGIC engine.add_document("doc3", "Cats make great pets")
# MAGIC
# MAGIC # Search
# MAGIC results = engine.search("coding in Python")
# MAGIC for doc, score in results:
# MAGIC     print(f"{score:.3f}: {doc['text']}")
# MAGIC ```
# MAGIC
# MAGIC ### Use Case 3: Question Answering System
# MAGIC
# MAGIC **Goal**: Extract answers from documents
# MAGIC
# MAGIC ```python
# MAGIC from transformers import pipeline
# MAGIC
# MAGIC qa_model = pipeline(
# MAGIC     "question-answering",
# MAGIC     model="distilbert-base-cased-distilled-squad"
# MAGIC )
# MAGIC
# MAGIC # Your knowledge base
# MAGIC context = """
# MAGIC The transformer architecture was introduced in 2017 by Vaswani et al.
# MAGIC It uses self-attention to process sequences in parallel, unlike RNNs.
# MAGIC This makes transformers much faster to train.
# MAGIC """
# MAGIC
# MAGIC # Ask questions
# MAGIC questions = [
# MAGIC     "When was the transformer introduced?",
# MAGIC     "Who introduced transformers?",
# MAGIC     "What is the advantage over RNNs?"
# MAGIC ]
# MAGIC
# MAGIC for question in questions:
# MAGIC     result = qa_model(question=question, context=context)
# MAGIC     print(f"Q: {question}")
# MAGIC     print(f"A: {result['answer']} (confidence: {result['score']:.1%})\n")
# MAGIC
# MAGIC # Output:
# MAGIC # Q: When was the transformer introduced?
# MAGIC # A: 2017 (confidence: 98.5%)
# MAGIC #
# MAGIC # Q: Who introduced transformers?
# MAGIC # A: Vaswani et al. (confidence: 95.2%)
# MAGIC #
# MAGIC # Q: What is the advantage over RNNs?
# MAGIC # A: much faster to train (confidence: 87.3%)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔬 When to Use These Notebooks Directly
# MAGIC
# MAGIC ### 1. **Learning & Understanding**
# MAGIC
# MAGIC You're here! ✅ These notebooks teach you:
# MAGIC - How attention mechanisms work mathematically
# MAGIC - Why multi-head attention captures different patterns
# MAGIC - How positional encoding preserves order
# MAGIC - Complete architecture from embeddings to output
# MAGIC
# MAGIC **Benefit**: Deep understanding helps you:
# MAGIC - Debug model issues
# MAGIC - Choose right architectures
# MAGIC - Read research papers
# MAGIC - Explain to others
# MAGIC
# MAGIC ### 2. **Research & Experimentation**
# MAGIC
# MAGIC Modify components to test ideas:
# MAGIC
# MAGIC ```python
# MAGIC # Example: Add learned position bias to attention
# MAGIC def custom_attention_with_bias(Q, K, V):
# MAGIC     # Your innovation!
# MAGIC     position_bias = nn.Parameter(torch.randn(seq_len, seq_len))
# MAGIC     scores = (Q @ K.T / np.sqrt(d_k)) + position_bias
# MAGIC     # ... rest of attention
# MAGIC     
# MAGIC # Test vs standard attention
# MAGIC performance_standard = evaluate(standard_attention)
# MAGIC performance_custom = evaluate(custom_attention_with_bias)
# MAGIC
# MAGIC print(f"Improvement: {performance_custom - performance_standard}")
# MAGIC ```
# MAGIC
# MAGIC ### 3. **Educational Tools**
# MAGIC
# MAGIC Create visualizations for teaching:
# MAGIC
# MAGIC ```python
# MAGIC # Interactive attention visualizer
# MAGIC def visualize_attention_patterns(sentence):
# MAGIC     output, attention_weights = encoder_block(sentence)
# MAGIC     
# MAGIC     for head_idx, weights in enumerate(attention_weights):
# MAGIC         plt.figure(figsize=(8, 6))
# MAGIC         sns.heatmap(weights, annot=True, cmap="Blues")
# MAGIC         plt.title(f"Head {head_idx+1}: Attention Pattern")
# MAGIC         plt.xlabel("Attending to (Key)")
# MAGIC         plt.ylabel("Attention from (Query)")
# MAGIC         plt.show()
# MAGIC         
# MAGIC         # Explain what this head learned
# MAGIC         analyze_pattern(weights)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Decision Tree: Which Approach?
# MAGIC
# MAGIC ```
# MAGIC Do you need to use transformers?
# MAGIC │
# MAGIC ├── YES → Is this for production/real project?
# MAGIC │        │
# MAGIC │        ├── YES → Use HuggingFace Transformers
# MAGIC │        │        │
# MAGIC │        │        ├── Pre-trained model works? → Use it directly!
# MAGIC │        │        └── Need customization? → Fine-tune on your data
# MAGIC │        │
# MAGIC │        └── NO (learning/research) → Use these notebooks!
# MAGIC │                 │
# MAGIC │                 ├── Understand concepts → Run & modify code
# MAGIC │                 ├── Test new ideas → Change architecture
# MAGIC │                 └── Teach others → Create visualizations
# MAGIC │
# MAGIC └── NO → Learn why transformers are powerful anyway!
# MAGIC           Deep understanding = better ML practitioner
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 Getting Started Checklist
# MAGIC
# MAGIC ### For Production Projects:
# MAGIC
# MAGIC - [ ] Install HuggingFace: `pip install transformers torch`
# MAGIC - [ ] Choose pre-trained model from [HuggingFace Hub](https://huggingface.co/models)
# MAGIC - [ ] Test on your data
# MAGIC - [ ] If accuracy insufficient, fine-tune
# MAGIC - [ ] Deploy (FastAPI + Docker)
# MAGIC
# MAGIC ### For Learning:
# MAGIC
# MAGIC - [ ] Run all 4 notebooks in order
# MAGIC - [ ] Modify code, see what breaks
# MAGIC - [ ] Change dimensions, heads, layers
# MAGIC - [ ] Visualize attention patterns
# MAGIC - [ ] Read original paper: "Attention Is All You Need"
# MAGIC
# MAGIC ### For Research:
# MAGIC
# MAGIC - [ ] Understand baseline (these notebooks)
# MAGIC - [ ] Implement your modification
# MAGIC - [ ] Compare performance
# MAGIC - [ ] Write paper!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Resources
# MAGIC
# MAGIC ### Documentation:
# MAGIC - **HuggingFace Transformers**: https://huggingface.co/docs/transformers
# MAGIC - **Original Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
# MAGIC - **BERT Paper**: [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
# MAGIC
# MAGIC ### Tutorials:
# MAGIC - HuggingFace Course: https://huggingface.co/course
# MAGIC - PyTorch Transformer Tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# MAGIC
# MAGIC ### Pre-trained Models:
# MAGIC - **HuggingFace Hub**: 100,000+ models: https://huggingface.co/models
# MAGIC - **BERT variants**: BERT, RoBERTa, ALBERT, DistilBERT
# MAGIC - **Sentence Embeddings**: sentence-transformers
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✨ Key Takeaway
# MAGIC
# MAGIC **You've learned the "how" - now leverage existing tools!**
# MAGIC
# MAGIC ### What These Notebooks Give You:
# MAGIC ✅ Deep understanding of transformer internals
# MAGIC ✅ Ability to debug and optimize models
# MAGIC ✅ Foundation for reading research papers
# MAGIC ✅ Knowledge to innovate new architectures
# MAGIC
# MAGIC ### For Real Projects:
# MAGIC 🔧 Use HuggingFace Transformers (battle-tested)
# MAGIC 🔧 Start with pre-trained models
# MAGIC 🔧 Fine-tune for your domain
# MAGIC 🔧 Only build from scratch for research
# MAGIC
# MAGIC ### You're Ready To:
# MAGIC 1. **Build production systems** with pre-trained models
# MAGIC 2. **Fine-tune** models for custom tasks
# MAGIC 3. **Debug** transformer behavior
# MAGIC 4. **Research** new architectures
# MAGIC 5. **Teach** others how transformers work
# MAGIC
# MAGIC **Congratulations on completing the transformer encoder journey! 🎉**