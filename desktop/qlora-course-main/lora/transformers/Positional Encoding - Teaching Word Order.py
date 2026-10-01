# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # Positional Encoding: Teaching Transformers Word Order
# MAGIC
# MAGIC ## The Order-Blindness Problem
# MAGIC
# MAGIC Transformers process all words **simultaneously in parallel** — but this creates a critical problem:
# MAGIC
# MAGIC ### They have NO inherent sense of word order!
# MAGIC
# MAGIC Consider these two sentences:
# MAGIC
# MAGIC 1. **"Dog bites man"** 🐕 → 👤
# MAGIC 2. **"Man bites dog"** 👤 → 🐕
# MAGIC
# MAGIC **Completely different meanings!** But to a transformer without positional encoding:
# MAGIC
# MAGIC ```
# MAGIC Both sentences contain: {"Dog", "bites", "man"}
# MAGIC ↓
# MAGIC Same words, same weights
# MAGIC ↓
# MAGIC IDENTICAL attention patterns!
# MAGIC ```
# MAGIC
# MAGIC The transformer treats them as a **"bag of words"** — it sees the same three words and produces the same output regardless of order.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## The Solution: Positional Encoding
# MAGIC
# MAGIC **Positional encoding adds a unique mathematical fingerprint to each position** so the model knows where each word sits.
# MAGIC
# MAGIC ```
# MAGIC "Dog bites man":
# MAGIC Dog₀   + position_0_encoding
# MAGIC bites₁ + position_1_encoding  
# MAGIC man₂   + position_2_encoding
# MAGIC
# MAGIC "Man bites dog":
# MAGIC Man₀   + position_0_encoding
# MAGIC bites₁ + position_1_encoding
# MAGIC Dog₂   + position_2_encoding
# MAGIC
# MAGIC Now they're different!
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## In This Notebook:
# MAGIC
# MAGIC We'll demonstrate:
# MAGIC 1. **The Problem**: Self-attention without positional encoding produces identical patterns for different word orders
# MAGIC 2. **The Solution**: How sinusoidal positional encoding makes each position unique
# MAGIC 3. **Visualization**: Before/after comparison showing how positional encoding rescues word order
# MAGIC 4. **Math**: The elegant sinusoidal formula that scales to any sequence length

# COMMAND ----------

# DBTITLE 1,Setup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Rectangle

np.random.seed(42)

print("✓ Libraries imported")

# COMMAND ----------

# DBTITLE 1,The Problem
# MAGIC %md
# MAGIC ## The Order-Blindness Problem
# MAGIC
# MAGIC Let's demonstrate with our two sentences:
# MAGIC - **"Dog bites man"**
# MAGIC - **"Man bites dog"**
# MAGIC
# MAGIC First, we'll show what self-attention looks like **WITHOUT** positional encoding.

# COMMAND ----------

# DBTITLE 1,Create Example Sentences
# Two sentences with different word orders
sentence1 = ["Dog", "bites", "man"]
sentence2 = ["Man", "bites", "dog"]

print("Sentence 1:", " ".join(sentence1))
print("Meaning: 🐕 attacks 👤\n")

print("Sentence 2:", " ".join(sentence2))
print("Meaning: 👤 attacks 🐕\n")

print("="*50)
print("CRITICAL OBSERVATION:")
print("Same words: {", ", ".join(sorted(set(sentence1))), "}")
print("Just different ORDER!")
print("="*50)

# COMMAND ----------

# DBTITLE 1,Create Word Embeddings
# Create word embeddings (same embeddings for same words)
d_model = 8  # Small dimension for visualization

# Vocabulary - include all variations
vocab = {"Dog": 0, "bites": 1, "man": 2, "Man": 2, "dog": 0}
# Note: "Man" and "man" map to same embedding (index 2)
# Note: "Dog" and "dog" map to same embedding (index 0)

# Create fixed embeddings for each unique word
np.random.seed(42)
word_embedding_matrix = np.random.randn(3, d_model) * 0.5  # 3 unique words

print("Word Embedding Matrix:")
print(f"Shape: {word_embedding_matrix.shape} (3 unique words × {d_model} dimensions)\n")

print("Word embeddings (by index):")
print(f"  Index 0 (Dog/dog): {word_embedding_matrix[0][:4]}... (showing first 4 dims)")
print(f"  Index 1 (bites):   {word_embedding_matrix[1][:4]}... (showing first 4 dims)")
print(f"  Index 2 (Man/man): {word_embedding_matrix[2][:4]}... (showing first 4 dims)")

print("\n⚠️  Key point: Same word = Same embedding everywhere")
print("    'Dog' (any case) always has the same vector, regardless of position!")
print("    'Man' (any case) always has the same vector, regardless of position!")

# COMMAND ----------

# DBTITLE 1,Self-Attention WITHOUT Positional Encoding
def self_attention_no_position(sentence, word_embeddings, vocab):
    """
    Compute self-attention WITHOUT positional encoding
    
    Returns:
        attention_weights: Attention pattern
        embeddings: Word embeddings used
    """
    # Get embeddings for this sentence
    embeddings = np.array([word_embeddings[vocab[word]] for word in sentence])
    
    # Create Q, K, V projection matrices
    W_Q = np.random.randn(d_model, d_model) * 0.1
    W_K = np.random.randn(d_model, d_model) * 0.1
    W_V = np.random.randn(d_model, d_model) * 0.1
    
    Q = embeddings @ W_Q
    K = embeddings @ W_K
    V = embeddings @ W_V
    
    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_model)
    
    # Softmax
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
    
    return attention_weights, embeddings

# Use SAME random seed for both to ensure identical W_Q, W_K, W_V
np.random.seed(100)
attention1_no_pos, emb1 = self_attention_no_position(sentence1, word_embedding_matrix, vocab)

np.random.seed(100)  # Same seed!
attention2_no_pos, emb2 = self_attention_no_position(sentence2, word_embedding_matrix, vocab)

print("WITHOUT Positional Encoding:\n")
print("Sentence 1 embeddings shape:", emb1.shape)
print("Sentence 2 embeddings shape:", emb2.shape)
print(f"\nAre the embedding SETS identical? {np.allclose(np.sort(emb1, axis=0), np.sort(emb2, axis=0))}")
print("(Sorting because same words appear in different positions)\n")

print("="*50)
print("⚠️  PROBLEM: Both sentences produce SIMILAR attention patterns")
print("    because the model can't distinguish positions!")
print("="*50)

# COMMAND ----------

# DBTITLE 1,Visualize Attention WITHOUT Position
# Visualize both attention patterns side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Sentence 1
sns.heatmap(attention1_no_pos, annot=True, fmt='.3f', cmap='YlOrRd',
           xticklabels=sentence1, yticklabels=sentence1, ax=ax1,
           vmin=0, vmax=1, cbar_kws={'label': 'Attention Weight'})
ax1.set_title('"Dog bites man"\n(WITHOUT Positional Encoding)', fontsize=12, weight='bold')
ax1.set_xlabel('Attending to', fontsize=10)
ax1.set_ylabel('Attending from', fontsize=10)

# Sentence 2
sns.heatmap(attention2_no_pos, annot=True, fmt='.3f', cmap='YlOrRd',
           xticklabels=sentence2, yticklabels=sentence2, ax=ax2,
           vmin=0, vmax=1, cbar_kws={'label': 'Attention Weight'})
ax2.set_title('"Man bites dog"\n(WITHOUT Positional Encoding)', fontsize=12, weight='bold')
ax2.set_xlabel('Attending to', fontsize=10)
ax2.set_ylabel('Attending from', fontsize=10)

plt.suptitle('⚠️  Problem: Cannot Distinguish Word Order!', fontsize=14, weight='bold', color='red')
plt.tight_layout()
plt.show()

print("\n🔍 Observation:")
print("   The attention patterns are VERY SIMILAR!")
print("   The model treats both as 'bag of words': {Dog, bites, man}")
print("   ❌ It can't tell 'Dog bites man' from 'Man bites dog'!")

# COMMAND ----------

# DBTITLE 1,The Solution
# MAGIC %md
# MAGIC ## The Solution: Positional Encoding
# MAGIC
# MAGIC We need to **inject position information** into the embeddings so the model knows:
# MAGIC - Position 0: First word
# MAGIC - Position 1: Second word  
# MAGIC - Position 2: Third word
# MAGIC
# MAGIC ### Requirements:
# MAGIC
# MAGIC 1. **Unique per position**: Each position gets a different encoding
# MAGIC 2. **Consistent**: Same position always gets the same encoding
# MAGIC 3. **Bounded**: Values don't explode for long sequences
# MAGIC 4. **Smooth**: Nearby positions have similar encodings
# MAGIC 5. **Extrapolates**: Works for sequences longer than training
# MAGIC
# MAGIC ### The Sinusoidal Solution:
# MAGIC
# MAGIC ```
# MAGIC PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# MAGIC PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
# MAGIC ```
# MAGIC
# MAGIC Where:
# MAGIC - `pos` = position index (0, 1, 2, ...)
# MAGIC - `i` = dimension index (0, 1, 2, ..., d_model/2)
# MAGIC - Even dimensions use **sine**
# MAGIC - Odd dimensions use **cosine**
# MAGIC
# MAGIC This creates a unique "fingerprint" for each position!

# COMMAND ----------

# DBTITLE 1,Implement Positional Encoding
def positional_encoding(max_len, d_model):
    """
    Generate sinusoidal positional encodings
    
    Args:
        max_len: Maximum sequence length
        d_model: Embedding dimension
    
    Returns:
        pe: Positional encoding matrix [max_len, d_model]
    """
    pe = np.zeros((max_len, d_model))
    position = np.arange(0, max_len)[:, np.newaxis]  # [max_len, 1]
    
    # Compute the div_term for each dimension
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Apply sine to even indices
    pe[:, 0::2] = np.sin(position * div_term)
    
    # Apply cosine to odd indices
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe

# Generate positional encodings
max_len = 10
pe = positional_encoding(max_len, d_model)

print(f"Positional Encoding Matrix Shape: {pe.shape}")
print(f"  - {pe.shape[0]} positions")
print(f"  - {pe.shape[1]} dimensions\n")

print("First 3 positions (showing all dimensions):")
for pos in range(3):
    print(f"\nPosition {pos}:")
    print(f"  {pe[pos]}")

print("\n✅ Key insight: Each position has a UNIQUE encoding vector!")
print("   Position 0 ≠ Position 1 ≠ Position 2")

# COMMAND ----------

# DBTITLE 1,Visualize Positional Encodings
# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Heatmap of positional encodings
ax1 = axes[0, 0]
sns.heatmap(pe[:8, :], cmap='RdBu_r', center=0, ax=ax1, 
           cbar_kws={'label': 'Encoding Value'})
ax1.set_xlabel('Dimension', fontsize=10)
ax1.set_ylabel('Position', fontsize=10)
ax1.set_title('Positional Encoding Matrix\n(Each row = unique position fingerprint)', 
             fontsize=11, weight='bold')

# 2. Position encodings as lines
ax2 = axes[0, 1]
for pos in range(6):
    ax2.plot(pe[pos], marker='o', label=f'Position {pos}', alpha=0.7)
ax2.set_xlabel('Dimension', fontsize=10)
ax2.set_ylabel('Encoding Value', fontsize=10)
ax2.set_title('Positional Encoding Patterns\n(Each position has unique pattern)', 
             fontsize=11, weight='bold')
ax2.legend(fontsize=8, loc='right')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 3. Sine/Cosine waves for different dimensions
ax3 = axes[1, 0]
positions = np.arange(50)
for dim in [0, 2, 4, 6]:  # Even dimensions (sine)
    div_term = np.exp(dim * -(np.log(10000.0) / d_model))
    wave = np.sin(positions * div_term)
    ax3.plot(positions, wave, label=f'Dim {dim} (sin)', alpha=0.7)
ax3.set_xlabel('Position', fontsize=10)
ax3.set_ylabel('Value', fontsize=10)
ax3.set_title('Sine Waves at Different Frequencies\n(Even dimensions)', 
             fontsize=11, weight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 4. How positions differ
ax4 = axes[1, 1]
pos_pairs = [(0, 1), (0, 2), (1, 2)]
differences = [np.abs(pe[p1] - pe[p2]) for p1, p2 in pos_pairs]
for (p1, p2), diff in zip(pos_pairs, differences):
    ax4.plot(diff, marker='o', label=f'|Pos{p1} - Pos{p2}|', alpha=0.7)
ax4.set_xlabel('Dimension', fontsize=10)
ax4.set_ylabel('Absolute Difference', fontsize=10)
ax4.set_title('How Different Are Position Encodings?\n(Larger = more distinguishable)', 
             fontsize=11, weight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.suptitle('Sinusoidal Positional Encoding: Unique Fingerprint Per Position', 
            fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

print("\n🎯 Key Properties:")
print("   ✓ Bounded: Values stay in [-1, 1]")
print("   ✓ Unique: Each position has distinct pattern")
print("   ✓ Smooth: Nearby positions similar but not identical")
print("   ✓ Deterministic: Same position always gets same encoding")
print("   ✓ Scalable: Works for any sequence length!")

# COMMAND ----------

# DBTITLE 1,Apply Positional Encoding
# MAGIC %md
# MAGIC ## Adding Positional Encoding to Embeddings
# MAGIC
# MAGIC Now we'll **add** the positional encodings to our word embeddings:
# MAGIC
# MAGIC ```
# MAGIC Final embedding = Word embedding + Positional encoding
# MAGIC ```
# MAGIC
# MAGIC This way:
# MAGIC - **"Dog" at position 0** gets a different final embedding than **"Dog" at position 2**
# MAGIC - Word identity (from word embedding) + Position (from positional encoding)

# COMMAND ----------

# DBTITLE 1,Self-Attention WITH Positional Encoding
def self_attention_with_position(sentence, word_embeddings, vocab, pe):
    """
    Compute self-attention WITH positional encoding
    
    Returns:
        attention_weights: Attention pattern
        final_embeddings: Word embeddings + positional encoding
    """
    # Get word embeddings
    word_emb = np.array([word_embeddings[vocab[word]] for word in sentence])
    
    # Add positional encodings
    seq_len = len(sentence)
    final_embeddings = word_emb + pe[:seq_len]
    
    # Create Q, K, V projection matrices
    W_Q = np.random.randn(d_model, d_model) * 0.1
    W_K = np.random.randn(d_model, d_model) * 0.1
    W_V = np.random.randn(d_model, d_model) * 0.1
    
    Q = final_embeddings @ W_Q
    K = final_embeddings @ W_K
    V = final_embeddings @ W_V
    
    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_model)
    
    # Softmax
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
    
    return attention_weights, final_embeddings

# Apply to both sentences with SAME random seed
np.random.seed(100)
attention1_with_pos, emb1_pos = self_attention_with_position(sentence1, word_embedding_matrix, vocab, pe)

np.random.seed(100)  # Same seed!
attention2_with_pos, emb2_pos = self_attention_with_position(sentence2, word_embedding_matrix, vocab, pe)

print("WITH Positional Encoding:\n")
print("Sentence 1 final embeddings shape:", emb1_pos.shape)
print("Sentence 2 final embeddings shape:", emb2_pos.shape)

print(f"\nAre embeddings at position 0 different?")
print(f"  Sentence 1 pos 0 (Dog):   {emb1_pos[0][:3]}...")
print(f"  Sentence 2 pos 0 (Man):   {emb2_pos[0][:3]}...")
print(f"  Different? {not np.allclose(emb1_pos[0], emb2_pos[0])} ✅")

print(f"\nAre embeddings at position 2 different?")
print(f"  Sentence 1 pos 2 (man):   {emb1_pos[2][:3]}...")
print(f"  Sentence 2 pos 2 (dog):   {emb2_pos[2][:3]}...")
print(f"  Different? {not np.allclose(emb1_pos[2], emb2_pos[2])} ✅")

print("\n" + "="*50)
print("✅ SUCCESS: Positional encoding makes positions distinguishable!")
print("   Now 'Dog at position 0' ≠ 'Dog at position 2'")
print("="*50)

# COMMAND ----------

# DBTITLE 1,Visualize Attention WITH Position
# Visualize both attention patterns side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Sentence 1
sns.heatmap(attention1_with_pos, annot=True, fmt='.3f', cmap='YlOrRd',
           xticklabels=sentence1, yticklabels=sentence1, ax=ax1,
           vmin=0, vmax=1, cbar_kws={'label': 'Attention Weight'})
ax1.set_title('"Dog bites man"\n(WITH Positional Encoding)', fontsize=12, weight='bold')
ax1.set_xlabel('Attending to', fontsize=10)
ax1.set_ylabel('Attending from', fontsize=10)

# Sentence 2
sns.heatmap(attention2_with_pos, annot=True, fmt='.3f', cmap='YlOrRd',
           xticklabels=sentence2, yticklabels=sentence2, ax=ax2,
           vmin=0, vmax=1, cbar_kws={'label': 'Attention Weight'})
ax2.set_title('"Man bites dog"\n(WITH Positional Encoding)', fontsize=12, weight='bold')
ax2.set_xlabel('Attending to', fontsize=10)
ax2.set_ylabel('Attending from', fontsize=10)

plt.suptitle('✅ Solution: Positional Encoding Distinguishes Word Order!', 
            fontsize=14, weight='bold', color='green')
plt.tight_layout()
plt.show()

print("\n🎯 Result:")
print("   The attention patterns are now DIFFERENT!")
print("   The model can distinguish positions")
print("   ✅ 'Dog bites man' ≠ 'Man bites dog'!")

# COMMAND ----------

# DBTITLE 1,Before vs After Comparison
# MAGIC %md
# MAGIC ## Before vs After: The Impact of Positional Encoding
# MAGIC
# MAGIC Let's directly compare how much the attention patterns differ with and without positional encoding.

# COMMAND ----------

# DBTITLE 1,Quantify the Difference
# Compute differences between sentence 1 and sentence 2
diff_without_pos = np.abs(attention1_no_pos - attention2_no_pos)
diff_with_pos = np.abs(attention1_with_pos - attention2_with_pos)

avg_diff_without = diff_without_pos.mean()
avg_diff_with = diff_with_pos.mean()

print("Average Absolute Difference in Attention Patterns:\n")
print(f"  WITHOUT positional encoding: {avg_diff_without:.4f}")
print(f"  WITH positional encoding:    {avg_diff_with:.4f}")
print(f"\n  Improvement: {(avg_diff_with / avg_diff_without):.1f}× more distinguishable!")

# Visualize the differences
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Without positional encoding
sns.heatmap(diff_without_pos, annot=True, fmt='.3f', cmap='Greens',
           xticklabels=['Dog', 'bites', 'man'], 
           yticklabels=['Dog', 'bites', 'man'], 
           ax=ax1, vmin=0, vmax=0.3,
           cbar_kws={'label': 'Absolute Difference'})
ax1.set_title(f'Difference: WITHOUT Positional Encoding\n(Avg diff: {avg_diff_without:.4f})', 
             fontsize=11, weight='bold')
ax1.set_xlabel('Position', fontsize=10)
ax1.set_ylabel('Position', fontsize=10)

# With positional encoding
sns.heatmap(diff_with_pos, annot=True, fmt='.3f', cmap='Greens',
           xticklabels=['Pos 0', 'Pos 1', 'Pos 2'], 
           yticklabels=['Pos 0', 'Pos 1', 'Pos 2'], 
           ax=ax2, vmin=0, vmax=0.3,
           cbar_kws={'label': 'Absolute Difference'})
ax2.set_title(f'Difference: WITH Positional Encoding\n(Avg diff: {avg_diff_with:.4f})', 
             fontsize=11, weight='bold')
ax2.set_xlabel('Position', fontsize=10)
ax2.set_ylabel('Position', fontsize=10)

plt.suptitle('How Different Are "Dog bites man" vs "Man bites dog"?\n(Larger values = More distinguishable)', 
            fontsize=13, weight='bold')
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print("\n❌ WITHOUT positional encoding:")
print("   → Very small differences (bag of words)")
print("   → Cannot distinguish word order")
print("   → 'Dog bites man' ≈ 'Man bites dog'")

print("\n✅ WITH positional encoding:")
print("   → Clear differences in attention patterns")
print("   → Can distinguish word order")
print("   → 'Dog bites man' ≠ 'Man bites dog'")
print("\n🎯 Positional encoding rescues word order!")

# COMMAND ----------

# DBTITLE 1,Visual Summary
# MAGIC %md
# MAGIC ## Visual Summary: The Complete Picture
# MAGIC
# MAGIC Let's visualize the complete transformation from word embeddings to final representations.

# COMMAND ----------

# DBTITLE 1,Complete Transformation Diagram
# Create visual diagram showing the transformation
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: "Dog bites man"
sentence = sentence1
word_emb = np.array([word_embedding_matrix[vocab[word]] for word in sentence])
pos_enc = pe[:len(sentence)]
final_emb = word_emb + pos_enc

# Word embeddings
ax1 = axes[0, 0]
im1 = ax1.imshow(word_emb.T, cmap='coolwarm', aspect='auto')
ax1.set_xticks(range(len(sentence)))
ax1.set_xticklabels(sentence, fontsize=10)
ax1.set_ylabel('Embedding Dimension', fontsize=9)
ax1.set_title('1. Word Embeddings\n(Same word = same vector)', fontsize=10, weight='bold')
plt.colorbar(im1, ax=ax1)

# Positional encodings
ax2 = axes[0, 1]
im2 = ax2.imshow(pos_enc.T, cmap='RdBu_r', aspect='auto')
ax2.set_xticks(range(len(sentence)))
ax2.set_xticklabels(['Pos 0', 'Pos 1', 'Pos 2'], fontsize=10)
ax2.set_ylabel('Dimension', fontsize=9)
ax2.set_title('2. Positional Encodings\n(Unique per position)', fontsize=10, weight='bold')
plt.colorbar(im2, ax=ax2)

# Final embeddings
ax3 = axes[0, 2]
im3 = ax3.imshow(final_emb.T, cmap='viridis', aspect='auto')
ax3.set_xticks(range(len(sentence)))
ax3.set_xticklabels([f'{w}\n@{i}' for i, w in enumerate(sentence)], fontsize=9)
ax3.set_ylabel('Dimension', fontsize=9)
ax3.set_title('3. Final = Word + Position\n(Position-aware!)', fontsize=10, weight='bold')
plt.colorbar(im3, ax=ax3)

# Add equation between plots
fig.text(0.28, 0.72, '+', fontsize=30, weight='bold', ha='center')
fig.text(0.57, 0.72, '=', fontsize=30, weight='bold', ha='center')

# Row 2: "Man bites dog"
sentence = sentence2
word_emb = np.array([word_embedding_matrix[vocab[word]] for word in sentence])
pos_enc = pe[:len(sentence)]
final_emb = word_emb + pos_enc

# Word embeddings
ax4 = axes[1, 0]
im4 = ax4.imshow(word_emb.T, cmap='coolwarm', aspect='auto')
ax4.set_xticks(range(len(sentence)))
ax4.set_xticklabels(sentence, fontsize=10)
ax4.set_ylabel('Embedding Dimension', fontsize=9)
ax4.set_title('1. Word Embeddings\n(Same words, different order)', fontsize=10, weight='bold')
plt.colorbar(im4, ax=ax4)

# Positional encodings (SAME as above!)
ax5 = axes[1, 1]
im5 = ax5.imshow(pos_enc.T, cmap='RdBu_r', aspect='auto')
ax5.set_xticks(range(len(sentence)))
ax5.set_xticklabels(['Pos 0', 'Pos 1', 'Pos 2'], fontsize=10)
ax5.set_ylabel('Dimension', fontsize=9)
ax5.set_title('2. Positional Encodings\n(SAME — position determined)', fontsize=10, weight='bold')
plt.colorbar(im5, ax=ax5)

# Final embeddings (DIFFERENT from row 1!)
ax6 = axes[1, 2]
im6 = ax6.imshow(final_emb.T, cmap='viridis', aspect='auto')
ax6.set_xticks(range(len(sentence)))
ax6.set_xticklabels([f'{w}\n@{i}' for i, w in enumerate(sentence)], fontsize=9)
ax6.set_ylabel('Dimension', fontsize=9)
ax6.set_title('3. Final = Word + Position\n(DIFFERENT from above!)', fontsize=10, weight='bold')
plt.colorbar(im6, ax=ax6)

# Add equations
fig.text(0.28, 0.28, '+', fontsize=30, weight='bold', ha='center')
fig.text(0.57, 0.28, '=', fontsize=30, weight='bold', ha='center')

# Add row labels
fig.text(0.02, 0.72, '"Dog bites man"', fontsize=12, weight='bold', rotation=90, va='center')
fig.text(0.02, 0.28, '"Man bites dog"', fontsize=12, weight='bold', rotation=90, va='center')

plt.suptitle('How Positional Encoding Makes Word Order Distinguishable', 
            fontsize=15, weight='bold', y=0.98)
plt.tight_layout(rect=[0.03, 0, 1, 0.96])
plt.show()

print("\n🎯 The Magic:")
print("   • Same words at different positions → Different final embeddings")
print("   • Position encoding acts as a 'fingerprint' for each location")
print("   • Result: Transformer can now understand word order!")

# COMMAND ----------

# DBTITLE 1,Key Insights
# MAGIC %md
# MAGIC ## Key Insights
# MAGIC
# MAGIC ### The Problem:
# MAGIC
# MAGIC **Transformers process all words in parallel** → Great for speed, but they lose word order!
# MAGIC
# MAGIC ```
# MAGIC "Dog bites man" = {Dog, bites, man}  ← Bag of words
# MAGIC "Man bites dog" = {Dog, bites, man}  ← Same bag!
# MAGIC ```
# MAGIC
# MAGIC Without positional information, these produce **identical attention patterns**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### The Solution:
# MAGIC
# MAGIC **Positional Encoding** adds a unique mathematical "fingerprint" to each position:
# MAGIC
# MAGIC ```
# MAGIC PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))  ← Even dimensions
# MAGIC PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))  ← Odd dimensions
# MAGIC ```
# MAGIC
# MAGIC #### Why Sinusoidal Functions?
# MAGIC
# MAGIC 1. **Bounded**: Values stay in [-1, 1] for any position
# MAGIC 2. **Unique**: Each position has a distinct pattern across dimensions
# MAGIC 3. **Smooth**: Nearby positions have similar (but not identical) encodings
# MAGIC 4. **Deterministic**: Same position always gets the same encoding
# MAGIC 5. **Extrapolates**: Can handle sequences longer than seen during training
# MAGIC 6. **Relative Position**: The model can learn to attend by relative positions
# MAGIC
# MAGIC #### Why 10,000?
# MAGIC
# MAGIC It's a "Goldilocks" base:
# MAGIC - Too small (e.g., 10): Patterns repeat too quickly
# MAGIC - Too large (e.g., 1,000,000): Patterns change too slowly, positions indistinguishable
# MAGIC - 10,000: Sweet spot for sequences up to ~10K tokens
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### The Result:
# MAGIC
# MAGIC ```
# MAGIC Word Embedding + Positional Encoding = Position-Aware Embedding
# MAGIC
# MAGIC "Dog" at position 0 ≠ "Dog" at position 2
# MAGIC "Man" at position 0 ≠ "Man" at position 2
# MAGIC ```
# MAGIC
# MAGIC Now the transformer can distinguish:
# MAGIC - "Dog bites man" (dog is subject)
# MAGIC - "Man bites dog" (man is subject)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Real-World Impact:
# MAGIC
# MAGIC Positional encoding is **critical** for:
# MAGIC - **Word order**: "not good" vs "good not"
# MAGIC - **Grammar**: Subject-verb-object relationships
# MAGIC - **Syntax**: Parse trees depend on position
# MAGIC - **Meaning**: "bank" (position matters for context)
# MAGIC
# MAGIC Without it, transformers would be **useless** for language understanding!

# COMMAND ----------

# DBTITLE 1,Summary
# MAGIC %md
# MAGIC ## Summary: Positional Encoding
# MAGIC
# MAGIC ### The Problem: Order-Blindness
# MAGIC
# MAGIC Transformers process words **simultaneously in parallel**:
# MAGIC - ✅ Fast (no sequential bottleneck)
# MAGIC - ❌ No inherent sense of word order
# MAGIC
# MAGIC **Example**:
# MAGIC ```
# MAGIC "Dog bites man" → {Dog, bites, man}  ← Bag of words
# MAGIC "Man bites dog" → {Dog, bites, man}  ← Identical!
# MAGIC ```
# MAGIC
# MAGIC ### The Solution: Positional Encoding
# MAGIC
# MAGIC Add a **unique mathematical fingerprint** to each position:
# MAGIC
# MAGIC ```python
# MAGIC PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))  # Even dims
# MAGIC PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))  # Odd dims
# MAGIC
# MAGIC Final_Embedding = Word_Embedding + Positional_Encoding
# MAGIC ```
# MAGIC
# MAGIC ### Key Properties:
# MAGIC
# MAGIC | Property | Benefit |
# MAGIC |----------|--------|
# MAGIC | **Unique** | Each position gets distinct encoding |
# MAGIC | **Bounded** | Values in [-1, 1], no explosion |
# MAGIC | **Smooth** | Nearby positions are similar |
# MAGIC | **Deterministic** | Consistent across training/inference |
# MAGIC | **Scalable** | Works for any sequence length |
# MAGIC
# MAGIC ### The Result:
# MAGIC
# MAGIC ```
# MAGIC WITHOUT Positional Encoding:
# MAGIC "Dog bites man" ≈ "Man bites dog"  ❌ Can't distinguish
# MAGIC
# MAGIC WITH Positional Encoding:
# MAGIC "Dog bites man" ≠ "Man bites dog"  ✅ Different meanings!
# MAGIC ```
# MAGIC
# MAGIC ### Why It Matters:
# MAGIC
# MAGIC Positional encoding enables transformers to:
# MAGIC - Understand **word order** ("not good" ≠ "good not")
# MAGIC - Capture **grammar** (subject-verb-object)
# MAGIC - Learn **syntax** (parse trees)
# MAGIC - Preserve **meaning** ("I didn't say he stole the money" — emphasis matters)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### The Big Picture:
# MAGIC
# MAGIC **Transformer Architecture** = Self-Attention + Positional Encoding
# MAGIC
# MAGIC - **Self-Attention**: Connects all words (parallel processing)
# MAGIC - **Positional Encoding**: Preserves word order
# MAGIC
# MAGIC Together, they solve the long-term dependency problem **without** losing position information!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Further Reading:
# MAGIC
# MAGIC To understand the complete picture:
# MAGIC 1. **Self-Attention Mechanism** - How words attend to each other
# MAGIC 2. **Multi-Head Attention** - Multiple attention perspectives
# MAGIC 3. **RNN vs Self-Attention** - Why transformers replaced RNNs
# MAGIC 4. **Complete Transformer** - Full encoder/decoder architecture
# MAGIC
# MAGIC **Original Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)