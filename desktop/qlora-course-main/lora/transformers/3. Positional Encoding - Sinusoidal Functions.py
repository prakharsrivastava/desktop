# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # Positional Encoding - Teaching Order to Transformers
# MAGIC
# MAGIC ## The Order-Blindness Problem
# MAGIC
# MAGIC **Sentence 1**: "Dog bites man"
# MAGIC **Sentence 2**: "Man bites dog"
# MAGIC
# MAGIC These have **completely different meanings**, but transformers can't tell the difference!
# MAGIC
# MAGIC ### Why?
# MAGIC
# MAGIC Transformers process all words **simultaneously in parallel**. They compute attention between every word pair:
# MAGIC
# MAGIC ```
# MAGIC Dog-bites, Dog-man, bites-Dog, bites-man, man-Dog, man-bites
# MAGIC ```
# MAGIC
# MAGIC But these calculations have **zero information about position**!
# MAGIC
# MAGIC It's like throwing words into a bag and shaking - the transformer sees the same collection of words regardless of order.
# MAGIC
# MAGIC ### The Solution: Positional Encoding
# MAGIC
# MAGIC We add a unique **mathematical fingerprint** to each word based on its position:
# MAGIC - Position 0 gets one pattern
# MAGIC - Position 1 gets a different pattern
# MAGIC - Position 2 gets another pattern
# MAGIC
# MAGIC These fingerprints are created using **sine and cosine waves** at different frequencies!

# COMMAND ----------

# DBTITLE 1,Setup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

print("✓ Setup complete")

# COMMAND ----------

# DBTITLE 1,Approach 1 - Raw Indices
# MAGIC %md
# MAGIC ## Approach 1: Raw Position Indices (FAILS)
# MAGIC
# MAGIC **Idea**: Just use position numbers directly
# MAGIC - Word 1 → add +1
# MAGIC - Word 2 → add +2
# MAGIC - Word 100 → add +100
# MAGIC
# MAGIC ### Why This Fails:
# MAGIC
# MAGIC Word embeddings are small decimal values like 0.3, -0.1, 0.5
# MAGIC
# MAGIC If you add +100 to position 100, it **crushes the word meaning**!
# MAGIC
# MAGIC The embedding becomes dominated by the giant position number.

# COMMAND ----------

# Simple sentence
tokens = ["Dog", "bites", "man"]
num_tokens = len(tokens)
d_model = 6  # Small dimension for visualization

# Create word embeddings (small decimal values)
embeddings = np.random.randn(num_tokens, d_model) * 0.3

print("Word Embeddings (small values):")
for i, token in enumerate(tokens):
    print(f"{token:6s}: {embeddings[i]}")

# COMMAND ----------

# Try adding raw position indices
raw_positions = np.arange(num_tokens)[:, None]  # [0, 1, 2] reshaped
print(f"\nRaw position indices: {raw_positions.flatten()}")

# COMMAND ----------

# DBTITLE 1,Raw Indices Demo
# This would fail for long sequences
print("\n❌ Problem: For a 100-word sentence, position 100 would overwhelm the embedding!")
print("   Word embedding values: ~0.3")
print("   Position 100: 100")
print("   Result: Word meaning gets crushed by position number")

# COMMAND ----------

# DBTITLE 1,Approach 2 - Normalized
# MAGIC %md
# MAGIC ## Approach 2: Normalized Positions (FAILS TOO)
# MAGIC
# MAGIC **Idea**: Normalize positions to range [0, 1]
# MAGIC - First word: 0.0
# MAGIC - Last word: 1.0
# MAGIC - Middle word: 0.5
# MAGIC
# MAGIC ### Why This Fails:
# MAGIC
# MAGIC Position values **change based on sentence length**!
# MAGIC
# MAGIC 3-word sentence:
# MAGIC - Position 2 → 1.0 (last word)
# MAGIC
# MAGIC 7-word sentence:
# MAGIC - Position 2 → 0.33 (early word)
# MAGIC
# MAGIC **Same position, different encoding!** Model can't generalize across sentence lengths.

# COMMAND ----------

def normalized_positions(seq_length):
    """Normalize positions to [0, 1]"""
    if seq_length == 1:
        return np.array([0.0])
    return np.arange(seq_length) / (seq_length - 1)


# COMMAND ----------


# Same word at position 2 in different sentence lengths
print("Position 2 in different sentence lengths:")
print()

# COMMAND ----------

# DBTITLE 1,Normalized Positions Demo


for length in [3, 5, 7, 10]:
    positions = normalized_positions(length)
    pos_2_value = positions[2] if length > 2 else None
    print(f"{length}-word sentence: Position 2 = {pos_2_value:.3f}" if pos_2_value else f"{length}-word sentence: Too short")

print("\n❌ Problem: Same position gets different encodings depending on sentence length!")
print("   Model cannot learn consistent positional patterns.")

# COMMAND ----------

# DBTITLE 1,Approach 3 - Learned
# MAGIC %md
# MAGIC ## Approach 3: Learned Positional Embeddings
# MAGIC
# MAGIC **Idea**: Create a lookup table, let model learn position patterns during training
# MAGIC
# MAGIC - Used by BERT and GPT-2
# MAGIC - Works great!
# MAGIC
# MAGIC ### The Catch:
# MAGIC
# MAGIC Fixed maximum sequence length. If trained on sequences up to 512 tokens:
# MAGIC - Position 513? **Crash!** No learned embedding for that position.
# MAGIC - Can't handle sequences longer than training maximum.
# MAGIC
# MAGIC ### We Need Better:
# MAGIC
# MAGIC We want:
# MAGIC 1. **Bounded** - values don't explode
# MAGIC 2. **Unique** - every position gets different encoding
# MAGIC 3. **Length-agnostic** - works for any sequence length
# MAGIC 4. **Smooth** - nearby positions have similar encodings
# MAGIC
# MAGIC **Answer**: Sine and Cosine functions!

# COMMAND ----------

# DBTITLE 1,Sinusoidal Approach
# MAGIC %md
# MAGIC ## The Solution: Sinusoidal Positional Encoding
# MAGIC
# MAGIC ### Key Idea:
# MAGIC
# MAGIC Use **sine and cosine waves** at different frequencies to create unique fingerprints!
# MAGIC
# MAGIC - Fast waves (high frequency) - change quickly, capture local position
# MAGIC - Slow waves (low frequency) - change slowly, capture global position
# MAGIC
# MAGIC ### Properties:
# MAGIC
# MAGIC ✅ **Bounded**: sine and cosine always between -1 and +1
# MAGIC ✅ **Unique**: combination of multiple frequencies = unique pattern for each position
# MAGIC ✅ **Length-agnostic**: continuous functions work for any position
# MAGIC ✅ **Smooth**: nearby positions have similar values
# MAGIC
# MAGIC ### The Formula:
# MAGIC
# MAGIC ```
# MAGIC PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# MAGIC PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
# MAGIC ```
# MAGIC
# MAGIC Where:
# MAGIC - `pos` = position in sequence (0, 1, 2, ...)
# MAGIC - `i` = dimension index (0, 1, 2, ...)
# MAGIC - `d_model` = embedding dimension

# COMMAND ----------

np.arange(3)[:, np.newaxis]

# COMMAND ----------

d_model

# COMMAND ----------

np.arange(0, d_model, 2)

# COMMAND ----------

np.log(10000.0)

# COMMAND ----------

np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

# COMMAND ----------

np.exp(np.arange(0,6, 2) * -(np.log(10000.0) / 6))

# COMMAND ----------

div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
position = np.arange(3)[:, np.newaxis]
np.sin(position * div_term)

# COMMAND ----------

# DBTITLE 1,Positional Encoding Implementation
def get_positional_encoding(max_seq_len, d_model):
    """
    Generate sinusoidal positional encodings
    
    Args:
        max_seq_len: Maximum sequence length
        d_model: Embedding dimension (must be even)
    
    Returns:
        pos_encoding: Positional encoding matrix (max_seq_len, d_model)
    """
    # Initialize positional encoding matrix
    pos_encoding = np.zeros((max_seq_len, d_model))
    
    # Create position indices [0, 1, 2, ..., max_seq_len-1]
    position = np.arange(max_seq_len)[:, np.newaxis]  # Shape: (max_seq_len, 1)
    
    # Create dimension indices [0, 1, 2, ..., d_model//2-1]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Apply sine to even dimensions
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    
    # Apply cosine to odd dimensions
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    return pos_encoding

print("✓ Positional encoding function defined")

# COMMAND ----------

# DBTITLE 1,Generate Encodings
# MAGIC %md
# MAGIC ## Generate Positional Encodings
# MAGIC
# MAGIC Let's create positional encodings for our example sentence: **"The dog chased the dog"**

# COMMAND ----------

get_positional_encoding(5, 6)


# COMMAND ----------

# DBTITLE 1,Create Positional Encodings
# Sentence: "The dog chased the dog"
tokens = ["The", "dog", "chased", "the", "dog"]
num_tokens = len(tokens)
d_model = 6  # Small dimension for clarity

# Generate positional encodings
pos_encoding = get_positional_encoding(num_tokens, d_model)

print(f"Positional Encoding shape: {pos_encoding.shape}")
print(f"\nPositional Encodings:")
print()

for pos in range(num_tokens):
    print(f"Position {pos} ({tokens[pos]:6s}): {pos_encoding[pos]}")

print("\n🔍 Notice: Each position has a UNIQUE pattern of values!")

# COMMAND ----------

# DBTITLE 1,Visualize Encodings
# MAGIC %md
# MAGIC ## Visualize the Positional Encoding Matrix
# MAGIC
# MAGIC Each row = one position, each column = one dimension

# COMMAND ----------

# DBTITLE 1,Heatmap Visualization
# Visualize positional encoding as heatmap
plt.figure(figsize=(10, 6))

sns.heatmap(pos_encoding, 
            annot=True, 
            fmt=".2f", 
            cmap="coolwarm",
            center=0,
            xticklabels=[f"Dim{i}" for i in range(d_model)],
            yticklabels=[f"Pos{i} ({tokens[i]})" for i in range(num_tokens)],
            cbar_kws={'label': 'Value'})

plt.title("Positional Encoding Matrix\n\nEach row is a unique fingerprint for a position")
plt.xlabel("Dimensions")
plt.ylabel("Positions")
plt.tight_layout()
plt.show()

print("\nColor coding:")
print("  🔵 Blue = Negative values")
print("  🔴 Red = Positive values")
print("  ⚪ White = Near zero")

# COMMAND ----------

# DBTITLE 1,Understanding Frequencies
# MAGIC %md
# MAGIC ## Understanding Different Frequencies
# MAGIC
# MAGIC The magic is in using **multiple frequencies simultaneously**:
# MAGIC
# MAGIC - **High frequency dimensions** (left): Change rapidly position-to-position
# MAGIC - **Low frequency dimensions** (right): Change slowly position-to-position
# MAGIC
# MAGIC Think of it like:
# MAGIC - High frequency = second hand on a clock (captures fine detail)
# MAGIC - Low frequency = hour hand on a clock (captures broad position)

# COMMAND ----------

# DBTITLE 1,Frequency Waves
# Generate longer sequence to see wave patterns
max_len = 100
d_model_viz = 8
pos_enc_long = get_positional_encoding(max_len, d_model_viz)

# Plot different frequency components
fig, axes = plt.subplots(4, 1, figsize=(14, 10))

positions = np.arange(max_len)

# Dimension 0 (highest frequency)
axes[0].plot(positions, pos_enc_long[:, 0], 'b-', linewidth=2)
axes[0].set_title("Dimension 0 (Highest Frequency) - Fast oscillation")
axes[0].set_ylabel("Value")
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Dimension 2 (medium-high frequency)
axes[1].plot(positions, pos_enc_long[:, 2], 'g-', linewidth=2)
axes[1].set_title("Dimension 2 (Medium Frequency) - Moderate oscillation")
axes[1].set_ylabel("Value")
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Dimension 4 (medium-low frequency)
axes[2].plot(positions, pos_enc_long[:, 4], 'orange', linewidth=2)
axes[2].set_title("Dimension 4 (Lower Frequency) - Slow oscillation")
axes[2].set_ylabel("Value")
axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Dimension 6 (lowest frequency)
axes[3].plot(positions, pos_enc_long[:, 6], 'r-', linewidth=2)
axes[3].set_title("Dimension 6 (Lowest Frequency) - Very slow oscillation")
axes[3].set_ylabel("Value")
axes[3].set_xlabel("Position")
axes[3].grid(True, alpha=0.3)
axes[3].axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.suptitle("Different Frequency Components of Positional Encoding", fontsize=14, y=0.995)
plt.tight_layout()
plt.show()

print("\n💡 Key Insight:")
print("  - Fast waves capture fine-grained local differences")
print("  - Slow waves capture broad global position in sequence")
print("  - Together they create unique fingerprint for EVERY position!")

# COMMAND ----------

# DBTITLE 1,Full Heatmap
# MAGIC %md
# MAGIC ## Full Positional Encoding Heatmap
# MAGIC
# MAGIC Let's visualize a larger positional encoding matrix (50 positions × 64 dimensions):

# COMMAND ----------

# DBTITLE 1,Large Heatmap
# Generate large positional encoding
max_len_large = 50
d_model_large = 64
pos_enc_large = get_positional_encoding(max_len_large, d_model_large)

# Create heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(pos_enc_large, 
            cmap="RdBu_r",
            center=0,
            cbar_kws={'label': 'Value'})

plt.title("Positional Encoding Heatmap (50 positions × 64 dimensions)\n\nEach row is a unique position fingerprint", 
          fontsize=14)
plt.xlabel("Dimensions\n\n← High Frequency (fast changes) | Low Frequency (slow changes) →", fontsize=11)
plt.ylabel("Positions", fontsize=11)
plt.tight_layout()
plt.show()

print("\n🔍 Observations:")
print("  - Left side: Rapid vertical stripes (high frequency)")
print("  - Right side: Slow gradual changes (low frequency)")
print("  - Every row (position) has unique pattern")
print("  - Nearby positions have similar patterns (smooth)")

# COMMAND ----------

# DBTITLE 1,Adding to Embeddings
# MAGIC %md
# MAGIC ## Adding Positional Encoding to Word Embeddings
# MAGIC
# MAGIC Now we combine:
# MAGIC 1. **Word embeddings** - capture word meaning
# MAGIC 2. **Positional encodings** - capture position
# MAGIC
# MAGIC **Formula**: `Final = Word_Embedding + Positional_Encoding`
# MAGIC
# MAGIC Element-wise addition merges meaning and position!

# COMMAND ----------

# DBTITLE 1,Combine Embeddings and Position
# Sentence: "The dog chased the dog"
tokens = ["The", "dog", "chased", "the", "dog"]
num_tokens = len(tokens)
d_model = 6

# Create word embeddings
word_embeddings = np.random.randn(num_tokens, d_model) * 0.5

print("Step 1: Word Embeddings (meaning only)")
for i, token in enumerate(tokens):
    print(f"{token:8s}: {word_embeddings[i]}")

# Create positional encodings
pos_enc = get_positional_encoding(num_tokens, d_model)

print("\nStep 2: Positional Encodings (position only)")
for i in range(num_tokens):
    print(f"Pos {i}:     {pos_enc[i]}")

# Add them together
final_embeddings = word_embeddings + pos_enc

print("\nStep 3: Final Embeddings (meaning + position)")
for i, token in enumerate(tokens):
    print(f"{token:8s}: {final_embeddings[i]}")

print("\n✨ Now each word knows BOTH what it means AND where it is!")

# COMMAND ----------

# DBTITLE 1,Visualize Addition
# MAGIC %md
# MAGIC ## Visualize the Addition Process

# COMMAND ----------

# DBTITLE 1,Addition Visualization
# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Word embeddings
sns.heatmap(word_embeddings, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=[f"D{i}" for i in range(d_model)],
            yticklabels=tokens, ax=axes[0], cbar_kws={'label': 'Value'},
            center=0)
axes[0].set_title("Word Embeddings\n(Semantic Meaning)")
axes[0].set_ylabel("Tokens")

# Positional encodings
sns.heatmap(pos_enc, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=[f"D{i}" for i in range(d_model)],
            yticklabels=[f"Pos {i}" for i in range(num_tokens)], 
            ax=axes[1], cbar_kws={'label': 'Value'},
            center=0)
axes[1].set_title("+ Positional Encoding\n(Position Information)")
axes[1].set_ylabel("")

# Final combined
sns.heatmap(final_embeddings, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=[f"D{i}" for i in range(d_model)],
            yticklabels=tokens, ax=axes[2], cbar_kws={'label': 'Value'},
            center=0)
axes[2].set_title("= Final Input\n(Meaning + Position)")
axes[2].set_ylabel("")

plt.suptitle("Adding Positional Encoding to Word Embeddings", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("\n✅ Element-wise addition merges semantic and positional information!")

# COMMAND ----------

# DBTITLE 1,Repeated Words
# MAGIC %md
# MAGIC ## Handling Repeated Words
# MAGIC
# MAGIC Notice our sentence has **"dog"** appear twice at positions 1 and 4.
# MAGIC
# MAGIC **Without positional encoding**: Both "dog" tokens would have IDENTICAL embeddings!
# MAGIC
# MAGIC **With positional encoding**: Each "dog" gets a unique representation based on its position.

# COMMAND ----------

# DBTITLE 1,Repeated Word Demo
# Focus on the two "dog" tokens
dog_positions = [1, 4]

print("The word 'dog' appears twice in different positions:")
print()

print("Original word embedding (same for both):")
print(f"  dog: {word_embeddings[1]}")
print()

print("Positional encodings (different for each position):")
for pos in dog_positions:
    print(f"  Position {pos}: {pos_enc[pos]}")
print()

print("Final embeddings (unique for each occurrence):")
for pos in dog_positions:
    print(f"  dog at pos {pos}: {final_embeddings[pos]}")
print()

print("✓ Same word, different positions = different final representations!")
print("  This lets the model distinguish between multiple occurrences.")

# Visualize difference
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Original embeddings (identical)
axes[0].bar(range(d_model), word_embeddings[1], color='steelblue', alpha=0.7)
axes[0].bar(range(d_model), word_embeddings[4], color='orange', alpha=0.5)
axes[0].set_title("Word Embeddings\n(Identical for both 'dog')")
axes[0].set_xlabel("Dimension")
axes[0].set_ylabel("Value")
axes[0].legend(['dog (pos 1)', 'dog (pos 4)'])
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Positional encodings (different)
axes[1].bar(range(d_model), pos_enc[1], color='steelblue', alpha=0.7)
axes[1].bar(range(d_model), pos_enc[4], color='orange', alpha=0.5)
axes[1].set_title("Positional Encodings\n(Different patterns)")
axes[1].set_xlabel("Dimension")
axes[1].legend(['Position 1', 'Position 4'])
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Final embeddings (unique)
axes[2].bar(range(d_model), final_embeddings[1], color='steelblue', alpha=0.7)
axes[2].bar(range(d_model), final_embeddings[4], color='orange', alpha=0.5)
axes[2].set_title("Final Embeddings\n(Unique for each occurrence)")
axes[2].set_xlabel("Dimension")
axes[2].legend(['dog at pos 1', 'dog at pos 4'])
axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.suptitle("How Positional Encoding Distinguishes Repeated Words", fontsize=14)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Why 10000
# MAGIC %md
# MAGIC ## Why 10,000 in the Formula?
# MAGIC
# MAGIC The formula uses **10,000** as the base:
# MAGIC
# MAGIC ```
# MAGIC PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
# MAGIC ```
# MAGIC
# MAGIC Why this specific number?
# MAGIC
# MAGIC ### The exponent creates a geometric progression:
# MAGIC
# MAGIC - Dimension 0: divisor = 10000^0 = **1** (highest frequency)
# MAGIC - Dimension d_model/2: divisor = 10000^1 = **10,000** (lowest frequency)
# MAGIC
# MAGIC This spreads frequencies across dimensions!
# MAGIC
# MAGIC ### Comparison:
# MAGIC
# MAGIC - **Too small (e.g., 10)**: Fast waves everywhere, positions collide
# MAGIC - **Just right (10,000)**: Perfect balance, handles ~10,000 tokens
# MAGIC - **Too large (e.g., 1,000,000)**: Waves too slow, can't distinguish nearby positions
# MAGIC
# MAGIC **10,000 is the Goldilocks value** - not too fast, not too slow!

# COMMAND ----------

# DBTITLE 1,Base Comparison
def get_positional_encoding_with_base(max_seq_len, d_model, base=10000):
    """Generate positional encoding with custom base"""
    pos_encoding = np.zeros((max_seq_len, d_model))
    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(base) / d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    return pos_encoding

# Compare different bases for the slowest dimension
max_len = 100
d_model_comp = 8

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
bases = [10, 10000, 1000000]
titles = ["Base = 10 (Too Small)", "Base = 10,000 (Just Right)", "Base = 1,000,000 (Too Large)"]

for idx, (base, title) in enumerate(zip(bases, titles)):
    pos_enc = get_positional_encoding_with_base(max_len, d_model_comp, base)
    
    # Plot the last dimension (slowest frequency)
    axes[idx].plot(np.arange(max_len), pos_enc[:, -1], linewidth=2)
    axes[idx].set_title(f"{title}\nLast dimension (slowest frequency wave)")
    axes[idx].set_ylabel("Value")
    axes[idx].grid(True, alpha=0.3)
    axes[idx].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[idx].set_ylim(-1.2, 1.2)
    
    if idx == 2:
        axes[idx].set_xlabel("Position")

plt.suptitle("Effect of Base Value on Slowest Frequency Component", fontsize=14)
plt.tight_layout()
plt.show()

print("\n🔍 Analysis:")
print("  - Base = 10: Wave cycles too fast, positions will collide")
print("  - Base = 10,000: Smooth wave, perfect for ~10K tokens")
print("  - Base = 1,000,000: Wave barely moves, can't distinguish positions")

# COMMAND ----------

# DBTITLE 1,Summary
# MAGIC %md
# MAGIC ## Summary: Positional Encoding
# MAGIC
# MAGIC ### The Problem:
# MAGIC - Transformers process words in parallel
# MAGIC - No inherent sense of order
# MAGIC - "Dog bites man" = "Man bites dog" without position info
# MAGIC
# MAGIC ### Failed Approaches:
# MAGIC 1. **Raw indices**: Values explode for long sequences
# MAGIC 2. **Normalized positions**: Same position gets different values in different sentence lengths
# MAGIC 3. **Learned embeddings**: Can't handle sequences longer than training maximum
# MAGIC
# MAGIC ### The Solution: Sinusoidal Encoding
# MAGIC
# MAGIC **Formula**:
# MAGIC ```
# MAGIC PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))  [even dimensions]
# MAGIC PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))  [odd dimensions]
# MAGIC ```
# MAGIC
# MAGIC ### Key Properties:
# MAGIC
# MAGIC ✅ **Bounded**: Values always between -1 and +1
# MAGIC ✅ **Unique**: Every position gets unique fingerprint
# MAGIC ✅ **Length-agnostic**: Works for any sequence length
# MAGIC ✅ **Smooth**: Nearby positions have similar encodings
# MAGIC
# MAGIC ### How It Works:
# MAGIC
# MAGIC 1. **Multiple frequencies**: 
# MAGIC    - Fast waves (high freq) capture local position
# MAGIC    - Slow waves (low freq) capture global position
# MAGIC
# MAGIC 2. **Alternating sine/cosine**: 
# MAGIC    - Even dimensions use sine
# MAGIC    - Odd dimensions use cosine
# MAGIC
# MAGIC 3. **Geometric progression**: 
# MAGIC    - Base 10,000 spreads frequencies perfectly
# MAGIC    - Handles sequences up to ~10,000 tokens
# MAGIC
# MAGIC 4. **Element-wise addition**: 
# MAGIC    - Add to word embeddings before attention
# MAGIC    - Final = Word_Embedding + Positional_Encoding
# MAGIC
# MAGIC ### Result:
# MAGIC
# MAGIC Every token now carries **both meaning and position**!
# MAGIC
# MAGIC The transformer can distinguish:
# MAGIC - "Dog bites man" vs "Man bites dog"
# MAGIC - First "dog" vs second "dog" in "The dog chased the dog"
# MAGIC
# MAGIC ### What's Next?
# MAGIC
# MAGIC Now that we have position-aware embeddings, we can build the **complete encoder** with multi-head attention, feed-forward networks, and layer normalization!