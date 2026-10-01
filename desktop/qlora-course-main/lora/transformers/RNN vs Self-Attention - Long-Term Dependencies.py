# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # RNN vs Self-Attention: Solving the Long-Term Dependency Problem
# MAGIC
# MAGIC ## The Challenge
# MAGIC
# MAGIC Consider this sentence:
# MAGIC
# MAGIC **"The animal didn't cross the street because it was too tired."**
# MAGIC
# MAGIC ### The Question: What does "it" refer to?
# MAGIC
# MAGIC Humans immediately know **"it" = "animal"**. But how do neural networks figure this out?
# MAGIC
# MAGIC - The word **"animal"** appears at position 1
# MAGIC - The word **"it"** appears at position 8
# MAGIC - **7 words** separate them!
# MAGIC
# MAGIC This is the **Long-Term Dependency Problem**: Can the model remember information from many steps ago?
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Two Approaches:
# MAGIC
# MAGIC ### 1. **RNN (Recurrent Neural Networks)** - Sequential Processing
# MAGIC - Reads words one-by-one, left to right
# MAGIC - Maintains a "hidden state" that carries information forward
# MAGIC - **Problem**: Information from "animal" must survive 7 steps to reach "it"
# MAGIC - Information gets **diluted, lost, or overwritten** along the way
# MAGIC
# MAGIC ### 2. **Self-Attention (Transformers)** - Parallel Processing
# MAGIC - Processes ALL words simultaneously
# MAGIC - Every word can directly "look at" every other word
# MAGIC - **Solution**: "it" can directly attend to "animal" regardless of distance
# MAGIC - No information loss due to sequential processing
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## In This Notebook:
# MAGIC
# MAGIC We'll **visualize both approaches** and show:
# MAGIC 1. How RNNs process the sentence sequentially (with information decay)
# MAGIC 2. How self-attention processes all words in parallel
# MAGIC 3. Attention scores showing "it" attending directly to "animal"
# MAGIC 4. Why transformers revolutionized NLP

# COMMAND ----------

# DBTITLE 1,Setup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

np.random.seed(42)

print("✓ Libraries imported")

# COMMAND ----------

# DBTITLE 1,The Sentence
# MAGIC %md
# MAGIC ## Our Example Sentence
# MAGIC
# MAGIC Let's tokenize and analyze this sentence:
# MAGIC
# MAGIC **"The animal didn't cross the street because it was too tired"**

# COMMAND ----------

# DBTITLE 1,Tokenize Sentence
# Our example sentence
sentence = "The animal didn't cross the street because it was too tired"
tokens = sentence.split()

print(f"Sentence: {sentence}")
print(f"\nTokens ({len(tokens)} words):")
for i, token in enumerate(tokens):
    marker = "  ← KEY WORD" if token in ["animal", "it"] else ""
    print(f"  Position {i}: {token:10s}{marker}")

print(f"\n💡 Key observation:")
print(f"   'animal' is at position 1")
print(f"   'it' is at position 8")
print(f"   Distance: 7 words apart!")

# COMMAND ----------

# DBTITLE 1,RNN Approach
# MAGIC %md
# MAGIC ## Approach 1: RNN (Sequential Processing)
# MAGIC
# MAGIC ### How RNNs Work:
# MAGIC
# MAGIC ```
# MAGIC Step 1: Read "The"     → Hidden state h1
# MAGIC Step 2: Read "animal"  → Hidden state h2 (contains info about "The" and "animal")
# MAGIC Step 3: Read "didn't"  → Hidden state h3 (contains info from h2)
# MAGIC Step 4: Read "cross"   → Hidden state h4
# MAGIC Step 5: Read "the"     → Hidden state h5
# MAGIC Step 6: Read "street"  → Hidden state h6
# MAGIC Step 7: Read "because" → Hidden state h7
# MAGIC Step 8: Read "it"      → Hidden state h8 (should remember "animal" from h2!)
# MAGIC ...
# MAGIC ```
# MAGIC
# MAGIC ### The Problem:
# MAGIC
# MAGIC By step 8, the information about **"animal"** from step 2 has:
# MAGIC - Been **passed through 6 intermediate states**
# MAGIC - Been **mixed with** information from 6 other words
# MAGIC - Potentially been **diluted or forgotten**
# MAGIC
# MAGIC This is called **vanishing gradient** or **information decay**.
# MAGIC
# MAGIC ### Visualization:
# MAGIC
# MAGIC Let's simulate how RNN hidden states lose information over time.

# COMMAND ----------

# DBTITLE 1,Simulate RNN Hidden States
def simulate_rnn_processing(tokens, decay_rate=0.7):
    """
    Simulate RNN processing with information decay
    
    Args:
        tokens: List of words
        decay_rate: How much previous info is retained (0-1)
    
    Returns:
        hidden_states: Simulated hidden state vectors
        animal_signal: How much 'animal' info remains at each step
    """
    num_tokens = len(tokens)
    hidden_dim = 10
    
    # Initialize
    hidden_states = []
    current_hidden = np.zeros(hidden_dim)
    
    # Track 'animal' signal strength
    animal_signal = []
    
    for i, token in enumerate(tokens):
        # Simulate word embedding
        word_embedding = np.random.randn(hidden_dim) * 0.5
        
        # RNN update: combine previous state with new word
        # This simulates: h_t = tanh(W * [h_{t-1}, x_t])
        current_hidden = decay_rate * current_hidden + (1 - decay_rate) * word_embedding
        
        hidden_states.append(current_hidden.copy())
        
        # Track 'animal' signal
        if token == "animal":
            signal_strength = 1.0  # Start tracking from here
        elif i > tokens.index("animal"):
            # After 'animal', signal decays
            steps_after = i - tokens.index("animal")
            signal_strength = decay_rate ** steps_after
        else:
            signal_strength = 0.0
        
        animal_signal.append(signal_strength)
    
    return np.array(hidden_states), np.array(animal_signal)

# Simulate RNN
hidden_states_rnn, animal_signal_strength = simulate_rnn_processing(tokens, decay_rate=0.7)

print(f"RNN Processing Simulation:")
print(f"  Hidden state dimension: {hidden_states_rnn.shape[1]}")
print(f"  Number of time steps: {hidden_states_rnn.shape[0]}")
print(f"\n'Animal' signal strength at each position:")
for i, (token, strength) in enumerate(zip(tokens, animal_signal_strength)):
    bar = "█" * int(strength * 20)
    print(f"  {i:2d}. {token:10s}: {bar} {strength:.3f}")

print(f"\n⚠️  Notice: By the time we reach 'it' (position 8),")
print(f"    the 'animal' signal has decayed to {animal_signal_strength[8]:.1%}!")

# COMMAND ----------

# DBTITLE 1,Visualize RNN Sequential Processing
# Create visualization of RNN processing
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Top plot: Sequential processing flow
ax1.set_xlim(0, len(tokens) + 1)
ax1.set_ylim(0, 3)
ax1.axis('off')
ax1.set_title("RNN: Sequential Processing (Left to Right)\n", fontsize=14, weight='bold')

# Draw boxes for each word
for i, token in enumerate(tokens):
    x = i + 0.5
    
    # Color based on importance
    if token == "animal":
        color = 'lightgreen'
        label = f"{token}\n(source)"
    elif token == "it":
        color = 'lightcoral'
        label = f"{token}\n(target)"
    else:
        color = 'lightgray'
        label = token
    
    # Draw word box
    box = FancyBboxPatch((x - 0.3, 1.5), 0.6, 0.6, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='black', 
                          facecolor=color, 
                          linewidth=2)
    ax1.add_patch(box)
    ax1.text(x, 1.8, label, ha='center', va='center', fontsize=9, weight='bold')
    
    # Draw arrow to next word
    if i < len(tokens) - 1:
        arrow = FancyArrowPatch((x + 0.3, 1.8), (x + 0.7, 1.8),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='blue', alpha=0.6)
        ax1.add_patch(arrow)
    
    # Draw hidden state indicator
    ax1.plot([x], [0.8], 'o', markersize=15, color='steelblue', alpha=0.7)
    ax1.text(x, 0.8, f'h{i}', ha='center', va='center', 
            fontsize=8, color='white', weight='bold')

# Add labels
ax1.text(0.5, 2.5, "Time →", fontsize=12, style='italic')
ax1.text(0.2, 0.8, "Hidden\nStates:", fontsize=10, ha='right')

# Add problem annotation
ax1.annotate('Information must travel through\n6 intermediate states!', 
            xy=(tokens.index('animal') + 0.5, 1.8), 
            xytext=(5, 0.3),
            arrowprops=dict(arrowstyle='->', lw=2, color='red', 
                          connectionstyle="arc3,rad=.3"),
            fontsize=11, color='red', weight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.7))

# Bottom plot: Information decay
ax2.plot(range(len(tokens)), animal_signal_strength, 
        marker='o', linewidth=3, markersize=10, 
        color='green', label='"Animal" signal strength')

# Highlight key positions
ax2.axvline(tokens.index('animal'), color='green', linestyle='--', alpha=0.5, label='"animal" position')
ax2.axvline(tokens.index('it'), color='red', linestyle='--', alpha=0.5, label='"it" position')

# Mark the positions
ax2.scatter([tokens.index('animal')], [animal_signal_strength[tokens.index('animal')]], 
           s=200, color='green', zorder=5, edgecolor='black', linewidth=2)
ax2.scatter([tokens.index('it')], [animal_signal_strength[tokens.index('it')]], 
           s=200, color='red', zorder=5, edgecolor='black', linewidth=2)

ax2.set_xlabel('Position in Sentence', fontsize=12)
ax2.set_ylabel('Signal Strength', fontsize=12)
ax2.set_title('Information Decay: How "Animal" Signal Fades Over Time', fontsize=12, weight='bold')
ax2.set_xticks(range(len(tokens)))
ax2.set_xticklabels(tokens, rotation=45, ha='right')
ax2.set_ylim(-0.1, 1.1)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=10)

# Add annotation
ax2.annotate(f'Only {animal_signal_strength[tokens.index("it")]:.1%} of original signal!', 
            xy=(tokens.index('it'), animal_signal_strength[tokens.index('it')]), 
            xytext=(tokens.index('it') + 1, 0.4),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'),
            fontsize=11, color='red', weight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))

plt.tight_layout()
plt.show()

print("\n⚠️  RNN Problem: Sequential processing causes information decay!")
print(f"   By position {tokens.index('it')}, only {animal_signal_strength[tokens.index('it')]:.1%} of 'animal' info remains.")

# COMMAND ----------

# DBTITLE 1,Self-Attention Approach
# MAGIC %md
# MAGIC ## Approach 2: Self-Attention (Parallel Processing)
# MAGIC
# MAGIC ### How Self-Attention Works:
# MAGIC
# MAGIC ```
# MAGIC ALL words processed simultaneously!
# MAGIC
# MAGIC "The" can look at: [The, animal, didn't, cross, the, street, because, it, was, too, tired]
# MAGIC "animal" can look at: [The, animal, didn't, cross, the, street, because, it, was, too, tired]
# MAGIC "it" can look at: [The, animal, didn't, cross, the, street, because, it, was, too, tired]
# MAGIC ...
# MAGIC
# MAGIC Every word sees EVERY other word directly!
# MAGIC ```
# MAGIC
# MAGIC ### The Solution:
# MAGIC
# MAGIC **"it"** can directly attend to **"animal"** with **ZERO intermediate steps**!
# MAGIC
# MAGIC - No sequential processing
# MAGIC - No information decay
# MAGIC - **Direct connection** between any two words
# MAGIC - Distance doesn't matter!
# MAGIC
# MAGIC ### Let's Implement It:

# COMMAND ----------

# DBTITLE 1,Implement Self-Attention
def self_attention_for_sentence(tokens):
    """
    Compute self-attention for the sentence
    
    Returns:
        attention_weights: How much each word attends to every other word
    """
    num_tokens = len(tokens)
    d_model = 8  # Small dimension for this demo
    
    # Create word embeddings
    word_embeddings = np.random.randn(num_tokens, d_model) * 0.5
    
    # Create Q, K, V matrices (simplified)
    W_Q = np.random.randn(d_model, d_model) * 0.1
    W_K = np.random.randn(d_model, d_model) * 0.1
    W_V = np.random.randn(d_model, d_model) * 0.1
    
    Q = word_embeddings @ W_Q
    K = word_embeddings @ W_K
    V = word_embeddings @ W_V
    
    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_model)
    
    # Apply softmax to get attention weights
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
    
    # Boost 'it' -> 'animal' connection for demonstration
    # (In real transformers, this emerges from training)
    it_idx = tokens.index('it')
    animal_idx = tokens.index('animal')
    attention_weights[it_idx, animal_idx] *= 2.5
    # Re-normalize that row
    attention_weights[it_idx] /= attention_weights[it_idx].sum()
    
    return attention_weights

# Compute self-attention
attention_matrix = self_attention_for_sentence(tokens)

print(f"Self-Attention Matrix Shape: {attention_matrix.shape}")
print(f"  - {attention_matrix.shape[0]} queries (one per word)")
print(f"  - {attention_matrix.shape[1]} keys (attending to all words)")

print(f"\n'it' (position {tokens.index('it')}) attention distribution:")
it_attention = attention_matrix[tokens.index('it')]
for token, score in zip(tokens, it_attention):
    bar = "█" * int(score * 50)
    print(f"  {token:10s}: {bar} {score:.3f}")

print(f"\n✅ Notice: 'it' pays {it_attention[tokens.index('animal')]:.1%} attention to 'animal'!")
print(f"   Direct connection, no information loss!")

# COMMAND ----------

# DBTITLE 1,Visualize Self-Attention
# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.5, 1], hspace=0.3, wspace=0.3)

# 1. Parallel processing diagram
ax1 = fig.add_subplot(gs[0, :])
ax1.set_xlim(0, len(tokens) + 1)
ax1.set_ylim(0, 3)
ax1.axis('off')
ax1.set_title("Self-Attention: All Words Processed Simultaneously (Parallel)", 
             fontsize=14, weight='bold')

# Draw all words
for i, token in enumerate(tokens):
    x = i + 0.5
    
    if token == "animal":
        color = 'lightgreen'
        label = f"{token}\n(source)"
    elif token == "it":
        color = 'lightcoral'
        label = f"{token}\n(target)"
    else:
        color = 'lightblue'
        label = token
    
    box = FancyBboxPatch((x - 0.3, 1.5), 0.6, 0.6, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='black', 
                          facecolor=color, 
                          linewidth=2)
    ax1.add_patch(box)
    ax1.text(x, 1.8, label, ha='center', va='center', fontsize=9, weight='bold')

# Draw direct connection from 'it' to 'animal'
animal_x = tokens.index('animal') + 0.5
it_x = tokens.index('it') + 0.5

arrow = FancyArrowPatch(
    (it_x, 1.5), (animal_x, 1.5),
    arrowstyle='<->', mutation_scale=30, 
    linewidth=4, color='red', alpha=0.8,
    connectionstyle="arc3,rad=.3"
)
ax1.add_patch(arrow)

ax1.text((it_x + animal_x) / 2, 0.8, 
        'Direct Connection!\nNo intermediate steps', 
        ha='center', fontsize=11, color='red', weight='bold',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))

# 2. Full attention heatmap
ax2 = fig.add_subplot(gs[1, :])
sns.heatmap(attention_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
           xticklabels=tokens, yticklabels=tokens, ax=ax2,
           cbar_kws={'label': 'Attention Weight'}, vmin=0, vmax=0.5)
ax2.set_title('Complete Attention Matrix: Every Word Attends to Every Other Word', 
             fontsize=12, weight='bold')
ax2.set_xlabel('Attending to (Key)', fontsize=11)
ax2.set_ylabel('Attention from (Query)', fontsize=11)

# Highlight the 'it' -> 'animal' connection
ax2.add_patch(plt.Rectangle((tokens.index('animal'), tokens.index('it')), 
                            1, 1, fill=False, edgecolor='red', linewidth=4))

# 3. Focus on 'it' attention
ax3 = fig.add_subplot(gs[2, 0])
it_attention = attention_matrix[tokens.index('it')]
colors = ['green' if token == 'animal' else 'red' if token == 'it' else 'steelblue' 
          for token in tokens]
ax3.bar(range(len(tokens)), it_attention, color=colors, alpha=0.7, edgecolor='black')
ax3.set_xlabel('Words', fontsize=11)
ax3.set_ylabel('Attention Weight', fontsize=11)
ax3.set_title('"it" Attention Distribution', fontsize=12, weight='bold')
ax3.set_xticks(range(len(tokens)))
ax3.set_xticklabels(tokens, rotation=45, ha='right')
ax3.grid(True, alpha=0.3, axis='y')

# Highlight animal bar
ax3.bar([tokens.index('animal')], [it_attention[tokens.index('animal')]], 
       color='green', alpha=0.9, edgecolor='black', linewidth=3)
ax3.annotate(f'{it_attention[tokens.index("animal")]:.1%}', 
            xy=(tokens.index('animal'), it_attention[tokens.index('animal')]), 
            xytext=(tokens.index('animal'), it_attention[tokens.index('animal')] + 0.05),
            fontsize=12, weight='bold', ha='center')

# 4. Focus on 'animal' attention
ax4 = fig.add_subplot(gs[2, 1])
animal_attention = attention_matrix[tokens.index('animal')]
colors = ['green' if token == 'animal' else 'red' if token == 'it' else 'steelblue' 
          for token in tokens]
ax4.bar(range(len(tokens)), animal_attention, color=colors, alpha=0.7, edgecolor='black')
ax4.set_xlabel('Words', fontsize=11)
ax4.set_ylabel('Attention Weight', fontsize=11)
ax4.set_title('"animal" Attention Distribution', fontsize=12, weight='bold')
ax4.set_xticks(range(len(tokens)))
ax4.set_xticklabels(tokens, rotation=45, ha='right')
ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle('Self-Attention: Direct Connections Between All Words', 
            fontsize=16, weight='bold', y=0.995)
plt.show()

print("\n✅ Self-Attention Advantage: Direct connections with NO information loss!")

# COMMAND ----------

# DBTITLE 1,Side-by-Side Comparison
# MAGIC %md
# MAGIC ## Direct Comparison: RNN vs Self-Attention
# MAGIC
# MAGIC Let's compare both approaches side-by-side for connecting **"it"** to **"animal"**:

# COMMAND ----------

# DBTITLE 1,Comparison Visualization
# Create comparison figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# LEFT: RNN approach
ax1.set_title('RNN: Sequential Processing\n(Information Decays)', fontsize=14, weight='bold')
ax1.set_xlim(-1, 10)
ax1.set_ylim(0, 8)
ax1.axis('off')

# Draw animal
ax1.add_patch(FancyBboxPatch((0, 6), 2, 1, boxstyle="round,pad=0.1",
                             facecolor='lightgreen', edgecolor='black', linewidth=2))
ax1.text(1, 6.5, 'animal\n(position 1)', ha='center', va='center', fontsize=11, weight='bold')

# Draw intermediate words
for i, word in enumerate(['didn\'t', 'cross', 'the', 'street', 'because']):
    y = 5 - i
    alpha = 0.7 ** (i + 1)  # Decay
    ax1.add_patch(FancyBboxPatch((3, y - 0.5), 2, 1, boxstyle="round,pad=0.1",
                                 facecolor='lightgray', edgecolor='black', 
                                 linewidth=2, alpha=alpha))
    ax1.text(4, y, word, ha='center', va='center', fontsize=10)
    
    # Arrow
    if i == 0:
        ax1.annotate('', xy=(3, y), xytext=(2, 6.2),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=alpha))
    else:
        ax1.annotate('', xy=(3, y), xytext=(5, prev_y),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=alpha))
    prev_y = y

# Draw 'it'
ax1.add_patch(FancyBboxPatch((7, 0), 2, 1, boxstyle="round,pad=0.1",
                             facecolor='lightcoral', edgecolor='black', linewidth=2))
ax1.text(8, 0.5, 'it\n(position 8)', ha='center', va='center', fontsize=11, weight='bold')
ax1.annotate('', xy=(7, 0.5), xytext=(5, 0.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=0.2))

# Add labels
ax1.text(5, 7.5, 'Information travels through\n6 intermediate steps', 
        ha='center', fontsize=11, color='red', weight='bold',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.7))
ax1.text(8, -0.8, f'Signal strength: {animal_signal_strength[tokens.index("it")]:.1%}', 
        ha='center', fontsize=10, color='red', weight='bold')

# RIGHT: Self-Attention approach
ax2.set_title('Self-Attention: Parallel Processing\n(Direct Connection)', fontsize=14, weight='bold')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 8)
ax2.axis('off')

# Draw animal
ax2.add_patch(FancyBboxPatch((1, 6), 2, 1, boxstyle="round,pad=0.1",
                             facecolor='lightgreen', edgecolor='black', linewidth=2))
ax2.text(2, 6.5, 'animal\n(position 1)', ha='center', va='center', fontsize=11, weight='bold')

# Draw other words (all in parallel)
other_words = ['The', 'didn\'t', 'cross', 'the', 'street', 'because', 'was', 'too', 'tired']
for i, word in enumerate(other_words):
    x = 1 + (i % 3) * 2
    y = 4 - (i // 3) * 1.5
    ax2.add_patch(FancyBboxPatch((x - 0.5, y - 0.4), 1, 0.8, boxstyle="round,pad=0.05",
                                 facecolor='lightblue', edgecolor='black', 
                                 linewidth=1, alpha=0.5))
    ax2.text(x, y, word, ha='center', va='center', fontsize=8)

# Draw 'it'
ax2.add_patch(FancyBboxPatch((7, 2), 2, 1, boxstyle="round,pad=0.1",
                             facecolor='lightcoral', edgecolor='black', linewidth=2))
ax2.text(8, 2.5, 'it\n(position 8)', ha='center', va='center', fontsize=11, weight='bold')

# DIRECT arrow from 'it' to 'animal'
ax2.annotate('', xy=(3, 6.2), xytext=(7.5, 2.8),
            arrowprops=dict(arrowstyle='<->', lw=4, color='red', 
                          connectionstyle="arc3,rad=.3"))

ax2.text(5, 5, 'DIRECT CONNECTION!\nZero intermediate steps\nNo information loss', 
        ha='center', fontsize=12, color='red', weight='bold',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.9))
ax2.text(8, 1.2, f'Signal strength: 100%', 
        ha='center', fontsize=10, color='green', weight='bold')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)
print(f"\nRNN (Sequential):")
print(f"  ❌ Must process words one-by-one")
print(f"  ❌ Information decays over distance")
print(f"  ❌ 'animal' signal at 'it': {animal_signal_strength[tokens.index('it')]:.1%}")
print(f"  ❌ Takes {len(tokens)} time steps")

print(f"\nSelf-Attention (Parallel):")
print(f"  ✅ Processes all words simultaneously")
print(f"  ✅ Direct connections between any words")
print(f"  ✅ 'animal' signal at 'it': {it_attention[tokens.index('animal')]:.1%}")
print(f"  ✅ Takes 1 time step (parallel)")

print(f"\n🎯 Winner: Self-Attention solves the long-term dependency problem!")

# COMMAND ----------

# DBTITLE 1,Key Insights
# MAGIC %md
# MAGIC ## Key Insights
# MAGIC
# MAGIC ### Why RNNs Struggle:
# MAGIC
# MAGIC 1. **Sequential Processing**:
# MAGIC    - Must process words one-by-one
# MAGIC    - Information from word 1 must pass through all intermediate words to reach word 8
# MAGIC    - Each step potentially loses or dilutes information
# MAGIC
# MAGIC 2. **Vanishing Gradient Problem**:
# MAGIC    - During training, gradients must backpropagate through time
# MAGIC    - Gradients get smaller (vanish) with each step backward
# MAGIC    - Hard to learn long-range dependencies
# MAGIC
# MAGIC 3. **Hidden State Bottleneck**:
# MAGIC    - All information must be compressed into fixed-size hidden state
# MAGIC    - Older information gets overwritten by newer information
# MAGIC
# MAGIC ### Why Self-Attention Succeeds:
# MAGIC
# MAGIC 1. **Parallel Processing**:
# MAGIC    - All words processed simultaneously
# MAGIC    - No sequential bottleneck
# MAGIC    - Much faster computation (on GPUs)
# MAGIC
# MAGIC 2. **Direct Connections**:
# MAGIC    - Every word can attend to every other word directly
# MAGIC    - No intermediate steps → no information loss
# MAGIC    - Distance doesn't matter!
# MAGIC
# MAGIC 3. **Flexible Attention**:
# MAGIC    - Model learns which words are important to attend to
# MAGIC    - Different attention patterns for different contexts
# MAGIC    - Multi-head attention captures multiple relationship types
# MAGIC
# MAGIC ### The Transformer Revolution:
# MAGIC
# MAGIC Self-attention (introduced in "Attention Is All You Need", 2017) **revolutionized NLP** because:
# MAGIC
# MAGIC ✅ **Solves long-term dependencies** - Direct connections between any words
# MAGIC ✅ **Faster training** - Parallel processing on GPUs
# MAGIC ✅ **Better performance** - Can capture complex relationships
# MAGIC ✅ **Scalable** - Can stack many layers without vanishing gradients
# MAGIC
# MAGIC ### Real-World Impact:
# MAGIC
# MAGIC This architectural change enabled:
# MAGIC - **BERT** (2018) - Bidirectional encoder
# MAGIC - **GPT** (2018-2024) - Decoder-only models
# MAGIC - **T5**, **BART** - Encoder-decoder models
# MAGIC - All modern LLMs!
# MAGIC
# MAGIC Without self-attention, we wouldn't have:
# MAGIC - ChatGPT
# MAGIC - Claude
# MAGIC - Gemini
# MAGIC - Modern translation systems
# MAGIC - Question-answering systems

# COMMAND ----------

# DBTITLE 1,Summary
# MAGIC %md
# MAGIC ## Summary: RNN vs Self-Attention
# MAGIC
# MAGIC ### The Problem:
# MAGIC
# MAGIC **Long-Term Dependencies**: When words are far apart, how does the model connect them?
# MAGIC
# MAGIC Example: "The animal didn't cross the street because **it** was too tired."
# MAGIC - "animal" at position 1
# MAGIC - "it" at position 8
# MAGIC - 7 words apart!
# MAGIC
# MAGIC ### The RNN Solution (Old Way):
# MAGIC
# MAGIC ```
# MAGIC Animal → didn't → cross → the → street → because → it
# MAGIC   100%     80%      64%    51%    41%      33%      26%
# MAGIC   
# MAGIC ❌ Information decays with distance
# MAGIC ❌ Sequential processing (slow)
# MAGIC ❌ Vanishing gradient problem
# MAGIC ```
# MAGIC
# MAGIC ### The Self-Attention Solution (New Way):
# MAGIC
# MAGIC ```
# MAGIC All words processed simultaneously!
# MAGIC
# MAGIC Animal ←─────── Direct Connection ────────→ it
# MAGIC   100%                                      100%
# MAGIC   
# MAGIC ✅ No information loss
# MAGIC ✅ Parallel processing (fast)
# MAGIC ✅ Direct connections
# MAGIC ```
# MAGIC
# MAGIC ### Key Formulas:
# MAGIC
# MAGIC **RNN**:
# MAGIC ```
# MAGIC h_t = f(h_{t-1}, x_t)  # Sequential
# MAGIC ```
# MAGIC
# MAGIC **Self-Attention**:
# MAGIC ```
# MAGIC Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V  # Parallel
# MAGIC ```
# MAGIC
# MAGIC ### Why It Matters:
# MAGIC
# MAGIC Self-attention's ability to connect any two words **directly** (regardless of distance) is why transformers:
# MAGIC - Outperform RNNs on virtually all NLP tasks
# MAGIC - Train faster (parallel computation)
# MAGIC - Scale to billions of parameters
# MAGIC - Power modern AI (GPT, BERT, Claude, etc.)
# MAGIC
# MAGIC ### The Takeaway:
# MAGIC
# MAGIC **Distance is irrelevant with self-attention!**
# MAGIC
# MAGIC Whether words are 1 position apart or 1000 positions apart, self-attention connects them with equal ease. This is the fundamental breakthrough that enabled modern large language models.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Further Reading:
# MAGIC
# MAGIC To understand how self-attention works in detail, check out the other notebooks:
# MAGIC 1. **Self-Attention Mechanism** - Matrix math and implementation
# MAGIC 2. **Multi-Head Attention** - Multiple perspectives simultaneously
# MAGIC 3. **Positional Encoding** - Adding word order information
# MAGIC 4. **Complete Transformer Encoder** - Full architecture
# MAGIC
# MAGIC **Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)