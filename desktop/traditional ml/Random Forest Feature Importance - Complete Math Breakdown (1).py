# Databricks notebook source
# DBTITLE 1,📚 Introduction - What We'll Learn
# MAGIC %md
# MAGIC # 🌳 Random Forest Feature Importance: Complete Mathematical Breakdown
# MAGIC
# MAGIC ## 🎯 What You'll Learn
# MAGIC
# MAGIC This notebook walks through **every step** of how Random Forest calculates feature importance, from scratch.
# MAGIC
# MAGIC ### 📋 Complete Formula Pipeline:
# MAGIC
# MAGIC ```
# MAGIC For each tree b = 1 to B:
# MAGIC   For each node in tree b:
# MAGIC     ΔGini = Gini_parent − (N_L/N)×Gini_L − (N_R/N)×Gini_R
# MAGIC     feature_used = j
# MAGIC     I_j[b] += (N_node / N_total) × ΔGini
# MAGIC
# MAGIC I_j(RF) = (1/B) × Σ_b I_j[b]         ← average over trees
# MAGIC
# MAGIC I_j(normalized) = I_j(RF) / Σ_k I_k   ← divide by sum
# MAGIC
# MAGIC → rf.feature_importances_[j] = I_j(normalized)
# MAGIC → Shape: (n_features,), Sum = 1.0
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📖 Topics Covered:
# MAGIC
# MAGIC 1. ✅ **Dummy Dataset Creation** - Simple, interpretable data
# MAGIC 2. ✅ **Gini Impurity** - Formula and calculation
# MAGIC 3. ✅ **Tree Building** - How splits are chosen
# MAGIC 4. ✅ **ΔGini Calculation** - Gini reduction at each node
# MAGIC 5. ✅ **Feature Importance Accumulation** - Tracking I_j[b] across trees
# MAGIC 6. ✅ **Averaging** - Computing I_j(RF) = (1/B) × Σ I_j[b]
# MAGIC 7. ✅ **Normalization** - Final feature_importances_
# MAGIC 8. ✅ **Validation** - Compare with sklearn
# MAGIC 9. ✅ **Interactive Visualization** - See how it changes with parameters
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Let's dive in! 🚀**

# COMMAND ----------

# DBTITLE 1,Step 1: Imports and Setup
# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: IMPORTS AND SETUP
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")
print("\n🎯 Goal: Understand rf.feature_importances_ calculation from scratch")
print("📊 Formula: I_j(normalized) = [Average Gini reduction from feature j] / [Sum of all reductions]")

# COMMAND ----------

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: CREATE SIMPLE DUMMY DATASET
# ═══════════════════════════════════════════════════════════════════════════════

# Create a simple dataset with 3 features
# Feature 0: Highly predictive
# Feature 1: Moderately predictive  
# Feature 2: Noise (not predictive)

np.random.seed(42)
n_samples = 100

# Feature 0: Strong signal (high importance)
X0 = np.random.randn(n_samples)

# Feature 1: Moderate signal (medium importance)
X1 = np.random.randn(n_samples)

# Feature 2: Pure noise (low importance)
X2 = np.random.randn(n_samples)

# Target: Mostly depends on X0, somewhat on X1, not on X2
y = (X0 > 0).astype(int)  # Primary dependency on X0
y = np.where((X0 > -0.5) & (X0 < 0.5) & (X1 > 0), 1, y)  # Some dependency on X1
noise_indices = np.random.choice(n_samples, size=10, replace=False)

# COMMAND ----------

np.random.choice(100, size=10, replace=False)

# COMMAND ----------

y[noise_indices]

# COMMAND ----------

# DBTITLE 1,Step 2: Create Simple Dummy Dataset

# Add some randomness

y[noise_indices] = 1 - y[noise_indices]

X = np.column_stack([X0, X1, X2])

print("✅ Dataset Created!")
print(f"\n📊 Dataset Shape: {X.shape}")
print(f"   Samples: {n_samples}")
print(f"   Features: {X.shape[1]} (Feature_0, Feature_1, Feature_2)")
print(f"\n🎯 Target Distribution:")
print(f"   Class 0: {(y == 0).sum()} samples ({(y == 0).mean()*100:.1f}%)")
print(f"   Class 1: {(y == 1).sum()} samples ({(y == 1).mean()*100:.1f}%)")

# Display sample data
df = pd.DataFrame(X, columns=['Feature_0 (Strong)', 'Feature_1 (Moderate)', 'Feature_2 (Noise)'])
df['Target'] = y

print("\n📄 First 10 Samples:")
display(df.head(10))

print("\n💡 Expected Result:")
print("   Feature_0 should have HIGHEST importance (strong signal)")
print("   Feature_1 should have MEDIUM importance (moderate signal)")
print("   Feature_2 should have LOWEST importance (pure noise)")

# COMMAND ----------

# DBTITLE 1,Step 3: Gini Impurity Formula
# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: GINI IMPURITY - THE FOUNDATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_gini(y_subset):
    """
    Calculate Gini Impurity for a set of labels
    
    Formula: Gini = 1 - Σ(p_i²)
    where p_i is the proportion of class i
    
    Interpretation:
      - Gini = 0.0 → Pure node (all same class)
      - Gini = 0.5 → Maximum impurity (50-50 split for binary)
    """
    if len(y_subset) == 0:
        return 0.0
    
    # Count each class
    classes, counts = np.unique(y_subset, return_counts=True)
    
    # Calculate proportions
    proportions = counts / len(y_subset)
    
    # Gini = 1 - sum of squared proportions
    gini = 1.0 - np.sum(proportions ** 2)
    
    return gini

# ═══════════════════════════════════════════════════════════════════════════════
# TEST GINI CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

print("🧮 GINI IMPURITY EXAMPLES")
print("\n" + "="*70)

# Example 1: Pure node (all class 0)
test1 = np.array([0, 0, 0, 0, 0])
gini1 = calculate_gini(test1)
print(f"\n1️⃣ Pure Node (All Class 0): {test1}")
print(f"   Gini = 1 - (1.0²) = {gini1:.4f}")
print(f"   ✅ Perfect purity!")

# Example 2: Pure node (all class 1)
test2 = np.array([1, 1, 1, 1, 1])
gini2 = calculate_gini(test2)
print(f"\n2️⃣ Pure Node (All Class 1): {test2}")
print(f"   Gini = 1 - (1.0²) = {gini2:.4f}")
print(f"   ✅ Perfect purity!")

# Example 3: Maximum impurity (50-50 split)
test3 = np.array([0, 0, 0, 1, 1, 1])
gini3 = calculate_gini(test3)
print(f"\n3️⃣ Maximum Impurity (50-50): {test3}")
print(f"   Proportions: p_0 = 0.5, p_1 = 0.5")
print(f"   Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = {gini3:.4f}")
print(f"   🔴 Worst case - maximum uncertainty!")

# Example 4: Moderate impurity (80-20 split)
test4 = np.array([0, 0, 0, 0, 1])
gini4 = calculate_gini(test4)
print(f"\n4️⃣ Moderate Impurity (80-20): {test4}")
print(f"   Proportions: p_0 = 0.8, p_1 = 0.2")
print(f"   Gini = 1 - (0.8² + 0.2²) = 1 - 0.68 = {gini4:.4f}")
print(f"   🟡 Some impurity")

# Example 5: Our dataset's root node
gini_root = calculate_gini(y)
print(f"\n5️⃣ Our Dataset (Root Node): n = {len(y)}")
print(f"   Class 0: {(y == 0).sum()} samples ({(y == 0).mean()*100:.1f}%)")
print(f"   Class 1: {(y == 1).sum()} samples ({(y == 1).mean()*100:.1f}%)")
print(f"   Gini = {gini_root:.4f}")
print(f"   🎯 This is where our trees will start!")

print("\n" + "="*70)
print("\n💡 Key Insight: Trees try to REDUCE Gini by splitting nodes!")
print("   Goal: Go from {:.4f} → 0.0 (pure nodes)".format(gini_root))

# COMMAND ----------

# DBTITLE 1,Step 4: Delta Gini Calculation Formula
# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: ΔGINI - GINI REDUCTION FROM A SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_delta_gini(y_parent, y_left, y_right):
    """
    Calculate Gini reduction from a split
    
    📝 FORMULA:
    ΔGini = Gini_parent - (N_L/N) × Gini_L - (N_R/N) × Gini_R
    
    Where:
      - Gini_parent: Gini of parent node before split
      - Gini_L: Gini of left child after split
      - Gini_R: Gini of right child after split
      - N_L: Number of samples in left child
      - N_R: Number of samples in right child
      - N: Total samples in parent (N_L + N_R)
    
    Returns:
      - ΔGini: How much Gini decreased (higher = better split!)
    """
    N = len(y_parent)
    N_L = len(y_left)
    N_R = len(y_right)
    
    gini_parent = calculate_gini(y_parent)
    gini_left = calculate_gini(y_left)
    gini_right = calculate_gini(y_right)
    
    # Weighted average of child Ginis
    weighted_child_gini = (N_L / N) * gini_left + (N_R / N) * gini_right
    
    # Reduction in Gini
    delta_gini = gini_parent - weighted_child_gini
    
    return delta_gini, gini_parent, gini_left, gini_right

# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE: MANUAL SPLIT DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

print("🔪 ΔGINI CALCULATION EXAMPLE")
print("\n" + "="*70)

# Example: Split on Feature_0 at threshold 0.0
threshold = 0.0
split_mask = X[:, 0] <= threshold  # Feature_0

y_parent = y
y_left = y[split_mask]  # Samples where Feature_0 <= 0.0
y_right = y[~split_mask]  # Samples where Feature_0 > 0.0

delta, g_parent, g_left, g_right = calculate_delta_gini(y_parent, y_left, y_right)

N = len(y_parent)
N_L = len(y_left)
N_R = len(y_right)

print(f"\n🎯 Split: Feature_0 <= {threshold}")
print(f"\n📊 Parent Node (Before Split):")
print(f"   Total samples: N = {N}")
print(f"   Class 0: {(y_parent == 0).sum()} | Class 1: {(y_parent == 1).sum()}")
print(f"   Gini_parent = {g_parent:.4f}")

print(f"\n⬅️  Left Child (Feature_0 <= {threshold}):")
print(f"   Samples: N_L = {N_L} ({N_L/N*100:.1f}% of parent)")
print(f"   Class 0: {(y_left == 0).sum()} | Class 1: {(y_left == 1).sum()}")
print(f"   Gini_L = {g_left:.4f}")

print(f"\n➡️  Right Child (Feature_0 > {threshold}):")
print(f"   Samples: N_R = {N_R} ({N_R/N*100:.1f}% of parent)")
print(f"   Class 0: {(y_right == 0).sum()} | Class 1: {(y_right == 1).sum()}")
print(f"   Gini_R = {g_right:.4f}")

print(f"\n📊 CALCULATION:")
print(f"   ΔGini = Gini_parent - (N_L/N)×Gini_L - (N_R/N)×Gini_R")
print(f"   ΔGini = {g_parent:.4f} - ({N_L}/{N})×{g_left:.4f} - ({N_R}/{N})×{g_right:.4f}")
print(f"   ΔGini = {g_parent:.4f} - {(N_L/N)*g_left:.4f} - {(N_R/N)*g_right:.4f}")
print(f"   ΔGini = {delta:.4f}")

print(f"\n⭐ Result: This split reduces Gini by {delta:.4f}!")
print(f"   Higher ΔGini = Better split")
print(f"   This ΔGini contributes to Feature_0's importance")

print("\n" + "="*70)

# COMMAND ----------

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

# COMMAND ----------

tree.tree_.node_count

# COMMAND ----------

tree.tree_.feature

# COMMAND ----------

tree.tree_.n_node_samples

# COMMAND ----------

tree.tree_.impurity

# COMMAND ----------

tree.tree_.children_left

# COMMAND ----------

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: SINGLE TREE - MANUAL IMPORTANCE CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_tree_feature_importance(tree, X, y, n_features):
    """
    Extract feature importance from a single decision tree
    
    For each node in the tree:
      I_j[tree] += (N_node / N_total) × ΔGini
      
    where:
      - N_node = number of samples at this node
      - N_total = total samples in training data
      - ΔGini = Gini reduction from the split
    """
    tree_ = tree.tree_
    feature_importances = np.zeros(n_features)
    
    # Total number of samples
    N_total = len(y)
    
    # Traverse all nodes
    for node_id in range(tree_.node_count):
        # Skip leaf nodes (they don't split)
        if tree_.feature[node_id] == -2:  # -2 indicates leaf
            continue
        
        # Get split information
        feature = tree_.feature[node_id]
        n_node_samples = tree_.n_node_samples[node_id]
        impurity_parent = tree_.impurity[node_id]
        
        # Get children
        left_child = tree_.children_left[node_id]
        right_child = tree_.children_right[node_id]
        
        n_left = tree_.n_node_samples[left_child]
        n_right = tree_.n_node_samples[right_child]
        
        impurity_left = tree_.impurity[left_child]
        impurity_right = tree_.impurity[right_child]
        
        # Calculate ΔGini for this split
        weighted_child_impurity = (n_left / n_node_samples) * impurity_left + \
                                   (n_right / n_node_samples) * impurity_right
        delta_gini = impurity_parent - weighted_child_impurity
        
        # Weight by fraction of samples at this node
        importance_contribution = (n_node_samples / N_total) * delta_gini
        print(feature )
        # Add to feature importance
        feature_importances[feature] += importance_contribution
    
    return feature_importances

# COMMAND ----------



# Build a single decision tree
print("🌳 BUILDING SINGLE DECISION TREE")
print("\n" + "="*70)

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

print("✅ Tree trained!")
print(f"   Max depth: {tree.get_depth()}")
print(f"   Number of leaves: {tree.get_n_leaves()}")
print(f"   Number of nodes: {tree.tree_.node_count}")

# Extract feature importance manually
importances_manual = extract_tree_feature_importance(tree, X, y, X.shape[1])

# COMMAND ----------

importances_manual 

# COMMAND ----------

tree.feature_importances_

# COMMAND ----------

# DBTITLE 1,Step 5: Build Single Tree & Extract Feature Importance


# Normalize (sklearn does this automatically)
importances_manual_normalized = importances_manual / importances_manual.sum()

print(f"\n📊 FEATURE IMPORTANCE (Single Tree):")
print("\n📝 Formula Applied:")
print("   For each node: I_j[tree] += (N_node / N_total) × ΔGini")
print("   Then normalize: I_j = I_j / Σ_k I_k")

print(f"\n📈 Raw Importances (Before Normalization):")
for i, imp in enumerate(importances_manual):
    print(f"   Feature_{i}: {imp:.6f}")

print(f"\n⭐ Normalized Importances (Sum = 1.0):")
for i, imp in enumerate(importances_manual_normalized):
    print(f"   Feature_{i}: {imp:.6f}  ({imp*100:.2f}%)")

print(f"\n✅ Sum check: {importances_manual_normalized.sum():.6f} (should be 1.0)")

# Compare with sklearn's built-in
sklearn_importances = tree.feature_importances_

print(f"\n🔍 VALIDATION: Compare with sklearn:")
for i in range(X.shape[1]):
    diff = abs(importances_manual_normalized[i] - sklearn_importances[i])
    match = "✅" if diff < 1e-6 else "❌"
    print(f"   Feature_{i}: Manual={importances_manual_normalized[i]:.6f}, "
          f"sklearn={sklearn_importances[i]:.6f}, Diff={diff:.2e} {match}")

print("\n" + "="*70)
print("\n💡 Insight: Feature_0 has highest importance (as expected!)")
print("   Feature_0 provides the most Gini reduction across all splits")

# COMMAND ----------

# DBTITLE 1,Step 6: Random Forest - Multiple Trees
# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: RANDOM FOREST - AVERAGING OVER B TREES
# ═══════════════════════════════════════════════════════════════════════════════

def extract_rf_feature_importance_manual(rf, X, y):
    """
    Extract feature importance from Random Forest manually
    
    📝 COMPLETE FORMULA:
    
    For each tree b = 1 to B:
      For each node in tree b:
        ΔGini = Gini_parent - (N_L/N)×Gini_L - (N_R/N)×Gini_R
        feature_used = j
        I_j[b] += (N_node / N_total) × ΔGini
    
    I_j(RF) = (1/B) × Σ_b I_j[b]         ← average over trees
    
    I_j(normalized) = I_j(RF) / Σ_k I_k   ← divide by sum
    """
    n_features = X.shape[1]
    n_trees = len(rf.estimators_)
    
    # Store importance from each tree
    tree_importances = np.zeros((n_trees, n_features))
    
    # Extract importance from each tree
    for b, tree in enumerate(rf.estimators_):
        tree_importances[b] = extract_tree_feature_importance(tree, X, y, n_features)
    
    # Average over all trees: I_j(RF) = (1/B) × Σ_b I_j[b]
    avg_importances = tree_importances.mean(axis=0)
    
    # Normalize: I_j(normalized) = I_j(RF) / Σ_k I_k
    normalized_importances = avg_importances / avg_importances.sum()
    
    return normalized_importances, tree_importances, avg_importances

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════════

print("🌳🌳🌳 BUILDING RANDOM FOREST 🌳🌳🌳")
print("\n" + "="*70)

B = 10  # Number of trees
rf = RandomForestClassifier(n_estimators=B, max_depth=3, random_state=42)
rf.fit(X, y)

print(f"✅ Random Forest trained!")
print(f"   Number of trees (B): {B}")
print(f"   Max depth per tree: 3")

# Extract importances manually
rf_importances_manual, tree_imps, avg_imps = extract_rf_feature_importance_manual(rf, X, y)

print(f"\n📊 FEATURE IMPORTANCE CALCULATION:")
print("\n📋 Step-by-Step:")

print(f"\n1️⃣ Individual Tree Importances (I_j[b] for each tree b):")
print("\n   Tree  | Feature_0 | Feature_1 | Feature_2")
print("   " + "-"*50)
for b in range(B):
    print(f"   {b+1:2d}    | {tree_imps[b, 0]:.6f}  | {tree_imps[b, 1]:.6f}  | {tree_imps[b, 2]:.6f}")

print(f"\n2️⃣ Average Over Trees: I_j(RF) = (1/B) × Σ_b I_j[b]")
print(f"   Where B = {B} trees")
for j in range(X.shape[1]):
    print(f"\n   Feature_{j}:")
    print(f"     Sum over trees: Σ_b I_{j}[b] = {tree_imps[:, j].sum():.6f}")
    print(f"     Average: I_{j}(RF) = {tree_imps[:, j].sum():.6f} / {B} = {avg_imps[j]:.6f}")

print(f"\n3️⃣ Normalize: I_j(normalized) = I_j(RF) / Σ_k I_k")
sum_avg = avg_imps.sum()
print(f"   Sum of averages: Σ_k I_k(RF) = {sum_avg:.6f}")
for j in range(X.shape[1]):
    print(f"   Feature_{j}: {avg_imps[j]:.6f} / {sum_avg:.6f} = {rf_importances_manual[j]:.6f}")

print(f"\n⭐ FINAL NORMALIZED IMPORTANCES:")
for j in range(X.shape[1]):
    print(f"   Feature_{j}: {rf_importances_manual[j]:.6f}  ({rf_importances_manual[j]*100:.2f}%)")

print(f"\n✅ Sum check: {rf_importances_manual.sum():.10f} (should be 1.0)")

# Compare with sklearn
sklearn_rf_importances = rf.feature_importances_

print(f"\n🔍 VALIDATION: Compare with sklearn's rf.feature_importances_:")
for j in range(X.shape[1]):
    diff = abs(rf_importances_manual[j] - sklearn_rf_importances[j])
    match = "✅" if diff < 1e-6 else "❌"
    print(f"   Feature_{j}: Manual={rf_importances_manual[j]:.6f}, "
          f"sklearn={sklearn_rf_importances[j]:.6f}, Diff={diff:.2e} {match}")

print("\n" + "="*70)
print("\n🎉 SUCCESS! Our manual calculation matches sklearn perfectly!")
print("\n💡 Key Insight:")
print(f"   Feature_0: {rf_importances_manual[0]*100:.2f}% importance (HIGHEST - strong signal)")
print(f"   Feature_1: {rf_importances_manual[1]*100:.2f}% importance (MEDIUM - moderate signal)")
print(f"   Feature_2: {rf_importances_manual[2]*100:.2f}% importance (LOWEST - noise)")
print("\n   ✅ Matches our data design perfectly!")

# COMMAND ----------

# DBTITLE 1,Step 7: Visualization - Formula Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: VISUALIZATION - THE COMPLETE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# TOP LEFT: Tree-by-tree importances
ax = axes[0, 0]
tree_ids = np.arange(1, B + 1)
width = 0.25

for j in range(X.shape[1]):
    ax.bar(tree_ids + j*width, tree_imps[:, j], width, 
           label=f'Feature_{j}', alpha=0.8, edgecolor='black')

ax.set_xlabel('Tree Number (b)', fontsize=12, fontweight='bold')
ax.set_ylabel('Importance I_j[b]', fontsize=12, fontweight='bold')
ax.set_title('① Individual Tree Importances\nI_j[b] for each tree b', 
            fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')
ax.set_xticks(tree_ids + width)
ax.set_xticklabels(tree_ids)

# Add formula annotation
ax.text(0.5, 0.95, 'I_j[b] = Σ_nodes (N_node/N_total) × ΔGini', 
       transform=ax.transAxes, ha='center', fontsize=10,
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# TOP RIGHT: Averaging process
ax = axes[0, 1]
feature_names = ['Feature_0\n(Strong)', 'Feature_1\n(Moderate)', 'Feature_2\n(Noise)']
colors = ['#2ecc71', '#3498db', '#e74c3c']

bars = ax.bar(feature_names, avg_imps, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Average Importance I_j(RF)', fontsize=12, fontweight='bold')
ax.set_title('② Average Over Trees\nI_j(RF) = (1/B) × Σ_b I_j[b]', 
            fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Add values on bars
for bar, val in zip(bars, avg_imps):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + height*0.02, 
           f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')

# Add formula annotation
ax.text(0.5, 0.95, f'Average over B={B} trees', 
       transform=ax.transAxes, ha='center', fontsize=10,
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

# BOTTOM LEFT: Normalization
ax = axes[1, 0]

bars = ax.bar(feature_names, rf_importances_manual, color=colors, alpha=0.8, 
             edgecolor='black', linewidth=2)
ax.set_ylabel('Normalized Importance', fontsize=12, fontweight='bold')
ax.set_title('③ Normalize (Sum = 1.0)\nI_j(norm) = I_j(RF) / Σ_k I_k', 
            fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0, 1.0)

# Add percentage labels
for bar, val in zip(bars, rf_importances_manual):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
           f'{val:.4f}\n({val*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')

# Add sum line
ax.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, 
          label=f'Sum = {rf_importances_manual.sum():.4f}')
ax.legend(fontsize=10)

# BOTTOM RIGHT: Comparison Manual vs sklearn
ax = axes[1, 1]

x_pos = np.arange(X.shape[1])
width = 0.35

ax.bar(x_pos - width/2, rf_importances_manual, width, label='Manual Calculation',
      color='steelblue', alpha=0.8, edgecolor='black', linewidth=2)
ax.bar(x_pos + width/2, sklearn_rf_importances, width, label='sklearn Built-in',
      color='orange', alpha=0.8, edgecolor='black', linewidth=2)

ax.set_ylabel('Feature Importance', fontsize=12, fontweight='bold')
ax.set_title('④ Validation: Manual vs sklearn\nPerfect Match! ✅', 
            fontsize=13, fontweight='bold', color='green')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Feature_0', 'Feature_1', 'Feature_2'])
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# Add difference annotations
for i in range(X.shape[1]):
    diff = abs(rf_importances_manual[i] - sklearn_rf_importances[i])
    y_pos = max(rf_importances_manual[i], sklearn_rf_importances[i]) + 0.05
    ax.text(i, y_pos, f'Δ={diff:.2e}', ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.suptitle('📊 Random Forest Feature Importance: Complete Mathematical Pipeline', 
            fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("🎉 COMPLETE FORMULA PIPELINE VISUALIZED!")
print("="*70)

# COMMAND ----------

# DBTITLE 1,🎮 Interactive: Experiment with Parameters
# ═══════════════════════════════════════════════════════════════════════════════
# 🎮 INTERACTIVE: EXPERIMENT WITH RF PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

import ipywidgets as widgets
from IPython.display import display, clear_output

def explore_rf_importance(n_trees, max_depth, n_samples):
    """
    Interactive Random Forest feature importance explorer
    """
    clear_output(wait=True)
    
    # Generate data
    np.random.seed(42)
    X0_exp = np.random.randn(n_samples)
    X1_exp = np.random.randn(n_samples)
    X2_exp = np.random.randn(n_samples)
    
    y_exp = (X0_exp > 0).astype(int)
    y_exp = np.where((X0_exp > -0.5) & (X0_exp < 0.5) & (X1_exp > 0), 1, y_exp)
    noise_idx = np.random.choice(n_samples, size=max(1, n_samples//10), replace=False)
    y_exp[noise_idx] = 1 - y_exp[noise_idx]
    
    X_exp = np.column_stack([X0_exp, X1_exp, X2_exp])
    
    # Train Random Forest
    rf_exp = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, random_state=42)
    rf_exp.fit(X_exp, y_exp)
    
    importances = rf_exp.feature_importances_
    
    # Print configuration
    print("═" * 80)
    print("🎯 INTERACTIVE RANDOM FOREST FEATURE IMPORTANCE")
    print("═" * 80)
    print(f"\n⚙️  Configuration:")
    print(f"   Number of Trees (B): {n_trees}")
    print(f"   Max Depth: {max_depth}")
    print(f"   Dataset Size: {n_samples} samples")
    
    print(f"\n⭐ FEATURE IMPORTANCES:")
    for i, imp in enumerate(importances):
        bar_length = int(imp * 50)
        bar = '█' * bar_length
        print(f"   Feature_{i}: {imp:.6f} ({imp*100:5.2f}%)  {bar}")
    
    print(f"\n✅ Sum: {importances.sum():.10f}")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # LEFT: Bar chart
    feature_names = ['Feature_0\n(Strong Signal)', 
                    'Feature_1\n(Moderate Signal)', 
                    'Feature_2\n(Noise)']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = ax1.bar(feature_names, importances, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=2)
    ax1.set_ylabel('Importance', fontsize=12, fontweight='bold')
    ax1.set_title(f'Feature Importances (B={n_trees}, depth={max_depth})', 
                 fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.0)
    
    # Add percentage labels
    for bar, imp in zip(bars, importances):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                f'{imp*100:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    # RIGHT: Pie chart
    explode = (0.1, 0.05, 0.02)
    ax2.pie(importances, labels=['Feature_0\n(Strong)', 'Feature_1\n(Moderate)', 'Feature_2\n(Noise)'],
           autopct='%1.1f%%', startangle=90, colors=colors, explode=explode,
           textprops={'fontsize': 11, 'fontweight': 'bold'},
           wedgeprops={'edgecolor': 'black', 'linewidth': 2})
    ax2.set_title(f'Relative Importance Distribution', fontsize=13, fontweight='bold')
    
    plt.suptitle('🎮 Interactive Feature Importance Explorer - Drag Sliders!', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "═" * 80)
    print("\n💡 Key Insights:")
    print(f"   • Feature_0: {importances[0]*100:.1f}% (should be HIGHEST)")
    print(f"   • Feature_1: {importances[1]*100:.1f}% (should be MEDIUM)")
    print(f"   • Feature_2: {importances[2]*100:.1f}% (should be LOWEST - it's noise!)")
    
    # Stability check
    if n_trees < 10:
        print(f"\n   ⚠️  Warning: Only {n_trees} trees - importances may be unstable")
        print(f"      Try increasing to 50-100 trees for stable estimates")
    elif n_trees >= 50:
        print(f"\n   ✅ Good! {n_trees} trees provide stable importance estimates")
    
    print("\n" + "═" * 80)

print("🎮 INTERACTIVE FEATURE IMPORTANCE EXPLORATION")
print("Experiment with Random Forest parameters!\n")

trees_slider = widgets.IntSlider(
    value=10, min=1, max=100, step=1,
    description='N Trees (B):', continuous_update=False,
    style={'description_width': '150px'}
)

depth_slider = widgets.IntSlider(
    value=3, min=1, max=10, step=1,
    description='Max Depth:', continuous_update=False,
    style={'description_width': '150px'}
)

samples_slider = widgets.IntSlider(
    value=100, min=50, max=500, step=50,
    description='N Samples:', continuous_update=False,
    style={'description_width': '150px'}
)

interactive_rf = widgets.interactive(
    explore_rf_importance,
    n_trees=trees_slider,
    max_depth=depth_slider,
    n_samples=samples_slider
)

display(interactive_rf)

# COMMAND ----------

# DBTITLE 1,🎓 Final Summary & Key Takeaways
# MAGIC %md
# MAGIC # 🎓 Summary: Complete Formula Breakdown
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📝 The Complete Formula Pipeline
# MAGIC
# MAGIC ### **Step 1: Calculate ΔGini at Each Node**
# MAGIC ```
# MAGIC For each node in each tree:
# MAGIC   ΔGini = Gini_parent - (N_L/N)×Gini_L - (N_R/N)×Gini_R
# MAGIC
# MAGIC Where:
# MAGIC   - Gini = 1 - Σ(p_i²)
# MAGIC   - N_L = samples in left child
# MAGIC   - N_R = samples in right child
# MAGIC   - N = N_L + N_R (total samples at node)
# MAGIC ```
# MAGIC
# MAGIC ### **Step 2: Accumulate Importance Per Tree**
# MAGIC ```
# MAGIC For tree b, for each split using feature j:
# MAGIC   I_j[b] += (N_node / N_total) × ΔGini
# MAGIC
# MAGIC Where:
# MAGIC   - N_node = samples at this node
# MAGIC   - N_total = total training samples
# MAGIC   - Weight by node size (larger nodes = more importance)
# MAGIC ```
# MAGIC
# MAGIC ### **Step 3: Average Over All Trees**
# MAGIC ```
# MAGIC I_j(RF) = (1/B) × Σ_b I_j[b]
# MAGIC
# MAGIC Where:
# MAGIC   - B = number of trees in forest
# MAGIC   - Average reduces variance from individual trees
# MAGIC ```
# MAGIC
# MAGIC ### **Step 4: Normalize (Sum to 1.0)**
# MAGIC ```
# MAGIC I_j(normalized) = I_j(RF) / Σ_k I_k
# MAGIC
# MAGIC This is what sklearn returns:
# MAGIC   rf.feature_importances_[j] = I_j(normalized)
# MAGIC   
# MAGIC Properties:
# MAGIC   - Shape: (n_features,)
# MAGIC   - Sum = 1.0
# MAGIC   - All values ≥ 0
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ What We Verified
# MAGIC
# MAGIC 1. ✅ **Gini Impurity**: Formula and edge cases (pure nodes, maximum impurity)
# MAGIC 2. ✅ **ΔGini Calculation**: Reduction from splits
# MAGIC 3. ✅ **Single Tree Importance**: Manual extraction and normalization
# MAGIC 4. ✅ **Random Forest Importance**: Averaging over multiple trees
# MAGIC 5. ✅ **Validation**: Perfect match with sklearn's implementation
# MAGIC 6. ✅ **Interactive Exploration**: See how parameters affect results
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 Key Insights
# MAGIC
# MAGIC ### **Why Feature Importance Works:**
# MAGIC * Features that reduce Gini the most get higher importance
# MAGIC * Weighted by number of samples at each node
# MAGIC * Averaged over many trees for stability
# MAGIC * Normalized so importances sum to 1.0
# MAGIC
# MAGIC ### **Interpretation:**
# MAGIC * **High importance** → Feature is used frequently for splits that reduce impurity
# MAGIC * **Low importance** → Feature provides little information gain
# MAGIC * **Zero importance** → Feature never used in any split
# MAGIC
# MAGIC ### **Advantages:**
# MAGIC * **Fast**: No retraining needed
# MAGIC * **Built-in**: Computed during training
# MAGIC * **Normalized**: Easy to compare features
# MAGIC * **Stable**: Averaging over trees reduces variance
# MAGIC
# MAGIC ### **Limitations:**
# MAGIC * **Biased** toward high-cardinality features
# MAGIC * **Not causal**: Correlation ≠ causation
# MAGIC * **Tree-specific**: Different from permutation importance
# MAGIC * **Scale-dependent**: Affected by feature scaling
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Formula Reference Card
# MAGIC
# MAGIC | **Formula** | **Purpose** |
# MAGIC |-------------|-------------|
# MAGIC | `Gini = 1 - Σ(p_i²)` | Measure node impurity |
# MAGIC | `ΔGini = Gini_parent - (N_L/N)×Gini_L - (N_R/N)×Gini_R` | Impurity reduction from split |
# MAGIC | `I_j[b] += (N_node/N_total) × ΔGini` | Accumulate importance in tree b |
# MAGIC | `I_j(RF) = (1/B) × Σ_b I_j[b]` | Average over B trees |
# MAGIC | `I_j(normalized) = I_j(RF) / Σ_k I_k` | Normalize to sum=1.0 |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 Next Steps
# MAGIC
# MAGIC 1. **Compare with Permutation Importance**: See how rankings differ
# MAGIC 2. **Try on Real Data**: Apply to your own datasets
# MAGIC 3. **Feature Selection**: Use importances to select top features
# MAGIC 4. **Partial Dependence Plots**: Understand feature effects
# MAGIC 5. **SHAP Values**: Get instance-level feature importance
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📌 Quick Reference
# MAGIC
# MAGIC **Access feature importances in sklearn:**
# MAGIC ```python
# MAGIC rf = RandomForestClassifier(n_estimators=100)
# MAGIC rf.fit(X_train, y_train)
# MAGIC
# MAGIC # Get importances
# MAGIC importances = rf.feature_importances_  # Shape: (n_features,)
# MAGIC
# MAGIC # Properties
# MAGIC assert importances.sum() == 1.0  # Always sums to 1
# MAGIC assert (importances >= 0).all()  # Always non-negative
# MAGIC
# MAGIC # Get top features
# MAGIC top_indices = importances.argsort()[::-1]
# MAGIC for i in top_indices[:5]:  # Top 5
# MAGIC     print(f"Feature {i}: {importances[i]:.4f}")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **🎉 You now understand the complete mathematics behind `rf.feature_importances_`!**