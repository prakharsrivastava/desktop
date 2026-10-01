# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # Fixing Underfitting with Polynomial Features
# MAGIC
# MAGIC ## The Problem: High Bias (Underfitting)
# MAGIC
# MAGIC When a model is **too simple** to capture the underlying patterns in the data:
# MAGIC * ❌ Both training AND test accuracies are LOW
# MAGIC * ❌ Model can't learn complex relationships
# MAGIC * ❌ Adding more data won't help!
# MAGIC
# MAGIC ## The Solution: Polynomial Features
# MAGIC
# MAGIC Transform features to capture **non-linear relationships** and **interactions**:
# MAGIC * Original features: `[x₁, x₂]`
# MAGIC * Polynomial features (degree 2): `[1, x₁, x₂, x₁², x₁·x₂, x₂²]`
# MAGIC * This increases model complexity without changing the algorithm!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC This notebook demonstrates how polynomial features fix underfitting in a classification task.

# COMMAND ----------

# DBTITLE 1,Setup and Data Generation
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

print("✅ Libraries imported successfully!")

# Generate classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\n📊 Dataset created:")
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Total features: {X_train.shape[1]}")
print(f"  Classes: {len(np.unique(y_train))}")

# COMMAND ----------

# DBTITLE 1,Problem: Simple Model (Underfitting)
# ════════════════════════════════════════════════════════════════════════
# PROBLEM: Model Too Simple (Using Only 2 of 20 Features)
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PROBLEM: HIGH BIAS (UNDERFITTING)")
print("=" * 70)

# Use only 2 features (too simple for 20-feature problem)
X_train_simple = X_train[:, :2]
X_test_simple = X_test[:, :2]

print(f"\n🔧 Model Configuration:")
print(f"  Algorithm: Logistic Regression")
print(f"  Features used: 2 out of {X_train.shape[1]}")
print(f"  ⚠️ This is intentionally too simple!")

# Train simple model
model_simple = LogisticRegression(max_iter=1000, random_state=42)
model_simple.fit(X_train_simple, y_train)

# Predictions
y_pred_train = model_simple.predict(X_train_simple)
y_pred_test = model_simple.predict(X_test_simple)

# Accuracies
train_acc = accuracy_score(y_train, y_pred_train) * 100
test_acc = accuracy_score(y_test, y_pred_test) * 100
gap = train_acc - test_acc

print(f"\n📊 Results:")
print(f"  Training Accuracy: {train_acc:.1f}%")
print(f"  Test Accuracy:     {test_acc:.1f}%")
print(f"  Gap:               {gap:.1f}%")

print(f"\n🔍 Diagnosis:")
if train_acc < 70 and test_acc < 70:
    print("  ❌ HIGH BIAS detected!")
    print("  📌 Both accuracies are LOW")
    print("  📌 Model is too simple to learn patterns")
    print("  📌 Gap is small because model fails equally on both sets")

print("\n" + "=" * 70)

# COMMAND ----------

# DBTITLE 1,Solution: Add Polynomial Features
# ════════════════════════════════════════════════════════════════════════
# SOLUTION: Add Polynomial Features to Increase Complexity
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("SOLUTION: ADD POLYNOMIAL FEATURES")
print("=" * 70)

# Create polynomial features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=True)
X_train_poly = poly.fit_transform(X_train_simple)
X_test_poly = poly.transform(X_test_simple)

print(f"\n🔧 Feature Engineering:")
print(f"  Original features: {X_train_simple.shape[1]}")
print(f"  After polynomial expansion: {X_train_poly.shape[1]}")
print(f"\n  New features include:")
print(f"    • Bias term: 1")
print(f"    • Linear terms: x₁, x₂")
print(f"    • Quadratic terms: x₁², x₂²")
print(f"    • Interaction term: x₁·x₂")

# Train model with polynomial features
model_poly = LogisticRegression(max_iter=1000, random_state=42)
model_poly.fit(X_train_poly, y_train)

# Predictions
y_pred_train_poly = model_poly.predict(X_train_poly)
y_pred_test_poly = model_poly.predict(X_test_poly)

# Accuracies
train_acc_poly = accuracy_score(y_train, y_pred_train_poly) * 100
test_acc_poly = accuracy_score(y_test, y_pred_test_poly) * 100
gap_poly = train_acc_poly - test_acc_poly

print(f"\n📊 Results (With Polynomial Features):")
print(f"  Training Accuracy: {train_acc_poly:.1f}%")
print(f"  Test Accuracy:     {test_acc_poly:.1f}%")
print(f"  Gap:               {gap_poly:.1f}%")

print(f"\n✅ Improvement:")
print(f"  Train: {train_acc:.1f}% → {train_acc_poly:.1f}% (+{train_acc_poly - train_acc:.1f}%)")
print(f"  Test:  {test_acc:.1f}% → {test_acc_poly:.1f}% (+{test_acc_poly - test_acc:.1f}%)")

if train_acc_poly > 70 and test_acc_poly > 70:
    print("\n🎉 SUCCESS! Polynomial features fixed the underfitting!")
    print("   Both accuracies are now acceptable.")

print("\n" + "=" * 70)

# COMMAND ----------

# DBTITLE 1,Visual Comparison: Before vs After
# ════════════════════════════════════════════════════════════════════════
# VISUALIZATION: Before vs After
# ════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# ─────────────────────────────────────────────────────────────────────────
# LEFT: Simple model (underfitting)
# ─────────────────────────────────────────────────────────────────────────
ax = axes[0]
models = ['Train', 'Test']
scores_simple = [train_acc, test_acc]
colors = ['steelblue', 'coral']

bars = ax.bar(models, scores_simple, color=colors, edgecolor='black', linewidth=2)
ax.axhline(70, color='red', linestyle='--', linewidth=2, label='Target (70%)')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('BEFORE: Simple Model (2 features)\n❌ HIGH BIAS (Underfitting)', 
            fontsize=13, fontweight='bold', color='darkred')
ax.set_ylim(0, 100)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 3,
            f'{height:.1f}%',
            ha='center', fontsize=12, fontweight='bold')

# Gap label
ax.text(0.5, 15, f'Gap: {gap:.1f}%', ha='center', fontsize=11, 
       bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', linewidth=2),
       fontweight='bold')

# ─────────────────────────────────────────────────────────────────────────
# RIGHT: With polynomial features (fixed)
# ─────────────────────────────────────────────────────────────────────────
ax = axes[1]
scores_poly = [train_acc_poly, test_acc_poly]

bars = ax.bar(models, scores_poly, color=colors, edgecolor='black', linewidth=2)
ax.axhline(70, color='green', linestyle='--', linewidth=2, label='Target (70%)')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title(f'AFTER: Polynomial Features ({X_train_poly.shape[1]} features)\n✅ FIXED!', 
            fontsize=13, fontweight='bold', color='darkgreen')
ax.set_ylim(0, 100)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 3,
            f'{height:.1f}%',
            ha='center', fontsize=12, fontweight='bold')

# Gap label
ax.text(0.5, 15, f'Gap: {gap_poly:.1f}%', ha='center', fontsize=11,
       bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', linewidth=2),
       fontweight='bold')

plt.suptitle('Fixing High Bias (Underfitting) with Polynomial Features', 
            fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)

# COMMAND ----------

# DBTITLE 1,Interactive Experimentation
# MAGIC %md
# MAGIC ## 🎮 Interactive Experimentation
# MAGIC
# MAGIC Now you can **interact** with the model! Use the sliders below to:
# MAGIC * Adjust the number of features used from the original dataset
# MAGIC * Change the polynomial degree
# MAGIC * See how it affects train/test accuracy in real-time
# MAGIC
# MAGIC This helps you understand:
# MAGIC * How increasing features reduces underfitting
# MAGIC * How polynomial degree affects complexity
# MAGIC * The trade-off between complexity and performance

# COMMAND ----------

# DBTITLE 1,Interactive Widget - Drag to Explore
# ════════════════════════════════════════════════════════════════════════
# INTERACTIVE VISUALIZATION - Drag sliders to see changes!
# ════════════════════════════════════════════════════════════════════════

import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings
warnings.filterwarnings('ignore')

def train_and_visualize(n_features, poly_degree):
    """
    Train models with specified parameters and visualize results
    """
    clear_output(wait=True)
    
    # Select features
    X_train_subset = X_train[:, :n_features]
    X_test_subset = X_test[:, :n_features]
    
    # Train simple model (no polynomial)
    model_simple = LogisticRegression(max_iter=1000, random_state=42)
    model_simple.fit(X_train_subset, y_train)
    
    y_pred_train_simple = model_simple.predict(X_train_subset)
    y_pred_test_simple = model_simple.predict(X_test_subset)
    
    train_acc_simple = accuracy_score(y_train, y_pred_train_simple) * 100
    test_acc_simple = accuracy_score(y_test, y_pred_test_simple) * 100
    gap_simple = train_acc_simple - test_acc_simple
    
    # Train with polynomial features
    poly = PolynomialFeatures(degree=poly_degree, include_bias=True)
    X_train_poly = poly.fit_transform(X_train_subset)
    X_test_poly = poly.transform(X_test_subset)
    
    model_poly = LogisticRegression(max_iter=2000, random_state=42)
    model_poly.fit(X_train_poly, y_train)
    
    y_pred_train_poly = model_poly.predict(X_train_poly)
    y_pred_test_poly = model_poly.predict(X_test_poly)
    
    train_acc_poly = accuracy_score(y_train, y_pred_train_poly) * 100
    test_acc_poly = accuracy_score(y_test, y_pred_test_poly) * 100
    gap_poly = train_acc_poly - test_acc_poly
    
    # Print results
    print("═" * 80)
    print(f"CONFIGURATION: {n_features} features | Polynomial degree: {poly_degree}")
    print("═" * 80)
    print(f"\n📊 SIMPLE MODEL ({n_features} features):")
    print(f"   Train: {train_acc_simple:.1f}%  |  Test: {test_acc_simple:.1f}%  |  Gap: {gap_simple:.1f}%")
    
    print(f"\n🔬 WITH POLYNOMIAL FEATURES ({X_train_poly.shape[1]} features):")
    print(f"   Train: {train_acc_poly:.1f}%  |  Test: {test_acc_poly:.1f}%  |  Gap: {gap_poly:.1f}%")
    
    improvement_train = train_acc_poly - train_acc_simple
    improvement_test = test_acc_poly - test_acc_simple
    print(f"\n✅ IMPROVEMENT: Train +{improvement_train:.1f}% | Test +{improvement_test:.1f}%")
    
    # Diagnosis
    print(f"\n🔍 DIAGNOSIS:")
    if train_acc_poly < 70 and test_acc_poly < 70:
        print("   ⚠️ Still underfitting - try more features or higher degree")
    elif train_acc_poly > 90 and test_acc_poly < 70:
        print("   ⚠️ Overfitting detected - gap too large!")
    elif train_acc_poly > 70 and test_acc_poly > 70 and abs(gap_poly) < 10:
        print("   ✅ Good fit! Both accuracies acceptable with small gap")
    else:
        print("   📊 Moderate fit - acceptable but could be improved")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # LEFT: Simple model
    ax = axes[0]
    models = ['Train', 'Test']
    scores_simple = [train_acc_simple, test_acc_simple]
    colors = ['steelblue', 'coral']
    
    bars = ax.bar(models, scores_simple, color=colors, edgecolor='black', linewidth=2)
    ax.axhline(70, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target (70%)')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Simple Model ({n_features} features)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    ax.text(0.5, 10, f'Gap: {gap_simple:.1f}%', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5))
    
    # RIGHT: With polynomial
    ax = axes[1]
    scores_poly = [train_acc_poly, test_acc_poly]
    
    bars = ax.bar(models, scores_poly, color=colors, edgecolor='black', linewidth=2)
    ax.axhline(70, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target (70%)')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Polynomial (degree {poly_degree}, {X_train_poly.shape[1]} features)', 
                fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    gap_color = 'green' if abs(gap_poly) < 10 else 'orange' if abs(gap_poly) < 20 else 'red'
    ax.text(0.5, 10, f'Gap: {gap_poly:.1f}%', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor=gap_color, linewidth=1.5))
    
    plt.suptitle('🎮 Interactive Model Comparison - Drag Sliders to Experiment!', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "═" * 80)

# Create interactive sliders
print("🎮 INTERACTIVE EXPERIMENTATION")
print("Drag the sliders below to see how parameters affect model performance!\n")

feature_slider = widgets.IntSlider(
    value=2,
    min=1,
    max=20,
    step=1,
    description='Features:',
    continuous_update=False,
    style={'description_width': '120px'}
)

degree_slider = widgets.IntSlider(
    value=2,
    min=1,
    max=4,
    step=1,
    description='Poly Degree:',
    continuous_update=False,
    style={'description_width': '120px'}
)

# Display interactive widget
interactive_plot = widgets.interactive(
    train_and_visualize,
    n_features=feature_slider,
    poly_degree=degree_slider
)

display(interactive_plot)

# COMMAND ----------

# DBTITLE 1,Key Takeaways
# MAGIC %md
# MAGIC ## 📝 Key Takeaways
# MAGIC
# MAGIC ### What is High Bias (Underfitting)?
# MAGIC * **Symptom**: Both training AND test accuracies are LOW
# MAGIC * **Cause**: Model is too simple to capture patterns in the data
# MAGIC * **Indicator**: Small gap between train and test (both fail equally)
# MAGIC
# MAGIC ### Why Polynomial Features Help
# MAGIC * **Increases model complexity** without changing the algorithm
# MAGIC * **Captures non-linear relationships**: `x²` terms model curved patterns
# MAGIC * **Captures feature interactions**: `x₁·x₂` terms model combined effects
# MAGIC * **Example**: `[x₁, x₂]` → `[1, x₁, x₂, x₁², x₁·x₂, x₂²]` (2 → 6 features)
# MAGIC
# MAGIC ### When to Use Polynomial Features
# MAGIC ✅ **Use when**:
# MAGIC * Both train and test accuracies are low
# MAGIC * You suspect non-linear relationships
# MAGIC * Simple linear model fails
# MAGIC
# MAGIC ❌ **Don't use when**:
# MAGIC * Already overfitting (high train, low test)
# MAGIC * Dataset is very small (risk of overfitting)
# MAGIC * Features are already complex/high-dimensional
# MAGIC
# MAGIC ### Alternative Solutions for Underfitting
# MAGIC 1. **Use more features** from the original dataset
# MAGIC 2. **Try a more complex model** (e.g., Random Forest, Neural Network)
# MAGIC 3. **Add domain-specific engineered features**
# MAGIC 4. **Remove regularization** if applied
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Interview Tip 💡
# MAGIC When asked "Train 55%, Test 53%" → Immediately recognize **HIGH BIAS**. The solution is NOT more data, but **increased model complexity** through:
# MAGIC * Polynomial features
# MAGIC * More features
# MAGIC * More complex algorithm
# MAGIC * Reduced regularization