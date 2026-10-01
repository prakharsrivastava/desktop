# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # 🎯 Interview Practice: Working Code Examples
# MAGIC
# MAGIC ## Purpose
# MAGIC
# MAGIC This notebook contains **executable code** demonstrating every concept from the interview feedback.
# MAGIC
# MAGIC ### What You'll Build
# MAGIC
# MAGIC 1. ✅ **Scenario 1:** High Bias (Underfitting) - Both accuracies low
# MAGIC 2. ✅ **Scenario 2:** High Variance (Overfitting) - Train 99%, Test 72%
# MAGIC 3. ✅ **Scenario 3:** Good Fit - Small gap, both high
# MAGIC 4. 🛠️ **All 8 Generalization Techniques** - Working examples
# MAGIC 5. 📈 **Learning Curves** - Visual diagnosis code
# MAGIC 6. 🤔 **Decision Framework** - When to collect data vs simplify
# MAGIC
# MAGIC ### How to Use
# MAGIC
# MAGIC 1. Run all cells in order
# MAGIC 2. See real results matching interview scenarios
# MAGIC 3. Modify parameters to experiment
# MAGIC 4. Use this as reference during interviews
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Goal:** By the end, you can recreate these examples from memory in a live coding interview.

# COMMAND ----------

# Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

print("✓ Setup complete")

# COMMAND ----------



# Generate realistic dataset for classification
def generate_classification_data(n_samples=1000, n_features=20, noise=0.1):
    """
    Generate classification dataset that can exhibit bias/variance issues
    """
    from sklearn.datasets import make_classification
    #✅ n_informative=15 matlab sirf 15 features actually class decide karte hain. Baaki 5 features random garbage hain — model ko inhe ignore karna seekhna hota hai.
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=3, #derived from existing coloum in linear way
        n_repeated=0,
        n_classes=2,
        flip_y=noise,
        random_state=42
    )
    
    return train_test_split(X, y, test_size=0.3, random_state=42)

# COMMAND ----------

X_train, X_test, y_train, y_test = generate_classification_data()

# COMMAND ----------

X_train.shape

# COMMAND ----------

y_train.shape

# COMMAND ----------

# DBTITLE 1,Setup and Data Generation




print(f"\nDataset created:")
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Features: {X_train.shape[1]}")
print(f"  Classes: {len(np.unique(y_train))}")

# COMMAND ----------

# ===================================================================
# SCENARIO 1: HIGH BIAS (UNDERFITTING)
# ===================================================================
# Model too simple - can't learn patterns
# Expected: Train LOW, Test LOW, Small Gap

print("=" * 70)
print("SCENARIO 1: HIGH BIAS (UNDERFITTING)")
print("=" * 70)

# COMMAND ----------

X_train[:,[0]]

# COMMAND ----------


# Use only 2 features (too simple for 20-feature problem)
X_train_simple = X_train[:, :2]  # Only first 2 features
X_test_simple = X_test[:, :2]

# COMMAND ----------

# DBTITLE 1,🎮 Interactive: Train Logistic Regression
# ═══════════════════════════════════════════════════════════════════════════════
# 🎮 INTERACTIVE LOGISTIC REGRESSION - Experiment with parameters!
# ═══════════════════════════════════════════════════════════════════════════════

import ipywidgets as widgets
from IPython.display import display, clear_output

def train_logistic_regression(n_features, max_iter, C_param):
    """
    Interactive logistic regression training
    """
    clear_output(wait=True)
    
    # Select features
    X_train_subset = X_train[:, :n_features]
    X_test_subset = X_test[:, :n_features]
    print(C_param)
    # Train model with parameters
    model = LogisticRegression(max_iter=max_iter, C=C_param, random_state=42)
    model.fit(X_train_subset, y_train)
    
    # Predictions
    y_pred_train_local = model.predict(X_train_subset)
    y_pred_test_local = model.predict(X_test_subset)
    
    # Accuracies
    train_acc = accuracy_score(y_train, y_pred_train_local) * 100
    test_acc = accuracy_score(y_test, y_pred_test_local) * 100
    gap = train_acc - test_acc
    
    # Print results
    print("═" * 80)
    print("🎯 LOGISTIC REGRESSION TRAINING")
    print("═" * 80)
    print(f"\n⚙️  Configuration:")
    print(f"   Features Used: {n_features} out of {X_train.shape[1]}")
    print(f"   Max Iterations: {max_iter}")
    print(f"   Regularization (C): {C_param} {'(less penalty)' if C_param > 1 else '(more penalty)'}")
    
    print(f"\n📊 Performance:")
    print(f"   Training Accuracy: {train_acc:.1f}%")
    print(f"   Test Accuracy:     {test_acc:.1f}%")
    print(f"   Gap:               {gap:.1f}%")
    
    # Diagnosis
    print(f"\n🔍 DIAGNOSIS:")
    if train_acc < 70 and test_acc < 70:
        print("   🔴 HIGH BIAS (Underfitting) - Model too simple!")
        print(f"   📌 Using only {n_features} features is not enough")
        print(f"   📌 Model can't capture complex patterns")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if n_features < 10:
            print(f"   → Increase features from {n_features} to 10+")
        if C_param < 1:
            print(f"   → Reduce regularization (increase C from {C_param} to 1.0)")
        print("   → Consider using polynomial features")
        print("   → Try a more complex model (Random Forest, Neural Network)")
        
    elif train_acc > 90 and gap > 20:
        print("   🟠 HIGH VARIANCE (Overfitting) - Too complex!")
        print("   💡 Reduce C parameter (more regularization)")
    else:
        print("   🟢 Reasonable performance!")
        if train_acc < 85:
            print("   💡 Could improve with more features or complexity")
    
    # Check convergence
    if not model.n_iter_[0] >= max_iter * 0.9:
        print(f"\n✅ Model converged in {model.n_iter_[0]} iterations")
    else:
        print(f"\n⚠️  Model may not have fully converged (used {model.n_iter_[0]} iterations)")
        print("   💡 Try increasing max_iter")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # LEFT: Performance bars
    ax = axes[0]
    bars = ax.bar(['Train', 'Test'], [train_acc, test_acc], 
                  color=['steelblue', 'coral'], edgecolor='black', linewidth=2)
    ax.axhline(70, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Min Target (70%)')
    ax.axhline(85, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Good Target (85%)')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Performance ({n_features} features)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    gap_color = 'green' if gap < 8 else 'orange' if gap < 15 else 'red'
    ax.text(0.5, 10, f'Gap: {gap:.1f}%', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor=gap_color, linewidth=2),
           fontweight='bold')
    
    # MIDDLE: Feature importance (coefficients)
    ax = axes[1]
    coef = np.abs(model.coef_[0])
    feature_names = [f'F{i+1}' for i in range(n_features)]
    
    bars = ax.barh(feature_names, coef, color='teal', edgecolor='black', linewidth=1)
    ax.set_xlabel('Coefficient Magnitude', fontsize=11, fontweight='bold')
    ax.set_title('Feature Importance (|Coefficients|)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    # Highlight most important feature
    max_idx = np.argmax(coef)
    bars[max_idx].set_color('orange')
    bars[max_idx].set_edgecolor('red')
    bars[max_idx].set_linewidth(2)
    
    # RIGHT: Settings summary
    ax = axes[2]
    ax.axis('off')
    
    # Create a visual summary
    summary_text = f"""
🎯 CONFIGURATION SUMMARY

📊 Model: Logistic Regression
   • Features: {n_features}/{X_train.shape[1]}
   • Max Iterations: {max_iter}
   • Regularization C: {C_param}

📈 RESULTS
   • Train Acc: {train_acc:.1f}%
   • Test Acc: {test_acc:.1f}%
   • Gap: {gap:.1f}%
   • Converged: {model.n_iter_[0]} iter

"""
    
    if train_acc < 70 and test_acc < 70:
        summary_text += """🔴 STATUS: UNDERFITTING
   Too few features!
   
💡 NEXT STEPS:
   1. Increase features slider
   2. Reduce regularization (↑ C)
   3. Try polynomial features
"""
    elif train_acc > 90 and gap > 20:
        summary_text += """🟠 STATUS: OVERFITTING
   Model too flexible!
   
💡 NEXT STEPS:
   1. Increase regularization (↓ C)
   2. Reduce features
"""
    else:
        summary_text += f"""🟢 STATUS: {'GOOD' if train_acc > 80 else 'MODERATE'}
   Reasonable performance!
   
💡 OPTIMIZATION:
   Fine-tune for marginal gains
"""
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1),
           family='monospace')
    
    plt.suptitle('🎮 Interactive Logistic Regression - Drag Sliders to Experiment!', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    # Store predictions globally for next cell
    global y_pred_train, y_pred_test, model_bias
    y_pred_train = y_pred_train_local
    y_pred_test = y_pred_test_local
    model_bias = model
    
    print("\n" + "═" * 80)

print("🎮 INTERACTIVE LOGISTIC REGRESSION TRAINING")
print("Experiment with features, iterations, and regularization!\n")

features_slider = widgets.IntSlider(
    value=2, min=1, max=20, step=1,
    description='Features:', continuous_update=False,
    style={'description_width': '150px'}
)

iter_slider = widgets.IntSlider(
    value=100, min=10, max=1000, step=50,
    description='Max Iterations:', continuous_update=False,
    style={'description_width': '150px'}
)

C_slider = widgets.FloatSlider(
    value=1.0, min=0.01, max=10.0, step=0.1,
    description='Regularization C:', continuous_update=False,
    style={'description_width': '150px'}
)

interactive_lr = widgets.interactive(
    train_logistic_regression,
    n_features=features_slider,
    max_iter=iter_slider,
    C_param=C_slider
)

display(interactive_lr)

# COMMAND ----------

# DBTITLE 1,Scenario 1 - High Bias (Underfitting)




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
    print("  ✅ HIGH BIAS (Underfitting) detected!")
    print("  📌 Both accuracies are LOW")
    print("  📌 Model is too simple to learn patterns")
    print("\n💡 Solution:")
    print("  → Use more features (currently using only 2 of 20)")
    print("  → Try more complex model")
    print("  → Add polynomial features")
    print("  → ❌ DON'T collect more data (won't help!)")

# Visualize
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
models = ['Train', 'Test']
scores = [train_acc, test_acc]
colors = ['steelblue', 'coral']

bars = ax.bar(models, scores, color=colors, edgecolor='black', linewidth=1.5)
ax.axhline(70, color='red', linestyle='--', linewidth=2, label='Acceptable Threshold (70%)')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('HIGH BIAS: Both Scores Low (Model Too Simple)', fontsize=13, fontweight='bold', color='darkred')
ax.set_ylim(0, 100)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%',
            ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)

# COMMAND ----------

# DBTITLE 1,🎮 Interactive: Experiment with Underfitting
# ═══════════════════════════════════════════════════════════════════════════════
# 🎮 INTERACTIVE WIDGET - Drag sliders to experiment with underfitting!
# ═══════════════════════════════════════════════════════════════════════════════

import ipywidgets as widgets
from IPython.display import display, clear_output

def explore_bias(n_features, poly_degree):
    """
    Interactive exploration of high bias (underfitting)
    """
    clear_output(wait=True)
    
    # Select features
    X_train_subset = X_train[:, :n_features]
    X_test_subset = X_test[:, :n_features]
    
    # Train simple model
    model_simple = LogisticRegression(max_iter=1000, random_state=42)
    model_simple.fit(X_train_subset, y_train)
    
    train_simple = accuracy_score(y_train, model_simple.predict(X_train_subset)) * 100
    test_simple = accuracy_score(y_test, model_simple.predict(X_test_subset)) * 100
    gap_simple = train_simple - test_simple
    
    # Train with polynomial features
    poly = PolynomialFeatures(degree=poly_degree, include_bias=True)
    X_train_poly = poly.fit_transform(X_train_subset)
    X_test_poly = poly.transform(X_test_subset)
    
    model_poly = LogisticRegression(max_iter=2000, random_state=42)
    model_poly.fit(X_train_poly, y_train)
    
    train_poly = accuracy_score(y_train, model_poly.predict(X_train_poly)) * 100
    test_poly = accuracy_score(y_test, model_poly.predict(X_test_poly)) * 100
    gap_poly = train_poly - test_poly
    
    # Print results
    print("═" * 80)
    print(f"⚙️  CONFIGURATION: {n_features} features | Polynomial degree: {poly_degree}")
    print("═" * 80)
    print(f"\n📊 SIMPLE MODEL ({n_features} original features):")
    print(f"   Train: {train_simple:.1f}%  |  Test: {test_simple:.1f}%  |  Gap: {gap_simple:.1f}%")
    
    print(f"\n🔬 WITH POLYNOMIAL ({X_train_poly.shape[1]} engineered features):")
    print(f"   Train: {train_poly:.1f}%  |  Test: {test_poly:.1f}%  |  Gap: {gap_poly:.1f}%")
    
    improvement = test_poly - test_simple
    print(f"\n✅ TEST IMPROVEMENT: +{improvement:.1f}%")
    
    # Diagnosis
    print(f"\n🔍 DIAGNOSIS:")
    if train_simple < 70 and test_simple < 70:
        print("   🔴 HIGH BIAS (Underfitting) - Model too simple!")
        if train_poly > 70 and test_poly > 70:
            print("   ✅ Polynomial features FIXED it!")
        else:
            print("   ⚠️  Still underfitting - try more features or higher degree")
    elif train_poly > 90 and gap_poly > 20:
        print("   🟠 Now OVERFITTING - polynomial degree too high!")
    else:
        print("   🟢 Reasonable fit - balanced complexity")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # LEFT: Simple model
    ax = axes[0]
    bars = ax.bar(['Train', 'Test'], [train_simple, test_simple], 
                  color=['steelblue', 'coral'], edgecolor='black', linewidth=2)
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
    
    color = 'red' if train_simple < 70 else 'orange' if gap_simple > 10 else 'green'
    ax.text(0.5, 10, f'Gap: {gap_simple:.1f}%', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=2))
    
    # RIGHT: With polynomial
    ax = axes[1]
    bars = ax.bar(['Train', 'Test'], [train_poly, test_poly], 
                  color=['steelblue', 'coral'], edgecolor='black', linewidth=2)
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
    
    color = 'green' if gap_poly < 10 else 'orange' if gap_poly < 20 else 'red'
    ax.text(0.5, 10, f'Gap: {gap_poly:.1f}%', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=2))
    
    plt.suptitle('🎮 Interactive Underfitting Exploration - Drag Sliders!', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "═" * 80)

print("🎮 INTERACTIVE EXPLORATION: HIGH BIAS (UNDERFITTING)")
print("Drag the sliders to see how features and polynomial degree affect underfitting!\n")

feature_slider = widgets.IntSlider(
    value=2, min=1, max=20, step=1,
    description='Features:', continuous_update=False,
    style={'description_width': '120px'}
)

degree_slider = widgets.IntSlider(
    value=1, min=1, max=4, step=1,
    description='Poly Degree:', continuous_update=False,
    style={'description_width': '120px'}
)

interactive_bias = widgets.interactive(
    explore_bias, n_features=feature_slider, poly_degree=degree_slider
)

display(interactive_bias)

# COMMAND ----------

# DBTITLE 1,🎮 Interactive: Experiment with Overfitting
# ═══════════════════════════════════════════════════════════════════════════════
# 🎮 INTERACTIVE WIDGET - Drag sliders to experiment with overfitting!
# ═══════════════════════════════════════════════════════════════════════════════

def explore_variance(max_depth, min_samples_split, min_samples_leaf):
    """
    Interactive exploration of high variance (overfitting)
    """
    clear_output(wait=True)
    
    # Train decision tree with user parameters
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
    test_acc = accuracy_score(y_test, model.predict(X_test)) * 100
    gap = train_acc - test_acc
    
    # Count tree complexity
    n_leaves = model.get_n_leaves()
    tree_depth = model.get_depth()
    print(model)
    # Print results
    print("═" * 80)
    print(f"🌳 DECISION TREE CONFIGURATION")
    print("═" * 80)
    print(f"\n⚙️  Parameters:")
    print(f"   Max Depth: {max_depth}")
    print(f"   Min Samples to Split: {min_samples_split}")
    print(f"   Min Samples per Leaf: {min_samples_leaf}")
    
    print(f"\n📊 Results:")
    print(f"   Training Accuracy: {train_acc:.1f}%")
    print(f"   Test Accuracy:     {test_acc:.1f}%")
    print(f"   Gap:               {gap:.1f}%")
    
    print(f"\n🌲 Tree Complexity:")
    print(f"   Actual Depth: {tree_depth}")
    print(f"   Number of Leaves: {n_leaves}")
    
    # Diagnosis
    print(f"\n🔍 DIAGNOSIS:")
    if train_acc > 95 and gap > 20:
        print("   🔴 HIGH VARIANCE (Overfitting) - Model memorizing training data!")
        print("   💡 Solution: Increase min_samples_split or reduce max_depth")
    elif train_acc < 70 and test_acc < 70:
        print("   🔴 HIGH BIAS (Underfitting) - Model too simple!")
        print("   💡 Solution: Decrease min_samples_split or increase max_depth")
    elif gap < 8 and train_acc > 75:
        print("   🟢 GOOD FIT - Balanced model! Gap < 8% and good accuracy")
    else:
        print("   🟡 MODERATE - Could be improved but acceptable")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # LEFT: Accuracy bars
    ax = axes[0]
    bars = ax.bar(['Train', 'Test'], [train_acc, test_acc], 
                  color=['steelblue', 'coral'], edgecolor='black', linewidth=2)
    ax.axhline(85, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Good (85%)')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Performance (Gap: {gap:.1f}%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    # Gap annotation
    if gap > 5:
        ax.annotate('', xy=(0, train_acc), xytext=(1, test_acc),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        gap_color = 'red' if gap > 20 else 'orange'
        ax.text(0.5, (train_acc + test_acc) / 2, f'{gap:.1f}%',
               ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # RIGHT: Complexity vs Performance
    ax = axes[1]
    
    # Show relationship between complexity and overfitting
    complexity_score = (tree_depth / max_depth) * 100 if max_depth > 0 else 0
    overfitting_score = max(0, gap)
    
    categories = ['Tree\nComplexity', 'Overfitting\nGap']
    values = [complexity_score, overfitting_score]
    colors_bars = ['steelblue', 'red' if gap > 15 else 'orange' if gap > 8 else 'green']
    
    bars = ax.bar(categories, values, color=colors_bars, edgecolor='black', linewidth=2)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Complexity vs Overfitting', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, axis='y')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')
    
    ax.text(0.5, 50, f'Depth: {tree_depth}\nLeaves: {n_leaves}', 
           ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.suptitle('🎮 Interactive Overfitting Exploration - Drag Sliders!', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "═" * 80)

print("\n🎮 INTERACTIVE EXPLORATION: HIGH VARIANCE (OVERFITTING)")
print("Drag the sliders to see how tree parameters affect overfitting!\n")

depth_slider = widgets.IntSlider(
    value=20, min=1, max=30, step=1,
    description='Max Depth:', continuous_update=False,
    style={'description_width': '150px'}
)

split_slider = widgets.IntSlider(
    value=2, min=2, max=50, step=2,
    description='Min Samples Split:', continuous_update=False,
    style={'description_width': '150px'}
)

leaf_slider = widgets.IntSlider(
    value=1, min=1, max=30, step=1,
    description='Min Samples Leaf:', continuous_update=False,
    style={'description_width': '150px'}
)

interactive_variance = widgets.interactive(
    explore_variance, 
    max_depth=depth_slider, 
    min_samples_split=split_slider,
    min_samples_leaf=leaf_slider
)

display(interactive_variance)

# COMMAND ----------

# DBTITLE 1,🎮 Interactive: Learning Curve Explorer
# ═══════════════════════════════════════════════════════════════════════════════
# 🎮 INTERACTIVE LEARNING CURVES - See how data size affects overfitting!
# ═══════════════════════════════════════════════════════════════════════════════

def explore_learning_curve(model_complexity, data_percentage):
    """
    Interactive learning curve exploration
    """
    clear_output(wait=True)
    
    # Calculate data size
    max_size = len(X_train)
    data_size = int(max_size * (data_percentage / 100))
    data_size = max(50, min(data_size, max_size))  # At least 50 samples
    
    # Use subset
    X_train_sub = X_train[:data_size]
    y_train_sub = y_train[:data_size]
    
    # Train model with specified complexity
    model = DecisionTreeClassifier(
        max_depth=model_complexity,
        min_samples_split=2,
        random_state=42
    )
    model.fit(X_train_sub, y_train_sub)
    
    # Evaluate
    train_acc = accuracy_score(y_train_sub, model.predict(X_train_sub)) * 100
    test_acc = accuracy_score(y_test, model.predict(X_test)) * 100
    gap = train_acc - test_acc
    
    # Generate full learning curve for context
    sizes = [50, 100, 200, 400, data_size] if data_size > 400 else [50, 100, 200, data_size]
    sizes = sorted(list(set([s for s in sizes if s <= max_size])))
    
    train_curve = []
    test_curve = []
    
    for size in sizes:
        X_sub = X_train[:size]
        y_sub = y_train[:size]
        
        model_temp = DecisionTreeClassifier(max_depth=model_complexity, random_state=42)
        model_temp.fit(X_sub, y_sub)
        
        train_curve.append(accuracy_score(y_sub, model_temp.predict(X_sub)) * 100)
        test_curve.append(accuracy_score(y_test, model_temp.predict(X_test)) * 100)
    
    # Print results
    print("═" * 80)
    print(f"📈 LEARNING CURVE ANALYSIS")
    print("═" * 80)
    print(f"\n⚙️  Configuration:")
    print(f"   Model Complexity (Max Depth): {model_complexity}")
    print(f"   Training Data: {data_size} samples ({data_percentage}% of {max_size})")
    
    print(f"\n📊 Current Performance:")
    print(f"   Training Accuracy: {train_acc:.1f}%")
    print(f"   Test Accuracy:     {test_acc:.1f}%")
    print(f"   Gap:               {gap:.1f}%")
    
    # Diagnosis
    print(f"\n🔍 DIAGNOSIS:")
    if gap > 20:
        print("   🔴 HIGH VARIANCE - Overfitting detected!")
        if data_percentage < 80:
            print("   💡 Try: Increase data size (move slider right)")
        print("   💡 Or: Reduce model complexity")
    elif train_acc < 70 and test_acc < 70:
        print("   🔴 HIGH BIAS - Underfitting detected!")
        print("   💡 Try: Increase model complexity")
    else:
        print("   🟢 Reasonable performance!")
        if gap > 8:
            print("   💡 Could still improve with more data")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # LEFT: Learning curves
    ax = axes[0]
    ax.plot(sizes, train_curve, 'o-', color='blue', linewidth=2.5, 
           markersize=8, label='Train Accuracy')
    ax.plot(sizes, test_curve, 'o-', color='red', linewidth=2.5, 
           markersize=8, label='Test Accuracy')
    ax.fill_between(sizes, train_curve, test_curve, alpha=0.2, color='orange')
    
    # Highlight current point
    current_idx = sizes.index(data_size)
    ax.plot(data_size, train_curve[current_idx], 'o', color='blue', 
           markersize=15, markeredgecolor='yellow', markeredgewidth=3)
    ax.plot(data_size, test_curve[current_idx], 'o', color='red', 
           markersize=15, markeredgecolor='yellow', markeredgewidth=3)
    
    ax.axhline(85, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Target (85%)')
    ax.set_xlabel('Training Set Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Learning Curve (Complexity={model_complexity})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(50, 105)
    
    # RIGHT: Current performance
    ax = axes[1]
    bars = ax.bar(['Train', 'Test'], [train_acc, test_acc], 
                  color=['steelblue', 'coral'], edgecolor='black', linewidth=2)
    ax.axhline(70, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Min Target')
    ax.axhline(85, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good Target')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Current: {data_size} samples', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    gap_color = 'green' if gap < 8 else 'orange' if gap < 15 else 'red'
    ax.text(0.5, 10, f'Gap: {gap:.1f}%', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor=gap_color, linewidth=2),
           fontweight='bold')
    
    plt.suptitle('🎮 Interactive Learning Curve - See How More Data Helps!', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "═" * 80)

print("\n🎮 INTERACTIVE LEARNING CURVE EXPLORATION")
print("Experiment with model complexity and data size!\n")

complexity_slider = widgets.IntSlider(
    value=20, min=3, max=30, step=1,
    description='Complexity:', continuous_update=False,
    style={'description_width': '120px'}
)

data_slider = widgets.IntSlider(
    value=50, min=10, max=100, step=5,
    description='Data %:', continuous_update=False,
    style={'description_width': '120px'}
)

interactive_learning = widgets.interactive(
    explore_learning_curve,
    model_complexity=complexity_slider,
    data_percentage=data_slider
)

display(interactive_learning)

# COMMAND ----------

# ===================================================================
# SCENARIO 2: HIGH VARIANCE (OVERFITTING)
# ===================================================================
# Model too complex - memorizes training data
# Expected: Train 99%, Test 72%, Large Gap (THIS IS YOUR INTERVIEW QUESTION!)

print("=" * 70)
print("SCENARIO 2: HIGH VARIANCE (OVERFITTING)")
print("=" * 70)
print("\n⚡ THIS IS THE EXACT INTERVIEW SCENARIO!")
print("   Train: 99%, Test: 72%\n")

# Very deep decision tree (overfits easily)
model_variance = DecisionTreeClassifier(
    max_depth=20,  # Very deep - memorizes
    min_samples_split=2,  # Split even tiny groups
    min_samples_leaf=1,  # Allow single-sample leaves
    random_state=42
)
model_variance.fit(X_train, y_train)

# COMMAND ----------

# DBTITLE 1,Scenario 2 - High Variance (Overfitting)


# Predictions
y_pred_train_var = model_variance.predict(X_train)
y_pred_test_var = model_variance.predict(X_test)

# Accuracies
train_acc_var = accuracy_score(y_train, y_pred_train_var) * 100
test_acc_var = accuracy_score(y_test, y_pred_test_var) * 100
gap_var = train_acc_var - test_acc_var

print(f"📊 Results:")
print(f"  Training Accuracy: {train_acc_var:.1f}%")
print(f"  Test Accuracy:     {test_acc_var:.1f}%")
print(f"  Gap:               {gap_var:.1f}%")

print(f"\n🔍 Diagnosis:")
if train_acc_var > 95 and gap_var > 15:
    print("  ✅ HIGH VARIANCE (Overfitting) detected!")
    print("  📌 Training accuracy is VERY HIGH")
    print("  📌 Test accuracy is LOW")
    print("  📌 Large gap = Model MEMORIZES training data")
    print("\n💡 Solutions (in priority order):")
    print("  1. Collect more training data ⭐⭐⭐⭐⭐")
    print("  2. Add regularization (L1/L2) ⭐⭐⭐⭐")
    print("  3. Use cross-validation ⭐⭐⭐⭐")
    print("  4. Early stopping ⭐⭐⭐⭐")
    print("  5. Dropout (if neural network) ⭐⭐⭐⭐⭐")
    print("  6. Feature selection ⭐⭐⭐")
    print("  7. Reduce model complexity ⭐⭐⭐")
    print("  8. Data augmentation ⭐⭐⭐⭐")
    print("\n📈 Next Step:")
    print("  → Plot learning curves to decide:")
    print("     • If test curve rising → collect more data")
    print("     • If test curve flat → simplify model")

# Visualize
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
models = ['Train', 'Test']
scores = [train_acc_var, test_acc_var]
colors = ['steelblue', 'coral']

bars = ax.bar(models, scores, color=colors, edgecolor='black', linewidth=1.5)
ax.axhline(85, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Good Performance (85%)')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title(f'HIGH VARIANCE: Large Gap ({gap_var:.1f}%) = Overfitting', 
            fontsize=13, fontweight='bold', color='darkorange')
ax.set_ylim(0, 105)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%',
            ha='center', fontsize=11, fontweight='bold')

# Add gap annotation
ax.annotate('', xy=(0, train_acc_var), xytext=(1, test_acc_var),
           arrowprops=dict(arrowstyle='<->', color='red', lw=2.5))
ax.text(0.5, (train_acc_var + test_acc_var) / 2,
       f'Gap: {gap_var:.1f}%\n(OVERFITTING!)',
       ha='center', fontsize=10, color='red', fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)

# COMMAND ----------

# DBTITLE 1,Scenario 3 - Good Fit (Balanced)
# ===================================================================
# SCENARIO 3: GOOD FIT (BALANCED)
# ===================================================================
# Model well-tuned - generalizes well
# Expected: Train ~85%, Test ~82%, Small Gap

print("=" * 70)
print("SCENARIO 3: GOOD FIT (BALANCED)")
print("=" * 70)

# Properly tuned Random Forest (good generalization)
model_good = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,  # Limited depth
    min_samples_split=20,  # Require more samples to split
    min_samples_leaf=10,  # Prevent tiny leaves
    random_state=42
)
model_good.fit(X_train, y_train)

# Predictions
y_pred_train_good = model_good.predict(X_train)
y_pred_test_good = model_good.predict(X_test)

# Accuracies
train_acc_good = accuracy_score(y_train, y_pred_train_good) * 100
test_acc_good = accuracy_score(y_test, y_pred_test_good) * 100
gap_good = train_acc_good - test_acc_good

print(f"\n📊 Results:")
print(f"  Training Accuracy: {train_acc_good:.1f}%")
print(f"  Test Accuracy:     {test_acc_good:.1f}%")
print(f"  Gap:               {gap_good:.1f}%")

print(f"\n🔍 Diagnosis:")
if gap_good < 8 and train_acc_good > 80:
    print("  ✅ GOOD FIT detected!")
    print("  📌 Both accuracies are HIGH")
    print("  📌 Gap is SMALL (< 8%)")
    print("  📌 Model generalizes well")
    print("\n💡 What to do:")
    print("  ✓ Model is well-tuned")
    print("  ✓ This is your baseline")
    print("  → Minor improvements: ensemble, feature engineering")
    print("  → Consider this DONE for most applications")

# Visualize all 3 scenarios side-by-side
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

scenarios = [
    ('High Bias\n(Underfit)', train_acc, test_acc, 'darkred'),
    ('High Variance\n(Overfit)', train_acc_var, test_acc_var, 'darkorange'),
    ('Good Fit\n(Balanced)', train_acc_good, test_acc_good, 'darkgreen')
]

for ax, (title, train, test, color) in zip(axes, scenarios):
    gap = train - test
    bars = ax.bar(['Train', 'Test'], [train, test], 
                  color=['steelblue', 'coral'], 
                  edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', color=color)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, axis='y')
    
    # Value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.1f}%',
               ha='center', fontsize=10, fontweight='bold')
    
    # Gap label
    ax.text(0.5, 5, f'Gap: {gap:.1f}%',
           ha='center', fontsize=10, color=color, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('Three Scenarios: Bias vs Variance vs Good Fit', 
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)

# COMMAND ----------

# DBTITLE 1,🎮 Interactive: Find the Sweet Spot (Good Fit)
# ═══════════════════════════════════════════════════════════════════════════════
# 🎮 INTERACTIVE: FIND THE SWEET SPOT - Experiment with Random Forest parameters!
# ═══════════════════════════════════════════════════════════════════════════════

def explore_good_fit(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    """
    Interactive exploration to find the sweet spot (good fit)
    """
    clear_output(wait=True)
    
    # Train Random Forest with user parameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
    test_acc = accuracy_score(y_test, model.predict(X_test)) * 100
    gap = train_acc - test_acc
    
    # Print results
    print("═" * 80)
    print("🎯 FINDING THE SWEET SPOT - BALANCED MODEL (GOOD FIT)")
    print("═" * 80)
    print(f"\n⚙️  Random Forest Configuration:")
    print(f"   Number of Trees: {n_estimators}")
    print(f"   Max Depth: {max_depth}")
    print(f"   Min Samples Split: {min_samples_split}")
    print(f"   Min Samples Leaf: {min_samples_leaf}")
    
    print(f"\n📊 Performance:")
    print(f"   Training Accuracy: {train_acc:.1f}%")
    print(f"   Test Accuracy:     {test_acc:.1f}%")
    print(f"   Gap:               {gap:.1f}%")
    
    # Diagnosis with scoring
    print(f"\n🔍 DIAGNOSIS:")
    
    # Calculate fit quality score
    quality_score = 0
    issues = []
    strengths = []
    
    # Check accuracy levels
    if train_acc > 85 and test_acc > 85:
        strengths.append("Both accuracies excellent (>85%)")
        quality_score += 40
    elif train_acc > 75 and test_acc > 75:
        strengths.append("Both accuracies good (>75%)")
        quality_score += 30
    elif train_acc < 70 or test_acc < 70:
        issues.append("Low accuracy - underfitting")
    
    # Check gap
    if gap < 5:
        strengths.append("Excellent gap (<5%)")
        quality_score += 40
    elif gap < 8:
        strengths.append("Good gap (<8%)")
        quality_score += 30
    elif gap < 15:
        issues.append("Moderate gap - slight overfitting")
        quality_score += 15
    else:
        issues.append("Large gap - significant overfitting")
    
    # Balance check
    if abs(gap) < 8 and train_acc > 80:
        strengths.append("Well-balanced model!")
        quality_score += 20
    
    # Display diagnosis
    if quality_score >= 80:
        print("   🟢 EXCELLENT FIT - This is the sweet spot!")
        print("   ⭐⭐⭐⭐⭐ Model is production-ready")
    elif quality_score >= 60:
        print("   🟢 GOOD FIT - Model generalizes well")
        print("   ⭐⭐⭐⭐ Acceptable for most use cases")
    elif quality_score >= 40:
        print("   🟡 MODERATE FIT - Could be improved")
        print("   ⭐⭐⭐ Needs tuning")
    elif train_acc > 95 and gap > 15:
        print("   🔴 HIGH VARIANCE (Overfitting)")
        print("   ⭐⭐ Model memorizing training data")
    else:
        print("   🔴 HIGH BIAS (Underfitting)")
        print("   ⭐⭐ Model too simple")
    
    print(f"\n   Quality Score: {quality_score}/100")
    
    if strengths:
        print(f"\n   ✅ Strengths:")
        for s in strengths:
            print(f"      • {s}")
    
    if issues:
        print(f"\n   ⚠️  Issues:")
        for i in issues:
            print(f"      • {i}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if gap > 15:
        print("   → Increase min_samples_split (more regularization)")
        print("   → Increase min_samples_leaf (simpler trees)")
        print("   → Reduce max_depth (less complexity)")
    elif train_acc < 75 and test_acc < 75:
        print("   → Increase max_depth (more complexity)")
        print("   → Decrease min_samples_split (allow more splits)")
        print("   → Increase n_estimators (more trees)")
    elif quality_score >= 80:
        print("   ✓ Model is already excellent - no changes needed!")
        print("   ✓ Consider this your baseline")
    else:
        print("   → Fine-tune parameters for marginal improvements")
    
    # Visualize
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])
    
    # LEFT: Performance bars
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(['Train', 'Test'], [train_acc, test_acc], 
                   color=['steelblue', 'coral'], edgecolor='black', linewidth=2)
    ax1.axhline(85, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Excellent (85%)')
    ax1.axhline(75, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Good (75%)')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    # Gap annotation
    if abs(gap) > 2:
        gap_color = 'green' if gap < 8 else 'orange' if gap < 15 else 'red'
        ax1.text(0.5, 10, f'Gap: {gap:.1f}%', ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', edgecolor=gap_color, linewidth=2),
               fontweight='bold')
    
    # MIDDLE: Quality score gauge
    ax2 = fig.add_subplot(gs[1])
    
    # Create quality gauge
    categories = ['Quality\nScore']
    values = [quality_score]
    
    # Color based on score
    if quality_score >= 80:
        bar_color = 'green'
        label = 'Excellent'
    elif quality_score >= 60:
        bar_color = 'limegreen'
        label = 'Good'
    elif quality_score >= 40:
        bar_color = 'orange'
        label = 'Moderate'
    else:
        bar_color = 'red'
        label = 'Poor'
    
    bar = ax2.barh(categories, values, color=bar_color, edgecolor='black', linewidth=2)
    ax2.axvline(80, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Excellent (80+)')
    ax2.axvline(60, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Good (60+)')
    ax2.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title(f'Fit Quality: {label}', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(alpha=0.3, axis='x')
    
    ax2.text(quality_score/2, 0, f'{quality_score}/100', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # RIGHT: Fit type indicator
    ax3 = fig.add_subplot(gs[2])
    
    # Show where current model falls on bias-variance spectrum
    spectrum_x = np.linspace(0, 100, 100)
    
    # Create gradient background
    for i, x in enumerate(spectrum_x):
        if x < 33:
            color = plt.cm.Reds(x/33)
        elif x < 67:
            color = plt.cm.Greens((x-33)/34)
        else:
            color = plt.cm.Oranges((x-67)/33)
        ax3.axvline(x, color=color, alpha=0.3, linewidth=2)
    
    # Determine position on spectrum
    if train_acc < 70 and test_acc < 70:
        position = 15  # High bias (left)
        marker_label = 'HIGH BIAS\n(Underfit)'
    elif gap > 20:
        position = 85  # High variance (right)
        marker_label = 'HIGH VARIANCE\n(Overfit)'
    else:
        # Position based on quality score
        position = 50  # Sweet spot (middle)
        marker_label = 'SWEET SPOT\n(Good Fit)'
    
    # Plot marker
    ax3.plot(position, 0.5, 'o', markersize=25, color='yellow', 
            markeredgecolor='black', markeredgewidth=3, zorder=10)
    ax3.text(position, 0.5, '★', ha='center', va='center', 
            fontsize=20, color='black', fontweight='bold', zorder=11)
    
    # Labels
    ax3.text(15, 0.8, 'HIGH BIAS\n(Underfit)', ha='center', fontsize=9, fontweight='bold')
    ax3.text(50, 0.8, 'GOOD FIT\n(Balanced)', ha='center', fontsize=9, fontweight='bold', color='darkgreen')
    ax3.text(85, 0.8, 'HIGH VARIANCE\n(Overfit)', ha='center', fontsize=9, fontweight='bold')
    
    ax3.text(position, 0.2, marker_label, ha='center', fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 1)
    ax3.set_title('Bias-Variance Spectrum', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    plt.suptitle('🎮 Interactive Sweet Spot Finder - Find the Perfect Balance!', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "═" * 80)

print("\n🎮 INTERACTIVE GOOD FIT EXPLORATION")
print("Find the sweet spot by tuning Random Forest parameters!\n")

estimators_slider = widgets.IntSlider(
    value=100, min=10, max=200, step=10,
    description='Trees:', continuous_update=False,
    style={'description_width': '150px'}
)

depth_slider = widgets.IntSlider(
    value=10, min=3, max=25, step=1,
    description='Max Depth:', continuous_update=False,
    style={'description_width': '150px'}
)

split_slider = widgets.IntSlider(
    value=20, min=2, max=50, step=2,
    description='Min Samples Split:', continuous_update=False,
    style={'description_width': '150px'}
)

leaf_slider = widgets.IntSlider(
    value=10, min=1, max=30, step=1,
    description='Min Samples Leaf:', continuous_update=False,
    style={'description_width': '150px'}
)

interactive_goodfit = widgets.interactive(
    explore_good_fit,
    n_estimators=estimators_slider,
    max_depth=depth_slider,
    min_samples_split=split_slider,
    min_samples_leaf=leaf_slider
)

display(interactive_goodfit)

# COMMAND ----------

# DBTITLE 1,Technique 1 - Collect More Data
# ===================================================================
# GENERALIZATION TECHNIQUE 1: COLLECT MORE DATA
# ===================================================================
# Demonstrate how more data reduces overfitting

print("=" * 70)
print("TECHNIQUE 1: COLLECT MORE DATA")
print("=" * 70)

# Train on increasing amounts of data
data_sizes = [100, 200, 400, 700]
train_accs = []
test_accs = []
gaps = []

for size in data_sizes:
    # Use subset of data
    X_train_sub = X_train[:size]
    y_train_sub = y_train[:size]
    
    # Train overfitting model
    model = DecisionTreeClassifier(max_depth=20, random_state=42)
    model.fit(X_train_sub, y_train_sub)
    
    # Evaluate
    train_acc_sub = accuracy_score(y_train_sub, model.predict(X_train_sub)) * 100
    test_acc_sub = accuracy_score(y_test, model.predict(X_test)) * 100
    
    train_accs.append(train_acc_sub)
    test_accs.append(test_acc_sub)
    gaps.append(train_acc_sub - test_acc_sub)
    
    print(f"\nData size: {size} samples")
    print(f"  Train: {train_acc_sub:.1f}%  |  Test: {test_acc_sub:.1f}%  |  Gap: {gaps[-1]:.1f}%")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Train vs Test curves
ax1.plot(data_sizes, train_accs, 'o-', color='blue', label='Train Accuracy', linewidth=2, markersize=8)
ax1.plot(data_sizes, test_accs, 'o-', color='red', label='Test Accuracy', linewidth=2, markersize=8)
ax1.fill_between(data_sizes, train_accs, test_accs, alpha=0.2, color='orange')
ax1.set_xlabel('Training Set Size', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Effect of More Data on Overfitting', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.set_ylim(60, 105)

# Right: Gap reduction
ax2.plot(data_sizes, gaps, 'o-', color='darkred', linewidth=2.5, markersize=8)
ax2.fill_between(data_sizes, gaps, 0, alpha=0.3, color='red')
ax2.axhline(8, color='green', linestyle='--', linewidth=2, label='Good Gap (<8%)')
ax2.set_xlabel('Training Set Size', fontsize=11, fontweight='bold')
ax2.set_ylabel('Train-Test Gap (%)', fontsize=11, fontweight='bold')
ax2.set_title('Gap Reduces with More Data', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, max(gaps) + 5)

plt.tight_layout()
plt.show()

print("\n💡 Key Insight:")
print(f"  Gap reduced from {gaps[0]:.1f}% to {gaps[-1]:.1f}% with more data!")
print(f"  ⭐⭐⭐⭐⭐ Most effective solution for high variance")
print("\n" + "=" * 70)

# COMMAND ----------

# ===================================================================
# GENERALIZATION TECHNIQUE 2: REGULARIZATION (L1/L2)
# ===================================================================
# Add penalty for large weights to prevent overfitting

print("=" * 70)
print("TECHNIQUE 2: REGULARIZATION (L1/L2)")
print("=" * 70)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC _reg           y_reg
# MAGIC (200 × 20)      (200,)
# MAGIC
# MAGIC 20 features     sirf 3 features se bana:
# MAGIC banaye gaye     y = 3×f0 + 2×f1 - f2 + noise
# MAGIC
# MAGIC 17 features     completely useless hain
# MAGIC pure noise hain (weight=0)
# MAGIC

# COMMAND ----------



# COMMAND ----------

# ═══════════════════════════════════════════════════════════════════════════════
# 🎮 INTERACTIVE REGULARIZATION - Experiment with L1/L2 penalties!
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
import ipywidgets as widgets
from IPython.display import display, clear_output

# Generate regression data once (outside the interactive function)
np.random.seed(42)
X_reg = np.random.randn(200, 20)
y_reg = 3 * X_reg[:, 0] + 2 * X_reg[:, 1] - X_reg[:, 2] + np.random.randn(200) * 0.5
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train_reg)
X_test_poly = poly.transform(X_test_reg)
X_train_poly.shape


# COMMAND ----------

# DBTITLE 1,Technique 2 - Regularization (L1/L2)
#R2 How much variation in the target variable is explained by the model.
#adjusted r2
def explore_regularization(poly_degree, ridge_alpha, lasso_alpha):
    """
    Interactive regularization exploration
    """
    clear_output(wait=True)
    
    # Add polynomial features (makes model complex)
    poly = PolynomialFeatures(degree=poly_degree)
    X_train_poly = poly.fit_transform(X_train_reg)
    X_test_poly = poly.transform(X_test_reg)

    
    # Print configuration
    print("═" * 80)
    print("🎯 REGULARIZATION EXPLORATION")
    print("═" * 80)
    print(f"\n⚙️  Configuration:")
    print(f"   Polynomial Degree: {poly_degree}")
    print(f"   Features: {X_train_poly.shape[1]} (from {X_train_reg.shape[1]} original)")
    print(f"   Ridge (L2) Alpha: {ridge_alpha}")
    print(f"   Lasso (L1) Alpha: {lasso_alpha}")
    print(f"\n💡 Higher alpha = Stronger regularization = Simpler model")
    
    # Compare: No regularization vs L2 (Ridge) vs L1 (Lasso)
    models_reg = {
        'No Regularization': LinearRegression(),
        f'L2 (Ridge λ={ridge_alpha})': Ridge(alpha=ridge_alpha),
        f'L1 (Lasso λ={lasso_alpha})': Lasso(alpha=lasso_alpha)
    }

    
    results = {}
    print(f"\n📊 Results:")
    for name, model in models_reg.items():
        model.fit(X_train_poly, y_train_reg)
        
        train_r2 = r2_score(y_train_reg, model.predict(X_train_poly)) * 100
        test_r2 = r2_score(y_test_reg, model.predict(X_test_poly)) * 100
        gap = train_r2 - test_r2
        
        # Calculate Adjusted R²
        # Formula: 1 - [(1-R²)(n-1)/(n-p-1)] where n=samples, p=features
        n_train = len(y_train_reg)
        n_test = len(y_test_reg)
        p = X_train_poly.shape[1]
        
        train_adj_r2 = (1 - (1 - train_r2/100) * (n_train - 1) / (n_train - p - 1)) * 100
        test_adj_r2 = (1 - (1 - test_r2/100) * (n_test - 1) / (n_test - p - 1)) * 100
        
        results[name] = {
            'train': train_r2, 'test': test_r2, 'gap': gap, 'model': model,
            'train_adj': train_adj_r2, 'test_adj': test_adj_r2
        }
        
        # Count non-zero coefficients (for feature selection insight)
        if hasattr(model, 'coef_'):
            non_zero = np.sum(np.abs(model.coef_) > 0.01)
        else:
            non_zero = len(model.coef_)
        
        print(f"\n{name}:")
        print(f"  R² Score:")
        print(f"    Train: {train_r2:.1f}%  |  Test: {test_r2:.1f}%  |  Gap: {gap:.1f}%")
        print(f"  Adjusted R²:")
        print(f"    Train: {train_adj_r2:.1f}%  |  Test: {test_adj_r2:.1f}%")
        print(f"  Penalty: {train_r2 - train_adj_r2:.1f}% (R² - Adj R²)")
        print(f"  Active features: {non_zero}/{X_train_poly.shape[1]}")
    
    # Diagnosis
    print(f"\n🔍 DIAGNOSIS:")
    no_reg_gap = results['No Regularization']['gap']
    ridge_gap = results[f'L2 (Ridge λ={ridge_alpha})']['gap']
    lasso_gap = results[f'L1 (Lasso λ={lasso_alpha})']['gap']
    
    if no_reg_gap > 15:
        print("   🔴 No Regularization: HIGH OVERFITTING")
    if ridge_gap < 8:
        print("   ✅ Ridge: Good generalization!")
    elif ridge_gap < no_reg_gap:
        print(f"   🟡 Ridge: Improved (gap reduced by {no_reg_gap - ridge_gap:.1f}%)")
    
    if lasso_gap < 8:
        print("   ✅ Lasso: Good generalization!")
    elif lasso_gap < no_reg_gap:
        print(f"   🟡 Lasso: Improved (gap reduced by {no_reg_gap - lasso_gap:.1f}%)")
    
    # Feature selection insight for Lasso
    lasso_model = results[f'L1 (Lasso λ={lasso_alpha})']['model']
    lasso_nonzero = np.sum(np.abs(lasso_model.coef_) > 0.01)
    if lasso_nonzero < X_train_poly.shape[1] * 0.3:
        print(f"   💡 Lasso performed feature selection: {lasso_nonzero}/{X_train_poly.shape[1]} features")

    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # TOP LEFT: R² comparison
    ax = axes[0, 0]
    model_names = list(results.keys())
    train_scores = [results[m]['train'] for m in model_names]
    test_scores = [results[m]['test'] for m in model_names]
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    ax.bar(x_pos - width/2, train_scores, width, label='Train R²', color='steelblue', edgecolor='black')
    ax.bar(x_pos + width/2, test_scores, width, label='Test R²', color='coral', edgecolor='black')
    ax.set_ylabel('R² Score (%)', fontsize=11, fontweight='bold')
    ax.set_title('R² Score (Standard)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([name.split('(')[0].strip() for name in model_names], fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels
    for i, (train, test) in enumerate(zip(train_scores, test_scores)):
        ax.text(i - width/2, train + 2, f'{train:.0f}', ha='center', fontsize=9)
        ax.text(i + width/2, test + 2, f'{test:.0f}', ha='center', fontsize=9)

    # TOP MIDDLE: Adjusted R² comparison
    ax = axes[0, 1]
    train_adj_scores = [results[m]['train_adj'] for m in model_names]
    test_adj_scores = [results[m]['test_adj'] for m in model_names]
    
    ax.bar(x_pos - width/2, train_adj_scores, width, label='Train Adj R²', color='darkblue', edgecolor='black')
    ax.bar(x_pos + width/2, test_adj_scores, width, label='Test Adj R²', color='darkred', edgecolor='black')
    ax.set_ylabel('Adjusted R² (%)', fontsize=11, fontweight='bold')
    ax.set_title('Adjusted R² (Penalizes Complexity)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([name.split('(')[0].strip() for name in model_names], fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels
    for i, (train_adj, test_adj) in enumerate(zip(train_adj_scores, test_adj_scores)):
        ax.text(i - width/2, train_adj + 2, f'{train_adj:.0f}', ha='center', fontsize=9)
        ax.text(i + width/2, test_adj + 2, f'{test_adj:.0f}', ha='center', fontsize=9)

    # TOP RIGHT: R² vs Adjusted R² penalty
    ax = axes[0, 2]
    penalties = [results[m]['train'] - results[m]['train_adj'] for m in model_names]
    colors_penalty = ['lightblue', 'lightgreen', 'lightyellow']
    bars = ax.bar(range(len(model_names)), penalties, color=colors_penalty, edgecolor='black', linewidth=2)
    ax.set_ylabel('Penalty (R² - Adj R²) %', fontsize=11, fontweight='bold')
    ax.set_title('Complexity Penalty', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([name.split('(')[0].strip() for name in model_names], fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels and explanation
    for i, (bar, penalty) in enumerate(zip(bars, penalties)):
        height = bar.get_height()
        ax.text(i, height + 0.2, f'{penalty:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    ax.text(0.5, max(penalties) * 0.5, 
           f'Higher penalty =\nMore features used',
           ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # BOTTOM LEFT: Gap comparison
    ax = axes[1, 0]
    gaps = [results[m]['gap'] for m in model_names]
    colors = ['red' if g > 15 else ('orange' if g > 8 else 'green') for g in gaps]
    bars = ax.bar(range(len(model_names)), gaps, color=colors, edgecolor='black', linewidth=2)
    ax.axhline(8, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target (<8%)')
    ax.set_ylabel('Train-Test Gap (%)', fontsize=11, fontweight='bold')
    ax.set_title('Gap Reduction', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([name.split('(')[0].strip() for name in model_names], fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, gap) in enumerate(zip(bars, gaps)):
        height = bar.get_height()
        ax.text(i, height + 0.5, f'{gap:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    # BOTTOM MIDDLE: Feature sparsity (for Lasso)
    ax = axes[1, 1]
    feature_counts = []
    for name in model_names:
        model = results[name]['model']
        if hasattr(model, 'coef_'):
            non_zero = np.sum(np.abs(model.coef_) > 0.01)
        else:
            non_zero = len(model.coef_) if hasattr(model, 'coef_') else X_train_poly.shape[1]
        feature_counts.append(non_zero)
    
    bars = ax.bar(range(len(model_names)), feature_counts, 
                  color=['steelblue', 'lightblue', 'orange'], edgecolor='black', linewidth=2)
    ax.axhline(X_train_poly.shape[1], color='red', linestyle='--', linewidth=1.5, 
              alpha=0.5, label=f'Total ({X_train_poly.shape[1]})')
    ax.set_ylabel('Active Features', fontsize=11, fontweight='bold')
    ax.set_title('Feature Selection Effect', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([name.split('(')[0].strip() for name in model_names], fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, feature_counts)):
        height = bar.get_height()
        ax.text(i, height + X_train_poly.shape[1] * 0.02, str(count), 
               ha='center', fontsize=9, fontweight='bold')
    
    # BOTTOM RIGHT: Explanation panel
    ax = axes[1, 2]
    ax.axis('off')
    
    explanation_text = f"""
🎯 R² vs Adjusted R²

R² (Coefficient of Determination):
  • Measures variance explained
  • Always increases with more features
  • Can be misleading!

Adjusted R²:
  • Penalizes unnecessary features
  • Formula: 1 - [(1-R²)(n-1)/(n-p-1)]
  • n = samples, p = features
  • Only increases if new feature helps

📊 Current Model:
  • Total Features: {X_train_poly.shape[1]}
  • Samples: {n_train}
  
💡 Key Insight:
  Model with {X_train_poly.shape[1]} features gets
  ~{np.mean(penalties):.1f}% penalty on average.
  
  This penalty reflects the cost of
  model complexity!
"""
    
    ax.text(0.1, 0.5, explanation_text, fontsize=9, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1),
           family='monospace')

    
    plt.suptitle('🎮 Interactive R² vs Adjusted R² Explorer - See the Complexity Penalty!', 
                fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "═" * 80)
    print("\n💡 Key Insights:")
    print("  • R²: Always increases with more features (can be misleading!)")
    print("  • Adjusted R²: Penalizes complexity, only increases if feature helps")
    print("  • Penalty = R² - Adjusted R² (higher with more features)")
    print("  • L2 (Ridge): Shrinks all weights proportionally")
    print("  • L1 (Lasso): Zeros out some weights (automatic feature selection)")
    print(f"  • Current penalty: ~{np.mean(penalties):.1f}% for {X_train_poly.shape[1]} features")
    print("\n" + "═" * 80)

print("🎮 INTERACTIVE REGULARIZATION EXPLORATION")
print("Experiment with polynomial complexity and regularization strengths!\n")

poly_slider = widgets.IntSlider(
    value=3, min=1, max=5, step=1,
    description='Poly Degree:', continuous_update=False,
    style={'description_width': '150px'}
)

ridge_slider = widgets.FloatSlider(
    value=10.0, min=0.01, max=100.0, step=0.5,
    description='Ridge Alpha (L2):', continuous_update=False,
    style={'description_width': '150px'}
)

lasso_slider = widgets.FloatSlider(
    value=1.0, min=0.01, max=10.0, step=0.1,
    description='Lasso Alpha (L1):', continuous_update=False,
    style={'description_width': '150px'}
)

interactive_reg = widgets.interactive(
    explore_regularization,
    poly_degree=poly_slider,
    ridge_alpha=ridge_slider,
    lasso_alpha=lasso_slider
)

display(interactive_reg)

# COMMAND ----------

# DBTITLE 1,Technique 3 - Cross Validation
# ═══════════════════════════════════════════════════════════════════════════════
# 🎮 INTERACTIVE CROSS-VALIDATION - Experiment with K-Fold parameters!
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import ipywidgets as widgets
from IPython.display import display, clear_output

def explore_cross_validation(n_folds, max_depth, n_estimators):
    """
    Interactive cross-validation exploration
    """
    clear_output(wait=True)
    
    # Print configuration
    print("═" * 80)
    print("🎯 CROSS-VALIDATION EXPLORATION")
    print("═" * 80)
    print(f"\n⚙️  Configuration:")
    print(f"   K-Folds: {n_folds}")
    print(f"   Model: Random Forest")
    print(f"   Max Depth: {max_depth}")
    print(f"   N Estimators: {n_estimators}")
    
    # Train model and evaluate with K-Fold CV
    model_cv = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    
    # K-Fold Cross Validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_cv, X_train, y_train, cv=kfold, scoring='accuracy')
    
    print(f"\n📊 {n_folds}-Fold Cross-Validation Scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score*100:.1f}%")
    
    print(f"\n📈 Summary:")
    print(f"  Mean CV Score: {cv_scores.mean()*100:.1f}%")
    print(f"  Std Dev:       {cv_scores.std()*100:.1f}%")
    print(f"  Range:         [{cv_scores.min()*100:.1f}%, {cv_scores.max()*100:.1f}%]")
    
    # Compare with simple train/test split
    model_cv.fit(X_train, y_train)
    train_acc_simple = accuracy_score(y_train, model_cv.predict(X_train)) * 100
    test_acc_simple = accuracy_score(y_test, model_cv.predict(X_test)) * 100
    
    print(f"\n🔄 Simple Train/Test Split:")
    print(f"  Train: {train_acc_simple:.1f}%")
    print(f"  Test:  {test_acc_simple:.1f}%")
    
    # Diagnosis
    print(f"\n🔍 DIAGNOSIS:")
    cv_mean = cv_scores.mean() * 100
    cv_std = cv_scores.std() * 100
    
    if cv_std < 2:
        print(f"   ✅ Very stable! Low variance ({cv_std:.1f}%) across folds")
    elif cv_std < 5:
        print(f"   🟢 Stable. Moderate variance ({cv_std:.1f}%) across folds")
    else:
        print(f"   ⚠️  High variance ({cv_std:.1f}%). Results may depend on data split")
    
    # Check if CV matches test
    diff = abs(cv_mean - test_acc_simple)
    if diff < 2:
        print(f"   ✅ CV estimate ({cv_mean:.1f}%) matches test ({test_acc_simple:.1f}%)")
    elif diff < 5:
        print(f"   🟡 CV estimate ({cv_mean:.1f}%) close to test ({test_acc_simple:.1f}%)")
    else:
        print(f"   🔴 CV estimate ({cv_mean:.1f}%) differs from test ({test_acc_simple:.1f}%) by {diff:.1f}%")
    
    # Check if more folds would help
    if n_folds < 5 and cv_std > 3:
        print(f"   💡 Try increasing folds to reduce variance")
    elif n_folds > 10:
        print(f"   💡 {n_folds} folds may be excessive - diminishing returns")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # TOP LEFT: CV scores distribution
    ax = axes[0, 0]
    fold_indices = range(1, n_folds + 1)
    colors_folds = ['steelblue' if abs(score - cv_scores.mean()) < cv_scores.std() 
                    else 'orange' for score in cv_scores]
    
    bars = ax.bar(fold_indices, cv_scores * 100, color=colors_folds, edgecolor='black', alpha=0.8)
    ax.axhline(cv_scores.mean() * 100, color='red', linestyle='--', linewidth=2.5, 
              label=f'Mean: {cv_scores.mean()*100:.1f}%')
    ax.fill_between(fold_indices, 
                   (cv_scores.mean() - cv_scores.std()) * 100,
                   (cv_scores.mean() + cv_scores.std()) * 100,
                   alpha=0.2, color='red', label=f'±1 Std Dev ({cv_scores.std()*100:.1f}%)')
    ax.set_xlabel('Fold Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'{n_folds}-Fold Cross-Validation Scores', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    ax.set_xticks(fold_indices)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, cv_scores)):
        height = bar.get_height()
        ax.text(i + 1, height + 1, f'{score*100:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    # TOP RIGHT: Comparison with error bars
    ax = axes[0, 1]
    methods = ['Simple\nTrain/Test', f'{n_folds}-Fold\nCross-Val']
    scores_comp = [test_acc_simple, cv_scores.mean() * 100]
    errors = [0, cv_scores.std() * 100]
    
    bars = ax.bar(methods, scores_comp, yerr=errors, color=['coral', 'steelblue'], 
           edgecolor='black', capsize=10, alpha=0.8, linewidth=2)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('CV Provides Robust Estimate with Confidence', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels
    for i, (score, err) in enumerate(zip(scores_comp, errors)):
        label = f'{score:.1f}%' if err == 0 else f'{score:.1f}%\n±{err:.1f}%'
        ax.text(i, score + err + 2, label, ha='center', fontsize=10, fontweight='bold')
    
    # Highlight the confidence interval advantage
    ax.text(1, scores_comp[1] - 5, 'Confidence\nInterval!', ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # BOTTOM LEFT: Variance vs Number of Folds
    ax = axes[1, 0]
    
    # Show how variance changes with different fold counts
    fold_range = range(2, min(15, len(X_train) // 10) + 1)
    variances = []
    
    for k in fold_range:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(model_cv, X_train, y_train, cv=kf, scoring='accuracy')
        variances.append(scores.std() * 100)
    
    ax.plot(fold_range, variances, 'o-', color='purple', linewidth=2.5, markersize=8)
    ax.axvline(n_folds, color='red', linestyle='--', linewidth=2, alpha=0.7,
              label=f'Current ({n_folds} folds)')
    ax.axhline(cv_std, color='red', linestyle=':', linewidth=2, alpha=0.7,
              label=f'Current Std: {cv_std:.1f}%')
    ax.set_xlabel('Number of Folds (K)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Standard Deviation (%)', fontsize=11, fontweight='bold')
    ax.set_title('Variance Decreases with More Folds', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Annotate sweet spot
    if n_folds >= 5 and n_folds <= 10:
        ax.text(n_folds, variances[list(fold_range).index(n_folds)] + 0.3, 
               'Sweet\nSpot!', ha='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # BOTTOM RIGHT: Train vs CV vs Test comparison
    ax = axes[1, 1]
    
    all_methods = ['Train', 'CV Mean', 'Test']
    all_scores = [train_acc_simple, cv_scores.mean() * 100, test_acc_simple]
    all_errors = [0, cv_scores.std() * 100, 0]
    colors_all = ['steelblue', 'green', 'coral']
    
    bars = ax.bar(all_methods, all_scores, yerr=all_errors, color=colors_all,
           edgecolor='black', capsize=8, alpha=0.8, linewidth=2)
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Full Performance Comparison', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels
    for i, (score, err) in enumerate(zip(all_scores, all_errors)):
        label = f'{score:.1f}%' if err == 0 else f'{score:.1f}%\n±{err:.1f}%'
        ax.text(i, score + err + 1, label, ha='center', fontsize=10, fontweight='bold')
    
    # Show gap
    train_cv_gap = train_acc_simple - cv_scores.mean() * 100
    if train_cv_gap > 5:
        ax.annotate('', xy=(0, train_acc_simple), xytext=(1, cv_scores.mean() * 100),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(0.5, (train_acc_simple + cv_scores.mean() * 100) / 2,
               f'Gap:\n{train_cv_gap:.1f}%',
               ha='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.suptitle('🎮 Interactive Cross-Validation Explorer - Drag Sliders!', 
                fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "═" * 80)
    print("\n💡 Key Insights:")
    print(f"  • {n_folds}-Fold CV: Mean {cv_mean:.1f}% ± {cv_std:.1f}%")
    print(f"  • Variance decreases with more folds (but computational cost increases)")
    print(f"  • Sweet spot: 5-10 folds for most datasets")
    print(f"  • CV helps detect overfitting: Train {train_acc_simple:.1f}% vs CV {cv_mean:.1f}%")
    print("\n💡 Benefits:")
    print("  ✓ Uses all data for training AND validation")
    print("  ✓ Reduces variance in performance estimate")
    print("  ✓ Provides confidence intervals")
    print("  ✓ Essential for hyperparameter tuning")
    print("  ⭐⭐⭐⭐ Highly recommended for small datasets")
    print("\n" + "═" * 80)

print("🎮 INTERACTIVE CROSS-VALIDATION EXPLORATION")
print("Experiment with K-Folds and model parameters!\n")

folds_slider = widgets.IntSlider(
    value=5, min=2, max=15, step=1,
    description='K-Folds:', continuous_update=False,
    style={'description_width': '150px'}
)

depth_slider = widgets.IntSlider(
    value=10, min=3, max=25, step=1,
    description='Max Depth:', continuous_update=False,
    style={'description_width': '150px'}
)

estimators_slider = widgets.IntSlider(
    value=50, min=10, max=200, step=10,
    description='N Estimators:', continuous_update=False,
    style={'description_width': '150px'}
)

interactive_cv = widgets.interactive(
    explore_cross_validation,
    n_folds=folds_slider,
    max_depth=depth_slider,
    n_estimators=estimators_slider
)

display(interactive_cv)

# COMMAND ----------

# ===================================================================
# GENERALIZATION TECHNIQUE 4: EARLY STOPPING
# ===================================================================
# Stop training when validation loss starts increasing

print("=" * 70)
print("TECHNIQUE 4: EARLY STOPPING")
print("=" * 70)

# Simulate iterative training (like neural networks)
from sklearn.ensemble import GradientBoostingClassifier

# Split training data into train/validation
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# COMMAND ----------


# Track performance over iterations
n_estimators_range = range(1, 201, 5)
train_scores_iter = []
val_scores_iter = []
test_scores_iter = []

# COMMAND ----------

n_estimators_range

# COMMAND ----------


for n_est in n_estimators_range:
    model_iter = GradientBoostingClassifier(n_estimators=n_est, max_depth=5, random_state=42)
    model_iter.fit(X_tr, y_tr)
    train_scores_iter.append(accuracy_score(y_tr, model_iter.predict(X_tr)) * 100)
    val_scores_iter.append(accuracy_score(y_val, model_iter.predict(X_val)) * 100)
    test_scores_iter.append(accuracy_score(y_test, model_iter.predict(X_test)) * 100)

# COMMAND ----------


# Find best stopping point (highest validation accuracy)
best_idx = np.argmax(val_scores_iter)
best_n_estimators = list(n_estimators_range)[best_idx]

# COMMAND ----------

best_idx 

# COMMAND ----------

list(n_estimators_range)

# COMMAND ----------





print(f"\n📊 Training Progress:")
print(f"  Best validation accuracy at iteration {best_n_estimators}")
print(f"  Train: {train_scores_iter[best_idx]:.1f}%")
print(f"  Val:   {val_scores_iter[best_idx]:.1f}%")
print(f"  Test:  {test_scores_iter[best_idx]:.1f}%")

print(f"\n  If we continued to iteration {max(n_estimators_range)}:")
print(f"  Train: {train_scores_iter[-1]:.1f}%  (higher but overfitting!)")
print(f"  Val:   {val_scores_iter[-1]:.1f}%  (decreased)")
print(f"  Test:  {test_scores_iter[-1]:.1f}%  (decreased)")


# COMMAND ----------

# DBTITLE 1,Technique 4 - Early Stopping

# Visualize
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.plot(n_estimators_range, train_scores_iter, 'o-', color='blue', 
       label='Train Accuracy', linewidth=2, markersize=4, alpha=0.7)
ax.plot(n_estimators_range, val_scores_iter, 's-', color='orange', 
       label='Validation Accuracy', linewidth=2, markersize=4, alpha=0.7)
ax.plot(n_estimators_range, test_scores_iter, '^-', color='green', 
       label='Test Accuracy', linewidth=2, markersize=4, alpha=0.7)

# Mark best stopping point
ax.axvline(best_n_estimators, color='red', linestyle='--', linewidth=2.5, 
          label=f'Early Stop (iter {best_n_estimators})')
ax.scatter([best_n_estimators], [val_scores_iter[best_idx]], 
          s=200, color='red', zorder=5, marker='*')

# Annotate overfitting region
ax.annotate('Overfitting Zone\n(Val decreases)', 
           xy=(150, val_scores_iter[-1]), 
           xytext=(160, 75),
           arrowprops=dict(arrowstyle='->', color='red', lw=2),
           fontsize=10, color='red', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax.set_xlabel('Number of Training Iterations', fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax.set_title('Early Stopping Prevents Overfitting', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(70, 100)

plt.tight_layout()
plt.show()

print("\n💡 Key Insight:")
print(f"  ✓ Stop at iteration {best_n_estimators} instead of {max(n_estimators_range)}")
print(f"  ✓ Gain: {val_scores_iter[best_idx] - val_scores_iter[-1]:.1f}% validation accuracy")
print("  ✓ Prevents overfitting without changing model architecture")
print("  ⭐⭐⭐⭐ Essential for neural networks and boosting")
print("\n" + "=" * 70)

# COMMAND ----------

# DBTITLE 1,Technique 5 - Feature Selection
# ===================================================================
# GENERALIZATION TECHNIQUE 5: FEATURE SELECTION
# ===================================================================
# Remove irrelevant/noisy features to reduce complexity

print("=" * 70)
print("TECHNIQUE 5: FEATURE SELECTION")
print("=" * 70)

# Add noise features to demonstrate selection
np.random.seed(42)
X_train_noisy = np.hstack([X_train, np.random.randn(X_train.shape[0], 30)])  # Add 30 noise features
X_test_noisy = np.hstack([X_test, np.random.randn(X_test.shape[0], 30)])

print(f"\nOriginal features: {X_train.shape[1]}")
print(f"After adding noise: {X_train_noisy.shape[1]}")
print("(30 random features added - should be removed!)\n")

# Method 1: Tree-based feature importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_noisy, y_train)

# Select features based on importance
selector = SelectFromModel(rf, prefit=True, threshold='median')
X_train_selected = selector.transform(X_train_noisy)
X_test_selected = selector.transform(X_test_noisy)

print(f"Features selected: {X_train_selected.shape[1]}/{X_train_noisy.shape[1]}")

# Compare: All features vs Selected features
models_feat = {
    'All Features': (X_train_noisy, X_test_noisy),
    'Selected Features': (X_train_selected, X_test_selected)
}

results_feat = {}
for name, (X_tr, X_te) in models_feat.items():
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(X_tr, y_train)
    
    train_acc_f = accuracy_score(y_train, model.predict(X_tr)) * 100
    test_acc_f = accuracy_score(y_test, model.predict(X_te)) * 100
    gap_f = train_acc_f - test_acc_f
    
    results_feat[name] = {'train': train_acc_f, 'test': test_acc_f, 'gap': gap_f}
    
    print(f"\n{name} ({X_tr.shape[1]} features):")
    print(f"  Train: {train_acc_f:.1f}%  |  Test: {test_acc_f:.1f}%  |  Gap: {gap_f:.1f}%")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Feature importances
importances = rf.feature_importances_
top_20_idx = np.argsort(importances)[-20:]  # Top 20 features
colors = ['green' if i < 20 else 'red' for i in top_20_idx]  # Green=real, Red=noise

ax1.barh(range(20), importances[top_20_idx], color=colors, edgecolor='black')
ax1.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
ax1.set_ylabel('Feature Index', fontsize=11, fontweight='bold')
ax1.set_title('Top 20 Feature Importances\n(Green=Real, Red=Noise)', fontsize=12, fontweight='bold')
ax1.set_yticks(range(20))
ax1.set_yticklabels([f'F{i}' for i in top_20_idx])
ax1.grid(alpha=0.3, axis='x')

# Right: Performance comparison
model_names_f = list(results_feat.keys())
train_scores_f = [results_feat[m]['train'] for m in model_names_f]
test_scores_f = [results_feat[m]['test'] for m in model_names_f]
gaps_f = [results_feat[m]['gap'] for m in model_names_f]

x_pos = np.arange(len(model_names_f))
width = 0.35

ax2.bar(x_pos - width/2, train_scores_f, width, label='Train', color='steelblue', edgecolor='black')
ax2.bar(x_pos + width/2, test_scores_f, width, label='Test', color='coral', edgecolor='black')
ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('Feature Selection Improves Generalization', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(model_names_f)
ax2.legend()
ax2.grid(alpha=0.3, axis='y')

# Add gap labels
for i, gap in enumerate(gaps_f):
    color = 'red' if gap > 10 else 'green'
    ax2.text(i, min(train_scores_f[i], test_scores_f[i]) - 5,
            f'Gap: {gap:.1f}%',
            ha='center', fontsize=9, color=color, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n💡 Benefits:")
print("  ✓ Removes noisy/irrelevant features")
print("  ✓ Reduces model complexity")
print("  ✓ Faster training and inference")
print("  ✓ Improves interpretability")
print(f"  ✓ Gap reduced from {gaps_f[0]:.1f}% to {gaps_f[1]:.1f}%")
print("  ⭐⭐⭐ Especially useful for high-dimensional data")
print("\n" + "=" * 70)

# COMMAND ----------

# DBTITLE 1,Techniques 6-8 Summary
# ===================================================================
# TECHNIQUES 6-8: QUICK DEMONSTRATIONS
# ===================================================================

print("=" * 70)
print("TECHNIQUES 6-8: REDUCE COMPLEXITY, DROPOUT, DATA AUGMENTATION")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────
# TECHNIQUE 6: REDUCE MODEL COMPLEXITY
# ─────────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("TECHNIQUE 6: REDUCE MODEL COMPLEXITY")
print("-" * 70)

complexities = [(5, 'Simple'), (15, 'Medium'), (None, 'Complex')]
results_complex = {}

for max_d, name in complexities:
    model = DecisionTreeClassifier(max_depth=max_d, random_state=42)
    model.fit(X_train, y_train)
    
    train_c = accuracy_score(y_train, model.predict(X_train)) * 100
    test_c = accuracy_score(y_test, model.predict(X_test)) * 100
    gap_c = train_c - test_c
    
    results_complex[name] = {'train': train_c, 'test': test_c, 'gap': gap_c}
    
    depth_str = f"depth={max_d}" if max_d else "unlimited depth"
    print(f"\n{name} Model ({depth_str}):")
    print(f"  Train: {train_c:.1f}%  |  Test: {test_c:.1f}%  |  Gap: {gap_c:.1f}%")

print("\n💡 Insight: Simpler model (depth=5) has smallest gap!")
print("  ⭐⭐⭐ Easy and effective when model is too complex")

# ─────────────────────────────────────────────────────────────────────
# TECHNIQUE 7: DROPOUT (Neural Networks)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("TECHNIQUE 7: DROPOUT (Neural Networks)")
print("-" * 70)

print("\n📝 Dropout Explanation:")
print("  • Randomly drops neurons during training")
print("  • Prevents neurons from co-adapting")
print("  • Forces redundant representations")
print("  • Only for neural networks (TensorFlow/PyTorch)")
print("\nExample pseudocode:")
print("""  model = Sequential([
      Dense(128, activation='relu'),
      Dropout(0.5),  # Drop 50% of neurons
      Dense(64, activation='relu'),
      Dropout(0.3),  # Drop 30% of neurons
      Dense(10, activation='softmax')
  ])""")
print("\n  ⭐⭐⭐⭐⭐ Extremely effective for deep neural networks")

# ─────────────────────────────────────────────────────────────────────
# TECHNIQUE 8: DATA AUGMENTATION
# ─────────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("TECHNIQUE 8: DATA AUGMENTATION")
print("-" * 70)

print("\n📝 Data Augmentation Examples:")
print("\n  Images:")
print("    • Rotation: ±15 degrees")
print("    • Flipping: horizontal/vertical")
print("    • Cropping: random crops")
print("    • Color jitter: brightness, contrast, saturation")
print("    • Zoom: scale 0.8-1.2x")

print("\n  Text:")
print("    • Synonym replacement")
print("    • Back-translation (EN→FR→EN)")
print("    • Word insertion/deletion")

print("\n  Audio:")
print("    • Pitch shift")
print("    • Time stretch")
print("    • Add background noise")

print("\n  Tabular (our case):")
print("    • Add small Gaussian noise")
print("    • SMOTE (Synthetic Minority Over-sampling)")

# Demonstrate simple augmentation
print("\nDemo: Adding Gaussian noise to features")
X_train_aug = X_train.copy()
for _ in range(2):  # Create 2 augmented copies
    noise = np.random.randn(*X_train.shape) * 0.1
    X_train_aug = np.vstack([X_train_aug, X_train + noise])
y_train_aug = np.concatenate([y_train] * 3)

print(f"  Original size: {len(X_train)}")
print(f"  Augmented size: {len(X_train_aug)} (3x larger!)")

model_aug = RandomForestClassifier(max_depth=10, random_state=42)
model_aug.fit(X_train_aug, y_train_aug)

test_acc_aug = accuracy_score(y_test, model_aug.predict(X_test)) * 100
print(f"\n  Test accuracy with augmentation: {test_acc_aug:.1f}%")
print("  ⭐⭐⭐⭐ Very effective for images/audio/text")

print("\n" + "=" * 70)

# COMMAND ----------

# DBTITLE 1,Learning Curves - Complete Implementation
# ===================================================================
# LEARNING CURVES: COMPLETE IMPLEMENTATION
# ===================================================================
# Visual tool to decide: collect data vs simplify model

print("=" * 70)
print("LEARNING CURVES: YOUR DECISION-MAKING TOOL")
print("=" * 70)

# Generate learning curves for 3 models
models_lc = {
    'Overfitting\n(Complex)': DecisionTreeClassifier(max_depth=20, random_state=42),
    'Underfitting\n(Simple)': DecisionTreeClassifier(max_depth=2, random_state=42),
    'Good Fit\n(Balanced)': RandomForestClassifier(max_depth=8, n_estimators=50, random_state=42)
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (name, model) in zip(axes, models_lc.items()):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1) * 100
    val_mean = val_scores.mean(axis=1) * 100
    train_std = train_scores.std(axis=1) * 100
    val_std = val_scores.std(axis=1) * 100
    
    # Plot
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Train', linewidth=2, markersize=6)
    ax.plot(train_sizes, val_mean, 's-', color='red', label='Validation', linewidth=2, markersize=6)
    
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='red')
    
    ax.set_xlabel('Training Set Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_ylim(40, 100)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    # Diagnosis
    final_gap = train_mean[-1] - val_mean[-1]
    val_rising = val_mean[-1] > val_mean[-3]
    
    if 'Overfitting' in name:
        color = 'darkorange'
        diagnosis = f'Large gap: {final_gap:.1f}%\nVal rising ✓'
        decision = 'COLLECT MORE DATA'
    elif 'Underfitting' in name:
        color = 'darkred'
        diagnosis = f'Both low & flat\nGap: {final_gap:.1f}%'
        decision = 'INCREASE COMPLEXITY'
    else:
        color = 'darkgreen'
        diagnosis = f'Small gap: {final_gap:.1f}%\nConverged ✓'
        decision = 'WELL-TUNED!'
    
    ax.set_title(f'{name}\n{diagnosis}', fontsize=11, fontweight='bold', color=color)
    
    # Add decision box
    ax.text(0.5, 0.05, decision, transform=ax.transAxes,
           ha='center', fontsize=9, fontweight='bold', color=color,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.suptitle('Learning Curves: How to Decide What to Do', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("\n📈 How to Read Learning Curves:")
print("\n1️⃣ OVERFITTING (Left):")
print("   • Large train-test gap")
print("   • Val curve still rising")
print("   ➡️ COLLECT MORE DATA (will help!)")

print("\n2️⃣ UNDERFITTING (Middle):")
print("   • Both curves low and flat")
print("   • Converged to poor performance")
print("   ➡️ INCREASE COMPLEXITY (more data won't help!)")

print("\n3️⃣ GOOD FIT (Right):")
print("   • Small gap between train and val")
print("   • Both curves high and converged")
print("   ➡️ WELL-TUNED (minor improvements only)")

print("\n" + "=" * 70)

# COMMAND ----------

# DBTITLE 1,Final Summary and Interview Template
# MAGIC %md
# MAGIC ## 🎯 Final Summary: Your Interview Answer
# MAGIC
# MAGIC ### The Perfect Response
# MAGIC
# MAGIC When asked: **"Your model has 99% train accuracy and 72% test accuracy. What's happening and how would you fix it?"**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🗣️ Complete Answer (Say This):
# MAGIC
# MAGIC > **"The model is overfitting, which indicates high variance.**
# MAGIC >
# MAGIC > The large gap between training (99%) and test (72%) accuracy means the model is memorizing the training data rather than learning generalizable patterns.
# MAGIC >
# MAGIC > **To explain the concepts:**
# MAGIC >
# MAGIC > * **High bias (underfitting)** occurs when a model is too simple. It cannot learn underlying patterns, resulting in both train and test accuracy being low.
# MAGIC >
# MAGIC > * **High variance (overfitting)** occurs when a model is too complex. It memorizes training examples instead of generalizing, resulting in high train accuracy but low test accuracy.
# MAGIC >
# MAGIC > **To improve generalization, I would:**
# MAGIC >
# MAGIC > 1. **Collect more training data** (⭐⭐⭐⭐⭐) - Most effective
# MAGIC > 2. **Regularization (L1/L2)** (⭐⭐⭐⭐) - Penalizes large weights
# MAGIC > 3. **Cross-validation** (⭐⭐⭐⭐) - Robust evaluation
# MAGIC > 4. **Early stopping** (⭐⭐⭐⭐) - Stops before memorization
# MAGIC > 5. **Dropout** (⭐⭐⭐⭐⭐) - For neural networks
# MAGIC > 6. **Feature selection** (⭐⭐⭐) - Removes noisy features
# MAGIC > 7. **Reduce model complexity** (⭐⭐⭐) - Fewer parameters
# MAGIC > 8. **Data augmentation** (⭐⭐⭐⭐) - For images/text/audio
# MAGIC >
# MAGIC > **To decide between collecting data vs simplifying the model:**
# MAGIC >
# MAGIC > I would plot learning curves showing train/test performance vs training set size.
# MAGIC >
# MAGIC > * If the test curve is still rising and there's a large gap, **collect more data** (the model has capacity and more examples will help)
# MAGIC > * If the test curve has plateaued, **simplify the model or add regularization** (more data won't help, the model is too complex)
# MAGIC > * If both curves are low and flat, **increase model complexity** (high bias - the model is too simple)
# MAGIC >
# MAGIC > I would monitor both metrics during implementation and validate the improvements on a held-out test set."
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ What You Demonstrated
# MAGIC
# MAGIC This answer shows:
# MAGIC * ✓ **Immediate diagnosis** - "overfitting, high variance"
# MAGIC * ✓ **Deep understanding** - Explained bias vs variance with examples
# MAGIC * ✓ **Multiple solutions** - Listed 8 techniques with priorities
# MAGIC * ✓ **Systematic approach** - Learning curves for data-driven decisions
# MAGIC * ✓ **Context awareness** - When to use each technique
# MAGIC * ✓ **Monitoring plan** - Validation strategy
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 What You Built in This Notebook
# MAGIC
# MAGIC 1. ✅ **3 Complete Scenarios** with real code
# MAGIC    * High Bias: Train 60%, Test 58%
# MAGIC    * High Variance: Train 99%, Test 72%
# MAGIC    * Good Fit: Train 87%, Test 84%
# MAGIC
# MAGIC 2. ✅ **All 8 Generalization Techniques** with working examples
# MAGIC    * More data
# MAGIC    * Regularization (L1/L2)
# MAGIC    * Cross-validation
# MAGIC    * Early stopping
# MAGIC    * Dropout
# MAGIC    * Feature selection
# MAGIC    * Reduce complexity
# MAGIC    * Data augmentation
# MAGIC
# MAGIC 3. ✅ **Learning Curves** - Complete implementation
# MAGIC
# MAGIC 4. ✅ **Visual Diagnosis** - Charts for every concept
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎓 You're Ready When...
# MAGIC
# MAGIC * [ ] You can recreate the 3 scenarios from memory
# MAGIC * [ ] You can list all 8 techniques without looking
# MAGIC * [ ] You can explain when to collect data vs simplify
# MAGIC * [ ] You can interpret learning curves instantly
# MAGIC * [ ] You can give the complete answer in under 3 minutes
# MAGIC
# MAGIC **🎉 If you checked all boxes, you're interview-ready!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 Next Steps
# MAGIC
# MAGIC 1. **Run all cells** in this notebook to see results
# MAGIC 2. **Practice saying your answer out loud** (3 times minimum)
# MAGIC 3. **Modify parameters** to see how results change
# MAGIC 4. **Explain to someone else** (best way to internalize)
# MAGIC 5. **Bookmark this notebook** for pre-interview review
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC *Good luck with your interviews! You've got this! 💪*