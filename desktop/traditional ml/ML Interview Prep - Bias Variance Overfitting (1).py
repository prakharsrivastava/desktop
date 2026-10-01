# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # 🎯 ML Interview Prep: Bias-Variance Tradeoff & Generalization
# MAGIC
# MAGIC ## The Classic Interview Question
# MAGIC
# MAGIC > **Interviewer:** "Your model has 99% training accuracy but only 72% test accuracy. What's happening and how would you fix it?"
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## What This Notebook Covers
# MAGIC
# MAGIC 1. ✅ **Identifying the Problem** - Overfitting vs Underfitting
# MAGIC 2. 📊 **Bias vs Variance** - Complete explanation with examples
# MAGIC 3. 🛠️ **Generalization Techniques** - 8 practical solutions
# MAGIC 4. 📈 **Learning Curves** - Visual diagnosis
# MAGIC 5. 🤔 **Decision Framework** - When to collect data vs simplify model
# MAGIC 6. 💪 **Strong Interview Answer** - What senior candidates say
# MAGIC 7. 🎮 **Interactive Examples** - Explore with real data
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Your Goal
# MAGIC
# MAGIC By the end of this notebook, you'll be able to:
# MAGIC * Instantly recognize overfitting from train/test accuracy gaps
# MAGIC * Explain bias vs variance like a senior ML engineer
# MAGIC * List and apply 8+ generalization techniques
# MAGIC * Analyze learning curves to guide decisions
# MAGIC * Give confident, comprehensive interview answers

# COMMAND ----------

# DBTITLE 1,Setup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

print("✓ Libraries loaded successfully")

# COMMAND ----------

# DBTITLE 1,The Interview Scenario
# MAGIC %md
# MAGIC ## 📝 The Interview Question
# MAGIC
# MAGIC ### Scenario
# MAGIC
# MAGIC ```
# MAGIC Your trained model shows:
# MAGIC   Training Accuracy: 99%
# MAGIC   Test Accuracy:     72%
# MAGIC   
# MAGIC Question: What's happening? How do you fix it?
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚡ Quick Answer (What Interviewer Wants to Hear First)
# MAGIC
# MAGIC **"The model is overfitting. It has high variance."**
# MAGIC
# MAGIC ✅ This immediately shows you understand the problem.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚫 Incomplete Answer (Don't Stop Here!)
# MAGIC
# MAGIC > "Bias means underfitting, variance means overfitting."
# MAGIC
# MAGIC This is **correct but shallow**. Senior candidates go deeper...

# COMMAND ----------

# DBTITLE 1,Complete Explanation - Bias vs Variance
# MAGIC %md
# MAGIC ## 🎓 Complete Explanation: Bias vs Variance
# MAGIC
# MAGIC ### 🔴 HIGH BIAS (Underfitting)
# MAGIC
# MAGIC **What it means:**
# MAGIC * Model is **too simple**
# MAGIC * Cannot learn underlying patterns
# MAGIC * Makes wrong assumptions about data
# MAGIC
# MAGIC **Symptoms:**
# MAGIC ```
# MAGIC Train Accuracy: LOW  (e.g., 60%)
# MAGIC Test Accuracy:  LOW  (e.g., 58%)
# MAGIC
# MAGIC Both are bad!
# MAGIC ```
# MAGIC
# MAGIC **Examples:**
# MAGIC * Using linear regression for non-linear data
# MAGIC * Using 1-layer neural network for complex images
# MAGIC * Too much regularization
# MAGIC
# MAGIC **Visual analogy:** 🎯 Arrows grouped together but missing the target → **Systematic error**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔵 HIGH VARIANCE (Overfitting)
# MAGIC
# MAGIC **What it means:**
# MAGIC * Model is **too complex**
# MAGIC * **Memorizes** training data instead of learning patterns
# MAGIC * Cannot generalize to new data
# MAGIC
# MAGIC **Symptoms:**
# MAGIC ```
# MAGIC Train Accuracy: HIGH (e.g., 99%)
# MAGIC Test Accuracy:  LOW  (e.g., 72%)
# MAGIC
# MAGIC Large gap = Overfitting!
# MAGIC ```
# MAGIC
# MAGIC **Examples:**
# MAGIC * Deep neural network with millions of parameters on small dataset
# MAGIC * Decision tree with no depth limit
# MAGIC * No regularization on complex model
# MAGIC
# MAGIC **Visual analogy:** 🎯 Arrows scattered all over → **Random error, unstable**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 The Goldilocks Zone
# MAGIC
# MAGIC ```
# MAGIC Train Accuracy: ~85%
# MAGIC Test Accuracy:  ~82%
# MAGIC
# MAGIC Small gap = Good generalization!
# MAGIC ```
# MAGIC
# MAGIC **This is what you're aiming for.**

# COMMAND ----------

# DBTITLE 1,Visualize Bias vs Variance
# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y_true = 2 * X.ravel() + 3 * np.sin(X.ravel()) + np.random.randn(100) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)

# Three models: Underfitting, Good fit, Overfitting
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# 1. High Bias (Underfitting) - Simple linear model
model_underfit = LinearRegression()
model_underfit.fit(X_train, y_train)

# 2. Good Fit - Polynomial degree 3
poly_good = PolynomialFeatures(degree=3)
X_train_poly_good = poly_good.fit_transform(X_train)
X_test_poly_good = poly_good.transform(X_test)
model_good = Ridge(alpha=1.0)
model_good.fit(X_train_poly_good, y_train)

# 3. High Variance (Overfitting) - Deep decision tree
model_overfit = DecisionTreeRegressor(max_depth=10, random_state=42)
model_overfit.fit(X_train, y_train)

# Predictions
y_pred_under_train = model_underfit.predict(X_train)
y_pred_under_test = model_underfit.predict(X_test)

y_pred_good_train = model_good.predict(X_train_poly_good)
y_pred_good_test = model_good.predict(X_test_poly_good)

y_pred_over_train = model_overfit.predict(X_train)
y_pred_over_test = model_overfit.predict(X_test)

# Calculate R² scores (accuracy equivalent for regression)
from sklearn.metrics import r2_score

scores = {
    'High Bias\n(Underfit)': {
        'train': r2_score(y_train, y_pred_under_train),
        'test': r2_score(y_test, y_pred_under_test)
    },
    'Good Fit\n(Balanced)': {
        'train': r2_score(y_train, y_pred_good_train),
        'test': r2_score(y_test, y_pred_good_test)
    },
    'High Variance\n(Overfit)': {
        'train': r2_score(y_train, y_pred_over_train),
        'test': r2_score(y_test, y_pred_over_test)
    }
}

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot decision boundaries
X_plot = np.linspace(0, 10, 300).reshape(-1, 1)

# Underfitting
ax = axes[0, 0]
y_plot_under = model_underfit.predict(X_plot)
ax.scatter(X_train, y_train, alpha=0.6, s=30, label='Train', color='blue')
ax.scatter(X_test, y_test, alpha=0.6, s=30, label='Test', color='red')
ax.plot(X_plot, y_plot_under, 'g-', lw=2, label='Model')
ax.set_title('HIGH BIAS (Underfitting)\nToo Simple', fontsize=12, fontweight='bold', color='darkred')
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.grid(alpha=0.3)

# Good fit
ax = axes[0, 1]
X_plot_poly = poly_good.transform(X_plot)
y_plot_good = model_good.predict(X_plot_poly)
ax.scatter(X_train, y_train, alpha=0.6, s=30, label='Train', color='blue')
ax.scatter(X_test, y_test, alpha=0.6, s=30, label='Test', color='red')
ax.plot(X_plot, y_plot_good, 'g-', lw=2, label='Model')
ax.set_title('GOOD FIT (Balanced)\nJust Right ✓', fontsize=12, fontweight='bold', color='darkgreen')
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.grid(alpha=0.3)

# Overfitting
ax = axes[0, 2]
y_plot_over = model_overfit.predict(X_plot)
ax.scatter(X_train, y_train, alpha=0.6, s=30, label='Train', color='blue')
ax.scatter(X_test, y_test, alpha=0.6, s=30, label='Test', color='red')
ax.plot(X_plot, y_plot_over, 'g-', lw=2, label='Model')
ax.set_title('HIGH VARIANCE (Overfitting)\nToo Complex', fontsize=12, fontweight='bold', color='darkorange')
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.grid(alpha=0.3)

# Accuracy comparison
ax = axes[1, :]
ax = axes[1, 0]
ax.axis('off')
ax = axes[1, 1]
ax.axis('off')
ax = axes[1, 2]
ax.axis('off')

ax_main = fig.add_subplot(2, 1, 2)
models = list(scores.keys())
train_scores = [scores[m]['train'] * 100 for m in models]
test_scores = [scores[m]['test'] * 100 for m in models]

x_pos = np.arange(len(models))
width = 0.35

bars1 = ax_main.bar(x_pos - width/2, train_scores, width, label='Train Accuracy', color='steelblue', edgecolor='black')
bars2 = ax_main.bar(x_pos + width/2, test_scores, width, label='Test Accuracy', color='coral', edgecolor='black')

ax_main.set_ylabel('Accuracy (R² Score %)', fontsize=11, fontweight='bold')
ax_main.set_title('Train vs Test Accuracy Comparison', fontsize=13, fontweight='bold')
ax_main.set_xticks(x_pos)
ax_main.set_xticklabels(models)
ax_main.legend()
ax_main.grid(alpha=0.3, axis='y')
ax_main.axhline(0, color='black', linewidth=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax_main.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add gap annotations
for i, model in enumerate(models):
    gap = train_scores[i] - test_scores[i]
    color = 'darkred' if gap > 15 else ('darkgreen' if gap < 8 else 'orange')
    ax_main.text(i, min(train_scores[i], test_scores[i]) - 8,
                f'Gap: {gap:.1f}%',
                ha='center', fontsize=9, color=color, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n📊 Analysis:")
print("="*60)
for model, score in scores.items():
    gap = (score['train'] - score['test']) * 100
    print(f"\n{model}:")
    print(f"  Train: {score['train']*100:.1f}%")
    print(f"  Test:  {score['test']*100:.1f}%")
    print(f"  Gap:   {gap:.1f}%", end="")
    
    if 'Bias' in model:
        print(" ← Both low (can't learn patterns)")
    elif 'Good' in model:
        print(" ← Small gap (good generalization) ✓")
    elif 'Variance' in model:
        print(" ← Large gap (memorizing training data)")
print("="*60)

# COMMAND ----------

# DBTITLE 1,8 Generalization Techniques
# MAGIC %md
# MAGIC ## 🛠️ How to Improve Generalization (Fix Overfitting)
# MAGIC
# MAGIC ### ⚠️ What the Interviewer is Testing
# MAGIC
# MAGIC When they ask **"How would you fix it?"**, they want to see:
# MAGIC 1. Do you know **multiple** techniques?
# MAGIC 2. Can you explain **when** to use each?
# MAGIC 3. Do you understand the **tradeoffs**?
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💪 The 8 Core Techniques
# MAGIC
# MAGIC ### 1️⃣ Collect More Training Data
# MAGIC
# MAGIC **Why it works:**
# MAGIC * More examples → harder to memorize all of them
# MAGIC * Model forced to learn general patterns
# MAGIC
# MAGIC **When to use:**
# MAGIC * You have access to more data
# MAGIC * Learning curves show data would help (we'll visualize this!)
# MAGIC * Model complexity is appropriate
# MAGIC
# MAGIC **Tradeoff:**
# MAGIC * 👍 Most effective solution
# MAGIC * 👎 Expensive, time-consuming
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2️⃣ Regularization (L1/L2)
# MAGIC
# MAGIC **Why it works:**
# MAGIC * Adds penalty for large weights
# MAGIC * Forces model to use simpler patterns
# MAGIC
# MAGIC **L2 (Ridge):** `Loss = MSE + λ · Σ(weights²)`
# MAGIC * Shrinks all weights
# MAGIC * Doesn't eliminate features
# MAGIC
# MAGIC **L1 (Lasso):** `Loss = MSE + λ · Σ|weights|`
# MAGIC * Can zero out weights
# MAGIC * Performs feature selection
# MAGIC
# MAGIC **When to use:**
# MAGIC * Model has too many parameters
# MAGIC * You suspect some features are noise
# MAGIC
# MAGIC **Tradeoff:**
# MAGIC * 👍 Easy to implement
# MAGIC * 👎 Need to tune λ (hyperparameter)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3️⃣ Cross-Validation
# MAGIC
# MAGIC **Why it works:**
# MAGIC * Uses all data for both training and validation
# MAGIC * Gives robust estimate of generalization
# MAGIC
# MAGIC **K-Fold CV:** Split data into K folds, train K times
# MAGIC
# MAGIC **When to use:**
# MAGIC * Small dataset (can't afford separate val set)
# MAGIC * Hyperparameter tuning
# MAGIC * Model selection
# MAGIC
# MAGIC **Tradeoff:**
# MAGIC * 👍 Robust evaluation
# MAGIC * 👎 K times slower training
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4️⃣ Early Stopping
# MAGIC
# MAGIC **Why it works:**
# MAGIC * Stop training when validation loss starts increasing
# MAGIC * Prevents model from memorizing training set
# MAGIC
# MAGIC **When to use:**
# MAGIC * Training neural networks
# MAGIC * Iterative algorithms (gradient descent)
# MAGIC
# MAGIC **Tradeoff:**
# MAGIC * 👍 Free (just monitor val loss)
# MAGIC * 👎 Need validation set
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5️⃣ Dropout (Neural Networks)
# MAGIC
# MAGIC **Why it works:**
# MAGIC * Randomly drops neurons during training
# MAGIC * Prevents neurons from co-adapting
# MAGIC * Forces redundant representations
# MAGIC
# MAGIC **When to use:**
# MAGIC * Deep neural networks
# MAGIC * Fully connected layers
# MAGIC
# MAGIC **Tradeoff:**
# MAGIC * 👍 Very effective for NNs
# MAGIC * 👎 Only applicable to neural networks
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 6️⃣ Feature Selection
# MAGIC
# MAGIC **Why it works:**
# MAGIC * Removes irrelevant/noisy features
# MAGIC * Reduces model complexity
# MAGIC * Improves interpretability
# MAGIC
# MAGIC **Methods:**
# MAGIC * Filter: Correlation, mutual information
# MAGIC * Wrapper: Forward/backward selection
# MAGIC * Embedded: L1 regularization, tree importances
# MAGIC
# MAGIC **When to use:**
# MAGIC * High-dimensional data
# MAGIC * Many irrelevant features
# MAGIC
# MAGIC **Tradeoff:**
# MAGIC * 👍 Simpler, faster model
# MAGIC * 👎 May lose useful interactions
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 7️⃣ Reduce Model Complexity
# MAGIC
# MAGIC **Why it works:**
# MAGIC * Simpler model = less capacity to memorize
# MAGIC
# MAGIC **Examples:**
# MAGIC * Fewer layers in neural network
# MAGIC * Shallower decision trees
# MAGIC * Lower polynomial degree
# MAGIC * Fewer hidden units
# MAGIC
# MAGIC **When to use:**
# MAGIC * Model is obviously too complex for problem size
# MAGIC * Dataset is small
# MAGIC
# MAGIC **Tradeoff:**
# MAGIC * 👍 Faster training/inference
# MAGIC * 👎 May underfit if too simple
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 8️⃣ Data Augmentation
# MAGIC
# MAGIC **Why it works:**
# MAGIC * Artificially increases dataset size
# MAGIC * Teaches invariances (rotation, scaling, etc.)
# MAGIC
# MAGIC **Examples:**
# MAGIC * Images: Flip, rotate, crop, color jitter
# MAGIC * Text: Synonym replacement, back-translation
# MAGIC * Audio: Pitch shift, time stretch
# MAGIC
# MAGIC **When to use:**
# MAGIC * Image/audio/text problems
# MAGIC * Limited training data
# MAGIC * Known invariances exist
# MAGIC
# MAGIC **Tradeoff:**
# MAGIC * 👍 Effective for certain domains
# MAGIC * 👎 Domain-specific, requires expertise
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Summary Table
# MAGIC
# MAGIC | Technique | Effectiveness | Ease | Domain |
# MAGIC |-----------|--------------|------|--------|
# MAGIC | More Data | ⭐⭐⭐⭐⭐ | 👎 Hard | All |
# MAGIC | Regularization | ⭐⭐⭐⭐ | 👍 Easy | All |
# MAGIC | Cross-Validation | ⭐⭐⭐⭐ | 👍 Easy | All |
# MAGIC | Early Stopping | ⭐⭐⭐⭐ | 👍 Easy | Iterative |
# MAGIC | Dropout | ⭐⭐⭐⭐⭐ | 👍 Easy | Neural Nets |
# MAGIC | Feature Selection | ⭐⭐⭐ | 👍 Medium | High-dim |
# MAGIC | Simplify Model | ⭐⭐⭐ | 👍 Easy | All |
# MAGIC | Data Augmentation | ⭐⭐⭐⭐ | 👎 Hard | Vision/NLP |

# COMMAND ----------

# DBTITLE 1,Demonstrate Regularization Effect
# Demonstrate regularization reducing overfitting
np.random.seed(42)

# Generate data with noise
X_reg = np.linspace(0, 10, 50).reshape(-1, 1)
y_reg = 2 * X_reg.ravel() + np.sin(2 * X_reg.ravel()) + np.random.randn(50) * 0.5

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Create polynomial features (degree 15 = very complex)
poly_15 = PolynomialFeatures(degree=15)
X_train_poly15 = poly_15.fit_transform(X_train_reg)
X_test_poly15 = poly_15.transform(X_test_reg)

# Three models with different regularization
from sklearn.linear_model import LinearRegression

models_reg = {
    'No Regularization\n(λ=0)': LinearRegression(),
    'Light Regularization\n(λ=1)': Ridge(alpha=1.0),
    'Strong Regularization\n(λ=100)': Ridge(alpha=100.0)
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

X_plot_reg = np.linspace(0, 10, 300).reshape(-1, 1)
X_plot_poly15 = poly_15.transform(X_plot_reg)

results_reg = {}

for idx, (name, model) in enumerate(models_reg.items()):
    # Train
    model.fit(X_train_poly15, y_train_reg)
    
    # Predict
    y_pred_train = model.predict(X_train_poly15)
    y_pred_test = model.predict(X_test_poly15)
    y_pred_plot = model.predict(X_plot_poly15)
    
    # Scores
    train_r2 = r2_score(y_train_reg, y_pred_train) * 100
    test_r2 = r2_score(y_test_reg, y_pred_test) * 100
    gap = train_r2 - test_r2
    
    results_reg[name] = {'train': train_r2, 'test': test_r2, 'gap': gap}
    
    # Plot
    ax = axes[0, idx]
    ax.scatter(X_train_reg, y_train_reg, alpha=0.6, s=40, label='Train', color='blue')
    ax.scatter(X_test_reg, y_test_reg, alpha=0.6, s=40, label='Test', color='red')
    ax.plot(X_plot_reg, y_pred_plot, 'g-', lw=2, label='Model')
    
    color = 'darkred' if gap > 20 else ('darkgreen' if gap < 10 else 'orange')
    ax.set_title(f'{name}\nTrain: {train_r2:.1f}% | Test: {test_r2:.1f}% | Gap: {gap:.1f}%',
                fontsize=10, fontweight='bold', color=color)
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.grid(alpha=0.3)
    ax.set_ylim(-5, 25)

# Comparison bar chart
ax_bar = fig.add_subplot(2, 1, 2)
model_names = list(results_reg.keys())
train_vals = [results_reg[m]['train'] for m in model_names]
test_vals = [results_reg[m]['test'] for m in model_names]
gap_vals = [results_reg[m]['gap'] for m in model_names]

x_pos = np.arange(len(model_names))
width = 0.35

bars1 = ax_bar.bar(x_pos - width/2, train_vals, width, label='Train R²', color='steelblue', edgecolor='black')
bars2 = ax_bar.bar(x_pos + width/2, test_vals, width, label='Test R²', color='coral', edgecolor='black')

ax_bar.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
ax_bar.set_title('Effect of Regularization on Overfitting', fontsize=13, fontweight='bold')
ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(model_names)
ax_bar.legend()
ax_bar.grid(alpha=0.3, axis='y')

# Add gap annotations
for i, gap in enumerate(gap_vals):
    color = 'darkred' if gap > 20 else ('darkgreen' if gap < 10 else 'orange')
    ax_bar.text(i, min(train_vals[i], test_vals[i]) - 10,
                f'Gap: {gap:.1f}%',
                ha='center', fontsize=9, color=color, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n📈 Regularization Analysis:")
print("="*60)
for name, scores in results_reg.items():
    print(f"\n{name}")
    print(f"  Train: {scores['train']:.1f}%")
    print(f"  Test:  {scores['test']:.1f}%")
    print(f"  Gap:   {scores['gap']:.1f}%", end="")
    
    if scores['gap'] > 20:
        print(" ← Overfitting!")
    elif scores['gap'] < 10:
        print(" ← Good generalization ✓")
    else:
        print(" ← Moderate overfitting")

print("\n" + "="*60)
print("✅ Key Insight: Regularization reduces the gap!")
print("   → Strong regularization (λ=100) gives best generalization")
print("   → But too much can cause underfitting (both scores drop)")
print("   → Need to tune λ on validation set")

# COMMAND ----------

# DBTITLE 1,Learning Curves - Visual Diagnosis
# MAGIC %md
# MAGIC ## 📈 Learning Curves: Your Diagnostic Tool
# MAGIC
# MAGIC ### What are Learning Curves?
# MAGIC
# MAGIC Plots showing train/test performance vs **training set size**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔍 Pattern 1: High Variance (Overfitting)
# MAGIC
# MAGIC ```
# MAGIC      |
# MAGIC 100% |     ========= Train (high, flat)
# MAGIC      |
# MAGIC  70% |  ____--------- Test (low, rising slowly)
# MAGIC      |
# MAGIC      +---------------------->
# MAGIC            Training Set Size
# MAGIC ```
# MAGIC
# MAGIC **What this tells you:**
# MAGIC * ✅ **Collecting more data WILL help**
# MAGIC * Train-test gap is large
# MAGIC * Test curve is still rising → hasn't plateaued
# MAGIC
# MAGIC **Action:** Get more training data
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔍 Pattern 2: High Bias (Underfitting)
# MAGIC
# MAGIC ```
# MAGIC      |
# MAGIC  65% |  ____--------- Train (low, plateaus quickly)
# MAGIC      |
# MAGIC  60% | ____---------- Test (low, plateaus quickly)
# MAGIC      |
# MAGIC      +---------------------->
# MAGIC            Training Set Size
# MAGIC ```
# MAGIC
# MAGIC **What this tells you:**
# MAGIC * ❌ **More data WON'T help**
# MAGIC * Both curves are low and flat
# MAGIC * Model can't learn patterns even with more data
# MAGIC
# MAGIC **Action:** Increase model complexity (more features, deeper model)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔍 Pattern 3: Good Fit
# MAGIC
# MAGIC ```
# MAGIC      |
# MAGIC  85% |  ____--------- Train (high, converged)
# MAGIC      |
# MAGIC  82% | ____---------- Test (close to train)
# MAGIC      |
# MAGIC      +---------------------->
# MAGIC            Training Set Size
# MAGIC ```
# MAGIC
# MAGIC **What this tells you:**
# MAGIC * ✓ Model is well-tuned
# MAGIC * Small train-test gap
# MAGIC * Both curves have converged
# MAGIC
# MAGIC **Action:** You're done! (Or try minor improvements)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Interview Tip
# MAGIC
# MAGIC When asked **"Should you collect more data or simplify the model?"**
# MAGIC
# MAGIC **Answer:**
# MAGIC > "I would plot learning curves. If the test curve is still rising and there's a large train-test gap, more data will help. If both curves are low and plateaued, the model is too simple and I need to increase complexity, not collect more data."

# COMMAND ----------

# DBTITLE 1,Generate Learning Curves
# Generate learning curves for three scenarios
from sklearn.model_selection import learning_curve

# Prepare data
np.random.seed(42)
X_lc = np.random.randn(500, 5)
y_lc = 2 * X_lc[:, 0] + 3 * X_lc[:, 1] - X_lc[:, 2]**2 + np.random.randn(500) * 0.3

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Scenario 1: High Variance (Overfitting) - Complex model
model_overfit_lc = Ridge(alpha=0.0001)  # Almost no regularization
poly_high = PolynomialFeatures(degree=10)
X_lc_poly = poly_high.fit_transform(X_lc)

train_sizes, train_scores_over, test_scores_over = learning_curve(
    model_overfit_lc, X_lc_poly, y_lc,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='r2', n_jobs=-1
)

train_mean_over = train_scores_over.mean(axis=1) * 100
test_mean_over = test_scores_over.mean(axis=1) * 100

ax = axes[0]
ax.plot(train_sizes, train_mean_over, 'o-', color='blue', label='Train Score', linewidth=2, markersize=6)
ax.plot(train_sizes, test_mean_over, 'o-', color='red', label='Test Score', linewidth=2, markersize=6)
ax.fill_between(train_sizes,
                train_scores_over.mean(axis=1) * 100 - train_scores_over.std(axis=1) * 100,
                train_scores_over.mean(axis=1) * 100 + train_scores_over.std(axis=1) * 100,
                alpha=0.1, color='blue')
ax.fill_between(train_sizes,
                test_scores_over.mean(axis=1) * 100 - test_scores_over.std(axis=1) * 100,
                test_scores_over.mean(axis=1) * 100 + test_scores_over.std(axis=1) * 100,
                alpha=0.1, color='red')
ax.set_xlabel('Training Set Size', fontsize=11)
ax.set_ylabel('Score (R² %)', fontsize=11)
ax.set_title('HIGH VARIANCE (Overfitting)\n✅ More data will help!', fontsize=12, fontweight='bold', color='darkorange')
ax.legend(loc='best')
ax.grid(alpha=0.3)
ax.set_ylim(0, 105)

# Add annotation
gap_final_over = train_mean_over[-1] - test_mean_over[-1]
ax.annotate(f'Gap: {gap_final_over:.1f}%\nTest rising →\nMore data helps!',
           xy=(train_sizes[-1], test_mean_over[-1]),
           xytext=(train_sizes[-3], test_mean_over[-1] - 15),
           arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
           fontsize=9, color='darkred', fontweight='bold')

# Scenario 2: High Bias (Underfitting) - Too simple model
model_underfit_lc = LinearRegression()

train_sizes, train_scores_under, test_scores_under = learning_curve(
    model_underfit_lc, X_lc,  y_lc,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='r2', n_jobs=-1
)

train_mean_under = train_scores_under.mean(axis=1) * 100
test_mean_under = test_scores_under.mean(axis=1) * 100

ax = axes[1]
ax.plot(train_sizes, train_mean_under, 'o-', color='blue', label='Train Score', linewidth=2, markersize=6)
ax.plot(train_sizes, test_mean_under, 'o-', color='red', label='Test Score', linewidth=2, markersize=6)
ax.fill_between(train_sizes,
                train_scores_under.mean(axis=1) * 100 - train_scores_under.std(axis=1) * 100,
                train_scores_under.mean(axis=1) * 100 + train_scores_under.std(axis=1) * 100,
                alpha=0.1, color='blue')
ax.fill_between(train_sizes,
                test_scores_under.mean(axis=1) * 100 - test_scores_under.std(axis=1) * 100,
                test_scores_under.mean(axis=1) * 100 + test_scores_under.std(axis=1) * 100,
                alpha=0.1, color='red')
ax.set_xlabel('Training Set Size', fontsize=11)
ax.set_ylabel('Score (R² %)', fontsize=11)
ax.set_title('HIGH BIAS (Underfitting)\n❌ More data won\'t help!', fontsize=12, fontweight='bold', color='darkred')
ax.legend(loc='best')
ax.grid(alpha=0.3)
ax.set_ylim(0, 105)

# Add annotation
ax.annotate('Both low & flat\n→ Need more\ncomplex model!',
           xy=(train_sizes[-1], train_mean_under[-1]),
           xytext=(train_sizes[-3], train_mean_under[-1] + 15),
           arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
           fontsize=9, color='darkred', fontweight='bold')

# Scenario 3: Good Fit - Well-tuned model
model_good_lc = Ridge(alpha=10.0)
poly_good_lc = PolynomialFeatures(degree=3)
X_lc_poly_good = poly_good_lc.fit_transform(X_lc)

train_sizes, train_scores_good, test_scores_good = learning_curve(
    model_good_lc, X_lc_poly_good, y_lc,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='r2', n_jobs=-1
)

train_mean_good = train_scores_good.mean(axis=1) * 100
test_mean_good = test_scores_good.mean(axis=1) * 100

ax = axes[2]
ax.plot(train_sizes, train_mean_good, 'o-', color='blue', label='Train Score', linewidth=2, markersize=6)
ax.plot(train_sizes, test_mean_good, 'o-', color='red', label='Test Score', linewidth=2, markersize=6)
ax.fill_between(train_sizes,
                train_scores_good.mean(axis=1) * 100 - train_scores_good.std(axis=1) * 100,
                train_scores_good.mean(axis=1) * 100 + train_scores_good.std(axis=1) * 100,
                alpha=0.1, color='blue')
ax.fill_between(train_sizes,
                test_scores_good.mean(axis=1) * 100 - test_scores_good.std(axis=1) * 100,
                test_scores_good.mean(axis=1) * 100 + test_scores_good.std(axis=1) * 100,
                alpha=0.1, color='red')
ax.set_xlabel('Training Set Size', fontsize=11)
ax.set_ylabel('Score (R² %)', fontsize=11)
ax.set_title('GOOD FIT (Balanced)\n✓ Well-tuned model!', fontsize=12, fontweight='bold', color='darkgreen')
ax.legend(loc='best')
ax.grid(alpha=0.3)
ax.set_ylim(0, 105)

# Add annotation
gap_final_good = train_mean_good[-1] - test_mean_good[-1]
ax.annotate(f'Small gap: {gap_final_good:.1f}%\nBoth converged\n→ Well-tuned!',
           xy=(train_sizes[-1], test_mean_good[-1]),
           xytext=(train_sizes[-4], test_mean_good[-1] - 20),
           arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
           fontsize=9, color='darkgreen', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n📊 Learning Curves Analysis:")
print("="*70)
print("\n1️⃣ HIGH VARIANCE (Overfitting):")
print(f"   Final Train Score: {train_mean_over[-1]:.1f}%")
print(f"   Final Test Score:  {test_mean_over[-1]:.1f}%")
print(f"   Gap: {gap_final_over:.1f}%")
print("   ✅ Decision: Collect more data (test curve still rising)")

print("\n2️⃣ HIGH BIAS (Underfitting):")
print(f"   Final Train Score: {train_mean_under[-1]:.1f}%")
print(f"   Final Test Score:  {test_mean_under[-1]:.1f}%")
print(f"   Gap: {train_mean_under[-1] - test_mean_under[-1]:.1f}%")
print("   ❌ Decision: More data won't help (both curves low & flat)")
print("   ✅ Decision: Increase model complexity instead")

print("\n3️⃣ GOOD FIT (Balanced):")
print(f"   Final Train Score: {train_mean_good[-1]:.1f}%")
print(f"   Final Test Score:  {test_mean_good[-1]:.1f}%")
print(f"   Gap: {gap_final_good:.1f}%")
print("   ✓ Decision: Model is well-tuned, you're done!")
print("="*70)

# COMMAND ----------

# DBTITLE 1,Decision Framework
# MAGIC %md
# MAGIC ## 🤔 Decision Framework: More Data vs Simplify Model
# MAGIC
# MAGIC ### The Million Dollar Question
# MAGIC
# MAGIC > "Should I collect more data or simplify my model?"
# MAGIC
# MAGIC This is where **senior candidates shine**. Junior candidates guess. Senior candidates **use data to decide**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 The Decision Tree
# MAGIC
# MAGIC ```
# MAGIC Start: Model is overfitting (high train, low test)
# MAGIC     |
# MAGIC     v
# MAGIC Plot Learning Curves
# MAGIC     |
# MAGIC     +---> Test curve still rising?
# MAGIC     |         |
# MAGIC     |         +---> YES → ✅ COLLECT MORE DATA
# MAGIC     |         |           (test hasn't plateaued,
# MAGIC     |         |            more data will help)
# MAGIC     |         |
# MAGIC     |         +---> NO  → Test curve has plateaued?
# MAGIC     |                   |
# MAGIC     |                   +---> YES → ❌ DON'T collect more data
# MAGIC     |                   |           ✅ Simplify model OR
# MAGIC     |                   |           ✅ Add regularization
# MAGIC     |                   |
# MAGIC     |                   +---> NO  → Need more samples
# MAGIC     |                              to see pattern
# MAGIC     |
# MAGIC     +---> Large train-test gap?
# MAGIC               |
# MAGIC               +---> YES (>15%) → High variance
# MAGIC               |                  ✅ Regularization first
# MAGIC               |                  ✅ Then try more data
# MAGIC               |
# MAGIC               +---> NO (<8%)   → ✓ Good fit
# MAGIC                                    Minor tuning only
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📝 Detailed Decision Logic
# MAGIC
# MAGIC ### Scenario 1: Test Curve Rising + Large Gap
# MAGIC
# MAGIC **Symptoms:**
# MAGIC * Train: 99%, Test: 72%
# MAGIC * Test curve hasn't flattened
# MAGIC * Gap is large (27%)
# MAGIC
# MAGIC **Diagnosis:** High Variance (Overfitting)
# MAGIC
# MAGIC **Solution Priority:**
# MAGIC 1. **Collect more data** (most effective)
# MAGIC 2. Add regularization
# MAGIC 3. Cross-validation
# MAGIC 4. Early stopping
# MAGIC
# MAGIC **Why:** Model is capable but needs more examples to generalize.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Scenario 2: Both Curves Low + Plateaued
# MAGIC
# MAGIC **Symptoms:**
# MAGIC * Train: 65%, Test: 62%
# MAGIC * Both curves are flat
# MAGIC * Small gap (3%)
# MAGIC
# MAGIC **Diagnosis:** High Bias (Underfitting)
# MAGIC
# MAGIC **Solution Priority:**
# MAGIC 1. **Increase model complexity** (add features, layers, polynomial degree)
# MAGIC 2. Reduce regularization
# MAGIC 3. Feature engineering
# MAGIC 4. Try different model architecture
# MAGIC
# MAGIC **Why:** Model is too simple. More data won't fix a fundamentally weak model.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Scenario 3: Test Curve Plateaued + Large Gap
# MAGIC
# MAGIC **Symptoms:**
# MAGIC * Train: 98%, Test: 75%
# MAGIC * Test curve is flat
# MAGIC * Gap is large (23%)
# MAGIC
# MAGIC **Diagnosis:** Model too complex for available data
# MAGIC
# MAGIC **Solution Priority:**
# MAGIC 1. **Simplify model** (reduce parameters)
# MAGIC 2. **Strong regularization**
# MAGIC 3. Feature selection
# MAGIC 4. Dropout (if neural network)
# MAGIC
# MAGIC **Why:** Test curve flat means more data won't help. Model is memorizing noise.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Scenario 4: Small Gap + High Scores
# MAGIC
# MAGIC **Symptoms:**
# MAGIC * Train: 87%, Test: 84%
# MAGIC * Both curves converged
# MAGIC * Small gap (3%)
# MAGIC
# MAGIC **Diagnosis:** Good fit!
# MAGIC
# MAGIC **Solution:**
# MAGIC * ✓ Model is well-tuned
# MAGIC * Minor tweaks only (ensemble, fine-tune hyperparameters)
# MAGIC * Consider this your baseline
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ❗ Common Interview Mistakes
# MAGIC
# MAGIC ### Mistake 1: "Just collect more data"
# MAGIC **Why wrong:** Assumes more data always helps. Not true for high bias!
# MAGIC
# MAGIC ### Mistake 2: "Simplify the model"
# MAGIC **Why wrong:** If test curve is rising, you're throwing away capacity that could be used with more data.
# MAGIC
# MAGIC ### Mistake 3: "Try everything"
# MAGIC **Why wrong:** Shows lack of systematic thinking. Use learning curves to guide you.
# MAGIC
# MAGIC ### Mistake 4: "Check accuracy"
# MAGIC **Why wrong:** Accuracy alone doesn't tell you whether it's bias or variance. Need the gap!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 What Interviewers Want to Hear
# MAGIC
# MAGIC > "I would first plot learning curves to diagnose whether we have high bias or high variance. If the test curve is still rising and there's a large gap, more training data will help close that gap. If both curves have plateaued at a low value with a small gap, the model is too simple and needs more capacity. If the test curve has plateaued but there's still a large gap, the model is too complex and needs regularization or simplification."
# MAGIC
# MAGIC This shows:
# MAGIC * ✓ Systematic approach
# MAGIC * ✓ Data-driven decision
# MAGIC * ✓ Understanding of tradeoffs
# MAGIC * ✓ Clear reasoning

# COMMAND ----------

# DBTITLE 1,Strong Interview Answer Template
# MAGIC %md
# MAGIC ## 💪 The Perfect Interview Answer
# MAGIC
# MAGIC ### The Complete Response
# MAGIC
# MAGIC When asked: **"Your model has 99% train accuracy and 72% test accuracy. What's happening and how do you fix it?"**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🏆 SENIOR-LEVEL ANSWER
# MAGIC
# MAGIC > **Part 1: Identify the Problem**
# MAGIC >
# MAGIC > "The model is **overfitting** — it has **high variance**. The large gap between training (99%) and test (72%) accuracy means the model is memorizing the training data rather than learning generalizable patterns.
# MAGIC
# MAGIC > **Part 2: Explain Concepts**
# MAGIC >
# MAGIC > High variance occurs when a model is too complex relative to the amount of training data, causing it to fit noise instead of signal. This is different from high bias, where a model is too simple and both train and test accuracy would be low.
# MAGIC
# MAGIC > **Part 3: Diagnosis**
# MAGIC >
# MAGIC > To determine the best solution, I would plot learning curves showing train and test performance versus training set size. This reveals whether collecting more data would help or if we need to adjust model complexity.
# MAGIC
# MAGIC > **Part 4: Solutions (Prioritized)**
# MAGIC >
# MAGIC > If the test curve is still rising:
# MAGIC > 1. **Collect more training data** — most effective for high variance
# MAGIC > 2. **Cross-validation** — ensure robust evaluation
# MAGIC > 3. **Regularization (L1/L2)** — penalize large weights
# MAGIC > 4. **Early stopping** — stop before memorization begins
# MAGIC >
# MAGIC > If the test curve has plateaued:
# MAGIC > 1. **Reduce model complexity** — fewer parameters
# MAGIC > 2. **Feature selection** — remove noisy features
# MAGIC > 3. **Strong regularization** — constrain model capacity
# MAGIC >
# MAGIC > Additional techniques:
# MAGIC > - **Dropout** (for neural networks)
# MAGIC > - **Data augmentation** (for images/text)
# MAGIC > - **Ensemble methods** (reduce variance through averaging)
# MAGIC
# MAGIC > **Part 5: Monitoring**
# MAGIC >
# MAGIC > I would track both train and test metrics during implementation, and use validation curves to tune hyperparameters like regularization strength."
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🟢 What Makes This Strong
# MAGIC
# MAGIC ✓ **Immediate identification** — "overfitting, high variance"
# MAGIC ✓ **Clear explanation** — distinguishes bias from variance
# MAGIC ✓ **Systematic diagnosis** — learning curves, not guessing
# MAGIC ✓ **Prioritized solutions** — context-dependent (curve shape)
# MAGIC ✓ **Multiple techniques** — shows breadth of knowledge
# MAGIC ✓ **Monitoring plan** — shows you think about deployment
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔴 Weak Answer (What NOT to Say)
# MAGIC
# MAGIC > "The model is overfitting. I would try regularization or maybe get more data. Or reduce the model size. Bias means underfitting."
# MAGIC
# MAGIC **Why this fails:**
# MAGIC * ❌ Too vague ("try regularization or maybe...")
# MAGIC * ❌ No diagnosis (why regularization vs more data?)
# MAGIC * ❌ Surface-level ("bias means underfitting" — no depth)
# MAGIC * ❌ No prioritization (listing options without context)
# MAGIC * ❌ Shows uncertainty ("maybe", "or")
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🟡 Mid-Level Answer (Better, But Not Senior)
# MAGIC
# MAGIC > "This is overfitting. I would add L2 regularization, use cross-validation, and try dropout if it's a neural network. I might also collect more data."
# MAGIC
# MAGIC **What's good:**
# MAGIC * ✓ Correct identification
# MAGIC * ✓ Multiple solutions
# MAGIC * ✓ Domain-specific (dropout for NNs)
# MAGIC
# MAGIC **What's missing:**
# MAGIC * ❌ No explanation of WHY overfitting occurs
# MAGIC * ❌ No diagnostic approach (learning curves)
# MAGIC * ❌ No prioritization logic
# MAGIC * ❌ Doesn't distinguish when to use each solution
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Key Phrases to Use
# MAGIC
# MAGIC **Problem identification:**
# MAGIC * "The model is overfitting — high variance"
# MAGIC * "Large train-test gap indicates memorization"
# MAGIC * "Poor generalization to unseen data"
# MAGIC
# MAGIC **Diagnosis:**
# MAGIC * "I would plot learning curves"
# MAGIC * "Analyze whether the test curve has plateaued"
# MAGIC * "Check if more data would close the gap"
# MAGIC
# MAGIC **Solutions:**
# MAGIC * "Regularization constrains model capacity"
# MAGIC * "Cross-validation gives robust evaluation"
# MAGIC * "Early stopping prevents memorization"
# MAGIC * "Feature selection removes noise"
# MAGIC
# MAGIC **Decision-making:**
# MAGIC * "The approach depends on whether..."
# MAGIC * "Learning curves will reveal if..."
# MAGIC * "I would prioritize X because..."
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📢 Practice Saying It Out Loud
# MAGIC
# MAGIC Read the senior-level answer above **out loud** 3 times.
# MAGIC Then close this notebook and explain it to someone (or a rubber duck).
# MAGIC You'll remember it in the interview!

# COMMAND ----------

# DBTITLE 1,Quick Quiz - Test Yourself
# MAGIC %md
# MAGIC ## 📝 Quick Quiz: Test Your Understanding
# MAGIC
# MAGIC ### Question 1
# MAGIC
# MAGIC **Scenario:**
# MAGIC ```
# MAGIC Train Accuracy: 78%
# MAGIC Test Accuracy:  76%
# MAGIC ```
# MAGIC
# MAGIC **What's the problem?**
# MAGIC
# MAGIC <details>
# MAGIC <summary>Click to reveal answer</summary>
# MAGIC
# MAGIC **Answer:** This is actually **good!** Small gap (2%) and reasonable scores suggest the model is well-balanced. You might try minor improvements (better features, ensemble), but this is a solid baseline.
# MAGIC
# MAGIC **Not** high variance (gap is small) and **not** high bias (scores are reasonable).
# MAGIC </details>
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Question 2
# MAGIC
# MAGIC **Scenario:**
# MAGIC ```
# MAGIC Train Accuracy: 99%
# MAGIC Test Accuracy:  95%
# MAGIC ```
# MAGIC
# MAGIC **Is this overfitting? Should you simplify the model?**
# MAGIC
# MAGIC <details>
# MAGIC <summary>Click to reveal answer</summary>
# MAGIC
# MAGIC **Answer:** **Small gap (4%)** suggests this is actually a **great model!** High scores on both train and test with small gap = excellent generalization.
# MAGIC
# MAGIC **Don't simplify!** You might lose performance. This is the goal.
# MAGIC
# MAGIC **Key insight:** High train accuracy alone doesn't mean overfitting. It's the *gap* that matters.
# MAGIC </details>
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Question 3
# MAGIC
# MAGIC **Scenario:**
# MAGIC ```
# MAGIC Train Accuracy: 62%
# MAGIC Test Accuracy:  60%
# MAGIC Learning curve: Both flat after 1000 samples
# MAGIC ```
# MAGIC
# MAGIC **Should you collect more data or simplify the model?**
# MAGIC
# MAGIC <details>
# MAGIC <summary>Click to reveal answer</summary>
# MAGIC
# MAGIC **Answer:** **Neither!** This is **high bias (underfitting)**. Both scores are low and curves have plateaued.
# MAGIC
# MAGIC **Correct action:** **Increase model complexity**
# MAGIC * Add more features
# MAGIC * Use more powerful model
# MAGIC * Increase polynomial degree
# MAGIC * Add hidden layers (if NN)
# MAGIC * Reduce regularization
# MAGIC
# MAGIC **Why not more data?** Flat learning curves mean model can't learn patterns even with current data. More data won't help a fundamentally weak model.
# MAGIC </details>
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Question 4
# MAGIC
# MAGIC **Scenario:**
# MAGIC ```
# MAGIC Train Accuracy: 99%
# MAGIC Test Accuracy:  72%
# MAGIC Learning curve: Test rising steadily
# MAGIC ```
# MAGIC
# MAGIC **Should you collect more data or add regularization?**
# MAGIC
# MAGIC <details>
# MAGIC <summary>Click to reveal answer</summary>
# MAGIC
# MAGIC **Answer:** **Collect more data** should be first priority!
# MAGIC
# MAGIC **Why?** Test curve is still rising → model has capacity and more data will close the gap.
# MAGIC
# MAGIC **But also:** Add regularization as a complementary technique. They're not mutually exclusive!
# MAGIC
# MAGIC **Best approach:**
# MAGIC 1. Add regularization (quick win)
# MAGIC 2. Collect more data (bigger impact)
# MAGIC 3. Monitor learning curves again
# MAGIC </details>
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Question 5
# MAGIC
# MAGIC **Scenario:**
# MAGIC ```
# MAGIC Train Accuracy: 98%
# MAGIC Test Accuracy:  74%
# MAGIC Learning curve: Test plateaued, train high
# MAGIC ```
# MAGIC
# MAGIC **What's the diagnosis and solution?**
# MAGIC
# MAGIC <details>
# MAGIC <summary>Click to reveal answer</summary>
# MAGIC
# MAGIC **Answer:** **High variance, but data won't help.**
# MAGIC
# MAGIC **Diagnosis:** Large gap (24%) = overfitting. Test curve plateaued = more data won't improve test performance.
# MAGIC
# MAGIC **Solution (priority order):**
# MAGIC 1. **Strong regularization** (L2, dropout)
# MAGIC 2. **Simplify model** (reduce parameters)
# MAGIC 3. **Feature selection** (remove noisy features)
# MAGIC 4. **Ensemble methods** (reduce variance)
# MAGIC
# MAGIC **Why not more data?** Test curve has plateaued → model is already seeing enough variety. Problem is model complexity, not data quantity.
# MAGIC </details>
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Scoring Guide
# MAGIC
# MAGIC * **5/5:** You're ready for senior interviews! 🎉
# MAGIC * **3-4/5:** Solid foundation, review weak areas
# MAGIC * **1-2/5:** Re-read the learning curves section
# MAGIC * **0/5:** Start from the top of this notebook
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Remember These Key Points
# MAGIC
# MAGIC 1. **Gap matters more than absolute accuracy**
# MAGIC    * Train 99%, Test 95% = ✓ Great
# MAGIC    * Train 70%, Test 68% = ✓ OK (might be hard problem)
# MAGIC    * Train 99%, Test 70% = ❌ Overfitting
# MAGIC
# MAGIC 2. **Learning curves guide decisions**
# MAGIC    * Test rising → more data helps
# MAGIC    * Test flat → more data won't help
# MAGIC    * Both low & flat → need more complex model
# MAGIC
# MAGIC 3. **Solutions depend on context**
# MAGIC    * High variance + data helps → collect data
# MAGIC    * High variance + data doesn't help → regularize
# MAGIC    * High bias → increase complexity
# MAGIC
# MAGIC 4. **Multiple techniques can combine**
# MAGIC    * Regularization + more data
# MAGIC    * Cross-validation + early stopping
# MAGIC    * Feature selection + ensemble
# MAGIC
# MAGIC 5. **Always explain your reasoning**
# MAGIC    * "I would X because Y"
# MAGIC    * Not "I would try X or maybe Y"

# COMMAND ----------

# DBTITLE 1,Summary Cheat Sheet
# MAGIC %md
# MAGIC ## 📜 One-Page Cheat Sheet
# MAGIC
# MAGIC ### 🔴 Problem Identification
# MAGIC
# MAGIC | Symptoms | Diagnosis | Train | Test | Gap |
# MAGIC |----------|-----------|-------|------|-----|
# MAGIC | Both low | High Bias (Underfit) | Low | Low | Small |
# MAGIC | Large gap, train high | High Variance (Overfit) | High | Low | Large |
# MAGIC | Both high, small gap | Good Fit | High | High | Small |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📈 Learning Curve Patterns
# MAGIC
# MAGIC | Pattern | What It Means | Action |
# MAGIC |---------|---------------|--------|
# MAGIC | Test rising | More data will help | ✅ Collect data |
# MAGIC | Both flat & low | Model too simple | ✅ Increase complexity |
# MAGIC | Test flat & large gap | Model too complex | ✅ Regularize/simplify |
# MAGIC | Both high & converged | Well-tuned | ✓ Done |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🛠️ Solution Techniques
# MAGIC
# MAGIC **For High Variance (Overfitting):**
# MAGIC 1. More training data ⭐⭐⭐⭐⭐
# MAGIC 2. Regularization (L1/L2) ⭐⭐⭐⭐
# MAGIC 3. Cross-validation ⭐⭐⭐⭐
# MAGIC 4. Early stopping ⭐⭐⭐⭐
# MAGIC 5. Dropout (NNs) ⭐⭐⭐⭐⭐
# MAGIC 6. Feature selection ⭐⭐⭐
# MAGIC 7. Simplify model ⭐⭐⭐
# MAGIC 8. Data augmentation ⭐⭐⭐⭐
# MAGIC
# MAGIC **For High Bias (Underfitting):**
# MAGIC 1. Increase model complexity
# MAGIC 2. Add more features
# MAGIC 3. Reduce regularization
# MAGIC 4. Use more powerful architecture
# MAGIC 5. Feature engineering
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ❓ Decision Tree
# MAGIC
# MAGIC ```
# MAGIC Overfitting detected?
# MAGIC     |
# MAGIC     v
# MAGIC Plot learning curves
# MAGIC     |
# MAGIC     +---> Test rising? → YES → Collect more data
# MAGIC     |                  → NO  → vvv
# MAGIC     |
# MAGIC     +---> Test flat? → YES → Regularize/Simplify
# MAGIC                      → NO  → Need more samples
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🗣️ Interview Answer Template
# MAGIC
# MAGIC 1. **Identify:** "This is overfitting — high variance"
# MAGIC 2. **Explain:** "Large gap means memorization, not generalization"
# MAGIC 3. **Diagnose:** "I'd plot learning curves to guide the solution"
# MAGIC 4. **Solve:** "If test rising → collect data. If flat → regularize"
# MAGIC 5. **List alternatives:** "Also: cross-validation, dropout, early stopping"
# MAGIC 6. **Monitor:** "Track train/test gap during implementation"
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚡ Common Mistakes to Avoid
# MAGIC
# MAGIC ❌ "Just collect more data" (What if high bias?)
# MAGIC ❌ "High train accuracy = overfitting" (What about the gap?)
# MAGIC ❌ "Accuracy is good" (Train or test? What's the gap?)
# MAGIC ❌ "Try everything" (No systematic approach)
# MAGIC ❌ Confusing bias/variance definitions
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Key Numbers to Remember
# MAGIC
# MAGIC * **Gap > 15%:** High variance, likely overfitting
# MAGIC * **Gap < 8%:** Good generalization
# MAGIC * **Both < 70%:** Check for high bias
# MAGIC * **Train > 95%, Test < 75%:** Classic overfitting
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📚 Further Reading
# MAGIC
# MAGIC * Andrew Ng's ML Course (bias-variance tradeoff)
# MAGIC * "Learning Curves" — scikit-learn docs
# MAGIC * "Regularization for Deep Learning" — Goodfellow et al.
# MAGIC * "A Few Useful Things to Know About Machine Learning" — Domingos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ You're Ready When You Can:
# MAGIC
# MAGIC * [ ] Identify overfitting from train/test scores
# MAGIC * [ ] Explain bias vs variance with examples
# MAGIC * [ ] List 8+ generalization techniques
# MAGIC * [ ] Interpret learning curves
# MAGIC * [ ] Decide: collect data vs regularize vs simplify
# MAGIC * [ ] Give a complete 2-minute answer
# MAGIC
# MAGIC **🎉 If you checked all boxes, you're interview-ready!**

# COMMAND ----------

# DBTITLE 1,Practice Notebook Title
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC # 🎯 Practice, Practice, Practice!
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Run all cells** in this notebook to see the visualizations
# MAGIC 2. **Read your answers out loud** to practice articulation
# MAGIC 3. **Modify the code** to generate different scenarios
# MAGIC 4. **Explain to someone** else (best way to solidify understanding)
# MAGIC 5. **Review before interviews** (bookmark this notebook!)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📝 Mock Interview Questions to Practice
# MAGIC
# MAGIC 1. "Your model shows 99% train and 72% test accuracy. Diagnose and fix."
# MAGIC 2. "Both train and test accuracy are 65%. What's wrong?"
# MAGIC 3. "When would collecting more data NOT help?"
# MAGIC 4. "Explain the bias-variance tradeoff."
# MAGIC 5. "How do learning curves inform your decisions?"
# MAGIC 6. "List 5 techniques to reduce overfitting."
# MAGIC 7. "What's the difference between L1 and L2 regularization?"
# MAGIC 8. "Your test accuracy is higher than train accuracy. What happened?"
# MAGIC 9. "How would you diagnose if regularization is too strong?"
# MAGIC 10. "Walk me through your process for tuning a model."
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎓 Confidence Builder
# MAGIC
# MAGIC You now know:
# MAGIC * ✅ More than 80% of candidates
# MAGIC * ✅ Enough for senior ML engineer roles  
# MAGIC * ✅ The systematic approach that separates good from great
# MAGIC
# MAGIC **Go ace that interview! 🚀**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC *Created for ML interview preparation. Share with fellow interviewees!*