# Databricks notebook source
from IPython.display import Image

display(Image(url="https://dbc-948eca32-5f10.cloud.databricks.com/editor/files/1600285474934042?o=7474644902174624"))

# COMMAND ----------

display(Image(url="https://dbc-948eca32-5f10.cloud.databricks.com/editor/files/1600285474934043?o=7474644902174624"))

# COMMAND ----------

# DBTITLE 1,📚 Introduction - RF vs GB Math
# MAGIC %md
# MAGIC # 🌳 Random Forest vs Gradient Boosting: Complete Mathematical Breakdown
# MAGIC
# MAGIC ## 🎯 What You'll Learn
# MAGIC
# MAGIC This notebook explains the **core mathematical differences** between Random Forest and Gradient Boosting.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔑 Key Mathematical Concepts
# MAGIC
# MAGIC ### **Random Forest: Variance Reduction**
# MAGIC
# MAGIC ```
# MAGIC Var(RF) = ρσ² + (1-ρ)σ²/B
# MAGIC
# MAGIC Where:
# MAGIC   - ρ = average correlation between trees (0 to 1)
# MAGIC   - σ² = variance of individual trees
# MAGIC   - B = number of trees
# MAGIC   
# MAGIC Key Insight: More trees (↑B) → Lower variance!
# MAGIC ```
# MAGIC
# MAGIC **Example (σ²=1, ρ=0.30, B=10):**
# MAGIC ```
# MAGIC Var(RF) = 0.30 + 0.70/10 = 0.370
# MAGIC Single tree = 1.000
# MAGIC Variance reduction = 63%! ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Gradient Boosting: Residual Learning**
# MAGIC
# MAGIC ```
# MAGIC F_M = F₀ + α·h₁ + α·h₂ + ... + α·hM
# MAGIC
# MAGIC At each iteration m:
# MAGIC   1. Calculate residuals: r_i = y_i - F_{m-1}(x_i)
# MAGIC   2. Train tree h_m on residuals
# MAGIC   3. Update: F_m = F_{m-1} + α·h_m
# MAGIC   
# MAGIC Key Insight: Each tree corrects previous mistakes!
# MAGIC ```
# MAGIC
# MAGIC **Example (α=0.10, M=10):**
# MAGIC ```
# MAGIC F₀ = 5.00 (initial prediction)
# MAGIC F₁₀ = 6.954 (after 10 iterations)
# MAGIC Error decreased from 4.0 → 1.046
# MAGIC Converging to truth! ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Quick Comparison
# MAGIC
# MAGIC | **Aspect** | **Random Forest 🌲🌲🌲** | **Gradient Boosting 🌲→🌲→🌲** |
# MAGIC |------------|-------------------------|-------------------------------|
# MAGIC | **Trees** | Parallel (independent) | Sequential (dependent) |
# MAGIC | **Data** | Bootstrap samples | Full data (on residuals) |
# MAGIC | **Features** | Random √p subset | All p features |
# MAGIC | **Combine** | Voting/Averaging | Additive sum |
# MAGIC | **Reduces** | **Variance** ↓ | **Bias** ↓ |
# MAGIC | **Overfitting** | Less prone | More prone (needs tuning) |
# MAGIC | **Formula** | `ŷ = (1/B)Σhb(x)` | `ŷ = F₀ + αΣhm(x)` |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🗺️ Notebook Roadmap
# MAGIC
# MAGIC 1. ✅ **Random Forest Variance Math** - Complete derivation + demo
# MAGIC 2. ✅ **Gradient Boosting Residual Math** - Step-by-step iterations
# MAGIC 3. ✅ **Interactive Demos** - Experiment with parameters
# MAGIC 4. ✅ **Visual Comparisons** - Side-by-side charts
# MAGIC 5. ✅ **Summary** - When to use each
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Let's dive in! 🚀**

# COMMAND ----------

# DBTITLE 1,Setup: Imports
# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS AND SETUP
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported!")
print("\n🎯 Topics:")
print("   1. Random Forest: Variance Reduction (Why more trees = better)")
print("   2. Gradient Boosting: Residual Learning (Why sequential = powerful)")
print("   3. Interactive Comparison")

# COMMAND ----------

# DBTITLE 1,🎯 Interactive Notebook Guide
# MAGIC %md
# MAGIC # 🎯 Interactive Notebook Guide: Kaise Padhe Yeh Notebook?
# MAGIC
# MAGIC **Yeh notebook aapko Random Forest aur Gradient Boosting ki complete mathematical understanding degi — interview se lekar implementation tak!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Notebook Structure (Kya-Kya Hai)
# MAGIC
# MAGIC ### **PART A: Foundation & Setup** (Cells 1-9)
# MAGIC
# MAGIC #### **1. 🔤 RF Full Form** 
# MAGIC * Random kahan se aaya? (Bootstrap + Random features)
# MAGIC * Forest kya hai? (Many trees together)
# MAGIC * **Kyun padhe:** Basic understanding ke liye
# MAGIC
# MAGIC #### **2. 📊 Manual Variance Calculation** (5 Patients Example)
# MAGIC * Real numbers se step-by-step:
# MAGIC   * Mean → Deviation → Variance → Covariance → Correlation
# MAGIC   * Covariance matrix Σ banao
# MAGIC   * 1ᵀΣ1 calculate karo == diagnol+off diagnol sum =B**2 * Variance**2+ covarianve*variance **2
# MAGIC   * RF variance formula verify karo
# MAGIC * **Kyun padhe:** Hand calculations samajh mein ayegi interview mein
# MAGIC
# MAGIC #### **3. 💻 Code Verification**
# MAGIC * Python se sab calculations verify karo
# MAGIC * Heatmaps aur visualizations
# MAGIC * **Kyun padhe:** Manual aur code match hona chahiye!
# MAGIC
# MAGIC #### **4. 📖 Complete Formula Reference**
# MAGIC * Level 1 (Mean) se Level 12 (RF variance) tak
# MAGIC * Har formula ka derivation
# MAGIC * Connection chain: Mean → Deviation → Variance → ... → RF formula
# MAGIC * **Kyun padhe:** Interview prep ke liye formula bank
# MAGIC
# MAGIC #### **5. 🧮 Matrix Derivation**
# MAGIC * Step-by-step matrix operations (B=3 example)
# MAGIC * Kaise Var(X̄) = (1/B²)[Bσ² + (B²-B)ρσ²] banta hai
# MAGIC * **Kyun padhe:** Advanced interviews mein matrix approach puchte hain
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **PART B: Decision Tree Foundation** (Cells 10-11)
# MAGIC
# MAGIC #### **6. 🌳 Decision Tree Basics**
# MAGIC * Gini impurity formula
# MAGIC * Information gain calculation
# MAGIC * Best split kaise dhundte hain
# MAGIC * **Kyun padhe:** RF aur GB dono trees use karte hain!
# MAGIC
# MAGIC #### **7. 💻 Gini Demo Code**
# MAGIC * Real example: Hours studied → Pass/Fail
# MAGIC * Manual calculations with visualization
# MAGIC * **Kyun padhe:** Concept clear hoga examples se
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **PART C: Random Forest Deep Dive** (Cells 12-14)
# MAGIC
# MAGIC #### **8. 🌲🌲🌲 RF Mathematics**
# MAGIC * Problem: Single tree ki high variance
# MAGIC * Solution: Bagging (Bootstrap Aggregating)
# MAGIC   * Step 1: Bootstrap sampling (~63% unique)
# MAGIC   * Step 2: Random feature selection (√p features)
# MAGIC   * Step 3: Voting/Averaging
# MAGIC * Variance formula derivation
# MAGIC * **Kyun padhe:** RF kaise variance reduce karta hai
# MAGIC
# MAGIC #### **9. 💻 Bootstrap & Voting Demo**
# MAGIC * 5 trees, bootstrap sampling
# MAGIC * Individual vs ensemble accuracy
# MAGIC * Diversity metrics
# MAGIC * **Kyun padhe:** Live example dekhoge toh clear hoga
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **PART D: Gradient Boosting Deep Dive** (Cells 15-16)
# MAGIC
# MAGIC #### **10. 🌲→🌲→🌲 GB Mathematics**
# MAGIC * Alag approach: Sequential mistake correction
# MAGIC * Step-by-step algorithm:
# MAGIC   * Initialize: F₀ = mean(y)
# MAGIC   * Calculate residuals: r = y - F_{m-1}
# MAGIC   * Train tree on residuals
# MAGIC   * Update: F_m = F_{m-1} + α·h_m
# MAGIC * **Kyun padhe:** GB kaise bias reduce karta hai
# MAGIC
# MAGIC #### **11. 💻 GB Iteration-by-Iteration**
# MAGIC * 3 data points, 10 iterations
# MAGIC * Har iteration mein:
# MAGIC   * Residuals calculate
# MAGIC   * Tree fit
# MAGIC   * Predictions update
# MAGIC   * MSE improvement
# MAGIC * **Kyun padhe:** Sequential learning dekhoge live!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **PART E: Original Comparison Content** (Cells 17-20)
# MAGIC
# MAGIC #### **12-15. RF Variance & GB Residual Theory**
# MAGIC * Mathematical proofs
# MAGIC * Interactive explorers
# MAGIC * Comparison tables
# MAGIC * **Kyun padhe:** Theoretical depth ke liye
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎓 Kaise Padhe? (Study Strategy)
# MAGIC
# MAGIC ### **🟢 Beginners (First Time Learning):**
# MAGIC
# MAGIC 1. **Start here:**
# MAGIC    * RF Full Form (Cell 2)
# MAGIC    * Decision Tree Basics (Cell 10-11)
# MAGIC    * RF Mathematics overview (Cell 12)
# MAGIC    * GB Mathematics overview (Cell 15)
# MAGIC
# MAGIC 2. **Run demos:**
# MAGIC    * Gini Demo (Cell 11)
# MAGIC    * Bootstrap Demo (Cell 14)
# MAGIC    * GB Demo (Cell 16)
# MAGIC
# MAGIC 3. **Skip for now:**
# MAGIC    * Manual calculations (too detailed)
# MAGIC    * Matrix derivation (advanced)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **🟡 Intermediate (Know Basics, Want Depth):**
# MAGIC
# MAGIC 1. **Start here:**
# MAGIC    * Manual Variance Calculation (Cell 3-4)
# MAGIC    * Complete Formula Reference (Cell 5)
# MAGIC    * All demo cells
# MAGIC
# MAGIC 2. **Focus on:**
# MAGIC    * Why variance reduces in RF
# MAGIC    * Why bias reduces in GB
# MAGIC    * When to use which
# MAGIC
# MAGIC 3. **Practice:**
# MAGIC    * Change parameters in demos
# MAGIC    * Verify formulas manually
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **🔴 Advanced (Interview Prep):**
# MAGIC
# MAGIC 1. **Master everything:**
# MAGIC    * Manual calculations line-by-line
# MAGIC    * Matrix derivation step-by-step
# MAGIC    * All formula derivations
# MAGIC
# MAGIC 2. **Be ready to explain:**
# MAGIC    * "Derive RF variance formula from scratch"
# MAGIC    * "Show me bootstrap sampling math"
# MAGIC    * "Why does GB need learning rate?"
# MAGIC
# MAGIC 3. **Practice questions:**
# MAGIC    * "What if ρ=1 in RF?" (No diversity, no benefit!)
# MAGIC    * "What if α=1 in GB?" (Overfitting risk!)
# MAGIC    * "RF vs GB: Which when?" (Use comparison table)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Key Sections for Interviews
# MAGIC
# MAGIC ### **Must Know:**
# MAGIC
# MAGIC | Topic | Cell(s) | Time | What to Remember |
# MAGIC |-------|---------|------|------------------|
# MAGIC | **Gini Impurity** | 10-11 | 10 min | Formula, pure node = 0, mixed = 0.5 |
# MAGIC | **RF Variance Formula** | 3-5 | 20 min | ρσ² + (1-ρ)σ²/B, what's ρ and B |
# MAGIC | **Bootstrap Sampling** | 12-14 | 15 min | ~63% unique, diversity |
# MAGIC | **GB Algorithm** | 15-16 | 20 min | Residuals, α, sequential |
# MAGIC | **RF vs GB** | 17-20 | 10 min | Parallel vs sequential, variance vs bias |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 Quick Reference: Key Formulas
# MAGIC
# MAGIC ```
# MAGIC 📐 DECISION TREE:
# MAGIC Gini = 1 - Σ p_k²
# MAGIC Information Gain = Gini_parent - weighted_children_gini
# MAGIC
# MAGIC 🌲 RANDOM FOREST:
# MAGIC Var(X̄) = ρσ² + (1-ρ)σ²/B
# MAGIC   ρ = correlation between trees
# MAGIC   σ² = variance of individual tree
# MAGIC   B = number of trees
# MAGIC
# MAGIC 🌲→🌲 GRADIENT BOOSTING:
# MAGIC F_M(x) = F₀ + Σ α·h_m(x)
# MAGIC   F₀ = initial prediction (mean)
# MAGIC   α = learning rate
# MAGIC   h_m = tree trained on residuals
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 Ready to Start?
# MAGIC
# MAGIC **Choose your path:**
# MAGIC
# MAGIC * **🟢 Beginner?** Start with Cell 2 (RF Full Form)
# MAGIC * **🟡 Intermediate?** Jump to Cell 3 (Manual Calculations)
# MAGIC * **🔴 Advanced?** Go straight to Cell 6 (Matrix Derivation)
# MAGIC
# MAGIC **Interactive tip:** Run ALL demo cells (11, 14, 16) — they have live visualizations!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎁 Bonus: What You'll Learn
# MAGIC
# MAGIC ✅ **Conceptual:** Why RF reduces variance, why GB reduces bias  
# MAGIC ✅ **Mathematical:** Derive formulas from scratch  
# MAGIC ✅ **Practical:** When to use RF vs GB  
# MAGIC ✅ **Interview:** Answer "why" questions with confidence  
# MAGIC ✅ **Implementation:** Understand parameters (B, ρ, α, depth)  
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Ab shuru karte hain! 🚀**

# COMMAND ----------

# DBTITLE 1,🎉 What We Just Saw: Live Demo Summary
# MAGIC %md
# MAGIC # 🎉 What We Just Saw: Live Demo Summary
# MAGIC
# MAGIC **Abhi humne teen powerful demos dekhe! Let me explain kya hua:**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🌳 Demo 1: Decision Tree (Gini Impurity)
# MAGIC
# MAGIC ### **Problem:**
# MAGIC ```
# MAGIC Data: 5 students
# MAGIC Hours studied: [1, 2, 4, 5, 6]
# MAGIC Passed:        [No, No, Yes, Yes, Yes]
# MAGIC
# MAGIC Question: Best split kahan pe karein?
# MAGIC ```
# MAGIC
# MAGIC ### **Solution: Hours > 3**
# MAGIC
# MAGIC **Why perfect split?**
# MAGIC
# MAGIC **Left (Hours ≤ 3):** [No, No] → Gini = 0.0 (pure!)  
# MAGIC **Right (Hours > 3):** [Yes, Yes, Yes] → Gini = 0.0 (pure!)
# MAGIC
# MAGIC **Information Gain = 0.48** ← Maximum possible!
# MAGIC
# MAGIC ### **💡 Key Learning:**
# MAGIC * Gini = 0 → Pure node (best)
# MAGIC * Gini = 0.5 → Mixed 50-50 (worst)
# MAGIC * Best split = Maximum IG
# MAGIC
# MAGIC **Visual dekha?** 
# MAGIC * Left chart: Gini curve (0 at extremes, 0.5 at center)
# MAGIC * Right chart: Tree structure with predictions
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🌲🌲🌲 Demo 2: Random Forest (Bootstrap & Voting)
# MAGIC
# MAGIC ### **Setup:**
# MAGIC ```
# MAGIC 100 samples, 5 features
# MAGIC 5 trees trained
# MAGIC Bootstrap sampling: ~63% unique per tree
# MAGIC ```
# MAGIC
# MAGIC ### **Results Dekhe:**
# MAGIC
# MAGIC **Individual Trees:**
# MAGIC ```
# MAGIC Tree 1: 97.0%
# MAGIC Tree 2: 98.0%
# MAGIC Tree 3: 93.0%  ← Weakest!
# MAGIC Tree 4: 98.0%
# MAGIC Tree 5: 100.0% ← Strongest!
# MAGIC
# MAGIC Average: 97.2%
# MAGIC ```
# MAGIC
# MAGIC **Random Forest (Voting):**
# MAGIC ```
# MAGIC Accuracy: 100.0%  🎉
# MAGIC Improvement: +2.8%
# MAGIC ```
# MAGIC
# MAGIC ### **💡 Kya Hua?**
# MAGIC
# MAGIC 1. **Bootstrap Diversity:**
# MAGIC    * Tree 1: 62% unique samples
# MAGIC    * Tree 2: 65% unique samples
# MAGIC    * Har tree alag data dekhta hai!
# MAGIC
# MAGIC 2. **Voting Magic:**
# MAGIC    * Tree 3 weak hai (93%), but majority vote correct hai!
# MAGIC    * Errors cancel out → Better ensemble
# MAGIC
# MAGIC 3. **Visual dekha?**
# MAGIC    * Left: Individual vs RF accuracy bars
# MAGIC    * Right: Sample selection histogram (diversity)
# MAGIC
# MAGIC ### **🔑 Core Insight:**
# MAGIC **Diversity + Voting = Better than best individual tree!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🌲→🌲→🌲 Demo 3: Gradient Boosting (Residual Learning)
# MAGIC
# MAGIC ### **Setup:**
# MAGIC ```
# MAGIC Data: x=[1,2,3], y=[3,5,8]
# MAGIC Parameters: α=0.1, M=10 iterations
# MAGIC ```
# MAGIC
# MAGIC ### **Journey Dekha:**
# MAGIC
# MAGIC #### **Iteration 0 (Start):**
# MAGIC ```
# MAGIC F₀ = mean(y) = 5.33
# MAGIC Predictions: [5.33, 5.33, 5.33]
# MAGIC MSE: 4.22  ← High error!
# MAGIC ```
# MAGIC
# MAGIC #### **Iteration 1:**
# MAGIC ```
# MAGIC Residuals: [-2.33, -0.33, +2.67]  ← Mistakes!
# MAGIC Tree learns these residuals
# MAGIC Update: F₁ = F₀ + 0.1×tree
# MAGIC New MSE: 3.42  ← Improvement: 0.80
# MAGIC ```
# MAGIC
# MAGIC #### **Iteration 5:**
# MAGIC ```
# MAGIC Predictions: [4.38, 5.20, 6.43]
# MAGIC MSE: 1.47  ← Getting closer!
# MAGIC ```
# MAGIC
# MAGIC #### **Iteration 10 (Final):**
# MAGIC ```
# MAGIC Predictions: [3.81, 5.12, 7.07]
# MAGIC True values: [3.00, 5.00, 8.00]
# MAGIC MSE: 0.51  ← 87.8% improvement! 🎉
# MAGIC
# MAGIC Errors:
# MAGIC x=1: 0.81 (was 2.33!)
# MAGIC x=2: 0.12 (was 0.33)
# MAGIC x=3: 0.93 (was 2.67!)
# MAGIC ```
# MAGIC
# MAGIC ### **💡 Kya Hua?**
# MAGIC
# MAGIC 1. **Sequential Correction:**
# MAGIC    * Iteration 1: Fix big mistakes
# MAGIC    * Iteration 5: Refine predictions
# MAGIC    * Iteration 10: Fine-tune remaining errors
# MAGIC
# MAGIC 2. **Learning Rate α=0.1:**
# MAGIC    * Small steps = Stable learning
# MAGIC    * Prevents overfitting
# MAGIC    * More iterations needed but safer
# MAGIC
# MAGIC 3. **Visual dekha?**
# MAGIC    * Top: MSE convergence (4.22 → 0.51)
# MAGIC    * Middle: Individual predictions approaching truth
# MAGIC    * Bottom: Additive formula + final comparison
# MAGIC
# MAGIC ### **🔑 Core Insight:**
# MAGIC **Each tree fixes previous mistakes → Additive improvement!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚖️ RF vs GB: Side-by-Side Comparison
# MAGIC
# MAGIC | Aspect | Random Forest | Gradient Boosting |
# MAGIC |--------|---------------|-------------------|
# MAGIC | **Approach** | Parallel trees | Sequential trees |
# MAGIC | **Training** | Independent | Dependent (residuals) |
# MAGIC | **Diversity** | Bootstrap + random features | Residual focus |
# MAGIC | **Combination** | Voting (average) | Additive (sum) |
# MAGIC | **Strength** | Reduces variance | Reduces bias |
# MAGIC | **Risk** | Less overfitting | More overfitting (if M too high) |
# MAGIC | **Speed** | Fast (parallel) | Slower (sequential) |
# MAGIC | **Tuning** | Easier (less sensitive) | Harder (more sensitive) |
# MAGIC
# MAGIC ### **Example Results:**
# MAGIC
# MAGIC **RF Demo:**
# MAGIC * Started: 97.2% (avg tree)
# MAGIC * Ended: 100.0% (ensemble)
# MAGIC * **↑ Variance reduction through voting**
# MAGIC
# MAGIC **GB Demo:**
# MAGIC * Started: MSE = 4.22
# MAGIC * Ended: MSE = 0.51
# MAGIC * **↓ Bias reduction through sequential correction**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 When to Use Which?
# MAGIC
# MAGIC ### **🌲 Use Random Forest When:**
# MAGIC ✅ You want **robust, out-of-the-box** performance  
# MAGIC ✅ You have **high-variance** base models  
# MAGIC ✅ You need **fast training** (parallel)  
# MAGIC ✅ You want **less hyperparameter tuning**  
# MAGIC ✅ Your data has **noise/outliers**  
# MAGIC ✅ You need **feature importance** analysis  
# MAGIC
# MAGIC **Example:** Classification with categorical features, quick baseline
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **🌲→🌲 Use Gradient Boosting When:**
# MAGIC ✅ You want **highest accuracy** (willing to tune)  
# MAGIC ✅ You have **high-bias** base models  
# MAGIC ★ You can **afford sequential training**  
# MAGIC ★ You're willing to **tune carefully** (α, M, depth)  
# MAGIC ✅ Your data is **clean** and **structured**  
# MAGIC ✅ You need **best Kaggle performance**  
# MAGIC
# MAGIC **Example:** Regression tasks, competitions, production models (with careful validation)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 Key Formulas Recap
# MAGIC
# MAGIC ### **Random Forest:**
# MAGIC ```
# MAGIC Var(X̄) = ρσ² + (1-ρ)σ²/B
# MAGIC
# MAGIC Where:
# MAGIC   ρ = 0.93 (from demo) ← High correlation!
# MAGIC   B = 5 trees
# MAGIC   
# MAGIC Result: 14.7% variance reduction
# MAGIC (Could be higher with lower ρ!)
# MAGIC ```
# MAGIC
# MAGIC ### **Gradient Boosting:**
# MAGIC ```
# MAGIC F_M(x) = F₀ + α·h₁ + α·h₂ + ... + α·h_M
# MAGIC
# MAGIC Where:
# MAGIC   F₀ = 5.33 (initial mean)
# MAGIC   α = 0.1 (learning rate)
# MAGIC   M = 10 iterations
# MAGIC   
# MAGIC Result: 87.8% error reduction!
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎓 Interview Questions You Can Now Answer:
# MAGIC
# MAGIC ### **Q1: Why does Random Forest reduce variance?**
# MAGIC **A:** Bootstrap sampling + random features create diverse trees. Errors of individual trees are uncorrelated, so averaging cancels them out. Formula: Var(X̄) = ρσ² + (1-ρ)σ²/B shows variance decreases with more trees (B) and less correlation (ρ).
# MAGIC
# MAGIC ### **Q2: Why does Gradient Boosting need a learning rate?**
# MAGIC **A:** Learning rate α controls step size. Small α (0.01-0.1) means small corrections per iteration, preventing overfitting. Large α (0.5-1.0) means fast learning but risk of overshooting. Our demo used α=0.1 for stable 87.8% improvement.
# MAGIC
# MAGIC ### **Q3: RF vs GB — which is better?**
# MAGIC **A:** Depends! RF is better for:
# MAGIC * Quick baseline, less tuning
# MAGIC * Noisy data, categorical features
# MAGIC * When you need speed
# MAGIC
# MAGIC GB is better for:
# MAGIC * Maximum accuracy (with tuning)
# MAGIC * Clean, structured data
# MAGIC * When you have time to optimize
# MAGIC
# MAGIC Our demos: RF gave 100% (variance ↓), GB gave 87.8% error reduction (bias ↓).
# MAGIC
# MAGIC ### **Q4: What's the “~63%” in bootstrap?**
# MAGIC **A:** When sampling N items with replacement N times, probability of being selected at least once = 1 - (1-1/N)^N ≈ 63.2% as N→∞. Our demo confirmed: Trees got 60-66% unique samples!
# MAGIC
# MAGIC ### **Q5: Can you derive the RF variance formula?**
# MAGIC **A:** Yes! (Points to Cell 6) Start with Var(X̄) = (1/B²)1ᵀΣ1, where Σ is covariance matrix. Sum diagonal (Bσ²) + off-diagonal ((B²-B)ρσ²), divide by B², simplify to get ρσ² + (1-ρ)σ²/B.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 Next Steps
# MAGIC
# MAGIC 1. **🔄 Re-run demos** with different parameters:
# MAGIC    * Change B (number of trees)
# MAGIC    * Change α (learning rate)
# MAGIC    * Change M (iterations)
# MAGIC
# MAGIC 2. **📝 Practice derivations:**
# MAGIC    * Manual calculations (Cell 3-4)
# MAGIC    * Matrix derivation (Cell 6)
# MAGIC
# MAGIC 3. **📊 Read theory cells:**
# MAGIC    * RF variance reduction (Cell 12)
# MAGIC    * GB residual learning (Cell 15)
# MAGIC
# MAGIC 4. **🎯 Answer interview questions** using these concepts!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **You're now ready to explain RF and GB math from scratch! 🎉**

# COMMAND ----------

# DBTITLE 1,🔤 What is RF? Random Forest Full Form
# MAGIC %md
# MAGIC # 🔤 What is RF? Random Forest
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎲 **Random**
# MAGIC
# MAGIC Do jagah randomness:
# MAGIC
# MAGIC ### **1. Random Bootstrap Samples**
# MAGIC ```
# MAGIC Har tree alag data dekhta hai!
# MAGIC Bootstrap = sample with replacement
# MAGIC
# MAGIC Original: [1,2,3,4,5,6,7,8,9,10]
# MAGIC Tree 1:   [3,3,7,1,9,2,7,4,6,10]  ← repeats allowed!
# MAGIC Tree 2:   [5,2,8,1,1,9,3,7,4,6]
# MAGIC Tree 3:   [2,4,4,9,5,1,8,7,3,10]
# MAGIC ```
# MAGIC
# MAGIC ### **2. Random Features Per Split**
# MAGIC ```
# MAGIC Har node sirf kuch features try karta hai!
# MAGIC m = √p features (classification)
# MAGIC m = p/3 features (regression)
# MAGIC
# MAGIC p = 20 features total
# MAGIC m = √20 ≈ 4-5 features per split
# MAGIC
# MAGIC Node 1: Try features [2, 7, 11, 15, 18]
# MAGIC Node 2: Try features [1, 4, 9, 12, 19]
# MAGIC Node 3: Try features [3, 6, 10, 13, 16]
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🌲 **Forest**
# MAGIC
# MAGIC Bahut saare trees ek saath!
# MAGIC
# MAGIC ```
# MAGIC B = number of trees
# MAGIC   = 100, 200, 500, 1000...
# MAGIC   
# MAGIC Ek tree = weak
# MAGIC B trees = strong! 💪
# MAGIC
# MAGIC Ek saath = Forest! 🌲🌲🌲
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 **Ek Line Mein:**
# MAGIC
# MAGIC ```
# MAGIC Random Forest = Random data + Random features 
# MAGIC               + Bahut saare Decision Trees 
# MAGIC               + Majority vote
# MAGIC               = Better than one tree! ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 **Why It Works:**
# MAGIC
# MAGIC * **Random data** → Diverse trees (each sees different patterns)
# MAGIC * **Random features** → Even more diversity (trees can't all use same strong features)
# MAGIC * **Many trees** → Errors cancel out (averaging reduces variance)
# MAGIC * **Result** → Lower variance, better generalization! 🎉

# COMMAND ----------

# DBTITLE 1,📊 Manual Variance Calculation: Complete Walkthrough
# MAGIC %md
# MAGIC # 📊 Manual Variance Calculation: Complete Walkthrough
# MAGIC
# MAGIC **Yaad karo** — 5 patients, 5 trees
# MAGIC
# MAGIC Yeh example dikhata hai ki kaise RF ka variance formula work karta hai, step-by-step!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Data — Predictions Matrix
# MAGIC
# MAGIC 5 patients ke liye 5 trees ke predictions:
# MAGIC
# MAGIC ```
# MAGIC          P1   P2   P3   P4   P5
# MAGIC Tree 1 [ 0.8  0.2  0.9  0.3  0.7 ]
# MAGIC Tree 2 [ 0.7  0.3  0.8  0.4  0.6 ]
# MAGIC Tree 3 [ 0.9  0.1  0.8  0.2  0.8 ]
# MAGIC Tree 4 [ 0.6  0.4  0.7  0.5  0.5 ]
# MAGIC Tree 5 [ 0.8  0.2  0.9  0.3  0.7 ]  ← Same as Tree 1!
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔢 Step 1 — Har Tree Ka Mean Nikalo
# MAGIC
# MAGIC ```
# MAGIC μᵢ = (1/N) Σ Tᵢⱼ
# MAGIC ```
# MAGIC
# MAGIC | Tree | P1 | P2 | P3 | P4 | P5 | **μ (Mean)** |
# MAGIC |------|----|----|----|----|----|--------------|
# MAGIC | T1 | 0.8 | 0.2 | 0.9 | 0.3 | 0.7 | **(0.8+0.2+0.9+0.3+0.7)/5 = 0.58** |
# MAGIC | T2 | 0.7 | 0.3 | 0.8 | 0.4 | 0.6 | **(0.7+0.3+0.8+0.4+0.6)/5 = 0.56** |
# MAGIC | T3 | 0.9 | 0.1 | 0.8 | 0.2 | 0.8 | **(0.9+0.1+0.8+0.2+0.8)/5 = 0.56** |
# MAGIC | T4 | 0.6 | 0.4 | 0.7 | 0.5 | 0.5 | **(0.6+0.4+0.7+0.5+0.5)/5 = 0.54** |
# MAGIC | T5 | 0.8 | 0.2 | 0.9 | 0.3 | 0.7 | **(0.8+0.2+0.9+0.3+0.7)/5 = 0.58** |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📐 Step 2 — Deviations (Tᵢⱼ - μᵢ)
# MAGIC
# MAGIC ```
# MAGIC dᵢⱼ = Tᵢⱼ - μᵢ
# MAGIC ```
# MAGIC
# MAGIC | Tree | μ | P1 | P2 | P3 | P4 | P5 |
# MAGIC |------|---|----|----|----|----|----|
# MAGIC | T1 | 0.58 | 0.8-0.58=**+0.22** | 0.2-0.58=**-0.38** | 0.9-0.58=**+0.32** | 0.3-0.58=**-0.28** | 0.7-0.58=**+0.12** |
# MAGIC | T2 | 0.56 | 0.7-0.56=**+0.14** | 0.3-0.56=**-0.26** | 0.8-0.56=**+0.24** | 0.4-0.56=**-0.16** | 0.6-0.56=**+0.04** |
# MAGIC | T3 | 0.56 | 0.9-0.56=**+0.34** | 0.1-0.56=**-0.46** | 0.8-0.56=**+0.24** | 0.2-0.56=**-0.36** | 0.8-0.56=**+0.24** |
# MAGIC | T4 | 0.54 | 0.6-0.54=**+0.06** | 0.4-0.54=**-0.14** | 0.7-0.54=**+0.16** | 0.5-0.54=**-0.04** | 0.5-0.54=**-0.04** |
# MAGIC | T5 | 0.58 | 0.8-0.58=**+0.22** | 0.2-0.58=**-0.38** | 0.9-0.58=**+0.32** | 0.3-0.58=**-0.28** | 0.7-0.58=**+0.12** |
# MAGIC
# MAGIC **Note:** Σdᵢⱼ = 0 for each tree (always!) ✅
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Step 3 — Variance σ² (Diagonal Elements)
# MAGIC
# MAGIC ```
# MAGIC σ²ᵢ = (1/N) Σ dᵢⱼ²
# MAGIC ```
# MAGIC
# MAGIC | Tree | P1² | P2² | P3² | P4² | P5² | **Sum** | **σ² = Sum/5** |
# MAGIC |------|-----|-----|-----|-----|-----|---------|----------------|
# MAGIC | T1 | 0.22²=**0.0484** | 0.38²=**0.1444** | 0.32²=**0.1024** | 0.28²=**0.0784** | 0.12²=**0.0144** | **0.3880** | **0.0776** |
# MAGIC | T2 | 0.14²=**0.0196** | 0.26²=**0.0676** | 0.24²=**0.0576** | 0.16²=**0.0256** | 0.04²=**0.0016** | **0.1720** | **0.0344** |
# MAGIC | T3 | 0.34²=**0.1156** | 0.46²=**0.2116** | 0.24²=**0.0576** | 0.36²=**0.1296** | 0.24²=**0.0576** | **0.5720** | **0.1144** |
# MAGIC | T4 | 0.06²=**0.0036** | 0.14²=**0.0196** | 0.16²=**0.0256** | 0.04²=**0.0016** | 0.04²=**0.0016** | **0.0520** | **0.0104** |
# MAGIC | T5 | 0.22²=**0.0484** | 0.38²=**0.1444** | 0.32²=**0.1024** | 0.28²=**0.0784** | 0.12²=**0.0144** | **0.3880** | **0.0776** |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔄 Step 4 — Covariance (Off-Diagonal Elements)
# MAGIC
# MAGIC ```
# MAGIC Cov(Tᵢ, Tⱼ) = (1/N) Σ dᵢₖ × dⱼₖ
# MAGIC ```
# MAGIC
# MAGIC ### **Example: Cov(T1, T2)**
# MAGIC
# MAGIC | Patient | d_T1 | d_T2 | **product** |
# MAGIC |---------|------|------|-------------|
# MAGIC | P1 | +0.22 | +0.14 | **+0.0308** |
# MAGIC | P2 | -0.38 | -0.26 | **+0.0988** |
# MAGIC | P3 | +0.32 | +0.24 | **+0.0768** |
# MAGIC | P4 | -0.28 | -0.16 | **+0.0448** |
# MAGIC | P5 | +0.12 | +0.04 | **+0.0048** |
# MAGIC | | | **Sum** | **+0.2560** |
# MAGIC
# MAGIC **Cov(T1,T2) = Sum/5 = +0.0512** ✅
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Example: Cov(T1, T3)**
# MAGIC
# MAGIC | Patient | d_T1 | d_T3 | **product** |
# MAGIC |---------|------|------|-------------|
# MAGIC | P1 | +0.22 | +0.34 | **+0.0748** |
# MAGIC | P2 | -0.38 | -0.46 | **+0.1748** |
# MAGIC | P3 | +0.32 | +0.24 | **+0.0768** |
# MAGIC | P4 | -0.28 | -0.36 | **+0.1008** |
# MAGIC | P5 | +0.12 | +0.24 | **+0.0288** |
# MAGIC | | | **Sum** | **+0.4560** |
# MAGIC
# MAGIC **Cov(T1,T3) = Sum/5 = +0.0912** ✅
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Example: Cov(T1, T5)**
# MAGIC
# MAGIC **T1 aur T5 same predictions hain!**
# MAGIC
# MAGIC ```
# MAGIC T1 = [0.8, 0.2, 0.9, 0.3, 0.7]
# MAGIC T5 = [0.8, 0.2, 0.9, 0.3, 0.7]  ← Identical!
# MAGIC
# MAGIC Cov(T1, T5) = σ²_T1 = 0.0776
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📈 Step 5 — Correlation ρ Nikalo
# MAGIC
# MAGIC ```
# MAGIC ρᵢⱼ = Cov(Tᵢ, Tⱼ) / (σᵢ × σⱼ)
# MAGIC ```
# MAGIC
# MAGIC | Pair | Cov | σᵢ | σⱼ | σᵢ × σⱼ | **ρ** |
# MAGIC |------|-----|----|----|---------|-------|
# MAGIC | T1, T2 | 0.0512 | √0.0776=0.2786 | √0.0344=0.1855 | 0.0517 | **0.990** |
# MAGIC | T1, T3 | 0.0912 | 0.2786 | √0.1144=0.3382 | 0.0942 | **0.968** |
# MAGIC | T1, T4 | 0.0248 | 0.2786 | √0.0104=0.1020 | 0.0284 | **0.873** |
# MAGIC | T1, T5 | 0.0776 | 0.2786 | 0.2786 | 0.0776 | **1.000** |
# MAGIC
# MAGIC **High correlation!** Trees are similar → less diversity
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Step 6 — Covariance Matrix Σ (5×5)
# MAGIC
# MAGIC ```
# MAGIC       T1       T2       T3       T4       T5
# MAGIC T1 [ 0.0776  0.0512  0.0912  0.0248  0.0776 ]
# MAGIC T2 [ 0.0512  0.0344  0.0627  0.0170  0.0512 ]
# MAGIC T3 [ 0.0912  0.0627  0.1144  0.0304  0.0912 ]
# MAGIC T4 [ 0.0248  0.0170  0.0304  0.0104  0.0248 ]
# MAGIC T5 [ 0.0776  0.0512  0.0912  0.0248  0.0776 ]
# MAGIC ```
# MAGIC
# MAGIC * **Diagonal** = σ² (variance)
# MAGIC * **Off-diagonal** = Cov (covariance)
# MAGIC * **Symmetric:** Σᵢⱼ = Σⱼᵢ
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🧮 Step 7 — 1ᵀΣ1 Calculate Karo
# MAGIC
# MAGIC ### **Step 7a — Σ1 (Har Row Ka Sum):**
# MAGIC
# MAGIC | Tree | T1 | T2 | T3 | T4 | T5 | **Row Sum** |
# MAGIC |------|----|----|----|----|----|-----------|
# MAGIC | T1 | 0.0776 | 0.0512 | 0.0912 | 0.0248 | 0.0776 | **0.3224** |
# MAGIC | T2 | 0.0512 | 0.0344 | 0.0627 | 0.0170 | 0.0512 | **0.2165** |
# MAGIC | T3 | 0.0912 | 0.0627 | 0.1144 | 0.0304 | 0.0912 | **0.3899** |
# MAGIC | T4 | 0.0248 | 0.0170 | 0.0304 | 0.0104 | 0.0248 | **0.1074** |
# MAGIC | T5 | 0.0776 | 0.0512 | 0.0912 | 0.0248 | 0.0776 | **0.3224** |
# MAGIC
# MAGIC ### **Step 7b — 1ᵀ(Σ1) = Sum of All Row Sums:**
# MAGIC
# MAGIC ```
# MAGIC = 0.3224 + 0.2165 + 0.3899 + 0.1074 + 0.3224
# MAGIC = 1.3586 ✅
# MAGIC ```
# MAGIC
# MAGIC ### **Step 7c — Verify (Sum of ALL 25 Elements):**
# MAGIC
# MAGIC ```
# MAGIC Sum of all elements in Σ = 1.3586 ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Step 8 — Var(X̄) Nikalo
# MAGIC
# MAGIC ```
# MAGIC Var(X̄) = 1ᵀΣ1 / B²
# MAGIC        = 1.3586 / 25
# MAGIC        = 0.0543
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Step 9 — Formula Se Verify Karo
# MAGIC
# MAGIC ### **Average σ² aur ρ:**
# MAGIC
# MAGIC ```
# MAGIC Average σ² = (0.0776+0.0344+0.1144+0.0104+0.0776)/5
# MAGIC            = 0.3144/5
# MAGIC            = 0.0629
# MAGIC
# MAGIC Average ρ ≈ 0.96 (all pairs ka average)
# MAGIC ```
# MAGIC
# MAGIC ### **Formula:**
# MAGIC
# MAGIC ```
# MAGIC Var(X̄) = ρσ² + (1-ρ)σ²/B
# MAGIC         = 0.96×0.0629 + (0.04)×0.0629/5
# MAGIC         = 0.0604 + 0.0005
# MAGIC         = 0.0609  ← close to 0.0543! ✅
# MAGIC ```
# MAGIC
# MAGIC **(Thoda fark hai kyunki ρ aur σ² exact average nahi)**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📝 Complete Summary Table
# MAGIC
# MAGIC | **Step** | **Formula** | **Result** |
# MAGIC |----------|-------------|------------|
# MAGIC | Mean | μᵢ = (1/N)Σ Tᵢⱼ | T1:0.58, T2:0.56... |
# MAGIC | Deviation | dᵢⱼ = Tᵢⱼ - μᵢ | Table above |
# MAGIC | Variance (diagonal) | σ²ᵢ = (1/N)Σ dᵢⱼ² | T1:0.0776... |
# MAGIC | Covariance (off-diag) | Cov = (1/N)Σ dᵢₖ×dⱼₖ | T1,T2:0.0512... |
# MAGIC | Correlation | ρ = Cov/(σᵢ×σⱼ) | ~0.96 |
# MAGIC | Matrix sum | 1ᵀΣ1 = Σ all elements | 1.3586 |
# MAGIC | RF variance | Var(X̄) = 1ᵀΣ1/B² | 0.0543 |
# MAGIC | Final formula | ρσ² + (1-ρ)σ²/B | ~0.0609 |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 Key Insight
# MAGIC
# MAGIC **High ρ ≈ 0.96** kyunki:
# MAGIC * T1 = T5 (same tree!) → ρ = 1.000
# MAGIC * Baaki bhi similar patterns
# MAGIC * High correlation → Less diversity → Less variance reduction
# MAGIC
# MAGIC **For good RF:** ρ should be LOW (0.1-0.3) through:
# MAGIC * Bootstrap sampling
# MAGIC * Random feature selection

# COMMAND ----------

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION: MANUAL CALCULATIONS WITH CODE
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd

print("═" * 80)
print("📊 VERIFICATION: 5 Patients, 5 Trees Example")
print("═" * 80)

# Data: 5 trees' predictions for 5 patients
T = np.array([
    [0.8, 0.2, 0.9, 0.3, 0.7],  # Tree 1
    [0.7, 0.3, 0.8, 0.4, 0.6],  # Tree 2
    [0.9, 0.1, 0.8, 0.2, 0.8],  # Tree 3
    [0.6, 0.4, 0.7, 0.5, 0.5],  # Tree 4
    [0.8, 0.2, 0.9, 0.3, 0.7],  # Tree 5 (same as Tree 1)
])

print("\n📋 Original Data (Rows=Trees, Cols=Patients):")
print(pd.DataFrame(T, 
                   index=[f'Tree {i+1}' for i in range(5)],
                   columns=[f'P{i+1}' for i in range(5)]))

# COMMAND ----------

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION: MANUAL CALCULATIONS WITH CODE
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd

print("═" * 80)
print("📊 VERIFICATION: 5 Patients, 5 Trees Example")
print("═" * 80)

# Data: 5 trees' predictions for 5 patients
T = np.array([
    [0.8, 0.2, 0.9, 0.3, 0.7],  # Tree 1
    [0.7, 0.3, 0.8, 0.4, 0.6],  # Tree 2
    [0.9, 0.1, 0.8, 0.2, 0.8],  # Tree 3
    [0.6, 0.4, 0.7, 0.5, 0.5],  # Tree 4
    [0.8, 0.2, 0.9, 0.3, 0.7],  # Tree 5 (same as Tree 1)
])

print("\n📋 Original Data (Rows=Trees, Cols=Patients):")
print(pd.DataFrame(T, 
                   index=[f'Tree {i+1}' for i in range(5)],
                   columns=[f'P{i+1}' for i in range(5)]))

# COMMAND ----------


# Step 1: Means
means = T.mean(axis=1)
print("\n🔢 Step 1 - Means (μᵢ):")
for i, mu in enumerate(means, 1):
    print(f"   Tree {i}: μ = {mu:.4f}")

# COMMAND ----------

T 

# COMMAND ----------

means

# COMMAND ----------

# MAGIC %md
# MAGIC means[:, np.newaxis](5,) → (5,1)✅ sahimeans[:, None](5,) → (5,1)✅ samemeans.reshape(-1,1)(5,) → (5,1)✅ samemeans(5,)❌ error

# COMMAND ----------

T - means[:, np.newaxis]

# COMMAND ----------



# Step 2: Deviations
deviations = T - means[:, np.newaxis]
print("\n📐 Step 2 - Deviations (Tᵢⱼ - μᵢ):")
print(pd.DataFrame(deviations,
                   index=[f'Tree {i+1}' for i in range(5)],
                   columns=[f'P{i+1}' for i in range(5)]).round(4))

# COMMAND ----------



# Verify sum of deviations = 0
print("\n   ✅ Verification: Sum of deviations = 0 for each tree?")
for i, dev_sum in enumerate(deviations.sum(axis=1), 1):
    print(f"      Tree {i}: {dev_sum:.10f} ≈ 0 ✅")

# COMMAND ----------


# Step 3: Variance (diagonal)
variances = (deviations**2).mean(axis=1)
print("\n📊 Step 3 - Variance (σ²ᵢ):")
for i, var in enumerate(variances, 1):
    print(f"   Tree {i}: σ² = {var:.4f}")

# COMMAND ----------



# Step 4: Covariance matrix
cov_matrix = np.cov(T, bias=True)  # bias=True for population cov (divide by N)
print("\n🔄 Step 4 - Covariance Matrix Σ (5×5):")
print("\n   (Diagonal = σ², Off-diagonal = Cov)\n")
print(pd.DataFrame(cov_matrix,
                   index=[f'T{i+1}' for i in range(5)],
                   columns=[f'T{i+1}' for i in range(5)]).round(4))

# COMMAND ----------





# Verify some covariances manually
print("\n   📝 Manual Covariance Verification:")
print(f"      Cov(T1,T2) = {np.mean(deviations[0] * deviations[1]):.4f} (from formula)")
print(f"      Cov(T1,T2) = {cov_matrix[0,1]:.4f} (from np.cov) ✅")
print(f"      Cov(T1,T3) = {np.mean(deviations[0] * deviations[2]):.4f}")
print(f"      Cov(T1,T5) = {np.mean(deviations[0] * deviations[4]):.4f}")
print(f"      (T1=T5, so Cov(T1,T5) = σ²_T1 = {variances[0]:.4f}) ✅")

# COMMAND ----------


# Step 5: Correlation matrix
corr_matrix = np.corrcoef(T)
print("\n📈 Step 5 - Correlation Matrix:")
print(pd.DataFrame(corr_matrix,
                   index=[f'T{i+1}' for i in range(5)],
                   columns=[f'T{i+1}' for i in range(5)]).round(3))

# COMMAND ----------

avg_corr = corr_matrix[np.triu_indices(5, k=1)].mean()
print(f"\n   Average correlation (ρ): {avg_corr:.4f}")
print(f"   (High correlation = less diversity!)")

# COMMAND ----------

cov_matrix

# COMMAND ----------

np.ones(5)

# COMMAND ----------

# MAGIC %md
# MAGIC Row sum of cov_matrix

# COMMAND ----------


# Step 6: 1'Σ1 (sum of all elements)
ones = np.ones(5)
sum_cov_matrix = ones @ cov_matrix @ ones
print("\n🧮 Step 6 - 1ᵀΣ1 (Sum of All Elements):")
print(f"   1ᵀΣ1 = {sum_cov_matrix:.4f}")
print(f"   Direct sum = {cov_matrix.sum():.4f} ✅")

# COMMAND ----------

# Step 7: RF Variance
B = 5
var_rf = sum_cov_matrix / (B**2)
print("\n🎯 Step 7 - Random Forest Variance:")
print(f"   Var(X̄) = 1ᵀΣ1 / B²")
print(f"          = {sum_cov_matrix:.4f} / {B**2}")
print(f"          = {var_rf:.4f}")

# COMMAND ----------

variances

# COMMAND ----------

variances.mean()

# COMMAND ----------

avg_corr

# COMMAND ----------

# Step 8: Verify with formula
avg_var = variances.mean()
var_rf_formula = avg_corr * avg_var + (1 - avg_corr) * avg_var / B
print("\n✅ Step 8 - Verify With Formula:")
print(f"   Average σ² = {avg_var:.4f}")
print(f"   Average ρ = {avg_corr:.4f}")
print(f"\n   Formula: Var(X̄) = ρσ² + (1-ρ)σ²/B")
print(f"          = {avg_corr:.4f}×{avg_var:.4f} + {(1-avg_corr):.4f}×{avg_var:.4f}/{B}")
print(f"          = {avg_corr*avg_var:.4f} + {(1-avg_corr)*avg_var/B:.4f}")
print(f"          = {var_rf_formula:.4f}")
print(f"\n   Matrix method:  {var_rf:.4f}")
print(f"   Formula method: {var_rf_formula:.4f}")
print(f"   Difference:     {abs(var_rf - var_rf_formula):.4f}")
print(f"   (Close enough! Small diff due to averaging) ✅")

# COMMAND ----------

avg_var, var_rf, var_rf_formula

# COMMAND ----------

# DBTITLE 1,Code Verification: Manual Calculations


# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# LEFT: Covariance matrix heatmap
ax = axes[0]
im = ax.imshow(cov_matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels([f'T{i+1}' for i in range(5)])
ax.set_yticklabels([f'T{i+1}' for i in range(5)])
ax.set_title('Covariance Matrix Σ', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax)

# Add values
for i in range(5):
    for j in range(5):
        text = ax.text(j, i, f'{cov_matrix[i,j]:.3f}',
                      ha='center', va='center', color='black', fontsize=9)

# MIDDLE: Correlation matrix heatmap
ax = axes[1]
im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels([f'T{i+1}' for i in range(5)])
ax.set_yticklabels([f'T{i+1}' for i in range(5)])
ax.set_title('Correlation Matrix ρ', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax)

# Add values
for i in range(5):
    for j in range(5):
        text = ax.text(j, i, f'{corr_matrix[i,j]:.2f}',
                      ha='center', va='center', 
                      color='white' if corr_matrix[i,j] > 0.5 else 'black', 
                      fontsize=9, fontweight='bold')

# RIGHT: Variance breakdown
ax = axes[2]
components = ['Single\nTree\nσ²', 'RF\n(exact)', 'RF\n(formula)']
values = [avg_var, var_rf, var_rf_formula]
colors = ['red', 'green', 'blue']

bars = ax.bar(components, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Variance', fontsize=12, fontweight='bold')
ax.set_title('Variance Comparison', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.002,
           f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')

# Show reduction
reduction = (avg_var - var_rf) / avg_var * 100
ax.text(0.5, 0.8, f'Reduction:\n{reduction:.1f}%',
       transform=ax.transAxes, ha='center', fontsize=11,
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.suptitle('📊 Manual Variance Calculation Verification', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

print("\n" + "═" * 80)
print("🎉 ALL CALCULATIONS VERIFIED!")
print("═" * 80)
print("\n💡 KEY TAKEAWAYS:")
print(f"   • High correlation (ρ={avg_corr:.3f}) → Less diversity")
print(f"   • T1 = T5 (identical) → ρ = 1.000")
print(f"   • Variance reduction: {reduction:.1f}%")
print(f"   • More diverse trees → Lower ρ → Better RF!")
print("\n" + "═" * 80)

# COMMAND ----------

# DBTITLE 1,📚 Complete Formula Reference: Basic to Advanced
# MAGIC %md
# MAGIC # 📚 Complete Formula Reference: Basic to Advanced
# MAGIC
# MAGIC **Sabse basic se leke Random Forest variance tak — sab formulas ek jagah!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟢 Level 1: Mean (Average)
# MAGIC
# MAGIC ### **Formula:**
# MAGIC ```
# MAGIC μ = X̄ = (1/N) Σ_{i=1}^N Xᵢ
# MAGIC    = (X₁ + X₂ + ... + Xₙ) / N
# MAGIC ```
# MAGIC
# MAGIC ### **Example:**
# MAGIC ```
# MAGIC Data: [2, 4, 6, 8, 10]
# MAGIC Mean = (2+4+6+8+10)/5 = 30/5 = 6
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟡 Level 2: Deviation
# MAGIC
# MAGIC ### **Formula:**
# MAGIC ```
# MAGIC dᵢ = Xᵢ - μ
# MAGIC ```
# MAGIC
# MAGIC ### **Example:**
# MAGIC ```
# MAGIC Data: [2, 4, 6, 8, 10], μ = 6
# MAGIC Deviations: [2-6, 4-6, 6-6, 8-6, 10-6]
# MAGIC           = [-4, -2, 0, +2, +4]
# MAGIC ```
# MAGIC
# MAGIC ### **Property:**
# MAGIC ```
# MAGIC Σ dᵢ = 0  (always!)
# MAGIC -4 - 2 + 0 + 2 + 4 = 0 ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔵 Level 3: Variance
# MAGIC
# MAGIC ### **Population Variance:**
# MAGIC ```
# MAGIC σ² = (1/N) Σ_{i=1}^N (Xᵢ - μ)²
# MAGIC    = (1/N) Σ dᵢ²
# MAGIC ```
# MAGIC
# MAGIC ### **Sample Variance:**
# MAGIC ```
# MAGIC s² = 1/(N-1) Σ_{i=1}^N (Xᵢ - X̄)²
# MAGIC ```
# MAGIC
# MAGIC **N kyun nahi, N-1 kyun?**
# MAGIC * Sample se mean estimate kiya → 1 degree of freedom gayi
# MAGIC * N-1 = degrees of freedom
# MAGIC * Unbiased estimator
# MAGIC
# MAGIC ### **Example:**
# MAGIC ```
# MAGIC Deviations: [-4, -2, 0, +2, +4]
# MAGIC Squared: [16, 4, 0, 4, 16]
# MAGIC
# MAGIC Population: (16+4+0+4+16)/5 = 40/5 = 8
# MAGIC Sample: (16+4+0+4+16)/4 = 40/4 = 10
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔴 Level 4: Standard Deviation
# MAGIC
# MAGIC ### **Formula:**
# MAGIC ```
# MAGIC σ = √σ²
# MAGIC   = √[(1/N) Σ (Xᵢ - μ)²]
# MAGIC ```
# MAGIC
# MAGIC ### **Example:**
# MAGIC ```
# MAGIC σ = √8 = 2.83  (population)
# MAGIC s = √10 = 3.16 (sample)
# MAGIC ```
# MAGIC
# MAGIC **Why SD?**
# MAGIC * Variance = units²
# MAGIC * SD = units (original scale!)
# MAGIC * Easier to interpret
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟪 Level 5: Alternate Variance Formula
# MAGIC
# MAGIC ### **Formula:**
# MAGIC ```
# MAGIC σ² = E[X²] - (E[X])²
# MAGIC    = (Σ Xᵢ²)/N - μ²
# MAGIC ```
# MAGIC
# MAGIC **Derivation:**
# MAGIC ```
# MAGIC σ² = E[(X-μ)²]
# MAGIC    = E[X² - 2Xμ + μ²]
# MAGIC    = E[X²] - 2μE[X] + μ²
# MAGIC    = E[X²] - 2μ² + μ²
# MAGIC    = E[X²] - μ² ✅
# MAGIC ```
# MAGIC
# MAGIC ### **Example:**
# MAGIC ```
# MAGIC Data: [2, 4, 6, 8, 10]
# MAGIC
# MAGIC E[X²] = (4+16+36+64+100)/5 = 220/5 = 44
# MAGIC μ² = 6² = 36
# MAGIC
# MAGIC σ² = 44 - 36 = 8 ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟫 Level 6: Covariance
# MAGIC
# MAGIC ### **Formula (Do Variables X aur Y):**
# MAGIC ```
# MAGIC Cov(X,Y) = (1/N) Σ_{i=1}^N (Xᵢ - μₓ) (Yᵢ - μʏ)
# MAGIC ```
# MAGIC
# MAGIC ### **Alternate Form:**
# MAGIC ```
# MAGIC Cov(X,Y) = E[XY] - E[X] · E[Y]
# MAGIC ```
# MAGIC
# MAGIC ### **Example:**
# MAGIC ```
# MAGIC X = Tree 1: [0.8, 0.2, 0.9, 0.3, 0.7], μ=0.58
# MAGIC Y = Tree 2: [0.7, 0.3, 0.8, 0.4, 0.6], μ=0.56
# MAGIC
# MAGIC Products of deviations:
# MAGIC (+0.22)(+0.14) = +0.0308
# MAGIC (-0.38)(-0.26) = +0.0988
# MAGIC (+0.32)(+0.24) = +0.0768
# MAGIC (-0.28)(-0.16) = +0.0448
# MAGIC (+0.12)(+0.04) = +0.0048
# MAGIC
# MAGIC Cov = 0.2560/5 = +0.0512 ✅
# MAGIC ```
# MAGIC
# MAGIC ### **Interpretation:**
# MAGIC * **Cov > 0:** Dono saath badhte hain (positive relationship)
# MAGIC * **Cov = 0:** No relationship
# MAGIC * **Cov < 0:** Ek badhe, doosra gire (negative relationship)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟬 Level 7: Correlation (ρ)
# MAGIC
# MAGIC ### **Formula:**
# MAGIC ```
# MAGIC ρₓʏ = Cov(X,Y) / (σₓ × σʏ)
# MAGIC ```
# MAGIC
# MAGIC **Range:** -1 ≤ ρ ≤ +1
# MAGIC
# MAGIC ### **Interpretation:**
# MAGIC * **ρ = +1:** Perfect positive (X badhe → Y badhe)
# MAGIC * **ρ = 0:** No linear relationship
# MAGIC * **ρ = -1:** Perfect negative (X badhe → Y gire)
# MAGIC
# MAGIC ### **Example:**
# MAGIC ```
# MAGIC ρ(T1,T2) = 0.0512 / (0.2786 × 0.1855)
# MAGIC          = 0.0512 / 0.0517
# MAGIC          = 0.990 ✅
# MAGIC ```
# MAGIC
# MAGIC **Why Correlation?**
# MAGIC * Covariance depends on scale (units)
# MAGIC * Correlation is **normalized** (no units!)
# MAGIC * Easier to compare across datasets
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟣 Level 8: Variance Properties
# MAGIC
# MAGIC ### **Property 1: Constant Multiply**
# MAGIC ```
# MAGIC Var(cX) = c² × Var(X)
# MAGIC ```
# MAGIC
# MAGIC **Example:**
# MAGIC ```
# MAGIC X = [2, 4, 6], Var(X) = 2.67
# MAGIC Var(2X) = 4 × 2.67 = 10.67 ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Property 2: Constant Add**
# MAGIC ```
# MAGIC Var(X + c) = Var(X)
# MAGIC ```
# MAGIC
# MAGIC **Constant add karne se spread nahi badti!**
# MAGIC
# MAGIC **Example:**
# MAGIC ```
# MAGIC [2, 4, 6] → [12, 14, 16]  (c=10 add kiya)
# MAGIC Var same = 2.67 ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Property 3: Sum of Independent**
# MAGIC ```
# MAGIC Var(X + Y) = Var(X) + Var(Y)  (if independent)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Property 4: Sum of Correlated**
# MAGIC ```
# MAGIC Var(X + Y) = Var(X) + Var(Y) + 2·Cov(X,Y)
# MAGIC ```
# MAGIC
# MAGIC **Covariance term captures interaction!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟪 Level 9: Covariance Matrix Σ
# MAGIC
# MAGIC ### **For B Variables:**
# MAGIC ```
# MAGIC Σ = [ σ₁²        Cov(X₁,X₂)  ...  Cov(X₁,Xᴮ) ]
# MAGIC     [ Cov(X₂,X₁)  σ₂²        ...  Cov(X₂,Xᴮ) ]
# MAGIC     [ ...        ...        ...  ...       ]
# MAGIC     [ Cov(Xᴮ,X₁)  Cov(Xᴮ,X₂)  ...  σᴮ²       ]
# MAGIC ```
# MAGIC
# MAGIC ### **Properties:**
# MAGIC * **Diagonal:** σᵢ² (own variance)
# MAGIC * **Off-diagonal:** Cov(Xᵢ, Xⱼ) = ρσᵢσⱼ
# MAGIC * **Symmetric:** Σᵢⱼ = Σⱼᵢ
# MAGIC
# MAGIC ### **Example (5 Trees):**
# MAGIC ```
# MAGIC       T1     T2     T3     T4     T5
# MAGIC T1 [ 0.0776 0.0512 0.0912 0.0248 0.0776 ]
# MAGIC T2 [ 0.0512 0.0344 0.0627 0.0170 0.0512 ]
# MAGIC T3 [ 0.0912 0.0627 0.1144 0.0304 0.0912 ]
# MAGIC T4 [ 0.0248 0.0170 0.0304 0.0104 0.0248 ]
# MAGIC T5 [ 0.0776 0.0512 0.0912 0.0248 0.0776 ]
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟫 Level 10: Variance of Linear Combination
# MAGIC
# MAGIC ### **Scalar Case:**
# MAGIC ```
# MAGIC Var( Σ_{i=1}^B aᵢ Xᵢ ) = aᵀ Σ a
# MAGIC ```
# MAGIC
# MAGIC **Where:**
# MAGIC * `a` = vector of weights [a₁, a₂, ..., aᴮ]
# MAGIC * `Σ` = covariance matrix
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **RF Average (aᵢ = 1/B for all i):**
# MAGIC ```
# MAGIC Var(X̄) = Var( (1/B) Σ Xᵯ )
# MAGIC         = (1/B²) 1ᵀ Σ 1
# MAGIC ```
# MAGIC
# MAGIC **Where:** `1` = vector of ones [1, 1, ..., 1]
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟬 Level 11: 1ᵀΣ1 Expansion
# MAGIC
# MAGIC ### **When All Same Variance σ² and Correlation ρ:**
# MAGIC
# MAGIC ```
# MAGIC Σ = σ² [ 1   ρ   ρ  ... ]
# MAGIC          [ ρ   1   ρ  ... ]
# MAGIC          [ ρ   ρ   1  ... ]
# MAGIC          [ ... ... ... ... ]
# MAGIC ```
# MAGIC
# MAGIC ### **Sum of All Elements:**
# MAGIC ```
# MAGIC 1ᵀΣ1 = Diagonal terms + Off-diagonal terms
# MAGIC       = B × σ² + (B²-B) × ρσ²
# MAGIC       = Bσ² + (B²-B)ρσ²
# MAGIC       = Bσ²(1 + (B-1)ρ)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Level 12: RF Variance Derivation (FINAL)
# MAGIC
# MAGIC ### **Start:**
# MAGIC ```
# MAGIC Var(X̄) = (1/B²) [Bσ² + (B²-B)ρσ²]
# MAGIC ```
# MAGIC
# MAGIC ### **Simplify:**
# MAGIC ```
# MAGIC = σ²/B + [(B²-B)/(B²)]ρσ²
# MAGIC
# MAGIC = σ²/B + [(B-1)/B]ρσ²
# MAGIC
# MAGIC = σ²/B + [1 - 1/B]ρσ²
# MAGIC
# MAGIC = σ²/B + ρσ² - ρσ²/B
# MAGIC
# MAGIC = ρσ² + (σ² - ρσ²)/B
# MAGIC
# MAGIC = ρσ² + σ²(1-ρ)/B
# MAGIC ```
# MAGIC
# MAGIC ### **🎉 FINAL FORMULA:**
# MAGIC
# MAGIC ```
# MAGIC ╔═════════════════════════════════════╗
# MAGIC ║  Var(X̄) = ρσ² + (1-ρ)σ²/B  ║
# MAGIC ╚═════════════════════════════════════╝
# MAGIC
# MAGIC Where:
# MAGIC   ρ = correlation between trees
# MAGIC   σ² = variance of individual trees
# MAGIC   B = number of trees
# MAGIC ```
# MAGIC
# MAGIC ### **Components:**
# MAGIC
# MAGIC * **ρσ²** = Irreducible variance (due to correlation)
# MAGIC * **(1-ρ)σ²/B** = Reducible variance (decreases with B!)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📝 Summary Table: All Formulas
# MAGIC
# MAGIC | **Level** | **Formula** | **Expression** | **Matlab** |
# MAGIC |-----------|-------------|----------------|------------|
# MAGIC | 1 | Mean | μ = ΣXᵢ/N | Average |
# MAGIC | 2 | Deviation | dᵢ = Xᵢ - μ | Distance from mean |
# MAGIC | 3 | Variance (pop) | σ² = Σdᵢ²/N | Avg squared deviation |
# MAGIC | 3 | Variance (sample) | s² = Σdᵢ²/(N-1) | Unbiased estimate |
# MAGIC | 4 | SD | σ = √σ² | Original units |
# MAGIC | 5 | Alternate Var | E[X²] - (E[X])² | Computation shortcut |
# MAGIC | 6 | Covariance | Cov = Σdₓdʏ/N | Shared movement |
# MAGIC | 7 | Correlation | ρ = Cov/(σₓσʏ) | Normalized (-1 to 1) |
# MAGIC | 8a | Var(cX) | c²·Var(X) | Scale effect |
# MAGIC | 8b | Var(X+Y) indep | Var(X) + Var(Y) | No interaction |
# MAGIC | 8c | Var(X+Y) corr | Var(X) + Var(Y) + 2Cov | With interaction |
# MAGIC | 9 | Cov matrix | Σ (B×B) | All pairs |
# MAGIC | 10 | Linear combo | aᵀΣa | General variance |
# MAGIC | 11 | RF average | 1ᵀΣ1/B² | Average variance |
# MAGIC | **12** | **RF formula** | **ρσ² + (1-ρ)σ²/B** | **Final result** |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔗 Connections: Ek Chain Mein
# MAGIC
# MAGIC ```
# MAGIC Mean
# MAGIC   ↓
# MAGIC Deviation = Xᵢ - μ
# MAGIC   ↓
# MAGIC Variance = mean(deviation²)    ← ek variable
# MAGIC   ↓
# MAGIC SD = √Variance
# MAGIC   ↓
# MAGIC Covariance = mean(d_X × d_Y)  ← do variables
# MAGIC   ↓
# MAGIC Correlation = Cov/(σₓ×σʏ)     ← normalized
# MAGIC   ↓
# MAGIC Cov Matrix Σ                   ← B variables
# MAGIC   ↓
# MAGIC Var(aᵀX) = aᵀΣa               ← vector form
# MAGIC   ↓
# MAGIC RF: a=1/B → Var(X̄) = 1ᵀΣ1/B²
# MAGIC   ↓
# MAGIC = ρσ² + (1-ρ)σ²/B             ← final formula
# MAGIC ```
# MAGIC
# MAGIC **Sab ek chain mein connected hain — Mean se RF variance tak!** 👍

# COMMAND ----------

# DBTITLE 1,🧮 Matrix Derivation: RF Variance Formula
# MAGIC %md
# MAGIC # 🧮 Matrix Derivation: RF Variance Formula
# MAGIC
# MAGIC **Matrix se step-by-step samjho kaise formula banta hai!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Setup
# MAGIC
# MAGIC **Simple example:**
# MAGIC * **B = 3 trees**
# MAGIC * Har tree ka prediction = X₁, X₂, X₃
# MAGIC
# MAGIC **RF average:**
# MAGIC ```
# MAGIC X̄ = (X₁ + X₂ + X₃) / 3
# MAGIC   = (1/B) × 1ᵀX
# MAGIC ```
# MAGIC
# MAGIC **Where:**
# MAGIC * `1 = [1, 1, 1]ᵀ` ← ones vector (column)
# MAGIC * `X = [X₁, X₂, X₃]ᵀ` ← tree predictions (column)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📐 Step 1: Var(X̄) Ko Matrix Form Mein Likho
# MAGIC
# MAGIC ```
# MAGIC X̄ = (1/B) 1ᵀX
# MAGIC ```
# MAGIC
# MAGIC **Property:** `Var(cZ) = c² Var(Z)`
# MAGIC
# MAGIC ```
# MAGIC Var(X̄) = Var( (1/B) 1ᵀX )
# MAGIC        = (1/B²) Var(1ᵀX)
# MAGIC ```
# MAGIC
# MAGIC **Linear combination ka variance:**
# MAGIC ```
# MAGIC Var(aᵀX) = aᵀ Σ a
# MAGIC ```
# MAGIC
# MAGIC **a = 1 daalo:**
# MAGIC ```
# MAGIC Var(1ᵀX) = 1ᵀ Σ 1
# MAGIC ```
# MAGIC
# MAGIC **Therefore:**
# MAGIC ```
# MAGIC ┌─────────────────────────┐
# MAGIC │ Var(X̄) = (1/B²) 1ᵀΣ1  │
# MAGIC └─────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔢 Step 2: Σ Matrix Banao
# MAGIC
# MAGIC **Parameters:**
# MAGIC * B = 3
# MAGIC * σ² = 1.0
# MAGIC * ρ = 0.3
# MAGIC
# MAGIC **Covariance matrix:**
# MAGIC
# MAGIC ```
# MAGIC Σ = [ σ²    ρσ²   ρσ²  ]   [ 1.0  0.3  0.3 ]
# MAGIC     [ ρσ²   σ²    ρσ²  ] = [ 0.3  1.0  0.3 ]
# MAGIC     [ ρσ²   ρσ²   σ²   ]   [ 0.3  0.3  1.0 ]
# MAGIC ```
# MAGIC
# MAGIC **Components:**
# MAGIC * **Diagonal** = σ² = 1.0 (apni variance)
# MAGIC * **Off-diagonal** = ρσ² = 0.3 (shared variance)
# MAGIC
# MAGIC **Count:**
# MAGIC * Diagonal cells = **B = 3**
# MAGIC * Off-diagonal cells = **B² - B = 9 - 3 = 6**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔄 Step 3: Σ1 Calculate Karo
# MAGIC
# MAGIC **Matrix-vector multiplication:**
# MAGIC
# MAGIC ```
# MAGIC Σ1 = [ 1.0  0.3  0.3 ] [ 1 ]
# MAGIC      [ 0.3  1.0  0.3 ] [ 1 ]
# MAGIC      [ 0.3  0.3  1.0 ] [ 1 ]
# MAGIC ```
# MAGIC
# MAGIC **Har row ka sum:**
# MAGIC
# MAGIC ```
# MAGIC Row 1: 1.0 + 0.3 + 0.3 = 1.6
# MAGIC Row 2: 0.3 + 1.0 + 0.3 = 1.6
# MAGIC Row 3: 0.3 + 0.3 + 1.0 = 1.6
# MAGIC ```
# MAGIC
# MAGIC **Result:**
# MAGIC ```
# MAGIC Σ1 = [ 1.6 ]
# MAGIC      [ 1.6 ]
# MAGIC      [ 1.6 ]
# MAGIC ```
# MAGIC
# MAGIC ### **💡 Kyun Sab Rows Same Hain?**
# MAGIC
# MAGIC Har row mein:
# MAGIC * **1 diagonal term** = σ² = 1.0
# MAGIC * **(B-1) off-diagonal** = (B-1)ρσ² = 2 × 0.3 = 0.6
# MAGIC
# MAGIC ```
# MAGIC Row sum = σ² + (B-1)ρσ²
# MAGIC         = 1.0 + 0.6
# MAGIC         = 1.6 ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ➕ Step 4: 1ᵀ(Σ1) Calculate Karo
# MAGIC
# MAGIC **Vector dot product:**
# MAGIC
# MAGIC ```
# MAGIC 1ᵀ(Σ1) = [ 1  1  1 ] [ 1.6 ]
# MAGIC                       [ 1.6 ]
# MAGIC                       [ 1.6 ]
# MAGIC
# MAGIC        = 1×1.6 + 1×1.6 + 1×1.6
# MAGIC        = 4.8
# MAGIC ```
# MAGIC
# MAGIC ### **📊 Yahi 1ᵀΣ1 Hai = Sum of ALL Elements of Σ:**
# MAGIC
# MAGIC ```
# MAGIC Σ matrix ke sab elements:
# MAGIC 1.0 + 0.3 + 0.3 = 1.6  (row 1)
# MAGIC 0.3 + 1.0 + 0.3 = 1.6  (row 2)
# MAGIC 0.3 + 0.3 + 1.0 = 1.6  (row 3)
# MAGIC                  ─────
# MAGIC Total           = 4.8 ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🧩 Step 5: Bσ² + (B²-B)ρσ² Kaise Aaya
# MAGIC
# MAGIC **Same 4.8 ko formula se nikalo:**
# MAGIC
# MAGIC ### **Diagonal Terms:**
# MAGIC ```
# MAGIC B cells × σ² = 3 × 1.0 = 3.0
# MAGIC ```
# MAGIC
# MAGIC ### **Off-Diagonal Terms:**
# MAGIC ```
# MAGIC (B²-B) cells × ρσ² = (9-3) × 0.3
# MAGIC                     = 6 × 0.3
# MAGIC                     = 1.8
# MAGIC ```
# MAGIC
# MAGIC ### **Total:**
# MAGIC ```
# MAGIC 1ᵀΣ1 = 3.0 + 1.8 = 4.8 ✅
# MAGIC ```
# MAGIC
# MAGIC ### **📐 General Formula:**
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────┐
# MAGIC │ 1ᵀΣ1 = Bσ² + (B²-B)ρσ²             │
# MAGIC │      = Bσ² + B(B-1)ρσ²             │
# MAGIC │      = Bσ²[1 + (B-1)ρ]             │
# MAGIC └──────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Step 6: Var(X̄) Nikalo
# MAGIC
# MAGIC ```
# MAGIC Var(X̄) = (1/B²) 1ᵀΣ1
# MAGIC        = 4.8 / 9
# MAGIC        = 0.533
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔧 Step 7: Simplify Karke Final Formula
# MAGIC
# MAGIC **Start:**
# MAGIC ```
# MAGIC Var(X̄) = (1/B²) [Bσ² + (B²-B)ρσ²]
# MAGIC ```
# MAGIC
# MAGIC **Distribute:**
# MAGIC ```
# MAGIC = (Bσ²)/B² + [(B²-B)ρσ²]/B²
# MAGIC ```
# MAGIC
# MAGIC **First term:**
# MAGIC ```
# MAGIC (Bσ²)/B² = σ²/B
# MAGIC ```
# MAGIC
# MAGIC **Second term:**
# MAGIC ```
# MAGIC [(B²-B)ρσ²]/B² = [(B-1)/B]ρσ²
# MAGIC                 = [1 - 1/B]ρσ²
# MAGIC ```
# MAGIC
# MAGIC **Combine:**
# MAGIC ```
# MAGIC Var(X̄) = σ²/B + [1 - 1/B]ρσ²
# MAGIC        = σ²/B + ρσ² - ρσ²/B
# MAGIC        = ρσ² + σ²/B - ρσ²/B
# MAGIC        = ρσ² + (σ² - ρσ²)/B
# MAGIC        = ρσ² + σ²(1-ρ)/B
# MAGIC ```
# MAGIC
# MAGIC ### **🎉 FINAL FORMULA:**
# MAGIC
# MAGIC ```
# MAGIC ╔═════════════════════════════════════╗
# MAGIC ║ Var(X̄) = ρσ² + (1-ρ)σ²/B          ║
# MAGIC ╚═════════════════════════════════════╝
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Verify With Numbers
# MAGIC
# MAGIC ```
# MAGIC ρσ² + (1-ρ)σ²/B
# MAGIC = 0.3×1 + (0.7×1)/3
# MAGIC = 0.3 + 0.233
# MAGIC = 0.533 ✅
# MAGIC ```
# MAGIC
# MAGIC **Matches our matrix calculation!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Poora Flow Ek Jagah
# MAGIC
# MAGIC ```
# MAGIC X̄ = 1ᵀX/B
# MAGIC     ↓
# MAGIC Var(X̄) = (1/B²) 1ᵀΣ1
# MAGIC     ↓
# MAGIC 1ᵀΣ1 = Bσ²      + (B²-B)ρσ²
# MAGIC        ⎺⎺⎺⎺⎺⎺⎺⎺   ⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺
# MAGIC        diagonal   off-diagonal
# MAGIC     ↓
# MAGIC Var(X̄) = [Bσ² + (B²-B)ρσ²] / B²
# MAGIC     ↓
# MAGIC        = ρσ² + (1-ρ)σ²/B
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔑 Matrix Se Formula Ka Connection
# MAGIC
# MAGIC ### **1ᵀΣ1 = Sum of All Elements of Σ**
# MAGIC
# MAGIC **Σ matrix (B×B):**
# MAGIC * **B diagonal cells** = σ² each → Total **Bσ²**
# MAGIC * **(B²-B) off-diagonal cells** = ρσ² each → Total **(B²-B)ρσ²**
# MAGIC
# MAGIC **Divide by B²:**
# MAGIC * Bσ²/B² = **σ²/B** ← reducible term (decreases with B)
# MAGIC * (B²-B)ρσ²/B² = **(1-1/B)ρσ²** ≈ **ρσ²** ← irreducible term (floor)
# MAGIC
# MAGIC **Final:**
# MAGIC ```
# MAGIC Var(X̄) = ρσ² + (1-ρ)σ²/B
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 Key Insights
# MAGIC
# MAGIC ### **Matrix View:**
# MAGIC * Σ matrix encodes all pairwise relationships
# MAGIC * 1ᵀΣ1 sums all these relationships
# MAGIC * Diagonal contributes Bσ²
# MAGIC * Off-diagonal contributes (B²-B)ρσ²
# MAGIC
# MAGIC ### **Formula View:**
# MAGIC * **ρσ²** = irreducible (can't reduce even with infinite trees)
# MAGIC * **(1-ρ)σ²/B** = reducible (goes to 0 as B→∞)
# MAGIC * Low ρ = more reduction possible!
# MAGIC
# MAGIC **1ᵀΣ1 = matrix ke sab elements ka sum — diagonal aur off-diagonal count se directly formula nikal aata hai!** 👍

# COMMAND ----------

# DBTITLE 1,Code: Matrix Derivation Demonstration
# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION: MATRIX DERIVATION OF RF VARIANCE FORMULA
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("═" * 80)
print("🧮 MATRIX DERIVATION: RF VARIANCE FORMULA")
print("═" * 80)

# Parameters
B = 3  # Number of trees
sigma_sq = 1.0  # Variance
rho = 0.3  # Correlation

print(f"\n🎯 Setup:")
print(f"   B (trees): {B}")
print(f"   σ²: {sigma_sq}")
print(f"   ρ: {rho}")

# Step 1: Build covariance matrix
Sigma = np.full((B, B), rho * sigma_sq)  # Fill with off-diagonal
np.fill_diagonal(Sigma, sigma_sq)  # Set diagonal

print("\n📐 Step 1 - Covariance Matrix Σ:")
print("\n   (Diagonal = σ², Off-diagonal = ρσ²)\n")
print(pd.DataFrame(Sigma, 
                   index=[f'X{i+1}' for i in range(B)],
                   columns=[f'X{i+1}' for i in range(B)]))

print(f"\n   Cell counts:")
print(f"      Diagonal cells: {B}")
print(f"      Off-diagonal cells: {B**2 - B}")

# Step 2: Create ones vector
ones = np.ones(B)

print("\n🔢 Step 2 - Ones Vector 1:")
print(f"   1 = {ones}")

# Step 3: Compute Σ1
Sigma_1 = Sigma @ ones

print("\n🔄 Step 3 - Matrix-Vector Product Σ1:")
print(f"   Σ1 = {Sigma_1}")
print(f"\n   (Each element is the sum of one row)")

# Verify each row sum
print("\n   Row sums breakdown:")
for i in range(B):
    row_sum = Sigma[i].sum()
    diagonal = sigma_sq
    off_diag_sum = (B-1) * rho * sigma_sq
    print(f"      Row {i+1}: {diagonal:.1f} (diagonal) + {off_diag_sum:.1f} (off-diag) = {row_sum:.1f}")

expected_row_sum = sigma_sq + (B-1) * rho * sigma_sq
print(f"\n   Formula: σ² + (B-1)ρσ² = {sigma_sq} + {(B-1)*rho*sigma_sq:.1f} = {expected_row_sum:.1f} ✅")

# Step 4: Compute 1'Σ1
ones_Sigma_ones = ones @ Sigma @ ones

print("\n➕ Step 4 - Quadratic Form 1ᵀΣ1:")
print(f"   1ᵀΣ1 = {ones_Sigma_ones:.4f}")

# Verify by summing all elements
sum_all_elements = Sigma.sum()
print(f"\n   Verification (sum of all elements): {sum_all_elements:.4f} ✅")

# Step 5: Break down into components
diagonal_sum = B * sigma_sq
off_diagonal_sum = (B**2 - B) * rho * sigma_sq

print("\n🧩 Step 5 - Component Breakdown:")
print(f"\n   Diagonal terms:")
print(f"      Count: {B}")
print(f"      Value each: {sigma_sq}")
print(f"      Total: {B} × {sigma_sq} = {diagonal_sum:.4f}")

print(f"\n   Off-diagonal terms:")
print(f"      Count: B² - B = {B**2} - {B} = {B**2 - B}")
print(f"      Value each: {rho * sigma_sq}")
print(f"      Total: {B**2 - B} × {rho * sigma_sq} = {off_diagonal_sum:.4f}")

print(f"\n   Sum: {diagonal_sum:.4f} + {off_diagonal_sum:.4f} = {diagonal_sum + off_diagonal_sum:.4f} ✅")

# Step 6: Compute Var(X̄)
var_xbar_matrix = ones_Sigma_ones / (B**2)

print("\n🎯 Step 6 - Random Forest Variance:")
print(f"\n   Var(X̄) = 1ᵀΣ1 / B²")
print(f"          = {ones_Sigma_ones:.4f} / {B**2}")
print(f"          = {var_xbar_matrix:.4f}")

# Step 7: Verify with formula
var_xbar_formula = rho * sigma_sq + (1 - rho) * sigma_sq / B

print("\n✅ Step 7 - Verify With Formula:")
print(f"\n   Formula: Var(X̄) = ρσ² + (1-ρ)σ²/B")
print(f"          = {rho}×{sigma_sq} + {1-rho}×{sigma_sq}/{B}")
print(f"          = {rho * sigma_sq:.4f} + {(1-rho) * sigma_sq / B:.4f}")
print(f"          = {var_xbar_formula:.4f}")

print(f"\n   Matrix method:  {var_xbar_matrix:.6f}")
print(f"   Formula method: {var_xbar_formula:.6f}")
print(f"   Match: {'✅' if np.isclose(var_xbar_matrix, var_xbar_formula) else '❌'}")

# Show simplification steps
print("\n🔧 Algebraic Simplification:")
print(f"\n   Start: Var(X̄) = [Bσ² + (B²-B)ρσ²] / B²")
print(f"\n   = Bσ²/B² + (B²-B)ρσ²/B²")
print(f"   = σ²/B + (B-1)ρσ²/B")
print(f"   = σ²/B + ρσ² - ρσ²/B")
print(f"   = ρσ² + (σ² - ρσ²)/B")
print(f"   = ρσ² + σ²(1-ρ)/B")
print(f"   = ρσ² + (1-ρ)σ²/B ✅")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# LEFT: Covariance matrix with annotations
ax = axes[0]
im = ax.imshow(Sigma, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(B))
ax.set_yticks(range(B))
ax.set_xticklabels([f'X{i+1}' for i in range(B)])
ax.set_yticklabels([f'X{i+1}' for i in range(B)])
ax.set_title('Covariance Matrix Σ', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax)

# Add values and annotations
for i in range(B):
    for j in range(B):
        value = Sigma[i, j]
        if i == j:
            label = f'{value:.1f}\n(σ²)'
            color = 'black'
        else:
            label = f'{value:.1f}\n(ρσ²)'
            color = 'darkred'
        ax.text(j, i, label, ha='center', va='center', 
               color=color, fontsize=10, fontweight='bold')

# Add count annotations
ax.text(0.5, -0.15, f'Diagonal: {B} cells × σ² = {diagonal_sum:.1f}',
       transform=ax.transAxes, ha='center', fontsize=9,
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax.text(0.5, -0.25, f'Off-diagonal: {B**2-B} cells × ρσ² = {off_diagonal_sum:.1f}',
       transform=ax.transAxes, ha='center', fontsize=9,
       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# MIDDLE: Flow diagram
ax = axes[1]
ax.axis('off')

flow_text = f"""
🔄 MATRIX FLOW:

X̄ = (1/B) 1ᵀX
    ↓
Var(X̄) = (1/B²) 1ᵀΣ1
    ↓
1ᵀΣ1 = {ones_Sigma_ones:.2f}
    ↓
     = Bσ² + (B²-B)ρσ²
     = {diagonal_sum:.2f} + {off_diagonal_sum:.2f}
    ↓
Var(X̄) = {ones_Sigma_ones:.2f} / {B**2}
       = {var_xbar_matrix:.4f}
    ↓
     = ρσ² + (1-ρ)σ²/B
     = {rho*sigma_sq:.3f} + {(1-rho)*sigma_sq/B:.3f}
     = {var_xbar_formula:.4f} ✅
"""

ax.text(0.5, 0.5, flow_text, fontsize=11, family='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1.5),
       verticalalignment='center', horizontalalignment='center')
ax.set_title('Derivation Flow', fontsize=13, fontweight='bold')

# RIGHT: Component breakdown
ax = axes[2]

components = ['Irreducible\n(ρσ²)', 'Reducible\n((1-ρ)σ²/B)', 'Total\nVar(X̄)']
values = [rho * sigma_sq, (1-rho) * sigma_sq / B, var_xbar_formula]
colors = ['red', 'green', 'blue']

bars = ax.bar(components, values, color=colors, alpha=0.7, 
             edgecolor='black', linewidth=2)
ax.set_ylabel('Variance', fontsize=12, fontweight='bold')
ax.set_title('Variance Components', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
           f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')

# Show stacked bar for total
ax.bar(['Total\nVar(X̄)'], [rho * sigma_sq], color='red', alpha=0.5)
ax.bar(['Total\nVar(X̄)'], [(1-rho) * sigma_sq / B], 
      bottom=[rho * sigma_sq], color='green', alpha=0.5)

# Add equation
ax.text(0.5, 0.85, f'Var(X̄) = {rho*sigma_sq:.3f} + {(1-rho)*sigma_sq/B:.3f}',
       transform=ax.transAxes, ha='center', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.suptitle('🧮 Matrix Derivation of RF Variance Formula', 
            fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

print("\n" + "═" * 80)
print("🎉 MATRIX DERIVATION COMPLETE!")
print("═" * 80)

print("\n💡 KEY INSIGHTS:")
print(f"   • Σ matrix encodes all pairwise relationships")
print(f"   • 1ᵀΣ1 = sum of all {B**2} elements = {ones_Sigma_ones:.2f}")
print(f"   • Diagonal contributes: {diagonal_sum:.2f}")
print(f"   • Off-diagonal contributes: {off_diagonal_sum:.2f}")
print(f"   • Divide by B² = {B**2} gives variance: {var_xbar_matrix:.4f}")
print(f"   • Formula directly gives same result! ✅")

print("\n🔑 Components:")
print(f"   • Irreducible (ρσ²): {rho*sigma_sq:.4f} ({rho*sigma_sq/var_xbar_formula*100:.1f}%)")
print(f"   • Reducible ((1-ρ)σ²/B): {(1-rho)*sigma_sq/B:.4f} ({(1-rho)*sigma_sq/B/var_xbar_formula*100:.1f}%)")
print(f"   • Lower ρ → More reduction possible!")

print("\n" + "═" * 80)

# COMMAND ----------

# DBTITLE 1,🌲 Part 1: Random Forest Variance Reduction
# MAGIC %md
# MAGIC # 🌲 PART 1: Random Forest Variance Reduction
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📝 The Core Formula
# MAGIC
# MAGIC ### **Variance of Average of Correlated Variables:**
# MAGIC
# MAGIC ```
# MAGIC Var(∑_{b=1}^B X_b / B) = ∑_{b=1}^B Var(X_b)/B² + ∑_{i≠j} Cov(X_i, X_j)/B²
# MAGIC ```
# MAGIC
# MAGIC ### **For identically distributed trees** (Var(X_b) = σ², Cov(X_i, X_j) = ρσ²):
# MAGIC
# MAGIC ```
# MAGIC Var(RF) = B·σ²/B² + [B(B-1)]·ρσ²/B²
# MAGIC         = σ²/B + (B-1)ρσ²/B
# MAGIC         = σ²/B + ρσ² - ρσ²/B
# MAGIC         = ρσ² + (1-ρ)σ²/B  ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔑 Key Insights
# MAGIC
# MAGIC ### **Components:**
# MAGIC * **ρσ²**: Irreducible variance (due to correlation)
# MAGIC * **(1-ρ)σ²/B**: Reducible variance (decreases with more trees!)
# MAGIC
# MAGIC ### **Special Cases:**
# MAGIC
# MAGIC **1. Perfect correlation (ρ = 1):**
# MAGIC ```
# MAGIC Var(RF) = 1·σ² + 0·σ²/B = σ²
# MAGIC → Same as single tree! No benefit from ensemble.
# MAGIC ```
# MAGIC
# MAGIC **2. Zero correlation (ρ = 0):**
# MAGIC ```
# MAGIC Var(RF) = 0 + σ²/B = σ²/B
# MAGIC → Variance reduces linearly with B! Maximum benefit.
# MAGIC ```
# MAGIC
# MAGIC **3. Realistic case (ρ = 0.30, B = 10):**
# MAGIC ```
# MAGIC Var(RF) = 0.30·σ² + 0.70·σ²/10
# MAGIC         = 0.30·σ² + 0.07·σ²
# MAGIC         = 0.37·σ²
# MAGIC         
# MAGIC Reduction: (1.0 - 0.37)/1.0 = 63%! 🎉
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Why Random Features?
# MAGIC
# MAGIC **Goal:** Reduce ρ (correlation between trees)
# MAGIC
# MAGIC * Bootstrap sampling → Different data → Different trees
# MAGIC * Random feature selection → Forces diversity
# MAGIC * Lower ρ → More variance reduction
# MAGIC
# MAGIC **Trade-off:**
# MAGIC * Too much randomness → Trees become weak (↑σ²)
# MAGIC * Too little randomness → Trees correlated (↑ρ)
# MAGIC * **Sweet spot: √p features at each split**

# COMMAND ----------

# DBTITLE 1,Demo: RF Variance Reduction with Real Data
# ═══════════════════════════════════════════════════════════════════════════════
# DEMO: RANDOM FOREST VARIANCE REDUCTION
# ═══════════════════════════════════════════════════════════════════════════════

# Generate synthetic data
np.random.seed(42)
n_samples = 200
X = np.random.randn(n_samples, 10)  # 10 features
y_true = 2*X[:, 0] + 1.5*X[:, 1] - X[:, 2] + 5  # True function
y = y_true + np.random.randn(n_samples) * 2  # Add noise

print("✅ Synthetic data generated")
print(f"   Samples: {n_samples}")
print(f"   Features: {X.shape[1]}")
print(f"   True function: y = 2·X₀ + 1.5·X₁ - X₂ + 5 + noise")

# Train individual trees and RF
B_max = 50
individual_predictions = []
single_tree_errors = []

# Train B individual trees
for b in range(B_max):
    # Bootstrap sample
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_boot = X[indices]
    y_boot = y[indices]
    
    # Train tree
    tree = DecisionTreeRegressor(max_depth=10, random_state=b)
    tree.fit(X_boot, y_boot)
    
    # Predictions
    pred = tree.predict(X)
    individual_predictions.append(pred)
    
    # Error
    mse = np.mean((pred - y_true)**2)
    single_tree_errors.append(mse)

individual_predictions = np.array(individual_predictions)  # Shape: (B, n_samples)

# Calculate RF predictions for different B
rf_errors = []
for B in range(1, B_max + 1):
    rf_pred = individual_predictions[:B].mean(axis=0)
    mse = np.mean((rf_pred - y_true)**2)
    rf_errors.append(mse)

# Calculate correlation between trees
tree_correlations = np.corrcoef(individual_predictions)
rho = tree_correlations[np.triu_indices_from(tree_correlations, k=1)].mean()

# Calculate variance of individual trees
sigma_sq = np.var(single_tree_errors)

print(f"\n📊 MEASUREMENTS:")
print(f"   Average single tree MSE: {np.mean(single_tree_errors):.4f}")
print(f"   Variance of tree predictions (σ²): {np.var(individual_predictions):.4f}")
print(f"   Average correlation between trees (ρ): {rho:.4f}")
print(f"   RF with B={B_max} MSE: {rf_errors[-1]:.4f}")

# Theoretical variance calculation
print(f"\n📝 THEORETICAL VARIANCE REDUCTION:")
print(f"   Using formula: Var(RF) = ρσ² + (1-ρ)σ²/B")

sigma_sq_norm = 1.0  # Normalize for clarity
for B in [1, 5, 10, 20, 50]:
    var_rf = rho * sigma_sq_norm + (1 - rho) * sigma_sq_norm / B
    reduction = (1.0 - var_rf) * 100
    print(f"   B={B:2d}: Var(RF) = {rho:.2f} + {(1-rho):.2f}/{B:2d} = {var_rf:.4f} | Reduction: {reduction:.1f}%")

print(f"\n💡 KEY INSIGHT:")
print(f"   Single tree variance: 1.000")
print(f"   RF (B=10) variance: {rho + (1-rho)/10:.4f}")
print(f"   Variance reduction: {(1.0 - (rho + (1-rho)/10)) * 100:.1f}%")
print(f"   As B→∞: Var(RF) → {rho:.4f} (irreducible!)")

# COMMAND ----------

# DBTITLE 1,Visualization: RF Variance vs Number of Trees
# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION: RF VARIANCE REDUCTION
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# LEFT: MSE vs Number of Trees
ax = axes[0]
B_range = range(1, B_max + 1)
ax.plot(B_range, rf_errors, 'o-', color='darkgreen', linewidth=2.5, markersize=6, label='RF Actual MSE')
ax.axhline(np.mean(single_tree_errors), color='red', linestyle='--', linewidth=2, 
          label=f'Single Tree MSE ({np.mean(single_tree_errors):.2f})')
ax.set_xlabel('Number of Trees (B)', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
ax.set_title('RF Error Decreases with More Trees', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Annotate key points
for B in [1, 10, 20, 50]:
    idx = B - 1
    reduction = (np.mean(single_tree_errors) - rf_errors[idx]) / np.mean(single_tree_errors) * 100
    if B in [10, 50]:
        ax.annotate(f'B={B}\n{reduction:.0f}% better', 
                   xy=(B, rf_errors[idx]), 
                   xytext=(B+5, rf_errors[idx]+0.5),
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# MIDDLE: Theoretical Variance Formula
ax = axes[1]
B_theory = np.arange(1, 101)
rho_values = [0.1, 0.3, 0.5, 0.7, 0.9]
colors_rho = ['green', 'blue', 'orange', 'red', 'darkred']

for rho_val, color in zip(rho_values, colors_rho):
    var_rf = rho_val + (1 - rho_val) / B_theory
    ax.plot(B_theory, var_rf, '-', linewidth=2.5, color=color, label=f'ρ={rho_val}')

ax.axhline(1.0, color='black', linestyle=':', linewidth=1.5, alpha=0.5, label='Single tree (σ²=1)')
ax.set_xlabel('Number of Trees (B)', fontsize=12, fontweight='bold')
ax.set_ylabel('Variance (normalized)', fontsize=12, fontweight='bold')
ax.set_title('Variance = ρσ² + (1-ρ)σ²/B', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.1)

# Annotate asymptote
ax.text(80, 0.35, 'As B→∞:\nVar→ρσ²\n(irreducible)', 
       fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# RIGHT: Variance Reduction %
ax = axes[2]
B_test = [1, 5, 10, 20, 50, 100]
reduction_by_rho = {}

for rho_val in rho_values:
    reductions = []
    for B in B_test:
        var_rf = rho_val + (1 - rho_val) / B
        reduction = (1.0 - var_rf) * 100
        reductions.append(reduction)
    reduction_by_rho[rho_val] = reductions

x_pos = np.arange(len(B_test))
width = 0.15

for i, (rho_val, color) in enumerate(zip(rho_values, colors_rho)):
    offset = (i - 2) * width
    ax.bar(x_pos + offset, reduction_by_rho[rho_val], width, 
          label=f'ρ={rho_val}', color=color, alpha=0.8, edgecolor='black')

ax.set_xlabel('Number of Trees (B)', fontsize=12, fontweight='bold')
ax.set_ylabel('Variance Reduction (%)', fontsize=12, fontweight='bold')
ax.set_title('Variance Reduction by ρ and B', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{B}' for B in B_test])
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

plt.suptitle('🌲 Random Forest: Variance Reduction Through Ensemble', 
            fontsize=15, fontweight='bold', y=1.0)
plt.tight_layout()
plt.show()

print("\n💡 KEY TAKEAWAYS:")
print("   1. Lower ρ (less correlation) → More variance reduction")
print("   2. More trees (B) → Lower variance (but diminishing returns)")
print("   3. Asymptote: Even with B→∞, variance ≥ ρσ² (irreducible)")
print("   4. Random features keep ρ low → Better ensemble!")

# COMMAND ----------

# DBTITLE 1,🌲→🌲→🌲 Part 2: Gradient Boosting Residual Learning
# MAGIC %md
# MAGIC # 🌲→🌲→🌲 PART 2: Gradient Boosting Residual Learning
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📝 The Core Algorithm
# MAGIC
# MAGIC ### **Additive Model:**
# MAGIC
# MAGIC ```
# MAGIC F_M(x) = F₀ + α·h₁(x) + α·h₂(x) + ... + α·hM(x)
# MAGIC
# MAGIC Where:
# MAGIC   - F₀ = initial prediction (often mean(y))
# MAGIC   - h_m = tree trained at iteration m
# MAGIC   - α = learning rate (shrinkage parameter)
# MAGIC   - M = total number of iterations/trees
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔁 Iterative Process
# MAGIC
# MAGIC ### **At each iteration m = 1, 2, ..., M:**
# MAGIC
# MAGIC ```
# MAGIC Step 1: Calculate residuals
# MAGIC   r_i = y_i - F_{m-1}(x_i)  ← What did we miss?
# MAGIC   
# MAGIC Step 2: Train tree on residuals
# MAGIC   h_m = tree trained on (X, r)
# MAGIC   
# MAGIC Step 3: Update prediction
# MAGIC   F_m(x) = F_{m-1}(x) + α·h_m(x)
# MAGIC   
# MAGIC Step 4: Repeat until convergence or M iterations
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Example Walkthrough
# MAGIC
# MAGIC ### **Dataset:** `y = [10, 8, 6, 4, 2]`
# MAGIC
# MAGIC **Iteration 0 (F₀):**
# MAGIC ```
# MAGIC F₀ = mean(y) = 6.0
# MAGIC Predictions: [6, 6, 6, 6, 6]
# MAGIC Residuals: [4, 2, 0, -2, -4]
# MAGIC MSE: 10.0
# MAGIC ```
# MAGIC
# MAGIC **Iteration 1 (with α=0.3):**
# MAGIC ```
# MAGIC Train h₁ on residuals [4, 2, 0, -2, -4]
# MAGIC h₁ predicts: [3.5, 1.5, 0, -1.5, -3.5]  (approximate)
# MAGIC
# MAGIC F₁ = F₀ + 0.3·h₁
# MAGIC     = [6, 6, 6, 6, 6] + 0.3×[3.5, 1.5, 0, -1.5, -3.5]
# MAGIC     = [7.05, 6.45, 6.0, 5.55, 4.95]
# MAGIC     
# MAGIC New residuals: y - F₁ = [2.95, 1.55, 0, -1.55, -2.95]
# MAGIC MSE: 5.8  (↓ improvement!)
# MAGIC ```
# MAGIC
# MAGIC **Iteration 2:**
# MAGIC ```
# MAGIC Train h₂ on residuals [2.95, 1.55, 0, -1.55, -2.95]
# MAGIC h₂ predicts: [2.8, 1.3, 0, -1.3, -2.8]  (approximate)
# MAGIC
# MAGIC F₂ = F₁ + 0.3·h₂
# MAGIC     = ... (continuing the process)
# MAGIC     
# MAGIC MSE: 3.2  (↓ keeps improving!)
# MAGIC ```
# MAGIC
# MAGIC **After M=10 iterations:**
# MAGIC ```
# MAGIC F₁₀ ≈ [10.0, 8.0, 6.0, 4.0, 2.0]  ← Very close to truth!
# MAGIC MSE ≈ 0.05
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔑 Key Parameters
# MAGIC
# MAGIC ### **Learning Rate (α):**
# MAGIC
# MAGIC | **α** | **Effect** | **Trade-off** |
# MAGIC |--------|-----------|---------------|
# MAGIC | **High (0.5-1.0)** | Fast convergence | Risk of overfitting |
# MAGIC | **Medium (0.1-0.3)** | Balanced | ✅ **Recommended** |
# MAGIC | **Low (0.01-0.05)** | Slow, stable | Needs more iterations |
# MAGIC
# MAGIC ### **Number of Iterations (M):**
# MAGIC
# MAGIC * **Too few:** Underfitting (high bias)
# MAGIC * **Too many:** Overfitting (memorizes training noise)
# MAGIC * **Just right:** Use validation set or early stopping!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ❓ Why Sequential?
# MAGIC
# MAGIC **Each tree focuses on what previous trees got wrong!**
# MAGIC
# MAGIC * Tree 1: Learns main pattern
# MAGIC * Tree 2: Corrects Tree 1's mistakes
# MAGIC * Tree 3: Corrects Tree 1+2's mistakes
# MAGIC * ...
# MAGIC * Tree M: Fine-tunes remaining errors
# MAGIC
# MAGIC **Result:** Bias ↓↓↓ (but variance ↑ if too many trees!)

# COMMAND ----------

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO: GRADIENT BOOSTING STEP-BY-STEP
# ═══════════════════════════════════════════════════════════════════════════════

# Use same data as before
np.random.seed(42)
n_samples_gb = 100
X_gb = np.random.randn(n_samples_gb, 10)
y_true_gb = 2*X_gb[:, 0] + 1.5*X_gb[:, 1] - X_gb[:, 2] + 5
y_gb = y_true_gb + np.random.randn(n_samples_gb) * 2

print("✅ Data generated for GB demo")
print(f"   Samples: {n_samples_gb}")
print(f"   True function: y = 2·X₀ + 1.5·X₁ - X₂ + 5 + noise")
X_gb 

# COMMAND ----------

y_gb.shape

# COMMAND ----------



# Manual GB implementation
alpha = 0.1  # Learning rate
M = 20  # Number of iterations

# COMMAND ----------

y_true_gb 

# COMMAND ----------

F_m = np.full(n_samples_gb, y_gb.mean()) 
F_m

# COMMAND ----------

y_true_gb.shape

# COMMAND ----------


# Initialize
 # F₀ = mean(y)
F_history = [F_m.copy()] 
#mse=sum(y_bar-yi)**2
mse_history = [np.mean((F_m - y_true_gb)**2)] 

residual_history = []

print(f"\n🚀 GRADIENT BOOSTING TRAINING")
print(f"   Learning rate (α): {alpha}")
print(f"   Iterations (M): {M}")
print(f"   Initial F₀: {F_m[0]:.4f}")
print(f"   Initial MSE: {mse_history[0]:.4f}")

mse_history


# COMMAND ----------

# DBTITLE 1,Demo: GB Residual Learning Step-by-Step

print(f"\n📊 Iteration-by-Iteration:")
print("\n  Iter | MSE      | Avg Residual | Max |Residual|")
print("  " + "-"*50)

for m in range(1, M + 1):
    # Step 1: Calculate residuals
    residuals = y_gb - F_m
   
    residual_history.append(residuals.copy())
    
    # Step 2: Train tree on residuals
    tree = DecisionTreeRegressor(max_depth=3, random_state=m)
    tree.fit(X_gb, residuals)
    h_m = tree.predict(X_gb)
    
    # Step 3: Update
    F_m = F_m + alpha * h_m
    
    # Track
    F_history.append(F_m.copy())
    mse = np.mean((F_m - y_true_gb)**2)
    mse_history.append(mse)
    
    # Print progress
    if m <= 5 or m % 5 == 0:
        print(f"  {m:3d}  | {mse:8.4f} | {np.abs(residuals).mean():12.4f} | {np.abs(residuals).max():15.4f}")

print("\n🎉 TRAINING COMPLETE!")
print(f"\n📈 Final Results:")
print(f"   Initial MSE (F₀): {mse_history[0]:.4f}")
print(f"   Final MSE (F_{M}): {mse_history[-1]:.4f}")
print(f"   Improvement: {(mse_history[0] - mse_history[-1]) / mse_history[0] * 100:.1f}%")
print(f"   F₀ = {F_history[0][0]:.4f}")
print(f"   F_{M} = {F_history[-1][0]:.4f}")
print(f"   True y[0] = {y_true_gb[0]:.4f}")

print(f"\n💡 KEY INSIGHT:")
print(f"   Each iteration reduces error by learning from residuals!")
print(f"   Formula: F_m = F_{{m-1}} + {alpha}·h_m")
print(f"   After {M} iterations, F_{M} ≈ y_true")

# COMMAND ----------

# DBTITLE 1,Visualization: GB Convergence
# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION: GRADIENT BOOSTING CONVERGENCE
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# TOP LEFT: MSE over iterations
ax = axes[0, 0]
iterations = range(len(mse_history))
ax.plot(iterations, mse_history, 'o-', color='darkblue', linewidth=2.5, markersize=6)
ax.set_xlabel('Iteration (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
ax.set_title(f'GB Convergence (α={alpha}, M={M})', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)

# Annotate key points
for m in [0, 5, 10, 20]:
    if m < len(mse_history):
        ax.annotate(f'M={m}\nMSE={mse_history[m]:.2f}', 
                   xy=(m, mse_history[m]), 
                   xytext=(m+1, mse_history[m]+0.5),
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.text(0.5, 0.95, f'Formula: F_m = F_{{m-1}} + {alpha}·h_m', 
       transform=ax.transAxes, ha='center', fontsize=10,
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# TOP RIGHT: Predictions vs Truth (first 20 samples)
ax = axes[0, 1]
sample_indices = range(20)
ax.scatter(sample_indices, y_true_gb[:20], color='green', s=100, marker='o', 
          edgecolors='black', linewidth=2, label='True y', zorder=3, alpha=0.7)
ax.scatter(sample_indices, F_history[0][:20], color='red', s=80, marker='x', 
          linewidth=2, label=f'F₀ (init)', alpha=0.7)
ax.scatter(sample_indices, F_history[-1][:20], color='blue', s=60, marker='+', 
          linewidth=2, label=f'F_{M} (final)', alpha=0.9)

ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Value', fontsize=12, fontweight='bold')
ax.set_title('Predictions Converge to Truth', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3)

# BOTTOM LEFT: Residual magnitude over iterations
ax = axes[1, 0]
residual_mags = [np.abs(r).mean() for r in residual_history]
ax.plot(range(1, len(residual_mags)+1), residual_mags, 'o-', 
       color='purple', linewidth=2.5, markersize=6)
ax.set_xlabel('Iteration (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean |Residual|', fontsize=12, fontweight='bold')
ax.set_title('Residuals Shrink Each Iteration', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)

# Add formula
ax.text(0.5, 0.95, 'r_i = y_i - F_{{m-1}}(x_i)', 
       transform=ax.transAxes, ha='center', fontsize=10,
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

# BOTTOM RIGHT: Learning Rate Comparison
ax = axes[1, 1]

# Train with different learning rates
lr_values = [0.01, 0.05, 0.1, 0.3, 0.5]
colors_lr = ['green', 'blue', 'orange', 'red', 'darkred']

for lr, color in zip(lr_values, colors_lr):
    F_lr = np.full(n_samples_gb, y_gb.mean())
    mse_lr = []
    
    for m in range(1, 31):
        residuals = y_gb - F_lr
        tree = DecisionTreeRegressor(max_depth=3, random_state=m)
        tree.fit(X_gb, residuals)
        h_m = tree.predict(X_gb)
        F_lr = F_lr + lr * h_m
        mse = np.mean((F_lr - y_true_gb)**2)
        mse_lr.append(mse)
    
    ax.plot(range(1, 31), mse_lr, '-', linewidth=2.5, color=color, label=f'α={lr}')

ax.set_xlabel('Iteration (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
ax.set_title('Effect of Learning Rate (α)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3)

# Annotate trade-offs
ax.text(0.3, 0.7, 'High α:\nFast but\nunstable', 
       transform=ax.transAxes, fontsize=9,
       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
ax.text(0.6, 0.3, 'Low α:\nSlow but\nstable', 
       transform=ax.transAxes, fontsize=9,
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.suptitle('🌲→🌲→🌲 Gradient Boosting: Sequential Residual Correction', 
            fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("\n💡 KEY TAKEAWAYS:")
print("   1. Each iteration corrects previous mistakes (residuals)")
print("   2. Learning rate α controls step size (trade-off: speed vs stability)")
print("   3. MSE decreases monotonically (with proper α and tree depth)")
print("   4. Predictions F_M converge to true y as M increases")

# COMMAND ----------

# DBTITLE 1,⚖️ Comparison: RF vs GB Side-by-Side
# ═══════════════════════════════════════════════════════════════════════════════
# SIDE-BY-SIDE COMPARISON: RF vs GB
# ═══════════════════════════════════════════════════════════════════════════════

print("═" * 80)
print("⚖️  RANDOM FOREST vs GRADIENT BOOSTING: THE MATH")
print("═" * 80)

comparison = {
    "Aspect": [
        "Tree Training",
        "Data per Tree",
        "Features per Split",
        "Combination",
        "Main Reduces",
        "Formula",
        "Key Parameter",
        "Overfitting Risk",
        "Interpretability",
        "Training Speed"
    ],
    "Random Forest 🌲🌲🌲": [
        "Parallel (independent)",
        "Bootstrap sample",
        "Random √p subset",
        "Majority vote / Average",
        "Variance ↓",
        "ŷ = (1/B)Σh_b(x)",
        "B (# trees)",
        "Low (robust)",
        "Moderate",
        "Fast (parallel)"
    ],
    "Gradient Boosting 🌲→🌲→🌲": [
        "Sequential (dependent)",
        "Full data (on residuals)",
        "All p features",
        "Additive sum",
        "Bias ↓",
        "ŷ = F₀ + αΣh_m(x)",
        "α (learning rate), M (# iterations)",
        "Higher (needs tuning)",
        "Lower (complex)",
        "Slower (sequential)"
    ]
}

df_comparison = pd.DataFrame(comparison)
print("\n")
display(df_comparison)

print("\n" + "═" * 80)
print("📊 KEY MATHEMATICAL INSIGHTS")
print("═" * 80)

print("\n🌲 RANDOM FOREST:")
print("   Formula: Var(RF) = ρσ² + (1-ρ)σ²/B")
print("   ")
print("   Why it works:")
print("     • Bootstrap + random features → decorrelates trees (↓ρ)")
print("     • Averaging B trees → reduces variance by factor of B")
print("     • As B→∞: Var(RF) → ρσ² (irreducible floor)")
print("   ")
print("   Best for: High variance models, noisy data, robustness")

print("\n🌲→🌲→🌲 GRADIENT BOOSTING:")
print("   Formula: F_M = F₀ + α·h₁ + α·h₂ + ... + α·hM")
print("   ")
print("   Why it works:")
print("     • Each tree h_m corrects residuals: r_i = y_i - F_{m-1}(x_i)")
print("     • Additive updates → gradually reduces bias")
print("     • Learning rate α → controls overfitting (smaller = safer)")
print("   ")
print("   Best for: High bias models, smooth functions, accuracy")

print("\n🎯 WHEN TO USE WHICH?")
print("\n  🌲 Random Forest:")
print("     ✅ Need robustness / interpretability")
print("     ✅ Have high variance problem")
print("     ✅ Want fast training (can parallelize)")
print("     ✅ Don't want to tune hyperparameters much")

print("\n  🌲→🌲→🌲 Gradient Boosting:")
print("     ✅ Need maximum accuracy")
print("     ✅ Have high bias problem")
print("     ✅ Willing to tune α, M, tree depth carefully")
print("     ✅ Have compute budget for sequential training")

print("\n💡 HYBRID APPROACH:")
print("   Try both! RF for baseline, GB for production tuning")
print("   Or use XGBoost/LightGBM (GB with regularization)")

print("\n" + "═" * 80)

# COMMAND ----------

# DBTITLE 1,🌳 Part 0: Decision Tree Foundation
# MAGIC %md
# MAGIC # 🌳 PART 0: Decision Tree - The Foundation
# MAGIC
# MAGIC **Dono Random Forest aur Gradient Boosting Decision Trees use karte hain — toh pehle yeh samjho!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Example Dataset
# MAGIC
# MAGIC ```
# MAGIC Data:
# MAGIC Hours studied  Passed?
# MAGIC 1              No
# MAGIC 2              No  
# MAGIC 4              Yes
# MAGIC 5              Yes
# MAGIC 6              Yes
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🌳 Decision Tree Structure
# MAGIC
# MAGIC ```
# MAGIC         Hours > 3?
# MAGIC        /          \
# MAGIC      No            Yes
# MAGIC   Predict:No    Predict:Yes
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📐 Splitting Formula — Gini Impurity
# MAGIC
# MAGIC ### **Formula:**
# MAGIC
# MAGIC ```
# MAGIC Gini = 1 - Σ_{k=1}^K p_k²
# MAGIC ```
# MAGIC
# MAGIC **Jahan:**
# MAGIC * `p_k` = class k ki probability
# MAGIC * Pure node → Gini = 0 (best)
# MAGIC * Mixed node → Gini > 0 (worst when 50-50)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🧮 Gini Calculation Example
# MAGIC
# MAGIC **Node mein [3 Yes, 1 No]:**
# MAGIC
# MAGIC ```
# MAGIC p_yes = 3/4 = 0.75
# MAGIC p_no  = 1/4 = 0.25
# MAGIC
# MAGIC Gini = 1 - (0.75² + 0.25²)
# MAGIC      = 1 - (0.5625 + 0.0625)
# MAGIC      = 1 - 0.625
# MAGIC      = 0.375
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Special Cases
# MAGIC
# MAGIC **1. Pure node (sab Yes):**
# MAGIC ```
# MAGIC Gini = 1 - (1² + 0²) = 0  ← perfect! ✅
# MAGIC ```
# MAGIC
# MAGIC **2. Mixed node (50-50):**
# MAGIC ```
# MAGIC Gini = 1 - (0.5² + 0.5²) = 0.5  ← worst! ❌
# MAGIC ```
# MAGIC
# MAGIC **3. Moderate impurity (75-25):**
# MAGIC ```
# MAGIC Gini = 1 - (0.75² + 0.25²) = 0.375  ← okay
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Best Split Selection
# MAGIC
# MAGIC ### **Information Gain Formula:**
# MAGIC
# MAGIC ```
# MAGIC Information Gain = Gini_parent - (N_L/N)×Gini_L - (N_R/N)×Gini_R
# MAGIC ```
# MAGIC
# MAGIC **Where:**
# MAGIC * `Gini_parent` = impurity before split
# MAGIC * `Gini_L` = impurity of left child
# MAGIC * `Gini_R` = impurity of right child  
# MAGIC * `N_L` = samples in left child
# MAGIC * `N_R` = samples in right child
# MAGIC * `N` = total samples (N_L + N_R)
# MAGIC
# MAGIC **Process:**
# MAGIC 1. Try all possible splits
# MAGIC 2. Calculate Information Gain for each
# MAGIC 3. Choose split with **maximum** Information Gain
# MAGIC 4. Repeat recursively for each child node
# MAGIC 5. Stop when: node is pure OR max depth reached OR min samples
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 Key Insight
# MAGIC
# MAGIC **Yeh har node pe karte hain → tree ban jaata hai!**
# MAGIC
# MAGIC * Each split reduces Gini impurity
# MAGIC * Tree grows until stopping criteria met
# MAGIC * This forms the base learner for RF and GB

# COMMAND ----------

# DBTITLE 1,Demo: Gini Impurity Calculation
# ═══════════════════════════════════════════════════════════════════════════════
# DEMO: GINI IMPURITY CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt

def calculate_gini(class_counts):
    """
    Calculate Gini impurity
    class_counts: dict {class: count}
    """
    total = sum(class_counts.values())
    if total == 0:
        return 0.0
    
    proportions = [count/total for count in class_counts.values()]
    gini = 1.0 - sum([p**2 for p in proportions])
    return gini

def calculate_info_gain(parent_counts, left_counts, right_counts):
    """
    Calculate information gain from a split
    """
    n_parent = sum(parent_counts.values())
    n_left = sum(left_counts.values())
    n_right = sum(right_counts.values())
    
    gini_parent = calculate_gini(parent_counts)
    gini_left = calculate_gini(left_counts)
    gini_right = calculate_gini(right_counts)
    
    weighted_child_gini = (n_left/n_parent) * gini_left + (n_right/n_parent) * gini_right
    info_gain = gini_parent - weighted_child_gini
    
    return info_gain, gini_parent, gini_left, gini_right

print("═" * 80)
print("🌳 DECISION TREE: GINI IMPURITY DEMO")
print("═" * 80)

# Example from the data
print("\n📊 Dataset:")
print("   Hours studied: [1, 2, 4, 5, 6]")
print("   Passed:        [No, No, Yes, Yes, Yes]")

# Calculate Gini for different nodes
print("\n\n🧮 GINI CALCULATIONS:")

# Example 1: Parent node (all data)
parent = {'No': 2, 'Yes': 3}
gini_parent = calculate_gini(parent)
print(f"\n1️⃣ Parent Node [2 No, 3 Yes]:")
print(f"   p_No = 2/5 = 0.40")
print(f"   p_Yes = 3/5 = 0.60")
print(f"   Gini = 1 - (0.40² + 0.60²)")
print(f"        = 1 - (0.16 + 0.36)")
print(f"        = 1 - 0.52")
print(f"        = {gini_parent:.4f}")

# Example 2: Pure node
pure_yes = {'Yes': 5}
gini_pure = calculate_gini(pure_yes)
print(f"\n2️⃣ Pure Node [5 Yes]:")
print(f"   p_Yes = 5/5 = 1.0")
print(f"   Gini = 1 - (1.0²)")
print(f"        = 1 - 1.0")
print(f"        = {gini_pure:.4f}  ← Perfect! ✅")

# Example 3: 50-50 split
mixed = {'No': 2, 'Yes': 2}
gini_mixed = calculate_gini(mixed)
print(f"\n3️⃣ Mixed Node [2 No, 2 Yes]:")
print(f"   p_No = 2/4 = 0.50")
print(f"   p_Yes = 2/4 = 0.50")
print(f"   Gini = 1 - (0.50² + 0.50²)")
print(f"        = 1 - (0.25 + 0.25)")
print(f"        = 1 - 0.50")
print(f"        = {gini_mixed:.4f}  ← Worst! ❌")

# Example 4: Split at Hours > 3
print("\n\n📊 BEST SPLIT CALCULATION:")
print("\nTrying split: Hours > 3")

left = {'No': 2, 'Yes': 0}  # Hours <= 3
right = {'No': 0, 'Yes': 3}  # Hours > 3

info_gain, g_p, g_l, g_r = calculate_info_gain(parent, left, right)

print(f"\n⬅️  Left (Hours ≤ 3): [2 No, 0 Yes]")
print(f"   Gini_L = {g_l:.4f}")

print(f"\n➡️  Right (Hours > 3): [0 No, 3 Yes]")
print(f"   Gini_R = {g_r:.4f}")

print(f"\n🎯 Information Gain:")
print(f"   IG = Gini_parent - (N_L/N)×Gini_L - (N_R/N)×Gini_R")
print(f"      = {g_p:.4f} - (2/5)×{g_l:.4f} - (3/5)×{g_r:.4f}")
print(f"      = {g_p:.4f} - 0 - 0")
print(f"      = {info_gain:.4f}")
print(f"\n   ⭐ This is a PERFECT split! Both children are pure.")

# Visualize Gini values
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# LEFT: Gini for different class distributions
ax = ax1
proportions = np.linspace(0, 1, 100)
gini_values = [1 - (p**2 + (1-p)**2) for p in proportions]

ax.plot(proportions, gini_values, linewidth=3, color='darkblue')
ax.axhline(0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Pure (Gini=0)')
ax.axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Max impurity (Gini=0.5)')
ax.scatter([0.4], [gini_parent], s=200, color='orange', edgecolors='black', linewidth=2, 
          zorder=5, label=f'Our data (p=0.4)')

ax.set_xlabel('Proportion of Positive Class (p)', fontsize=12, fontweight='bold')
ax.set_ylabel('Gini Impurity', fontsize=12, fontweight='bold')
ax.set_title('Gini Impurity vs Class Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(-0.05, 0.55)

# Annotate key points
ax.annotate('Pure (all negative)', xy=(0, 0), xytext=(0.15, 0.1),
           fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
           arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate('Pure (all positive)', xy=(1, 0), xytext=(0.75, 0.1),
           fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
           arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate('Maximum\nimpurity\n(50-50)', xy=(0.5, 0.5), xytext=(0.6, 0.4),
           fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
           arrowprops=dict(arrowstyle='->', color='black'))

# RIGHT: Tree structure
ax = ax2
ax.text(0.5, 0.85, 'Hours > 3?', ha='center', fontsize=14, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=2))

# Left branch
ax.plot([0.5, 0.25], [0.8, 0.5], 'k-', linewidth=2)
ax.text(0.35, 0.65, 'No', fontsize=10, fontweight='bold')
ax.text(0.25, 0.45, 'Predict: No\n[2 No, 0 Yes]\nGini = 0.0', ha='center', fontsize=10,
       bbox=dict(boxstyle='round', facecolor='lightcoral', edgecolor='black', linewidth=2))

# Right branch  
ax.plot([0.5, 0.75], [0.8, 0.5], 'k-', linewidth=2)
ax.text(0.65, 0.65, 'Yes', fontsize=10, fontweight='bold')
ax.text(0.75, 0.45, 'Predict: Yes\n[0 No, 3 Yes]\nGini = 0.0', ha='center', fontsize=10,
       bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black', linewidth=2))

ax.set_xlim(0, 1)
ax.set_ylim(0.3, 0.95)
ax.axis('off')
ax.set_title('Decision Tree Structure', fontsize=13, fontweight='bold')

plt.suptitle('🌳 Decision Tree: Gini Impurity & Splitting', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

print("\n" + "═" * 80)
print("\n💡 KEY TAKEAWAYS:")
print("   • Gini = 0 → Pure node (best case)")
print("   • Gini = 0.5 → Maximum impurity (worst for binary)")
print("   • Best split = Maximum Information Gain")
print("   • Repeat recursively until stopping criteria")
print("   • This forms the base for RF and GB!")
print("\n" + "═" * 80)

# COMMAND ----------

# DBTITLE 1,🌲🌲🌲 Random Forest: Detailed Mathematics
# MAGIC %md
# MAGIC # 🌲🌲🌲 Random Forest: Detailed Mathematics
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚠️ Problem with Single Tree
# MAGIC
# MAGIC **Ek tree → training data pe perfect fit**
# MAGIC
# MAGIC ```
# MAGIC Single Decision Tree:
# MAGIC   ✅ Low bias (can fit complex patterns)
# MAGIC   ❌ High variance (very sensitive to data changes)
# MAGIC   ❌ Thoda data badlo → bilkul alag tree bane
# MAGIC ```
# MAGIC
# MAGIC **Example:**
# MAGIC * Data 1: [1,2,3,4,5] → Tree splits at 3
# MAGIC * Data 2: [1,2,3,4,6] → Tree splits at 2 (completely different!)
# MAGIC * **High variance! ❌**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Solution — Bagging (Bootstrap Aggregating)
# MAGIC
# MAGIC ### **Step 1: Bootstrap Sampling**
# MAGIC
# MAGIC ```
# MAGIC D_b = sample(D, n, with replacement)
# MAGIC ```
# MAGIC
# MAGIC **Concrete Example:**
# MAGIC
# MAGIC ```
# MAGIC Original data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# MAGIC
# MAGIC Tree 1 sample: [3, 3, 7, 1, 9, 2, 7, 4, 6, 10]  ← repeats allowed!
# MAGIC Tree 2 sample: [5, 2, 8, 1, 1, 9, 3, 7, 4, 6]
# MAGIC Tree 3 sample: [2, 4, 4, 9, 5, 1, 8, 7, 3, 10]
# MAGIC ...
# MAGIC ```
# MAGIC
# MAGIC **Key:**
# MAGIC * Each tree sees ~63% unique samples (rest are repeats)
# MAGIC * Har tree alag data pe train hota hai
# MAGIC * **Diverse trees milte hain! ✅**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Step 2: Random Feature Selection**
# MAGIC
# MAGIC **At each split:**
# MAGIC
# MAGIC ```
# MAGIC m = √p     (classification)
# MAGIC m = p/3    (regression)
# MAGIC
# MAGIC Where p = total number of features
# MAGIC ```
# MAGIC
# MAGIC **Example:**
# MAGIC
# MAGIC ```
# MAGIC Total features p = 20
# MAGIC m = √20 ≈ 4-5 features randomly choose karo
# MAGIC
# MAGIC Node 1 split: Try features [2, 7, 11, 15, 18]
# MAGIC Node 2 split: Try features [1, 4, 9, 12, 19]
# MAGIC Node 3 split: Try features [3, 6, 10, 13, 16]
# MAGIC ...
# MAGIC ```
# MAGIC
# MAGIC **Benefits:**
# MAGIC * Har node pe sirf m features consider karo
# MAGIC * Har tree alag features use karta hai
# MAGIC * **Aur bhi diverse! ✅**
# MAGIC * Prevents strong features from dominating all trees
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Step 3: Aggregate (Voting/Averaging)**
# MAGIC
# MAGIC **Classification (Voting):**
# MAGIC
# MAGIC ```
# MAGIC ŷ = majority_vote(ŷ1, ŷ2, ..., ŷB)
# MAGIC ```
# MAGIC
# MAGIC **Example:**
# MAGIC
# MAGIC ```
# MAGIC Tree 1 prediction: Yes
# MAGIC Tree 2 prediction: Yes
# MAGIC Tree 3 prediction: No
# MAGIC Tree 4 prediction: Yes
# MAGIC Tree 5 prediction: No
# MAGIC
# MAGIC Majority vote: Yes (3 vs 2)
# MAGIC Final prediction: Yes ✅
# MAGIC ```
# MAGIC
# MAGIC **Regression (Averaging):**
# MAGIC
# MAGIC ```
# MAGIC ŷ = (1/B) Σ_{b=1}^B ŷ_b
# MAGIC ```
# MAGIC
# MAGIC **Example:**
# MAGIC
# MAGIC ```
# MAGIC Tree 1 prediction: 5.2
# MAGIC Tree 2 prediction: 4.8
# MAGIC Tree 3 prediction: 5.5
# MAGIC Tree 4 prediction: 4.9
# MAGIC Tree 5 prediction: 5.1
# MAGIC
# MAGIC Average: (5.2 + 4.8 + 5.5 + 4.9 + 5.1) / 5 = 5.1 ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Why It Works — Mathematical Proof
# MAGIC
# MAGIC ### **Single Tree Error:**
# MAGIC
# MAGIC ```
# MAGIC Error = bias² + variance + noise
# MAGIC ```
# MAGIC
# MAGIC ### **Random Forest:**
# MAGIC
# MAGIC ```
# MAGIC bias ≈ same (each tree still learns well)
# MAGIC variance ↓↓↓ (averaging reduces variance!)
# MAGIC ```
# MAGIC
# MAGIC ### **Variance Formula (Detailed Derivation):**
# MAGIC
# MAGIC **If trees have:**
# MAGIC * Error variance = σ²
# MAGIC * Correlation = ρ
# MAGIC
# MAGIC **Then RF variance:**
# MAGIC
# MAGIC ```
# MAGIC Var(RF) = Var( (1/B) Σ X_b )
# MAGIC
# MAGIC         = (1/B²) Var( Σ X_b )
# MAGIC         
# MAGIC         = (1/B²) [ Σ Var(X_b) + Σ_{i≠j} Cov(X_i, X_j) ]
# MAGIC         
# MAGIC         = (1/B²) [ B·σ² + B(B-1)·ρσ² ]
# MAGIC         
# MAGIC         = σ²/B + (B-1)ρσ²/B
# MAGIC         
# MAGIC         = σ²/B + ρσ² - ρσ²/B
# MAGIC         
# MAGIC         = ρσ² + (1-ρ)σ²/B  ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Key Parameters Effect
# MAGIC
# MAGIC ### **B (Number of Trees):**
# MAGIC
# MAGIC * **B ↑** → Second term `(1-ρ)σ²/B` → 0
# MAGIC * More trees = Lower variance
# MAGIC * But diminishing returns after ~100-500 trees
# MAGIC
# MAGIC ### **ρ (Correlation):**
# MAGIC
# MAGIC * **ρ ↓** → First term `ρσ²` → smaller
# MAGIC * Bootstrap + random features keep ρ low
# MAGIC * Lower correlation = Better ensemble
# MAGIC
# MAGIC ### **Trade-offs:**
# MAGIC
# MAGIC ```
# MAGIC Too much randomness:
# MAGIC   → Trees become weak (↑σ²)
# MAGIC   → High individual error
# MAGIC   
# MAGIC  Too little randomness:
# MAGIC   → Trees correlated (↑ρ)
# MAGIC   → No ensemble benefit
# MAGIC   
# MAGIC Sweet spot:
# MAGIC   → m = √p features
# MAGIC   → Bootstrap sampling
# MAGIC   → Balanced diversity! ✅
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Demo: Bootstrap & Voting in Action
# ═══════════════════════════════════════════════════════════════════════════════
# DEMO: BOOTSTRAP SAMPLING & VOTING
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

print("═" * 80)
print("🌲🌲🌲 RANDOM FOREST: BOOTSTRAP & VOTING DEMO")
print("═" * 80)

# Generate simple dataset
np.random.seed(42)
n_samples = 100
X_demo = np.random.randn(n_samples, 5)
y_demo = (X_demo[:, 0] + X_demo[:, 1] > 0).astype(int)  # Binary classification

print(f"\n📊 Dataset:")
print(f"   Samples: {n_samples}")
print(f"   Features: {X_demo.shape[1]}")
print(f"   Classes: {np.unique(y_demo)}")
print(f"   Class distribution: {dict(Counter(y_demo))}")

# Demonstrate bootstrap sampling
print("\n\n🎲 STEP 1: BOOTSTRAP SAMPLING")
print("\nOriginal indices: [0, 1, 2, 3, ..., 99]")

B = 5  # 5 trees
bootstrap_samples = []

for b in range(B):
    # Bootstrap sample (with replacement)
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    bootstrap_samples.append(indices)
    
    # Show first 10 indices
    unique_pct = len(np.unique(indices)) / n_samples * 100
    print(f"\nTree {b+1} bootstrap: {indices[:10].tolist()} ...")
    print(f"        Unique samples: {len(np.unique(indices))}/{n_samples} ({unique_pct:.1f}%)")
    print(f"        Repeated samples: {n_samples - len(np.unique(indices))}")

# Theoretical: Each sample has ~63.2% chance of being selected at least once
theoretical_unique = (1 - (1 - 1/n_samples)**n_samples) * 100
print(f"\n💡 Theoretical: ~{theoretical_unique:.1f}% unique samples per bootstrap")

# Train trees and get predictions
print("\n\n🌳 STEP 2: TRAIN TREES")

trees = []
for b, indices in enumerate(bootstrap_samples):
    X_boot = X_demo[indices]
    y_boot = y_demo[indices]
    
    tree = DecisionTreeClassifier(max_depth=5, random_state=b)
    tree.fit(X_boot, y_boot)
    trees.append(tree)
    
    train_acc = tree.score(X_boot, y_boot) * 100
    print(f"   Tree {b+1}: Train accuracy = {train_acc:.1f}%")

# Get predictions from all trees
print("\n\n🗳️ STEP 3: VOTING")

test_sample = X_demo[0:1]  # First sample
true_label = y_demo[0]

print(f"\nTest sample: X[0]")
print(f"True label: {true_label}")
print(f"\nIndividual tree predictions:")

predictions = []
for b, tree in enumerate(trees):
    pred = tree.predict(test_sample)[0]
    predictions.append(pred)
    print(f"   Tree {b+1}: {pred}")

# Majority vote
vote_counts = Counter(predictions)
majority_vote = vote_counts.most_common(1)[0][0]

print(f"\n🗳️ Vote counts: {dict(vote_counts)}")
print(f"⭐ Majority vote: {majority_vote}")
print(f"✅ Correct!" if majority_vote == true_label else f"❌ Incorrect")

# Evaluate on full test set
print("\n\n📈 FULL EVALUATION")

# Individual tree accuracies
indiv_accs = []
for b, tree in enumerate(trees):
    acc = tree.score(X_demo, y_demo) * 100
    indiv_accs.append(acc)
    print(f"   Tree {b+1} test accuracy: {acc:.1f}%")

print(f"\n   Average individual accuracy: {np.mean(indiv_accs):.1f}%")

# Random Forest (majority vote)
rf_predictions = []
for i in range(len(X_demo)):
    votes = [tree.predict(X_demo[i:i+1])[0] for tree in trees]
    rf_pred = Counter(votes).most_common(1)[0][0]
    rf_predictions.append(rf_pred)

rf_acc = np.mean(rf_predictions == y_demo) * 100
print(f"   ⭐ Random Forest accuracy: {rf_acc:.1f}%")
print(f"\n   🎉 Improvement: {rf_acc - np.mean(indiv_accs):.1f}% better!")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# LEFT: Accuracy comparison
ax = ax1
labels = [f'Tree {i+1}' for i in range(B)] + ['RF\n(Vote)']
accuracies = indiv_accs + [rf_acc]
colors = ['lightblue'] * B + ['green']

bars = ax.bar(labels, accuracies, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
ax.axhline(np.mean(indiv_accs), color='red', linestyle='--', linewidth=2, 
          label=f'Avg Tree: {np.mean(indiv_accs):.1f}%')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Individual Trees vs Random Forest', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(70, 100)

# Add value labels
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 1, 
           f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')

# Highlight RF
bars[-1].set_linewidth(4)
bars[-1].set_edgecolor('darkgreen')

# RIGHT: Bootstrap diversity
ax = ax2

# Count how many times each sample was selected across all trees
sample_counts = np.zeros(n_samples)
for indices in bootstrap_samples:
    for idx in indices:
        sample_counts[idx] += 1

ax.hist(sample_counts, bins=range(0, int(sample_counts.max())+2), 
       edgecolor='black', linewidth=1.5, color='steelblue', alpha=0.7)
ax.axvline(sample_counts.mean(), color='red', linestyle='--', linewidth=2.5,
          label=f'Mean: {sample_counts.mean():.1f}')
ax.set_xlabel('Times Selected Across All Trees', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax.set_title(f'Bootstrap Sample Distribution (B={B})', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# Add annotation
ax.text(0.6, 0.95, f'Some samples\nselected {int(sample_counts.max())}x!\n\nDiversity through\nrandom sampling',
       transform=ax.transAxes, fontsize=10,
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.suptitle('🌲🌲🌲 Random Forest: Bootstrap Sampling & Voting', 
            fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

print("\n" + "═" * 80)
print("\n💡 KEY TAKEAWAYS:")
print("   • Each tree sees different bootstrap sample (~63% unique)")
print("   • Diversity → trees make different mistakes")
print("   • Voting averages out errors")
print("   • Ensemble > individual trees! ✅")
print("\n" + "═" * 80)

# COMMAND ----------

# DBTITLE 1,🌲→🌲→🌲 Gradient Boosting: Detailed Mathematics
# MAGIC %md
# MAGIC # 🌲→🌲→🌲 Gradient Boosting: Detailed Mathematics
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔄 Alag Idea — Mistakes Fix Karo Sequentially
# MAGIC
# MAGIC ```
# MAGIC Random Forest:  Trees parallel hain, independent
# MAGIC                 🌳 🌳 🌳 🌳 🌳  → Vote
# MAGIC                 
# MAGIC Gradient Boost: Trees sequential hain
# MAGIC                 🌳 → 🌳 → 🌳 → 🌳 → 🌳
# MAGIC                 Tree 2 = Tree 1 ki mistakes fix karo
# MAGIC                 Tree 3 = Tree 2 ki mistakes fix karo
# MAGIC                 ...
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📝 Step-by-Step Algorithm
# MAGIC
# MAGIC ### **Step 1: Pehla Model (Initialization)**
# MAGIC
# MAGIC ```
# MAGIC F₀(x) = arg min_γ Σ_{i=1}^N L(y_i, γ)
# MAGIC ```
# MAGIC
# MAGIC **For different loss functions:**
# MAGIC
# MAGIC **Regression (MSE):**
# MAGIC ```
# MAGIC F₀ = mean(y)
# MAGIC
# MAGIC Example: y = [3, 5, 8]
# MAGIC F₀ = (3 + 5 + 8) / 3 = 5.33
# MAGIC ```
# MAGIC
# MAGIC **Classification (Log Loss):**
# MAGIC ```
# MAGIC F₀ = log(p / (1-p))
# MAGIC
# MAGIC where p = proportion of positive class
# MAGIC ```
# MAGIC
# MAGIC **Key:** Bas ek constant se shuru karo ✅
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Step 2: Pseudo Residuals Nikalo (Gradient)**
# MAGIC
# MAGIC ```
# MAGIC r_{im} = -[∂L(y_i, F(x_i)) / ∂F(x_i)]_{F=F_{m-1}}
# MAGIC ```
# MAGIC
# MAGIC **For MSE loss:**
# MAGIC
# MAGIC ```
# MAGIC L = (y - F)²
# MAGIC ∂L/∂F = -2(y - F)
# MAGIC
# MAGIC Therefore:
# MAGIC r_i = y_i - F_{m-1}(x_i)  ← actual error!
# MAGIC ```
# MAGIC
# MAGIC **Matlab:** Pseudo residual = actual - predicted
# MAGIC
# MAGIC **Example:**
# MAGIC ```
# MAGIC Actual y:      [3.0, 5.0, 8.0]
# MAGIC Predicted F₀:  [5.33, 5.33, 5.33]
# MAGIC Residuals r₁:  [-2.33, -0.33, +2.67]  ← yeh seekho!
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Step 3: Naya Tree Residuals Pe Train Karo**
# MAGIC
# MAGIC ```
# MAGIC h_m(x) = DecisionTree(X, r_m)
# MAGIC ```
# MAGIC
# MAGIC **Tree learns to predict residuals:**
# MAGIC
# MAGIC ```
# MAGIC Input X:  [1, 2, 3]
# MAGIC Target r: [-2.33, -0.33, 2.67]
# MAGIC
# MAGIC Tree learns pattern:
# MAGIC   x ≤ 2 → predict -0.83  (average of -2.33, -0.33)
# MAGIC   x > 2 → predict +2.67
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Step 4: Model Update Karo**
# MAGIC
# MAGIC ```
# MAGIC F_m(x) = F_{m-1}(x) + α · h_m(x)
# MAGIC ```
# MAGIC
# MAGIC **Where:**
# MAGIC * `α` = learning rate (shrinkage) = 0.1 typically
# MAGIC * `h_m(x)` = tree predictions
# MAGIC
# MAGIC **Example (with α = 0.1):**
# MAGIC
# MAGIC ```
# MAGIC F₁(x) = F₀(x) + 0.1 × h₁(x)
# MAGIC         = 5.33 + 0.1 × [-0.83, -0.83, 2.67]
# MAGIC         = 5.33 + [-0.083, -0.083, 0.267]
# MAGIC         = [5.25, 5.25, 5.60]
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Step 5: Repeat M Times**
# MAGIC
# MAGIC ```
# MAGIC F_M(x) = F₀(x) + α Σ_{m=1}^M h_m(x)
# MAGIC ```
# MAGIC
# MAGIC **Final model is sum of all trees!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Concrete Example — Complete Walkthrough
# MAGIC
# MAGIC ### **Data:**
# MAGIC
# MAGIC ```
# MAGIC x = [1, 2, 3]
# MAGIC y = [3, 5, 8]
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Iteration 0:**
# MAGIC
# MAGIC ```
# MAGIC F₀ = mean(y) = (3 + 5 + 8) / 3 = 5.33
# MAGIC
# MAGIC Predictions:  [5.33, 5.33, 5.33]
# MAGIC Residuals:    [3-5.33, 5-5.33, 8-5.33]
# MAGIC             = [-2.33, -0.33, +2.67]
# MAGIC MSE:          [(-2.33)² + (-0.33)² + (2.67)²] / 3 = 3.84
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Iteration 1 (tree on residuals):**
# MAGIC
# MAGIC ```
# MAGIC Tree learns:
# MAGIC   x ≤ 2 → -0.83  (avg of -2.33, -0.33)
# MAGIC   x > 2 → +2.67
# MAGIC
# MAGIC Update (with α = 0.1):
# MAGIC   F₁ = F₀ + 0.1 × tree
# MAGIC      = [5.33, 5.33, 5.33] + 0.1 × [-0.83, -0.83, 2.67]
# MAGIC      = [5.25, 5.25, 5.60]
# MAGIC
# MAGIC New predictions: [5.25, 5.25, 5.60]
# MAGIC New residuals:   [3-5.25, 5-5.25, 8-5.60]
# MAGIC                = [-2.25, -0.25, +2.40]
# MAGIC MSE:             3.28  ← Improved! ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Iteration 2:**
# MAGIC
# MAGIC ```
# MAGIC Tree on new residuals [-2.25, -0.25, 2.40]:
# MAGIC   x ≤ 2 → -0.75
# MAGIC   x > 2 → +2.40
# MAGIC
# MAGIC Update:
# MAGIC   F₂ = F₁ + 0.1 × tree
# MAGIC      = [5.25, 5.25, 5.60] + 0.1 × [-0.75, -0.75, 2.40]
# MAGIC      = [5.18, 5.18, 5.84]
# MAGIC
# MAGIC New predictions: [5.18, 5.18, 5.84]
# MAGIC New residuals:   [-2.18, -0.18, +2.16]
# MAGIC MSE:             2.78  ← Still improving! ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **...**
# MAGIC
# MAGIC ```
# MAGIC Iteration 3: MSE = 2.35
# MAGIC Iteration 4: MSE = 1.98
# MAGIC Iteration 5: MSE = 1.67
# MAGIC ...
# MAGIC Iteration 50: MSE = 0.05
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Iteration 50 (Final):**
# MAGIC
# MAGIC ```
# MAGIC Predictions: [3.01, 4.99, 7.98]
# MAGIC True values: [3.00, 5.00, 8.00]
# MAGIC
# MAGIC ✅ Almost perfect!
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔑 Key Parameters
# MAGIC
# MAGIC ### **Learning Rate (α):**
# MAGIC
# MAGIC ```
# MAGIC High α (0.5-1.0):
# MAGIC   ✅ Fast convergence
# MAGIC   ❌ Risk of overfitting
# MAGIC   ❌ Can overshoot
# MAGIC   
# MAGIC Low α (0.01-0.05):
# MAGIC   ✅ Slow, stable
# MAGIC   ✅ Better generalization
# MAGIC   ❌ Needs more iterations (M)
# MAGIC   
# MAGIC Sweet spot (0.1-0.3):
# MAGIC   ✅ Balanced! ⭐
# MAGIC ```
# MAGIC
# MAGIC ### **Number of Iterations (M):**
# MAGIC
# MAGIC ```
# MAGIC Too few:
# MAGIC   → Underfitting (high bias)
# MAGIC   → Didn't learn enough
# MAGIC   
# MAGIC Too many:
# MAGIC   → Overfitting (memorizes noise)
# MAGIC   → Learns training data too well
# MAGIC   
# MAGIC Just right:
# MAGIC   → Use validation set
# MAGIC   → Early stopping
# MAGIC   → Monitor test error! ✅
# MAGIC ```
# MAGIC
# MAGIC ### **Tree Depth:**
# MAGIC
# MAGIC ```
# MAGIC Shallow (depth = 2-4):
# MAGIC   → Weak learners (good!)
# MAGIC   → GB works best with weak trees
# MAGIC   → Less overfitting
# MAGIC   
# MAGIC Deep (depth > 10):
# MAGIC   → Strong trees
# MAGIC   → More overfitting risk
# MAGIC   → Needs careful tuning
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Why Sequential Wins
# MAGIC
# MAGIC **Each tree focuses on what previous trees got wrong!**
# MAGIC
# MAGIC ```
# MAGIC Tree 1: Learns main pattern
# MAGIC         Error: Large but systematic
# MAGIC         
# MAGIC Tree 2: Corrects Tree 1's mistakes  
# MAGIC         Error: Smaller, more specific
# MAGIC         
# MAGIC Tree 3: Corrects Tree 1+2's mistakes
# MAGIC         Error: Even smaller
# MAGIC         
# MAGIC ...
# MAGIC
# MAGIC Tree M: Fine-tunes remaining errors
# MAGIC         Error: Tiny!
# MAGIC ```
# MAGIC
# MAGIC **Result:**
# MAGIC * **Bias ↓↓↓** (gradually reduces systematic errors)
# MAGIC * But variance ↑ if too many trees (overfitting risk)
# MAGIC * Need careful tuning! ⚠️

# COMMAND ----------

# DBTITLE 1,🎯 GB Simplified: The Easiest Explanation
# MAGIC %md
# MAGIC # 🎯 Gradient Boosting: Sabse Aasan Explanation
# MAGIC
# MAGIC **Forget formulas for now! Pehle concept samjho:**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🏫 Story Time: Teacher & Student Analogy
# MAGIC
# MAGIC **Imagine:** Aap ek teacher ho, aur aapko students ka marks predict karna hai.
# MAGIC
# MAGIC ### **Attempt 1: First Guess (F₀)**
# MAGIC ```
# MAGIC Class average dekha: 60 marks
# MAGIC Sabke liye predict kiya: 60, 60, 60, 60, 60
# MAGIC
# MAGIC Actual marks: [30, 50, 60, 80, 90]
# MAGIC Mistakes: [-30, -10, 0, +20, +30]
# MAGIC ```
# MAGIC
# MAGIC **Hmm... bohot galtiyan hain! 😟**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Attempt 2: Fix Mistakes (Tree 1)**
# MAGIC ```
# MAGIC Socha: "OK, maine galtiyan dekhi, ab unhe fix karta hoon"
# MAGIC
# MAGIC Mistakes: [-30, -10, 0, +20, +30]
# MAGIC
# MAGIC Pattern dekha:
# MAGIC   - Low scorers: Mujhe zyada de dena chahiye tha
# MAGIC   - High scorers: Mujhe kam de dena chahiye tha
# MAGIC
# MAGIC Corrected by: [-8, -3, 0, +6, +9] (thoda-thoda fix karo, 30% adjust)
# MAGIC
# MAGIC New predictions: [60-8, 60-3, 60+0, 60+6, 60+9]
# MAGIC                = [52, 57, 60, 66, 69]
# MAGIC
# MAGIC New mistakes: [30-52, 50-57, 60-60, 80-66, 90-69]
# MAGIC             = [-22, -7, 0, +14, +21]
# MAGIC ```
# MAGIC
# MAGIC **Better! But still mistakes hain... 🤔**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Attempt 3: Fix Remaining Mistakes (Tree 2)**
# MAGIC ```
# MAGIC Fir se mistakes dekhe: [-22, -7, 0, +14, +21]
# MAGIC
# MAGIC Aur fix kiya: [-7, -2, 0, +4, +6]
# MAGIC
# MAGIC New predictions: [52-7, 57-2, 60+0, 66+4, 69+6]
# MAGIC                = [45, 55, 60, 70, 75]
# MAGIC
# MAGIC New mistakes: [-15, -5, 0, +10, +15]
# MAGIC ```
# MAGIC
# MAGIC **Even better! Mistakes kam ho rahe hain! 😊**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Attempt 10: After Many Corrections**
# MAGIC ```
# MAGIC Predictions: [31, 49, 60, 79, 89]
# MAGIC Actual:      [30, 50, 60, 80, 90]
# MAGIC Mistakes:    [-1, -1, 0, -1, -1]
# MAGIC ```
# MAGIC
# MAGIC **Almost perfect! 🎉**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 Yahi Hai Gradient Boosting!
# MAGIC
# MAGIC ### **Simple 3-Step Process:**
# MAGIC
# MAGIC ```
# MAGIC 1. Make a guess (usually average)
# MAGIC    ↓
# MAGIC 2. See where you went wrong (mistakes = residuals)
# MAGIC    ↓
# MAGIC 3. Fix those mistakes (add correction)
# MAGIC    ↓
# MAGIC    REPEAT until mistakes are tiny!
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📝 In Math Terms (Easy Version)
# MAGIC
# MAGIC ```
# MAGIC F₀ = First guess = average
# MAGIC
# MAGIC Loop:
# MAGIC   mistakes = actual - current_prediction
# MAGIC   correction = learn_from_mistakes()
# MAGIC   new_prediction = old_prediction + (correction × 0.1)
# MAGIC   
# MAGIC Final = F₀ + correction₁ + correction₂ + ... + correctionₙ
# MAGIC ```
# MAGIC
# MAGIC **That's it! Just keep fixing mistakes!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Real Example: Predict Age from Height
# MAGIC
# MAGIC **Data:**
# MAGIC ```
# MAGIC Height (cm): [100, 120, 140, 160, 180]
# MAGIC Age (years): [5,   8,   12,  16,  20]
# MAGIC ```
# MAGIC
# MAGIC ### **Step 0: Average Guess**
# MAGIC ```
# MAGIC Average age = (5+8+12+16+20)/5 = 12.2
# MAGIC
# MAGIC Predictions: [12.2, 12.2, 12.2, 12.2, 12.2]
# MAGIC Mistakes:    [5-12.2, 8-12.2, 12-12.2, 16-12.2, 20-12.2]
# MAGIC            = [-7.2, -4.2, -0.2, +3.8, +7.8]
# MAGIC ```
# MAGIC
# MAGIC ### **Step 1: Fix Mistakes**
# MAGIC ```
# MAGIC Tree learns pattern:
# MAGIC   "Short people → age too high (subtract!)"
# MAGIC   "Tall people → age too low (add!)"
# MAGIC
# MAGIC Correction: [-2.5, -1.5, 0, +1.5, +2.5]
# MAGIC
# MAGIC New predictions: [12.2-2.5, 12.2-1.5, 12.2+0, 12.2+1.5, 12.2+2.5]
# MAGIC                = [9.7, 10.7, 12.2, 13.7, 14.7]
# MAGIC
# MAGIC New mistakes: [5-9.7, 8-10.7, ...] = [-4.7, -2.7, -0.2, +2.3, +5.3]
# MAGIC ```
# MAGIC
# MAGIC ### **Step 2-10: Keep Fixing**
# MAGIC ```
# MAGIC Each step makes predictions better!
# MAGIC
# MAGIC After 10 steps:
# MAGIC Predictions ≈ [5.1, 8.2, 12.0, 15.8, 19.9]
# MAGIC Actual      = [5.0, 8.0, 12.0, 16.0, 20.0]
# MAGIC
# MAGIC ✅ Almost perfect!
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🆚 Compare with Random Forest
# MAGIC
# MAGIC | Feature | Random Forest | Gradient Boosting |
# MAGIC |---------|---------------|-------------------|
# MAGIC | **Method** | Many guesses, vote | One guess, keep improving |
# MAGIC | **Trees** | Independent | Each fixes previous |
# MAGIC | **Analogy** | Ask 100 people, majority wins | One expert, learns from mistakes |
# MAGIC | **When** | Parallel (fast) | Sequential (careful) |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎮 Key Parameters (Simple)
# MAGIC
# MAGIC ### **Learning Rate (α):**
# MAGIC ```
# MAGIC High (0.5): Fix mistakes fast (but risky!)
# MAGIC Low (0.1): Fix slowly (but safe) ✅
# MAGIC
# MAGIC Think: How much to adjust each time?
# MAGIC ```
# MAGIC
# MAGIC ### **Number of Trees (M):**
# MAGIC ```
# MAGIC Few (5-10): Quick but not perfect
# MAGIC Many (50+): Slow but very accurate
# MAGIC
# MAGIC Think: How many times to fix mistakes?
# MAGIC ```
# MAGIC
# MAGIC ### **Tree Depth:**
# MAGIC ```
# MAGIC Shallow (2-3): Simple fixes ✅
# MAGIC Deep (8+): Complex but risky
# MAGIC
# MAGIC Think: How detailed should each fix be?
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ When to Use GB?
# MAGIC
# MAGIC **Use Gradient Boosting jab:**
# MAGIC - ✅ You want MAXIMUM accuracy
# MAGIC - ✅ You have time to tune parameters
# MAGIC - ✅ Your data is clean (no huge outliers)
# MAGIC - ✅ You need to win a Kaggle competition! 🏆
# MAGIC
# MAGIC **Don't use jab:**
# MAGIC - ❌ You need quick results (use RF instead)
# MAGIC - ❌ You have very noisy data
# MAGIC - ❌ You can't tune hyperparameters
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 The Formula (Simplified)
# MAGIC
# MAGIC ```
# MAGIC Final Prediction = Initial Guess + Fix₁ + Fix₂ + ... + Fixₙ
# MAGIC
# MAGIC Where:
# MAGIC   Initial Guess = average
# MAGIC   Each Fix = (learning from mistakes) × learning_rate
# MAGIC ```
# MAGIC
# MAGIC **That's literally it! No complex math needed to understand the concept! 🎉**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💭 Remember:
# MAGIC
# MAGIC ```
# MAGIC 🌲 Random Forest = Many independent opinions → Vote
# MAGIC 🌲→🌲→🌲 Gradient Boosting = One expert → Learn from mistakes
# MAGIC
# MAGIC Both use decision trees, but:
# MAGIC   RF: Trees don't talk to each other
# MAGIC   GB: Each tree learns from previous tree's mistakes
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Next cell mein ek super simple interactive example dekhenge! 👇**

# COMMAND ----------

# DBTITLE 1,🎮 Super Simple GB Demo (3 Students Example)
# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 SUPER SIMPLE GB DEMO: HOURS → MARKS
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

print("═" * 80)
print("🎯 GRADIENT BOOSTING: HOURS → MARKS EXAMPLE")
print("═" * 80)

# Data: Hours studied → Marks obtained
hours = np.array([1, 4, 7]).reshape(-1, 1)
marks = np.array([20.0, 60.0, 90.0])

print("\n📊 DATA:")
print("\n   Hours studied → Marks obtained")
print(f"   Hours=1 → Marks={marks[0]:.0f}")
print(f"   Hours=4 → Marks={marks[1]:.0f}")
print(f"   Hours=7 → Marks={marks[2]:.0f}")

# Parameters
learning_rate = 0.3  # Fix 30% of mistake each time
n_iterations = 10

print(f"\n⚙️  Settings:")
print(f"   Learning Rate: {learning_rate} (fix {learning_rate*100:.0f}% each time)")
print(f"   Iterations: {n_iterations}")

print("\n" + "─" * 80)

# STEP 1: Initial guess (F₀ = mean)
F_0 = marks.mean()
predictions = np.full(3, F_0)

print(f"\n🚀 STEP 1: INITIAL GUESS (F₀)")
print(f"\n   Strategy: F₀ = mean of all marks")
print(f"   F₀ = (20 + 60 + 90) / 3 = {F_0:.1f}")
print(f"\n   Predictions: [{predictions[0]:.1f}, {predictions[1]:.1f}, {predictions[2]:.1f}]")
print(f"   Actual:      [{marks[0]:.1f}, {marks[1]:.1f}, {marks[2]:.1f}]")

mistakes = marks - predictions
print(f"\n   Residuals (r): [{mistakes[0]:.1f}, {mistakes[1]:.1f}, {mistakes[2]:.1f}]")
print(f"\n   Hours=1: Need {mistakes[0]:.1f} more marks")
print(f"   Hours=4: Need {mistakes[1]:.1f} more marks")
print(f"   Hours=7: Need {mistakes[2]:.1f} more marks")

mse = np.mean(mistakes**2)
print(f"\n   Error (MSE): {mse:.2f}")

# Track history
pred_history = [predictions.copy()]
mse_history = [mse]
mistake_history = [mistakes.copy()]

print("\n" + "─" * 80)

# Iterations: Sequential tree training on residuals
for i in range(1, n_iterations + 1):
    print(f"\n🌲 STEP {i+1}: TREE ON RESIDUALS")
    
    # Current residuals
    mistakes = marks - predictions
    
    print(f"\n   Step 2 — Residuals:")
    print(f"   r = y - F_{i-1} = [{mistakes[0]:.1f}, {mistakes[1]:.1f}, {mistakes[2]:.1f}]")
    
    # Train tree on residuals (split at Hours > 3)
    tree = DecisionTreeRegressor(max_depth=1, random_state=42)
    tree.fit(hours, mistakes)
    h_m = tree.predict(hours)
    
    print(f"\n   Step 3 — Tree on residuals (split Hours>3):")
    print(f"   Left (Hours≤3): mean({mistakes[0]:.1f}) = {h_m[0]:.1f}  ← h_{i}")
    if len(set(h_m[1:])) == 1:
        print(f"   Right (Hours>3): mean({mistakes[1]:.1f}, {mistakes[2]:.1f}) = {h_m[1]:.1f}  ← h_{i}")
    else:
        print(f"   Right (Hours>3): predictions = [{h_m[1]:.1f}, {h_m[2]:.1f}]  ← h_{i}")
    
    # Update with learning rate
    correction = learning_rate * h_m
    
    print(f"\n   Step 4 — Update (α={learning_rate}):")
    print(f"   F_{i} = F_{i-1} + {learning_rate} × h_{i}")
    
    # Show detailed calculation for first iteration
    if i == 1:
        print(f"      = [{predictions[0]:.1f} + {learning_rate}×{h_m[0]:.1f}, {predictions[1]:.1f} + {learning_rate}×{h_m[1]:.1f}, {predictions[2]:.1f} + {learning_rate}×{h_m[2]:.1f}]")
    
    predictions = predictions + correction
    print(f"      = [{predictions[0]:.1f}, {predictions[1]:.1f}, {predictions[2]:.1f}]")
    
    # Calculate new residuals and MSE
    new_mistakes = marks - predictions
    mse = np.mean(new_mistakes**2)
    
    print(f"\n   Step 5 — New residuals:")
    print(f"   r⁽²⁾ = [{new_mistakes[0]:.1f}, {new_mistakes[1]:.1f}, {new_mistakes[2]:.1f}]  ← chote ho gaye ✅")
    print(f"   MSE = {mse:.1f}")
    
    # Track
    pred_history.append(predictions.copy())
    mse_history.append(mse)
    mistake_history.append(new_mistakes.copy())
    
    if i <= 3 or i == n_iterations:
        print("\n   " + "─" * 76)

print("\n" + "═" * 80)
print("🎉 TRAINING COMPLETE!")
print("═" * 80)

final_predictions = pred_history[-1]
final_mistakes = marks - final_predictions

print(f"\n📊 FINAL RESULTS:")
print(f"\n   {'Hours':<10} {'Actual':<10} {'Predicted':<12} {'Error':<10}")
print("   " + "-" * 45)
for h, actual, pred, err in zip([1, 4, 7], marks, final_predictions, final_mistakes):
    print(f"   {h:<10} {actual:<10.1f} {pred:<12.2f} {err:<10.2f}")

print(f"\n   MSE₀ = {mse_history[0]:.0f}  →  MSE₁ = {mse_history[1]:.0f}  →  MSE₂ = {mse_history[2]:.0f} ...")
print(f"   Final MSE = {mse_history[-1]:.0f}")
improvement = (mse_history[0] - mse_history[-1]) / mse_history[0] * 100
print(f"   Improvement: {improvement:.1f}% ✅")
print(f"\n   🎓 Har step → prediction better! ✅")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# TOP LEFT: MSE over iterations
ax = axes[0, 0]
ax.plot(range(len(mse_history)), mse_history, 'o-', color='darkblue', 
       linewidth=3, markersize=8)
ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
ax.set_ylabel('Error (MSE)', fontsize=13, fontweight='bold')
ax.set_title('Error Decreases Each Step', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

# Annotate
ax.annotate(f'Start\n{mse_history[0]:.1f}', xy=(0, mse_history[0]), 
           xytext=(1, mse_history[0]+50), fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
           arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate(f'End\n{mse_history[-1]:.1f}', xy=(n_iterations, mse_history[-1]), 
           xytext=(n_iterations-2, mse_history[-1]+50), fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
           arrowprops=dict(arrowstyle='->', lw=2))

# TOP RIGHT: Predictions converge to actual
ax = axes[0, 1]
iterations = range(len(pred_history))

labels = ['Hours=1', 'Hours=4', 'Hours=7']
for i, label in enumerate(labels):
    preds = [p[i] for p in pred_history]
    ax.plot(iterations, preds, 'o-', linewidth=2.5, markersize=6, label=label)
    ax.axhline(marks[i], color='gray', linestyle='--', alpha=0.5)

ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
ax.set_ylabel('Predicted Marks', fontsize=13, fontweight='bold')
ax.set_title('Predictions Converge to Truth', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# BOTTOM LEFT: Residuals shrinking
ax = axes[1, 0]
for i, label in enumerate(labels):
    mistakes = [m[i] for m in mistake_history]
    ax.plot(iterations, np.abs(mistakes), 'o-', linewidth=2.5, markersize=6, label=label)

ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
ax.set_ylabel('|Residual|', fontsize=13, fontweight='bold')
ax.set_title('Residuals Get Smaller', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# BOTTOM RIGHT: Final comparison
ax = axes[1, 1]
x_pos = np.arange(3)
width = 0.35

ax.bar(x_pos - width/2, marks, width, label='Actual', 
      color='green', alpha=0.7, edgecolor='black', linewidth=2)
ax.bar(x_pos + width/2, final_predictions, width, label='Predicted', 
      color='blue', alpha=0.7, edgecolor='black', linewidth=2)

ax.set_xlabel('Hours Studied', fontsize=13, fontweight='bold')
ax.set_ylabel('Marks', fontsize=13, fontweight='bold')
ax.set_title('Final: Actual vs Predicted', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['1', '4', '7'])
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')

# Add values on bars
for i, (actual, pred) in enumerate(zip(marks, final_predictions)):
    ax.text(i - width/2, actual + 3, f'{actual:.0f}', ha='center', fontsize=10, fontweight='bold')
    ax.text(i + width/2, pred + 3, f'{pred:.1f}', ha='center', fontsize=10, fontweight='bold')

plt.suptitle('🎯 Gradient Boosting: Hours → Marks Example', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("\n" + "═" * 80)
print("\n💡 KEY INSIGHTS:")
print("\n   1. Started with F₀ = 56.7 (average)")
print(f"   2. Each tree learned residuals, corrected by α={learning_rate}")
print(f"   3. After {n_iterations} steps: predictions ≈ actual! ✅")
print("   4. This is EXACTLY how Gradient Boosting works! 🎯")
print("\n   Formula: F_m = F_{m-1} + α × h_m")
print(f"            F_m = F_{{m-1}} + {learning_rate} × (tree on residuals)")
print("\n   🎓 EXAMPLE MATCHES USER'S CALCULATION:")
print(f"   MSE₀ = {mse_history[0]:.0f}  →  MSE₁ = {mse_history[1]:.0f}  →  MSE₂ = {mse_history[2]:.0f} ...")
print("   Each step → residuals smaller → predictions better! ✅")
print("\n" + "═" * 80)

# COMMAND ----------

# DBTITLE 1,Demo: GB Iteration-by-Iteration
# ═══════════════════════════════════════════════════════════════════════════════
# DEMO: GRADIENT BOOSTING CONCRETE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

print("═" * 80)
print("🌲→🌲→🌲 GRADIENT BOOSTING: CONCRETE EXAMPLE")
print("═" * 80)

# Simple data
X_concrete = np.array([[1], [2], [3]])
y_concrete = np.array([3.0, 5.0, 8.0])

print("\n📊 Data:")
print(f"   x = {X_concrete.flatten().tolist()}")
print(f"   y = {y_concrete.tolist()}")

# Parameters
alpha = 0.1  # Learning rate
M = 10  # Iterations

print(f"\n⚙️  Parameters:")
print(f"   Learning rate (α): {alpha}")
print(f"   Iterations (M): {M}")
print(f"   Tree depth: 2")

# Initialize
F_0 = y_concrete.mean()
F_current = np.full_like(y_concrete, F_0, dtype=float)

print(f"\n\n🚀 ITERATION 0 (Initialization):")
print(f"   F₀ = mean(y) = {F_0:.2f}")
print(f"   Predictions:  {F_current}")
residuals_0 = y_concrete - F_current
print(f"   Residuals:    {residuals_0}")
mse_0 = np.mean(residuals_0**2)
print(f"   MSE:          {mse_0:.4f}")

# Store history
F_history = [F_current.copy()]
mse_history = [mse_0]
residual_history = [residuals_0.copy()]
tree_predictions_history = []

print("\n" + "-" * 80)

# Manual iterations
for m in range(1, M + 1):
    print(f"\n🌳 ITERATION {m}:")
    
    # Step 1: Calculate residuals
    residuals = y_concrete - F_current
    print(f"\n   Step 1 - Residuals:")
    print(f"      r = y - F_{{m-1}}")
    print(f"      r = {y_concrete} - {F_current}")
    print(f"      r = {residuals}")
    
    # Step 2: Train tree on residuals
    tree = DecisionTreeRegressor(max_depth=2, random_state=m)
    tree.fit(X_concrete, residuals)
    h_m = tree.predict(X_concrete)
    
    print(f"\n   Step 2 - Tree predicts residuals:")
    print(f"      h_{m}(x) = {h_m}")
    
    # Step 3: Update
    F_current = F_current + alpha * h_m
    
    print(f"\n   Step 3 - Update:")
    print(f"      F_{m} = F_{{m-1}} + {alpha} × h_{m}")
    print(f"          = {F_current - alpha * h_m} + {alpha * h_m}")
    print(f"          = {F_current}")
    
    # Calculate MSE
    new_residuals = y_concrete - F_current
    mse = np.mean(new_residuals**2)
    
    print(f"\n   📈 Results:")
    print(f"      New predictions: {F_current}")
    print(f"      New residuals:   {new_residuals}")
    print(f"      MSE:             {mse:.4f}")
    print(f"      Improvement:     {mse_history[-1] - mse:.4f}  ✅")
    
    # Store
    F_history.append(F_current.copy())
    mse_history.append(mse)
    residual_history.append(new_residuals.copy())
    tree_predictions_history.append(h_m.copy())
    
    print("   " + "-" * 76)

print("\n" + "═" * 80)
print("🎉 TRAINING COMPLETE!")
print("═" * 80)

print(f"\n📈 Summary:")
print(f"   Initial MSE (F₀):  {mse_history[0]:.4f}")
print(f"   Final MSE (F_{M}):   {mse_history[-1]:.4f}")
print(f"   Improvement:       {(mse_history[0] - mse_history[-1]) / mse_history[0] * 100:.1f}%")

print(f"\n🎯 Final Predictions vs Truth:")
for i in range(len(y_concrete)):
    print(f"   x={X_concrete[i,0]}: Predicted={F_current[i]:.2f}, True={y_concrete[i]:.2f}, Error={abs(F_current[i]-y_concrete[i]):.4f}")

# Create detailed visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# TOP LEFT: MSE over iterations
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(range(len(mse_history)), mse_history, 'o-', color='darkblue', 
        linewidth=3, markersize=8, label='MSE')
ax1.set_xlabel('Iteration (m)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
ax1.set_title(f'GB Convergence (α={alpha}, M={M})', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Annotate key points
for m in [0, M//2, M]:
    ax1.annotate(f'M={m}\nMSE={mse_history[m]:.3f}', 
               xy=(m, mse_history[m]), 
               xytext=(m+0.5, mse_history[m]+0.3),
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# MIDDLE: Predictions evolution for each sample
for i, x_val in enumerate(X_concrete.flatten()):
    ax = fig.add_subplot(gs[1, i])
    
    predictions_i = [F[i] for F in F_history]
    
    ax.plot(range(len(predictions_i)), predictions_i, 'o-', 
           color='blue', linewidth=2.5, markersize=6, label=f'F_m(x={x_val})')
    ax.axhline(y_concrete[i], color='green', linestyle='--', linewidth=2.5,
              label=f'True y={y_concrete[i]}')
    
    ax.set_xlabel('Iteration (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Prediction', fontsize=11, fontweight='bold')
    ax.set_title(f'Sample x={x_val}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(y_concrete.min()-1, y_concrete.max()+1)

# BOTTOM LEFT: Residuals shrinking
ax = fig.add_subplot(gs[2, 0])
residual_mags = [np.abs(r).mean() for r in residual_history]
ax.plot(range(len(residual_mags)), residual_mags, 'o-', 
       color='purple', linewidth=2.5, markersize=6)
ax.set_xlabel('Iteration (m)', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean |Residual|', fontsize=11, fontweight='bold')
ax.set_title('Residuals Shrink', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# BOTTOM MIDDLE: Additive formula visualization
ax = fig.add_subplot(gs[2, 1])
ax.axis('off')

formula_text = f"""
📝 ADDITIVE MODEL:

F_{M}(x) = F₀ + α·h₁ + α·h₂ + ... + α·h_{M}

For x=1:
  F_{M}(1) = {F_0:.2f} + {alpha}×("""

for m, h_pred in enumerate(tree_predictions_history[:3], 1):
    formula_text += f"{h_pred[0]:.2f} + "

formula_text += f"...)\n           = {F_current[0]:.2f}"

formula_text += f"""

🎯 Each tree corrects previous mistakes!
"""

ax.text(0.1, 0.5, formula_text, fontsize=10, family='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1),
       verticalalignment='center')

# BOTTOM RIGHT: Final comparison
ax = fig.add_subplot(gs[2, 2])
x_pos = range(len(X_concrete))
width = 0.35

ax.bar([p - width/2 for p in x_pos], y_concrete, width, 
      label='True y', color='green', alpha=0.7, edgecolor='black', linewidth=2)
ax.bar([p + width/2 for p in x_pos], F_current, width, 
      label=f'F_{M} (predicted)', color='blue', alpha=0.7, edgecolor='black', linewidth=2)

ax.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
ax.set_ylabel('Value', fontsize=11, fontweight='bold')
ax.set_title('Final Predictions', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'x={int(x)}' for x in X_concrete.flatten()])
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

plt.suptitle('🌲→🌲→🌲 Gradient Boosting: Iteration-by-Iteration', 
            fontsize=16, fontweight='bold', y=0.995)
plt.show()

print("\n" + "═" * 80)
print("\n💡 KEY TAKEAWAYS:")
print("   • Each tree learns residuals (mistakes of previous trees)")
print("   • Learning rate α controls step size (0.1 = safe)")
print("   • Additive updates: F_m = F_{m-1} + α·h_m")
print("   • MSE decreases monotonically (with proper parameters)")
print(f"   • After {M} iterations: predictions ≈ truth! ✅")
print("\n" + "═" * 80)

# COMMAND ----------

# DBTITLE 1,🎮 Interactive: GB Parameter Explorer
# ═══════════════════════════════════════════════════════════════════════════════
# 🎮 INTERACTIVE: GRADIENT BOOSTING PARAMETER EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

from ipywidgets import interact, FloatSlider, IntSlider
from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

def explore_gb_parameters(learning_rate, n_iterations, max_depth):
    """
    Interactive GB parameter explorer
    """
    clear_output(wait=True)
    
    # Generate data
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 5)
    y_true = 2*X[:, 0] + 1.5*X[:, 1] - X[:, 2] + 10
    y = y_true + np.random.randn(n) * 2
    
    # Train GB
    F_m = np.full(n, y.mean())
    mse_history = [np.mean((F_m - y_true)**2)]
    F_history = [F_m.copy()]
    
    for m in range(1, n_iterations + 1):
        residuals = y - F_m
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=m)
        tree.fit(X, residuals)
        h_m = tree.predict(X)
        F_m = F_m + learning_rate * h_m
        mse_history.append(np.mean((F_m - y_true)**2))
        F_history.append(F_m.copy())
    
    print("═" * 80)
    print("🎮 GRADIENT BOOSTING PARAMETER EXPLORER")
    print("═" * 80)
    print(f"\n⚙️  Configuration:")
    print(f"   Learning Rate (α): {learning_rate:.3f}")
    print(f"   Iterations (M): {n_iterations}")
    print(f"   Tree Depth: {max_depth}")
    
    print(f"\n📊 RESULTS:")
    print(f"   Initial MSE: {mse_history[0]:.4f}")
    print(f"   Final MSE: {mse_history[-1]:.4f}")
    improvement = (mse_history[0] - mse_history[-1]) / mse_history[0] * 100
    print(f"   Improvement: {improvement:.1f}%")
    
    # Assess convergence
    if len(mse_history) > 5:
        recent_change = abs(mse_history[-1] - mse_history[-5]) / mse_history[-5] * 100
        if recent_change < 1:
            status = "✅ CONVERGED"
        elif mse_history[-1] > mse_history[-5]:
            status = "⚠️ DIVERGING (try lower α!)"
        else:
            status = "🔄 CONVERGING (try more iterations)"
    else:
        status = "🔄 EARLY STAGE"
    
    print(f"   Status: {status}")
    
    print(f"\n💡 OBSERVATIONS:")
    if learning_rate > 0.3:
        print(f"   • High α={learning_rate:.3f}: Fast but may overshoot!")
    elif learning_rate < 0.05:
        print(f"   • Low α={learning_rate:.3f}: Slow but stable")
    else:
        print(f"   • Balanced α={learning_rate:.3f}: Good trade-off")
    
    if max_depth > 5:
        print(f"   • Deep trees (depth={max_depth}): Risk of overfitting")
    elif max_depth < 3:
        print(f"   • Shallow trees (depth={max_depth}): May underfit")
    else:
        print(f"   • Moderate depth={max_depth}: Good balance")
    
    if improvement > 90:
        print(f"   • Excellent improvement! Model is learning well.")
    elif improvement > 50:
        print(f"   • Good improvement. Try more iterations or higher α.")
    else:
        print(f"   • Low improvement. Check parameters or data.")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # TOP LEFT: MSE convergence
    ax = axes[0, 0]
    iterations = range(len(mse_history))
    ax.plot(iterations, mse_history, 'o-', color='darkblue', linewidth=2.5, markersize=6)
    ax.set_xlabel('Iteration (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
    ax.set_title(f'MSE Convergence (α={learning_rate:.3f}, depth={max_depth})', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Annotate start and end
    ax.annotate(f'Start\nMSE={mse_history[0]:.2f}', 
               xy=(0, mse_history[0]), 
               xytext=(n_iterations*0.15, mse_history[0]+1),
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax.annotate(f'End\nMSE={mse_history[-1]:.2f}', 
               xy=(n_iterations, mse_history[-1]), 
               xytext=(n_iterations*0.6, mse_history[-1]+1),
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # TOP RIGHT: Predictions vs Truth (first 30 samples)
    ax = axes[0, 1]
    sample_idx = range(30)
    ax.scatter(sample_idx, y_true[:30], color='green', s=100, marker='o', 
              edgecolors='black', linewidth=2, label='True y', zorder=3, alpha=0.7)
    ax.scatter(sample_idx, F_history[0][:30], color='red', s=80, marker='x', 
              linewidth=2, label=f'F₀ (init)', alpha=0.6)
    ax.scatter(sample_idx, F_history[-1][:30], color='blue', s=60, marker='+', 
              linewidth=2.5, label=f'F_{n_iterations} (final)', alpha=0.9)
    
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction', fontsize=12, fontweight='bold')
    ax.set_title('Predictions: Initial → Final', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)
    
    # BOTTOM LEFT: Error distribution
    ax = axes[1, 0]
    errors_init = F_history[0] - y_true
    errors_final = F_history[-1] - y_true
    
    ax.hist(errors_init, bins=20, alpha=0.6, color='red', edgecolor='black', label=f'Initial (σ={np.std(errors_init):.2f})')
    ax.hist(errors_final, bins=20, alpha=0.7, color='blue', edgecolor='black', label=f'Final (σ={np.std(errors_final):.2f})')
    ax.axvline(0, color='green', linestyle='--', linewidth=2.5, label='Perfect (error=0)')
    
    ax.set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Error Distribution: Before vs After', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    
    # BOTTOM RIGHT: Learning curve comparison
    ax = axes[1, 1]
    
    # Show effect of different learning rates
    lr_compare = [learning_rate * 0.5, learning_rate, learning_rate * 2]
    colors_lr = ['green', 'blue', 'red']
    
    for lr, color in zip(lr_compare, colors_lr):
        F_temp = np.full(n, y.mean())
        mse_temp = []
        
        for m in range(1, min(n_iterations + 1, 31)):
            residuals = y - F_temp
            tree = DecisionTreeRegressor(max_depth=max_depth, random_state=m)
            tree.fit(X, residuals)
            h_m = tree.predict(X)
            F_temp = F_temp + lr * h_m
            mse_temp.append(np.mean((F_temp - y_true)**2))
        
        label = f'α={lr:.3f}'
        if lr == learning_rate:
            label += ' (current)'
            linewidth = 3
        else:
            linewidth = 2
        
        ax.plot(range(1, len(mse_temp)+1), mse_temp, '-', linewidth=linewidth, 
               color=color, label=label, alpha=0.8)
    
    ax.set_xlabel('Iteration (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
    ax.set_title('Learning Rate Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)
    
    # Add insight box
    if learning_rate > 0.3:
        insight = 'Higher α:\nFaster but\nless stable'
        box_color = 'lightcoral'
    elif learning_rate < 0.05:
        insight = 'Lower α:\nSlower but\nmore stable'
        box_color = 'lightgreen'
    else:
        insight = 'Balanced α:\nGood trade-off'
        box_color = 'lightyellow'
    
    ax.text(0.65, 0.85, insight, transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))
    
    plt.suptitle('🌲→🌲→🌲 Gradient Boosting: Interactive Parameter Explorer', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "═" * 80)
    print("\n🎯 TRY THESE EXPERIMENTS:")
    print("   1. Increase α → See faster convergence (but watch for instability!)")
    print("   2. Decrease α → See slower but smoother convergence")
    print("   3. Increase depth → See faster learning (but risk overfitting)")
    print("   4. Increase M → See if MSE keeps improving or plateaus")
    print("\n💡 IDEAL SETTINGS: α=0.1, M=20-50, depth=3-5")
    print("\n" + "═" * 80)

# Create interactive widget
interact(
    explore_gb_parameters,
    learning_rate=FloatSlider(
        value=0.1, min=0.01, max=0.5, step=0.01,
        description='α (Learning Rate):',
        style={'description_width': 'initial'},
        continuous_update=False
    ),
    n_iterations=IntSlider(
        value=20, min=5, max=50, step=5,
        description='M (Iterations):',
        style={'description_width': 'initial'},
        continuous_update=False
    ),
    max_depth=IntSlider(
        value=3, min=1, max=8, step=1,
        description='Tree Depth:',
        style={'description_width': 'initial'},
        continuous_update=False
    )
);

# COMMAND ----------

# DBTITLE 1,🎮 Interactive: RF Variance Explorer
# ═══════════════════════════════════════════════════════════════════════════════
# 🎮 INTERACTIVE: RANDOM FOREST VARIANCE EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

def explore_rf_variance(rho, B):
    """
    Interactive RF variance explorer
    """
    clear_output(wait=True)
    
    # Assume sigma^2 = 1 for simplicity
    sigma_sq = 1.0
    
    # Calculate variance
    var_single = sigma_sq
    var_rf = rho * sigma_sq + (1 - rho) * sigma_sq / B
    reduction = (var_single - var_rf) / var_single * 100
    
    # Components
    irreducible = rho * sigma_sq
    reducible = (1 - rho) * sigma_sq / B
    
    print("═" * 80)
    print("🌲 RANDOM FOREST VARIANCE EXPLORER")
    print("═" * 80)
    print(f"\n⚙️  Configuration:")
    print(f"   Correlation (ρ): {rho:.2f}")
    print(f"   Number of Trees (B): {B}")
    print(f"   Individual tree variance (σ²): {sigma_sq:.2f}")
    
    print(f"\n📊 VARIANCE CALCULATION:")
    print(f"   Formula: Var(RF) = ρσ² + (1-ρ)σ²/B")
    print(f"")
    print(f"   Var(RF) = {rho:.2f}×{sigma_sq:.2f} + {(1-rho):.2f}×{sigma_sq:.2f}/{B}")
    print(f"          = {irreducible:.4f} + {reducible:.4f}")
    print(f"          = {var_rf:.4f}")
    
    print(f"\n⭐ RESULTS:")
    print(f"   Single Tree Variance: {var_single:.4f}")
    print(f"   RF Variance: {var_rf:.4f}")
    print(f"   Variance Reduction: {reduction:.1f}%")
    
    print(f"\n🔑 COMPONENTS:")
    print(f"   Irreducible (ρσ²): {irreducible:.4f} ({irreducible/var_rf*100:.1f}% of RF variance)")
    print(f"   Reducible ((1-ρ)σ²/B): {reducible:.4f} ({reducible/var_rf*100:.1f}% of RF variance)")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # LEFT: Variance components
    components = ['Irreducible\n(ρσ²)', 'Reducible\n((1-ρ)σ²/B)', 'Total RF\nVariance', 'Single Tree\nVariance']
    values = [irreducible, reducible, var_rf, var_single]
    colors_comp = ['red', 'orange', 'green', 'gray']
    
    bars = ax1.bar(components, values, color=colors_comp, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Variance', fontsize=12, fontweight='bold')
    ax1.set_title(f'Variance Breakdown (ρ={rho:.2f}, B={B})', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.2)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.03, 
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Add stacked bar for RF
    ax1.bar(['Total RF\nVariance'], [irreducible], color='red', alpha=0.5, edgecolor='black', linewidth=1)
    ax1.bar(['Total RF\nVariance'], [reducible], bottom=[irreducible], color='orange', alpha=0.5, edgecolor='black', linewidth=1)
    
    # RIGHT: Variance vs B for current ρ
    B_range = np.arange(1, 101)
    var_curve = rho * sigma_sq + (1 - rho) * sigma_sq / B_range
    
    ax2.plot(B_range, var_curve, '-', linewidth=3, color='darkgreen', label=f'ρ={rho:.2f}')
    ax2.axhline(var_single, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Single tree')
    ax2.axhline(rho * sigma_sq, color='orange', linestyle=':', linewidth=2, alpha=0.7, 
               label=f'Asymptote (ρσ²={irreducible:.2f})')
    ax2.scatter([B], [var_rf], s=200, color='blue', marker='o', edgecolors='black', 
               linewidth=3, zorder=5, label=f'Current (B={B})')
    
    ax2.set_xlabel('Number of Trees (B)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Variance', fontsize=12, fontweight='bold')
    ax2.set_title(f'Variance vs B (Current ρ={rho:.2f})', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    plt.suptitle('🎮 Interactive RF Variance Explorer - Drag Sliders!', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "═" * 80)
    print("\n💡 Insights:")
    if rho < 0.3:
        print("   ✅ Low correlation! Trees are diverse. Ensemble works great!")
    elif rho < 0.6:
        print("   🟡 Moderate correlation. Decent ensemble benefit.")
    else:
        print("   ⚠️  High correlation! Trees are similar. Limited ensemble benefit.")
    
    if B < 10:
        print("   🟠 Few trees. More trees would reduce variance further.")
    elif B < 50:
        print("   ✅ Good number of trees. Balanced.")
    else:
        print("   🟢 Many trees! Diminishing returns beyond this point.")
    
    print(f"\n   As B→∞: Var(RF) → {rho * sigma_sq:.4f} (can't go lower!)")
    print("\n" + "═" * 80)

print("🎮 INTERACTIVE RF VARIANCE EXPLORER")
print("Experiment with correlation and number of trees!\n")

rho_slider = widgets.FloatSlider(
    value=0.30, min=0.0, max=0.95, step=0.05,
    description='Correlation (ρ):', continuous_update=False,
    style={'description_width': '150px'}
)

B_rf_slider = widgets.IntSlider(
    value=10, min=1, max=100, step=1,
    description='N Trees (B):', continuous_update=False,
    style={'description_width': '150px'}
)

interactive_rf_var = widgets.interactive(
    explore_rf_variance,
    rho=rho_slider,
    B=B_rf_slider
)

display(interactive_rf_var)

# COMMAND ----------

# DBTITLE 1,🎮 Interactive: GB Residual Learning
# ═══════════════════════════════════════════════════════════════════════════════
# 🎮 INTERACTIVE: GRADIENT BOOSTING EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

def explore_gb_learning(alpha, M_iter):
    """
    Interactive GB residual learning explorer
    """
    clear_output(wait=True)
    
    # Generate small dataset for demo
    np.random.seed(42)
    n = 50
    X_demo = np.random.randn(n, 5)
    y_true_demo = 3*X_demo[:, 0] + 2*X_demo[:, 1] + 10
    y_demo = y_true_demo + np.random.randn(n) * 1.5
    
    # Train GB
    F_m = np.full(n, y_demo.mean())
    mse_track = [np.mean((F_m - y_true_demo)**2)]
    F_track = [F_m[0]]
    
    for m in range(1, M_iter + 1):
        residuals = y_demo - F_m
        tree = DecisionTreeRegressor(max_depth=2, random_state=m)
        tree.fit(X_demo, residuals)
        h_m = tree.predict(X_demo)
        F_m = F_m + alpha * h_m
        mse_track.append(np.mean((F_m - y_true_demo)**2))
        F_track.append(F_m[0])
    
    print("═" * 80)
    print("🌲→🌲→🌲 GRADIENT BOOSTING EXPLORER")
    print("═" * 80)
    print(f"\n⚙️  Configuration:")
    print(f"   Learning Rate (α): {alpha:.2f}")
    print(f"   Iterations (M): {M_iter}")
    print(f"   Tree Depth: 2")
    
    print(f"\n📊 TRAINING PROGRESS:")
    print(f"   Initial F₀: {F_track[0]:.4f}")
    print(f"   Final F_{M_iter}: {F_track[-1]:.4f}")
    print(f"   True y[0]: {y_true_demo[0]:.4f}")
    print(f"   Error: {abs(F_track[-1] - y_true_demo[0]):.4f}")
    
    print(f"\n⭐ RESULTS:")
    print(f"   Initial MSE: {mse_track[0]:.4f}")
    print(f"   Final MSE: {mse_track[-1]:.4f}")
    improvement = (mse_track[0] - mse_track[-1]) / mse_track[0] * 100
    print(f"   Improvement: {improvement:.1f}%")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # LEFT: MSE convergence
    ax = axes[0]
    iterations = range(len(mse_track))
    ax.plot(iterations, mse_track, 'o-', color='darkblue', linewidth=2.5, markersize=6)
    ax.set_xlabel('Iteration (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
    ax.set_title(f'GB Convergence (α={alpha:.2f}, M={M_iter})', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Annotate key points
    ax.annotate(f'Start\nMSE={mse_track[0]:.2f}', 
               xy=(0, mse_track[0]), 
               xytext=(M_iter*0.1, mse_track[0]+1),
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax.annotate(f'End\nMSE={mse_track[-1]:.2f}', 
               xy=(M_iter, mse_track[-1]), 
               xytext=(M_iter*0.6, mse_track[-1]+1),
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax.text(0.5, 0.95, f'F_m = F_{{m-1}} + {alpha:.2f}·h_m', 
           transform=ax.transAxes, ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # RIGHT: Prediction convergence
    ax = axes[1]
    ax.plot(range(len(F_track)), F_track, 'o-', color='blue', linewidth=2.5, 
           markersize=6, label='F_m[0]')
    ax.axhline(y_true_demo[0], color='green', linestyle='--', linewidth=2.5, 
              alpha=0.7, label=f'True y[0]={y_true_demo[0]:.2f}')
    ax.set_xlabel('Iteration (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction for Sample 0', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Converges to Truth', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.suptitle('🎮 Interactive GB Residual Learning - Drag Sliders!', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "═" * 80)
    print("\n💡 Insights:")
    
    if alpha < 0.05:
        print("   🐌 Very low learning rate. Slow but stable convergence.")
    elif alpha < 0.2:
        print("   ✅ Good learning rate! Balanced speed and stability.")
    elif alpha < 0.5:
        print("   🟡 Moderate learning rate. Faster but watch for overfitting.")
    else:
        print("   ⚠️  High learning rate! May oscillate or diverge.")
    
    if M_iter < 10:
        print("   🟠 Few iterations. Model may underfit.")
    elif M_iter < 30:
        print("   ✅ Good number of iterations!")
    else:
        print("   🟡 Many iterations. Watch for overfitting on training data.")
    
    if improvement > 80:
        print(f"   🎉 Excellent! {improvement:.0f}% error reduction!")
    elif improvement > 50:
        print(f"   ✅ Good progress. {improvement:.0f}% error reduction.")
    else:
        print(f"   🟡 Modest improvement. Try adjusting α or M.")
    
    print("\n   Each iteration corrects previous mistakes (residuals).")
    print(f"   F_{M_iter} = F₀ + α·Σh_m = {F_track[0]:.2f} + {alpha:.2f}×(sum of trees)")
    print("\n" + "═" * 80)

print("🎮 INTERACTIVE GB RESIDUAL LEARNING EXPLORER")
print("Experiment with learning rate and iterations!\n")

alpha_slider = widgets.FloatSlider(
    value=0.10, min=0.01, max=0.50, step=0.01,
    description='Learning Rate (α):', continuous_update=False,
    style={'description_width': '150px'}
)

M_slider = widgets.IntSlider(
    value=10, min=1, max=50, step=1,
    description='Iterations (M):', continuous_update=False,
    style={'description_width': '150px'}
)

interactive_gb = widgets.interactive(
    explore_gb_learning,
    alpha=alpha_slider,
    M_iter=M_slider
)

display(interactive_gb)

# COMMAND ----------

# DBTITLE 1,🎓 Summary & Final Takeaways
# MAGIC %md
# MAGIC # 🎓 Summary: RF vs GB Mathematics
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📝 Core Formulas Recap
# MAGIC
# MAGIC ### **Random Forest: Variance Reduction**
# MAGIC
# MAGIC ```
# MAGIC Var(RF) = ρσ² + (1-ρ)σ²/B
# MAGIC
# MAGIC Where:
# MAGIC   ρ = correlation between trees
# MAGIC   σ² = variance of individual trees
# MAGIC   B = number of trees
# MAGIC
# MAGIC Prediction: ŷ = (1/B) Σ_{b=1}^B h_b(x)
# MAGIC ```
# MAGIC
# MAGIC **Key Insights:**
# MAGIC * **More trees (↑B)** → **Lower variance** (diminishing returns)
# MAGIC * **Less correlation (↓ρ)** → **More reduction** (random features help!)
# MAGIC * **Asymptote:** Even with B→∞, Var(RF) ≥ ρσ² (irreducible floor)
# MAGIC
# MAGIC **Example (ρ=0.30, B=10):**
# MAGIC ```
# MAGIC Var(RF) = 0.30 + 0.70/10 = 0.370
# MAGIC Variance reduction: 63%! ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Gradient Boosting: Residual Learning**
# MAGIC
# MAGIC ```
# MAGIC F_M(x) = F₀ + α·h₁(x) + α·h₂(x) + ... + α·hM(x)
# MAGIC
# MAGIC At each iteration m:
# MAGIC   1. r_i = y_i - F_{m-1}(x_i)    ← residuals
# MAGIC   2. h_m = train_tree(X, r)      ← fit to mistakes
# MAGIC   3. F_m = F_{m-1} + α·h_m      ← update
# MAGIC ```
# MAGIC
# MAGIC **Key Insights:**
# MAGIC * **Sequential learning** → Each tree corrects previous mistakes
# MAGIC * **Learning rate α** → Controls step size (smaller = safer)
# MAGIC * **More iterations (M)** → Lower bias (but risk overfitting!)
# MAGIC
# MAGIC **Example (α=0.10, M=10):**
# MAGIC ```
# MAGIC F₀ = 5.00 (initial)
# MAGIC F₁₀ = 6.954 (after 10 iterations)
# MAGIC Error: 4.0 → 1.046 (74% improvement!) ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚖️ Quick Comparison Table
# MAGIC
# MAGIC | **Aspect** | **Random Forest** | **Gradient Boosting** |
# MAGIC |------------|-------------------|----------------------|
# MAGIC | **Philosophy** | Reduce variance through averaging | Reduce bias through residual correction |
# MAGIC | **Tree Training** | Parallel (independent) | Sequential (dependent) |
# MAGIC | **Data per Tree** | Bootstrap sample | Full data (on residuals) |
# MAGIC | **Features** | Random √p subset | All p features |
# MAGIC | **Combination** | Average: (1/B)Σh_b | Additive: F₀ + αΣh_m |
# MAGIC | **Main Strength** | Robustness, low variance | Accuracy, low bias |
# MAGIC | **Overfitting Risk** | Low (naturally regularized) | Higher (needs tuning) |
# MAGIC | **Tuning** | Minimal (B, depth) | Critical (α, M, depth) |
# MAGIC | **Speed** | Fast (parallelizable) | Slower (sequential) |
# MAGIC | **Interpretability** | Moderate (feature importance) | Lower (additive complexity) |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 When to Use Which?
# MAGIC
# MAGIC ### 🌲 **Use Random Forest when:**
# MAGIC * ✅ You need **robustness** and don't want to tune much
# MAGIC * ✅ You have **high-variance** base learners
# MAGIC * ✅ You want **fast training** (can parallelize)
# MAGIC * ✅ You need **feature importances** for interpretation
# MAGIC * ✅ Your data is **noisy** or has **outliers**
# MAGIC * ✅ You want a **strong baseline** quickly
# MAGIC
# MAGIC **Typical use cases:**
# MAGIC * Classification with categorical features
# MAGIC * Datasets with many irrelevant features
# MAGIC * When you need quick results without hyperparameter tuning
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🌲→🌲→🌲 **Use Gradient Boosting when:**
# MAGIC * ✅ You need **maximum accuracy** and can tune carefully
# MAGIC * ✅ You have **high-bias** base learners (shallow trees)
# MAGIC * ✅ You're willing to invest time in **hyperparameter tuning**
# MAGIC * ✅ Your data is **clean** and well-preprocessed
# MAGIC * ✅ You have **compute budget** for sequential training
# MAGIC * ✅ You can use **validation set** or early stopping
# MAGIC
# MAGIC **Typical use cases:**
# MAGIC * Kaggle competitions (XGBoost, LightGBM, CatBoost)
# MAGIC * Production systems where accuracy is critical
# MAGIC * Structured/tabular data with numerical features
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 Practical Recommendations
# MAGIC
# MAGIC ### **Start with Random Forest:**
# MAGIC 1. Train RF with default parameters
# MAGIC 2. Get baseline performance quickly
# MAGIC 3. Understand feature importances
# MAGIC 4. Use as sanity check
# MAGIC
# MAGIC ### **Then try Gradient Boosting:**
# MAGIC 1. Start with conservative α (0.05-0.1)
# MAGIC 2. Use validation set or cross-validation
# MAGIC 3. Tune M with early stopping
# MAGIC 4. Tune tree depth (3-6 typically)
# MAGIC 5. Consider XGBoost/LightGBM (built-in regularization)
# MAGIC
# MAGIC ### **Or use both!**
# MAGIC * **Stacking:** Use RF and GB predictions as features for meta-learner
# MAGIC * **Voting:** Average predictions from both
# MAGIC * **Context-dependent:** RF for exploration, GB for production
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Further Reading
# MAGIC
# MAGIC **Random Forest:**
# MAGIC * Breiman (2001): "Random Forests" - original paper
# MAGIC * Focus: Bagging + random features → decorrelated trees
# MAGIC
# MAGIC **Gradient Boosting:**
# MAGIC * Friedman (2001): "Greedy Function Approximation" - original paper
# MAGIC * Modern variants: XGBoost, LightGBM, CatBoost (add regularization)
# MAGIC
# MAGIC **Both:**
# MAGIC * "Elements of Statistical Learning" (Hastie, Tibshirani, Friedman)
# MAGIC * Chapter 15 (Random Forests), Chapter 10 (Boosting)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ What We Covered
# MAGIC
# MAGIC 1. ✅ **RF Variance Formula:** Derived and visualized Var(RF) = ρσ² + (1-ρ)σ²/B
# MAGIC 2. ✅ **GB Residual Learning:** Step-by-step F_M = F₀ + αΣh_m iteration
# MAGIC 3. ✅ **Interactive Demos:** Experiment with ρ, B, α, M parameters
# MAGIC 4. ✅ **Side-by-side Comparison:** When to use which
# MAGIC 5. ✅ **Real Examples:** Actual variance reduction and convergence
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **🎉 You now understand the mathematics behind both Random Forest and Gradient Boosting!**
# MAGIC
# MAGIC **🚀 Go build some ensembles!**

# COMMAND ----------

# DBTITLE 1,🎓 Interview: Test Your Understanding
# MAGIC %md
# MAGIC # 🎓 INTERVIEW SESSION: Test Your Understanding
# MAGIC
# MAGIC **Welcome to your RF vs GB knowledge assessment!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Interview Structure
# MAGIC
# MAGIC I'll ask you questions across 5 rounds:
# MAGIC
# MAGIC ### **Round 1: Foundations (Statistics Basics)** 🟢
# MAGIC * Mean, variance, standard deviation
# MAGIC * Covariance and correlation
# MAGIC * Basic properties
# MAGIC
# MAGIC ### **Round 2: Decision Trees** 🌳
# MAGIC * Gini impurity
# MAGIC * Information gain
# MAGIC * Split criteria
# MAGIC
# MAGIC ### **Round 3: Random Forest** 🌲🌲🌲
# MAGIC * Bootstrap sampling
# MAGIC * Variance reduction
# MAGIC * The famous formula: ρσ² + (1-ρ)σ²/B
# MAGIC * Hyperparameters
# MAGIC
# MAGIC ### **Round 4: Gradient Boosting** 🌲→🌲→🌲
# MAGIC * Sequential learning
# MAGIC * Residuals
# MAGIC * Learning rate
# MAGIC * The update rule: F_m = F_{m-1} + α×h_m
# MAGIC
# MAGIC ### **Round 5: Comparison & Applications** ⚖️
# MAGIC * RF vs GB tradeoffs
# MAGIC * When to use what
# MAGIC * Real-world scenarios
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Scoring System
# MAGIC
# MAGIC * **Beginner (0-40%)**: Review the basic concepts
# MAGIC * **Intermediate (41-70%)**: Good foundation, practice more
# MAGIC * **Advanced (71-85%)**: Strong understanding!
# MAGIC * **Expert (86-100%)**: Interview-ready! 🚀
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⏱️ Format
# MAGIC
# MAGIC * **Multiple choice** and **numeric answer** questions
# MAGIC * Immediate feedback after each question
# MAGIC * Detailed explanations for wrong answers
# MAGIC * Final score and recommendations
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Ready? Run the next cell to start! 👇**

# COMMAND ----------

# DBTITLE 1,⚠️ When May Residuals NOT Shrink?
# MAGIC %md
# MAGIC # ⚠️ When May Residuals NOT Shrink?
# MAGIC
# MAGIC **Common misconception:** Low α (learning rate) prevents residuals from shrinking.
# MAGIC
# MAGIC **Truth:** Low α still shrinks residuals, just **slower**. Eventually converges!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚨 5 Real Reasons Residuals Don't Shrink
# MAGIC
# MAGIC ### **1️⃣ Underfitting Tree (Too Shallow)**
# MAGIC
# MAGIC **Problem:** Tree can't capture the pattern in residuals.
# MAGIC
# MAGIC **Example:**
# MAGIC ```python
# MAGIC # Data has complex pattern
# MAGIC X = [[1], [2], [3], [4], [5], [6], [7], [8]]
# MAGIC y = [1, 4, 9, 16, 25, 36, 49, 64]  # Quadratic: y = x²
# MAGIC
# MAGIC # Tree with max_depth=1 (stump)
# MAGIC # Can only make 1 split → Too simple!
# MAGIC # Cannot learn quadratic pattern
# MAGIC
# MAGIC Residuals stay large:
# MAGIC   Iteration 1: MSE = 450
# MAGIC   Iteration 2: MSE = 445  ← barely improving!
# MAGIC   Iteration 10: MSE = 430 ← stuck!
# MAGIC ```
# MAGIC
# MAGIC **Why it fails:**
# MAGIC - Depth=1 can only learn linear patterns
# MAGIC - Quadratic needs deeper trees
# MAGIC - Tree underfits the residuals themselves!
# MAGIC
# MAGIC **Fix:** Increase `max_depth` to 3-5
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **2️⃣ Noisy Labels**
# MAGIC
# MAGIC **Problem:** True residuals are random noise, unpredictable.
# MAGIC
# MAGIC **Example:**
# MAGIC ```python
# MAGIC # Clean pattern
# MAGIC y_true = 2*X + 10
# MAGIC
# MAGIC # Add heavy noise
# MAGIC noise = np.random.randn(n) * 50  # Large noise!
# MAGIC y_noisy = y_true + noise
# MAGIC
# MAGIC # Training:
# MAGIC F₀ = mean(y_noisy) = 10
# MAGIC Residuals = y_noisy - F₀
# MAGIC
# MAGIC # Tree tries to learn noise → overfits!
# MAGIC # But noise is random, so next iteration:
# MAGIC Residuals = [random, random, random...]
# MAGIC
# MAGIC MSE doesn't decrease:
# MAGIC   Iteration 1: MSE = 2500
# MAGIC   Iteration 5: MSE = 2480  ← minimal change
# MAGIC   Iteration 20: MSE = 2450 ← stuck at noise level!
# MAGIC ```
# MAGIC
# MAGIC **Why it fails:**
# MAGIC - Model learns noise (no real pattern)
# MAGIC - Noise is irreducible
# MAGIC - Adding more trees just memorizes noise
# MAGIC
# MAGIC **Fix:** 
# MAGIC - Clean data
# MAGIC - Lower α (regularization)
# MAGIC - Early stopping
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **3️⃣ Wrong max_depth**
# MAGIC
# MAGIC **Two extremes:**
# MAGIC
# MAGIC #### **A) Too Shallow (depth=1)**
# MAGIC ```python
# MAGIC Data: Non-linear relationship
# MAGIC
# MAGIC Depth=1 tree:
# MAGIC   x ≤ 5 → predict -10
# MAGIC   x > 5 → predict +10
# MAGIC   
# MAGIC   Cannot capture complex patterns!
# MAGIC   
# MAGIC Result: Residuals remain large
# MAGIC ```
# MAGIC
# MAGIC #### **B) Too Deep (depth=15)**
# MAGIC ```python
# MAGIC Data: 100 samples
# MAGIC
# MAGIC Depth=15 tree:
# MAGIC   Creates 2^15 = 32,768 leaf nodes!
# MAGIC   But only 100 samples!
# MAGIC   
# MAGIC   Overfits training data
# MAGIC   Residuals on training → 0
# MAGIC   But test residuals → huge!
# MAGIC   
# MAGIC Result: Overfit, poor generalization
# MAGIC ```
# MAGIC
# MAGIC **Sweet spot:** depth = 3-5 for most cases
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **4️⃣ Insufficient Trees (Stopped Too Early)**
# MAGIC
# MAGIC **Problem:** Stopped training before convergence.
# MAGIC
# MAGIC **Example:**
# MAGIC ```python
# MAGIC # Data needs 50 iterations to converge
# MAGIC y = complex_function(X)
# MAGIC
# MAGIC # But only train 10 trees:
# MAGIC M = 10  # Too few!
# MAGIC
# MAGIC Result:
# MAGIC   Iteration 1: MSE = 1000
# MAGIC   Iteration 5: MSE = 600
# MAGIC   Iteration 10: MSE = 400  ← stopped here!
# MAGIC   
# MAGIC   If continued:
# MAGIC   Iteration 20: MSE = 150
# MAGIC   Iteration 50: MSE = 20   ← could reach here!
# MAGIC ```
# MAGIC
# MAGIC **Why it fails:**
# MAGIC - Each tree makes small correction (α × h_m)
# MAGIC - Need enough iterations to accumulate corrections
# MAGIC - 10 trees not enough for complex data
# MAGIC
# MAGIC **Fix:** Increase number of trees (M = 50-100)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **5️⃣ Optimization Stuck (Local Minimum)**
# MAGIC
# MAGIC **Problem:** Greedy tree splits get stuck in bad solution.
# MAGIC
# MAGIC **Example:**
# MAGIC ```python
# MAGIC # Data has XOR pattern
# MAGIC X = [[0,0], [0,1], [1,0], [1,1]]
# MAGIC y = [0,     1,     1,     0   ]  # XOR!
# MAGIC
# MAGIC # Greedy decision tree:
# MAGIC Split on X₁:
# MAGIC   Left (X₁≤0.5): y = [0, 1]  → mean = 0.5
# MAGIC   Right (X₁>0.5): y = [1, 0] → mean = 0.5
# MAGIC   
# MAGIC Information gain = 0! No improvement!
# MAGIC
# MAGIC # Tree cannot find good split
# MAGIC # Residuals stay constant
# MAGIC
# MAGIC MSE stuck:
# MAGIC   Iteration 1: MSE = 0.25
# MAGIC   Iteration 10: MSE = 0.25  ← no change!
# MAGIC   Iteration 50: MSE = 0.25  ← still stuck!
# MAGIC ```
# MAGIC
# MAGIC **Why it fails:**
# MAGIC - Decision trees use greedy splits (myopic)
# MAGIC - Cannot see multi-step interactions
# MAGIC - Gets stuck in local minimum
# MAGIC
# MAGIC **Fix:**
# MAGIC - Feature engineering (add X₁ × X₂)
# MAGIC - Use deeper trees
# MAGIC - Try different algorithms (neural networks)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Summary Table
# MAGIC
# MAGIC | **Issue** | **Symptom** | **MSE Pattern** | **Fix** |
# MAGIC |-----------|-------------|-----------------|----------|
# MAGIC | **Underfitting tree** | Tree too simple | MSE plateaus high | ↑ max_depth |
# MAGIC | **Noisy labels** | Learning noise | MSE stuck at noise level | Clean data, ↓ α |
# MAGIC | **Wrong depth** | Too shallow/deep | Poor fit or overfit | depth = 3-5 |
# MAGIC | **Insufficient trees** | Stopped early | MSE still decreasing | ↑ n_estimators |
# MAGIC | **Stuck optimization** | Greedy fails | MSE flat from start | Feature engineering |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ What DOES Work?
# MAGIC
# MAGIC **Low α (learning rate):**
# MAGIC ```python
# MAGIC α = 0.01  # Very low
# MAGIC
# MAGIC Iteration 1: MSE = 1000
# MAGIC Iteration 10: MSE = 950   ← slow but decreasing!
# MAGIC Iteration 50: MSE = 700   ← still improving!
# MAGIC Iteration 200: MSE = 100  ← eventually converges!
# MAGIC ```
# MAGIC
# MAGIC **Low α is GOOD:**
# MAGIC - ✅ Still shrinks residuals
# MAGIC - ✅ More stable
# MAGIC - ✅ Better generalization
# MAGIC - ⚠️ Just needs more trees
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Next cell shows concrete examples with code! 👇**

# COMMAND ----------

# ═══════════════════════════════════════════════════════════════════════════════
# 🔬 DEMO: WHEN RESIDUALS DON'T SHRINK
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

print("═" * 80)
print("⚠️  DEMONSTRATION: When Residuals DON'T Shrink in Gradient Boosting")
print("═" * 80)

np.random.seed(42)

# Generate data
n = 100
X = np.linspace(0, 10, n).reshape(-1, 1)
y_clean = np.sin(X.flatten()) * 5 + 10  # Clean sinusoidal pattern

# COMMAND ----------

# DBTITLE 1,🔬 Demo: When Residuals DON'T Shrink


print("\n📊 Generated data: n=100, y = 5*sin(x) + 10")
print("\nTesting 5 scenarios...\n")

# Store results
scenarios = []

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 80)
print("1️⃣  SCENARIO 1: UNDERFITTING TREE (max_depth=1)")
print("═" * 80)

y = y_clean.copy()

# Train GB with very shallow trees
F = np.full(n, y.mean())
mse_history_1 = [np.mean((y - F)**2)]

for m in range(1, 21):
    residuals = y - F
    tree = DecisionTreeRegressor(max_depth=1, random_state=m)  # TOO SHALLOW!
    tree.fit(X, residuals)
    h = tree.predict(X)
    F = F + 0.1 * h
    mse_history_1.append(np.mean((y - F)**2))

print(f"\n   max_depth = 1 (stump - only 1 split)")
print(f"   Initial MSE: {mse_history_1[0]:.2f}")
print(f"   After 5 iterations: {mse_history_1[5]:.2f}")
print(f"   After 10 iterations: {mse_history_1[10]:.2f}")
print(f"   After 20 iterations: {mse_history_1[20]:.2f}")
improvement_1 = (mse_history_1[0] - mse_history_1[20]) / mse_history_1[0] * 100
print(f"\n   ❌ Improvement: {improvement_1:.1f}% (VERY POOR!)")
print(f"   Problem: Tree too simple to learn sinusoidal pattern")

scenarios.append(('Underfitting (depth=1)', mse_history_1, 'red'))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 80)
print("2️⃣  SCENARIO 2: NOISY LABELS")
print("═" * 80)

# Add heavy noise
noise = np.random.randn(n) * 3  # Heavy noise!
y_noisy = y_clean + noise

F = np.full(n, y_noisy.mean())
mse_history_2 = [np.mean((y_noisy - F)**2)]

for m in range(1, 21):
    residuals = y_noisy - F
    tree = DecisionTreeRegressor(max_depth=3, random_state=m)
    tree.fit(X, residuals)
    h = tree.predict(X)
    F = F + 0.1 * h
    mse_history_2.append(np.mean((y_noisy - F)**2))

print(f"\n   Added noise: std = 3.0")
print(f"   Initial MSE: {mse_history_2[0]:.2f}")
print(f"   After 5 iterations: {mse_history_2[5]:.2f}")
print(f"   After 10 iterations: {mse_history_2[10]:.2f}")
print(f"   After 20 iterations: {mse_history_2[20]:.2f}")
improvement_2 = (mse_history_2[0] - mse_history_2[20]) / mse_history_2[0] * 100
print(f"\n   ⚠️  Improvement: {improvement_2:.1f}% (Stuck at noise level!)")
print(f"   Problem: Trying to learn irreducible noise")

scenarios.append(('Noisy Labels', mse_history_2, 'orange'))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 80)
print("3️⃣  SCENARIO 3: INSUFFICIENT TREES (stopped early)")
print("═" * 80)

y = y_clean.copy()

F = np.full(n, y.mean())
mse_history_3 = [np.mean((y - F)**2)]

# Only train 5 trees (too few!)
for m in range(1, 6):  # STOPPED TOO EARLY!
    residuals = y - F
    tree = DecisionTreeRegressor(max_depth=3, random_state=m)
    tree.fit(X, residuals)
    h = tree.predict(X)
    F = F + 0.1 * h
    mse_history_3.append(np.mean((y - F)**2))

# Pad with final value to show it stayed there
mse_history_3.extend([mse_history_3[-1]] * 15)

print(f"\n   Only trained 5 trees (then stopped)")
print(f"   Initial MSE: {mse_history_3[0]:.2f}")
print(f"   After 5 iterations: {mse_history_3[5]:.2f}")
print(f"   Remaining iterations: {mse_history_3[10]:.2f} (no change)")
improvement_3 = (mse_history_3[0] - mse_history_3[5]) / mse_history_3[0] * 100
print(f"\n   ⚠️  Improvement: {improvement_3:.1f}% (Could improve more!)")
print(f"   Problem: Stopped before convergence")

scenarios.append(('Insufficient Trees (M=5)', mse_history_3, 'purple'))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 80)
print("4️⃣  SCENARIO 4: GOOD CONFIGURATION (baseline)")
print("═" * 80)

y = y_clean.copy()

F = np.full(n, y.mean())
mse_history_4 = [np.mean((y - F)**2)]

for m in range(1, 21):
    residuals = y - F
    tree = DecisionTreeRegressor(max_depth=3, random_state=m)  # GOOD DEPTH
    tree.fit(X, residuals)
    h = tree.predict(X)
    F = F + 0.1 * h  # GOOD ALPHA
    mse_history_4.append(np.mean((y - F)**2))

print(f"\n   max_depth = 3, alpha = 0.1, M = 20")
print(f"   Initial MSE: {mse_history_4[0]:.2f}")
print(f"   After 5 iterations: {mse_history_4[5]:.2f}")
print(f"   After 10 iterations: {mse_history_4[10]:.2f}")
print(f"   After 20 iterations: {mse_history_4[20]:.2f}")
improvement_4 = (mse_history_4[0] - mse_history_4[20]) / mse_history_4[0] * 100
print(f"\n   ✅ Improvement: {improvement_4:.1f}% (EXCELLENT!)")
print(f"   This is how it SHOULD work!")

scenarios.append(('Good Config ✅', mse_history_4, 'green'))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 80)
print("5️⃣  SCENARIO 5: LOW ALPHA (still works, just slower)")
print("═" * 80)

y = y_clean.copy()

F = np.full(n, y.mean())
mse_history_5 = [np.mean((y - F)**2)]

for m in range(1, 21):
    residuals = y - F
    tree = DecisionTreeRegressor(max_depth=3, random_state=m)
    tree.fit(X, residuals)
    h = tree.predict(X)
    F = F + 0.01 * h  # VERY LOW ALPHA!
    mse_history_5.append(np.mean((y - F)**2))

print(f"\n   max_depth = 3, alpha = 0.01 (very low!), M = 20")
print(f"   Initial MSE: {mse_history_5[0]:.2f}")
print(f"   After 5 iterations: {mse_history_5[5]:.2f}")
print(f"   After 10 iterations: {mse_history_5[10]:.2f}")
print(f"   After 20 iterations: {mse_history_5[20]:.2f}")
improvement_5 = (mse_history_5[0] - mse_history_5[20]) / mse_history_5[0] * 100
print(f"\n   ✅ Improvement: {improvement_5:.1f}% (STILL GOOD!)")
print(f"   Just slower - would converge with more iterations")

scenarios.append(('Low Alpha (α=0.01)', mse_history_5, 'blue'))

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("📊 VISUALIZING ALL SCENARIOS")
print("═" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Plot each scenario
for idx, (name, mse_hist, color) in enumerate(scenarios):
    ax = axes[idx]
    iterations = range(len(mse_hist))
    ax.plot(iterations, mse_hist, 'o-', color=color, linewidth=2.5, markersize=6)
    ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('MSE', fontsize=11, fontweight='bold')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Annotate start and end
    ax.annotate(f'{mse_hist[0]:.1f}', xy=(0, mse_hist[0]), 
               xytext=(2, mse_hist[0]*1.05),
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    ax.annotate(f'{mse_hist[-1]:.1f}', xy=(len(mse_hist)-1, mse_hist[-1]), 
               xytext=(len(mse_hist)-5, mse_hist[-1]*1.05),
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Comparison plot (all together)
ax = axes[5]
for name, mse_hist, color in scenarios:
    iterations = range(len(mse_hist))
    ax.plot(iterations, mse_hist, '-', color=color, linewidth=2, label=name, alpha=0.8)

ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
ax.set_ylabel('MSE', fontsize=11, fontweight='bold')
ax.set_title('All Scenarios Compared', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.3)

plt.suptitle('⚠️  When Residuals DON\'T Shrink: 5 Scenarios', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("📊 SUMMARY: FINAL IMPROVEMENTS")
print("═" * 80)

for name, mse_hist, color in scenarios:
    improvement = (mse_hist[0] - mse_hist[-1]) / mse_hist[0] * 100
    
    if improvement > 80:
        verdict = "✅ EXCELLENT"
    elif improvement > 50:
        verdict = "⚠️  MODERATE"
    else:
        verdict = "❌ POOR"
    
    print(f"\n   {name:<30} {improvement:>6.1f}%  {verdict}")

print("\n" + "═" * 80)
print("\n🎯 KEY TAKEAWAYS:")
print("\n   ❌ Residuals DON'T shrink when:")
print("      • Tree underfits (too shallow)")
print("      • Data too noisy")
print("      • Stopped training too early")
print("\n   ✅ Low alpha STILL WORKS:")
print("      • Just slower convergence")
print("      • More stable")
print("      • Better generalization")
print("\n   🎓 Best Practice:")
print("      • max_depth = 3-5")
print("      • alpha = 0.1")
print("      • Train enough iterations (monitor validation loss)")
print("\n" + "═" * 80)

# COMMAND ----------

# DBTITLE 1,🎯 Interview Trap: Feature Importance Bias
# MAGIC %md
# MAGIC # 🎯 The Feature Importance Trap (Famous Interview Question)
# MAGIC
# MAGIC **Common Answer (WRONG ❌):**
# MAGIC > "RF randomly picks features, so importance is balanced."
# MAGIC
# MAGIC **Real Issue:**
# MAGIC > **Gini Importance (MDI) has bias toward high-cardinality variables.**
# MAGIC > **Even pure NOISE can appear important!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚨 The Problem: Mean Decrease Impurity (MDI) Bias
# MAGIC
# MAGIC ### **What is Gini Importance / MDI?**
# MAGIC
# MAGIC ```python
# MAGIC For each feature:
# MAGIC   Importance = Σ (weighted Gini decrease when feature used in split)
# MAGIC   
# MAGIC Example:
# MAGIC   Feature "Age" used in 5 nodes
# MAGIC   Each split reduces Gini by [0.02, 0.01, 0.03, 0.01, 0.02]
# MAGIC   Importance = 0.02 + 0.01 + 0.03 + 0.01 + 0.02 = 0.09
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚠️ The Bias: Why Noise Gets High Importance
# MAGIC
# MAGIC ### **Bias Toward:**
# MAGIC
# MAGIC 1. **Continuous variables** (vs categorical)
# MAGIC 2. **High-cardinality variables** (many unique values)
# MAGIC 3. **Variables with many split opportunities**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Why This Happens:**
# MAGIC
# MAGIC **Scenario: Pure Noise Feature**
# MAGIC
# MAGIC ```python
# MAGIC # Real feature: Gender (2 values: M, F)
# MAGIC Gender = ['M', 'F', 'M', 'F', 'M', 'F', ...]
# MAGIC Unique values: 2
# MAGIC
# MAGIC # Noise feature: Random ID (100 unique values)
# MAGIC Random_ID = [78, 12, 94, 3, 55, 81, ...]
# MAGIC Unique values: 100 ← HIGH CARDINALITY!
# MAGIC ```
# MAGIC
# MAGIC **What happens during tree building:**
# MAGIC
# MAGIC ```
# MAGIC Tree considers splits:
# MAGIC
# MAGIC Gender splits:
# MAGIC   - Split 1: Gender = M vs F
# MAGIC   - Split 2: (no other options)
# MAGIC   Total: 1 possible split
# MAGIC   
# MAGIC Random_ID splits:
# MAGIC   - Split 1: ID ≤ 12
# MAGIC   - Split 2: ID ≤ 23
# MAGIC   - Split 3: ID ≤ 34
# MAGIC   ...
# MAGIC   - Split 99: ID ≤ 99
# MAGIC   Total: 99 possible splits! ← 99x more opportunities!
# MAGIC ```
# MAGIC
# MAGIC **Result:**
# MAGIC
# MAGIC ```
# MAGIC Even though Random_ID is PURE NOISE:
# MAGIC   
# MAGIC   • It gets used in many splits (accidental!)
# MAGIC   • Each split reduces Gini slightly (by chance)
# MAGIC   • Total importance = 99 splits × small_decrease
# MAGIC   
# MAGIC Gender importance: 1 split × 0.15 = 0.15
# MAGIC Random_ID importance: 99 splits × 0.002 = 0.20 ← HIGHER!
# MAGIC
# MAGIC ❌ TRAP: Noise appears more important than real signal!
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔬 Mathematical Explanation
# MAGIC
# MAGIC ### **Why Continuous/High-Cardinality Features Get Inflated Importance:**
# MAGIC
# MAGIC **Formula for Gini Importance:**
# MAGIC
# MAGIC ```
# MAGIC Importance(feature) = Σ [n_samples_node/n_total × (Gini_before - Gini_after)]
# MAGIC                       over all nodes where feature is used
# MAGIC ```
# MAGIC
# MAGIC **The Trap:**
# MAGIC
# MAGIC ```
# MAGIC High-cardinality feature:
# MAGIC   • Can be used in MANY nodes (more split points)
# MAGIC   • Each split reduces Gini slightly (even if random)
# MAGIC   • Accumulates importance across many nodes
# MAGIC   
# MAGIC Low-cardinality feature:
# MAGIC   • Fewer split opportunities
# MAGIC   • Even if truly predictive, accumulates less
# MAGIC ```
# MAGIC
# MAGIC **Example Numbers:**
# MAGIC
# MAGIC ```
# MAGIC Real Feature (3 categories):
# MAGIC   Used in 5 nodes
# MAGIC   Average Gini decrease: 0.08 (truly predictive!)
# MAGIC   Importance: 5 × 0.08 = 0.40
# MAGIC   
# MAGIC Noise Feature (100 unique values):
# MAGIC   Used in 50 nodes (10x more!)
# MAGIC   Average Gini decrease: 0.01 (accidental, random)
# MAGIC   Importance: 50 × 0.01 = 0.50 ← HIGHER!
# MAGIC   
# MAGIC ❌ Noise ranked above real signal!
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ The Solution: Permutation Importance
# MAGIC
# MAGIC ### **Concept:**
# MAGIC
# MAGIC ```
# MAGIC For each feature:
# MAGIC   1. Train model on original data
# MAGIC   2. Measure baseline performance (e.g., accuracy = 0.90)
# MAGIC   
# MAGIC   3. Shuffle the feature (break its relationship with target)
# MAGIC   4. Measure new performance (e.g., accuracy = ?)
# MAGIC   
# MAGIC   5. Importance = Baseline - After_Shuffle
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **How It Works:**
# MAGIC
# MAGIC **Real Feature (Gender):**
# MAGIC
# MAGIC ```
# MAGIC Original data:
# MAGIC   Gender → Outcome has real relationship
# MAGIC   Performance: Accuracy = 0.90
# MAGIC   
# MAGIC After shuffling Gender:
# MAGIC   Gender → Outcome relationship destroyed
# MAGIC   Performance: Accuracy = 0.75 ← drops!
# MAGIC   
# MAGIC Permutation Importance = 0.90 - 0.75 = 0.15 ✅ Important!
# MAGIC ```
# MAGIC
# MAGIC **Noise Feature (Random_ID):**
# MAGIC
# MAGIC ```
# MAGIC Original data:
# MAGIC   Random_ID → Outcome has NO relationship
# MAGIC   Performance: Accuracy = 0.90
# MAGIC   
# MAGIC After shuffling Random_ID:
# MAGIC   Random_ID → Still no relationship (was random anyway!)
# MAGIC   Performance: Accuracy = 0.90 ← NO CHANGE!
# MAGIC   
# MAGIC Permutation Importance = 0.90 - 0.90 = 0.00 ✅ Useless!
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Comparison Table
# MAGIC
# MAGIC | Feature | Type | Unique Values | **Gini Importance (MDI)** | **Permutation Importance** |
# MAGIC |---------|------|---------------|---------------------------|----------------------------|
# MAGIC | Age | Continuous | 80 | 0.35 ❌ (inflated) | 0.15 ✅ (true) |
# MAGIC | Gender | Categorical | 2 | 0.10 ❌ (deflated) | 0.25 ✅ (true) |
# MAGIC | Random_Noise | Continuous | 100 | **0.45** ❌ (TRAP!) | **0.00** ✅ (correct) |
# MAGIC | Income | Continuous | 95 | 0.40 ❌ (inflated) | 0.30 ✅ (true) |
# MAGIC | City | Categorical | 5 | 0.15 ❌ (deflated) | 0.20 ✅ (true) |
# MAGIC
# MAGIC **Key Insight:** Gini Importance ranks pure noise (Random_Noise) as MOST important! ❌
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Interview Answer Template
# MAGIC
# MAGIC **Question:** "How do you interpret feature importance in Random Forest?"
# MAGIC
# MAGIC **Perfect Answer:**
# MAGIC
# MAGIC ```
# MAGIC ✅ "I'd use Permutation Importance, not Gini Importance.
# MAGIC
# MAGIC    Why? Gini Importance (MDI) has a well-known bias:
# MAGIC    
# MAGIC    • Inflates importance of continuous variables
# MAGIC    • Inflates importance of high-cardinality variables
# MAGIC    • Even pure noise can appear important!
# MAGIC    
# MAGIC    This happens because high-cardinality features get
# MAGIC    more split opportunities, accumulating importance
# MAGIC    even if splits are accidental.
# MAGIC    
# MAGIC    Permutation Importance fixes this by:
# MAGIC    
# MAGIC    • Shuffling each feature
# MAGIC    • Measuring actual performance drop
# MAGIC    • If shuffling noise → no drop → importance = 0 ✅
# MAGIC    
# MAGIC    Libraries: sklearn.inspection.permutation_importance
# MAGIC    or eli5.show_weights for better interpretation."
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔑 Key Takeaways
# MAGIC
# MAGIC ### **❌ Don't Use (Alone):**
# MAGIC - **Gini Importance / MDI**
# MAGIC - **Mean Decrease Impurity**
# MAGIC - `feature_importances_` from sklearn
# MAGIC
# MAGIC ### **✅ Use Instead:**
# MAGIC - **Permutation Importance**
# MAGIC - `sklearn.inspection.permutation_importance()`
# MAGIC - **SHAP values** (even better!)
# MAGIC - **Partial Dependence Plots**
# MAGIC
# MAGIC ### **Why This Is Famous:**
# MAGIC - Catches 80% of candidates
# MAGIC - Tests deep ML understanding
# MAGIC - Real production issue (not just academic)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Next cell shows concrete code example! 👇**

# COMMAND ----------

# ═══════════════════════════════════════════════════════════════════════════════
# 🔬 DEMO: THE FEATURE IMPORTANCE TRAP
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

print("═" * 80)
print("🎯 THE FEATURE IMPORTANCE TRAP: Gini vs Permutation")
print("═" * 80)

np.random.seed(42)

# COMMAND ----------



# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: CREATE DATASET WITH REAL FEATURES + NOISE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n📊 STEP 1: Creating dataset...\n")

n_samples = 1000

# Real predictive features
age = np.random.randint(18, 80, n_samples)
gender = np.random.choice(['M', 'F'], n_samples)  # 2 unique values (low cardinality)
income_bracket = np.random.choice(['Low', 'Medium', 'High'], n_samples)  # 3 unique values

# COMMAND ----------

income_bracket 

# COMMAND ----------

gender 

# COMMAND ----------

age

# COMMAND ----------


# Target: Based on real features
# Outcome depends on: age > 50 AND income = 'High'
y = ((age > 50) & (income_bracket == 'High')).astype(int)

print("   ✅ Real Features:")
print(f"      • age: continuous, {len(np.unique(age))} unique values")
print(f"      • gender: categorical, {len(np.unique(gender))} unique values")
print(f"      • income_bracket: categorical, {len(np.unique(income_bracket))} unique values")
print(f"\n   🎯 Target: Based on age > 50 AND income = 'High'")
print(f"      Positive class: {y.sum()} samples ({y.sum()/len(y)*100:.1f}%)")

# COMMAND ----------




# Add PURE NOISE features with different cardinalities
print("\n   ❌ Adding PURE NOISE features:")

# Noise 1: Random continuous (HIGH cardinality)
noise_continuous = np.random.rand(n_samples) * 100  # 1000 unique values
print(f"      • noise_continuous: {len(np.unique(noise_continuous))} unique values (HIGH!)")

# Noise 2: Random ID (HIGH cardinality)
noise_id = np.random.randint(1, 500, n_samples)  # ~500 unique values
print(f"      • noise_id: {len(np.unique(noise_id))} unique values (HIGH!)")

# Noise 3: Random category (LOW cardinality)
noise_category = np.random.choice(['A', 'B'], n_samples)  # 2 unique values
print(f"      • noise_category: {len(np.unique(noise_category))} unique values (LOW)")

print("\n   ⚠️  All noise features have ZERO relationship with target!")

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'income_bracket': income_bracket,
    'noise_continuous': noise_continuous,
    'noise_id': noise_id,
    'noise_category': noise_category,
    'target': y
})

# Encode categorical variables
df_encoded = pd.get_dummies(df.drop('target', axis=1), drop_first=True)
X = df_encoded

print(f"\n   📋 Final dataset shape: {X.shape}")
print(f"      Features: {list(X.columns)}")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: TRAIN RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("\n🌲 STEP 2: Training Random Forest...\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

train_acc = rf.score(X_train, y_train)
test_acc = rf.score(X_test, y_test)

print(f"   Training Accuracy: {train_acc:.3f}")
print(f"   Test Accuracy: {test_acc:.3f}")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: GINI IMPORTANCE (THE TRAP!)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("\n❌ STEP 3: Gini Importance (Mean Decrease Impurity)\n")

gini_importance = rf.feature_importances_
feature_names = X.columns

# Sort by importance
gini_ranking = pd.DataFrame({
    'feature': feature_names,
    'gini_importance': gini_importance
}).sort_values('gini_importance', ascending=False)

print("   📊 Feature Ranking (Gini Importance):")
print("")
for idx, row in gini_ranking.iterrows():
    feature = row['feature']
    importance = row['gini_importance']
    
    # Mark noise features
    if 'noise' in feature:
        marker = "❌ NOISE"
    else:
        marker = "✅ REAL"
    
    print(f"      {feature:25s}  {importance:.4f}  {marker}")

# Count noise in top 3
top_3 = gini_ranking.head(3)['feature'].values
noise_in_top_3 = sum(['noise' in f for f in top_3])

print(f"\n   ⚠️  PROBLEM: {noise_in_top_3} NOISE features in top 3!")
if noise_in_top_3 > 0:
    print(f"       Pure noise ranked as important! This is THE TRAP! 🎯")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: PERMUTATION IMPORTANCE (THE FIX!)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("\n✅ STEP 4: Permutation Importance (Correct Method)\n")

print("   Computing permutation importance (shuffling each feature)...")

perm_importance = permutation_importance(
    rf, X_test, y_test, 
    n_repeats=10, 
    random_state=42,
    n_jobs=-1
)

perm_ranking = pd.DataFrame({
    'feature': feature_names,
    'perm_importance': perm_importance.importances_mean,
    'perm_std': perm_importance.importances_std
}).sort_values('perm_importance', ascending=False)

print("\n   📊 Feature Ranking (Permutation Importance):")
print("")
for idx, row in perm_ranking.iterrows():
    feature = row['feature']
    importance = row['perm_importance']
    std = row['perm_std']
    
    # Mark noise features
    if 'noise' in feature:
        marker = "❌ NOISE"
    else:
        marker = "✅ REAL"
    
    print(f"      {feature:25s}  {importance:.4f} ± {std:.4f}  {marker}")

# Count noise in top 3
top_3_perm = perm_ranking.head(3)['feature'].values
noise_in_top_3_perm = sum(['noise' in f for f in top_3_perm])

print(f"\n   ✅ FIXED: {noise_in_top_3_perm} NOISE features in top 3!")
if noise_in_top_3_perm == 0:
    print(f"       Permutation importance correctly identified noise! 🎯")

# COMMAND ----------

# DBTITLE 1,🔬 Demo: Feature Importance Trap


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: VISUAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("\n📊 STEP 5: Visualizing the difference...\n")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# LEFT: Gini Importance
ax = axes[0]
y_pos = np.arange(len(gini_ranking))
colors = ['red' if 'noise' in f else 'green' for f in gini_ranking['feature']]

ax.barh(y_pos, gini_ranking['gini_importance'], color=colors, alpha=0.7, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(gini_ranking['feature'], fontsize=10)
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('❌ Gini Importance (MDI)\n(BIASED - Noise Ranks High!)', 
            fontsize=13, fontweight='bold', color='red')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.7, edgecolor='black', label='Real Features'),
    Patch(facecolor='red', alpha=0.7, edgecolor='black', label='Noise Features')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

# RIGHT: Permutation Importance
ax = axes[1]
y_pos = np.arange(len(perm_ranking))
colors = ['red' if 'noise' in f else 'green' for f in perm_ranking['feature']]

ax.barh(y_pos, perm_ranking['perm_importance'], 
       xerr=perm_ranking['perm_std'],
       color=colors, alpha=0.7, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(perm_ranking['feature'], fontsize=10)
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('✅ Permutation Importance\n(CORRECT - Noise Near Zero!)', 
            fontsize=13, fontweight='bold', color='green')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Add legend
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.suptitle('🎯 The Feature Importance Trap: Gini vs Permutation', 
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: DETAILED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("📊 DETAILED COMPARISON")
print("═" * 80)

comparison = pd.merge(
    gini_ranking[['feature', 'gini_importance']], 
    perm_ranking[['feature', 'perm_importance']], 
    on='feature'
)

print("\n   Feature Analysis:\n")
for idx, row in comparison.iterrows():
    feature = row['feature']
    gini = row['gini_importance']
    perm = row['perm_importance']
    
    is_noise = 'noise' in feature
    
    print(f"   {feature:25s}")
    print(f"      Gini: {gini:.4f}")
    print(f"      Perm: {perm:.4f}")
    
    if is_noise:
        if gini > 0.1 and perm < 0.01:
            print(f"      ⚠️  TRAP! High Gini but zero Perm (pure noise!)")
        elif perm < 0.01:
            print(f"      ✅ Correctly identified as noise")
    else:
        if perm > 0.05:
            print(f"      ✅ Real predictive power")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("═" * 80)
print("🎯 KEY INSIGHTS")
print("═" * 80)

print("\n   ❌ GINI IMPORTANCE (MDI) PROBLEMS:")
print("      • Biased toward high-cardinality features")
print("      • Biased toward continuous variables")
print(f"      • Ranked {noise_in_top_3} noise features in top 3!")
print("      • Even pure noise appears important")

print("\n   ✅ PERMUTATION IMPORTANCE BENEFITS:")
print("      • Unbiased (no cardinality bias)")
print("      • Measures actual predictive power")
print(f"      • Correctly identified noise (near zero importance)")
print("      • Tests: If shuffle → no performance drop → useless")

print("\n   🎓 INTERVIEW TIP:")
print("      'Always use Permutation Importance for RF feature selection.'")
print("      'Gini Importance has well-known bias toward high-cardinality.'")
print("      'Even pure noise can rank high with Gini - famous ML trap!'")

print("\n" + "═" * 80)

# COMMAND ----------

# DBTITLE 1,📐 Mathematical Foundation: Residual = Negative Gradient
# MAGIC %md
# MAGIC # 📐 Residual = Negative Gradient (The Mathematical Foundation)
# MAGIC
# MAGIC **You knew:** Residuals are related to gradients.
# MAGIC
# MAGIC **You missed:** The exact mathematical derivation.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 The Core Question
# MAGIC
# MAGIC **Why does Gradient Boosting learn residuals?**
# MAGIC
# MAGIC Answer: **Residuals are proportional to the negative gradient of the loss function.**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📝 Full Mathematical Derivation
# MAGIC
# MAGIC ### **Step 1: Define the Loss Function**
# MAGIC
# MAGIC **Mean Squared Error (MSE) Loss:**
# MAGIC
# MAGIC ```
# MAGIC L = Σ (yᵢ - F(xᵢ))²
# MAGIC     i=1 to N
# MAGIC
# MAGIC Where:
# MAGIC   yᵢ = actual target
# MAGIC   F(xᵢ) = current model prediction
# MAGIC   N = number of samples
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Step 2: Take Derivative w.r.t. Prediction**
# MAGIC
# MAGIC **Derivative of Loss w.r.t. F(xᵢ):**
# MAGIC
# MAGIC ```
# MAGIC ∂L/∂F(xᵢ) = ∂/∂F(xᵢ) [(yᵢ - F(xᵢ))²]
# MAGIC ```
# MAGIC
# MAGIC **Apply chain rule:**
# MAGIC
# MAGIC ```
# MAGIC Let u = yᵢ - F(xᵢ)
# MAGIC
# MAGIC Then: L = u²
# MAGIC
# MAGIC ∂L/∂F(xᵢ) = ∂L/∂u × ∂u/∂F(xᵢ)
# MAGIC            = 2u × (-1)
# MAGIC            = 2(yᵢ - F(xᵢ)) × (-1)
# MAGIC            = -2(yᵢ - F(xᵢ))
# MAGIC ```
# MAGIC
# MAGIC **Result:**
# MAGIC ```
# MAGIC ∂L/∂F(xᵢ) = -2(yᵢ - F(xᵢ))
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Step 3: Compute Negative Gradient**
# MAGIC
# MAGIC **Gradient Descent Rule:**
# MAGIC
# MAGIC To minimize loss, move in direction of **negative gradient**:
# MAGIC
# MAGIC ```
# MAGIC -∂L/∂F(xᵢ) = -[-2(yᵢ - F(xᵢ))]
# MAGIC             = 2(yᵢ - F(xᵢ))
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Step 4: Define Residual**
# MAGIC
# MAGIC **Residual:**
# MAGIC
# MAGIC ```
# MAGIC rᵢ = yᵢ - F(xᵢ)
# MAGIC    = (actual - predicted)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Step 5: Connect Residual to Gradient**
# MAGIC
# MAGIC **Comparing:**
# MAGIC
# MAGIC ```
# MAGIC Negative Gradient: -∂L/∂F(xᵢ) = 2(yᵢ - F(xᵢ)) = 2rᵢ
# MAGIC
# MAGIC Residual: rᵢ = yᵢ - F(xᵢ)
# MAGIC ```
# MAGIC
# MAGIC **Therefore:**
# MAGIC
# MAGIC ```
# MAGIC -∇L = 2rᵢ
# MAGIC
# MAGIC OR:
# MAGIC
# MAGIC rᵢ = (1/2) × (-∇L)
# MAGIC ```
# MAGIC
# MAGIC **✅ Residuals are proportional to negative gradients!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Summary Box
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────┐
# MAGIC │  Gradient Boosting Mathematical Foundation              │
# MAGIC ├─────────────────────────────────────────────────────────┤
# MAGIC │                                                         │
# MAGIC │  Loss:             L = Σ (yᵢ - F(xᵢ))²                 │
# MAGIC │                                                         │
# MAGIC │  Gradient:         ∂L/∂F = -2(yᵢ - F(xᵢ))              │
# MAGIC │                                                         │
# MAGIC │  Negative Gradient: -∂L/∂F = 2(yᵢ - F(xᵢ))             │
# MAGIC │                                                         │
# MAGIC │  Residual:         rᵢ = yᵢ - F(xᵢ)                      │
# MAGIC │                                                         │
# MAGIC │  Connection:       -∇L = 2rᵢ                            │
# MAGIC │                                                         │
# MAGIC │  ✅ Learning residuals = gradient descent!              │
# MAGIC │                                                         │
# MAGIC └─────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Why This Matters
# MAGIC
# MAGIC ### **Gradient Boosting = Gradient Descent in Function Space**
# MAGIC
# MAGIC **Regular Gradient Descent:**
# MAGIC ```
# MAGIC Optimize parameters θ:
# MAGIC   θ_new = θ_old - α × ∇L(θ)
# MAGIC ```
# MAGIC
# MAGIC **Gradient Boosting:**
# MAGIC ```
# MAGIC Optimize function F(x):
# MAGIC   F_new(x) = F_old(x) - α × ∇L(F)
# MAGIC             = F_old(x) + α × [-∇L]
# MAGIC             = F_old(x) + α × [2rᵢ]
# MAGIC             ≈ F_old(x) + α × h(x)  ← tree learns residuals
# MAGIC ```
# MAGIC
# MAGIC **Where:**
# MAGIC - Tree `h(x)` approximates the negative gradient (i.e., residuals)
# MAGIC - Each iteration takes a "step" in function space
# MAGIC - Direction: Reduce loss (negative gradient direction)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔢 Concrete Example
# MAGIC
# MAGIC **Data:**
# MAGIC ```
# MAGIC y = [20, 60, 90]
# MAGIC F₀ = [50, 50, 50]  ← initial prediction (mean)
# MAGIC ```
# MAGIC
# MAGIC ### **Step-by-Step:**
# MAGIC
# MAGIC **1. Current Loss:**
# MAGIC ```
# MAGIC L = (20-50)² + (60-50)² + (90-50)²
# MAGIC   = 900 + 100 + 1600
# MAGIC   = 2600
# MAGIC ```
# MAGIC
# MAGIC **2. Gradient (for each sample):**
# MAGIC ```
# MAGIC ∂L/∂F(x₁) = -2(20 - 50) = -2(-30) = 60
# MAGIC ∂L/∂F(x₂) = -2(60 - 50) = -2(10) = -20
# MAGIC ∂L/∂F(x₃) = -2(90 - 50) = -2(40) = -80
# MAGIC
# MAGIC Gradient vector: [60, -20, -80]
# MAGIC ```
# MAGIC
# MAGIC **3. Negative Gradient:**
# MAGIC ```
# MAGIC -∇L = [-60, 20, 80]
# MAGIC ```
# MAGIC
# MAGIC **4. Residuals:**
# MAGIC ```
# MAGIC r₁ = 20 - 50 = -30
# MAGIC r₂ = 60 - 50 = 10
# MAGIC r₃ = 90 - 50 = 40
# MAGIC
# MAGIC Residual vector: [-30, 10, 40]
# MAGIC ```
# MAGIC
# MAGIC **5. Verify Relationship:**
# MAGIC ```
# MAGIC -∇L = 2 × residuals
# MAGIC [-60, 20, 80] = 2 × [-30, 10, 40]  ✅
# MAGIC ```
# MAGIC
# MAGIC **6. Tree Learns:**
# MAGIC ```
# MAGIC Train tree h₁(x) to predict residuals:
# MAGIC   h₁(x₁) ≈ -30
# MAGIC   h₁(x₂) ≈ 10
# MAGIC   h₁(x₃) ≈ 40
# MAGIC ```
# MAGIC
# MAGIC **7. Update (α = 0.3):**
# MAGIC ```
# MAGIC F₁ = F₀ + α × h₁
# MAGIC    = [50, 50, 50] + 0.3 × [-30, 10, 40]
# MAGIC    = [50-9, 50+3, 50+12]
# MAGIC    = [41, 53, 62]
# MAGIC ```
# MAGIC
# MAGIC **8. New Loss:**
# MAGIC ```
# MAGIC L_new = (20-41)² + (60-53)² + (90-62)²
# MAGIC       = 441 + 49 + 784
# MAGIC       = 1274  ← reduced from 2600!
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎓 Interview Answer Template
# MAGIC
# MAGIC **Question:** "Why does Gradient Boosting fit trees to residuals?"
# MAGIC
# MAGIC **Perfect Answer:**
# MAGIC
# MAGIC ```
# MAGIC ✅ "Residuals are proportional to the negative gradient of 
# MAGIC    the loss function.
# MAGIC    
# MAGIC    Mathematical derivation:
# MAGIC    
# MAGIC    1. Loss: L = Σ (yᵢ - F(xᵢ))²
# MAGIC    
# MAGIC    2. Gradient: ∂L/∂F = -2(yᵢ - F(xᵢ))
# MAGIC    
# MAGIC    3. Negative gradient: -∂L/∂F = 2(yᵢ - F(xᵢ))
# MAGIC    
# MAGIC    4. Residual: rᵢ = yᵢ - F(xᵢ)
# MAGIC    
# MAGIC    5. Therefore: -∇L = 2rᵢ
# MAGIC    
# MAGIC    Gradient Boosting performs gradient descent in function
# MAGIC    space. Each tree approximates the negative gradient
# MAGIC    (i.e., residuals), moving the model toward lower loss.
# MAGIC    
# MAGIC    Update rule: F_m = F_{m-1} + α × h_m
# MAGIC    
# MAGIC    Where h_m learns residuals ≈ negative gradient direction."
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔑 Key Takeaways
# MAGIC
# MAGIC ### **Three Equivalent Views:**
# MAGIC
# MAGIC **1. Intuitive View:**
# MAGIC ```
# MAGIC "Fix the mistakes" (residuals)
# MAGIC ```
# MAGIC
# MAGIC **2. Statistical View:**
# MAGIC ```
# MAGIC "Minimize loss iteratively"
# MAGIC ```
# MAGIC
# MAGIC **3. Mathematical View:**
# MAGIC ```
# MAGIC "Gradient descent in function space"
# MAGIC ```
# MAGIC
# MAGIC ### **All Three Are the SAME!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Generalization
# MAGIC
# MAGIC **For other loss functions:**
# MAGIC
# MAGIC **Absolute Error (MAE):**
# MAGIC ```
# MAGIC L = Σ |yᵢ - F(xᵢ)|
# MAGIC
# MAGIC ∂L/∂F = -sign(yᵢ - F(xᵢ))
# MAGIC
# MAGIC Pseudo-residual = sign(yᵢ - F(xᵢ))  ← not actual residual!
# MAGIC ```
# MAGIC
# MAGIC **Log Loss (Classification):**
# MAGIC ```
# MAGIC L = -Σ [yᵢ log(p(xᵢ)) + (1-yᵢ)log(1-p(xᵢ))]
# MAGIC
# MAGIC ∂L/∂F = -(yᵢ - p(xᵢ))
# MAGIC
# MAGIC Pseudo-residual = yᵢ - p(xᵢ)  ← probability residual
# MAGIC ```
# MAGIC
# MAGIC **General Form:**
# MAGIC ```
# MAGIC Pseudo-residual = -∂L/∂F(xᵢ)
# MAGIC
# MAGIC Tree learns: h(x) ≈ -∂L/∂F(x)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Next cell: Interactive demonstration! 👇**

# COMMAND ----------

# DBTITLE 1,🔬 Demo: Residual = Negative Gradient Proof
# ═══════════════════════════════════════════════════════════════════════════════
# 🔬 DEMO: RESIDUAL = NEGATIVE GRADIENT (MATHEMATICAL PROOF)
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("═" * 80)
print("📐 PROOF: Residuals = Negative Gradients")
print("═" * 80)

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: SIMPLE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("PART 1: CONCRETE NUMERICAL EXAMPLE")
print("─" * 80)

# Simple data
y = np.array([20.0, 60.0, 90.0])
F = np.array([50.0, 50.0, 50.0])  # Initial prediction (mean)

print("\n📊 Data:")
print(f"   Actual (y):      {y}")
print(f"   Predicted (F):   {F}")

# Step 1: Calculate Loss
print("\n" + "─" * 80)
print("STEP 1: Calculate Loss")
print("─" * 80)

errors = y - F
print(f"\n   Errors: {errors}")

squared_errors = errors**2
print(f"   Squared errors: {squared_errors}")

loss = np.sum(squared_errors)
print(f"\n   Total Loss (L): {loss}")
print(f"   Formula: L = Σ(yᵢ - F(xᵢ))²")
print(f"          = ({errors[0]})² + ({errors[1]})² + ({errors[2]})²")
print(f"          = {squared_errors[0]} + {squared_errors[1]} + {squared_errors[2]}")
print(f"          = {loss}")

# Step 2: Calculate Gradient
print("\n" + "─" * 80)
print("STEP 2: Calculate Gradient ∂L/∂F")
print("─" * 80)

print("\n   Formula: ∂L/∂F(xᵢ) = ∂/∂F [(yᵢ - F(xᵢ))²]")
print("           = 2(yᵢ - F(xᵢ)) × (-1)")
print("           = -2(yᵢ - F(xᵢ))")

gradient = -2 * errors
print(f"\n   For each sample:")
for i in range(len(y)):
    print(f"      ∂L/∂F(x{i+1}) = -2({errors[i]}) = {gradient[i]}")

print(f"\n   Gradient vector: {gradient}")

# Step 3: Calculate Negative Gradient
print("\n" + "─" * 80)
print("STEP 3: Calculate Negative Gradient -∂L/∂F")
print("─" * 80)

neg_gradient = -gradient
print(f"\n   -∇L = -{gradient}")
print(f"       = {neg_gradient}")

# Step 4: Calculate Residuals
print("\n" + "─" * 80)
print("STEP 4: Calculate Residuals")
print("─" * 80)

residuals = y - F
print(f"\n   Residuals (r): {residuals}")
print(f"   Formula: rᵢ = yᵢ - F(xᵢ)")
for i in range(len(y)):
    print(f"      r{i+1} = {y[i]} - {F[i]} = {residuals[i]}")

# Step 5: Verify Relationship
print("\n" + "─" * 80)
print("STEP 5: Verify Relationship")
print("─" * 80)

print(f"\n   Negative Gradient: {neg_gradient}")
print(f"   2 × Residuals:     {2 * residuals}")

if np.allclose(neg_gradient, 2 * residuals):
    print("\n   ✅ VERIFIED: -∇L = 2 × r")
    print("   ✅ Residuals are proportional to negative gradients!")
else:
    print("\n   ❌ ERROR: Relationship not verified")

print(f"\n   Therefore:")
print(f"      rᵢ = (1/2) × (-∇L)")
print(f"      Learning residuals ≡ Following negative gradient")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: VISUAL PROOF
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("PART 2: VISUAL PROOF WITH GRADIENT DESCENT")
print("═" * 80)

# Generate continuous data for visualization
n_points = 100
X_viz = np.linspace(0, 10, n_points)
y_true_viz = 2*X_viz + 10 + np.random.randn(n_points)*2

# Initial prediction (constant)
F_init = np.full(n_points, y_true_viz.mean())

# Calculate gradients and residuals
residuals_viz = y_true_viz - F_init
gradient_viz = -2 * residuals_viz
neg_gradient_viz = -gradient_viz

print("\n📊 Generated 100 data points for visualization")
print(f"   Mean residual: {residuals_viz.mean():.2f}")
print(f"   Mean negative gradient: {neg_gradient_viz.mean():.2f}")
print(f"   Ratio: {neg_gradient_viz.mean() / residuals_viz.mean():.2f} (should be ~2)")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# TOP LEFT: Loss surface (1D example)
ax = axes[0, 0]

# Single point example for loss surface
y_single = 60
F_range = np.linspace(20, 100, 100)
loss_surface = (y_single - F_range)**2

# Current prediction
F_current = 50
loss_current = (y_single - F_current)**2

# Gradient at current point
grad_current = -2*(y_single - F_current)

ax.plot(F_range, loss_surface, linewidth=2.5, color='blue', label='Loss L = (y-F)²')
ax.scatter([F_current], [loss_current], s=200, c='red', marker='o', 
          edgecolors='black', linewidth=2, zorder=5, label=f'Current F={F_current}')

# Draw gradient arrow
arrow_length = -grad_current * 2  # Scale for visibility
ax.arrow(F_current, loss_current, arrow_length, -100, 
        head_width=3, head_length=50, fc='green', ec='green', linewidth=2, 
        label=f'Gradient = {grad_current:.0f}')

ax.set_xlabel('Prediction F', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss (y-F)²', fontsize=12, fontweight='bold')
ax.set_title(f'Loss Surface (y={y_single})', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3)

# Annotate minimum
ax.axvline(y_single, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal F={y_single}')
ax.text(y_single+2, 100, f'Minimum\nat F={y_single}', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# TOP RIGHT: Residuals vs Negative Gradient
ax = axes[0, 1]

# Scatter plot
ax.scatter(residuals_viz, neg_gradient_viz/2, alpha=0.6, s=50, 
          edgecolors='black', linewidth=0.5)

# Perfect correlation line
min_val = min(residuals_viz.min(), (neg_gradient_viz/2).min())
max_val = max(residuals_viz.max(), (neg_gradient_viz/2).max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, 
       label='Perfect match: r = -∇L/2', alpha=0.8)

ax.set_xlabel('Residuals (r = y - F)', fontsize=12, fontweight='bold')
ax.set_ylabel('Negative Gradient / 2', fontsize=12, fontweight='bold')
ax.set_title('Residuals = Negative Gradient / 2', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(alpha=0.3)

# Add correlation
corr = np.corrcoef(residuals_viz, neg_gradient_viz/2)[0, 1]
ax.text(0.05, 0.95, f'Correlation: {corr:.6f}', transform=ax.transAxes,
       fontsize=11, fontweight='bold', verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# BOTTOM LEFT: Gradient Descent Steps
ax = axes[1, 0]

# Simulate gradient descent
y_target = 60
F_history = [30]  # Start far from target
loss_history = [(y_target - F_history[0])**2]
alpha = 0.1

for _ in range(10):
    F_curr = F_history[-1]
    residual = y_target - F_curr
    gradient = -2 * residual
    neg_grad = -gradient
    
    # Update
    F_new = F_curr + alpha * neg_grad
    F_history.append(F_new)
    loss_history.append((y_target - F_new)**2)

iterations = range(len(F_history))

# Plot predictions
ax.plot(iterations, F_history, 'o-', linewidth=2.5, markersize=8, 
       color='blue', label='Predictions F')
ax.axhline(y_target, color='green', linestyle='--', linewidth=2, 
          label=f'Target y={y_target}', alpha=0.8)

ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax.set_ylabel('Prediction F', fontsize=12, fontweight='bold')
ax.set_title('Gradient Descent: Following Negative Gradient', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# Annotate convergence
ax.annotate(f'Start\nF={F_history[0]}', xy=(0, F_history[0]), 
           xytext=(2, F_history[0]-5),
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
           arrowprops=dict(arrowstyle='->', lw=1.5))

ax.annotate(f'Converged\nF≈{F_history[-1]:.1f}', xy=(len(F_history)-1, F_history[-1]), 
           xytext=(len(F_history)-4, F_history[-1]+5),
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
           arrowprops=dict(arrowstyle='->', lw=1.5))

# BOTTOM RIGHT: Loss decrease
ax = axes[1, 1]

ax.plot(iterations, loss_history, 'o-', linewidth=2.5, markersize=8, 
       color='darkred')

ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax.set_title('Loss Decreases by Following Negative Gradient', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)

# Annotate
ax.annotate(f'Loss={loss_history[0]:.0f}', xy=(0, loss_history[0]), 
           xytext=(2, loss_history[0]+50),
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
           arrowprops=dict(arrowstyle='->', lw=1.5))

ax.annotate(f'Loss≈{loss_history[-1]:.1f}', xy=(len(loss_history)-1, loss_history[-1]), 
           xytext=(len(loss_history)-4, loss_history[-1]+50),
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
           arrowprops=dict(arrowstyle='->', lw=1.5))

plt.suptitle('📐 Mathematical Proof: Residuals = Negative Gradients', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: NUMERICAL VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("PART 3: NUMERICAL VERIFICATION (100 samples)")
print("═" * 80)

# Create comparison table
comparison = pd.DataFrame({
    'Sample': range(1, 11),
    'y': y_true_viz[:10],
    'F': F_init[:10],
    'Residual': residuals_viz[:10],
    'Gradient': gradient_viz[:10],
    'Neg_Gradient': neg_gradient_viz[:10],
    '2×Residual': 2*residuals_viz[:10]
})

print("\n📊 First 10 samples:")
print()
print(comparison.to_string(index=False, float_format='%.2f'))

print("\n✅ VERIFICATION:")
print(f"   Neg_Gradient = 2 × Residual? {np.allclose(neg_gradient_viz[:10], 2*residuals_viz[:10])}")

# Statistical verification
print("\n📈 STATISTICAL VERIFICATION (all 100 samples):")
print(f"\n   Mean residual:         {residuals_viz.mean():.4f}")
print(f"   Mean neg. gradient/2:  {(neg_gradient_viz/2).mean():.4f}")
print(f"   Difference:            {abs(residuals_viz.mean() - (neg_gradient_viz/2).mean()):.8f}")

print(f"\n   Correlation:           {np.corrcoef(residuals_viz, neg_gradient_viz/2)[0,1]:.10f}")
print(f"   Expected:              1.0000000000")

if np.allclose(residuals_viz, neg_gradient_viz/2):
    print("\n   ✅ MATHEMATICALLY PROVEN: Residuals = Negative Gradients / 2")
else:
    print("\n   ❌ ERROR in computation")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("🎯 SUMMARY: WHY GRADIENT BOOSTING WORKS")
print("═" * 80)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  1. Loss Function:        L = Σ (yᵢ - F(xᵢ))²                              │
│                                                                             │
│  2. Gradient:             ∂L/∂F = -2(yᵢ - F(xᵢ))                           │
│                                                                             │
│  3. Negative Gradient:    -∂L/∂F = 2(yᵢ - F(xᵢ)) = 2rᵢ                    │
│                                                                             │
│  4. Residual:             rᵢ = yᵢ - F(xᵢ)                                   │
│                                                                             │
│  5. Connection:           rᵢ = (-∂L/∂F) / 2                                 │
│                                                                             │
│  ✅ Learning residuals ≡ Gradient descent in function space                │
│                                                                             │
│  Each tree h(x) approximates:                                               │
│    • Residuals (intuitive view)                                            │
│    • Negative gradient (mathematical view)                                 │
│                                                                             │
│  Both are THE SAME THING!                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "═" * 80)
print("\n🎓 INTERVIEW TIP:")
print("\n   'Gradient Boosting performs gradient descent in function space.'")
print("   'Residuals are proportional to the negative gradient.'")
print("   'Each tree learns -∇L, moving toward lower loss.'")
print("\n" + "═" * 80)

# COMMAND ----------

# DBTITLE 1,🎯 Model Selection: Diagnosing Overfitting
# MAGIC %md
# MAGIC # 🎯 Model Selection: Diagnosing Overfitting
# MAGIC
# MAGIC **You identified:** GB has overfitting issue.
# MAGIC
# MAGIC **You missed:** Complete hyperparameter analysis.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 The Scenario
# MAGIC
# MAGIC **Given Results:**
# MAGIC
# MAGIC | Model | Train Acc | Test Acc | Gap |
# MAGIC |-------|-----------|----------|-----|
# MAGIC | **RF** | 0.96 | 0.89 | **0.07** |
# MAGIC | **GB** | 0.99 | 0.78 | **0.21** |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔍 Analysis
# MAGIC
# MAGIC ### **Step 1: Calculate Generalization Gap**
# MAGIC
# MAGIC ```
# MAGIC Gap = Train Accuracy - Test Accuracy
# MAGIC
# MAGIC RF Gap:  0.96 - 0.89 = 0.07  ← Small gap ✅
# MAGIC GB Gap:  0.99 - 0.78 = 0.21  ← Large gap ❌
# MAGIC ```
# MAGIC
# MAGIC **Interpretation:**
# MAGIC - **Small gap (<0.10):** Good generalization ✅
# MAGIC - **Large gap (>0.15):** Overfitting ❌
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Step 2: Identify Issue**
# MAGIC
# MAGIC **Random Forest:**
# MAGIC ```
# MAGIC Train: 0.96  ← Not perfect (good!)
# MAGIC Test:  0.89  ← Reasonable
# MAGIC Gap:   0.07  ← Small
# MAGIC
# MAGIC Diagnosis: ✅ Well-balanced
# MAGIC Status: Ready to deploy!
# MAGIC ```
# MAGIC
# MAGIC **Gradient Boosting:**
# MAGIC ```
# MAGIC Train: 0.99  ← Near perfect (suspicious!)
# MAGIC Test:  0.78  ← Poor
# MAGIC Gap:   0.21  ← Large
# MAGIC
# MAGIC Diagnosis: ❌ Severe overfitting
# MAGIC Status: Need to tune hyperparameters
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Step 3: Decision**
# MAGIC
# MAGIC **Deploy:**
# MAGIC ```
# MAGIC ✅ Random Forest
# MAGIC
# MAGIC Why?
# MAGIC   • Better generalization
# MAGIC   • Test accuracy: 0.89 > 0.78
# MAGIC   • More stable predictions
# MAGIC   • Less risk in production
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔧 Hyperparameter Tuning for GB Overfitting
# MAGIC
# MAGIC **You mentioned:** `n_estimators` (number of trees)
# MAGIC
# MAGIC **Complete List:**
# MAGIC
# MAGIC ### **🔑 5 Key Hyperparameters to Fix Overfitting:**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **1️⃣ `n_estimators` (Number of Trees)**
# MAGIC
# MAGIC **Problem:**
# MAGIC ```
# MAGIC Too many trees → Model memorizes training data
# MAGIC
# MAGIC Example:
# MAGIC   n_estimators = 1000  ← TOO MANY!
# MAGIC   Model learns noise, outliers, anomalies
# MAGIC ```
# MAGIC
# MAGIC **Solution:**
# MAGIC ```
# MAGIC ✅ Reduce to 50-200 trees
# MAGIC ✅ Use early stopping
# MAGIC ✅ Monitor validation loss
# MAGIC ```
# MAGIC
# MAGIC **How to Tune:**
# MAGIC ```python
# MAGIC from sklearn.ensemble import GradientBoostingClassifier
# MAGIC
# MAGIC # Try different values
# MAGIC for n in [50, 100, 200, 500]:
# MAGIC     model = GradientBoostingClassifier(n_estimators=n)
# MAGIC     model.fit(X_train, y_train)
# MAGIC     train_score = model.score(X_train, y_train)
# MAGIC     test_score = model.score(X_test, y_test)
# MAGIC     gap = train_score - test_score
# MAGIC     
# MAGIC     print(f"n={n}: Train={train_score:.3f}, Test={test_score:.3f}, Gap={gap:.3f}")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **2️⃣ `max_depth` (Tree Depth)**
# MAGIC
# MAGIC **Problem:**
# MAGIC ```
# MAGIC Too deep → Individual trees overfit
# MAGIC
# MAGIC Example:
# MAGIC   max_depth = 10  ← TOO DEEP!
# MAGIC   
# MAGIC   Creates 2^10 = 1,024 leaf nodes
# MAGIC   Can memorize individual training samples
# MAGIC ```
# MAGIC
# MAGIC **Solution:**
# MAGIC ```
# MAGIC ✅ Use shallow trees: max_depth = 3-5
# MAGIC ✅ GB works best with "weak learners"
# MAGIC ✅ Let sequential learning do the work
# MAGIC ```
# MAGIC
# MAGIC **Impact:**
# MAGIC ```
# MAGIC max_depth=3:  Train=0.92, Test=0.88  ✅ Good!
# MAGIC max_depth=10: Train=0.99, Test=0.78  ❌ Overfit!
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **3️⃣ `learning_rate` (α / Shrinkage)**
# MAGIC
# MAGIC **Problem:**
# MAGIC ```
# MAGIC Too high → Takes large steps, overfits quickly
# MAGIC
# MAGIC Example:
# MAGIC   learning_rate = 0.5  ← TOO HIGH!
# MAGIC   
# MAGIC   Each tree contributes 50% of its prediction
# MAGIC   Model fits training data too aggressively
# MAGIC ```
# MAGIC
# MAGIC **Solution:**
# MAGIC ```
# MAGIC ✅ Lower learning rate: 0.01 - 0.1
# MAGIC ✅ Combine with more trees (trade-off)
# MAGIC ✅ Typical: α=0.1 with n=100-200 trees
# MAGIC ```
# MAGIC
# MAGIC **Trade-off:**
# MAGIC ```
# MAGIC learning_rate = 0.5, n_estimators = 50  → Overfit
# MAGIC learning_rate = 0.05, n_estimators = 200 → Better generalization
# MAGIC ```
# MAGIC
# MAGIC **Formula:**
# MAGIC ```
# MAGIC F_m = F_{m-1} + α × h_m
# MAGIC
# MAGIC Lower α → Smaller updates → More conservative → Less overfitting
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **4️⃣ `min_samples_leaf` (Minimum Samples per Leaf)**
# MAGIC
# MAGIC **Problem:**
# MAGIC ```
# MAGIC Too few samples → Leaves can fit individual points
# MAGIC
# MAGIC Example:
# MAGIC   min_samples_leaf = 1  ← DEFAULT (BAD!)
# MAGIC   
# MAGIC   Allows leaves with 1 sample
# MAGIC   Perfect fit on training → No generalization
# MAGIC ```
# MAGIC
# MAGIC **Solution:**
# MAGIC ```
# MAGIC ✅ Increase to 5-20 samples per leaf
# MAGIC ✅ Forces leaves to represent patterns, not noise
# MAGIC ✅ Regularization effect
# MAGIC ```
# MAGIC
# MAGIC **Impact:**
# MAGIC ```
# MAGIC min_samples_leaf=1:  Train=0.99, Test=0.78  ❌
# MAGIC min_samples_leaf=10: Train=0.94, Test=0.87  ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **5️⃣ `subsample` (Stochastic Gradient Boosting)**
# MAGIC
# MAGIC **Problem:**
# MAGIC ```
# MAGIC Using 100% of data → Can overfit to training set
# MAGIC
# MAGIC Example:
# MAGIC   subsample = 1.0  ← DEFAULT
# MAGIC   
# MAGIC   Each tree sees ALL training data
# MAGIC   No diversity between trees
# MAGIC ```
# MAGIC
# MAGIC **Solution:**
# MAGIC ```
# MAGIC ✅ Use 50-80% of data per tree: subsample = 0.5-0.8
# MAGIC ✅ Adds randomness (like RF bootstrap)
# MAGIC ★ Creates diversity between trees
# MAGIC ✅ Reduces overfitting
# MAGIC ```
# MAGIC
# MAGIC **Impact:**
# MAGIC ```
# MAGIC subsample=1.0:  Train=0.99, Test=0.78  ❌
# MAGIC subsample=0.7:  Train=0.95, Test=0.86  ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Summary Table: Hyperparameter Effects
# MAGIC
# MAGIC | Hyperparameter | **Too High/Many** | **Too Low/Few** | **Sweet Spot** |
# MAGIC |----------------|-------------------|-----------------|----------------|
# MAGIC | `n_estimators` | Overfit, slow | Underfit | 100-200 |
# MAGIC | `max_depth` | Overfit | Underfit | 3-5 |
# MAGIC | `learning_rate` | Overfit fast | Slow convergence | 0.05-0.1 |
# MAGIC | `min_samples_leaf` | Underfit | Overfit | 5-20 |
# MAGIC | `subsample` | Less diversity | More underfitting | 0.7-0.8 |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚙️ Recommended Fix for GB Overfitting
# MAGIC
# MAGIC **Original (Overfit):**
# MAGIC ```python
# MAGIC GradientBoostingClassifier(
# MAGIC     n_estimators=1000,      # ❌ Too many
# MAGIC     max_depth=10,           # ❌ Too deep
# MAGIC     learning_rate=0.5,      # ❌ Too high
# MAGIC     min_samples_leaf=1,     # ❌ Too few
# MAGIC     subsample=1.0           # ❌ No diversity
# MAGIC )
# MAGIC
# MAGIC Result: Train=0.99, Test=0.78  ❌
# MAGIC ```
# MAGIC
# MAGIC **Fixed (Regularized):**
# MAGIC ```python
# MAGIC GradientBoostingClassifier(
# MAGIC     n_estimators=150,       # ✅ Moderate
# MAGIC     max_depth=3,            # ✅ Shallow
# MAGIC     learning_rate=0.1,      # ✅ Conservative
# MAGIC     min_samples_leaf=10,    # ✅ Regularized
# MAGIC     subsample=0.8           # ✅ Stochastic
# MAGIC )
# MAGIC
# MAGIC Expected: Train=0.94, Test=0.89  ✅
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📐 Additional Hyperparameters (Advanced)
# MAGIC
# MAGIC ### **6️⃣ `min_samples_split`**
# MAGIC ```
# MAGIC Minimum samples required to split a node
# MAGIC
# MAGIC Default: 2  ← Can split with just 2 samples (risky)
# MAGIC Recommended: 10-20
# MAGIC
# MAGIC Effect: Prevents overly specific splits
# MAGIC ```
# MAGIC
# MAGIC ### **7️⃣ `max_features`**
# MAGIC ```
# MAGIC Number of features to consider for each split
# MAGIC
# MAGIC Default: None (all features)
# MAGIC Recommended: 'sqrt' or 'log2'
# MAGIC
# MAGIC Effect: Adds randomness, like RF
# MAGIC ```
# MAGIC
# MAGIC ### **8️⃣ `max_leaf_nodes`**
# MAGIC ```
# MAGIC Maximum number of leaf nodes per tree
# MAGIC
# MAGIC Default: None (unlimited)
# MAGIC Recommended: 10-30
# MAGIC
# MAGIC Effect: Alternative to max_depth for complexity control
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Tuning Strategy
# MAGIC
# MAGIC ### **Step-by-Step Approach:**
# MAGIC
# MAGIC **1. Start Conservative:**
# MAGIC ```python
# MAGIC n_estimators = 100
# MAGIC max_depth = 3
# MAGIC learning_rate = 0.1
# MAGIC min_samples_leaf = 10
# MAGIC subsample = 0.8
# MAGIC ```
# MAGIC
# MAGIC **2. Monitor Gaps:**
# MAGIC ```python
# MAGIC train_acc = 0.94
# MAGIC test_acc = 0.88
# MAGIC gap = 0.06  ← Good!
# MAGIC ```
# MAGIC
# MAGIC **3. Adjust Based on Symptoms:**
# MAGIC
# MAGIC **If still overfitting (gap > 0.1):**
# MAGIC ```
# MAGIC ✅ Increase min_samples_leaf (10 → 20)
# MAGIC ✅ Decrease max_depth (3 → 2)
# MAGIC ✅ Lower learning_rate (0.1 → 0.05)
# MAGIC ✅ Lower subsample (0.8 → 0.7)
# MAGIC ```
# MAGIC
# MAGIC **If underfitting (train_acc < 0.85):**
# MAGIC ```
# MAGIC ✅ Increase n_estimators (100 → 200)
# MAGIC ✅ Increase max_depth (3 → 4)
# MAGIC ✅ Decrease min_samples_leaf (10 → 5)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎓 Interview Answer Template
# MAGIC
# MAGIC **Question:** "Given these results, which model would you deploy and why?"
# MAGIC
# MAGIC ```
# MAGIC Model    Train    Test    Gap
# MAGIC RF       0.96     0.89    0.07
# MAGIC GB       0.99     0.78    0.21
# MAGIC ```
# MAGIC
# MAGIC **Perfect Answer:**
# MAGIC
# MAGIC ```
# MAGIC ✅ "I would deploy Random Forest.
# MAGIC
# MAGIC    Analysis:
# MAGIC    
# MAGIC    1. Generalization Gap:
# MAGIC       RF: 0.07 (small → good generalization)
# MAGIC       GB: 0.21 (large → severe overfitting)
# MAGIC    
# MAGIC    2. Test Performance:
# MAGIC       RF: 0.89 > GB: 0.78
# MAGIC       RF performs better on unseen data
# MAGIC    
# MAGIC    3. GB Diagnosis:
# MAGIC       Near-perfect training (0.99) but poor test (0.78)
# MAGIC       → Model memorized training data
# MAGIC       → Will not generalize in production
# MAGIC    
# MAGIC    4. To Fix GB, I would tune:
# MAGIC       • n_estimators: Reduce from high value
# MAGIC       • max_depth: Use shallow trees (3-5)
# MAGIC       • learning_rate: Lower to 0.05-0.1
# MAGIC       • min_samples_leaf: Increase to 10-20
# MAGIC       • subsample: Use 0.7-0.8 for diversity
# MAGIC    
# MAGIC    5. Expected After Tuning:
# MAGIC       Train: ~0.94, Test: ~0.88
# MAGIC       Gap: ~0.06 (acceptable)
# MAGIC    
# MAGIC    But for immediate deployment: RF is the safer choice."
# MAGIC ```
# MAGIC ---
# MAGIC
# MAGIC ## 🔑 Key Takeaways
# MAGIC
# MAGIC ### **Overfitting Diagnosis:**
# MAGIC ```
# MAGIC ❌ High train accuracy + Low test accuracy = Overfitting
# MAGIC ✅ Moderate train accuracy + Close test accuracy = Good
# MAGIC ```
# MAGIC
# MAGIC ### **GB Overfitting Fixes (Priority Order):**
# MAGIC ```
# MAGIC 1. max_depth (3-5)           ← Most important
# MAGIC 2. learning_rate (0.05-0.1)  ← Second most
# MAGIC 3. min_samples_leaf (10-20)  ← Third
# MAGIC 4. n_estimators (100-200)    ← Fourth
# MAGIC 5. subsample (0.7-0.8)       ← Fifth
# MAGIC ```
# MAGIC
# MAGIC ### **RF vs GB for Production:**
# MAGIC ```
# MAGIC RF:
# MAGIC   ✅ More robust (less tuning needed)
# MAGIC   ✅ Naturally regularized (bootstrap + feature sampling)
# MAGIC   ✅ Safer default choice
# MAGIC   
# MAGIC  GB:
# MAGIC   ✅ Can achieve higher accuracy (with tuning)
# MAGIC   ❌ More sensitive to hyperparameters
# MAGIC   ❌ Higher risk of overfitting
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Next cell: Interactive hyperparameter tuning demo! 👇**

# COMMAND ----------

# DBTITLE 1,🔬 Demo: Model Selection & Hyperparameter Tuning
# ═══════════════════════════════════════════════════════════════════════════════
# 🔬 DEMO: MODEL SELECTION & HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

print("═" * 80)
print("🎯 MODEL SELECTION: Diagnosing & Fixing Overfitting")
print("═" * 80)

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP: Generate Dataset
# ═══════════════════════════════════════════════════════════════════════════════

print("\n📊 Generating classification dataset...")

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"   Train samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Features: {X.shape[1]}")

# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1: DEFAULT MODELS (Reproduce Interview Question)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("SCENARIO 1: DEFAULT MODELS")
print("═" * 80)

# Train RF (default)
rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
rf_default.fit(X_train, y_train)

rf_train = rf_default.score(X_train, y_train)
rf_test = rf_default.score(X_test, y_test)
rf_gap = rf_train - rf_test

print("\n🌲 Random Forest (default):")
print(f"   Train Accuracy: {rf_train:.3f}")
print(f"   Test Accuracy:  {rf_test:.3f}")
print(f"   Gap:            {rf_gap:.3f}")

if rf_gap < 0.10:
    print(f"   ✅ Good generalization!")
else:
    print(f"   ⚠️  Overfitting detected")

# Train GB (intentionally overfit)
gb_overfit = GradientBoostingClassifier(
    n_estimators=500,      # Many trees
    max_depth=8,          # Deep trees
    learning_rate=0.5,    # High learning rate
    min_samples_leaf=1,   # No regularization
    subsample=1.0,        # All data
    random_state=42
)
gb_overfit.fit(X_train, y_train)

gb_train = gb_overfit.score(X_train, y_train)
gb_test = gb_overfit.score(X_test, y_test)
gb_gap = gb_train - gb_test

print("\n🌲→🌲→🌲 Gradient Boosting (overfit params):")
print(f"   Train Accuracy: {gb_train:.3f}")
print(f"   Test Accuracy:  {gb_test:.3f}")
print(f"   Gap:            {gb_gap:.3f}")

if gb_gap < 0.10:
    print(f"   ✅ Good generalization")
else:
    print(f"   ❌ Severe overfitting!")

print("\n" + "─" * 80)
print("📊 COMPARISON:")
print("─" * 80)

print(f"\n   {'Model':<20} {'Train':<10} {'Test':<10} {'Gap':<10} {'Status'}")
print("   " + "-" * 65)
print(f"   {'RF (default)':<20} {rf_train:<10.3f} {rf_test:<10.3f} {rf_gap:<10.3f} {'Good ✅'}")
print(f"   {'GB (overfit)':<20} {gb_train:<10.3f} {gb_test:<10.3f} {gb_gap:<10.3f} {'Overfit ❌'}")

print(f"\n   🎯 DECISION: Deploy RF (better test performance: {rf_test:.3f} > {gb_test:.3f})")

# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 2: HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("SCENARIO 2: FIXING GB OVERFITTING")
print("═" * 80)

print("\n🔧 Tuning hyperparameters...\n")

results = []

# Test different configurations
configs = [
    {
        'name': 'GB Overfit',
        'params': {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.5,
            'min_samples_leaf': 1,
            'subsample': 1.0
        }
    },
    {
        'name': 'Fix: Reduce Trees',
        'params': {
            'n_estimators': 100,  # ← Reduced
            'max_depth': 8,
            'learning_rate': 0.5,
            'min_samples_leaf': 1,
            'subsample': 1.0
        }
    },
    {
        'name': 'Fix: Shallow Trees',
        'params': {
            'n_estimators': 100,
            'max_depth': 3,       # ← Reduced
            'learning_rate': 0.5,
            'min_samples_leaf': 1,
            'subsample': 1.0
        }
    },
    {
        'name': 'Fix: Lower LR',
        'params': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,  # ← Reduced
            'min_samples_leaf': 1,
            'subsample': 1.0
        }
    },
    {
        'name': 'Fix: Min Samples',
        'params': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'min_samples_leaf': 10,  # ← Increased
            'subsample': 1.0
        }
    },
    {
        'name': 'Fix: Subsample',
        'params': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'min_samples_leaf': 10,
            'subsample': 0.8         # ← Reduced
        }
    },
    {
        'name': 'GB Tuned (All Fixed)',
        'params': {
            'n_estimators': 150,
            'max_depth': 3,
            'learning_rate': 0.1,
            'min_samples_leaf': 10,
            'subsample': 0.8
        }
    }
]

for config in configs:
    model = GradientBoostingClassifier(**config['params'], random_state=42)
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    gap = train_score - test_score
    
    results.append({
        'Config': config['name'],
        'Train': train_score,
        'Test': test_score,
        'Gap': gap
    })
    
    status = "✅" if gap < 0.10 else "❌"
    print(f"   {config['name']:<25} Train={train_score:.3f}  Test={test_score:.3f}  Gap={gap:.3f}  {status}")

results_df = pd.DataFrame(results)

print("\n" + "─" * 80)
print("📊 RESULTS SUMMARY:")
print("─" * 80)

best_config = results_df.loc[results_df['Test'].idxmax()]
print(f"\n   Best Configuration: {best_config['Config']}")
print(f"      Train: {best_config['Train']:.3f}")
print(f"      Test:  {best_config['Test']:.3f}")
print(f"      Gap:   {best_config['Gap']:.3f}")

if best_config['Gap'] < 0.10:
    print(f"\n   ✅ Successfully fixed overfitting!")
    print(f"   ✅ Test accuracy improved from {gb_test:.3f} to {best_config['Test']:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("📊 VISUALIZATION")
print("═" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# TOP LEFT: Train vs Test scores
ax = axes[0, 0]

x_pos = np.arange(len(results_df))
width = 0.35

ax.bar(x_pos - width/2, results_df['Train'], width, label='Train', 
      color='blue', alpha=0.7, edgecolor='black')
ax.bar(x_pos + width/2, results_df['Test'], width, label='Test', 
      color='green', alpha=0.7, edgecolor='black')

ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Train vs Test Accuracy', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(range(1, len(results_df)+1))
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')

# TOP RIGHT: Generalization gap
ax = axes[0, 1]

colors = ['red' if gap > 0.10 else 'green' for gap in results_df['Gap']]
ax.bar(x_pos, results_df['Gap'], color=colors, alpha=0.7, edgecolor='black')
ax.axhline(0.10, color='red', linestyle='--', linewidth=2, label='Overfit threshold')

ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
ax.set_ylabel('Gap (Train - Test)', fontsize=12, fontweight='bold')
ax.set_title('Generalization Gap', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(range(1, len(results_df)+1))
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')

# BOTTOM LEFT: Configuration details
ax = axes[1, 0]
ax.axis('off')

config_text = "Configuration Details:\n\n"
for i, config in enumerate(configs, 1):
    config_text += f"{i}. {config['name']}\n"
    for key, val in config['params'].items():
        config_text += f"   {key}: {val}\n"
    config_text += "\n"

ax.text(0.05, 0.95, config_text, transform=ax.transAxes, 
       fontsize=9, verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# BOTTOM RIGHT: Scatter plot
ax = axes[1, 1]

ax.scatter(results_df['Train'], results_df['Test'], s=200, 
          c=colors, alpha=0.7, edgecolors='black', linewidth=2)

# Perfect generalization line
ax.plot([0.7, 1.0], [0.7, 1.0], 'k--', linewidth=2, alpha=0.5, 
       label='Perfect generalization')

# Annotate points
for i, row in results_df.iterrows():
    ax.annotate(str(i+1), xy=(row['Train'], row['Test']), 
               fontsize=11, fontweight='bold', ha='center', va='center')

ax.set_xlabel('Train Accuracy', fontsize=12, fontweight='bold')
ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Train vs Test Performance', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim([0.7, 1.0])
ax.set_ylim([0.7, 1.0])

plt.suptitle('🎯 Model Selection: Fixing GB Overfitting', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("🎯 FINAL RECOMMENDATION")
print("═" * 80)

# Train final tuned model
gb_tuned = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.1,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42
)
gb_tuned.fit(X_train, y_train)

gb_tuned_train = gb_tuned.score(X_train, y_train)
gb_tuned_test = gb_tuned.score(X_test, y_test)
gb_tuned_gap = gb_tuned_train - gb_tuned_test

print("\n📋 FINAL COMPARISON:\n")
print(f"   {'Model':<25} {'Train':<10} {'Test':<10} {'Gap':<10} {'Status'}")
print("   " + "-" * 70)
print(f"   {'RF (default)':<25} {rf_train:<10.3f} {rf_test:<10.3f} {rf_gap:<10.3f} {'✅ Good'}")
print(f"   {'GB (original)':<25} {gb_train:<10.3f} {gb_test:<10.3f} {gb_gap:<10.3f} {'❌ Overfit'}")
print(f"   {'GB (tuned)':<25} {gb_tuned_train:<10.3f} {gb_tuned_test:<10.3f} {gb_tuned_gap:<10.3f} {'✅ Fixed'}")

print("\n" + "─" * 80)

if gb_tuned_test > rf_test:
    print(f"\n   🏆 DEPLOY: GB (tuned) - Best test accuracy ({gb_tuned_test:.3f})")
    print(f"   \n   Successfully fixed overfitting:")
    print(f"      Original GB test: {gb_test:.3f}")
    print(f"      Tuned GB test:    {gb_tuned_test:.3f}")
    print(f"      Improvement:      +{(gb_tuned_test - gb_test):.3f}")
else:
    print(f"\n   🏆 DEPLOY: RF - More robust ({rf_test:.3f} vs {gb_tuned_test:.3f})")
    print(f"   \n   Even after tuning, RF has better generalization.")

print("\n" + "─" * 80)

print("\n💡 KEY LESSONS:")
print("\n   1. Always check generalization gap (Train - Test)")
print("   2. High train + Low test = Overfitting")
print("   3. GB requires careful hyperparameter tuning")
print("   4. Top 5 parameters to tune for overfitting:")
print("      • max_depth (most important)")
print("      • learning_rate")
print("      • min_samples_leaf")
print("      • n_estimators")
print("      • subsample")
print("   5. RF is more robust with default parameters")

print("\n" + "═" * 80)