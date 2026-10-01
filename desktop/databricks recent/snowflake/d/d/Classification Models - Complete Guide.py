# Databricks notebook source
# DBTITLE 1,Introduction and Learning Objectives
# MAGIC %md
# MAGIC # Classification Problems - Complete Guide
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC After completing this notebook, you will be able to:
# MAGIC
# MAGIC * **Understand classification problems** and their real-world applications
# MAGIC * **Build logistic regression models** using statsmodels and scikit-learn
# MAGIC * **Build decision tree classifiers** using scikit-learn
# MAGIC * **Split datasets** into training and validation sets
# MAGIC * **Measure model performance** using:
# MAGIC   - Confusion matrices
# MAGIC   - Sensitivity (Recall), Specificity, Precision, F1-Score
# MAGIC   - ROC curves and AUC
# MAGIC * **Find optimal classification thresholds** for decision-making
# MAGIC * **Perform model diagnostics** and assess statistical significance
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Table of Contents
# MAGIC
# MAGIC 1. Introduction to Classification
# MAGIC 2. Data Preparation
# MAGIC 3. Logistic Regression
# MAGIC 4. Model Evaluation Metrics
# MAGIC 5. Decision Trees
# MAGIC 6. Model Comparison

# COMMAND ----------

# DBTITLE 1,1. Introduction to Classification
# MAGIC %md
# MAGIC ## 1. Introduction to Classification
# MAGIC
# MAGIC ### What is Classification?
# MAGIC
# MAGIC Classification is a supervised learning technique where the goal is to predict **discrete class labels** for new observations based on past observations.
# MAGIC
# MAGIC ### Key Characteristics:
# MAGIC
# MAGIC * **Binary Classification**: 2 classes (e.g., spam/not spam, disease/no disease)
# MAGIC * **Multiclass Classification**: 3+ classes (e.g., low/medium/high risk)
# MAGIC * **Output**: Probability of belonging to each class
# MAGIC
# MAGIC ### Real-World Applications:
# MAGIC
# MAGIC 1. **Financial Services**: Credit risk assessment (good/bad credit)
# MAGIC 2. **Healthcare**: Disease diagnosis (positive/negative)
# MAGIC 3. **Marketing**: Customer churn prediction (will churn/won't churn)
# MAGIC 4. **E-commerce**: Fraud detection (fraudulent/legitimate)
# MAGIC 5. **HR**: Employee attrition (will leave/will stay)
# MAGIC
# MAGIC ### Common Algorithms:
# MAGIC
# MAGIC * Logistic Regression
# MAGIC * Decision Trees
# MAGIC * Random Forests
# MAGIC * Support Vector Machines
# MAGIC * Neural Networks
# MAGIC
# MAGIC In this notebook, we'll focus on **Logistic Regression** and **Decision Trees**.

# COMMAND ----------

# DBTITLE 1,Import Libraries
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
%matplotlib inline

print("Libraries imported successfully!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# COMMAND ----------

# DBTITLE 1,2. Data Preparation
# MAGIC %md
# MAGIC ## 2. Data Preparation
# MAGIC
# MAGIC For this tutorial, we'll use the **German Credit Dataset** from the UCI Machine Learning Repository.
# MAGIC
# MAGIC **Dataset Description:**
# MAGIC * 1000 observations
# MAGIC * Predicting credit risk: Good Credit (0) or Bad Credit (1)
# MAGIC * Features include: checking account status, credit duration, amount, age, employment status, etc.
# MAGIC
# MAGIC **Note**: You'll need to download the German Credit Data CSV file. For this demonstration, we'll create a sample dataset with similar characteristics.

# COMMAND ----------

# DBTITLE 1,Create Sample Dataset
# Create a sample dataset with characteristics similar to credit data
# This simulates the German Credit dataset structure

np.random.seed(42)
n_samples = 1000

# Generate synthetic credit data
data = {
    'duration': np.random.randint(6, 72, n_samples),  # loan duration in months
    'amount': np.random.randint(250, 18500, n_samples),  # loan amount
    'inst_rate': np.random.randint(1, 5, n_samples),  # installment rate
    'age': np.random.randint(19, 75, n_samples),  # age
    'num_credits': np.random.randint(1, 5, n_samples),  # number of credits
    'residing_since': np.random.randint(1, 5, n_samples),  # years at residence
}

# Categorical features
data['checkin_acc'] = np.random.choice(['A11', 'A12', 'A13', 'A14'], n_samples, p=[0.3, 0.3, 0.2, 0.2])
data['credit_history'] = np.random.choice(['A30', 'A31', 'A32', 'A33', 'A34'], n_samples)
data['savings_acc'] = np.random.choice(['A61', 'A62', 'A63', 'A64', 'A65'], n_samples)
data['present_emp_since'] = np.random.choice(['A71', 'A72', 'A73', 'A74', 'A75'], n_samples)
data['personal_status'] = np.random.choice(['A91', 'A92', 'A93', 'A94'], n_samples)
data['inst_plans'] = np.random.choice(['A141', 'A142', 'A143'], n_samples, p=[0.1, 0.1, 0.8])
data['job'] = np.random.choice(['A171', 'A172', 'A173', 'A174'], n_samples, p=[0.05, 0.15, 0.6, 0.2])

# Create target variable with some logic
# Higher duration, amount, and certain account types increase bad credit probability
risk_score = (
    (data['duration'] / 72) * 0.3 +
    (data['amount'] / 18500) * 0.2 +
    (data['age'] < 25).astype(int) * 0.2 +
    (pd.Series(data['checkin_acc']) == 'A11').astype(int) * 0.3 +
    np.random.random(n_samples) * 0.3  # add randomness
)

data['status'] = (risk_score > 0.5).astype(int)  # 1 = Bad Credit, 0 = Good Credit

# Create DataFrame
credit_df = pd.DataFrame(data)

print(f"Dataset created with {len(credit_df)} records")
print(f"\nClass distribution:")
print(credit_df['status'].value_counts())
print(f"\nBad credit rate: {credit_df['status'].mean():.1%}")

# COMMAND ----------

# DBTITLE 1,Explore Dataset
# Display dataset info and first few rows
print("Dataset Information:")
print("="*50)
credit_df.info()

print("\n" + "="*50)
print("\nFirst 5 rows:")
display(credit_df.head())

print("\nBasic Statistics:")
display(credit_df.describe())

# COMMAND ----------

# DBTITLE 1,Feature Engineering
# MAGIC %md
# MAGIC ### Encoding Categorical Features
# MAGIC
# MAGIC Machine learning algorithms require numerical input. We'll use **one-hot encoding** (dummy variables) to convert categorical features to numerical format.
# MAGIC
# MAGIC **Key points:**
# MAGIC * Creates binary columns for each category
# MAGIC * `drop_first=True` drops one category to avoid multicollinearity
# MAGIC * The dropped category becomes the baseline/reference category

# COMMAND ----------

# DBTITLE 1,Encode Categorical Variables
# Separate features and target
X_features = [col for col in credit_df.columns if col != 'status']
y_target = credit_df['status']

print(f"Features: {len(X_features)}")
print(X_features)

# Perform one-hot encoding on categorical features
encoded_credit_df = pd.get_dummies(credit_df[X_features], drop_first=True)

print(f"\nAfter encoding: {len(encoded_credit_df.columns)} features")
print("\nEncoded feature names:")
for i, col in enumerate(encoded_credit_df.columns, 1):
    print(f"{i:2d}. {col}")

print(f"\nShape: {encoded_credit_df.shape}")

# COMMAND ----------

# DBTITLE 1,Verify Encoding
# Example: Show how checkin_acc was encoded
print("Original checkin_acc categories:")
print(credit_df['checkin_acc'].value_counts().sort_index())

print("\nEncoded dummy variables (first 10 rows):")
dummy_cols = [col for col in encoded_credit_df.columns if col.startswith('checkin_acc')]
if dummy_cols:
    display(encoded_credit_df[dummy_cols].head(10))
    print("\nNote: When all dummy variables = 0, it represents the dropped category (baseline)")

# COMMAND ----------

# DBTITLE 1,Train-Test Split
# MAGIC %md
# MAGIC ### Train-Test Split
# MAGIC
# MAGIC Before building models, we split the data:
# MAGIC * **Training Set (70%)**: Used to train the model
# MAGIC * **Test Set (30%)**: Used to evaluate model performance on unseen data
# MAGIC
# MAGIC This prevents **overfitting** and gives us an honest estimate of model performance.

# COMMAND ----------

# DBTITLE 1,Split Data into Train and Test Sets
from sklearn.model_selection import train_test_split

# Split the data: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    encoded_credit_df, 
    y_target,
    test_size=0.3,
    random_state=42,
    stratify=y_target  # Maintain class proportions
)

print("Dataset Split Summary")
print("="*50)
print(f"Training set size: {len(X_train)} ({len(X_train)/len(encoded_credit_df):.1%})")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(encoded_credit_df):.1%})")
print(f"\nTraining set class distribution:")
print(y_train.value_counts())
print(f"\nTest set class distribution:")
print(y_test.value_counts())

# COMMAND ----------

# DBTITLE 1,3. Logistic Regression Theory
# MAGIC %md
# MAGIC ## 3. Logistic Regression
# MAGIC
# MAGIC ### What is Logistic Regression?
# MAGIC
# MAGIC Logistic regression models the **probability** that an observation belongs to a particular class.
# MAGIC
# MAGIC **Mathematical Form:**
# MAGIC
# MAGIC $$P(Y=1) = \frac{e^Z}{1 + e^Z} = \frac{1}{1 + e^{-Z}}$$
# MAGIC
# MAGIC where $Z = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_mX_m$
# MAGIC
# MAGIC **Key Properties:**
# MAGIC * Output is between 0 and 1 (probability)
# MAGIC * S-shaped (sigmoid) curve
# MAGIC * Linear relationship between features and log-odds
# MAGIC * Interpretable coefficients
# MAGIC
# MAGIC ### Log-Odds (Logit):
# MAGIC
# MAGIC $$\ln\left(\frac{P(Y=1)}{1-P(Y=1)}\right) = Z$$
# MAGIC
# MAGIC This is why it's called **logistic** regression!

# COMMAND ----------

# DBTITLE 1,Build Logistic Regression Model
import statsmodels.api as sm

# Add constant term for intercept
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Build logistic regression model
logit_model = sm.Logit(y_train, X_train_const)
logit_result = logit_model.fit()

print("Model fitting complete!")
print(f"Number of iterations: {logit_result.mle_retvals['iterations']}")
print(f"Converged: {logit_result.mle_retvals['converged']}")

# COMMAND ----------

# DBTITLE 1,Model Summary
# Display model summary
print("\n" + "="*80)
print("LOGISTIC REGRESSION MODEL SUMMARY")
print("="*80)

summary = logit_result.summary2()
print(summary)

# COMMAND ----------

# DBTITLE 1,Interpreting Model Summary
# MAGIC %md
# MAGIC ### Understanding the Model Summary
# MAGIC
# MAGIC **Key Metrics to Check:**
# MAGIC
# MAGIC 1. **Pseudo R-squared**: Measure of goodness-of-fit (0 to 1, higher is better)
# MAGIC 2. **Log-Likelihood**: Higher values indicate better fit
# MAGIC 3. **LLR p-value**: Tests if model is better than null model (want p < 0.05)
# MAGIC 4. **Coefficients**: 
# MAGIC    - Positive coefficient → increases probability of class 1
# MAGIC    - Negative coefficient → decreases probability of class 1
# MAGIC 5. **P > |z|**: Statistical significance of each feature (want p < 0.05)
# MAGIC
# MAGIC **Statistical Tests:**
# MAGIC * **Wald Test**: Tests significance of individual coefficients (z-statistic)
# MAGIC * **Likelihood Ratio Test**: Tests overall model significance (LLR p-value)

# COMMAND ----------

# DBTITLE 1,Identify Significant Features
# Extract statistically significant features (p < 0.05)
def get_significant_features(model, alpha=0.05):
    """Extract features with p-value < alpha"""
    p_values = model.pvalues
    significant = p_values[p_values < alpha]
    return significant.sort_values()

significant_features = get_significant_features(logit_result)

print("\nStatistically Significant Features (p < 0.05):")
print("="*60)
for feature, p_value in significant_features.items():
    coef = logit_result.params[feature]
    direction = "↑ Increases" if coef > 0 else "↓ Decreases"
    print(f"{feature:30s} | p={p_value:.4f} | {direction} bad credit risk")

print(f"\n{len(significant_features)} out of {len(logit_result.params)} features are significant")

# COMMAND ----------

# DBTITLE 1,Make Predictions
# Make predictions on test set
y_pred_proba = logit_result.predict(X_test_const)

# Create results dataframe
predictions_df = pd.DataFrame({
    'actual': y_test,
    'predicted_probability': y_pred_proba
})

print("Prediction Results (Sample):")
print("="*50)
display(predictions_df.sample(10, random_state=42).sort_index())

print(f"\nPredicted Probability Statistics:")
print(predictions_df['predicted_probability'].describe())

# COMMAND ----------

# DBTITLE 1,Visualize Predictions
# Visualize predicted probabilities by actual class
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(predictions_df[predictions_df['actual'] == 0]['predicted_probability'], 
             bins=30, alpha=0.6, label='Good Credit (0)', color='green')
axes[0].hist(predictions_df[predictions_df['actual'] == 1]['predicted_probability'], 
             bins=30, alpha=0.6, label='Bad Credit (1)', color='red')
axes[0].set_xlabel('Predicted Probability')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Predicted Probabilities by Actual Class')
axes[0].legend()
axes[0].axvline(0.5, color='black', linestyle='--', label='Default Threshold')

# Box plot
axes[1].boxplot([predictions_df[predictions_df['actual'] == 0]['predicted_probability'],
                 predictions_df[predictions_df['actual'] == 1]['predicted_probability']],
                labels=['Good Credit (0)', 'Bad Credit (1)'])
axes[1].set_ylabel('Predicted Probability')
axes[1].set_title('Predicted Probability by Actual Class')
axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nObservation: Good separation = better model performance")

# COMMAND ----------

# DBTITLE 1,4. Model Evaluation Metrics
# MAGIC %md
# MAGIC ## 4. Model Evaluation Metrics
# MAGIC
# MAGIC ### Classification Threshold
# MAGIC
# MAGIC Logistic regression outputs **probabilities**. We need a **threshold** to convert them to class predictions:
# MAGIC
# MAGIC * Probability ≥ threshold → Predict class 1 (Bad Credit)
# MAGIC * Probability < threshold → Predict class 0 (Good Credit)
# MAGIC
# MAGIC Default threshold is typically **0.5**, but we'll learn how to optimize this.
# MAGIC
# MAGIC ### Confusion Matrix
# MAGIC
# MAGIC A confusion matrix shows the counts of:
# MAGIC * **True Positives (TP)**: Correctly predicted class 1
# MAGIC * **True Negatives (TN)**: Correctly predicted class 0
# MAGIC * **False Positives (FP)**: Incorrectly predicted class 1 (Type I error)
# MAGIC * **False Negatives (FN)**: Incorrectly predicted class 0 (Type II error)
# MAGIC
# MAGIC |                | Predicted Negative | Predicted Positive |
# MAGIC |----------------|-------------------|--------------------|
# MAGIC | Actual Negative | TN               | FP                 |
# MAGIC | Actual Positive | FN               | TP                 |

# COMMAND ----------

# DBTITLE 1,Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score

# Apply threshold of 0.5
threshold = 0.5
y_pred = (y_pred_proba >= threshold).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create visualization
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Good Credit (0)', 'Bad Credit (1)'],
            yticklabels=['Good Credit (0)', 'Bad Credit (1)'],
            cbar_kws={'label': 'Count'},
            ax=ax)

ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title(f'Confusion Matrix (Threshold = {threshold})', fontsize=14, fontweight='bold')

# Add text annotations for quadrants
tn, fp, fn, tp = cm.ravel()
ax.text(0.5, 0.25, 'TN', ha='center', va='center', fontsize=10, color='darkblue', fontweight='bold')
ax.text(1.5, 0.25, 'FP\n(Type I Error)', ha='center', va='center', fontsize=10, color='darkred', fontweight='bold')
ax.text(0.5, 1.25, 'FN\n(Type II Error)', ha='center', va='center', fontsize=10, color='darkred', fontweight='bold')
ax.text(1.5, 1.25, 'TP', ha='center', va='center', fontsize=10, color='darkblue', fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\nConfusion Matrix Breakdown:")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")

# COMMAND ----------

# DBTITLE 1,Performance Metrics Theory
# MAGIC %md
# MAGIC ### Performance Metrics Formulas
# MAGIC
# MAGIC **Accuracy**: Overall correctness
# MAGIC $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
# MAGIC
# MAGIC **Sensitivity (Recall, TPR)**: Ability to find positives
# MAGIC $$\text{Sensitivity} = \frac{TP}{TP + FN}$$
# MAGIC
# MAGIC **Specificity (TNR)**: Ability to find negatives
# MAGIC $$\text{Specificity} = \frac{TN}{TN + FP}$$
# MAGIC
# MAGIC **Precision (PPV)**: Accuracy of positive predictions
# MAGIC $$\text{Precision} = \frac{TP}{TP + FP}$$
# MAGIC
# MAGIC **F1-Score**: Harmonic mean of precision and recall
# MAGIC $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

# COMMAND ----------

# DBTITLE 1,Calculate Performance Metrics
# Calculate all metrics manually
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall, TPR
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

print("\n" + "="*60)
print("PERFORMANCE METRICS SUMMARY")
print("="*60)
print(f"Accuracy:    {accuracy:.4f} ({accuracy:.1%})")
print(f"Sensitivity: {sensitivity:.4f} ({sensitivity:.1%}) - Recall/TPR")
print(f"Specificity: {specificity:.4f} ({specificity:.1%}) - TNR")
print(f"Precision:   {precision:.4f} ({precision:.1%})")
print(f"F1-Score:    {f1_score:.4f}")
print("="*60)

# Also get classification report from sklearn
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Good Credit (0)', 'Bad Credit (1)']))

# COMMAND ----------

