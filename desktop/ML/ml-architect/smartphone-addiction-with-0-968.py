# Databricks notebook source


# COMMAND ----------

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

# COMMAND ----------

!pip install -q lightgbm xgboost catboost

# COMMAND ----------

import os
import gc
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize

import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURATION
# ================================================================
SEED = 42
N_SPLITS = 5
TARGET = "addicted_label"
ID_COL = "id"

# ================================================================
# DATA LOADING & MEMORY OPTIMIZATION
# ================================================================
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and not pd.api.types.is_categorical_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df

train = pd.read_csv("/kaggle/input/competitions/playground-series-s6e8/train.csv")
test = pd.read_csv("/kaggle/input/competitions/playground-series-s6e8/test.csv")

# ================================================================
# FEATURE ENGINEERING (ADVANCED)
# ================================================================
def engineer_features(df):
    # Time Ratios
    if 'daily_screen_time_hours' in df.columns:
        # Productivity vs Entertainment
        if 'work_study_hours' in df.columns and 'social_media_hours' in df.columns:
            df['prod_ent_ratio'] = df['work_study_hours'] / (df['social_media_hours'] + df['gaming_hours'] + 0.5)
        
        # Screen time vs Sleep (Health Proxy)
        if 'sleep_hours' in df.columns:
            df['screen_sleep_diff'] = df['daily_screen_time_hours'] - df['sleep_hours']
            df['sleep_efficiency'] = df['sleep_hours'] / (df['daily_screen_time_hours'] + 1)

    # Intensity Features
    if 'notifications_per_day' in df.columns and 'app_opens_per_day' in df.columns:
        df['notification_intensity'] = df['notifications_per_day'] / (df['app_opens_per_day'] + 1)
    
    # Aggregated "Digital Footprint"
    digital_cols = ['social_media_hours', 'gaming_hours', 'daily_screen_time_hours']
    df['total_digital_hours'] = df[digital_cols].sum(axis=1)
    
    # Identify Categoricals
    cat_features = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_features:
        df[col] = df[col].fillna("Unknown").astype('category')
        
    return df

train = engineer_features(train)
test = engineer_features(test)

X = train.drop(columns=[TARGET, ID_COL])
y = train[TARGET]
X_test = test.drop(columns=[ID_COL])

cat_cols = X.select_dtypes(include=['category']).columns.tolist()

# ================================================================
# MODEL HYPERPARAMETERS (TUNED FOR S6E8)
# ================================================================
lgb_params = {
    'n_estimators': 5000,
    'learning_rate': 0.01,
    'num_leaves': 63,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'objective': 'binary',
    'metric': 'auc',
    'device': 'cpu', # Use 'gpu' if available
    'random_state': SEED,
    'verbose': -1
}

xgb_params = {
    'n_estimators': 5000,
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': -1,
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'enable_categorical': True, # Crucial for XGBoost performance
    'random_state': SEED
}

cat_params = {
    'iterations': 5000,
    'learning_rate': 0.02,
    'depth': 6,
    'l2_leaf_reg': 3,
    'eval_metric': 'AUC',
    'early_stopping_rounds': 200,
    'random_seed': SEED,
    'verbose': False,
    'cat_features': cat_cols
}

# ================================================================
# CROSS-VALIDATION LOOP
# ================================================================
oof_lgb, oof_xgb, oof_cat = [np.zeros(len(X)) for _ in range(3)]
pred_lgb, pred_xgb, pred_cat = [np.zeros(len(X_test)) for _ in range(3)]

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"--- FOLD {fold+1} ---")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 1. LightGBM
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                  callbacks=[lgb.early_stopping(200)])
    oof_lgb[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
    pred_lgb += model_lgb.predict_proba(X_test)[:, 1] / N_SPLITS

    # 2. XGBoost
    model_xgb = XGBClassifier(**xgb_params)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    pred_xgb += model_xgb.predict_proba(X_test)[:, 1] / N_SPLITS

    # 3. CatBoost
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(X_train, y_train, eval_set=(X_val, y_val))
    oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
    pred_cat += model_cat.predict_proba(X_test)[:, 1] / N_SPLITS

    print(f"LGB AUC: {roc_auc_score(y_val, oof_lgb[val_idx]):.5f} | "
          f"XGB AUC: {roc_auc_score(y_val, oof_xgb[val_idx]):.5f} | "
          f"CAT AUC: {roc_auc_score(y_val, oof_cat[val_idx]):.5f}")

# ================================================================
# ENSEMBLE OPTIMIZATION
# ================================================================
def objective(w):
    preds = w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat
    return -roc_auc_score(y, preds)

res = minimize(objective, [0.33, 0.33, 0.34], method='Nelder-Mead')
w = res.x / np.sum(res.x)

final_oof = w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat
final_pred = w[0]*pred_lgb + w[1]*pred_xgb + w[2]*pred_cat

print("\n" + "="*30)
print(f"FINAL OPTIMIZED AUC: {roc_auc_score(y, final_oof):.6f}")
print(f"Weights: LGB={w[0]:.2f}, XGB={w[1]:.2f}, CAT={w[2]:.2f}")
print("="*30)

# ================================================================
# SUBMISSION
# ================================================================
submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: final_pred})
submission.to_csv("submission.csv", index=False)