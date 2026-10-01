# Databricks notebook source
# DBTITLE 1,Install Dependencies
# MAGIC %pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# MAGIC %pip install fastai scikit-learn

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports & Global Setup

# COMMAND ----------

import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
from fastai.tabular.all import *
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
set_seed(42, reproducible=True)

TRAIN_PATH = "/Workspace/Users/prakhar1207srivastava@gmail.com/Drafts/train.csv"
TEST_PATH = "/Workspace/Users/prakhar1207srivastava@gmail.com/Drafts/test.csv"
SAMPLE_SUB_PATH = "/Workspace/Users/prakhar1207srivastava@gmail.com/Drafts/sample_submission.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
submission = pd.read_csv(SAMPLE_SUB_PATH)


# COMMAND ----------


TARGET = "addicted_label"
ID_COL = "id" if "id" in train_df.columns else None

initial_features = [c for c in train_df.columns if c not in [TARGET, ID_COL]]

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering

# COMMAND ----------



# COMMAND ----------

def engineer_features(df):
    
    df = df.copy()

    df["missing_count"] = df[initial_features].apply(
        lambda row: row.astype(str).isin(["nan", "None", "<NA>"]).sum(), axis=1
    ).astype(float)
    
    def to_float(col_name):
        if col_name in df.columns:
            return pd.to_numeric(df[col_name].astype(str), errors="coerce")
        return pd.Series(np.nan, index=df.index)
    
    s_media = to_float("social_media_hours")
    gaming = to_float("gaming_hours")
    work = to_float("work_study_hours")
    daily_screen = to_float("daily_screen_time_hours")
    sleep = to_float("sleep_hours")
    weekend_screen = to_float("weekend_screen_time")
    app_opens = to_float("app_opens_per_day")
    notifications = to_float("notifications_per_day")

    df["total_breakdown_hours"] = s_media.fillna(0.0) + gaming.fillna(0.0) + work.fillna(0.0)
    df["social_ratio"] = s_media / (daily_screen + 1e-5)
    df["gaming_ratio"] = gaming / (daily_screen + 1e-5)
    df["work_ratio"] = work / (daily_screen + 1e-5)
    df["unaccounted_screen_time"] = daily_screen - df["total_breakdown_hours"]
    df["screen_to_sleep_ratio"] = daily_screen / (sleep + 1e-5)
    df["weekend_vs_daily_ratio"] = weekend_screen / (daily_screen + 1e-5)
    df["app_opens_per_hour"] = app_opens / (daily_screen + 1e-5)
    df["notifications_per_hour"] = notifications / (daily_screen + 1e-5)
    
    engineered_cont_cols = [
        "total_breakdown_hours", "social_ratio", "gaming_ratio", "work_ratio",
        "unaccounted_screen_time", "screen_to_sleep_ratio", "weekend_vs_daily_ratio",
        "app_opens_per_hour", "notifications_per_hour"
    ]
    df[engineered_cont_cols] = df[engineered_cont_cols].fillna(0.0)
    
    return df

train_df = engineer_features(train_df)
test_df = engineer_features(test_df)


# COMMAND ----------

display(test_df)

# COMMAND ----------

display(train_df)

# COMMAND ----------

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df[TARGET])):
    print(fold)
    print(len(train_idx))
    print(len(val_idx))
    #display(train_df.iloc[train_idx])
    global_mean = float(train_df.iloc[train_idx][TARGET].mean())
    print(global_mean)

# COMMAND ----------

train_encoded = train_df.copy()
test_encoded = test_df.copy()

for col in initial_features:
    train_encoded[f"{col}_te"] = 0.0
    train_encoded[f"{col}_freq"] = 0.0
    test_encoded[f"{col}_freq"] = 0.0
    test_encoded[f"{col}_te"] = 0.0

display(train_encoded)

# COMMAND ----------

for col in initial_features:
            counts = train_df.iloc[train_idx][col].value_counts()
            print(counts)

# COMMAND ----------


def apply_encodings_cv(train_df, test_df, cat_cols, target_col, n_splits=5, smoothing=10, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    for col in cat_cols:
        train_encoded[f"{col}_te"] = 0.0
        train_encoded[f"{col}_freq"] = 0.0
        test_encoded[f"{col}_freq"] = 0.0
        test_encoded[f"{col}_te"] = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df[target_col])):
        tr_subset = train_df.iloc[train_idx]
        val_subset = train_df.iloc[val_idx]
        global_mean = float(tr_subset[target_col].mean())
        
        for col in cat_cols:
            counts = tr_subset[col].value_counts()
            
            val_freq = val_subset[col].map(counts).astype(float).fillna(0.0).values
            train_encoded.iloc[val_idx, train_encoded.columns.get_loc(f"{col}_freq")] = val_freq
            
            test_freq = test_df[col].map(counts).astype(float).fillna(0.0).values
            test_encoded[f"{col}_freq"] += test_freq / n_splits
            
            stats = tr_subset.groupby(col, observed=False)[target_col].agg(["count", "mean"])
            smoothed_te = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
            
            val_te = val_subset[col].map(smoothed_te).astype(float).fillna(global_mean).values
            train_encoded.iloc[val_idx, train_encoded.columns.get_loc(f"{col}_te")] = val_te
            
            test_te = test_df[col].map(smoothed_te).astype(float).fillna(global_mean).values
            test_encoded[f"{col}_te"] += test_te / n_splits

    return train_encoded, test_encoded, skf

train_df, test_df, skf = apply_encodings_cv(
    train_df, test_df, cat_cols=initial_features, target_col=TARGET
)

# COMMAND ----------

display(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Training

# COMMAND ----------

cat_names = [c for c in initial_features if train_df[c].dtype == 'object' or train_df[c].dtype.name == 'category']
cont_names = [c for c in train_df.columns if c not in cat_names + [TARGET, ID_COL]]

procs = [Categorify, FillMissing, Normalize]

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df[TARGET])):
    
    print(f"\n>>> FOLD {fold + 1} / 5 <<<")

    splits = (list(train_idx), list(val_idx))

    to = TabularPandas(
        train_df,
        procs=procs,
        cat_names=cat_names,
        cont_names=cont_names,
        y_names=TARGET,
        y_block=CategoryBlock(),
        splits=splits
    )
    
    dls = to.dataloaders(bs=512)
    
    config = tabular_config(
        ps=0.15,
        embed_p=0.05,
        use_bn=True,
        act_cls=Mish()
    )

    roc_auc_metric = RocAucBinary()
    roc_auc_metric.name = 'roc_auc'
    
    learn = tabular_learner(
        dls,
        layers=[256, 256, 128],
        emb_szs={c: 16 for c in cat_names},
        config=config,
        wd=0.035,
        n_out=2,
        opt_func=Adam,
        metrics=roc_auc_metric
    )

    cbs = [
        SaveModelCallback(monitor='roc_auc', comp=np.greater, fname=f'best_model_fold_{fold}')
    ]

    learn.fit_one_cycle(n_epoch=22, lr_max=1.5e-3, pct_start=0.2, cbs=cbs)

    learn.load(f'best_model_fold_{fold}')

    val_dl = dls.valid
    val_preds_res, _ = learn.get_preds(dl=val_dl)
    val_probs = val_preds_res[:, 1].numpy()
    oof_preds[val_idx] = val_probs

    fold_auc = roc_auc_score(train_df.iloc[val_idx][TARGET], val_probs)
    print(f"--> Fold {fold + 1} | ROC-AUC: {fold_auc:.5f}")

    tst_dl = dls.test_dl(test_df)
    tst_preds_res, _ = learn.get_preds(dl=tst_dl)
    test_preds += tst_preds_res[:, 1].numpy() / 5

overall_oof_auc = roc_auc_score(train_df[TARGET], oof_preds)
print("\n" + "=" * 60)
print(f"OVERALL OOF ROC-AUC SCORE: {overall_oof_auc:.5f}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC # Submission

# COMMAND ----------

oof_df = train_df[[ID_COL]].copy() if ID_COL else pd.DataFrame(index=train_df.index)
oof_df.loc[:, "oof_pred"] = oof_preds
oof_df.loc[:, TARGET] = train_df[TARGET]
oof_df.to_csv("oof.csv", index=False)

submission[TARGET] = test_preds
submission.to_csv("submission.csv", index=False)
submission.head()