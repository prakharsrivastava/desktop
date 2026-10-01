# Stellar Classification — Step-wise Code Guide
# Problem: GALAXY vs QSO vs STAR ko kaunsi cheez alag karti hai, aur predict kar sakte hain?
# Har step: KYUN kar rahe ho (1 line) + CODE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              classification_report, confusion_matrix)
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

train = pd.read_csv('your_data.csv')   # columns: u, g, r, i, z, redshift, class


# ============================================================
# STEP 1 — EDA: Histograms
# KYUN: shape dekhna hai (skew? scaling chahiye?)
# ============================================================
train[['u', 'g', 'r', 'i', 'z', 'redshift']].hist(figsize=(15, 8), bins=50)
plt.tight_layout(); plt.show()
# DEKHO: redshift right-skewed hoga, baaki ~normal -> scaling zaroori


# ============================================================
# STEP 2 — Correlation heatmap (raw bands)
# KYUN: dekhna hai bands aapas mein kitni redundant hain
# ============================================================
sns.heatmap(train[['u', 'g', 'r', 'i', 'z', 'redshift']].corr(),
           annot=True, cmap='coolwarm')
plt.show()
# DEKHO: bands aapas mein ~0.9+ correlated (brightness shared), redshift alag


# ============================================================
# STEP 3 — Colour indices banao (sabse important decision)
# KYUN: brightness cancel karke sirf spectral shape rakhna hai
# ============================================================
train['u_g'] = train['u'] - train['g']
train['g_r'] = train['g'] - train['r']
train['r_i'] = train['r'] - train['i']
train['i_z'] = train['i'] - train['z']

color_cols = ['u_g', 'g_r', 'r_i', 'i_z', 'redshift']

# heatmap dobara - ab correlation kam honi chahiye
sns.heatmap(train[color_cols].corr(), annot=True, cmap='coolwarm')
plt.show()


# ============================================================
# STEP 4 — Pairplot
# KYUN: dekhna hai kaunsa single feature classes ko alag karta hai
# ============================================================
sns.pairplot(train.sample(15000), vars=color_cols, hue='class')
plt.show()
# DEKHO: redshift wale boxes mein QSO saaf alag, baaki overlap


# ============================================================
# STEP 5 — PCA: Standardize -> Fit -> Explained variance
# KYUN: 5 features ko kam dimensions mein todna hai, info bachate hue
# ============================================================
pca_features = color_cols
X = StandardScaler().fit_transform(train[pca_features])

pca = PCA()
pca.fit(X)
print(pca.explained_variance_ratio_)
# YEH PER-COMPONENT HAI, PER-FEATURE NAHI! (0.40, 0.20, 0.16, 0.15, 0.08)
# PC1+PC2 = ~60% -> 2 components mein zyaadatar info


# ============================================================
# STEP 6a — Loadings table (har PC ka matlab kya)
# KYUN: PC1/PC2 physically kya represent karte hain
# ============================================================
loadings = pd.DataFrame(
    pca.components_.T,
    index=pca_features,
    columns=[f'PC{i+1}' for i in range(len(pca_features))]
)
print(loadings)
# DEKHO: har column mein sabse bade 1-2 number -> wahi PC ka matlab


# ============================================================
# STEP 6b — PC1 vs PC2 scatter (scores)
# KYUN: 2D mein dekhna hai classes alag hoti hain ya nahi
# ============================================================
pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X)

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=train['class'], alpha=0.3)
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.show()

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
# (5-component version baad ke steps ke liye chahiye)
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X)
pca_df = pd.DataFrame(X_pca_full, columns=[f'PC{i+1}' for i in range(5)])
pca_df['class'] = train['class'].values


# ============================================================
# STEP 6c — Biplot (points + feature arrows ek saath)
# KYUN: feature aur class ko jodna hai ("redshift -> QSO")
# ============================================================
plt.figure(figsize=(10, 8))
sample_df = pca_df.groupby('class', group_keys=False).sample(1000, random_state=42)
for cls in sample_df['class'].unique():
    s = sample_df[sample_df['class'] == cls]
    plt.scatter(s['PC1'], s['PC2'], alpha=0.3, s=10, label=cls)

scale = 3
for feature in pca_features:
    x, y = loadings.loc[feature, 'PC1'], loadings.loc[feature, 'PC2']
    plt.arrow(0, 0, x*scale, y*scale, color='black', head_width=0.05)
    plt.text(x*scale*1.15, y*scale*1.15, feature, fontweight='bold')
plt.legend(); plt.show()


# ============================================================
# STEP 7a — Class-wise mean/std (numeric proof)
# KYUN: "lagta hai PC1 best hai" ko number se confirm karna
# ============================================================
print(pca_df.groupby('class').agg(['mean', 'std']))
# DEKHO: kis PC par classes ke means sabse door hain


# ============================================================
# STEP 7b — Violin / Box plots
# KYUN: poora distribution dekhna, sirf mean nahi
# ============================================================
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for i in range(5):
    sns.violinplot(data=pca_df, x='class', y=f'PC{i+1}', hue='class', ax=axes[i])
plt.tight_layout(); plt.show()


# ============================================================
# STEP 7c — ANOVA F-test (final numeric ranking)
# KYUN: ek number mein "kaunsa PC sabse strong separator" batana
# F = between-class variance / within-class variance
# ============================================================
F, p = f_classif(pca_df[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']], pca_df['class'])
ftest_result = pd.DataFrame({
    'PC': ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], 'F_score': F, 'p_value': p
}).sort_values('F_score', ascending=False)
print(ftest_result)
# RANKING: PC1 >> PC2 > PC4 > PC3 > PC5


# ============================================================
# STEP 8a — Minimum Spanning Tree (local neighbourhood check)
# KYUN: dekhna hai paas wale objects same class hote hain ya nahi
# ============================================================
sample = pca_df.groupby('class').sample(150, random_state=42).reset_index(drop=True)
coords = sample[['PC1', 'PC2']].values
dist_matrix = squareform(pdist(coords, metric='euclidean'))
mst = minimum_spanning_tree(dist_matrix).toarray()

same_class, different_class = 0, 0
classes = sample['class'].values
for i in range(len(coords)):
    for j in range(len(coords)):
        if mst[i, j] > 0:
            if classes[i] == classes[j]:
                same_class += 1
            else:
                different_class += 1
ratio = same_class / (same_class + different_class)
print(f"Same-class ratio: {ratio:.3f}")   # ~0.80


# ============================================================
# STEP 8b — MST repeat 200-1000x (stability check) + dimensions test
# KYUN: ek-baar ka result luck na ho, aur dekhna 4 vs 5 PCs farak
# ============================================================
feature_sets_to_test = [['PC1'], ['PC1', 'PC2'], ['PC1', 'PC2', 'PC3'],
                        ['PC1', 'PC2', 'PC3', 'PC4'],
                        ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]
mst_results = []
for cols in feature_sets_to_test:
    ratios = []
    for seed in range(200):
        s = pca_df.groupby('class').sample(100, random_state=seed).reset_index(drop=True)
        c = s[cols].values
        m = minimum_spanning_tree(squareform(pdist(c))).toarray()
        same, total = 0, 0
        for i in range(len(c)):
            for j in range(len(c)):
                if m[i, j] > 0:
                    total += 1
                    if s['class'].values[i] == s['class'].values[j]:
                        same += 1
        ratios.append(same / total)
    mst_results.append({'cols': '+'.join(cols), 'mean_ratio': np.mean(ratios)})
print(pd.DataFrame(mst_results))
# DEKHO: 4 PCs ke baad plateau, PC5 se koi sudhaar nahi


# ============================================================
# STEP 9 — KNN Classifier (MST ki prediction test karo)
# KYUN: "paas wale same class" -> isliye KNN kaam karega
# ============================================================
feature_sets = {
    'PC1_PC2': ['PC1', 'PC2'],
    'PC1_PC2_PC3_PC4': ['PC1', 'PC2', 'PC3', 'PC4'],
    'ALL_PC': ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'],
}
param_grid = [(k, w) for k in [1, 3, 5, 7, 11, 21, 31] for w in ['uniform', 'distance']]

knn_results = []
for name, cols in feature_sets.items():
    X_feat, y_feat = pca_df[cols], pca_df['class']
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y_feat, test_size=0.2, random_state=42, stratify=y_feat)
    for k, w in param_grid:
        model = KNeighborsClassifier(n_neighbors=k, weights=w, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        knn_results.append({
            'features': name, 'k': k, 'weights': w,
            'accuracy': accuracy_score(y_test, preds),
            'balanced_accuracy': balanced_accuracy_score(y_test, preds)
        })
knn_results = pd.DataFrame(knn_results).sort_values('accuracy', ascending=False)
print(knn_results.head(10))
# BEST: ALL_PC, k=21, distance -> ~93.5% accuracy


# ============================================================
# STEP 10 — Error analysis (model kahan fail hota hai)
# KYUN: galtiyaan random hain ya structural, yeh dikhana hai
# ============================================================
# best model dobara fit karo (ALL_PC, k=21, distance) final test ke liye
best_cols = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
X_feat, y_feat = pca_df[best_cols], pca_df['class']
X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y_feat, test_size=0.2, random_state=42, stratify=y_feat)
best_model = KNeighborsClassifier(n_neighbors=21, weights='distance', n_jobs=-1)
best_model.fit(X_train, y_train)
preds = best_model.predict(X_test)

print(classification_report(y_test, preds))
# DEKHO: STAR ka f1 sabse kam (~0.84), GALAXY sabse zyada (~0.96)

cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=sorted(y_test.unique()), yticklabels=sorted(y_test.unique()))
plt.xlabel('Predicted'); plt.ylabel('True'); plt.show()
# DEKHO: STAR<->GALAXY sabse common confusion

# correct vs wrong predictions ke PC-means compare karo
analysis = X_test.copy()
analysis['true_class'] = y_test.values
analysis['pred_class'] = preds
analysis['correct'] = analysis['true_class'] == analysis['pred_class']

print("Correct mean:\n", analysis[analysis.correct][best_cols].mean())
print("Wrong mean:\n", analysis[~analysis.correct][best_cols].mean())
# DEKHO: wrong predictions ka mean 0 se door (overlap zone mein)

# pc_norm = origin se distance -> errors kahan concentrated hain
analysis['pc_norm'] = np.sqrt(sum(analysis[c]**2 for c in best_cols))
print(analysis.groupby('correct')['pc_norm'].describe())
# SURPRISE: galat predictions origin ke PAAS hote hain (crowded overlap zone)


# ============================================================
# FINAL ANSWER (yeh bolna hai interview mein)
# ============================================================
# 1. Colour indices banaye brightness cancel karne ke liye
# 2. PCA se 5D -> 2D, PC1+PC2 mein ~60-80% structure
# 3. F-test/MST se NUMERICALLY confirm: PC1 sabse strong separator
# 4. KNN se 93.5% accuracy -> separation REAL hai, sirf visual nahi
# 5. Errors random nahi -- classes ke genuine overlap zone mein hote hain
#    (STAR intermediate class hone ki wajah se sabse zyada confuse)
