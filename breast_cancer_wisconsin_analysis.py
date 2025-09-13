from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, classification_report, roc_curve
)

try:
    import umap
    _has_umap = True
except Exception:
    _has_umap = False

RANDOM_STATE = 42

# =========================
# Load the data
# =========================

local_path = './breast-cancer-wisconsin.data'


def load_data(local_path):
    cols = [
        'id', 'clump_thickness', 'uniformity_of_cell_size', 'uniformity_of_cell_shape',
        'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin',
        'normal_nucleoli', 'mitoses', 'class'
    ]

    if Path(local_path).exists():
        df = pd.read_csv(local_path, header=None, names=cols)
    else:
        raise FileNotFoundError(f"File not found: {local_path}")

    return df

print('Loading data...')
df = load_data(local_path)
print(f'Data shape: {df.shape}')

# Quick look
print(df.head())

# =========================
# Section 1: Basic EDA (Raw Data)
# =========================
df.replace('?', np.nan, inplace=True)

# Data types
print('\nColumn dtypes and missing counts:')
print(df.info())
print('\nMissing value counts:')
print(df.isna().sum())

# Convert numeric columns
num_cols = [c for c in df.columns if c not in ('id','class')]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Summary statistics
summary = df[num_cols].describe().T
summary['IQR'] = df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25)
print('\nSummary statistics:\n', summary)

# Histograms
for c in num_cols:
    plt.figure(figsize=(6,3))
    sns.histplot(df[c].dropna(), kde=False)
    plt.title(f'Histogram of {c}')
    plt.tight_layout()
    plt.show()

# Boxplots comparing benign vs malignant
# In this dataset class: 2 = benign, 4 = malignant
df['class_label'] = df['class'].map({2: 'benign', 4: 'malignant'})
for c in num_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(x='class_label', y=c, data=df)
    plt.title(f'{c} by class')
    plt.tight_layout()
    plt.show()

# Scatter plot between top 2 features by correlation
corr = df[num_cols].corr().abs().unstack().sort_values(ascending=False)
# get top pair (exclude self correlations)
pairs = [(i,j) for i,j in corr.index if i!=j]
if pairs:
    top_pair = pairs[0]
    print('\nTop correlated pair:', top_pair)
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=top_pair[0], y=top_pair[1], hue='class_label', data=df)
    plt.title(f'Scatter: {top_pair[0]} vs {top_pair[1]}')
    plt.tight_layout()
    plt.show()

# =========================
# Section 2: Preprocessing
# =========================
# Handle missing values: bare_nuclei may have missing values
print('\nRows with missing values:')
print(df[df.isna().any(axis=1)])

# Simple strategy: impute missing numeric with median
imputer = SimpleImputer(strategy='median')
df[num_cols] = imputer.fit_transform(df[num_cols])

# Re-check missing
print('\nMissing after imputation:')
print(df[num_cols].isna().sum())

# Normalize continuous features
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

# Prepare X and y
X = df_scaled[num_cols].copy()
y = df['class'].map({2:0,4:1})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print('\nTrain shape:', X_train.shape, 'Test shape:', X_test.shape)

# =========================
# Section 3: Feature Engineering & Reduction
# =========================
# Example derived features
X_train_fe = X_train.copy()
X_test_fe = X_test.copy()
X_train_fe['size_shape_ratio'] = X_train_fe['uniformity_of_cell_size'] / (X_train_fe['uniformity_of_cell_shape'] + 1e-6)
X_test_fe['size_shape_ratio'] = X_test_fe['uniformity_of_cell_size'] / (X_test_fe['uniformity_of_cell_shape'] + 1e-6)

# Recursive Feature Elimination with Decision Tree
dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
rfe = RFE(dt, n_features_to_select=5)
rfe.fit(X_train_fe, y_train)
selected = [f for f, s in zip(X_train_fe.columns, rfe.support_) if s]
print('\nSelected features by RFE:', selected)

# PCA
pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_fe)
X_test_pca = pca.transform(X_test_fe)
print('\nPCA components:', pca.n_components_, 'explained variance ratio sum:', pca.explained_variance_ratio_.sum())

# t-SNE (for visualization only; use small sample for speed)
sample_frac = 0.3
X_vis_sample = X_train_fe.sample(frac=sample_frac, random_state=RANDOM_STATE)
y_vis_sample = y_train.loc[X_vis_sample.index]
tsne = TSNE(n_components=2, random_state=RANDOM_STATE, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_vis_sample)
plt.figure(figsize=(6,4))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_vis_sample, cmap='viridis', alpha=0.7)
plt.title('t-SNE visualization (sample)')
plt.colorbar()
plt.show()

# UMAP (if available)
if _has_umap:
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
    X_umap = reducer.fit_transform(X_vis_sample)
    plt.figure(figsize=(6,4))
    plt.scatter(X_umap[:,0], X_umap[:,1], c=y_vis_sample, cmap='viridis', alpha=0.7)
    plt.title('UMAP visualization (sample)')
    plt.colorbar()
    plt.show()
else:
    print('\nUMAP not installed. To install: pip install umap-learn')

# =========================
# Section 4: Second EDA (Post-Processing)
# =========================
# Variance explained by PCA (already printed). Plot it.
plt.figure(figsize=(6,3))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA explained variance')
plt.grid(True)
plt.show()

# Feature importance from Decision Tree trained on full features
dt_full = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_full.fit(X_train_fe, y_train)
fi = pd.Series(dt_full.feature_importances_, index=X_train_fe.columns).sort_values(ascending=False)
print('\nDecision Tree feature importances:\n', fi)
plt.figure(figsize=(6,3))
fi.plot(kind='bar')
plt.title('Feature importances (Decision Tree)')
plt.tight_layout()
plt.show()

# =========================
# Section 5: ML Modeling
# =========================
# Prepare final X (use selected features from RFE)
X_train_final = X_train_fe[selected]
X_test_final = X_test_fe[selected]

models = {
    'NaiveBayes': GaussianNB(),
    'k-NN': KNeighborsClassifier(n_neighbors=5)
}

results = {}
for name, model in models.items():
    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)
    y_proba = model.predict_proba(X_test_final)[:,1] if hasattr(model, 'predict_proba') else None
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    results[name] = {'accuracy': acc, 'recall': rec, 'precision': prec, 'roc_auc': roc}
    print(f"\nModel: {name}\nAccuracy: {acc:.4f}\nRecall (sensitivity): {rec:.4f}\nPrecision: {prec:.4f}\nROC-AUC: {roc if roc is None else f'{roc:.4f}'}")
    print('\nClassification report:\n', classification_report(y_test, y_pred))

    # ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc:.3f})')
        plt.plot([0,1],[0,1],'--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend()
        plt.grid(True)
        plt.show()

# =========================
# Section 6: Evaluation
# =========================
# Compare models
print('\nSummary of results:')
print(pd.DataFrame(results).T)

# Discuss implications (print notes)
print('\nNotes:')
print('- False negatives (missed malignancies) are the most critical errors in this domain; prioritize high recall/sensitivity.')
print('- Depending on application, you may choose to adjust decision thresholds to increase sensitivity at cost of specificity.')
print('- Consider using cross-validation and calibration curves for probabilistic models.')

# =========================
# Section 7: Interpretation
# =========================
from sklearn.calibration import calibration_curve
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        prob_pos = model.predict_proba(X_test_final)[:,1]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)
        plt.figure(figsize=(6,4))
        plt.plot(mean_predicted_value, fraction_of_positives, 's-')
        plt.plot([0,1],[0,1], '--')
        plt.xlabel('Mean predicted value')
        plt.ylabel('Fraction of positives')
        plt.title(f'Calibration curve - {name}')
        plt.grid(True)
        plt.show()

print('\nScript finished.\n')
