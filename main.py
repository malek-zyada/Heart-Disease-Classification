# ============================================================
# HEART DISEASE PREDICTION - LOGISTIC REGRESSION (IMPROVED)
# ============================================================

# IMPORT LIBRARIES
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kagglehub
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score, recall_score, precision_score, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

RANDOM_STATE = 42

# ============================================================
# LOAD DATASET
# ============================================================
path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
print("Path to dataset files:", path)

data_files = os.listdir(path)
data_path = os.path.join(path, data_files[0])
df = pd.read_csv(data_path)

# Drop duplicates
before = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"Removed {before - df.shape[0]} duplicate rows. Remaining: {df.shape[0]}")

# ============================================================
# QUICK EDA (sanity checks before modeling)
# ============================================================
print("\n--- Missing values per column ---")
print(df.isnull().sum().sum(), "total missing values")

print("\n--- Target class balance ---")
print(df['target'].value_counts(normalize=True).rename("proportion"))

plt.figure(figsize=(5, 4))
sns.countplot(x='target', data=df, hue='target', palette='pastel', legend=False)
plt.title('Target Class Distribution')
plt.xlabel('0 = No Disease | 1 = Heart Disease')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300)
plt.close()

# ============================================================
# CATEGORICAL ENCODING
# ============================================================
# These columns are categorical codes, not continuous numbers.
# Leaving them as raw integers implies a false ordinal relationship
# (e.g. chest pain type 3 is not "more" than type 1).
categorical_cols = ['cp', 'restecg', 'slope', 'thal', 'ca']
categorical_cols = [c for c in categorical_cols if c in df.columns]

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ============================================================
# DATA SPLITTING & PREPROCESSING
# ============================================================
X = df_encoded.drop(columns='target')
y = df_encoded['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# HYPERPARAMETER TUNING (GridSearchCV)
# ============================================================
# Scoring changed from 'accuracy' to 'f1': for a medical diagnosis
# task, missing a true positive (false negative) is more costly
# than raw accuracy reflects, and f1 balances precision/recall.
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

log_reg_base = LogisticRegression(max_iter=10000, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator=log_reg_base,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_

print("\n--- Hyperparameter Tuning Results ---")
print(f"Best Parameters found: {grid_search.best_params_}")
print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

# ============================================================
# PREDICTIONS & FULL EVALUATION
# ============================================================
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n--- Test Set Performance ---")
print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}  (critical: fraction of real patients correctly caught)")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Heart Disease']))

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
coefficients = best_model.coef_[0]
df_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients
})
df_importance['Absolute_Impact'] = df_importance['Coefficient'].abs()
df_importance = df_importance.sort_values(by='Absolute_Impact', ascending=False)

plt.figure(figsize=(10, 7))
sns.barplot(
    x='Coefficient',
    y='Feature',
    data=df_importance,
    palette='coolwarm',
    hue='Feature',
    legend=False
)
plt.title('Logistic Regression Optimized Feature Importance', fontsize=14)
plt.xlabel('Coefficient Value (Direction & Strength)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()

# ============================================================
# CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Heart Disease'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Optimized Logistic Regression', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()

# ============================================================
# ROC CURVE
# ============================================================
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Optimized Logistic Regression')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300)
plt.close()

# ============================================================
# SAVE MODEL, SCALER & FEATURE ORDER (FOR DEPLOYMENT)
# ============================================================
# Saving the trained column order alongside the model/scaler is
# essential now that we one-hot encode categorical columns —
# any new inference data must be encoded and reordered identically.
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('heart_disease_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open('feature_columns.pkl', 'wb') as columns_file:
    pickle.dump(list(X.columns), columns_file)

print("\nSUCCESS: model, scaler, and feature_columns saved for deployment.")
