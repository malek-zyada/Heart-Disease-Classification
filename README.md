# Heart Disease Prediction — Logistic Regression

Machine learning project that predicts the presence of heart disease using
patient clinical data, built with an optimized (GridSearchCV-tuned)
Logistic Regression model.

## Dataset

[Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
(Kaggle, via `kagglehub`) — 303 patient records with 13 clinical features
and a binary target (`0` = no disease, `1` = heart disease).

## Project Pipeline

1. **Data Cleaning** — duplicate removal, missing-value check.
2. **Categorical Encoding** — one-hot encoding of categorical clinical
   codes (`cp`, `restecg`, `slope`, `thal`, `ca`) to avoid a false
   ordinal assumption.
3. **Preprocessing** — stratified train/test split (80/20) + feature
   scaling with `StandardScaler`.
4. **Hyperparameter Tuning** — `GridSearchCV` (5-fold CV) over `C`,
   `penalty`, and `solver`, optimized for **F1 score** (a medical
   diagnosis task penalizes missed positives more than plain accuracy
   captures).
5. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC,
   confusion matrix, and ROC curve.
6. **Deployment Artifacts** — the trained model, scaler, and exact
   feature-column order are all pickled together, since new inference
   data must be encoded and ordered identically.

## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 83.61% |
| Precision | 84.85% |
| Recall    | 84.85% |
| F1 Score  | 84.85% |
| ROC-AUC   | 90.26% |

**Best hyperparameters:** `C=1`, `penalty='l2'`, `solver='liblinear'`

## Project Structure

```
.
├── main.py                  # Full training & evaluation pipeline
├── requirements.txt         # Python dependencies
├── heart_disease_model.pkl  # Trained model (generated after running)
├── scaler.pkl                # Fitted StandardScaler (generated)
├── feature_columns.pkl       # Feature column order (generated)
├── feature_importance.png    # Coefficient importance plot (generated)
├── confusion_matrix.png      # Confusion matrix plot (generated)
├── roc_curve.png              # ROC curve plot (generated)
└── class_distribution.png     # Target class balance plot (generated)
```

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Run the pipeline
python main.py
```

The script automatically downloads the dataset via `kagglehub` on first run.

## Tech Stack

- Python, pandas, numpy
- scikit-learn (Logistic Regression, GridSearchCV)
- matplotlib, seaborn (visualizations)
- kagglehub (dataset access)

## License

This project is open source and available under the MIT License.
