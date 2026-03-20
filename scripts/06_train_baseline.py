import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

MODELING_READY = OUT / "modeling_ready.csv"
if not MODELING_READY.exists():
    raise FileNotFoundError("modeling_ready.csv not found. Run scripts/03_build_modeling_ready.py first.")

df = pd.read_csv(MODELING_READY, low_memory=False)
if "y_grad_outcome_high" not in df.columns:
    raise ValueError("modeling_ready.csv must include y_grad_outcome_high")

# Keep supervised dataset clean
model_df = df.dropna(subset=["y_grad_outcome_high"]).copy()
model_df["y_grad_outcome_high"] = pd.to_numeric(model_df["y_grad_outcome_high"], errors="coerce")
model_df = model_df.dropna(subset=["y_grad_outcome_high"]).copy()
model_df["y_grad_outcome_high"] = model_df["y_grad_outcome_high"].astype(int)

feature_cols = [column for column in model_df.columns if column not in {"UNITID", "y_grad_outcome_high"}]
if not feature_cols:
    raise ValueError("No feature columns available for modeling")

X = model_df[feature_cols]
y = model_df["y_grad_outcome_high"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [column for column in X_train.columns if column not in numeric_cols]

numeric_pipe_logit = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

numeric_pipe_rf = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocess_logit = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe_logit, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ],
    remainder="drop",
)

preprocess_rf = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe_rf, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ],
    remainder="drop",
)

logit_model = Pipeline(
    steps=[
        ("preprocess", preprocess_logit),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ]
)

rf_model = Pipeline(
    steps=[
        ("preprocess", preprocess_rf),
        (
            "model",
            RandomForestClassifier(
                n_estimators=400,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

logit_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)


def evaluate_model(name: str, model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series) -> dict:
    pred = model.predict(X_eval)
    proba = model.predict_proba(X_eval)[:, 1]
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_eval, pred)),
        "precision": float(precision_score(y_eval, pred, zero_division=0)),
        "recall": float(recall_score(y_eval, pred, zero_division=0)),
        "f1": float(f1_score(y_eval, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_eval, proba)),
    }

metrics_rows = [
    evaluate_model("logistic_regression", logit_model, X_test, y_test),
    evaluate_model("random_forest", rf_model, X_test, y_test),
]

metrics = {
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "target_train_distribution": {str(int(k)): int(v) for k, v in y_train.value_counts().to_dict().items()},
    "target_test_distribution": {str(int(k)): int(v) for k, v in y_test.value_counts().to_dict().items()},
    "models": metrics_rows,
}

# Feature importance exports
feature_importance_rows = []

logit_feature_names = logit_model.named_steps["preprocess"].get_feature_names_out()
logit_coef = logit_model.named_steps["model"].coef_[0]
for name, coef in zip(logit_feature_names, logit_coef):
    feature_importance_rows.append(
        {
            "model": "logistic_regression",
            "feature": str(name),
            "importance": float(abs(coef)),
            "signed_value": float(coef),
        }
    )

rf_feature_names = rf_model.named_steps["preprocess"].get_feature_names_out()
rf_importance = rf_model.named_steps["model"].feature_importances_
for name, importance in zip(rf_feature_names, rf_importance):
    feature_importance_rows.append(
        {
            "model": "random_forest",
            "feature": str(name),
            "importance": float(importance),
            "signed_value": float(importance),
        }
    )

feature_importance_df = pd.DataFrame(feature_importance_rows)
feature_importance_df = feature_importance_df.sort_values(["model", "importance"], ascending=[True, False])

metrics_path = OUT / "baseline_metrics.json"
importance_path = OUT / "baseline_feature_importance.csv"

with metrics_path.open("w", encoding="utf-8") as fp:
    json.dump(metrics, fp, indent=2)

feature_importance_df.to_csv(importance_path, index=False)

print("Wrote:")
print(f"- {metrics_path}")
print(f"- {importance_path}")
for row in metrics_rows:
    print(row)
