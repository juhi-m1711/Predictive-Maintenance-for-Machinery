"""
train_models.py
----------------
Replicates the exact preprocessing + training pipeline from
Predictive_Maintenance_for_Machinery.ipynb and saves all artifacts
needed by app.py (models, scalers, label encoders, metrics).

Run this once: python train_models.py
It produces a single file: model_artifacts.pkl
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

print("Loading dataset...")
df = pd.read_csv("maintenance_dataset.csv")
df = df.drop(columns=["UDI", "Product ID"])

numeric_cols = ['Air temperature [K]', 'Process temperature [K]',
                 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# ---- Encode Type ----
le = LabelEncoder()
df["Type_Encoded"] = le.fit_transform(df["Type"])  # H=0, L=1, M=2
type_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Type mapping:", type_mapping)

feature_cols = ["Type_Encoded"] + numeric_cols

# ============================================================
# STAGE 1: BINARY CLASSIFICATION (Machine failure: yes/no)
# ============================================================
X = df[feature_cols]
y = df["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

print("\nTraining binary models...")

binary_results = {}

def eval_binary(name, model):
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    binary_results[name] = {
        "metrics": metrics,
        "confusion_matrix": cm,
        "roc": (fpr, tpr),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
    print(f"  {name}: {metrics}")

log_reg = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
log_reg.fit(X_train_sm, y_train_sm)
eval_binary("Logistic Regression", log_reg)

dt = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8)
dt.fit(X_train_sm, y_train_sm)
eval_binary("Decision Tree", dt)

rf = RandomForestClassifier(random_state=RANDOM_STATE, max_depth=10, n_estimators=200)
rf.fit(X_train_sm, y_train_sm)
eval_binary("Random Forest", rf)

# Best model = Random Forest (per notebook's stated rationale: best recall/F1 balance)
best_binary_model_name = "Random Forest"
best_binary_model = rf

importances_binary = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

# ============================================================
# STAGE 2: MULTICLASS CLASSIFICATION (failure type)
# ============================================================
print("\nTraining multiclass model...")

failed_df = df[df["Machine failure"] == 1].copy()

def get_failure_type(row):
    for ft in failure_types:
        if row[ft] == 1:
            return ft
    return "Unknown"

failed_df["failure_type"] = failed_df.apply(get_failure_type, axis=1)
failed_df = failed_df[failed_df["failure_type"] != "Unknown"]

X_multi = failed_df[feature_cols]
y_multi = failed_df["failure_type"]

le_multi = LabelEncoder()
y_multi_encoded = le_multi.fit_transform(y_multi)
multi_class_mapping = dict(zip(range(len(le_multi.classes_)), le_multi.classes_))
print("Multiclass mapping:", multi_class_mapping)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi_encoded, test_size=0.25, random_state=RANDOM_STATE, stratify=y_multi_encoded
)

scaler_m = StandardScaler()
X_train_m_scaled = scaler_m.fit_transform(X_train_m)
X_test_m_scaled = scaler_m.transform(X_test_m)

model_m = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200, max_depth=8, class_weight="balanced")
model_m.fit(X_train_m_scaled, y_train_m)

y_pred_m = model_m.predict(X_test_m_scaled)
multi_report = classification_report(y_test_m, y_pred_m, target_names=le_multi.classes_, output_dict=True)
multi_cm = confusion_matrix(y_test_m, y_pred_m)
importances_multi = pd.Series(model_m.feature_importances_, index=feature_cols).sort_values(ascending=False)

print("Multiclass model trained.")

# ============================================================
# FAILURE TYPE FULL NAMES (for friendly display in the UI)
# ============================================================
failure_full_names = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}

# ============================================================
# SAVE EVERYTHING
# ============================================================
artifacts = {
    "feature_cols": feature_cols,
    "numeric_cols": numeric_cols,
    "type_mapping": type_mapping,          # {'H':0, 'L':1, 'M':2}

    # Binary stage
    "binary_scaler": scaler,
    "binary_models": {
        "Logistic Regression": log_reg,
        "Decision Tree": dt,
        "Random Forest": rf,
    },
    "best_binary_model_name": best_binary_model_name,
    "binary_results": binary_results,
    "binary_feature_importance": importances_binary,
    "binary_test_failure_rate": float(y_test.mean()),
    "binary_train_failure_rate": float(y_train.mean()),

    # Multiclass stage
    "multi_scaler": scaler_m,
    "multi_model": model_m,
    "multi_class_mapping": multi_class_mapping,   # {0:'HDF',1:'OSF',...}
    "multi_report": multi_report,
    "multi_confusion_matrix": multi_cm,
    "multi_feature_importance": importances_multi,
    "failure_full_names": failure_full_names,

    # Dataset-level stats for the UI (ranges to set sensible slider bounds)
    "data_ranges": {
        col: (float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        for col in numeric_cols
    },
    "overall_failure_rate": float(df["Machine failure"].mean()),
    "n_rows": int(len(df)),
}

with open("model_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("\nSaved all artifacts to model_artifacts.pkl")
print("Done.")