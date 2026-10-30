from pathlib import Path
import json
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from features import build_feature_union

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "emails.csv"
MODEL_PATH = ROOT / "phishing_model.joblib"
METRICS_PATH = ROOT / "metrics.json"
MATRIX_PATH = ROOT / "confusion_matrix.png"

def normalize_label(value):
    value = str(value).strip().lower()
    if value in {"phishing", "phish", "1", "spam", "malicious"}:
        return "Phishing"
    if value in {"safe", "legitimate", "legit", "ham", "0", "benign"}:
        return "Safe"
    raise ValueError(f"Unrecognized label: {value!r}. Use Phishing or Safe.")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    # Accept common column names from downloaded datasets.
    text_col = next((c for c in ["email_text", "text", "body", "message", "content"] if c in df.columns), None)
    label_col = next((c for c in ["label", "class", "category", "type"] if c in df.columns), None)
    if not text_col or not label_col:
        raise ValueError(
            "CSV must contain an email text column (email_text/text/body/message/content) "
            "and a label column (label/class/category/type)."
        )

    data = df[[text_col, label_col]].dropna().copy()
    data[text_col] = data[text_col].astype(str).str.strip()
    data["label_clean"] = data[label_col].apply(normalize_label)
    data = data[data[text_col].str.len() > 0]

    if data["label_clean"].nunique() != 2:
        raise ValueError("Dataset must contain both Phishing and Safe examples.")
    if data["label_clean"].value_counts().min() < 2:
        raise ValueError("Need at least 2 examples of each class for a train/test split.")

    X_train, X_test, y_train, y_test = train_test_split(
        data[text_col], data["label_clean"],
        test_size=0.25, random_state=42, stratify=data["label_clean"]
    )

    model = Pipeline([
        ("features", build_feature_union()),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, predictions))
    labels = ["Phishing", "Safe"]
    cm = confusion_matrix(y_test, predictions, labels=labels)

    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    joblib.dump(model, MODEL_PATH)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Phishing Email Detection - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(MATRIX_PATH, dpi=160)
    plt.close(fig)

    metrics = {
        "accuracy": accuracy,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "total_rows": int(len(data)),
        "class_counts": {k: int(v) for k, v in data["label_clean"].value_counts().items()},
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nMODEL TRAINING COMPLETE")
    print(f"Rows used: {len(data)} | Training: {len(X_train)} | Testing: {len(X_test)}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved confusion matrix: {MATRIX_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print("\nImportant: test accuracy is meaningful only when the CSV contains a sufficiently large, representative, correctly labelled dataset.")

if __name__ == "__main__":
    main()
