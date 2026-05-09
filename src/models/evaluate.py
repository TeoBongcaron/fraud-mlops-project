import os
import json
import joblib
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from src.data.preprocess import split_features_labels, scale_features


def load_artifacts(artifacts_dir="artifacts"):
    """Load trained model and scaler."""
    model_path = os.path.join(artifacts_dir, "model_xgb.pkl")
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found. Train the model first.")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler file not found. Run preprocessing first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


def evaluate_model(
    csv_path="data/creditcard.csv",
    artifacts_dir="artifacts"
):
    """
    Evaluate the trained model on a dataset.
    Useful for:
    - CI/CD validation
    - Monitoring
    - Retraining triggers
    """

    print("📥 Loading dataset for evaluation...")
    df = pd.read_csv(csv_path)

    print("🔧 Splitting features and labels...")
    X, y = split_features_labels(df)

    print("📏 Loading scaler and model...")
    model, scaler = load_artifacts(artifacts_dir)

    print("📐 Scaling features...")
    X_scaled = scaler.transform(X)

    print("📊 Running predictions...")
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]

    metrics = {
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, probs)
    }

    print("📈 Evaluation Metrics:", metrics)

    # Save evaluation results
    eval_path = os.path.join(artifacts_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"💾 Saved evaluation results to {eval_path}")

    return metrics


if __name__ == "__main__":
    evaluate_model()
