import os
import json
import joblib
import xgboost as xgb

from src.data.preprocess import preprocess_pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def load_previous_metrics(artifacts_dir="artifacts"):
    """Load previous test metrics for comparison."""
    metrics_path = os.path.join(artifacts_dir, "metrics.json")

    if not os.path.exists(metrics_path):
        print("⚠️ No previous metrics found. Retraining will proceed without comparison.")
        return None

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return metrics.get("test", None)


def compute_metrics(model, X, y):
    """Compute evaluation metrics."""
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    return {
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, probs)
    }


def retrain_model(
    csv_path="data/creditcard.csv",
    artifacts_dir="artifacts"
):
    """
    Retraining pipeline:
    - preprocess new data
    - train new model
    - evaluate new model
    - compare with previous metrics
    - replace model if improved
    """

    print("🔄 Starting retraining pipeline...")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = preprocess_pipeline(csv_path, use_smote=True, artifacts_dir=artifacts_dir)

    print("📦 Training new XGBoost model...")
    new_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        eval_metric="logloss",
        random_state=42
    )

    new_model.fit(X_train, y_train)

    print("📊 Evaluating new model...")
    new_metrics = compute_metrics(new_model, X_test, y_test)
    print("🆕 New Model Metrics:", new_metrics)

    print("📁 Loading previous metrics...")
    prev_metrics = load_previous_metrics(artifacts_dir)

    if prev_metrics is None:
        print("📌 No previous model found — saving new model.")
        save_new_model(new_model, new_metrics, artifacts_dir)
        return new_model, new_metrics

    print("📈 Previous Model Metrics:", prev_metrics)

    # Compare based on ROC-AUC (primary metric)
    if new_metrics["roc_auc"] > prev_metrics["roc_auc"]:
        print("✅ New model performs better — updating artifacts.")
        save_new_model(new_model, new_metrics, artifacts_dir)
    else:
        print("❌ New model did NOT outperform previous model — keeping old model.")

    return new_model, new_metrics


def save_new_model(model, metrics, artifacts_dir):
    """Save model + metrics to artifacts directory."""
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, "model_xgb.pkl")
    metrics_path = os.path.join(artifacts_dir, "metrics.json")

    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump({"test": metrics}, f, indent=4)

    print(f"💾 Saved updated model to {model_path}")
    print(f"📄 Saved updated metrics to {metrics_path}")


if __name__ == "__main__":
    retrain_model()
