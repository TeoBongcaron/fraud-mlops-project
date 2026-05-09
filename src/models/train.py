import os
import json
import joblib
import xgboost as xgb
import argparse
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.data.preprocess import preprocess_pipeline
from src.models.tune import tune_xgboost


def train_xgboost(
    csv_path="data/creditcard.csv",
    artifacts_dir="artifacts",
    use_tuning=False
):
    """
    Full training pipeline:
    - preprocess data
    - (optional) hyperparameter tuning
    - train XGBoost model
    - evaluate on validation + test
    - save model + metrics
    """

    print("🚀 Starting training pipeline...")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = preprocess_pipeline(csv_path, use_smote=True, artifacts_dir=artifacts_dir)

    # 🔍 Optional hyperparameter tuning
    if use_tuning:
        print("🎯 Running hyperparameter tuning...")
        best_params = tune_xgboost(X_train, y_train)
    else:
        best_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 1,
            "eval_metric": "logloss",
            "random_state": 42
        }

    print("📦 Training XGBoost model...")
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    print("📊 Evaluating model...")

    def compute_metrics(X, y):
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        return {
            "precision": precision_score(y, preds),
            "recall": recall_score(y, preds),
            "f1": f1_score(y, preds),
            "roc_auc": roc_auc_score(y, probs)
        }

    val_metrics = compute_metrics(X_val, y_val)
    test_metrics = compute_metrics(X_test, y_test)

    print("📈 Validation Metrics:", val_metrics)
    print("🧪 Test Metrics:", test_metrics)

    # Save model
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "model_xgb.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Saved model to {model_path}")

    # Save metrics
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {"validation": val_metrics, "test": test_metrics},
            f,
            indent=4
        )
    print(f"📄 Saved metrics to {metrics_path}")

    print("✅ Training pipeline completed successfully.")

    return model, val_metrics, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tuning", action="store_true", help="Enable hyperparameter tuning")
    args = parser.parse_args()

    train_xgboost(use_tuning=args.use_tuning)


