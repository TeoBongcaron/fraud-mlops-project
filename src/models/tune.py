import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
import numpy as np


def tune_xgboost(
    X_train,
    y_train,
    n_iter=20,
    cv=3,
    random_state=42
):
    """
    Hyperparameter tuning for XGBoost using RandomizedSearchCV.
    Optimizes for F1-score, which is critical for fraud detection.
    """

    print("🎯 Starting XGBoost hyperparameter tuning...")

    # Base model
    model = xgb.XGBClassifier(
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1
    )

    # Parameter search space
    param_dist = {
        "n_estimators": [200, 300, 400, 500],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": np.linspace(0.01, 0.2, 10),
        "subsample": np.linspace(0.6, 1.0, 5),
        "colsample_bytree": np.linspace(0.6, 1.0, 5),
        "gamma": [0, 0.1, 0.2, 0.3],
        "min_child_weight": [1, 3, 5, 7]
    }

    # Optimize for F1-score
    scorer = make_scorer(f1_score)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )

    # Run search
    search.fit(X_train, y_train)

    print("\n🏆 Best parameters found:")
    print(search.best_params_)

    print("\n📈 Best F1-score:")
    print(search.best_score_)

    return search.best_params_
