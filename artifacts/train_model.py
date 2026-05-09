import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import os

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("creditcard.csv")

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train model
# -----------------------------
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

model.fit(X_scaled, y)

# -----------------------------
# Save artifacts
# -----------------------------
os.makedirs("artifacts", exist_ok=True)

joblib.dump(model, "artifacts/model_xgb.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")

print("Training complete. Artifacts saved in /artifacts/")
