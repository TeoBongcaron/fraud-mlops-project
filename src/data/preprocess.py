import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the credit card fraud dataset."""
    df = pd.read_csv(csv_path)
    df = df.dropna()
    return df


def split_features_labels(df: pd.DataFrame):
    """Separate features and target."""
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


def scale_time_amount(X_train, X_val, X_test):
    """Scale ONLY Time and Amount (correct for this dataset)."""
    scaler = StandardScaler()

    # Copy to avoid modifying original
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    # Fit only on training set
    X_train_scaled[["Time", "Amount"]] = scaler.fit_transform(
        X_train[["Time", "Amount"]]
    )

    # Transform val/test
    X_val_scaled[["Time", "Amount"]] = scaler.transform(
        X_val[["Time", "Amount"]]
    )
    X_test_scaled[["Time", "Amount"]] = scaler.transform(
        X_test[["Time", "Amount"]]
    )

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train):
    """Handle class imbalance using SMOTE."""
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def preprocess_pipeline(
    csv_path: str,
    use_smote: bool = True,
    artifacts_dir: str = "artifacts"
):
    """
    Full preprocessing pipeline:
    - load data
    - split train/val/test
    - scale ONLY Time + Amount
    - apply SMOTE (optional)
    - save scaler + feature order
    """

    print("📥 Loading dataset...")
    df = load_data(csv_path)

    print("🔧 Splitting features and labels...")
    X, y = split_features_labels(df)

    print("📊 Creating train/val/test splits...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print("📏 Scaling Time and Amount...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_time_amount(
        X_train, X_val, X_test
    )

    if use_smote:
        print("⚖️ Applying SMOTE to handle class imbalance...")
        X_train_scaled, y_train = apply_smote(X_train_scaled, y_train)

    # Save scaler
    os.makedirs(artifacts_dir, exist_ok=True)
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"💾 Saved scaler to {scaler_path}")

    # Save feature order
    feature_path = os.path.join(artifacts_dir, "feature_order.pkl")
    joblib.dump(list(X.columns), feature_path)
    print(f"📄 Saved feature order to {feature_path}")

    return (
        X_train_scaled.values,
        X_val_scaled.values,
        X_test_scaled.values,
        y_train.values,
        y_val.values,
        y_test.values
    )


if __name__ == "__main__":
    preprocess_pipeline("data/creditcard.csv")
