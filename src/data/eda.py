import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

sns.set(style="whitegrid")


def load_data(csv_path="data/creditcard.csv"):
    return pd.read_csv(csv_path)


def save_plot(fig, filename, artifacts_dir="artifacts/eda"):
    os.makedirs(artifacts_dir, exist_ok=True)
    fig.savefig(os.path.join(artifacts_dir, filename), bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    df["Class"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Class Distribution (0 = Normal, 1 = Fraud)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    save_plot(fig, "class_distribution.png")


def plot_fraud_percentage(df):
    fraud_pct = df["Class"].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Fraud %"], [fraud_pct], color="red")
    ax.set_title("Fraud Percentage")
    ax.set_ylabel("Percentage")
    save_plot(fig, "fraud_percentage.png")


def plot_amount_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Amount"], bins=50, kde=True, ax=ax)
    ax.set_title("Transaction Amount Distribution")
    save_plot(fig, "amount_distribution.png")


def plot_time_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Time"], bins=50, kde=True, ax=ax)
    ax.set_title("Transaction Time Distribution")
    save_plot(fig, "time_distribution.png")


def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(14, 10))
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    save_plot(fig, "correlation_heatmap.png")


def plot_pca(df):
    features = df.drop(columns=["Class"])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        pca_result[:, 0],
        pca_result[:, 1],
        c=df["Class"],
        cmap="coolwarm",
        alpha=0.5
    )
    ax.set_title("PCA Visualization (2 Components)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.legend(*scatter.legend_elements(), title="Class")
    save_plot(fig, "pca_scatter.png")


def run_eda(csv_path="data/creditcard.csv"):
    print("🔍 Running EDA...")
    df = load_data(csv_path)

    print("📊 Generating plots...")
    plot_class_distribution(df)
    plot_fraud_percentage(df)
    plot_amount_distribution(df)
    plot_time_distribution(df)
    plot_correlation_heatmap(df)
    plot_pca(df)

    print("📁 EDA complete. Plots saved to artifacts/eda/")


if __name__ == "__main__":
    run_eda()
