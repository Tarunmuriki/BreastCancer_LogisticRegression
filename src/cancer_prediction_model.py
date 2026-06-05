import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = pd.Series(cancer.target)
    return X, y


def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)
    return model


def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "r--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_auc_curve.png"))
    plt.close()


def evaluate_model(y_test, y_pred, y_prob):
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["Malignant", "Benign"])
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    print("\nConfusion Matrix:\n", cm)
    print("\nPrecision:", precision)
    print("Recall:", recall)
    print("ROC AUC Score:", roc_auc)
    print("\nClassification Report:\n", report)

    plot_confusion_matrix(cm)
    plot_roc_curve(fpr, tpr, roc_auc)

    custom_threshold = 0.3
    y_custom = (y_prob >= custom_threshold).astype(int)
    print(f"\nThreshold tuning at {custom_threshold:.2f}:")
    print(
        classification_report(
            y_test,
            y_custom,
            target_names=["Malignant", "Benign"],
        )
    )


if __name__ == "__main__":
    X, y = load_data()
    print("Dataset Shape:", X.shape)
    print("Target Distribution:")
    print(y.value_counts())

    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(X, y)
    model = train_model(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    evaluate_model(y_test, y_pred, y_prob)
