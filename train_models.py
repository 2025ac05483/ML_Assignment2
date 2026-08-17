import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TARGET_COLUMN = "target"
MODEL_DIR = Path("model")


def get_dataset():
    dataset = load_breast_cancer(as_frame=True)
    data = dataset.frame.copy()
    data.columns = [column.replace(" ", "_") for column in data.columns]
    feature_names = [column.replace(" ", "_") for column in dataset.feature_names]
    return data, feature_names, list(dataset.target_names)


def build_models():
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=5,
                        min_samples_leaf=4,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "kNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Naive Bayes": Pipeline([("classifier", GaussianNB())]),
        "Random Forest (Ensemble)": Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=8,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
    }


def predict_positive_probability(model, x_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    decision_scores = model.decision_function(x_test)
    return (decision_scores - decision_scores.min()) / (
        decision_scores.max() - decision_scores.min()
    )


def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    probabilities = predict_positive_probability(model, x_test)
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    data, feature_names, target_names = get_dataset()
    x = data[feature_names]
    y = data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    test_data = x_test.copy()
    test_data[TARGET_COLUMN] = y_test
    test_data.to_csv("test_data.csv", index=False)

    rows = []
    model_files = {}
    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        file_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        file_name = file_name.replace("__", "_") + ".joblib"
        model_path = MODEL_DIR / file_name
        joblib.dump(model, model_path)
        model_files[model_name] = str(model_path)
        rows.append({"ML Model Name": model_name, **metrics})

    metrics_df = pd.DataFrame(rows)
    for column in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        metrics_df[column] = np.round(metrics_df[column], 4)
    metrics_df.to_csv(MODEL_DIR / "metrics.csv", index=False)

    metadata = {
        "feature_names": feature_names,
        "target_column": TARGET_COLUMN,
        "target_names": target_names,
        "model_files": model_files,
    }
    joblib.dump(metadata, MODEL_DIR / "metadata.joblib")

    print("Training complete.")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
