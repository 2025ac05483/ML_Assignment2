import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODEL_DIR = Path("model")
DEFAULT_TEST_DATA = Path("test_data.csv")


def apply_custom_styles():
    st.markdown(
        """
        <style>
        :root {
            --ink: #16213e;
            --muted: #5f6c7b;
            --blue: #2563eb;
            --teal: #0f9f8e;
            --coral: #ef5b5b;
            --amber: #f59e0b;
            --panel: #ffffff;
            --soft-blue: #eef5ff;
            --soft-teal: #e8fbf7;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.12), transparent 28rem),
                radial-gradient(circle at top right, rgba(15, 159, 142, 0.12), transparent 26rem),
                #f7fafc;
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #17324d 0%, #1f5f66 100%);
        }

        [data-testid="stSidebar"] * {
            color: #f8fbff;
        }

        [data-testid="stSidebar"] label {
            font-weight: 700;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #172033;
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .app-header {
            border-left: 8px solid var(--coral);
            background: linear-gradient(90deg, #ffffff 0%, var(--soft-blue) 50%, var(--soft-teal) 100%);
            border-radius: 8px;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1.3rem;
            box-shadow: 0 10px 28px rgba(22, 33, 62, 0.08);
        }

        .app-header h1 {
            color: var(--ink);
            font-size: 2rem;
            line-height: 1.15;
            margin: 0;
        }

        .app-header p {
            color: var(--muted);
            font-size: 1rem;
            margin: 0.45rem 0 0;
        }

        h2, h3 {
            color: var(--ink);
        }

        [data-testid="stMetric"] {
            background: var(--panel);
            border-top: 5px solid var(--blue);
            border-radius: 8px;
            padding: 0.95rem;
            box-shadow: 0 8px 22px rgba(22, 33, 62, 0.08);
        }

        div[data-testid="stMetric"]:nth-of-type(2n) {
            border-top-color: var(--teal);
        }

        div[data-testid="stMetric"]:nth-of-type(3n) {
            border-top-color: var(--amber);
        }

        div[data-testid="stMetric"]:nth-of-type(4n) {
            border-top-color: var(--coral);
        }

        [data-testid="stDataFrame"] {
            border: 1px solid rgba(22, 33, 62, 0.10);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 8px 20px rgba(22, 33, 62, 0.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    metadata = joblib.load(MODEL_DIR / "metadata.joblib")
    models = {
        model_name: joblib.load(model_path)
        for model_name, model_path in metadata["model_files"].items()
    }
    metrics = pd.read_csv(MODEL_DIR / "metrics.csv")
    return metadata, models, metrics


def evaluate(model, x_data, y_true):
    predictions = model.predict(x_data)
    probabilities = model.predict_proba(x_data)[:, 1]
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "AUC": roc_auc_score(y_true, probabilities),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predictions),
    }, predictions


def plot_confusion_matrix(y_true, predictions, target_names):
    matrix = confusion_matrix(y_true, predictions)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return fig


def main():
    st.set_page_config(page_title="ML Assignment 2", layout="wide")
    apply_custom_styles()
    st.markdown(
        """
        <div class="app-header">
            <h1>ML Assignment 2 - Classification Model Comparison</h1>
            <p>Compare five classification models on the Breast Cancer Wisconsin dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metadata, models, saved_metrics = load_artifacts()
    feature_names = metadata["feature_names"]
    target_column = metadata["target_column"]
    target_names = metadata["target_names"]

    st.sidebar.header("Input")
    uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])
    selected_model_name = st.sidebar.selectbox("Select model", list(models.keys()))

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv(DEFAULT_TEST_DATA)

    missing_features = [feature for feature in feature_names if feature not in data.columns]
    if missing_features:
        st.error(f"Uploaded CSV is missing required columns: {missing_features}")
        st.stop()

    x_data = data[feature_names]
    selected_model = models[selected_model_name]
    predictions = selected_model.predict(x_data)

    st.subheader("Dataset Preview")
    st.dataframe(data.head(20), width="stretch")

    if target_column in data.columns:
        y_true = data[target_column]
        selected_metrics, predictions = evaluate(selected_model, x_data, y_true)

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        metric_columns = [col1, col2, col3, col4, col5, col6]
        for column, (metric_name, metric_value) in zip(
            metric_columns, selected_metrics.items()
        ):
            column.metric(metric_name, f"{metric_value:.4f}")

        st.subheader("Saved Model Comparison")
        st.dataframe(saved_metrics, width="stretch")

        left, right = st.columns(2)
        with left:
            st.pyplot(plot_confusion_matrix(y_true, predictions, target_names))
        with right:
            report = classification_report(
                y_true,
                predictions,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
            )
            st.dataframe(pd.DataFrame(report).transpose(), width="stretch")
    else:
        st.info(
            "The uploaded CSV does not include the target column, so the app is showing predictions only."
        )

    output = data.copy()
    output["predicted_target"] = predictions
    output["predicted_label"] = [target_names[prediction] for prediction in predictions]

    st.subheader("Predictions")
    st.dataframe(output.head(50), width="stretch")


if __name__ == "__main__":
    main()
