import os
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless for Spaces
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from sklearn.feature_selection import f_classif

import gradio as gr

# --------------------
# Data locations
# --------------------
DATA_DIR = "data"
RED_LOCAL = os.path.join(DATA_DIR, "winequality-red.csv")
WHITE_LOCAL = os.path.join(DATA_DIR, "winequality-white.csv")

# UCI Wine Quality (red + white)
RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

TASK_BLURB = "Predicts the *discrete* quality class (e.g., 5, 6, 7). Metrics: Accuracy, Macro F1."

# --------------------
# Utilities
# --------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _download_csv(url: str, dest_path: str, sep: str = ";") -> bool:
    try:
        df = pd.read_csv(url, sep=sep)
        df.to_csv(dest_path, index=False)
        return True
    except Exception as e:
        print(f"[wine] Download failed from {url}: {e}")
        return False

def load_wine_data(cache_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Prefer local CSVs in ./data. If missing, try to download from UCI.
    Adds 'wine_type' (1=white, 0=red).
    """
    _ensure_dir(cache_dir)

    have_red = os.path.exists(RED_LOCAL)
    have_white = os.path.exists(WHITE_LOCAL)

    if not (have_red and have_white):
        if not have_red:
            print("[wine] red CSV missing locally; attempting download…")
            have_red = _download_csv(RED_URL, RED_LOCAL)
        if not have_white:
            print("[wine] white CSV missing locally; attempting download…")
            have_white = _download_csv(WHITE_URL, WHITE_LOCAL)

    if not (have_red and have_white):
        raise RuntimeError(
            "Wine CSVs not found and download failed.\n"
            "Fix: add these files to your repo under ./data/ :\n"
            "  - data/winequality-red.csv\n"
            "  - data/winequality-white.csv\n"
        )

    red = pd.read_csv(RED_LOCAL)
    white = pd.read_csv(WHITE_LOCAL)
    red["wine_type"] = 0
    white["wine_type"] = 1
    return pd.concat([red, white], ignore_index=True)

def split_features_target(df: pd.DataFrame, target: str = "quality") -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target].astype(int)  # classes
    return X, y

# --------------------
# Training / Evaluation
# --------------------
def train_nb(
    var_smoothing_exp: float,
    use_uniform_priors: bool,
    test_size: float,
    random_state: int,
):
    """
    Train a Gaussian Naive Bayes classifier and return model, metrics, feature scores, test preview.
    var_smoothing is set as 10 ** var_smoothing_exp for easier slider control.
    """
    df = load_wine_data()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    priors = None
    if use_uniform_priors:
        classes = np.unique(y_train)
        priors = np.ones_like(classes, dtype=float) / len(classes)

    model = GaussianNB(var_smoothing=10.0 ** var_smoothing_exp, priors=priors)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, preds))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, preds, average="macro", zero_division=0
    )
    metrics = {
        "Accuracy": accuracy,
        "Precision (macro)": float(prec_macro),
        "Recall (macro)": float(rec_macro),
        "F1 (macro)": float(f1_macro),
    }
    metrics_table = pd.DataFrame(
        {"metric": list(metrics.keys()),
         "value": [f"{v:.3f}" for v in metrics.values()]}
    )

    # Feature relevance proxy via ANOVA F-score (univariate)
    f_scores, pvals = f_classif(X_train, y_train)
    feature_scores = pd.DataFrame(
        {"feature": X.columns, "F_score": f_scores, "p_value": pvals}
    ).sort_values("F_score", ascending=False, ignore_index=True)

    return model, metrics, feature_scores, X.columns.tolist(), y_test.values, preds, metrics_table

def plot_feature_scores(scores: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(7, 4.5))
    plt.barh(scores["feature"], scores["F_score"])
    plt.gca().invert_yaxis()
    plt.title("Feature relevance (ANOVA F-score)")
    plt.xlabel("F-score")
    plt.ylabel("Feature")
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig

# --------------------
# Gradio callbacks
# --------------------
def on_train(
    var_smoothing_exp: float,
    use_uniform_priors: bool,
    test_size: float,
    random_state: int,
    _state: Dict[str, Any]
):
    (model, metrics, feat_scores, feat_list,
     y_test_arr, preds_arr, metrics_table) = train_nb(
        var_smoothing_exp=var_smoothing_exp,
        use_uniform_priors=use_uniform_priors,
        test_size=test_size,
        random_state=random_state,
    )

    _state["model"] = model
    _state["feature_names"] = feat_list

    metrics_md = (
        f"**Accuracy:** {metrics['Accuracy']:.3f} | "
        f"**Precision (macro):** {metrics['Precision (macro)']:.3f} | "
        f"**Recall (macro):** {metrics['Recall (macro)']:.3f} | "
        f"**F1 (macro):** {metrics['F1 (macro)']:.3f}"
    )

    labels_sorted = sorted(np.unique(y_test_arr).tolist())
    cm_plot = plot_confusion_matrix(y_test_arr, preds_arr, labels=labels_sorted)
    fs_plot = plot_feature_scores(feat_scores)

    return metrics_md, metrics_table, fs_plot, cm_plot, _state

def on_predict(
        fixed_acidity: float, volatile_acidity: float, citric_acid: float,
        residual_sugar: float, chlorides: float, free_sulfur_dioxide: float,
        total_sulfur_dioxide: float, density: float, pH: float, sulphates: float,
        alcohol: float, wine_type: str, _state: Dict[str, Any]
    ):
    model = _state.get("model")
    if model is None:
        return "Please train the model first (Train/Evaluate tab)."

    x = np.array([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol,
        1 if wine_type == "white" else 0
    ]])
    pred = model.predict(x)[0]
    return f"Predicted quality class: **{int(pred)}**"

# --------------------
# UI
# --------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Wine Dataset — Naive Bayes") as demo:
        gr.Markdown(
            """
            # Wine Dataset — Naive Bayes

            This interactive demo trains a **Gaussian Naive Bayes** classifier on the combined
            [red and white wine dataset from UCI](https://archive.ics.uci.edu/dataset/186/wine+quality).

            **Task**: Classification only.  
            """ + TASK_BLURB + """
            
            **What you’ll see after training**
            - Metrics (Accuracy, macro Precision/Recall/F1)  
            - Confusion matrix  
            - Feature relevance via **ANOVA F-score** (quick univariate proxy)
            """
        )

        state = gr.State({})

        with gr.Tab("Train/Evaluate"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Model hyperparameters")
                    var_smoothing_exp = gr.Slider(-12.0, -6.0, value=-9.0, step=0.5,
                                                  label="log10(var_smoothing)")
                    use_uniform_priors = gr.Checkbox(value=False, label="Use uniform class priors")

                with gr.Column(scale=1):
                    gr.Markdown("### Training config")
                    test_size = gr.Slider(0.1, 0.5, value=0.2, step=0.05, label="test_size")
                    random_state = gr.Number(value=42, precision=0, label="random_state")

            train_btn = gr.Button("Train", variant="primary")

            metrics_out = gr.Markdown("")                    # 1
            metrics_table = gr.Dataframe(headers=["metric","value"], label="Metrics", wrap=True)  # 2

            with gr.Tabs():
                with gr.TabItem("Feature Relevance (F-score)"):
                    fs_plot = gr.Plot(label="ANOVA F-scores") # 3
                with gr.TabItem("Confusion Matrix"):
                    cm_plot = gr.Plot(label="Confusion Matrix") # 4

            train_btn.click(
                on_train,
                inputs=[var_smoothing_exp, use_uniform_priors, test_size, random_state, state],
                outputs=[metrics_out, metrics_table, fs_plot, cm_plot, state]
            )

        with gr.Tab("Predict"):
            gr.Markdown("Provide physicochemical properties and wine type.")
            with gr.Row():
                fixed_acidity = gr.Slider(3.0, 16.0, value=7.4, step=0.1, label="fixed acidity")
                volatile_acidity = gr.Slider(0.08, 1.58, value=0.70, step=0.01, label="volatile acidity")
                citric_acid = gr.Slider(0.0, 1.0, value=0.00, step=0.01, label="citric acid")
                residual_sugar = gr.Slider(0.9, 65.0, value=1.9, step=0.1, label="residual sugar")
            with gr.Row():
                chlorides = gr.Slider(0.012, 0.611, value=0.076, step=0.001, label="chlorides")
                free_sulfur_dioxide = gr.Slider(1.0, 289.0, value=11.0, step=1.0, label="free sulfur dioxide")
                total_sulfur_dioxide = gr.Slider(6.0, 440.0, value=34.0, step=1.0, label="total sulfur dioxide")
                density = gr.Slider(0.987, 1.004, value=0.9978, step=0.0001, label="density")
            with gr.Row():
                pH = gr.Slider(2.5, 4.5, value=3.51, step=0.01, label="pH")
                sulphates = gr.Slider(0.22, 2.0, value=0.56, step=0.01, label="sulphates")
                alcohol = gr.Slider(8.0, 14.9, value=9.4, step=0.1, label="alcohol")
                wine_type = gr.Radio(choices=["red", "white"], value="red", label="wine type")

            predict_btn = gr.Button("Predict quality class", variant="primary")
            pred_out = gr.Markdown()

            predict_btn.click(
                on_predict,
                inputs=[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol,
                        wine_type, state],
                outputs=[pred_out]
            )

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
