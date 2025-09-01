# app.py — Wine Dataset — SVM (Classification & Regression)
# Matches your MLP demo structure & UI for a consistent portfolio.

import os
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from sklearn.inspection import permutation_importance

import gradio as gr

DATA_DIR = "data"
RED_LOCAL = os.path.join(DATA_DIR, "winequality-red.csv")
WHITE_LOCAL = os.path.join(DATA_DIR, "winequality-white.csv")

RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

TASK_MODES = {
    "Regression": {
        "blurb": "Predicts the numeric quality score (e.g., 5.8). Metrics: MAE, RMSE, R²."
    },
    "Classification": {
        "blurb": "Predicts the discrete quality class (e.g., 5, 6, 7). Metrics: Accuracy, Macro F1."
    },
}

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
    If download is blocked (Spaces w/o internet), raise a helpful error.
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
            "Fix: add these files under ./data :\n"
            "  - data/winequality-red.csv\n"
            "  - data/winequality-white.csv\n"
            "Get them from the UCI Wine Quality dataset."
        )

    red = pd.read_csv(RED_LOCAL)
    white = pd.read_csv(WHITE_LOCAL)
    red["wine_type"] = 0
    white["wine_type"] = 1
    return pd.concat([red, white], ignore_index=True)

def split_features_target(df: pd.DataFrame, target: str = "quality") -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def _perm_importance_dataframe(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
    max_samples: int = 600,
    n_repeats: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute permutation importance on a subset of test data for speed.
    Works for both SVC and SVR inside a Pipeline.
    """
    if len(X_test) > max_samples:
        X_eval = X_test.sample(max_samples, random_state=random_state)
        y_eval = y_test.loc[X_eval.index]
    else:
        X_eval = X_test
        y_eval = y_test

    r = permutation_importance(
        pipe, X_eval, y_eval,
        n_repeats=n_repeats, random_state=random_state, n_jobs=-1, scoring=None
    )
    df = pd.DataFrame({"feature": X_test.columns, "importance": r.importances_mean})
    return df.sort_values("importance", ascending=False, ignore_index=True)

def plot_importances(importances: 'pd.DataFrame') -> plt.Figure:
    fig = plt.figure(figsize=(7, 4.5))
    plt.barh(importances["feature"], importances["importance"])
    plt.gca().invert_yaxis()
    plt.title("Permutation Importance (test subset)")
    plt.xlabel("Mean score decrease")
    plt.ylabel("Feature")
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    im = ax.imshow(cm, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9, color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.tight_layout()
    return fig

def train_svm(
    task_type: str,
    kernel: str,
    C: float,
    gamma: str,
    epsilon: float,           # used for SVR
    degree: int,              # used for polynomial kernels
    class_weight: str,        # used for SVC
    test_size: float,
    random_state: int
):
    """
    Train an SVM (SVC or SVR) inside a pipeline with StandardScaler.
    Returns: pipeline, metrics, importances, y_test, preds, metrics_table.
    """
    df = load_wine_data()
    X, y = split_features_target(df)

    if task_type == "Classification":
        y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if task_type == "Classification" else None
    )

    # Build estimator
    if task_type == "Classification":
        est = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma if kernel in ("rbf", "poly", "sigmoid") else "scale",
            degree=degree if kernel == "poly" else 3,
            class_weight=None if class_weight == "None" else class_weight,
            probability=False,               # keep it fast/lightweight
            random_state=random_state
        )
    else:
        est = SVR(
            kernel=kernel,
            C=C,
            gamma=gamma if kernel in ("rbf", "poly", "sigmoid") else "scale",
            degree=degree if kernel == "poly" else 3,
            epsilon=epsilon
        )

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svm", est)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    # Metrics
    if task_type == "Classification":
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
            {"metric": list(metrics.keys()), "value": [f"{v:.3f}" for v in metrics.values()]}
        )
    else:
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
        metrics_table = pd.DataFrame(
            {"metric": ["MAE", "RMSE", "R²"], "value": [f"{mae:.3f}", f"{rmse:.3f}", f"{r2:.3f}"]}
        )

    # Permutation importances on a subset
    importances = _perm_importance_dataframe(pipe, X_test, y_test, task_type, random_state=random_state)

    return pipe, metrics, importances, y_test.values, preds, metrics_table

def on_train(
    task_type: str,
    kernel: str,
    C: float,
    gamma: str,
    epsilon: float,
    degree: int,
    class_weight: str,
    test_size: float,
    random_state: int,
    _state: Dict[str, Any]
):
    (pipe, metrics, importances, y_test_arr, preds_arr, metrics_table) = train_svm(
        task_type=task_type,
        kernel=kernel,
        C=C,
        gamma=gamma,
        epsilon=epsilon,
        degree=degree,
        class_weight=class_weight,
        test_size=test_size,
        random_state=random_state
    )

    _state["model"] = pipe
    _state["task_type"] = task_type

    if task_type == "Classification":
        metrics_md = (
            f"**Accuracy:** {metrics['Accuracy']:.3f} | "
            f"**Precision (macro):** {metrics['Precision (macro)']:.3f} | "
            f"**Recall (macro):** {metrics['Recall (macro)']:.3f} | "
            f"**F1 (macro):** {metrics['F1 (macro)']:.3f}"
        )
        labels_sorted = sorted(np.unique(y_test_arr).tolist())
        cm_plot = plot_confusion_matrix(y_test_arr, preds_arr, labels=labels_sorted)
        cm_note = ""
    else:
        metrics_md = f"**MAE:** {metrics['MAE']:.3f} | **RMSE:** {metrics['RMSE']:.3f} | **R²:** {metrics['R2']:.3f}"
        import matplotlib.pyplot as plt
        cm_plot = plt.figure(figsize=(6, 2))
        plt.axis("off")
        cm_note = "Confusion matrix applies to classification only."

    fi_plot = plot_importances(importances)

    return metrics_md, metrics_table, fi_plot, cm_plot, cm_note, _state

def on_predict(
    fixed_acidity: float, volatile_acidity: float, citric_acid: float,
    residual_sugar: float, chlorides: float, free_sulfur_dioxide: float,
    total_sulfur_dioxide: float, density: float, pH: float, sulphates: float,
    alcohol: float, wine_type: str, _state: Dict[str, Any]
):
    pipe = _state.get("model")
    if pipe is None:
        return "Please train the model first (Train/Evaluate tab)."

    x = np.array([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol,
        1 if wine_type == "white" else 0
    ]])
    pred = pipe.predict(x)[0]
    return (
        f"Predicted quality: **{pred:.2f}**"
        if _state.get("task_type") == "Regression"
        else f"Predicted quality class: **{int(round(pred))}**"
    )

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Wine Dataset — SVM") as demo:
        gr.Markdown(
            """
            # Wine Dataset — SVM
            This interactive demo trains and visualizes a **Support Vector Machine** on the combined
            [red and white wine dataset from UCI](https://archive.ics.uci.edu/dataset/186/wine+quality).

            **Tasks**
            - **Regression**: predicts a *numeric* wine quality score (e.g., 5.8).
            - **Classification**: predicts a *discrete* quality class (e.g., 5, 6, 7).

            **What you'll see after training**
            - **Metrics** (accuracy/precision/recall/F1 for classification; MAE/RMSE/R² for regression)
            - **Permutation importance** (model-agnostic feature effects)
            - **Confusion matrix** (for classification)
            """
        )

        state = gr.State({})

        with gr.Tab("Train/Evaluate"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Task")
                    task_type = gr.Radio(choices=list(TASK_MODES.keys()), value="Classification", label="Choose task")
                    task_help = gr.Markdown(TASK_MODES["Classification"]["blurb"])

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Model hyperparameters")
                    kernel = gr.Dropdown(choices=["rbf", "linear", "poly", "sigmoid"], value="rbf", label="kernel")
                    C = gr.Slider(0.1, 50.0, value=5.0, step=0.1, label="C (regularization strength)")
                    gamma = gr.Dropdown(choices=["scale", "auto"], value="scale", label="gamma (rbf/poly/sigmoid)")
                    degree = gr.Slider(2, 6, value=3, step=1, label="degree (poly only)")
                    epsilon = gr.Slider(0.01, 2.0, value=0.1, step=0.01, label="epsilon (SVR only)")
                    class_weight = gr.Dropdown(choices=["None", "balanced"], value="None", label="class_weight (SVC)")

                with gr.Column(scale=1):
                    gr.Markdown("### Training config")
                    test_size = gr.Slider(0.1, 0.5, value=0.2, step=0.05, label="test_size")
                    random_state = gr.Number(value=42, precision=0, label="random_state")

            def _on_task_change(task: str):
                return TASK_MODES[task]["blurb"]

            task_type.change(_on_task_change, inputs=[task_type], outputs=[task_help])

            train_btn = gr.Button("Train", variant="primary")

            metrics_out = gr.Markdown("")                                   # 1
            metrics_table = gr.Dataframe(headers=["metric","value"],         # 2
                                         label="Metrics", wrap=True)

            with gr.Tabs():
                with gr.TabItem("Permutation Importance"):
                    fi_plot = gr.Plot(label="Feature effects")               # 3
                with gr.TabItem("Confusion Matrix"):
                    cm_plot = gr.Plot(label="Confusion Matrix")              # 4
                    cm_note = gr.Markdown("")                                # 5

            train_btn.click(
                on_train,
                inputs=[task_type, kernel, C, gamma, epsilon, degree, class_weight, test_size, random_state, state],
                outputs=[metrics_out, metrics_table, fi_plot, cm_plot, cm_note, state]
            )

        with gr.Tab("Predict"):
            gr.Markdown("Provide physicochemical properties and wine type. Use ranges similar to the dataset.")
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

            predict_btn = gr.Button("Predict quality", variant="primary")
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
