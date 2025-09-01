import os
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless for servers/Spaces
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

import gradio as gr


# ---------- Data ----------
DATA_DIR = "data"
RED_LOCAL = os.path.join(DATA_DIR, "winequality-red.csv")
WHITE_LOCAL = os.path.join(DATA_DIR, "winequality-white.csv")

RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

TASK_MODES = {
    "Regression": {
        "models": ["LinearRegression", "Ridge", "Lasso"],
        "blurb": "Predicts the *numeric* quality score (e.g., 5.8). Metrics: MAE, RMSE, R²."
    },
    "Classification": {
        "models": ["LogisticRegression"],
        "blurb": "Predicts the *discrete* quality class (e.g., 5, 6, 7). Metrics: Accuracy, Macro F1."
    }
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
            "You can get them from the UCI Wine Quality dataset."
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


# ---------- Training / Evaluation ----------
def _build_pipeline(
    task_type: str,
    model_name: str,
    alpha: float,
    C_inv_reg: float,
    standardize: bool,
    random_state: int
) -> Pipeline:
    steps: List[tuple[str, Any]] = []
    if standardize:
        steps.append(("scaler", StandardScaler()))

    if task_type == "Regression":
        if model_name == "LinearRegression":
            model = LinearRegression()
        elif model_name == "Ridge":
            model = Ridge(alpha=alpha, random_state=random_state)
        elif model_name == "Lasso":
            model = Lasso(alpha=alpha, random_state=random_state, max_iter=10000)
        else:
            raise ValueError(f"Unknown regression model: {model_name}")
    else:
        # Classification
        # Use lbfgs for multinomial. C is inverse of regularization strength.
        model = LogisticRegression(
            C=C_inv_reg,
            penalty="l2",
            solver="lbfgs",
            multi_class="auto",
            max_iter=2000,
            random_state=random_state,
        )

    steps.append(("model", model))
    return Pipeline(steps)


def _coef_dataframe(pipeline: Pipeline, feature_names: List[str], task_type: str) -> pd.DataFrame:
    # Works for Linear/Ridge/Lasso and LogisticRegression.
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        coefs = np.atleast_2d(model.coef_)
        # For multiclass, average absolute magnitude across classes
        if coefs.shape[0] > 1:
            vals = np.mean(np.abs(coefs), axis=0)
        else:
            vals = coefs.flatten()
        return pd.DataFrame({"feature": feature_names, "coef": vals}).sort_values("coef", ascending=False, ignore_index=True)
    else:
        return pd.DataFrame({"feature": feature_names, "coef": np.zeros(len(feature_names))})


def plot_coefficients(coef_df: pd.DataFrame, top_k: int = 12) -> plt.Figure:
    top = coef_df.head(top_k)
    fig = plt.figure(figsize=(7, 4.5))
    plt.barh(top["feature"], np.abs(top["coef"]))
    plt.gca().invert_yaxis()
    plt.title("Top Feature Coefficient Magnitudes")
    plt.xlabel("|coefficient|")
    plt.ylabel("Feature")
    plt.tight_layout()
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    resid = y_true - y_pred
    fig = plt.figure(figsize=(6.5, 4))
    plt.scatter(y_pred, resid, s=12)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (true - pred)")
    plt.title("Residuals Plot (Regression)")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    im = ax.imshow(cm)  # default colormap
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def train_linear_models(
    task_type: str,
    model_name: str,
    alpha: float,
    C_inv_reg: float,
    standardize: bool,
    test_size: float,
    random_state: int,
):
    df = load_wine_data()
    X, y = split_features_target(df)

    if task_type == "Classification":
        y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if task_type == "Classification" else None
    )

    pipe = _build_pipeline(
        task_type=task_type,
        model_name=model_name,
        alpha=alpha,
        C_inv_reg=C_inv_reg,
        standardize=standardize,
        random_state=random_state
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

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
        metrics_table = pd.DataFrame({"metric": list(metrics.keys()), "value": [f"{v:.3f}" for v in metrics.values()]})
        cm_fig = plot_confusion_matrix(y_test, preds, labels=sorted(np.unique(y_test).tolist()))
        residuals_fig = _blank_fig("Residuals apply to regression only.")
        cm_note = ""
    else:
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
        metrics_table = pd.DataFrame({"metric": ["MAE", "RMSE", "R²"], "value": [f"{mae:.3f}", f"{rmse:.3f}", f"{r2:.3f}"]})
        cm_fig = _blank_fig("Confusion matrix applies to classification only.")
        cm_note = "Confusion matrix applies to classification only."
        residuals_fig = plot_residuals(np.array(y_test), np.array(preds))

    coef_df = _coef_dataframe(pipe, X.columns.tolist(), task_type)
    coef_plot = plot_coefficients(coef_df)

    return pipe, metrics, metrics_table, coef_df, coef_plot, cm_fig, cm_note, residuals_fig, X.columns.tolist(), y_test.values, preds


def _blank_fig(note: str = "") -> plt.Figure:
    fig = plt.figure(figsize=(6, 2))
    plt.axis("off")
    if note:
        plt.title(note)
    return fig


# ---------- Gradio callbacks ----------
def on_train(
    task_type: str,
    model_name: str,
    alpha: float,
    C_inv_reg: float,
    standardize: bool,
    test_size: float,
    random_state: int,
    _state: Dict[str, Any]
):
    (pipe, metrics, metrics_table, coef_df, coef_plot, cm_plot, cm_note, resid_plot,
     feat_list, y_test_arr, preds_arr) = train_linear_models(
        task_type=task_type,
        model_name=model_name,
        alpha=alpha,
        C_inv_reg=C_inv_reg,
        standardize=standardize,
        test_size=test_size,
        random_state=random_state,
    )

    _state["model"] = pipe
    _state["feature_names"] = feat_list
    _state["task_type"] = task_type

    if task_type == "Classification":
        metrics_md = (
            f"**Accuracy:** {metrics['Accuracy']:.3f} | "
            f"**Precision (macro):** {metrics['Precision (macro)']:.3f} | "
            f"**Recall (macro):** {metrics['Recall (macro)']:.3f} | "
            f"**F1 (macro):** {metrics['F1 (macro)']:.3f}"
        )
    else:
        metrics_md = f"**MAE:** {metrics['MAE']:.3f} | **RMSE:** {metrics['RMSE']:.3f} | **R²:** {metrics['R2']:.3f}"

    return (
        metrics_md,       # 1
        metrics_table,    # 2
        coef_plot,        # 3
        cm_plot,          # 4
        cm_note,          # 5
        resid_plot,       # 6
        _state            # 7
    )


def on_predict(
        fixed_acidity: float, volatile_acidity: float, citric_acid: float,
        residual_sugar: float, chlorides: float, free_sulfur_dioxide: float,
        total_sulfur_dioxide: float, density: float, pH: float, sulphates: float,
        alcohol: float, wine_type: str, _state: Dict[str, Any]
    ):
    model = _state.get("model")
    task_type = _state.get("task_type", "Regression")
    if model is None:
        return "Please train the model first (Train/Evaluate tab)."

    x = np.array([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol,
        1 if wine_type == "white" else 0
    ]])
    pred = model.predict(x)[0]
    if task_type == "Classification":
        return f"Predicted class (quality): **{int(pred)}**"
    else:
        return f"Predicted quality: **{pred:.2f}**"


# ---------- UI ----------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Wine Dataset — Linear Models") as demo:
        gr.Markdown(
            """
            # Wine Dataset — Linear Models (Linear / Ridge / Lasso / Logistic)

            This interactive demo trains and visualizes **linear models** on the combined
            [red and white wine dataset from UCI](https://archive.ics.uci.edu/dataset/186/wine+quality).

            You can experiment with two task types:
            - **Regression**: LinearRegression, Ridge, Lasso  
            - **Classification**: LogisticRegression

            After training, you will see:
            - **Metrics** (accuracy, precision, recall, F1 for classification; MAE, RMSE, R² for regression)  
            - **Coefficient magnitudes** (top features)  
            - **Confusion matrix** (classification)  
            - **Residuals plot** (regression)

            **Tips**
            - Linear models often benefit from **Standardize features**.
            - For Ridge/Lasso, adjust **alpha** to control regularization strength.
            """
        )

        state = gr.State({})

        with gr.Tab("Train/Evaluate"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Task")
                    task_type = gr.Radio(choices=list(TASK_MODES.keys()), value="Classification", label="Choose task")
                    task_help = gr.Markdown(TASK_MODES["Classification"]["blurb"])

                    def _on_task_change(task: str):
                        models = TASK_MODES[task]["models"]
                        return (
                            gr.update(choices=models, value=models[0]),
                            TASK_MODES[task]["blurb"],
                            gr.update(visible=(task == "Regression")),   # alpha row (Ridge/Lasso)
                            gr.update(visible=(task == "Classification")) # C row (LogReg)
                        )

                with gr.Column(scale=1):
                    gr.Markdown("### Model & Regularization")
                    model_name = gr.Dropdown(
                        choices=TASK_MODES["Classification"]["models"],
                        value=TASK_MODES["Classification"]["models"][0],
                        label="Model"
                    )
                    standardize = gr.Checkbox(value=True, label="Standardize features")

                    alpha = gr.Slider(0.0001, 5.0, value=1.0, step=0.0001, label="alpha (Ridge/Lasso)", visible=False)
                    C_inv_reg = gr.Slider(0.01, 10.0, value=1.0, step=0.01, label="C (LogisticRegression)", visible=True)

                    task_type.change(_on_task_change, inputs=[task_type], outputs=[model_name, task_help, alpha, C_inv_reg])

                with gr.Column(scale=1):
                    gr.Markdown("### Training config")
                    test_size = gr.Slider(0.1, 0.5, value=0.2, step=0.05, label="test_size")
                    random_state = gr.Number(value=42, precision=0, label="random_state")

            train_btn = gr.Button("Train", variant="primary")

            metrics_out = gr.Markdown("")                                   # 1
            metrics_table = gr.Dataframe(headers=["metric","value"],         # 2
                                         label="Metrics", wrap=True)

            with gr.Tabs():
                with gr.TabItem("Coefficient Magnitudes"):
                    coef_plot = gr.Plot(label="Top coefficients")            # 3

                with gr.TabItem("Confusion Matrix"):
                    cm_plot = gr.Plot(label="Confusion Matrix")              # 4
                    cm_note = gr.Markdown("")                                # 5

                with gr.TabItem("Residuals (Regression)"):
                    resid_plot = gr.Plot(label="Residuals Plot")             # 6

            train_btn.click(
                on_train,
                inputs=[task_type, model_name, alpha, C_inv_reg, standardize, test_size, random_state, state],
                outputs=[metrics_out, metrics_table, coef_plot, cm_plot, cm_note, resid_plot, state]
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

            predict_btn = gr.Button("Predict", variant="primary")
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
