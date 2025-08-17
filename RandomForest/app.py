import os
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless for servers/Spaces
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

import gradio as gr


DATA_DIR = "data"
RED_LOCAL = os.path.join(DATA_DIR, "winequality-red.csv")
WHITE_LOCAL = os.path.join(DATA_DIR, "winequality-white.csv")

# UCI Wine Quality (red + white)
RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

TASK_MODES = {
    "Regression": {
        "criteria": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
        "blurb": (
            "Predicts the *numeric* quality score (e.g., 5.8). "
            "Metrics: MAE, RMSE, R²."
        ),
    },
    "Classification": {
        "criteria": ["gini", "entropy", "log_loss"],
        "blurb": (
            "Predicts the *discrete* quality class (e.g., 5, 6, 7). "
            "Metrics: Accuracy, Macro F1."
        ),
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
        # Optional: print to server logs for debugging on Spaces
        print(f"[wine] Download failed from {url}: {e}")
        return False

def load_wine_data(cache_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Prefer local CSVs in ./data. If missing, try to download from UCI.
    If download is blocked (e.g., on Spaces without internet), raise a helpful error.
    Adds 'wine_type' (1=white, 0=red).
    """
    _ensure_dir(cache_dir)
    red_path = RED_LOCAL
    white_path = WHITE_LOCAL

    # 1) Local first
    have_red = os.path.exists(red_path)
    have_white = os.path.exists(white_path)

    # 2) If any missing, attempt download
    if not (have_red and have_white):
        if not have_red:
            print("[wine] red CSV missing locally; attempting download…")
            have_red = _download_csv(RED_URL, red_path)
        if not have_white:
            print("[wine] white CSV missing locally; attempting download…")
            have_white = _download_csv(WHITE_URL, white_path)

    # 3) Final check
    if not (have_red and have_white):
        raise RuntimeError(
            "Wine CSVs not found and download failed.\n"
            "Fix: add these files to your repo under ./data/ :\n"
            "  - data/winequality-red.csv\n"
            "  - data/winequality-white.csv\n"
            "You can get them from the UCI Wine Quality dataset."
        )

    # 4) Load + tag
    red = pd.read_csv(red_path)
    white = pd.read_csv(white_path)
    red["wine_type"] = 0
    white["wine_type"] = 1
    return pd.concat([red, white], ignore_index=True)


def split_features_target(df: pd.DataFrame, target: str = "quality") -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return X, y



def train_rf(
    max_depth: int,
    min_samples_split: int,
    n_estimators: int,
    max_features_in: str,
    test_size: float,
    random_state: int,
    task_type: str,
    criterion: str,
):
    """
    Train a RandomForest (regressor or classifier) and return model, metrics, importances, feature list, and test preview.
    """
    df = load_wine_data()
    X, y = split_features_target(df)

    # Classification uses integer labels; regression stays numeric
    if task_type == "Classification":
        y = y.astype(int)

    # Parse max_features (dropdown gives a string)
    # Supported: "sqrt", "log2", "1.0", "0.5", "None"
    if max_features_in == "None":
        max_features = None
    else:
        try:
            max_features = float(max_features_in)
        except ValueError:
            max_features = max_features_in  # keep "sqrt"/"log2"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if task_type == "Classification" else None
    )

    common_kwargs = dict(
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth > 0 else None,
        min_samples_split=min_samples_split,
        max_features=max_features,
        criterion=criterion,
        random_state=random_state,
        n_jobs=-1,
    )

    if task_type == "Classification":
        model = RandomForestClassifier(**common_kwargs)
    else:
        model = RandomForestRegressor(**common_kwargs)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

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

    importances = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False, ignore_index=True)

    return (
        model,                  # 1
        metrics,                # 2
        importances,            # 3
        X.columns.tolist(),     # 4
        y_test.values,          # 6
        preds,                  # 7
        metrics_table           # 8
    )

def plot_importances(importances: 'pd.DataFrame'):
    fig = plt.figure(figsize=(7, 4.5))
    plt.barh(importances["feature"], importances["importance"])
    plt.gca().invert_yaxis()
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    return fig


def plot_tree_fig(model: DecisionTreeRegressor, feature_names: List[str], plot_max_depth: int | None) -> plt.Figure:
    """Plot the trained tree. Use plot_max_depth to keep the figure readable."""
    fig = plt.figure(figsize=(16, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        filled=False,          # no specific colors
        rounded=False,
        impurity=True,
        proportion=True,
        max_depth=plot_max_depth if plot_max_depth and plot_max_depth > 0 else None,
        fontsize=8
    )
    plt.title("Random Forest - 1st Decision Tree Structure")
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



def on_train(
    # model hyperparams
    max_depth: int,
    min_samples_split: int,
    n_estimators: int,
    max_features_in: str,
    # training config
    test_size: float,
    random_state: int,
    # viz config
    plot_max_depth: int,
    # task controls
    task_type: str,
    criterion: str,
    _state: Dict[str, Any]
):
    (model, metrics, importances, feat_list, 
     y_test_arr, preds_arr, metrics_table) = train_rf(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        n_estimators=n_estimators,
        max_features_in=max_features_in,
        test_size=test_size,
        random_state=random_state,
        task_type=task_type,
        criterion=criterion,
    )

    _state["model"] = model
    _state["feature_names"] = feat_list
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
        import matplotlib.pyplot as plt  # ensure local
        cm_plot = plt.figure(figsize=(6, 2))
        plt.axis("off")
        cm_note = "Confusion matrix applies to classification only."

    # Feature importances
    fi_plot = plot_importances(importances)

    # Tree plot: show the first estimator in the forest so the tab remains useful
    if hasattr(model, "estimators_") and len(model.estimators_) > 0:
        first_tree = model.estimators_[0]
        tree_plot = plot_tree_fig(first_tree, feat_list, plot_max_depth)
    else:
        import matplotlib.pyplot as plt
        tree_plot = plt.figure(figsize=(6, 2))
        plt.axis("off")
        plt.title("No tree available to plot")

    return (
        metrics_md,
        metrics_table,
        fi_plot,
        cm_plot,
        cm_note,
        tree_plot,
        _state
    )

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
    return f"Predicted quality: **{pred:.2f}**"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Wine Dataset — Random Forest") as demo:
        gr.Markdown(
            """
            # Wine Dataset — Random Forest

            This interactive demo trains and visualizes a **Random Forest** model on the combined [red and white wine dataset from UCI](https://archive.ics.uci.edu/dataset/186/wine+quality).  
            You can experiment with two types of tasks:

            - **Regression**: predicts a *numeric* wine quality score (e.g., 5.8).  
            - **Classification**: predicts a *discrete* quality class (e.g., 5, 6, 7).

            After training, you’ll see:
            - **Metrics** (accuracy, precision, recall, F1 for classification; MAE, RMSE, R² for regression)  
            - **Feature importance** rankings  
            - **Confusion matrix** (for classification)  
            - **Tree plot** of the **first tree** in the forest (for interpretability)

            **Tips**
            - The **Tree plot** shows a single estimator (the first tree) from the forest. Use **Plot max depth** to keep it readable.  
            - *Model hyperparameters* like **n_estimators**, **max_features**, **max_depth**, and **min_samples_split** shape the learned **forest**, while *training config* controls the dataset split and randomness.  
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
                    max_depth = gr.Slider(0, 32, value=10, step=1, label="max_depth (0 means None)")
                    min_samples_split = gr.Slider(2, 50, value=4, step=1, label="min_samples_split")
                    n_estimators = gr.Slider(10, 500, value=200, step=10, label="n_estimators")
                    max_features_in = gr.Dropdown(
                        choices=["sqrt", "log2", "1.0", "0.5", "None"],
                        value="sqrt",
                        label="max_features"
                    )
                    criterion = gr.Dropdown(
                        choices=["squared_error", "friedman_mse", "absolute_error", "poisson", 
                                "gini", "entropy", "log_loss"],
                        value="gini" if task_type.value == "Classification" else "squared_error",
                        label="criterion"
                    )
                                # react to task changes: update criterion choices + help text
                    def _on_task_change(task: str):
                        choices = TASK_MODES[task]["criteria"]
                        return gr.update(choices=choices, value=choices[0]), TASK_MODES[task]["blurb"]

                    task_type.change(
                        _on_task_change,
                        inputs=[task_type],
                        outputs=[criterion, task_help]
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### Training config")
                    test_size = gr.Slider(0.1, 0.5, value=0.2, step=0.05, label="test_size")
                    random_state = gr.Number(value=42, precision=0, label="random_state")

                with gr.Column(scale=1):
                    gr.Markdown("### Visualization")
                    plot_max_depth = gr.Slider(0, 10, value=3, step=1, label="Plot max depth (0 = full tree)")

            train_btn = gr.Button("Train", variant="primary")


            # Metrics on top
            metrics_out = gr.Markdown("")                                   # 1
            metrics_table = gr.Dataframe(headers=["metric","value"],         # 2
                                        label="Metrics", wrap=True)

            with gr.Tabs():
                with gr.TabItem("Feature Importance"):
                    fi_plot = gr.Plot(label="Feature importances")           # 3

                with gr.TabItem("Confusion Matrix"):
                    cm_plot = gr.Plot(label="Confusion Matrix")              # 5
                    cm_note = gr.Markdown("")                                 # 6

                with gr.TabItem("Tree Plot"):
                    tree_plot = gr.Plot(label="Random Forest — first tree plot")          # 7

            train_btn.click(
                on_train,
                inputs=[max_depth, min_samples_split, n_estimators, max_features_in, test_size, random_state, plot_max_depth, task_type, criterion, state],
                outputs=[metrics_out, metrics_table, fi_plot, cm_plot, cm_note, tree_plot, state]
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
