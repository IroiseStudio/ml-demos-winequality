import os
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless for servers/Spaces
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from sklearn.inspection import permutation_importance

import gradio as gr


DATA_DIR = "data"
RED_LOCAL = os.path.join(DATA_DIR, "winequality-red.csv")
WHITE_LOCAL = os.path.join(DATA_DIR, "winequality-white.csv")

# UCI Wine Quality (red + white)
RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

TASK_MODES = {
    "Regression": {
        "criteria": ["euclidean", "manhattan", "minkowski"],
        "blurb": (
            "Predicts the *numeric* quality score (e.g., 5.8). "
            "Metrics: MAE, RMSE, R²."
        ),
    },
    "Classification": {
        "criteria": ["euclidean", "manhattan", "minkowski"],
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
        print(f"[wine] Download failed from {url}: {e}")
        return False

def load_wine_data(cache_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Prefer local CSVs in ./data. If missing, try to download from UCI.
    If download is blocked (e.g., on Spaces), raise a helpful error.
    Adds 'wine_type' (1=white, 0=red).
    """
    _ensure_dir(cache_dir)
    red_path, white_path = RED_LOCAL, WHITE_LOCAL

    have_red = os.path.exists(red_path)
    have_white = os.path.exists(white_path)

    if not (have_red and have_white):
        if not have_red:
            print("[wine] red CSV missing locally; attempting download…")
            have_red = _download_csv(RED_URL, red_path)
        if not have_white:
            print("[wine] white CSV missing locally; attempting download…")
            have_white = _download_csv(WHITE_URL, white_path)

    if not (have_red and have_white):
        raise RuntimeError(
            "Wine CSVs not found and download failed.\n"
            "Fix: add these files to your repo under ./data/ :\n"
            "  - data/winequality-red.csv\n"
            "  - data/winequality-white.csv\n"
            "You can get them from the UCI Wine Quality dataset."
        )

    red = pd.read_csv(red_path)
    white = pd.read_csv(white_path)
    red["wine_type"] = 0
    white["wine_type"] = 1
    return pd.concat([red, white], ignore_index=True)

def split_features_target(df: pd.DataFrame, target: str = "quality") -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def _metric_p_param(metric_name: str, p_in: int) -> Dict[str, Any]:
    """
    For scikit-learn KNN: 'minkowski' uses p, 'euclidean' == minkowski(p=2), 'manhattan' == minkowski(p=1).
    We harmonize user input into a consistent model param set.
    """
    if metric_name == "euclidean":
        return {"metric": "euclidean", "p": 2}
    if metric_name == "manhattan":
        return {"metric": "manhattan", "p": 1}
    # minkowski: respect user-supplied p
    return {"metric": "minkowski", "p": p_in}

def _compute_perm_importance(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, task_type: str, feature_names: List[str]) -> pd.DataFrame:
    # Choose a reasonable scoring default
    scoring = "accuracy" if task_type == "Classification" else "r2"
    try:
        r = permutation_importance(
            model, X_test, y_test, scoring=scoring, n_repeats=5, random_state=0, n_jobs=-1
        )
        importances = pd.DataFrame({"feature": feature_names, "importance": r.importances_mean})
        importances = importances.sort_values("importance", ascending=False, ignore_index=True)
    except Exception as e:
        print(f"[perm_importance] failed: {e}")
        importances = pd.DataFrame({"feature": feature_names, "importance": np.zeros(len(feature_names))})
    return importances

def train_knn(
    n_neighbors: int,
    weights: str,
    algorithm: str,
    leaf_size: int,
    p_in: int,
    metric_in: str,
    test_size: float,
    random_state: int,
    task_type: str,
):
    """
    Train a KNN (regressor or classifier) with a StandardScaler in pipeline.
    Return model, metrics, permutation importances, feature list, y_test, preds, metrics_table.
    """
    df = load_wine_data()
    X, y = split_features_target(df)

    if task_type == "Classification":
        y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if task_type == "Classification" else None
    )

    metric_params = _metric_p_param(metric_in, p_in)

    if task_type == "Classification":
        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            **metric_params
        )
    else:
        knn = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            **metric_params
        )

    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("knn", knn),
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

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

    # Permutation importance (model-agnostic)
    importances = _compute_perm_importance(model, X_test, y_test, task_type, X.columns.tolist())

    return (
        model,
        metrics,
        importances,
        X.columns.tolist(),
        y_test.values,
        preds,
        metrics_table
    )

def plot_importances(importances: 'pd.DataFrame') -> plt.Figure:
    fig = plt.figure(figsize=(7, 4.5))
    plt.barh(importances["feature"], importances["importance"])
    plt.gca().invert_yaxis()
    plt.title("Permutation Importance (KNN)")
    plt.xlabel("Importance (mean Δ score)")
    plt.ylabel("Feature")
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    ax.imshow(cm)  # default colormap
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
    n_neighbors: int,
    weights: str,
    algorithm: str,
    leaf_size: int,
    p_in: int,
    metric_in: str,
    # dataset split
    test_size: float,
    random_state: int,
    # task
    task_type: str,
    _state: Dict[str, Any]
):
    (model, metrics, importances, feat_list,
     y_test_arr, preds_arr, metrics_table) = train_knn(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p_in=p_in,
        metric_in=metric_in,
        test_size=test_size,
        random_state=random_state,
        task_type=task_type,
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
        import matplotlib.pyplot as plt
        cm_plot = plt.figure(figsize=(6, 2))
        plt.axis("off")
        cm_note = "Confusion matrix applies to classification only."

    fi_plot = plot_importances(importances)

    return (
        metrics_md,
        metrics_table,
        fi_plot,
        cm_plot,
        cm_note,
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
    return f"Predicted quality: **{pred:.2f}**" if _state.get("task_type") == "Regression" else f"Predicted class: **{int(pred)}**"

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Wine Dataset — K Nearest Neighbors") as demo:
        gr.Markdown(
            """
            # Wine Dataset — K Nearest Neighbors (KNN)

            This interactive demo applies a **KNN** model on the combined [red and white wine dataset from UCI](https://archive.ics.uci.edu/dataset/186/wine+quality).  
            You can experiment with two types of tasks:

            - **Regression**: predicts a *numeric* wine quality score (e.g., 5.8).  
            - **Classification**: predicts a *discrete* quality class (e.g., 5, 6, 7).

            **Note:**  
            Unlike trees or neural nets, **KNN does not learn parameters**. It simply stores the dataset and makes predictions by checking the *k* nearest neighbors in feature space.  
            """
        )

        state = gr.State({})

        with gr.Tab("Fit/Evaluate"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Task")
                    task_type = gr.Radio(choices=list(TASK_MODES.keys()), value="Classification", label="Choose task")
                    task_help = gr.Markdown(TASK_MODES["Classification"]["blurb"])

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Model hyperparameters")
                    n_neighbors = gr.Slider(1, 75, value=15, step=1, label="n_neighbors")
                    weights = gr.Dropdown(choices=["uniform", "distance"], value="uniform", label="weights")
                    algorithm = gr.Dropdown(choices=["auto", "ball_tree", "kd_tree", "brute"], value="auto", label="algorithm")
                    leaf_size = gr.Slider(10, 100, value=30, step=1, label="leaf_size")
                    metric_in = gr.Dropdown(
                        choices=TASK_MODES["Classification"]["criteria"],
                        value="minkowski",
                        label="metric"
                    )
                    p_in = gr.Slider(1, 5, value=2, step=1, label="p (for minkowski)")

                    def _on_task_change(task: str):
                        choices = TASK_MODES[task]["criteria"]
                        return gr.update(choices=choices, value=choices[0]), TASK_MODES[task]["blurb"]

                    task_type.change(_on_task_change, inputs=[task_type], outputs=[metric_in, task_help])

                with gr.Column(scale=1):
                    gr.Markdown("### Dataset split")
                    test_size = gr.Slider(0.1, 0.5, value=0.2, step=0.05, label="test_size")
                    random_state = gr.Number(value=42, precision=0, label="random_state")

            fit_btn = gr.Button("Fit (store dataset and evaluate)", variant="primary")

            metrics_out = gr.Markdown("")
            metrics_table = gr.Dataframe(headers=["metric", "value"], label="Metrics", wrap=True)

            with gr.Tabs():
                with gr.TabItem("Permutation Importance"):
                    fi_plot = gr.Plot(label="Permutation Importances")
                with gr.TabItem("Confusion Matrix"):
                    cm_plot = gr.Plot(label="Confusion Matrix")
                    cm_note = gr.Markdown("")

            fit_btn.click(
                on_train,
                inputs=[ n_neighbors, weights, algorithm, leaf_size, p_in, metric_in, test_size, random_state, task_type, state ],
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

            predict_btn = gr.Button("Predict quality/class", variant="primary")
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
