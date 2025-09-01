---
license: cc
title: KNN_WineQuality
sdk: gradio
colorFrom: purple
colorTo: indigo
short_description: K Nearest Neighbors demo for wine quality prediction
sdk_version: 5.42.0
---

# Wine Dataset — K Nearest Neighbors (KNN)

This interactive demo applies a **K Nearest Neighbors (KNN)** model to the combined [red and white wine dataset from UCI](https://archive.ics.uci.edu/dataset/186/wine+quality).

You can experiment with two types of tasks:

- **Regression**: predicts a _numeric_ wine quality score (e.g., 5.8)
- **Classification**: predicts a _discrete_ quality class (e.g., 5, 6, 7)

---

## What you’ll see after fitting

- **Metrics**
  - Classification: Accuracy, Precision, Recall, F1
  - Regression: MAE, RMSE, R²
- **Permutation importance** (model-agnostic feature effects)
- **Confusion matrix** (for classification tasks)

---

## Parameters

- **n_neighbors** — number of nearest neighbors to consider
- **weights** — `uniform` (all neighbors equal) or `distance` (closer neighbors count more)
- **metric** — distance metric (`euclidean`, `manhattan`, `minkowski`)
- **p** — Minkowski power parameter (`p=1` = Manhattan, `p=2` = Euclidean)
- **algorithm** — search method (`auto`, `ball_tree`, `kd_tree`, `brute`)
- **leaf_size** — affects tree-based search speed
- **test_size** — percentage of data held out for evaluation
- **random_state** — random seed for reproducibility

> Note: Inputs are standardized with `StandardScaler` since KNN is distance-based.  
> KNN does not learn parameters; it stores the dataset and predicts based on the _k_ nearest stored samples.

---

## Author

**Alban Delamarre**  
[Hugging Face Spaces](https://huggingface.co/AlbanDelamarre)
