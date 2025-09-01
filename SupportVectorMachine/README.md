---
license: cc
title: SVM_WineQuality
sdk: gradio
colorFrom: purple
colorTo: indigo
short_description: Support Vector Machine demo for wine quality prediction
sdk_version: 5.42.0
---

# Wine Dataset — Support Vector Machine (SVM)

This interactive demo trains and visualizes a **Support Vector Machine (SVM)** model on the combined [red and white wine dataset from UCI](https://archive.ics.uci.edu/dataset/186/wine+quality).

You can experiment with two types of tasks:

- **Regression**: predicts a _numeric_ wine quality score (e.g., 5.8)
- **Classification**: predicts a _discrete_ quality class (e.g., 5, 6, 7)

---

## What you’ll see after training

- **Metrics**
  - Classification: Accuracy, Precision, Recall, F1
  - Regression: MAE, RMSE, R²
- **Permutation importance** (model-agnostic feature effects)
- **Confusion matrix** (for classification tasks)

---

## Parameters

- **kernel** — type of SVM kernel (`linear`, `rbf`, `poly`, `sigmoid`)
- **C** — regularization strength
- **gamma** — kernel coefficient (`scale`, `auto`) for `rbf/poly/sigmoid`
- **degree** — degree for polynomial kernel
- **epsilon** — margin of tolerance for regression (SVR)
- **class_weight** — weighting strategy (`None`, `balanced`) for classification
- **test_size** — percentage of data held out for testing
- **random_state** — random seed for reproducibility

---

## Author

**Alban Delamarre**  
[Hugging Face Spaces](https://huggingface.co/AlbanDelamarre)
