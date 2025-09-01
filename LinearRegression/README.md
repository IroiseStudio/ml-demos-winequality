---
license: cc
title: Linear_WineQuality
sdk: gradio
colorFrom: purple
colorTo: indigo
short_description: Linear models demo for wine quality prediction
sdk_version: 5.42.0
---

# Wine Dataset — Linear Models (Linear, Ridge, Lasso, Logistic)

This interactive demo trains and visualizes **linear models** on the combined [red and white wine dataset from UCI](https://archive.ics.uci.edu/dataset/186/wine+quality).

You can experiment with two types of tasks:

- **Regression**: predicts a _numeric_ wine quality score (e.g., 5.8)  
  Models: `LinearRegression`, `Ridge`, `Lasso`
- **Classification**: predicts a _discrete_ quality class (e.g., 5, 6, 7)  
  Model: `LogisticRegression`

---

## What you’ll see after training

- **Metrics**
  - Classification: Accuracy, Precision, Recall, F1
  - Regression: MAE, RMSE, R²
- **Coefficient magnitudes** (top features)
- **Confusion matrix** (for classification tasks)
- **Residuals plot** (for regression tasks)

---

## Parameters

- **model** — choice of linear algorithm (`LinearRegression`, `Ridge`, `Lasso`, `LogisticRegression`)
- **alpha** — regularization strength for Ridge/Lasso
- **C** — inverse regularization strength for LogisticRegression
- **standardize** — apply `StandardScaler` before fitting
- **test_size** — percentage of data held out for testing
- **random_state** — random seed for reproducibility

> Note: Standardization is recommended for most linear models.

---

## Author

**Alban Delamarre**  
[Hugging Face Spaces](https://huggingface.co/AlbanDelamarre)
