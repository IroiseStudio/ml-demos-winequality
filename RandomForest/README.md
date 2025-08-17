---
license: apache-2.0
title: RF_WineQuality
sdk: gradio
colorFrom: purple
colorTo: indigo
short_description: Random Forest demo for wine quality prediction
sdk_version: 5.42.0
---

# Wine Dataset — Random Forest

This interactive demo trains and visualizes a **Random Forest** model on the combined [red and white wine dataset from UCI](https://archive.ics.uci.edu/dataset/186/wine+quality).

You can experiment with two types of tasks:

- **Regression**: predicts a _numeric_ wine quality score (e.g., 5.8)
- **Classification**: predicts a _discrete_ quality class (e.g., 5, 6, 7)

---

## What you’ll see after training

- **Metrics**
  - Classification: Accuracy, Precision, Recall, F1
  - Regression: MAE, RMSE, R²
- **Feature importance** rankings
- **Confusion matrix** (for classification tasks)
- **Tree plot** of the first estimator in the forest (for interpretability)

---

## Parameters

- **n_estimators** — number of trees in the forest
- **max_features** — how many features each split considers (`sqrt`, `log2`, or fraction)
- **max_depth** — maximum depth of each tree
- **min_samples_split** — minimum samples to split a node
- **criterion** — splitting criterion (`gini`, `entropy`, `log_loss` for classification; `squared_error`, `absolute_error` for regression)
- **test_size** — percentage of data held out for testing
- **random_state** — random seed for reproducibility
- **plot_max_depth** — depth limit for the tree visualization

---

## Author

**Alban Delamarre**  
[Hugging Face Spaces](https://huggingface.co/AlbanDelamarre)
