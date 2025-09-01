---
license: cc
title: MLP_WineQuality
sdk: gradio
colorFrom: purple
colorTo: indigo
short_description: MLP (neural net) demo for wine quality prediction
sdk_version: 5.42.0
---

# Wine Dataset — MLP (Neural Network)

This interactive demo trains and visualizes a **Multilayer Perceptron (MLP)** on the combined [red and white wine dataset from UCI](https://archive.ics.uci.edu/dataset/186/wine+quality).

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
- **Loss curve** (training convergence)

---

## Parameters

- **hidden_layer_sizes** — comma-separated sizes (e.g., `64,32`)
- **activation** — nonlinearity (`relu`, `tanh`, `logistic`, `identity`)
- **alpha** — L2 regularization strength
- **learning_rate_init** — initial learning rate
- **batch_size** — minibatch size
- **max_iter** — maximum training iterations
- **early_stopping** — stop when validation score stops improving
- **test_size** — percentage of data held out for testing
- **random_state** — random seed for reproducibility

> Note: Inputs are standardized with `StandardScaler` for stable MLP training.

---

## Author

**Alban Delamarre**  
[Hugging Face Spaces](https://huggingface.co/AlbanDelamarre)
