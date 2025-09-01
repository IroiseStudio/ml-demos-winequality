---
license: cc
title: NB_WineQuality
sdk: gradio
colorFrom: purple
colorTo: indigo
short_description: Naive Bayes demo for wine quality prediction
sdk_version: 5.42.0
---

# Wine Dataset — Naive Bayes

This interactive demo trains a **Gaussian Naive Bayes** model on the combined [red and white wine dataset from UCI](https://archive.ics.uci.edu/dataset/186/wine+quality).

- **Task**: Classification only (predict discrete wine quality class, e.g., 5, 6, 7)

---

## What you’ll see after training

- **Metrics**: Accuracy, Precision, Recall, F1 (macro)
- **Confusion matrix** (class-level performance)
- **Feature relevance** via ANOVA F-scores (quick univariate importance proxy)

---

## Parameters

- **log10(var_smoothing)** — controls numeric stability (`10^x`)
- **use_uniform_priors** — toggle uniform vs. learned class priors
- **test_size** — percentage of data held out for testing
- **random_state** — random seed for reproducibility

---

## Author

**Alban Delamarre**  
[Hugging Face Spaces](https://huggingface.co/AlbanDelamarre)
