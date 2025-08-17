# ML Demos — Wine Quality


This repository contains interactive ML demos built in Python with **Gradio** and designed for deployment on **Hugging Face Spaces**.  
Each demo uses the [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality).

## Demos

- **Decision Tree**  
  Train and visualize a Decision Tree for wine quality prediction.  
  [Open on Hugging Face](https://huggingface.co/spaces/AlbanDelamarre/DT_WineQuality)

- **Random Forest**  
  Train and evaluate a Random Forest, explore feature importances and metrics.  
  [Open on Hugging Face](https://huggingface.co/spaces/AlbanDelamarre/RF_WineQuality)

---

## Running Locally

Each demo folder contains its own `app.py`, `requirements.txt`, and `README.md`.

```bash
cd DecisionTree
pip install -r requirements.txt
python app.py