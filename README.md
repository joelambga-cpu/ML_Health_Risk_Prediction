# ML Health Risk Prediction — Educational Prototype

A neural-network learning exercise with TensorFlow, PyTorch and scikit-learn. This is not a production-ready system or a clinically validated risk model.

## What the code does

The loader uses scikit-learn's bundled diabetes regression dataset and converts its continuous target to a binary label: above versus at or below the dataset-wide mean. This label is an exercise-specific threshold, not a diagnosis or a clinical risk category.

The main script splits the data, fits a standard scaler on the training features, trains a TensorFlow network and prints test accuracy. A separate PyTorch training implementation is included but is not run by the main script. The inference helper is a minimal TensorFlow-style prediction wrapper, not a deployed service.

## Installation and usage

From a terminal with Python available:

```powershell
git clone https://github.com/joelambga-cpu/ML_Health_Risk_Prediction.git
cd ML_Health_Risk_Prediction/ml_health_risk
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

On macOS/Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

Use a Python version supported by the TensorFlow and PyTorch releases you install. Dependencies are not pinned.

## Limitations

The target threshold is computed before the train/test split and uses the full target distribution. This evaluation design needs revision before making generalization claims. A single split and accuracy alone do not establish robust performance. Training randomness is not fully controlled; model persistence, deployment and clinical validation are not implemented.

[Source code](ml_health_risk/src/) · [Dependencies](ml_health_risk/requirements.txt)

Author: Giuseppe Agnello
