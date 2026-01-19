import numpy as np

def accuracy(preds, y):
    preds = (preds > 0.5).astype(int)
    return (preds == y).mean()
