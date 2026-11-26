from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def save_confusion_matrix(cm, labels, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_roc_curve(y_true, y_score, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)