import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

def compute_apcer_bpcer(y_true, y_pred, real_label="Real"):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    is_real_true = y_true == real_label
    is_real_pred = y_pred == real_label
    spoof_mask = ~is_real_true
    apcer = (is_real_pred & spoof_mask).sum() / max(spoof_mask.sum(), 1)
    bpcer = ((~is_real_pred) & is_real_true).sum() / max(is_real_true.sum(), 1)
    return float(apcer), float(bpcer)

def compute_acer(apcer, bpcer):
    return (apcer + bpcer) / 2.0

def compute_auc(y_true_binary, y_score):
    return float(roc_auc_score(y_true_binary, y_score))

def compute_confusion_matrix(y_true, y_pred, labels):
    return confusion_matrix(y_true, y_pred, labels=labels)