import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, brier_score_loss
)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    pred = (y_prob >= 0.5).astype(int)
    conf = np.maximum(y_prob, 1 - y_prob)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i+1])
        if np.any(mask):
            ece += np.mean(mask) * abs(np.mean(correct[mask]) - np.mean(conf[mask]))
    return float(ece)

def find_best_threshold(y_true, y_prob):
    best_thr, best_bal = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 181):
        pred = (y_prob >= thr).astype(int)
        bal = balanced_accuracy_score(y_true, pred)
        if bal > best_bal:
            best_thr, best_bal = float(thr), float(bal)
    return best_thr, best_bal

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["roc_auc"] = np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["fpr"] = fp / (fp + tn + 1e-9)
    metrics["fnr"] = fn / (fn + tp + 1e-9)
    return metrics
