"""Unified metrics + multi-seed aggregation."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
)
from src.config import CLASS_NAMES


def compute_metrics(y_true, y_pred, y_prob=None):
    m = {}
    m["accuracy"] = accuracy_score(y_true, y_pred)
    m["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    m["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    m["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    m["per_class_f1"] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    m["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    if y_prob is not None:
        try:
            m["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            m["roc_auc"] = None
    else:
        m["roc_auc"] = None
    m["classification_report"] = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
    )
    return m


def print_metrics(metrics, model_name="Model"):
    print(f"\n{'='*60}\n  Results: {model_name}\n{'='*60}")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"  ROC-AUC:         {metrics['roc_auc']:.4f}")
    print("\nPer-class F1:")
    for n, f1 in zip(CLASS_NAMES, metrics['per_class_f1']):
        print(f"    {n:25s} {f1:.4f}")


def metrics_to_dataframe(results_dict):
    rows = []
    for name, m in results_dict.items():
        rows.append({
            "Model": name,
            "Accuracy": m.get("accuracy"),
            "Macro_F1": m.get("macro_f1"),
            "ROC_AUC": m.get("roc_auc"),
            "Train_Time_sec": m.get("train_time"),
        })
    return pd.DataFrame(rows)


def aggregate_multi_seed(seed_metrics_list):
    agg = {}
    for k in ["accuracy", "macro_f1", "macro_precision", "macro_recall", "roc_auc"]:
        vals = [m[k] for m in seed_metrics_list if m.get(k) is not None]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
    return agg
