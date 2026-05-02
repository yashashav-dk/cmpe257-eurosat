"""Plot helpers."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from src.config import CLASS_NAMES, NUM_CLASSES

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 120,
})


def plot_confusion_matrix(cm, title="Confusion Matrix", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_roc_curves(y_true, y_prob, title="ROC Curves (OvR)", save_path=None):
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=1.5, label=f"{CLASS_NAMES[i]} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_training_curves(history, title="Training", save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history["train_loss"], label="Train", color="blue")
    ax1.plot(epochs, history["val_loss"], label="Val", color="red")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title(f"{title} — Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["train_acc"], label="Train", color="blue")
    ax2.plot(epochs, history["val_acc"], label="Val", color="red")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title(f"{title} — Acc"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_optimizer_comparison(histories, metric="val_acc", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, h in histories.items():
        epochs = range(1, len(h[metric]) + 1)
        ax.plot(epochs, h[metric], label=name, lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Optimizer Comparison — {metric.replace('_', ' ').title()}")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_bar_comparison(names, values, ylabel="Accuracy", title="Models",
                        errors=None, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    bars = ax.bar(x, values, yerr=errors, capsize=4,
                  color=sns.color_palette("viridis", len(names)))
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
