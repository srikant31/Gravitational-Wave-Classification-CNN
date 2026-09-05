"""Plotting utilities: class samples, training curves, confusion matrix."""

import matplotlib.image as mpimg
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from . import config


def plot_class_samples(class_names: list) -> None:
    n_cols = 3
    n_rows = int(np.ceil(len(class_names) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    plt.subplots_adjust(hspace=0.5, top=0.92, bottom=0.08)
    axes_flat = axes.flatten() if n_rows > 1 else [axes]

    for index, class_name in enumerate(class_names):
        ax = axes_flat[index]
        class_dir = config.TRAIN_DIR / class_name
        if not class_dir.is_dir():
            continue

        images = [p for p in class_dir.iterdir() if p.is_file()]
        if not images:
            continue

        image = mpimg.imread(images[0])
        ax.imshow(image)
        ax.set_title(class_name)
        ax.axis("off")

    for i in range(len(class_names), len(axes_flat)):
        fig.delaxes(axes_flat[i])

    fig.suptitle("Type of Gravitational Wave Detections", fontsize=24)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIGURES_DIR / "gw_types.png", dpi=100)
    plt.close()


def plot_accuracy_loss(history) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.plot(history.history["accuracy"])
    ax1.plot(history.history["val_accuracy"])
    ax1.set_title("Model Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend(["Train", "Validation"], loc="lower right")

    ax2.plot(history.history["loss"])
    ax2.plot(history.history["val_loss"])
    ax2.set_title("Model Loss")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend(["Train", "Validation"], loc="upper right")

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIGURES_DIR / "model_accuracy_loss_plot.png", dpi=100)
    plt.close()


def plot_confusion_matrix(y_true: list, y_pred: list, class_names: list) -> None:
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / np.sum(cm)
    title = f"Confusion Matrix for Predictions - Accuracy: {accuracy:.2%}"

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        cm, annot=True, fmt="g", cmap="Blues", xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIGURES_DIR / "confusion_matrix_plot.png", dpi=100)
    plt.close()
