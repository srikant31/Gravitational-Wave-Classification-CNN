"""Train and evaluate the gravitational wave classification CNN.

Run from the repository root with:

    python -m pywavecnn.train

Fixes vs. the original ``main.py``:
  * The dataset re-download/re-split step was hardcoded behind ``if True:``
    despite a comment saying "once run once you can set this to False" —
    ``check_files()`` was imported but never actually called. This now
    calls ``prepare_dataset()``, which itself checks ``dataset_ready()``
    and skips re-downloading unless ``--force-redownload`` is passed.
  * Image size, batch size, epoch count and the random seed are CLI flags
    instead of magic numbers buried in the function body.
  * The tf.data pipelines use ``.cache()``/``.prefetch(AUTOTUNE)`` so
    training doesn't re-decode images from disk every epoch.
"""

import argparse
import logging
import os

import numpy as np
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

from . import config
from .data import prepare_dataset
from .models import build_cnn_model, get_early_stopping
from .plotting import plot_accuracy_loss, plot_class_samples, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PyWaveCNN gravitational wave classifier.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, nargs=2, default=(400, 400), metavar=("H", "W"))
    parser.add_argument("--seed", type=int, default=220301)
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Re-download and re-split the dataset even if a prepared split already exists.",
    )
    return parser.parse_args()


def build_datasets(img_size: tuple, batch_size: int, seed: int):
    autotune = tf.data.AUTOTUNE

    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR, seed=seed, image_size=img_size, shuffle=True, batch_size=batch_size
    )
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        config.VALID_DIR, seed=seed, image_size=img_size, batch_size=batch_size
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR, seed=seed, image_size=img_size, batch_size=batch_size
    )

    class_names = train_ds.class_names

    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    valid_ds = valid_ds.cache().prefetch(buffer_size=autotune)
    test_ds = test_ds.prefetch(buffer_size=autotune)

    return train_ds, valid_ds, test_ds, class_names


def main():
    args = parse_args()
    cfg = config.TrainingConfig(
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )

    prepare_dataset(cfg, force=args.force_redownload)

    train_ds, valid_ds, test_ds, class_names = build_datasets(
        cfg.img_size, cfg.batch_size, cfg.seed
    )
    plot_class_samples(class_names)

    model = build_cnn_model(cfg.img_size, num_classes=len(class_names))
    model.summary()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer="rmsprop",
        metrics=["accuracy"],
    )

    early_stopping = get_early_stopping(
        patience=cfg.early_stopping_patience, baseline=cfg.early_stopping_baseline
    )

    history = model.fit(
        train_ds, epochs=cfg.epochs, validation_data=valid_ds, callbacks=[early_stopping]
    )
    plot_accuracy_loss(history)

    scores = model.evaluate(test_ds)
    final_accuracy = scores[1]

    y_pred, y_true = [], []
    for img_batch, label_batch in test_ds:
        preds = model.predict(img_batch)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(label_batch.numpy())

    plot_confusion_matrix(y_true, y_pred, class_names)
    logger.info("Final test accuracy: %.2f%%", final_accuracy * 100)


if __name__ == "__main__":
    main()
