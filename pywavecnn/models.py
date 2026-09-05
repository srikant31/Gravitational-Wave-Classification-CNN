"""CNN model architecture for gravitational wave classification."""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


def build_cnn_model(img_size: tuple, num_classes: int) -> tf.keras.Model:
    """Build the classification CNN.

    ``num_classes`` is now a parameter instead of a hardcoded ``22`` in the
    final Dense layer, so the model always matches whatever classes are
    actually present in the training data directory.
    """
    inputs = tf.keras.Input(shape=img_size + (3,))

    x = layers.Resizing(img_size[0], img_size[1])(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="pywave_cnn")


def get_early_stopping(patience: int = 5, baseline: float = 0.95) -> EarlyStopping:
    return EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        mode="max",
        restore_best_weights=True,
        verbose=1,
        baseline=baseline,
    )
