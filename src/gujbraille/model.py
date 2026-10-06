"""CNN cell classifier.

Architecture (nine Keras layers; 117,440 trainable parameters for 64 outputs):

    Input 28x28x1
    Conv2D(32, 3x3, ReLU)  -> 26x26x32
    MaxPool(2x2)           -> 13x13x32
    Conv2D(64, 3x3, ReLU)  -> 11x11x64
    MaxPool(2x2)           ->  5x5x64
    Conv2D(128, 3x3, ReLU) ->  3x3x128
    MaxPool(2x2)           ->  1x1x128
    Flatten                ->  128
    Dense(128, ReLU)       ->  128
    Dense(64, softmax)     ->  64   (one output per six-dot pattern, 0 = blank)

The output index *is* the dot-pattern bit-mask, so the network predicts the
Braille cell; the Gujarati character is obtained afterwards by the rule-based
mapper (transliterate.braille_to_gujarati), which needs the context of the
neighbouring cells.
"""
from __future__ import annotations

N_CLASSES = 64


def build_cnn(n_classes: int = N_CLASSES, input_size: int = 28, lr: float = 1e-3):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    m = models.Sequential(name="gujbraille_cnn")
    m.add(layers.Input((input_size, input_size, 1)))
    m.add(layers.Conv2D(32, 3, activation="relu"))
    m.add(layers.MaxPooling2D(2))
    m.add(layers.Conv2D(64, 3, activation="relu"))
    m.add(layers.MaxPooling2D(2))
    m.add(layers.Conv2D(128, 3, activation="relu"))
    m.add(layers.MaxPooling2D(2))
    m.add(layers.Flatten())
    m.add(layers.Dense(128, activation="relu"))
    m.add(layers.Dense(n_classes, activation="softmax"))
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def augmentation(rotation_deg: float = 3.0, shift: float = 0.05, zoom: float = 0.05):
    from tensorflow.keras import layers, models
    return models.Sequential([
        layers.RandomRotation(rotation_deg / 360.0, fill_mode="nearest"),
        layers.RandomTranslation(shift, shift, fill_mode="nearest"),
        layers.RandomZoom(zoom, fill_mode="nearest"),
    ], name="augmentation")
