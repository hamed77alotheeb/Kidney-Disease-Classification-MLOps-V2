from typing import Optional, Tuple

import tensorflow as tf


def build_vgg16_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    weights: Optional[str] = "imagenet",
    freeze_backbone: bool = True,
    dropout_rate: float = 0.35,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Build a VGG16-based image classifier.

    Classes are expected to follow the data loader mapping:
        Cyst   -> 0
        Normal -> 1
        Stone  -> 2
        Tumor  -> 3

    Input images must use the VGG16 preprocessing function
    provided by cnnClassifier.components.data_loader.

    Args:
        input_shape: Image dimensions in channels-last format.
        num_classes: Number of output classes.
        weights: "imagenet" for pretrained weights or None
                 for random initialization.
        freeze_backbone: Whether to freeze the VGG16 feature extractor.
        dropout_rate: Dropout probability before the classification head.
        learning_rate: Optimizer learning rate.

    Returns:
        A compiled TensorFlow Keras model.

    This function builds and compiles a model but does not train it.
    """

    if len(input_shape) != 3:
        raise ValueError(
            "input_shape must contain height, width, and channels."
        )

    if any(dimension <= 0 for dimension in input_shape):
        raise ValueError("All input dimensions must be positive.")

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in the range [0, 1).")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    inputs = tf.keras.Input(
        shape=input_shape,
        name="image"
    )

    backbone = tf.keras.applications.VGG16(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )

    backbone.trainable = not freeze_backbone

    features = backbone(inputs)

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(features)

    x = tf.keras.layers.Dropout(
        rate=dropout_rate,
        name="classification_dropout"
    )(x)

    outputs = tf.keras.layers.Dense(
        units=num_classes,
        activation="softmax",
        name="predictions"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="kidney_vgg16_classifier"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model
