from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


CLASSES = ("Cyst", "Normal", "Stone", "Tumor")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


def create_data_generators(
    data_dir: Path,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    seed: int = 42,
    include_test: bool = True,
    include_calibration: bool = False,
) -> Dict[str, Any]:
    """
    Create train, validation, and test generators.

    VGG16 preprocessing is applied consistently to all three splits.
    Augmentation is applied only to training images.

    This function does not train or modify the dataset.
    """

    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}"
        )

    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")

    if (
        len(image_size) != 2
        or any(not isinstance(value, int) or value < 1
               for value in image_size)
    ):
        raise ValueError(
            "image_size must contain two positive integers"
        )

    split_names = ("train", "val", "test") if include_test else ("train", "val")

    # Verify the expected class directories before loading.
    for split_name in split_names:
        split_dir = data_dir / split_name

        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Missing split directory: {split_dir}"
            )

        for class_name in CLASSES:
            class_dir = split_dir / class_name

            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing class directory: {class_dir}"
                )

            image_files = [
                path for path in class_dir.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                )
            ]

            if not image_files:
                raise ValueError(
                    f"No images found in {class_dir}"
                )

    # Augmentation is limited to training data.
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=8,
        width_shift_range=0.03,
        height_shift_range=0.03,
        zoom_range=0.08,
        brightness_range=(0.85, 1.15),
        horizontal_flip=True,
    )

    # Validation images are never augmented.
    # When requested, split validation into two non-overlapping subsets:
    # one for model selection and one reserved for confidence calibration.
    common_args = {
        "target_size": image_size,
        "batch_size": batch_size,
        "class_mode": "sparse",
        "classes": list(CLASSES),
        "interpolation": "bilinear",
    }

    train_generator = train_datagen.flow_from_directory(
        directory=str(data_dir / "train"),
        shuffle=True,
        seed=seed,
        **common_args
    )

    calibration_generator = None

    if include_calibration:
        split_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=0.5,
        )

        val_generator = split_datagen.flow_from_directory(
            directory=str(data_dir / "val"),
            subset="training",
            shuffle=False,
            seed=seed,
            **common_args
        )

        calibration_generator = split_datagen.flow_from_directory(
            directory=str(data_dir / "val"),
            subset="validation",
            shuffle=False,
            seed=seed,
            **common_args
        )
    else:
        evaluation_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

        val_generator = evaluation_datagen.flow_from_directory(
            directory=str(data_dir / "val"),
            shuffle=False,
            **common_args
        )

    test_generator = None
    if include_test:
        # Test images use a separate, non-augmented loader.
        # Training will explicitly call this function with include_test=False.
        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        test_generator = test_datagen.flow_from_directory(
            directory=str(data_dir / "test"),
            shuffle=False,
            **common_args
        )

    # Lock the class-index mapping so labels remain consistent.
    expected_mapping = {
        class_name: index
        for index, class_name in enumerate(CLASSES)
    }

    generators_to_check = [
        ("train", train_generator),
        ("val", val_generator),
    ]
    if include_calibration:
        generators_to_check.append(("calibration", calibration_generator))
    if include_test:
        generators_to_check.append(("test", test_generator))

    for split_name, generator in generators_to_check:
        if generator.class_indices != expected_mapping:
            raise ValueError(
                f"Unexpected class mapping in {split_name}: "
                f"{generator.class_indices}"
            )

    # Balance class contributions during training.
    class_ids = np.arange(len(CLASSES))

    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_ids,
        y=train_generator.classes,
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(class_ids, weights)
    }

    result = {
        "train": train_generator,
        "val": val_generator,
        "class_weights": class_weights,
        "class_names": list(CLASSES),
        "image_size": image_size,
        "batch_size": batch_size,
    }
    if include_calibration:
        result["calibration"] = calibration_generator
    if include_test:
        result["test"] = test_generator

    return result
