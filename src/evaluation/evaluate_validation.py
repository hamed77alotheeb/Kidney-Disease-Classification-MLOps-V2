"""Evaluate the fine-tuned classifier on the validation split only."""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file) or {}

    data_config = config.get("data", {})
    training_config = config.get("training", {})
    evaluation_config = config.get("evaluation", {})

    class_names = list(data_config.get("classes", []))
    expected_classes = ["Cyst", "Normal", "Stone", "Tumor"]

    if class_names != expected_classes:
        raise ValueError("Unexpected class list or class order.")

    split_dir = PROJECT_ROOT / data_config.get("split_dir", "data/splits")
    validation_name = evaluation_config.get("validation_split", "val")

    if validation_name != "val":
        raise ValueError("This script is restricted to the val split.")

    validation_dir = split_dir / "val"
    model_path = PROJECT_ROOT / evaluation_config.get(
        "validation_model",
        "models/best_finetuned_model.h5",
    )

    if not validation_dir.is_dir():
        raise FileNotFoundError(
            "Validation directory not found: {}".format(validation_dir)
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            "Model file not found: {}".format(model_path)
        )

    image_size = tuple(training_config.get("image_size", [224, 224]))
    batch_size = int(training_config.get("batch_size", 16))

    if len(image_size) != 2 or batch_size < 1:
        raise ValueError("Invalid image size or batch size.")

    generator = ImageDataGenerator(
        preprocessing_function=preprocess_input
    ).flow_from_directory(
        directory=str(validation_dir),
        classes=class_names,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False,
        interpolation="bilinear",
    )

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
    )

    output_shape = model.output_shape
    if len(output_shape) != 2 or output_shape[-1] != len(class_names):
        raise ValueError(
            "Model output does not match the four expected classes."
        )

    probabilities = np.asarray(
        model.predict(generator, verbose=1),
        dtype=np.float64,
    )

    if (
        probabilities.shape != (generator.samples, len(class_names))
        or not np.isfinite(probabilities).all()
    ):
        raise RuntimeError("The model returned invalid predictions.")

    y_true = generator.classes.astype(np.int64)
    y_pred = np.argmax(probabilities, axis=1)
    labels = list(range(len(class_names)))

    class_report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    result = {
        "status": "completed",
        "evaluation_split": "val",
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "num_images": int(len(y_true)),
        "class_names": class_names,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix_actual_rows_predicted_columns": (
            confusion_matrix(y_true, y_pred, labels=labels).tolist()
        ),
        "classification_report": class_report,
        "test_split_loaded": False,
        "note": (
            "Validation data has been used during model development. "
            "This report is not an independent test estimate."
        ),
    }

    report_path = (
        PROJECT_ROOT
        / "reports"
        / "metrics"
        / "validation_evaluation_summary.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Validation report saved:", report_path)
    print("The test split was not loaded.")


if __name__ == "__main__":
    main()
