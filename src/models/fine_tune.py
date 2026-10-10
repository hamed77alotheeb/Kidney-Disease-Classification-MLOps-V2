import argparse
import shutil
from pathlib import Path
from datetime import datetime
import json
import random
import sys

import numpy as np
import tensorflow as tf
import yaml
from scipy.optimize import minimize_scalar


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from cnnClassifier.components.data_loader import CLASSES, create_data_generators


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def stable_softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(logits)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def calibrate_temperature(probabilities, labels):
    eps = 1e-7
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if len(labels) == 0 or len(probabilities) != len(labels):
        raise ValueError("Calibration predictions and labels do not match.")

    logits = np.log(np.clip(probabilities, eps, 1.0))

    def nll(log_temperature):
        temperature = float(np.exp(log_temperature))
        calibrated = stable_softmax(logits / temperature)
        correct = calibrated[np.arange(len(labels)), labels]
        return float(-np.mean(np.log(np.clip(correct, eps, 1.0))))

    result = minimize_scalar(
        nll,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
    )

    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Temperature calibration failed.")

    temperature = float(np.exp(result.x))
    calibrated = stable_softmax(logits / temperature)
    return temperature, calibrated, float(result.fun)


def main(force=False):
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    data_config = config.get("data", {})
    train_config = config.get("training", {})

    seed = int(train_config.get("seed", 42))
    epochs = int(train_config.get("fine_tune_epochs", 15))

    # A smaller batch is selected for the 4 GB GPU.
    batch_size = int(train_config.get("fine_tune_batch_size", 2))
    learning_rate = float(
        train_config.get("fine_tune_learning_rate", 1e-5)
    )
    image_size = tuple(train_config.get("image_size", [224, 224]))

    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Invalid fine-tuning configuration.")

    split_dir = PROJECT_ROOT / data_config.get("split_dir", "data/splits")

    # The previously trained model is the starting point.
    original_path = PROJECT_ROOT / "models" / "best_candidate_model.h5"

    # All outputs are separate; the original model will not be overwritten.
    output_path = PROJECT_ROOT / "models" / "best_finetuned_model.h5"
    history_path = PROJECT_ROOT / "logs" / "fine_tune_history.csv"
    summary_path = (
        PROJECT_ROOT / "reports" / "metrics" / "fine_tune_summary.json"
    )
    calibration_path = (
        PROJECT_ROOT / "models" / "fine_tuned_confidence_calibration.json"
    )

    if not original_path.is_file():
        raise FileNotFoundError(
            "Original trained model not found: {}".format(original_path)
        )

    output_paths = (
        output_path,
        history_path,
        summary_path,
        calibration_path,
    )
    existing_outputs = [path for path in output_paths if path.exists()]

    if existing_outputs and not force:
        raise FileExistsError(
            "Previous fine-tuning outputs exist. Nothing was overwritten. "
            "Use --force to archive existing outputs before a new run."
        )

    if existing_outputs:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        archive_root = PROJECT_ROOT / "archive" / (
            "fine_tune_run_" + run_id
        )

        for path in existing_outputs:
            destination = archive_root / path.relative_to(PROJECT_ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))
            print(
                "Archived previous output: {}".format(destination),
                flush=True,
            )

    for directory in (
        output_path.parent,
        history_path.parent,
        summary_path.parent,
        calibration_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    set_seeds(seed)

    print("=" * 68)
    print("FINE-TUNING THE EXISTING MODEL")
    print("=" * 68)
    print("Starting model:", original_path)
    print("New model:", output_path)
    print("Batch size:", batch_size)
    print("Maximum epochs:", epochs)
    print("Learning rate:", learning_rate)
    print("Test split: NOT LOADED")
    print("Original model: will remain unchanged")
    print("=" * 68)

    # This loader accesses train and validation directories only.
    data = create_data_generators(
        data_dir=split_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        include_test=False,
        include_calibration=True,
    )

    model = tf.keras.models.load_model(
        str(original_path),
        compile=False,
    )

    backbone = model.get_layer("vgg16")

    # Unfreeze only the convolutional layers in block 5.
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = layer.name in {
            "block5_conv1",
            "block5_conv2",
            "block5_conv3",
        }

    # Recompile AFTER changing trainable flags.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    trainable_params = int(
        sum(np.prod(weight.shape) for weight in model.trainable_weights)
    )

    print("\nTrainable parameters:", f"{trainable_params:,}")
    print("Unfrozen VGG16 layers:")
    for layer in backbone.layers:
        if layer.trainable:
            print("  ", layer.name)

    if trainable_params <= 2052:
        raise RuntimeError(
            "The VGG16 block-5 layers were not unfrozen as expected."
        )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(history_path),
            append=False,
        ),
    ]

    history = model.fit(
        data["train"],
        validation_data=data["val"],
        epochs=epochs,
        class_weight=data["class_weights"],
        callbacks=callbacks,
        verbose=1,
    )

    val_accuracy = np.asarray(
        history.history["val_accuracy"], dtype=np.float64
    )
    val_loss = np.asarray(
        history.history["val_loss"], dtype=np.float64
    )

    if len(val_accuracy) == 0 or len(val_loss) != len(val_accuracy):
        raise RuntimeError("Training history is incomplete.")

    if (
        not np.isfinite(val_accuracy).all()
        or not np.isfinite(val_loss).all()
    ):
        raise RuntimeError("Non-finite validation metrics detected.")

    best_index = int(np.argmax(val_accuracy))
    best_accuracy = float(val_accuracy[best_index])

    # Load the checkpoint selected by validation accuracy.
    best_model = tf.keras.models.load_model(
        str(output_path),
        compile=False,
    )

    calibration_generator = data["calibration"]
    calibration_generator.reset()

    calibration_probs = best_model.predict(
        calibration_generator,
        verbose=1,
    )
    calibration_labels = np.asarray(
        calibration_generator.classes,
        dtype=np.int64,
    )

    temperature, calibrated_probs, calibration_nll = calibrate_temperature(
        calibration_probs,
        calibration_labels,
    )

    calibration_summary = {
        "method": "temperature_scaling",
        "temperature": temperature,
        "calibration_samples": int(len(calibration_labels)),
        "calibration_accuracy": float(
            np.mean(
                np.argmax(calibrated_probs, axis=1) == calibration_labels
            )
        ),
        "negative_log_likelihood": calibration_nll,
        "mean_calibrated_confidence": float(
            np.mean(np.max(calibrated_probs, axis=1))
        ),
        "class_names": list(CLASSES),
        "source_model": str(
            output_path.relative_to(PROJECT_ROOT)
        ),
        "test_split_used": False,
        "note": (
            "Calibration confidence is not a guarantee of correctness."
        ),
    }

    with calibration_path.open("w", encoding="utf-8") as file:
        json.dump(calibration_summary, file, indent=2)

    summary = {
        "status": "completed",
        "source_model": str(original_path.relative_to(PROJECT_ROOT)),
        "fine_tuned_model": str(output_path.relative_to(PROJECT_ROOT)),
        "epochs_requested": epochs,
        "epochs_completed": int(len(val_accuracy)),
        "best_epoch": best_index + 1,
        "best_validation_accuracy": best_accuracy,
        "best_validation_accuracy_percent": round(
            best_accuracy * 100, 4
        ),
        "best_validation_loss": float(val_loss[best_index]),
        "original_model_overwritten": False,
        "test_split_loaded": False,
        "test_split_evaluated": False,
        "calibration_file": str(
            calibration_path.relative_to(PROJECT_ROOT)
        ),
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nFINE-TUNING COMPLETE")
    print(json.dumps(summary, indent=2))
    print("\nCONFIDENCE CALIBRATION")
    print(json.dumps(calibration_summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune the existing kidney image classifier."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Archive previous outputs before starting another run.",
    )
    args = parser.parse_args()
    main(force=args.force)
