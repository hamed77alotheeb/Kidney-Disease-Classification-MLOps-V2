from pathlib import Path
from datetime import datetime
import json
import random
import shutil
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
from cnnClassifier.components.model_builder import build_vgg16_classifier


VALIDATION_THRESHOLD = 0.95
INFERENCE_CONFIDENCE_THRESHOLD = 0.98


def configure_environment(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def archive_previous_outputs(paths, run_id):
    existing = [Path(p) for p in paths if Path(p).exists()]
    if not existing:
        return

    archive_root = PROJECT_ROOT / "archive" / f"training_run_{run_id}"

    for path in existing:
        destination = archive_root / path.relative_to(PROJECT_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
        print(f"Archived previous output: {destination}", flush=True)


def stable_softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(logits)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def calibrate_temperature(probabilities, labels):
    """
    Fit one temperature on the dedicated calibration subset.
    The independent test split is never used.
    """
    eps = 1e-7
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if probabilities.ndim != 2:
        raise ValueError("Expected a two-dimensional probability array.")
    if len(probabilities) != len(labels) or len(labels) == 0:
        raise ValueError("Calibration predictions and labels do not match.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite calibration probabilities detected.")

    clipped = np.clip(probabilities, eps, 1.0)
    logits = np.log(clipped)

    def negative_log_likelihood(log_temperature):
        temperature = float(np.exp(log_temperature))
        calibrated = stable_softmax(logits / temperature)
        correct_class_probs = calibrated[np.arange(len(labels)), labels]
        return float(-np.mean(np.log(np.clip(correct_class_probs, eps, 1.0))))

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
    )

    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Temperature calibration failed.")

    temperature = float(np.exp(result.x))
    calibrated = stable_softmax(logits / temperature)

    return temperature, calibrated, float(result.fun)


class BestEpochTracker(tf.keras.callbacks.Callback):
    """Save each new best validation-accuracy checkpoint."""

    def __init__(self, candidate_path, qualified_path):
        super().__init__()
        self.candidate_path = Path(candidate_path)
        self.qualified_path = Path(qualified_path)
        self.best_val_accuracy = -1.0
        self.best_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_accuracy = logs.get("val_accuracy")
        accuracy = logs.get("accuracy")
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")

        if val_accuracy is None or not np.isfinite(val_accuracy):
            print(
                f"Epoch {epoch + 1}: validation accuracy unavailable; "
                "no checkpoint selected.",
                flush=True,
            )
            return

        val_accuracy = float(val_accuracy)
        is_new_best = val_accuracy > self.best_val_accuracy

        if is_new_best:
            self.best_val_accuracy = val_accuracy
            self.best_epoch = epoch + 1
            self.model.save(str(self.candidate_path), overwrite=True)

            if val_accuracy > VALIDATION_THRESHOLD:
                self.model.save(str(self.qualified_path), overwrite=True)

        line = (
            f"Epoch {epoch + 1:02d} | "
            f"accuracy={float(accuracy):.2%} | "
            f"val_accuracy={val_accuracy:.2%} | "
            f"loss={float(loss):.4f} | "
            f"val_loss={float(val_loss):.4f}"
        )

        if is_new_best:
            line += " | ⭐ NEW BEST EPOCH"
        if val_accuracy > VALIDATION_THRESHOLD:
            line += " | Validation accuracy > 95%"

        print(line, flush=True)


def main():
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    data_config = config.get("data", {})
    train_config = config.get("training", {})
    model_config = config.get("model", {})

    seed = int(train_config.get("seed", 42))
    epochs = int(train_config.get("epochs", 25))
    batch_size = int(train_config.get("batch_size", 16))

    raw_size = train_config.get("image_size", [224, 224])
    if isinstance(raw_size, int):
        image_size = (raw_size, raw_size)
    else:
        image_size = tuple(int(value) for value in raw_size[:2])

    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError(f"Invalid image_size: {raw_size}")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")

    split_dir = PROJECT_ROOT / data_config.get("split_dir", "data/splits")
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {split_dir}")

    models_dir = PROJECT_ROOT / "models"
    logs_dir = PROJECT_ROOT / "logs"
    metrics_dir = PROJECT_ROOT / "reports" / "metrics"

    for directory in (models_dir, logs_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    candidate_path = models_dir / "best_candidate_model.h5"
    qualified_path = models_dir / "best_model.h5"
    history_path = logs_dir / "training_history.csv"
    summary_path = metrics_dir / "training_summary.json"
    calibration_path = models_dir / "confidence_calibration.json"

    archive_previous_outputs(
        [
            candidate_path,
            qualified_path,
            history_path,
            summary_path,
            calibration_path,
        ],
        run_id,
    )

    configure_environment(seed)

    print("=" * 72)
    print("KIDNEY CT CLASSIFICATION - TRAINING")
    print("=" * 72)
    print(f"Python: {sys.executable}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    print(f"Maximum epochs: {epochs}")
    print("Data splits: training + validation-selection + calibration")
    print("Test split: EXCLUDED")
    print("Best epoch: highest validation accuracy")
    print("Inference confidence threshold: 98% after calibration")
    print("=" * 72)

    # Critical: do not create, enumerate, or load the test generator.
    data = create_data_generators(
        data_dir=split_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        include_test=False,
        include_calibration=True,
    )

    required_keys = {"train", "val", "calibration", "class_weights"}
    if not required_keys.issubset(data.keys()):
        raise RuntimeError(
            f"Data loader did not return the required keys: {required_keys}"
        )
    if "test" in data:
        raise RuntimeError("Unexpected test generator returned by data loader.")

    expected_mapping = {
        name: index for index, name in enumerate(CLASSES)
    }
    for split_name in ("train", "val", "calibration"):
        generator = data[split_name]
        if generator.class_indices != expected_mapping:
            raise ValueError(
                f"Class mapping mismatch in {split_name}: "
                f"{generator.class_indices}"
            )

    class_weights = {
        int(key): float(value)
        for key, value in data["class_weights"].items()
    }

    model = build_vgg16_classifier(
        input_shape=(image_size[0], image_size[1], 3),
        num_classes=len(CLASSES),
        weights=model_config.get("weights", "imagenet"),
        freeze_backbone=bool(model_config.get("freeze_backbone", True)),
        dropout_rate=float(model_config.get("dropout_rate", 0.35)),
        learning_rate=float(model_config.get("learning_rate", 1e-3)),
    )

    tracker = BestEpochTracker(candidate_path, qualified_path)

    callbacks = [
        tracker,
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
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(history_path),
            append=False,
        ),
    ]

    print("\nStarting training. Keras will display batch progress.\n", flush=True)

    history = model.fit(
        data["train"],
        validation_data=data["val"],
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    val_accuracy = np.asarray(
        history.history.get("val_accuracy", []),
        dtype=np.float64,
    )
    val_loss = np.asarray(
        history.history.get("val_loss", []),
        dtype=np.float64,
    )

    if len(val_accuracy) == 0 or len(val_loss) != len(val_accuracy):
        raise RuntimeError("Training history is missing validation metrics.")
    if not np.isfinite(val_accuracy).all() or not np.isfinite(val_loss).all():
        raise RuntimeError("Non-finite validation metric detected.")
    if not candidate_path.is_file():
        raise RuntimeError("No candidate checkpoint was saved.")

    best_index = int(np.argmax(val_accuracy))
    best_accuracy = float(val_accuracy[best_index])

    # Ensure the candidate represents the restored best validation epoch.
    model.save(str(candidate_path), overwrite=True)

    validation_threshold_passed = best_accuracy > VALIDATION_THRESHOLD
    if validation_threshold_passed:
        model.save(str(qualified_path), overwrite=True)

    print("\nSelected best epoch:", best_index + 1)
    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print(f"Validation accuracy > 95%: {validation_threshold_passed}")

    # Calibrate probabilities only on the dedicated calibration subset.
    # This subset is not used for gradient updates or epoch selection.
    calibration_generator = data["calibration"]
    calibration_generator.reset()

    print("\nCalibrating prediction probabilities...", flush=True)
    calibration_probabilities = model.predict(
        calibration_generator,
        verbose=1,
    )
    calibration_labels = np.asarray(
        calibration_generator.classes,
        dtype=np.int64,
    )

    temperature, calibrated_probs, calibration_nll = calibrate_temperature(
        calibration_probabilities,
        calibration_labels,
    )

    raw_confidence = np.max(calibration_probabilities, axis=1)
    calibrated_confidence = np.max(calibrated_probs, axis=1)
    calibrated_predictions = np.argmax(calibrated_probs, axis=1)
    calibration_accuracy = float(
        np.mean(calibrated_predictions == calibration_labels)
    )
    confidence_coverage = float(
        np.mean(calibrated_confidence >= INFERENCE_CONFIDENCE_THRESHOLD)
    )

    calibration_summary = {
        "method": "temperature_scaling",
        "temperature": temperature,
        "confidence_threshold": INFERENCE_CONFIDENCE_THRESHOLD,
        "confidence_threshold_percent": 98.0,
        "calibration_samples": int(len(calibration_labels)),
        "calibration_accuracy": calibration_accuracy,
        "calibration_negative_log_likelihood": calibration_nll,
        "mean_raw_max_probability": float(np.mean(raw_confidence)),
        "mean_calibrated_max_probability": float(np.mean(calibrated_confidence)),
        "calibration_fraction_at_or_above_98_percent": confidence_coverage,
        "class_names": list(CLASSES),
        "model_path": str(candidate_path.relative_to(PROJECT_ROOT)),
        "calibration_data_source": "validation directory, held-out calibration subset",
        "test_split_used": False,
        "note": (
            "Calibration confidence is not a guarantee of correctness. "
            "The reported coverage is descriptive of the calibration subset, "
            "not an independent test result."
        ),
    }

    with calibration_path.open("w", encoding="utf-8") as file:
        json.dump(calibration_summary, file, indent=2, ensure_ascii=False)

    training_summary = {
        "status": "completed",
        "run_id": run_id,
        "classes": list(CLASSES),
        "seed": seed,
        "image_size": list(image_size),
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_completed": int(len(val_accuracy)),
        "selection_metric": "val_accuracy",
        "best_epoch": best_index + 1,
        "best_val_accuracy": best_accuracy,
        "best_val_accuracy_percent": round(best_accuracy * 100, 4),
        "val_loss_at_best_accuracy": float(val_loss[best_index]),
        "validation_accuracy_above_95_percent": bool(
            validation_threshold_passed
        ),
        "candidate_model_path": str(candidate_path.relative_to(PROJECT_ROOT)),
        "qualified_model_saved": bool(validation_threshold_passed),
        "qualified_model_path": (
            str(qualified_path.relative_to(PROJECT_ROOT))
            if validation_threshold_passed else None
        ),
        "training_history_path": str(history_path.relative_to(PROJECT_ROOT)),
        "calibration_path": str(calibration_path.relative_to(PROJECT_ROOT)),
        "test_split_loaded": False,
        "test_split_evaluated": False,
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(training_summary, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("TRAINING AND CALIBRATION FINISHED")
    print("=" * 72)
    print(json.dumps(training_summary, indent=2, ensure_ascii=False))
    print("\nConfidence calibration:")
    print(json.dumps(calibration_summary, indent=2, ensure_ascii=False))
    print("\nTest split was not loaded or evaluated.")
    print("No Git commit or upload was performed.")


if __name__ == "__main__":
    main()
