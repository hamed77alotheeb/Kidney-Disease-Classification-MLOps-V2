from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image


EXPECTED_CLASSES = ("Cyst", "Normal", "Stone", "Tumor")
DEFAULT_CONFIDENCE_THRESHOLD = 0.98


class KidneyImagePredictor:
    """
    Load the saved kidney image classifier and predict one uploaded image.

    The confidence value is a calibrated model score, not a guarantee
    of diagnostic correctness.
    """

    def __init__(
        self,
        model_path=None,
        calibration_path=None,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        project_root = Path(__file__).resolve().parents[3]

        self.model_path = Path(
            model_path
            if model_path is not None
            else project_root / "models" / "best_finetuned_model.h5"
        )

        self.calibration_path = Path(
            calibration_path
            if calibration_path is not None
            else project_root / "models" / "fine_tuned_confidence_uncalibrated.json"
        )

        self.confidence_threshold = float(confidence_threshold)

        if not 0.0 < self.confidence_threshold < 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

        if not self.model_path.is_file():
            raise FileNotFoundError(
                "Saved model not found: {}".format(self.model_path)
            )

        if not self.calibration_path.is_file():
            raise FileNotFoundError(
                "Calibration file not found: {}".format(
                    self.calibration_path
                )
            )

        with self.calibration_path.open("r", encoding="utf-8") as file:
            calibration = json.load(file)

        self.temperature = float(calibration["temperature"])

        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid calibration temperature.")

        saved_classes = tuple(calibration.get("class_names", ()))
        if saved_classes != EXPECTED_CLASSES:
            raise ValueError(
                "Class order in calibration file does not match the "
                "expected classifier order. Found: {}".format(saved_classes)
            )

        # Load once when the predictor is initialized, not per image.
        self.model = tf.keras.models.load_model(
            str(self.model_path),
            compile=False,
        )

        output_shape = self.model.output_shape
        if len(output_shape) != 2 or output_shape[-1] != len(EXPECTED_CLASSES):
            raise ValueError(
                "Unexpected model output shape: {}".format(output_shape)
            )

        input_shape = self.model.input_shape
        if (
            len(input_shape) != 4
            or input_shape[1] is None
            or input_shape[2] is None
            or input_shape[3] != 3
        ):
            raise ValueError(
                "Unexpected model input shape: {}".format(input_shape)
            )

        self.image_height = int(input_shape[1])
        self.image_width = int(input_shape[2])
        self.class_names = EXPECTED_CLASSES

    def predict_image(self, image_path):
        """
        Predict one image and return the top class and calibrated scores.
        """
        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                "Image file not found: {}".format(image_path)
            )

        try:
            with Image.open(str(image_path)) as opened_image:
                image = opened_image.convert("RGB")

                if hasattr(Image, "Resampling"):
                    resampling = Image.Resampling.BILINEAR
                else:
                    resampling = Image.BILINEAR

                image = image.resize(
                    (self.image_width, self.image_height),
                    resample=resampling,
                )

                image_array = np.asarray(image, dtype=np.float32)

        except Exception as exc:
            raise ValueError(
                "Unable to read the supplied image: {}".format(exc)
            )

        # Match the VGG16 preprocessing used by the training data loader.
        batch = np.expand_dims(image_array, axis=0)
        batch = tf.keras.applications.vgg16.preprocess_input(batch)

        raw_probabilities = np.asarray(
            self.model.predict(batch, verbose=0)[0],
            dtype=np.float64,
        )

        if (
            raw_probabilities.shape != (len(self.class_names),)
            or not np.isfinite(raw_probabilities).all()
        ):
            raise RuntimeError("The model returned invalid probabilities.")

        raw_probabilities = np.clip(raw_probabilities, 1e-7, 1.0)
        raw_probabilities /= raw_probabilities.sum()

        # Temperature scaling using the temperature saved during calibration.
        scaled_logits = np.log(raw_probabilities) / self.temperature
        scaled_logits -= np.max(scaled_logits)

        exp_values = np.exp(scaled_logits)
        calibrated_probabilities = exp_values / exp_values.sum()

        predicted_index = int(np.argmax(calibrated_probabilities))
        confidence = float(calibrated_probabilities[predicted_index])
        meets_threshold = confidence >= self.confidence_threshold

        return {
            "predicted_class": self.class_names[predicted_index],
            "confidence": confidence,
            "confidence_percent": round(confidence * 100.0, 2),
            "raw_confidence": float(np.max(raw_probabilities)),
            "confidence_threshold": self.confidence_threshold,
            "confidence_threshold_percent": (
                self.confidence_threshold * 100.0
            ),
            "meets_confidence_threshold": bool(meets_threshold),
            "class_probabilities": {
                name: float(calibrated_probabilities[index])
                for index, name in enumerate(self.class_names)
            },
            "notice": (
                "Model confidence is an estimate, not a guarantee "
                "of diagnostic correctness."
            ),
        }


class PredictionPipeline:
    """Compatibility wrapper for the original project structure."""

    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        result = KidneyImagePredictor().predict_image(self.filename)
        return [{"image": result["predicted_class"]}]
