
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class PrepareBaseModel:
    """Validate an existing four-class model without retraining it."""

    def __init__(self, config=None):
        self.config = config
        self.model_path = Path(
            getattr(
                config,
                "model_path",
                PROJECT_ROOT / "models" / "best_candidate_model.h5",
            )
        )
        self.model = None

    def get_base_model(self):
        if not self.model_path.is_file():
            raise FileNotFoundError(
                "Existing baseline model not found: {}".format(
                    self.model_path
                )
            )

        import tensorflow as tf

        self.model = tf.keras.models.load_model(
            str(self.model_path),
            compile=False,
        )

        input_shape = self.model.input_shape
        output_shape = self.model.output_shape

        if len(input_shape) != 4 or input_shape[-1] != 3:
            raise ValueError(
                "Unexpected baseline input shape: {}".format(input_shape)
            )

        if len(output_shape) != 2 or output_shape[-1] != 4:
            raise ValueError(
                "Expected four output classes, got {}".format(output_shape)
            )

        return self.model

    def update_base_model(self):
        # Compatibility method: validation only, no weight update.
        return self.get_base_model()

    @staticmethod
    def save_model(path, model):
        raise RuntimeError(
            "This component cannot write model weights. "
            "Use the explicit training command to create a new model."
        )
