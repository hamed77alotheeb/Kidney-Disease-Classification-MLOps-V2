
from pathlib import Path

import yaml

from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ConfigurationManager:
    """Read the current local four-class configuration."""

    def __init__(self, config_filepath=None, params_filepath=None):
        self.config_path = Path(
            config_filepath or PROJECT_ROOT / "config" / "config.yaml"
        )
        self.params_path = Path(
            params_filepath or PROJECT_ROOT / "params.yaml"
        )

        if not self.config_path.is_absolute():
            self.config_path = PROJECT_ROOT / self.config_path
        if not self.params_path.is_absolute():
            self.params_path = PROJECT_ROOT / self.params_path

        with self.config_path.open("r", encoding="utf-8-sig") as file:
            self.config = yaml.safe_load(file) or {}

        if self.params_path.is_file():
            with self.params_path.open(
                "r", encoding="utf-8-sig"
            ) as file:
                self.params = yaml.safe_load(file) or {}
        else:
            self.params = {}

        names = self.config.get("data", {}).get("classes", [])
        if names != ["Cyst", "Normal", "Stone", "Tumor"]:
            raise ValueError("Expected the four configured kidney classes.")

        if self.config["data"].get("num_classes") != 4:
            raise ValueError("Configured class count must be four.")

    def get_data_ingestion_config(self):
        data = self.config["data"]
        return DataIngestionConfig(
            root_dir=PROJECT_ROOT / data.get("split_dir", "data/splits"),
            manifest_path=PROJECT_ROOT / data.get(
                "manifest", "data/splits/data_manifest_unique.csv"
            ),
        )

    def get_prepare_base_model_config(self):
        training = self.config["training"]
        return PrepareBaseModelConfig(
            model_path=PROJECT_ROOT / "models" / "best_candidate_model.h5",
            classes=tuple(self.config["data"]["classes"]),
            image_size=tuple(training.get("image_size", [224, 224])),
        )

    def get_training_config(self):
        training = self.config["training"]
        return TrainingConfig(
            script_path=PROJECT_ROOT / "src" / "models" / "train.py",
            config_path=self.config_path,
            data_dir=PROJECT_ROOT / self.config["data"].get(
                "split_dir", "data/splits"
            ),
            epochs=int(training.get("epochs", 25)),
            batch_size=int(training.get("batch_size", 16)),
            image_size=tuple(training.get("image_size", [224, 224])),
            seed=int(training.get("seed", 42)),
        )

    def get_evaluation_config(self):
        training = self.config["training"]
        evaluation = self.config.get("evaluation", {})
        return EvaluationConfig(
            path_of_model=PROJECT_ROOT / evaluation.get(
                "validation_model", "models/best_finetuned_model.h5"
            ),
            validation_data=PROJECT_ROOT / self.config["data"].get(
                "split_dir", "data/splits"
            ) / "val",
            report_path=(
                PROJECT_ROOT / "reports" / "metrics"
                / "validation_evaluation_summary.json"
            ),
            image_size=tuple(training.get("image_size", [224, 224])),
            batch_size=int(training.get("batch_size", 16)),
        )
