
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    manifest_path: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    model_path: Path
    classes: Tuple[str, ...]
    image_size: Tuple[int, int]


@dataclass(frozen=True)
class TrainingConfig:
    script_path: Path
    config_path: Path
    data_dir: Path
    epochs: int
    batch_size: int
    image_size: Tuple[int, int]
    seed: int


@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    validation_data: Path
    report_path: Path
    image_size: Tuple[int, int]
    batch_size: int
