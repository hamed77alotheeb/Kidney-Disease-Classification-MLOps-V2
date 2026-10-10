
"""Compatibility import for notebooks using the previous module path."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from cnnClassifier.pipeline.prediction import (
    KidneyImagePredictor,
    EXPECTED_CLASSES,
)

__all__ = ["KidneyImagePredictor", "EXPECTED_CLASSES"]
