
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class DataIngestion:
    """Local-only data validation; remote downloads are disabled."""

    def __init__(self, config=None):
        self.config = config
        self.validator = (
            PROJECT_ROOT / "src" / "data" / "validate_dataset.py"
        )

    def validate_local_data(self):
        if not self.validator.is_file():
            raise FileNotFoundError(
                "Local dataset validator not found: {}".format(
                    self.validator
                )
            )

        return subprocess.run(
            [sys.executable, str(self.validator)],
            cwd=str(PROJECT_ROOT),
            check=True,
        ).returncode

    def download_file(self):
        raise RuntimeError(
            "Remote dataset downloads are disabled. "
            "Place the approved dataset in data/splits and validate it locally."
        )

    def extract_zip_file(self):
        raise RuntimeError(
            "Automatic archive extraction is disabled. "
            "This project uses the existing local data/splits directory."
        )
