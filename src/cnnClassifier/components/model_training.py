
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Training:
    """Adapter to the current four-class training implementation."""

    def __init__(self, config=None):
        self.config = config
        self.script = PROJECT_ROOT / "src" / "models" / "train.py"

    def train(self):
        if not self.script.is_file():
            raise FileNotFoundError(str(self.script))

        return subprocess.run(
            [sys.executable, str(self.script)],
            cwd=str(PROJECT_ROOT),
            check=True,
        ).returncode
