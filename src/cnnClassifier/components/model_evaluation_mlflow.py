
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Evaluation:
    """Run the current local validation evaluation; no remote MLflow."""

    def __init__(self, config=None):
        self.config = config
        self.script = (
            PROJECT_ROOT / "src" / "evaluation"
            / "evaluate_validation.py"
        )

    def evaluation(self):
        if not self.script.is_file():
            raise FileNotFoundError(str(self.script))

        return subprocess.run(
            [sys.executable, str(self.script)],
            cwd=str(PROJECT_ROOT),
            check=True,
        ).returncode

    def log_into_mlflow(self):
        raise RuntimeError(
            "External MLflow logging is disabled in this local project."
        )
