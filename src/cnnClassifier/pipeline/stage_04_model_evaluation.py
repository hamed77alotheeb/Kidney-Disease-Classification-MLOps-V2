"""Stage 04: evaluate the fine-tuned model on the validation split."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_SCRIPT = (
    PROJECT_ROOT / "src" / "evaluation" / "evaluate_validation.py"
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the classifier using validation data only."
    )
    parser.parse_args()

    if not EVALUATION_SCRIPT.is_file():
        raise FileNotFoundError(str(EVALUATION_SCRIPT))

    result = subprocess.run(
        [sys.executable, str(EVALUATION_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
