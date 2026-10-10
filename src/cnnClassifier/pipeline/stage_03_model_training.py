"""Stage 03: run the existing baseline-training implementation."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "models" / "train.py"


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline model training."
    )
    parser.parse_args()

    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(str(TRAIN_SCRIPT))

    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
