"""Stage 05: fine-tune the existing baseline model."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINE_TUNE_SCRIPT = PROJECT_ROOT / "src" / "models" / "fine_tune.py"


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune the existing classifier."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Archive previous outputs before another fine-tuning run.",
    )
    args = parser.parse_args()

    if not FINE_TUNE_SCRIPT.is_file():
        raise FileNotFoundError(str(FINE_TUNE_SCRIPT))

    command = [sys.executable, str(FINE_TUNE_SCRIPT)]
    if args.force:
        command.append("--force")

    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
