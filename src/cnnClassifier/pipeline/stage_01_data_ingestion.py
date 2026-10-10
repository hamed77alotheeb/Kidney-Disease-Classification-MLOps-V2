
import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(
        description="Validate the existing local dataset; do not download data."
    )
    parser.parse_args()

    validator = PROJECT_ROOT / "src" / "data" / "validate_dataset.py"
    if not validator.is_file():
        raise FileNotFoundError(str(validator))

    return subprocess.run(
        [sys.executable, str(validator)],
        cwd=str(PROJECT_ROOT),
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
