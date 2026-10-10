
import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Verify the existing four-class baseline model; do not train."
    )
    parser.parse_args()

    from cnnClassifier.components.prepare_base_model import PrepareBaseModel

    model = PrepareBaseModel().get_base_model()

    print("Existing baseline model: valid")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    print("No training or weight modification was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
