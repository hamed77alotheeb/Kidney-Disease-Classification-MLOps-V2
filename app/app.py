"""Compatibility wrapper for the root Flask app."""
import importlib.util
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_APP_PATH = PROJECT_ROOT / "app.py"

for import_root in (PROJECT_ROOT, PROJECT_ROOT / 'src'):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

SPEC = importlib.util.spec_from_file_location(
    "_kidney_root_flask_app", str(ROOT_APP_PATH)
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load Flask app from {ROOT_APP_PATH}")

ROOT_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ROOT_MODULE
SPEC.loader.exec_module(ROOT_MODULE)

app = ROOT_MODULE.app
__all__ = ["app"]
