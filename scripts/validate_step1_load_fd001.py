"""Run Step 1 validation: load and structurally verify FD001 files."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cmapss_loader import validate_fd001_dataset


def main() -> None:
    summary = validate_fd001_dataset(PROJECT_ROOT / "nasa")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
