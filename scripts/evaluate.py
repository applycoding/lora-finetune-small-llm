"""Score JSON validity and field accuracy from a YAML config."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maintlog.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main())
