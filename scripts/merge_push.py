"""Merge a LoRA adapter and push it. Requires HF_TOKEN unless --merge-only."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maintlog.merge_push import main

if __name__ == "__main__":
    raise SystemExit(main())
