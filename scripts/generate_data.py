"""Write the synthetic maintenance-log JSONL from configs/data.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maintlog.config import DataConfig
from maintlog.data import write_jsonl
from maintlog.synthetic import generate_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic maintenance-log dataset.")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)
    config = DataConfig.from_yaml(args.config)
    rows = generate_rows(config.n_train, config.n_eval, config.seed)
    path = write_jsonl(config.output_path, rows)
    train_count = sum(row["split"] == "train" for row in rows)
    eval_count = sum(row["split"] == "eval" for row in rows)
    print(f"Wrote {len(rows)} synthetic rows ({train_count} train, {eval_count} eval) to {path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
