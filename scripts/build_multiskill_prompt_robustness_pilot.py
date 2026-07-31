"""Build the approved local multi-Skill pilot bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skill_annealing.multiskill_prompt_robustness.builder import build_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="data/multiskill_prompt_robustness_pilot_20260724"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_bundle(Path(args.output_dir)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
