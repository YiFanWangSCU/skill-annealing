"""Build public, synthetic Full-only / Random-Mix / Naive-Staged SFT files.

All labels come from the deterministic rule engine.  The script never calls an
LLM and accepts only repository-relative paths.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skill_annealing.refund_decision.data_generator import generate_samples
from skill_annealing.refund_decision.prompt_builder import build_sft_record


EXPOSURES = ("full", "partial", "minimal", "no_skill")


def exposure_order(method: str, count: int, seed: int) -> list[str]:
    if method == "full_only":
        return ["full"] * count
    if count % 4:
        raise ValueError("--records must be divisible by 4 for mixed methods")
    if method == "random_mix":
        values = [exposure for exposure in EXPOSURES for _ in range(count // 4)]
        random.Random(seed).shuffle(values)
        return values
    if method == "naive_staged":
        if count % 12:
            raise ValueError("--records must be divisible by 12 for naive_staged")
        q = count // 12
        return (["full"] * (3 * q) + ["partial"] * q +
                ["partial"] * (2 * q) + ["minimal"] * (2 * q) +
                ["minimal"] * q + ["no_skill"] * (3 * q))
    raise ValueError(f"unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("full_only", "random_mix", "naive_staged"), required=True)
    parser.add_argument("--records", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples = generate_samples(sample_count=args.records, seed=args.seed)
    exposures = exposure_order(args.method, args.records, args.seed)
    rows = []
    for index, (sample, exposure) in enumerate(zip(samples, exposures, strict=True)):
        row = build_sft_record(sample, exposure)
        row["method"] = args.method
        row["record_index"] = index
        row["seed"] = args.seed
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"method": args.method, "records": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
