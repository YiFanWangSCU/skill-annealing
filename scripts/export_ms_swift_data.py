"""Export messages-only JSONL files for ms-swift SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DATASET_FILES = {
    "full_skill_sft": "full_skill_sft.jsonl",
    "no_skill_sft": "no_skill_sft.jsonl",
    "annealed_sft": "annealed_sft.jsonl",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def messages_only(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"messages": record["messages"]} for record in records]


def export_ms_swift_data(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"outputs": {}}
    for dataset_name, filename in DATASET_FILES.items():
        source_path = source_dir / filename
        records = messages_only(load_jsonl(source_path))
        output_path = output_dir / filename
        write_jsonl(output_path, records)
        summary["outputs"][dataset_name] = {
            "path": str(output_path),
            "count": len(records),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert refund_decision SFT JSONL to ms-swift messages-only JSONL."
    )
    parser.add_argument("--source-dir", default="data/refund_decision")
    parser.add_argument("--output-dir", default="data/ms_swift/refund_decision")
    args = parser.parse_args()

    summary = export_ms_swift_data(Path(args.source_dir), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
