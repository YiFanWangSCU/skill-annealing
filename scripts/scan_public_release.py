"""Conservative secret/privacy/LFS scan for a public checkout."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXT = {".py", ".md", ".json", ".jsonl", ".sh", ".toml", ".yml", ".yaml", ".env", ".txt"}
PATTERNS = {
    "openai_key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "hf_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "wandb_key": re.compile(r"(wandb|w\.andb)[-_]?[A-Za-z0-9]{20,}", re.I),
    "private_path": re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z]:\\|/home/|/data/|/root/|/Users/)", re.I),
    "credential_assignment": re.compile(r"(?:api[_-]?key|token|password)\s*[:=]\s*['\"][^'\"]+", re.I),
}

findings = []
large_files = []
for path in ROOT.rglob("*"):
    if not path.is_file() or ".git" in path.parts or path.name == Path(__file__).name:
        continue
    size = path.stat().st_size
    if size > 50 * 1024 * 1024:
        large_files.append({"path": str(path.relative_to(ROOT)), "bytes": size})
    if path.suffix.lower() not in TEXT_EXT:
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue
    for name, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({"type": name, "path": str(path.relative_to(ROOT)), "line": text[:match.start()].count("\n") + 1})

report = {
    "sha256": hashlib.sha256("\n".join(sorted(str(p) for p in ROOT.rglob("*"))).encode()).hexdigest(),
    "findings": findings,
    "large_files_over_50MiB": large_files,
    "git_lfs_pointer_files": [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and p.stat().st_size < 200 and p.suffix in {".bin", ".safetensors", ".pt", ".pth"}],
    "status": "pass" if not findings and not large_files else "review_required",
}
(ROOT / "results" / "secret-scan-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
sys.exit(0 if report["status"] == "pass" else 1)
