"""Run the no-model public smoke test: generation, export, and oracle eval."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "data" / "examples" / "smoke_generated"
subprocess.run([sys.executable, "scripts/generate_mvp_data.py", "--sample-count", "8", "--output-dir", str(out)], cwd=ROOT, check=True)
result = subprocess.run([sys.executable, "-m", "skill_annealing.refund_decision.evaluator", "--predictions", str(out / "eval_prompts.jsonl"), "--by-exposure", "--oracle"], cwd=ROOT, check=True, capture_output=True, text=True)
metrics = json.loads(result.stdout)
assert metrics["retention"]["skill_retention_score"] == 1.0
print(json.dumps({"generated": 8, "exposures": list(metrics["retention"]["skill_retention_curve"]), "oracle_retention": metrics["retention"]["skill_retention_score"]}, ensure_ascii=False))
