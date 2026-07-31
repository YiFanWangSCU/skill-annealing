# Public-release audit

## Scope review

- Included: original experiment utilities, synthetic data builders, small synthetic examples, aggregate result tables, documentation, and focused tests.
- Excluded: `.git` history, model weights, adapter/checkpoint files, optimizer states, raw predictions, logs, remote launch scripts, personal paths, server addresses, credentials, and internal/company-specific materials.
- Data review: `data/examples/` is regenerated from `skill_annealing/refund_decision/data_generator.py`; it contains synthetic fields and no external dataset.

## Automated scan

Run on this release candidate:

```text
python scripts/scan_public_release.py
status: pass
secret/private-path findings: 0
files over 50 MiB: 0
Git LFS pointer candidates: 0
```

The scanner checks common OpenAI/Hugging Face/W&B key formats, credential assignments, absolute private-path patterns, and large binary artifacts. It is a conservative static scan, not proof that secrets never existed.

## License review

- No model weights, datasets, or ms-swift source code are redistributed.
- Qwen and ms-swift must be installed separately and used under their own licenses.
- The release candidate intentionally has **no repository LICENSE** until the author confirms authorship/provenance for every retained file. `NOTICE.md` records this restriction.

## Reproducibility review

All public commands in `README.md` use repository-relative paths. The CPU smoke test runs generation, export, and oracle structured evaluation without network access or a model download. Full SFT requires a user-supplied, locally licensed model path through `MODEL_PATH`.
