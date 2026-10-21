# Release verification — 2026-10-09

This is a code/aggregate packaging verification, not a repeat of the training experiments.

- All 67 copied-asset registry entries verified; original research-workspace sources/results remained unchanged.
- All 33 scientific scripts, source configs and protocol/methods files in the recorded server snapshot matched the release.
- 19 synthetic geometry/finalization regression tests passed.
- 35 retained argparse command-line entry points loaded successfully with `--help`.
- Dependency/AST/config/preprocessing-reference checks passed.
- Reconstructed nested assignments from the released grouped manifest contained 29,775 rows and matched frozen SHA256 `f0d4590e47ba0e58b2d24236dbace32ea6cb62f45628d5470c2914744d166dbf`.
- Checked all five seeds, 14 classification systems, 350 fold/seed/system rows, 1,400 class/fold/seed/system rows, three primary contrasts, 5,000 classification bootstrap replicates, mean/SD/CI/p/Holm calculations, segmentation aggregates, 59,550 interpretation records by coverage and 20 fixed cases.
- `uv lock --check --offline`, preprocessing dry-run and `git diff --check` passed.
- All 117 removed files (22 Python scripts, 12 YAML configs, 64 CSV, 10 JSON, eight Markdown and one text file) were backed up byte-for-byte; exact paths/hashes are in `removed_files.csv`. The preceding Git revision retains them.
- No raw images, masks, model weights, case arrays, office documents or credentials were added.

Commands suitable for independently repeating the data-free checks:

```bash
python scripts/check_release.py
python -m unittest discover -s experiments/associate_editor_revision_v1 -p 'test_*.py' -v
python data_processing/prepare_manuscript_data.py --dry-run
uv lock --check --offline
```

The implementation still requires separately obtained licensed data and frozen checkpoints for full training/inference reproduction. No manuscript edits, model tuning or retraining were performed during this packaging task.
