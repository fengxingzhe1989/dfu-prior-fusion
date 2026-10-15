# Release scope and recoverable cleanup

This release replaces pre-revision entry points with the frozen associate-editor pipeline. It is a packaging cleanup, not a change to experimental design, predictions or result values.

## Retained

- All five-seed classification, stacking, segmentation, uncertainty and interpretation code, recovery helpers and tests.
- The later budget-first v3 training-size source and complete four-budget aggregates, kept separate from primary results and superseded v1/v2 attempts.
- Shared architecture, preprocessing and feature-building modules and ten frozen source configs. Older filenames do not imply these dependencies are unused.
- Dataset inclusion, grouping and cross-dataset overlap audits and required non-image metadata.
- Historical batch/split factorial code and verified summaries, original source-generator resolution audit and architecture-only benchmark still used as supporting evidence.

## Removed from the current tree

- Superseded full-OOF fusion search and candidate-selection outputs.
- Pre-revision result/sensitivity/interpretation pipelines not called by the frozen analysis.
- Old figure builders bound to superseded results. The frozen interpretation renderer remains; this release does not claim to regenerate every manually edited manuscript schematic.
- Unused hardware/grouped/exclusion/random-prior configs, result template and upload checklist.
- Superseded result tables/prediction files that could be confused with the current primary evidence.

Exact paths and original hashes are in `paper_assets/release_audit/removed_files.csv`. The earlier public tree remains recoverable at commit `6d4208639c802ed38442729dae8dacd8a8c76cdf`; Git history is not rewritten. Local research archives, source data, completed results and existing independently modified checkouts are not deleted.

## Provenance

The 11 October 2026 addition publishes the separately frozen three-RGB-model interpretability extension and combined five-model aggregates. `paper_assets/release_audit/rgb_extension_asset_registry.csv` records the original and released hashes. The four extension protocol/code/test files match the sealed execution files. This addition does not modify the initial two-model code, the original frozen-asset registry or any scientific result. Its only transformed aggregate metadata is the explicitly path-redacted input registry. Checkpoints, per-image predictions, medical images and case arrays remain excluded. See Sections 4–5 of `REPRODUCIBILITY.md` for the initial-versus-extension boundary, 50 + 75 = 125 job counts and the later display-only palette change.

The 15 October 2026 addition publishes the separately frozen v3 training-size analysis: seven byte-identical protocol/source/test files, one additional source dependency recorded in the original seal, all four budgets and five trajectories, and complete non-image aggregate tables with saved bootstrap draws. The additional dependency `experiments/dfuc2021_cls_baseline/summarize_strong_results.py` is restored because it was registered in the v3 seal; it is not a current-analysis entry point. `paper_assets/release_audit/training_size_asset_registry.csv` records copied-file hashes. The sole derived v3 metadata file is explicitly named `provenance_summary.json`, not `sealed_inputs.json`: it exposes original provenance digests and environment versions without private machine paths and is not usable as the runner's execution seal. Original markers remain byte-identical historical records. The original frozen registries and experiment/result files are not edited.

The v3 publication contains 400 fresh fits' aggregate evidence, not 300 fits or a reused primary 100% endpoint. It does not distribute fit checkpoints, per-image predictions, individual assignments, training logs or medical images. The original runner's archive/environment and sidecar dependencies are documented in Section 6 of `REPRODUCIBILITY.md`; a public aggregate check is not a full rerun. This addition also corrects only the public final-display palette description to `Greens`, opacity 0.35; sealed CAM renderers and numerical results are unchanged.

`paper_assets/release_audit/frozen_asset_registry.csv` records the byte-identical source/output copies. The one path-redacted registry is explicitly identified with its original hash. Documentation/publication wrappers are separate from sealed scientific code. Archived markers may refer to undistributed inference/model assets, not all-public inputs.

The retained historical generator-resolution JSON also replaces its old machine-specific checkpoint prefix with a repository-relative path; no numerical result was edited. The original JSON SHA256 is `53190f756043e5a043bc89a73a0284eff4266f2ef2fdb3ded457337e2b72976b` and remains recoverable in the preceding Git revision.

The historical factorial and new analysis have different validation/uncertainty limits. Historical scores are not substituted into current all-seed aggregates. Removing an obsolete entry point does not erase prior evidence or permit selective reporting.
