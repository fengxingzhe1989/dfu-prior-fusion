# Release scope and recoverable cleanup

This release replaces pre-revision entry points with the frozen associate-editor pipeline. It is a packaging cleanup, not a change to experimental design, predictions or result values.

## Retained

- All five-seed classification, stacking, segmentation, uncertainty and interpretation code, recovery helpers and tests.
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

`paper_assets/release_audit/frozen_asset_registry.csv` records the byte-identical source/output copies. The one path-redacted registry is explicitly identified with its original hash. Documentation/publication wrappers are separate from sealed scientific code. Archived markers may refer to undistributed inference/model assets, not all-public inputs.

The retained historical generator-resolution JSON also replaces its old machine-specific checkpoint prefix with a repository-relative path; no numerical result was edited. The original JSON SHA256 is `53190f756043e5a043bc89a73a0284eff4266f2ef2fdb3ded457337e2b72976b` and remains recoverable in the preceding Git revision.

The historical factorial and new analysis have different validation/uncertainty limits. Historical scores are not substituted into current all-seed aggregates. Removing an obsolete entry point does not erase prior evidence or permit selective reporting.
