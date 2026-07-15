# Fusion Search Logs

This directory contains non-image probability-fusion search evidence for the secondary grouped development analysis.

Files:

- `publication_baseline5_full_oof_fusion_search.csv`: full OOF candidate table from the baseline5 fusion search, sorted by Macro-F1.
- `publication_baseline5_single_metrics.csv`: single-method source metrics used by the search.
- `publication_baseline5_nested_metrics.csv`: legacy filename containing the precomputed-OOF fold-held-out selection-trace metric; this is not full end-to-end nested validation.
- `publication_baseline5_nested_selections.csv`: legacy filename containing fold-level second-stage selected candidates.
- `publication_baseline5_report.md`: search report emitted by the search script.
- `publication_baseline5_selection_summary.csv`: compact audit summary, including the selected prior-informed candidate rank.
- `publication_selected_prior_informed_candidate_trace.csv`: exact row for the manuscript-selected prior-informed probability-fusion candidate.

Important interpretation boundary:

The full search log is retained for transparency and includes unconstrained seed-only RGB probability-fusion candidates. The unrestricted winner was RGB-only. The highest-ranked candidate containing `strong_lpt_full_output_ensemble` ranked 13th and is retained only as a post hoc prior-informed full-OOF development analysis. It is not the global winner, an externally validated model, or the manuscript's primary comparison.

The archived search used `--top-methods 18 --top-triples 12 --top-bias-candidates 60`; these explicit arguments differ from the general-purpose implementation defaults.
