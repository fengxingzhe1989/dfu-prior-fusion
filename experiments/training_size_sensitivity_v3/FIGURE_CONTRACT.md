# Four-budget curve contract

- Purpose: display observed performance versus available DFUC2021 training data
  without presupposing improvement, superiority, saturation or sufficiency.
- Evidence: A Macro-F1; B ischaemia-only recall; C ischaemia-only precision;
  D ischaemia-only F1. Each shows the two matched models at all four budgets,
  all five trajectories and their mean +/- sample SD. Paired prior-contribution
  estimates/pointwise intervals are provided in the accompanying table.
- Archetype: compact quantitative grid. Fixed labels, no selective panels,
  smoothing, removed outliers or favourable-seed emphasis.
- Backend: Python, using the existing saved plotting preference. Matplotlib only.
- Horizontal coordinates: actual mean training image counts across five outer
  folds; nominal fractions are tick annotations, not a claim of exact image
  percentages. Source data include actual group and minority-image counts.
- Exports: editable SVG/PDF, 600 dpi lossless-compressed TIFF and a preview PNG;
  about 183 mm wide, white background, readable 8 pt typography, restrained
  blue/grey method colours, shared semantics across panels. Export all source
  points and the caption with hashes.
- Limits: five trajectories combine subsampling and optimisation variability.
  Fixed external prior and ImageNet pretraining are not included in the varying
  DFUC2021 label budget. No patient-level, anatomical or clinical validation.
- Gate: do not render until all 40 seed cells and all 400 new fits verify. Exports
  remain draft until an actual-image visual inspection; computation/export alone
  does not assert manuscript insertion or publication-readiness.

All four endpoints are fresh v3 fits. Five-trajectory variation includes subset,
internal-partition and optimisation variation. No historical predictions enter.
