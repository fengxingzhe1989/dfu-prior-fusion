# Publication Fusion Search

This analysis searches probability-level fusion weights from fold-strict OOF predictions.
The fold-held-out result is a second-level selection trace over precomputed OOF predictions, not full end-to-end nested validation; the full-OOF result is a development upper bound.

## Precomputed-OOF Fold-Held-Out Selection Trace

| method                             | macro_f1 | balanced_accuracy | macro_precision | macro_recall | accuracy |
| ---------------------------------- | -------- | ----------------- | --------------- | ------------ | -------- |
| nested_selected_publication_fusion | 0.952914 | 0.947379          | 0.958726        | 0.947379     | 0.947943 |

## Full OOF Development Upper Bound

| method                               | macro_f1 | balanced_accuracy | macro_precision | macro_recall | accuracy | biased | bias                                     | candidate                                                                                                                               |
| ------------------------------------ | -------- | ----------------- | --------------- | ------------ | -------- | ------ | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| full_oof_selected_publication_fusion | 0.964246 | 0.959659          | 0.969214        | 0.959659     | 0.953820 | True   | [0.1, -0.05, 0.0, 0.3]                   | 0.250*dfuc2021_cls_efficientnetv2s_seed2027 + 0.350*convnext_tiny + 0.400*dfuc2021_cls_efficientnetv2s_seed3407                         |
| full_oof_selected_publication_fusion | 0.964034 | 0.959037          | 0.969257        | 0.959037     | 0.954324 | True   | [0.12500000000000003, 0.25, 0.0, 0.0]    | 0.150*dfuc2021_cls_convnext_tiny_seed2027 + 0.400*convnext_tiny + 0.450*dfuc2021_cls_efficientnetv2s_seed3407                           |
| full_oof_selected_publication_fusion | 0.964030 | 0.959658          | 0.968538        | 0.959658     | 0.953149 | True   | [-0.35, 0.0, 0.0, -0.3]                  | 0.350*dfuc2021_cls_efficientnetv2s_seed2027 + 0.400*dfuc2021_cls_efficientnetv2s_seed3407 + 0.250*dfuc2021_cls_convnext_tiny_seed202607 |
| full_oof_selected_publication_fusion | 0.963992 | 0.959257          | 0.969090        | 0.959257     | 0.953652 | True   | [0.0, -0.1, 0.0, 0.0]                    | 0.250*dfuc2021_cls_efficientnetv2s_seed2027 + 0.300*convnext_tiny + 0.450*dfuc2021_cls_efficientnetv2s_seed3407                         |
| full_oof_selected_publication_fusion | 0.963981 | 0.958644          | 0.969640        | 0.958644     | 0.954324 | True   | [0.125, 0.1, 0.1, 0.05]                  | 0.450*dfuc2021_cls_efficientnetv2s_seed2027 + 0.150*dfuc2021_cls_convnext_tiny_seed2027 + 0.400*dfuc2021_cls_efficientnetv2s_seed3407   |
| full_oof_selected_publication_fusion | 0.963928 | 0.958741          | 0.969379        | 0.958741     | 0.954492 | True   | [0.0, 0.1, -0.1, 0.0]                    | 0.400*dfuc2021_cls_efficientnetv2s_seed2027 + 0.200*dfuc2021_cls_convnext_tiny_seed2027 + 0.400*dfuc2021_cls_efficientnetv2s_seed3407   |
| full_oof_selected_publication_fusion | 0.963848 | 0.960135          | 0.967700        | 0.960135     | 0.955164 | True   | [-0.15, 0.1, -0.25, 0.0]                 | 0.400*dfuc2021_cls_efficientnetv2s_seed2027 + 0.300*dfuc2021_cls_convnext_tiny_seed2027 + 0.300*dfuc2021_cls_efficientnetv2s_seed3407   |
| full_oof_selected_publication_fusion | 0.963795 | 0.959439          | 0.968314        | 0.959439     | 0.954492 | True   | [0.0, 0.2, 0.0, 0.0]                     | 0.450*dfuc2021_cls_efficientnetv2s_seed2027 + 0.200*dfuc2021_cls_convnext_tiny_seed2027 + 0.350*dfuc2021_cls_efficientnetv2s_seed3407   |
| full_oof_selected_publication_fusion | 0.963751 | 0.960037          | 0.967597        | 0.960037     | 0.954996 | True   | [-0.2, 0.1, -0.15, 0.0]                  | 0.400*dfuc2021_cls_efficientnetv2s_seed2027 + 0.250*dfuc2021_cls_convnext_tiny_seed2027 + 0.350*dfuc2021_cls_efficientnetv2s_seed3407   |
| full_oof_selected_publication_fusion | 0.963595 | 0.959354          | 0.968088        | 0.959354     | 0.953820 | True   | [-0.125, -0.1, 0.0, 0.0]                 | 0.300*dfuc2021_cls_efficientnetv2s_seed2027 + 0.250*convnext_tiny + 0.450*dfuc2021_cls_efficientnetv2s_seed3407                         |
| full_oof_selected_publication_fusion | 0.963595 | 0.959354          | 0.968096        | 0.959354     | 0.953820 | True   | [-0.07500000000000001, -0.05, 0.1, 0.0]  | 0.350*dfuc2021_cls_efficientnetv2s_seed2027 + 0.200*convnext_tiny + 0.450*dfuc2021_cls_efficientnetv2s_seed3407                         |
| full_oof_selected_publication_fusion | 0.963556 | 0.960329          | 0.966869        | 0.960329     | 0.955500 | True   | [-0.39999999999999997, 0.0, -0.25, 0.0]  | 0.400*dfuc2021_cls_efficientnetv2s_seed2027 + 0.350*dfuc2021_cls_convnext_tiny_seed2027 + 0.250*dfuc2021_cls_efficientnetv2s_seed3407   |
| full_oof_selected_publication_fusion | 0.963539 | 0.963916          | 0.963340        | 0.963916     | 0.957011 | True   | [-0.35, -0.025, -0.25, 0.0]              | 0.400*strong_lpt_full_output_ensemble + 0.350*dfuc2021_cls_convnext_tiny_seed2027 + 0.250*dfuc2021_cls_efficientnetv2s_seed3407         |
| full_oof_selected_publication_fusion | 0.963503 | 0.958461          | 0.968829        | 0.958461     | 0.952813 | True   | [0.0, 0.025, 0.0, -0.25]                 | 0.200*dfuc2021_cls_efficientnetv2s_seed2027 + 0.400*convnext_tiny + 0.400*dfuc2021_cls_efficientnetv2s_seed3407                         |
| full_oof_selected_publication_fusion | 0.963463 | 0.960157          | 0.966871        | 0.960157     | 0.954156 | True   | [-0.5, -0.25, -0.25, 0.0]                | 0.350*dfuc2021_cls_efficientnetv2s_seed2027 + 0.300*strong_lpt_full_output_ensemble + 0.350*dfuc2021_cls_efficientnetv2s_seed3407       |
| full_oof_selected_publication_fusion | 0.963417 | 0.959962          | 0.967006        | 0.959962     | 0.953820 | True   | [-0.39999999999999997, -0.225, 0.0, 0.0] | 0.400*dfuc2021_cls_efficientnetv2s_seed2027 + 0.400*dfuc2021_cls_efficientnetv2s_seed3407 + 0.200*dfuc2021_cls_convnext_tiny_seed202607 |
| full_oof_selected_publication_fusion | 0.963415 | 0.959060          | 0.968030        | 0.959060     | 0.953317 | True   | [0.0, 0.0, -0.05, -0.1]                  | 0.500*dfuc2021_cls_efficientnetv2s_seed2027 + 0.150*dfuc2021_cls_convnext_tiny_seed2027 + 0.350*dfuc2021_cls_efficientnetv2s_seed3407   |
| full_oof_selected_publication_fusion | 0.963407 | 0.959156          | 0.967815        | 0.959156     | 0.953484 | True   | [-0.275, 0.0, 0.0, 0.0]                  | 0.400*dfuc2021_cls_efficientnetv2s_seed2027 + 0.200*convnext_tiny + 0.400*dfuc2021_cls_efficientnetv2s_seed3407                         |
| full_oof_selected_publication_fusion | 0.963354 | 0.958457          | 0.968419        | 0.958457     | 0.953484 | True   | [-0.25, 0.025, 0.1, 0.0]                 | 0.350*dfuc2021_cls_efficientnetv2s_seed2027 + 0.400*dfuc2021_cls_efficientnetv2s_seed3407 + 0.250*efficientnetv2s                       |
| full_oof_selected_publication_fusion | 0.963350 | 0.958462          | 0.968534        | 0.958462     | 0.952813 | True   | [0.0, 0.0, 0.0, 0.0]                     | 0.200*dfuc2021_cls_efficientnetv2s_seed2027 + 0.350*convnext_tiny + 0.450*dfuc2021_cls_efficientnetv2s_seed3407                         |

## Best Single/Existing Methods

| method                                  | source                                                                                                    | macro_f1 | balanced_accuracy | macro_precision | macro_recall | accuracy |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------- | -------- | ----------------- | --------------- | ------------ | -------- |
| dfuc2021_cls_efficientnetv2s_seed2027   | dfuc2021_cls_efficientnetv2s_seed2027_oof_predictions.csv                                                 | 0.951094 | 0.947797          | 0.954510        | 0.947797     | 0.941562 |
| strong_cls_ensemble                     | similarity_grouped_reliability_prior_fusion_strong_cls_ensemble_oof_predictions.csv             | 0.950799 | 0.954017          | 0.947819        | 0.954017     | 0.948111 |
| strong_lpt_full_output_ensemble         | similarity_grouped_reliability_prior_fusion_strong_lpt_full_output_ensemble_oof_predictions.csv | 0.950113 | 0.950987          | 0.949353        | 0.950987     | 0.948447 |
| strong_lpt_prior_fusion                 | similarity_grouped_reliability_prior_fusion_strong_lpt_prior_fusion_oof_predictions.csv         | 0.949925 | 0.951106          | 0.948858        | 0.951106     | 0.947607 |
| strong_lpt_full_prior_fusion            | similarity_grouped_reliability_prior_fusion_strong_lpt_full_prior_fusion_oof_predictions.csv    | 0.948456 | 0.947903          | 0.949066        | 0.947903     | 0.946599 |
| strong_cls_prior_residual               | similarity_grouped_reliability_prior_fusion_strong_cls_prior_residual_oof_predictions.csv       | 0.947674 | 0.951525          | 0.944153        | 0.951525     | 0.945928 |
| dfuc2021_cls_convnext_tiny_seed2027     | dfuc2021_cls_convnext_tiny_seed2027_oof_predictions.csv                                                   | 0.947466 | 0.947900          | 0.947184        | 0.947900     | 0.940386 |
| convnext_tiny                           | dfuc2021_cls_convnext_tiny_fold{fold}_val_predictions.csv                                                 | 0.947305 | 0.945132          | 0.949579        | 0.945132     | 0.938035 |
| strong_lpt_output_ensemble              | similarity_grouped_reliability_prior_fusion_strong_lpt_output_ensemble_oof_predictions.csv      | 0.946102 | 0.949420          | 0.942973        | 0.949420     | 0.945760 |
| dfuc2021_cls_efficientnetv2s_seed3407   | dfuc2021_cls_efficientnetv2s_seed3407_oof_predictions.csv                                                 | 0.944816 | 0.936396          | 0.953921        | 0.936396     | 0.936356 |
| efficientnetv2s                         | dfuc2021_cls_efficientnetv2s_fold{fold}_val_predictions.csv                                               | 0.943470 | 0.941301          | 0.945798        | 0.941301     | 0.936692 |
| dfuc2021_cls_convnext_tiny_seed202607   | dfuc2021_cls_convnext_tiny_seed202607_oof_predictions.csv                                                 | 0.942531 | 0.937231          | 0.948062        | 0.937231     | 0.935181 |
| dfuc2021_cls_convnext_tiny_seed777      | dfuc2021_cls_convnext_tiny_seed777_oof_predictions.csv                                                    | 0.940804 | 0.936956          | 0.944771        | 0.936956     | 0.933669 |
| dfuc2021_cls_convnext_tiny_seed3407     | dfuc2021_cls_convnext_tiny_seed3407_oof_predictions.csv                                                   | 0.938252 | 0.937648          | 0.938952        | 0.937648     | 0.934173 |
| lpt_ablation_output_ensemble            | similarity_grouped_reliability_prior_fusion_lpt_ablation_output_ensemble_oof_predictions.csv    | 0.938055 | 0.935874          | 0.940359        | 0.935874     | 0.936356 |
| lpt_model_ensemble                      | similarity_grouped_reliability_prior_fusion_lpt_model_ensemble_oof_predictions.csv              | 0.935832 | 0.935468          | 0.936285        | 0.935468     | 0.936860 |
| lpt_output_ensemble                     | similarity_grouped_reliability_prior_fusion_lpt_output_ensemble_oof_predictions.csv             | 0.935658 | 0.933756          | 0.937668        | 0.933756     | 0.937364 |
| lpt_ablation_ensemble                   | similarity_grouped_reliability_prior_fusion_lpt_ablation_ensemble_oof_predictions.csv           | 0.934316 | 0.934381          | 0.934308        | 0.934381     | 0.935516 |
| dfuc2021_cls_efficientnetv2s_seed777    | dfuc2021_cls_efficientnetv2s_seed777_oof_predictions.csv                                                  | 0.933437 | 0.928297          | 0.938904        | 0.928297     | 0.926952 |
| dfuc2021_cls_efficientnetv2s_seed202607 | dfuc2021_cls_efficientnetv2s_seed202607_oof_predictions.csv                                               | 0.932731 | 0.927292          | 0.938551        | 0.927292     | 0.926952 |

## Nested Fold Selections

| fold | train_macro_f1 | biased | bias                                     | candidate                                                                                                                               |
| ---- | -------------- | ------ | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 0    | 0.966385       | True   | [-0.25, -0.05, -0.5, 0.0]                | 0.400*dfuc2021_cls_convnext_tiny_seed2027 + 0.400*dfuc2021_cls_efficientnetv2s_seed2027 + 0.200*dfuc2021_cls_efficientnetv2s_seed3407   |
| 1    | 0.962401       | True   | [-0.25, -0.1, 0.0, -0.25]                | 0.400*dfuc2021_cls_efficientnetv2s_seed2027 + 0.200*dfuc2021_cls_convnext_tiny_seed202607 + 0.400*dfuc2021_cls_efficientnetv2s_seed3407 |
| 2    | 0.967719       | True   | [-0.375, -0.1, -0.25, -0.1]              | 0.300*dfuc2021_cls_efficientnetv2s_seed2027 + 0.400*dfuc2021_cls_efficientnetv2s_seed3407 + 0.300*efficientnetv2s                       |
| 3    | 0.965388       | True   | [-0.35, -0.025, -0.25, 0.0]              | 0.400*strong_lpt_full_output_ensemble + 0.350*dfuc2021_cls_convnext_tiny_seed2027 + 0.250*dfuc2021_cls_efficientnetv2s_seed3407         |
| 4    | 0.968503       | True   | [0.475, 0.35000000000000003, 0.0, -0.25] | 0.350*dfuc2021_cls_efficientnetv2s_seed2027 + 0.350*convnext_tiny + 0.300*dfuc2021_cls_efficientnetv2s_seed3407                         |
