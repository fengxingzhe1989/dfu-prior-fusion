#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CLASS_NAMES = ("none", "infection", "ischaemia", "both")
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {idx: name for name, idx in LABEL_TO_ID.items()}
PROB_COLUMNS = tuple(f"prob_{name}" for name in CLASS_NAMES)
EPS = 1e-8


@dataclass(frozen=True)
class MethodPredictions:
    name: str
    source: str
    frame: pd.DataFrame


@dataclass(frozen=True)
class Candidate:
    name: str
    indexes: tuple[int, ...]
    weights: tuple[float, ...]


BASE_SPECS = {
    "resnet50": "dfuc2021_cls_resnet50_fold{fold}_val_predictions.csv",
    "efficientnetb0": "dfuc2021_cls_efficientnetb0_fold{fold}_val_predictions.csv",
    "convnext_tiny": "dfuc2021_cls_convnext_tiny_fold{fold}_val_predictions.csv",
    "efficientnetv2s": "dfuc2021_cls_efficientnetv2s_fold{fold}_val_predictions.csv",
    "lpt_full": "cross_task_lpt_resnet50_mobilenetv3_fold{fold}_val_predictions.csv",
    "lpt_no_alignment": "cross_task_lpt_no_alignment_fold{fold}_val_predictions.csv",
    "lpt_no_prior_mod": "cross_task_lpt_no_prior_mod_fold{fold}_val_predictions.csv",
    "lpt_no_roi": "cross_task_lpt_no_roi_fold{fold}_val_predictions.csv",
    "lpt_stats_only": "cross_task_lpt_stats_only_fold{fold}_val_predictions.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search publication-oriented probability fusions from fold-strict OOF predictions.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--prediction-dir", type=Path, default=Path("outputs/publication_grouped/predictions"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/publication_grouped/publication_fusion_search"))
    parser.add_argument("--output-prefix", default="publication_fusion")
    parser.add_argument("--top-methods", type=int, default=18)
    parser.add_argument("--top-triples", type=int, default=12)
    parser.add_argument("--pair-step", type=float, default=0.025)
    parser.add_argument("--triple-step", type=float, default=0.05)
    parser.add_argument("--top-bias-candidates", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument(
        "--skip-invalid-oof",
        action="store_true",
        help="Skip malformed *_oof_predictions.csv files after writing an explicit skip log. By default, malformed files fail the search.",
    )
    return parser.parse_args()


def resolve(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else repo_root / path


def labels_to_id(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).to_numpy()
    return series.astype(str).map(LABEL_TO_ID).astype(int).to_numpy()


def read_prediction(path: Path, fold: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "label" in frame.columns:
        y_true = labels_to_id(frame["label"])
    elif "true_label" in frame.columns:
        y_true = labels_to_id(frame["true_label"])
    else:
        raise ValueError(f"Missing label/true_label in {path}")
    if "fold" in frame.columns:
        fold_values = frame["fold"].astype(int).to_numpy()
    elif fold is not None:
        fold_values = np.full(len(frame), int(fold), dtype=int)
    else:
        raise ValueError(f"Missing fold in {path}")
    output = pd.DataFrame(
        {
            "image": frame["image"].astype(str),
            "label": y_true,
            "fold": fold_values,
        }
    )
    for column in PROB_COLUMNS:
        if column not in frame.columns:
            raise ValueError(f"Missing {column} in {path}")
        output[column] = frame[column].astype(float)
    probs = output[list(PROB_COLUMNS)].to_numpy(dtype=float)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), EPS, None)
    output[list(PROB_COLUMNS)] = probs
    output["pred_id"] = np.argmax(probs, axis=1)
    return output


def load_fold_spec(prediction_dir: Path, name: str, pattern: str) -> MethodPredictions | None:
    parts = []
    for fold in range(5):
        path = prediction_dir / pattern.format(fold=fold)
        if not path.exists():
            return None
        parts.append(read_prediction(path, fold=fold))
    frame = pd.concat(parts, ignore_index=True).sort_values("image").reset_index(drop=True)
    return MethodPredictions(name=name, source=pattern, frame=frame)


def prettify_oof_name(path: Path, output_prefix_hint: str) -> str:
    stem = path.name.removesuffix("_oof_predictions.csv")
    for prefix in (
        output_prefix_hint,
        "similarity_grouped_reliability_prior_fusion_",
        "reliability_prior_fusion_",
    ):
        stem = stem.removeprefix(prefix)
    return stem


def load_methods(prediction_dir: Path, output_prefix_hint: str, skip_invalid_oof: bool) -> tuple[list[MethodPredictions], list[dict[str, str]]]:
    methods: list[MethodPredictions] = []
    skipped: list[dict[str, str]] = []
    for name, pattern in BASE_SPECS.items():
        item = load_fold_spec(prediction_dir, name, pattern)
        if item is not None:
            methods.append(item)
    for path in sorted(prediction_dir.glob("*_oof_predictions.csv")):
        try:
            frame = read_prediction(path)
        except (KeyError, ValueError, pd.errors.ParserError) as exc:
            if not skip_invalid_oof:
                raise ValueError(f"Could not parse candidate OOF prediction file {path}: {exc}") from exc
            skipped.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        methods.append(MethodPredictions(name=prettify_oof_name(path, output_prefix_hint), source=path.name, frame=frame.sort_values("image").reset_index(drop=True)))
    if not methods:
        raise ValueError(f"No predictions found in {prediction_dir}")
    reference = methods[0].frame[["image", "label", "fold"]].reset_index(drop=True)
    for method in methods[1:]:
        current = method.frame[["image", "label", "fold"]].reset_index(drop=True)
        if not reference.equals(current):
            raise ValueError(f"Prediction rows are not aligned for {method.name}")
    return methods, skipped


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    encoded = len(CLASS_NAMES) * y_true.astype(int) + y_pred.astype(int)
    matrix = np.bincount(encoded, minlength=len(CLASS_NAMES) ** 2).reshape(len(CLASS_NAMES), len(CLASS_NAMES)).astype(float)
    tp = np.diag(matrix)
    pred_count = matrix.sum(axis=0)
    true_count = matrix.sum(axis=1)
    precision_values = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall_values = np.divide(tp, true_count, out=np.zeros_like(tp), where=true_count > 0)
    denom = precision_values + recall_values
    f1_values = np.divide(2.0 * precision_values * recall_values, denom, out=np.zeros_like(tp), where=denom > 0)
    total = max(1.0, float(matrix.sum()))
    return {
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recall_values)),
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "accuracy": float(tp.sum() / total),
    }


def weighted_probs(prob_stack: np.ndarray, candidate: Candidate) -> np.ndarray:
    weights = np.array(candidate.weights, dtype=float)
    probs = np.tensordot(weights, prob_stack[list(candidate.indexes)], axes=(0, 0))
    return probs / np.clip(probs.sum(axis=1, keepdims=True), EPS, None)


def apply_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probs, EPS, 1.0)) + bias.reshape(1, -1)
    return np.argmax(logits, axis=1)


def optimize_bias(probs: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, float]:
    bias = np.zeros(len(CLASS_NAMES), dtype=float)
    best_pred = apply_bias(probs, bias)
    best = metrics(y_true, best_pred)["macro_f1"]
    for step in (0.75, 0.5, 0.25, 0.1, 0.05, 0.025):
        improved = True
        while improved:
            improved = False
            for class_idx in range(len(CLASS_NAMES)):
                for delta in (-step, step):
                    candidate = bias.copy()
                    candidate[class_idx] = float(np.clip(candidate[class_idx] + delta, -3.0, 3.0))
                    score = metrics(y_true, apply_bias(probs, candidate))["macro_f1"]
                    if score > best + 1e-12:
                        best = score
                        bias = candidate
                        improved = True
    return bias, best


def weight_grid(size: int, step: float) -> Iterable[tuple[float, ...]]:
    units = int(round(1.0 / step))
    if size == 1:
        yield (1.0,)
    elif size == 2:
        for first in range(units + 1):
            w0 = first / units
            yield (w0, 1.0 - w0)
    elif size == 3:
        for first in range(units + 1):
            for second in range(units + 1 - first):
                third = units - first - second
                yield (first / units, second / units, third / units)
    else:
        raise ValueError(size)


def build_candidates(single_scores: list[tuple[int, float]], top_methods: int, top_triples: int, pair_step: float, triple_step: float) -> list[Candidate]:
    ranked = [idx for idx, _ in sorted(single_scores, key=lambda item: item[1], reverse=True)]
    top_pair = ranked[: min(top_methods, len(ranked))]
    top_triple = ranked[: min(top_triples, len(ranked))]
    candidates: list[Candidate] = []
    for idx in ranked:
        candidates.append(Candidate(name=f"single:{idx}", indexes=(idx,), weights=(1.0,)))
    for i, j in itertools.combinations(top_pair, 2):
        for weights in weight_grid(2, pair_step):
            if min(weights) <= 0.0:
                continue
            candidates.append(Candidate(name=f"pair:{i},{j}:{weights[0]:.3f},{weights[1]:.3f}", indexes=(i, j), weights=weights))
    for combo in itertools.combinations(top_triple, 3):
        for weights in weight_grid(3, triple_step):
            if min(weights) <= 0.0:
                continue
            candidates.append(
                Candidate(
                    name="triple:" + ",".join(map(str, combo)) + ":" + ",".join(f"{w:.3f}" for w in weights),
                    indexes=tuple(combo),
                    weights=weights,
                )
            )
    return candidates


def describe_candidate(candidate: Candidate, methods: list[MethodPredictions]) -> str:
    return " + ".join(f"{weight:.3f}*{methods[index].name}" for index, weight in zip(candidate.indexes, candidate.weights))


def evaluate_candidates(
    prob_stack: np.ndarray,
    y_true: np.ndarray,
    candidates: list[Candidate],
    top_bias_candidates: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scored: list[tuple[float, Candidate, np.ndarray]] = []
    for candidate in candidates:
        probs = weighted_probs(prob_stack, candidate)
        pred = np.argmax(probs, axis=1)
        score = metrics(y_true, pred)["macro_f1"]
        scored.append((score, candidate, probs))
        rows.append({"candidate": candidate, "bias": np.zeros(len(CLASS_NAMES)), "biased": False, **metrics(y_true, pred)})
    for _, candidate, probs in sorted(scored, key=lambda item: item[0], reverse=True)[:top_bias_candidates]:
        bias, _ = optimize_bias(probs, y_true)
        pred = apply_bias(probs, bias)
        rows.append({"candidate": candidate, "bias": bias, "biased": True, **metrics(y_true, pred)})
    return rows


def fold_held_out_selection(
    methods: list[MethodPredictions],
    prob_stack: np.ndarray,
    y_true: np.ndarray,
    folds: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[dict[str, object]]]:
    """Trace second-level selection across folds of precomputed OOF predictions.

    This excludes each assessment fold from candidate, weight, and bias selection,
    but it does not retrain the base CNNs inside each fold-held-out split. It is
    therefore not a full end-to-end nested-validation procedure.
    """
    final_probs = np.zeros((len(y_true), len(CLASS_NAMES)), dtype=float)
    selections: list[dict[str, object]] = []
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != fold
        val_mask = folds == fold
        single_scores = []
        for idx in range(len(methods)):
            pred = np.argmax(prob_stack[idx, train_mask], axis=1)
            single_scores.append((idx, metrics(y_true[train_mask], pred)["macro_f1"]))
        candidates = build_candidates(single_scores, args.top_methods, args.top_triples, args.pair_step, args.triple_step)
        rows = evaluate_candidates(prob_stack[:, train_mask, :], y_true[train_mask], candidates, args.top_bias_candidates)
        best = max(rows, key=lambda row: row["macro_f1"])
        candidate = best["candidate"]
        bias = best["bias"]
        probs = weighted_probs(prob_stack[:, val_mask, :], candidate)
        final_probs[val_mask] = probs
        selections.append(
            {
                "fold": int(fold),
                "train_macro_f1": float(best["macro_f1"]),
                "biased": bool(best["biased"]),
                "bias": json.dumps([float(x) for x in np.asarray(bias)]),
                "candidate": describe_candidate(candidate, methods),
            }
        )
    final_pred = np.array([apply_bias(final_probs[folds == fold], np.asarray(json.loads(sel["bias"]))) for fold, sel in zip(sorted(set(folds.tolist())), selections)], dtype=object)
    pred_ids = np.zeros(len(y_true), dtype=int)
    for fold, arr in zip(sorted(set(folds.tolist())), final_pred):
        pred_ids[folds == fold] = arr
    trace_metrics = pd.DataFrame([{"method": "precomputed_oof_fold_held_out_fusion_trace", **metrics(y_true, pred_ids)}])
    return trace_metrics, pd.DataFrame(selections), final_probs, selections


def save_predictions(path: Path, reference: pd.DataFrame, probs: np.ndarray, bias_by_fold: dict[int, np.ndarray] | None = None, global_bias: np.ndarray | None = None) -> None:
    pred_ids = np.zeros(len(reference), dtype=int)
    for fold in sorted(reference["fold"].unique()):
        mask = reference["fold"].to_numpy(dtype=int) == int(fold)
        bias = global_bias if global_bias is not None else bias_by_fold.get(int(fold), np.zeros(len(CLASS_NAMES)))
        pred_ids[mask] = apply_bias(probs[mask], bias)
    output = reference[["image", "label", "fold"]].copy()
    output["label_name"] = [ID_TO_LABEL[int(v)] for v in output["label"]]
    output["pred_label"] = [ID_TO_LABEL[int(v)] for v in pred_ids]
    for idx, class_name in enumerate(CLASS_NAMES):
        output[f"prob_{class_name}"] = probs[:, idx]
    output.to_csv(path, index=False)


def markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    show = frame.head(max_rows).copy()
    for column in show.columns:
        if pd.api.types.is_float_dtype(show[column]):
            show[column] = show[column].map(lambda x: f"{x:.6f}")
    widths = [max(len(str(column)), *(len(str(value)) for value in show[column])) for column in show.columns]
    header = "| " + " | ".join(str(column).ljust(widths[idx]) for idx, column in enumerate(show.columns)) + " |"
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(str(row[column]).ljust(widths[idx]) for idx, column in enumerate(show.columns)) + " |"
        for _, row in show.iterrows()
    ]
    return "\n".join([header, divider, *body])


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    prediction_dir = resolve(args.prediction_dir, repo_root)
    output_dir = resolve(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods, skipped_oof_files = load_methods(prediction_dir, args.output_prefix, args.skip_invalid_oof)
    reference = methods[0].frame[["image", "label", "fold"]].copy()
    y_true = reference["label"].to_numpy(dtype=int)
    folds = reference["fold"].to_numpy(dtype=int)
    prob_stack = np.stack([method.frame[list(PROB_COLUMNS)].to_numpy(dtype=float) for method in methods], axis=0)

    single_rows = []
    single_scores = []
    for idx, method in enumerate(methods):
        pred = np.argmax(prob_stack[idx], axis=1)
        row = {"method": method.name, "source": method.source, **metrics(y_true, pred)}
        single_rows.append(row)
        single_scores.append((idx, row["macro_f1"]))
    single_metrics = pd.DataFrame(single_rows).sort_values("macro_f1", ascending=False)

    candidates = build_candidates(single_scores, args.top_methods, args.top_triples, args.pair_step, args.triple_step)
    full_rows = evaluate_candidates(prob_stack, y_true, candidates, args.top_bias_candidates)
    full_table_rows = []
    for row in full_rows:
        candidate = row["candidate"]
        bias = np.asarray(row["bias"], dtype=float)
        full_table_rows.append(
            {
                "method": "full_oof_selected_publication_fusion",
                "macro_f1": row["macro_f1"],
                "balanced_accuracy": row["balanced_accuracy"],
                "macro_precision": row["macro_precision"],
                "macro_recall": row["macro_recall"],
                "accuracy": row["accuracy"],
                "biased": bool(row["biased"]),
                "bias": json.dumps([float(x) for x in bias]),
                "candidate": describe_candidate(candidate, methods),
                "_candidate_obj": candidate,
                "_bias_obj": bias,
            }
        )
    full_metrics = pd.DataFrame(full_table_rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    best_full = full_metrics.iloc[0]
    best_candidate: Candidate = best_full["_candidate_obj"]
    best_bias = np.asarray(best_full["_bias_obj"], dtype=float)
    best_probs = weighted_probs(prob_stack, best_candidate)

    trace_metrics, trace_selections, trace_probs, selections = fold_held_out_selection(methods, prob_stack, y_true, folds, args)
    bias_by_fold = {int(row["fold"]): np.asarray(json.loads(row["bias"]), dtype=float) for row in selections}

    single_metrics.to_csv(output_dir / f"{args.output_prefix}_single_metrics.csv", index=False)
    full_metrics.drop(columns=["_candidate_obj", "_bias_obj"]).to_csv(output_dir / f"{args.output_prefix}_full_oof_fusion_search.csv", index=False)
    # Legacy filenames are retained for compatibility with archived publication assets.
    trace_metrics.to_csv(output_dir / f"{args.output_prefix}_nested_metrics.csv", index=False)
    trace_selections.to_csv(output_dir / f"{args.output_prefix}_nested_selections.csv", index=False)
    if skipped_oof_files:
        pd.DataFrame(skipped_oof_files).to_csv(output_dir / f"{args.output_prefix}_skipped_oof_files.csv", index=False)
        print(f"Skipped {len(skipped_oof_files)} malformed OOF files; see skipped_oof_files CSV.", file=sys.stderr)
    save_predictions(output_dir / f"{args.output_prefix}_full_oof_selected_predictions.csv", reference, best_probs, global_bias=best_bias)
    save_predictions(output_dir / f"{args.output_prefix}_nested_selected_predictions.csv", reference, trace_probs, bias_by_fold=bias_by_fold)

    report = [
        "# Publication Fusion Search",
        "",
        "This analysis searches probability-level fusion weights from fold-strict OOF predictions.",
        "The full-OOF selected candidate is an internally optimised development estimate.",
        "The fold-held-out output traces second-level selection over precomputed OOF predictions; it is not full end-to-end nested validation.",
        "",
        "## Precomputed-OOF Fold-Held-Out Selection Trace",
        "",
        markdown_table(trace_metrics),
        "",
        "## Full OOF Development Upper Bound",
        "",
        markdown_table(full_metrics.drop(columns=["_candidate_obj", "_bias_obj"])),
        "",
        "## Best Single/Existing Methods",
        "",
        markdown_table(single_metrics),
        "",
        "## Fold-Held-Out Selections",
        "",
        markdown_table(trace_selections, max_rows=10),
        "",
    ]
    (output_dir / f"{args.output_prefix}_report.md").write_text("\n".join(report), encoding="utf-8")
    print(trace_metrics.to_string(index=False))
    print()
    print(full_metrics.drop(columns=["_candidate_obj", "_bias_obj"]).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
