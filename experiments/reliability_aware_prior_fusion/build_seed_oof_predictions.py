#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate seed-specific fold predictions into OOF prediction files.")
    parser.add_argument("--prediction-dir", type=Path, default=Path("outputs/publication_grouped/predictions"))
    parser.add_argument("--prefixes", nargs="+", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_dir = args.prediction_dir
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for prefix in args.prefixes:
        parts = []
        for fold in args.folds:
            path = prediction_dir / f"{prefix}_fold{fold}_val_predictions.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            if "fold" not in frame.columns:
                frame.insert(0, "fold", int(fold))
            parts.append(frame)
        output = pd.concat(parts, ignore_index=True)
        output.to_csv(prediction_dir / f"{prefix}_oof_predictions.csv", index=False)
        print(f"wrote {prediction_dir / f'{prefix}_oof_predictions.csv'} rows={len(output)}")


if __name__ == "__main__":
    main()
