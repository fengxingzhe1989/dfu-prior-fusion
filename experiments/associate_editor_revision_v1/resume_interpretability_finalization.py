#!/usr/bin/env python3
"""Resume only summary/render after all 50 frozen interpretation jobs complete.

Run after the original inference runner has exited, never concurrently with it.
This additive orchestration repair does not modify any sealed code, protocol,
case, placement, prediction, or checkpoint. Completed stages are hash-verified
and retained. A stage directory without its completion marker is moved intact
to a timestamped recovery archive before the original sealed stage is rerun.
The existing summary validator is reused; no PyTorch/CUDA environment is required.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path("experiments/associate_editor_revision_v1")
MANIFEST = Path("data_processed/dfuc2021_cls_v1/similarity_grouped_splits/dfuc2021_cls_cv5_similarity_grouped_manifest.csv")
SEEDS = tuple(range(20260925, 20260930))
MODELS = ("resnet50", "lpt_full")
STAGES = {"resnet50": "refit", "lpt_full": "direct_refit"}
N_IMAGES, N_GROUPS = 5955, 5670
SUMMARY_ASSETS = {
    "seed_metrics.csv", "model_summary.csv", "coverage_audit.csv",
    "paired_seed_metrics.csv", "bootstrap_intervals.csv", "bootstrap_replicates.csv",
    "input_registry.csv", "bootstrap_design.json", "summary.md",
}
SLUGS = ("neither", "infection", "ischaemia", "both")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def inside(root: Path, name: str, *, allow_absolute: bool = False) -> Path:
    candidate = Path(name)
    require(allow_absolute or not candidate.is_absolute(), f"Absolute asset path forbidden: {name}")
    path = (root / candidate).resolve()
    require(path.is_relative_to(root.resolve()), f"Asset path escapes root: {name}")
    require(path.is_file(), f"Asset absent: {path}")
    return path


def verify_assets(root: Path, assets: dict[str, str]) -> dict[Path, str]:
    require(isinstance(assets, dict) and bool(assets), f"Empty asset registry: {root}")
    checked = {}
    for name, digest in assets.items():
        path = inside(root, name)
        require(path not in checked, f"Duplicate resolved asset: {path}")
        require(sha(path) == digest, f"Asset hash mismatch: {path}")
        checked[path] = digest
    return checked


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp_" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit_jobs(repo: Path, out: Path) -> list[dict[str, str]]:
    """Reuse sealed validation of all 50 markers/assets and complete coverage."""
    spec = importlib.util.spec_from_file_location("dfu_frozen_summary_audit", repo / EXPERIMENT / "summarize_interpretability.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, _, inputs, _ = module.load_and_audit(repo, out)
    return inputs.to_dict("records")


class Audit:
    def __init__(self, repo: Path, out: Path) -> None:
        self.repo = repo.resolve()
        self.out = out.resolve()
        require(self.out.is_relative_to(self.repo), "Artifact root must be within its original repository")
        self.base = self.out / "interpretability"
        self.registry_path = self.base / "source_registry.json"
        self.registry_hash = sha(self.registry_path)
        self.summary_inputs: dict[Path, tuple[str, str]] = {}
        self.job_assets: dict[Path, str] = {}
        self.cases: list[dict[str, str]] = []

    def sources_and_jobs(self) -> None:
        registry = read_json(self.registry_path)
        require(registry.get("status") == "frozen", "Source registry is not frozen")
        require(tuple(registry.get("seeds", [])) == SEEDS and tuple(registry.get("models", [])) == MODELS,
                "Frozen model/seed matrix differs")
        require((registry.get("n_images"), registry.get("n_groups")) == (N_IMAGES, N_GROUPS), "Frozen sample counts differ")
        require(set(registry.get("assets", {})) == {
            "qualitative_cases.csv", "execution_contract.json", "placements.json", "input_manifest.csv",
        }, "Source-registry asset inventory differs")
        verify_assets(self.base, registry["assets"])
        code_assets = verify_assets(self.repo, registry.get("code_assets", {}))
        for name in ("run_interpretability.py", "interpretability_geometry.py",
                     "summarize_interpretability.py", "render_interpretability.py", "protocol.yaml"):
            require((self.repo / EXPERIMENT / name).resolve() in code_assets, f"Unsealed execution file: {name}")
        source_assets = verify_assets(self.repo, registry.get("source_assets", {}))
        manifest = self.repo / MANIFEST
        require(manifest.resolve() in source_assets, "Grouped manifest not sealed")
        input_rows = read_csv(self.base / "input_manifest.csv")
        require(len(input_rows) == N_IMAGES and len({r["image"] for r in input_rows}) == N_IMAGES, "Input inventory differs")
        for row in input_rows:
            for prefix in ("image", "prior"):
                path = inside(self.repo, row[prefix + "_path"])
                require(sha(path) == row[prefix + "_sha256"], f"Changed original {prefix}: {path}")
        self.cases = read_csv(self.base / "qualitative_cases.csv")
        require(registry.get("qualitative_cases_sha256") == sha(self.base / "qualitative_cases.csv"), "Case hash differs")
        for row in audit_jobs(self.repo, self.out):
            path = inside(self.repo, row["path"], allow_absolute=True)
            require(path not in self.summary_inputs, f"Duplicate audited input: {path}")
            self.summary_inputs[path] = (row["role"], row["sha256"])
            if row["role"] == "job_asset":
                self.job_assets[path] = row["sha256"]
        print("AUDIT_OK frozen sources and all 50 interpretation jobs", flush=True)

    def summary(self) -> dict[str, Any]:
        directory = self.base / "summary"
        marker = read_json(directory / "summary_complete.json")
        require(marker.get("status") == "complete" and marker.get("analysis") == "frozen_grouped_interpretability", "Invalid summary marker")
        require(tuple(marker.get("seeds", [])) == SEEDS and tuple(marker.get("models", [])) == MODELS, "Summary matrix differs")
        require((marker.get("n_images"), marker.get("n_groups"), marker.get("n_image_seed_model_rows"))
                == (N_IMAGES, N_GROUPS, N_IMAGES * len(SEEDS) * len(MODELS)), "Summary sample counts differ")
        require(marker.get("audited_inputs", {}).get("source_registry_sha256") == self.registry_hash
                and marker["audited_inputs"].get("qualitative_cases_sha256") == sha(self.base / "qualitative_cases.csv")
                and marker["audited_inputs"].get("audited_jobs") == 50, "Summary provenance differs")
        require(marker.get("bootstrap", {}).get("replicates") == 5000
                and marker["bootstrap"].get("seed") == 20261001, "Summary bootstrap setting differs")
        require(set(marker.get("assets", {})) == SUMMARY_ASSETS, "Summary asset inventory differs")
        verify_assets(directory, marker["assets"])
        observed = {}
        for row in read_csv(directory / "input_registry.csv"):
            path = inside(self.repo, row["path"], allow_absolute=True)
            require(path not in observed, f"Duplicate summary input: {path}")
            observed[path] = (row["role"], row["sha256"])
            require(sha(path) == row["sha256"], f"Changed summary input: {path}")
        require(observed == self.summary_inputs, "Summary input registry differs from all frozen jobs")
        return marker

    def figures(self) -> dict[str, Any]:
        directory = self.base / "figures"
        marker = read_json(directory / "figures_complete.json")
        require(marker.get("status") == "complete", "Invalid figure marker")
        require((marker.get("case_count"), marker.get("seed_count"), marker.get("model_count"),
                 marker.get("plate_count"), marker.get("source_npz_count")) == (20, 5, 2, 20, 200), "Figure counts differ")
        require(marker.get("dpi") == 300 and marker.get("cam_target") == "archived_oof_predicted_class", "Figure settings differ")
        expected_files = {f"qualitative_seed_{seed}_{slug}.{ext}" for seed in SEEDS for slug in SLUGS for ext in ("png", "pdf", "svg")}
        expected_files |= {f"qualitative_seed_{seed}_all_classes.pdf" for seed in SEEDS} | {"captions_and_methods.txt", "source_registry.csv"}
        checked = {}
        for asset in marker.get("assets", []):
            path = inside(self.out, asset["path"])
            require(path.parent == directory and path.name not in checked, f"Duplicate or misplaced figure asset: {path}")
            require(sha(path) == asset["sha256"] and path.stat().st_size == asset["size_bytes"], f"Figure asset differs: {path}")
            checked[path.name] = asset["sha256"]
        require(set(checked) == expected_files, "Figure asset inventory differs")
        expected_sources = {(self.base / "qualitative_cases.csv").resolve(): ("frozen_case_manifest", sha(self.base / "qualitative_cases.csv"))}
        renderer = (self.repo / EXPERIMENT / "render_interpretability.py").resolve()
        expected_sources[renderer] = ("renderer", sha(renderer))
        for seed in SEEDS:
            for case in self.cases:
                for model in MODELS:
                    path = (self.base / "jobs" / f"seed_{seed}" / f"outer_{case['fold']}" / model / "cases" / f"{Path(case['image']).stem}.npz").resolve()
                    require(path in self.job_assets, f"Case not sealed by job: {path}")
                    expected_sources[path] = ("case_array", self.job_assets[path])
        observed = {}
        for row in read_csv(directory / "source_registry.csv"):
            path = inside(self.repo if row["kind"] == "renderer" else self.out, row["path"])
            require(path not in observed, f"Duplicate figure input: {path}")
            require(sha(path) == row["sha256"] and path.stat().st_size == int(row["size_bytes"]), f"Changed figure input: {path}")
            observed[path] = (row["kind"], row["sha256"])
        require(observed == expected_sources, "Figure input registry differs from frozen cases/jobs")
        return marker


def archive_incomplete(directory: Path, marker_name: str) -> Path | None:
    require(not (directory / marker_name).exists(), "Never archive a stage with a completed marker automatically")
    if not directory.exists():
        return None
    require(directory.is_dir() and not directory.is_symlink(), f"Refusing non-directory/symlink stage: {directory}")
    archived = directory.with_name(directory.name + ".incomplete_archive_" + stamp())
    directory.rename(archived)
    print(f"PRESERVED incomplete stage: {archived}", flush=True)
    return archived


def finalize(repo: Path, out: Path, verify_only: bool = False) -> dict[str, Any]:
    audit = Audit(repo, out)
    audit.sources_and_jobs()
    records = []
    validators = {"summary": audit.summary, "figures": audit.figures}
    for stage, marker_name, script in [
        ("summary", "summary_complete.json", "summarize_interpretability.py"),
        ("figures", "figures_complete.json", "render_interpretability.py"),
    ]:
        directory = audit.base / stage
        marker_path = directory / marker_name
        if marker_path.exists():
            validators[stage]()
            print(f"VERIFIED_SKIP {stage}", flush=True)
            records.append({"stage": stage, "action": "verified_existing"})
        elif verify_only:
            raise FileNotFoundError(f"Stage not complete: {marker_path}")
        else:
            archived = archive_incomplete(directory, marker_name)
            subprocess.run([sys.executable, str(audit.repo / EXPERIMENT / script),
                            "--repo-root", str(audit.repo), "--output-root", str(audit.out)], cwd=audit.repo, check=True)
            validators[stage]()
            records.append({"stage": stage, "action": "completed_from_frozen_script", "archived_partial": str(archived) if archived else None})
    result = {
        "status": "complete", "jobs": 50, "images_per_model_per_seed": N_IMAGES,
        "source_registry_sha256": audit.registry_hash,
        "summary_complete_sha256": sha(audit.base / "summary/summary_complete.json"),
        "figures_complete_sha256": sha(audit.base / "figures/figures_complete.json"),
        "recovery_script_sha256": sha(Path(__file__).resolve()),
        "recovery_script": str(Path(__file__).resolve()),
        "visual_qa": "required_after_export; computational completion does not assert visual acceptance",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    final_path = audit.base / "interpretability_complete.json"
    previous = read_json(final_path) if final_path.exists() else None
    if previous:
        for key in ("status", "jobs", "images_per_model_per_seed", "source_registry_sha256", "summary_complete_sha256",
                    "figures_complete_sha256", "recovery_script_sha256"):
            if key in previous:
                require(previous[key] == result[key], f"Existing overall completion conflicts: {key}; no overwrite")
        if all(previous.get(key) == result[key] for key in ("figures_complete_sha256", "recovery_script_sha256")):
            return previous
    if verify_only:
        return {**result, "status": "verified_only", "overall_marker_updated": False}
    if previous:
        archived_marker = final_path.with_name(final_path.name + ".legacy_archive_" + stamp())
        final_path.rename(archived_marker)
        records.append({"stage": "overall", "action": "preserved_legacy_marker", "path": str(archived_marker)})
    audit_path = audit.base / ("finalization_recovery_audit_" + stamp() + ".json")
    atomic_json(audit_path, {"actions": records, **result})
    result["recovery_audit_path"] = str(audit_path)
    result["recovery_audit_sha256"] = sha(audit_path)
    atomic_json(final_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/associate_editor_revision_v1"))
    parser.add_argument("--verify-only", action="store_true", help="Read-only validation of all completed stages")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    out = args.output_root if args.output_root.is_absolute() else repo / args.output_root
    if args.verify_only:
        result = finalize(repo, out, True)
    else:
        # This serializes this recovery script only. The original runner must have exited.
        with (out / "interpretability/finalization_recovery.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = finalize(repo, out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
