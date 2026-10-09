"""Synthetic, CPU-only tests; never load real study artifacts or run analyses."""
from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("resume_interpretability_finalization.py")
SPEC = importlib.util.spec_from_file_location("dfu_finalizer", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def put(path: Path, value: str = "synthetic\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return path


def js(path: Path, value: dict) -> Path:
    return put(path, json.dumps(value))


def table(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dfu_finalizer_test_")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.out = self.repo / "outputs/associate_editor_revision_v1"
        self.base = self.out / "interpretability"
        self.base.mkdir(parents=True)
        self.patches = [patch.object(M, "N_IMAGES", 20), patch.object(M, "N_GROUPS", 20)]
        for context in self.patches:
            context.start()
            self.addCleanup(context.stop)
        references = [dict(image=f"image_{i:02d}.png", label=str(i // 5), similarity_group=f"g{i:02d}", fold=str(i % 5)) for i in range(20)]
        self.refs = references
        sources, codes = {}, {}
        for filename in ("run_interpretability.py", "interpretability_geometry.py", "summarize_interpretability.py", "render_interpretability.py", "protocol.yaml"):
            path = put(self.repo / M.EXPERIMENT / filename)
            codes[str(path.relative_to(self.repo))] = M.sha(path)
        manifest = table(self.repo / M.MANIFEST, references)
        sources[str(manifest.relative_to(self.repo))] = M.sha(manifest)
        inputs = []
        for row in references:
            image = put(self.repo / "data_raw/images" / row["image"], "RGB " + row["image"])
            prior = put(self.repo / "data_processed/priors" / row["image"], "PRIOR " + row["image"])
            inputs.append({**row, "image_path": str(image.relative_to(self.repo)), "image_sha256": M.sha(image),
                           "prior_path": str(prior.relative_to(self.repo)), "prior_sha256": M.sha(prior)})
        table(self.base / "input_manifest.csv", inputs)
        self.cases = [{"case_rank": i + 1, **row} for i, row in enumerate(references)]
        table(self.base / "qualitative_cases.csv", self.cases)
        js(self.base / "placements.json", {})
        js(self.base / "execution_contract.json", {"status": "synthetic"})
        for seed in M.SEEDS:
            for fold in range(5):
                for model in M.MODELS:
                    source = self.out / "classification" / f"seed_{seed}" / f"outer_{fold}" / model / M.STAGES[model]
                    ckpt = put(source / "model.pt")
                    pred = put(source / "predictions.csv")
                    done = js(source / "complete.json", {
                        "status": "complete", "base_seed": seed, "outer_fold": fold, "model": model, "stage": M.STAGES[model],
                        "checkpoint_sha256": M.sha(ckpt), "prediction_sha256": M.sha(pred),
                    })
                    for path in (ckpt, pred, done):
                        sources[str(path.relative_to(self.repo))] = M.sha(path)
        assets = {name: M.sha(self.base / name) for name in ("qualitative_cases.csv", "execution_contract.json", "placements.json", "input_manifest.csv")}
        registry = js(self.base / "source_registry.json", {
            "status": "frozen", "seeds": list(M.SEEDS), "models": list(M.MODELS), "n_images": 20, "n_groups": 20,
            "assets": assets, "code_assets": codes, "source_assets": sources,
            "qualitative_cases_sha256": assets["qualitative_cases.csv"],
        })
        self.registry_hash = M.sha(registry)
        for seed in M.SEEDS:
            for fold in range(5):
                for model in M.MODELS:
                    job = self.base / "jobs" / f"seed_{seed}" / f"outer_{fold}" / model
                    rows = [{"image": row["image"], "label": row["label"], "similarity_group": row["similarity_group"],
                             "outer_fold": fold, "seed": seed, "model": model} for row in references if int(row["fold"]) == fold]
                    paths = [table(job / "metrics.csv", rows), put(job / "placements.csv"), js(job / "replay.json", {})]
                    paths.extend(put(job / "cases" / f"{Path(row['image']).stem}.npz") for row in references if int(row["fold"]) == fold)
                    js(job / "complete.json", {
                        "status": "complete", "seed": seed, "outer_fold": fold, "model": model, "images": 4, "smoke": False,
                        "source_registry_sha256": self.registry_hash, "checkpoint_sha256": M.sha(self.out / "classification" / f"seed_{seed}" / f"outer_{fold}" / model / M.STAGES[model] / "model.pt"),
                        "assets": {str(path.relative_to(job)): M.sha(path) for path in paths},
                    })
        # Stage-resumption tests isolate the previously tested sealed scientific
        # validator; the stand-in still verifies all 50 marker/asset hash chains.
        context = patch.object(M, "audit_jobs", side_effect=self.fake_job_audit)
        context.start()
        self.addCleanup(context.stop)

    def fake_job_audit(self, repo, out):
        rows = []
        def add(path, role):
            rows.append({"path": str(path.resolve()), "role": role, "sha256": M.sha(path)})
        for path, role in [(repo / M.EXPERIMENT / "protocol.yaml", "protocol"),
                           (repo / M.MANIFEST, "grouped_manifest"),
                           (self.base / "source_registry.json", "source_registry"),
                           (self.base / "qualitative_cases.csv", "qualitative_cases")]:
            add(path, role)
        for seed in M.SEEDS:
            for fold in range(5):
                for model in M.MODELS:
                    job = self.base / "jobs" / f"seed_{seed}" / f"outer_{fold}" / model
                    marker = M.read_json(job / "complete.json")
                    M.require(marker["status"] == "complete", "incomplete synthetic job")
                    M.require(marker["source_registry_sha256"] == self.registry_hash, "wrong synthetic source")
                    add(job / "complete.json", "job_completion")
                    for path in M.verify_assets(job, marker["assets"]):
                        add(path, "job_asset")
        return rows

    def audited(self):
        audit = M.Audit(self.repo, self.out)
        audit.sources_and_jobs()
        return audit

    def summary(self):
        audit = self.audited()
        directory = self.base / "summary"
        for name in M.SUMMARY_ASSETS - {"input_registry.csv"}:
            put(directory / name)
        table(directory / "input_registry.csv", [{"role": role, "path": str(path), "sha256": digest}
                                                 for path, (role, digest) in audit.summary_inputs.items()])
        js(directory / "summary_complete.json", {
            "status": "complete", "analysis": "frozen_grouped_interpretability", "seeds": list(M.SEEDS), "models": list(M.MODELS),
            "n_images": 20, "n_groups": 20, "n_image_seed_model_rows": 200,
            "audited_inputs": {"source_registry_sha256": self.registry_hash, "qualitative_cases_sha256": M.sha(self.base / "qualitative_cases.csv"), "audited_jobs": 50},
            "bootstrap": {"replicates": 5000, "seed": 20261001},
            "assets": {name: M.sha(directory / name) for name in M.SUMMARY_ASSETS},
        })

    def figures(self):
        directory = self.base / "figures"
        filenames = {f"qualitative_seed_{seed}_{slug}.{ext}" for seed in M.SEEDS for slug in M.SLUGS for ext in ("png", "pdf", "svg")}
        filenames |= {f"qualitative_seed_{seed}_all_classes.pdf" for seed in M.SEEDS} | {"captions_and_methods.txt", "source_registry.csv"}
        for name in filenames:
            put(directory / name)
        sources = [("frozen_case_manifest", self.base / "qualitative_cases.csv", self.out),
                   ("renderer", self.repo / M.EXPERIMENT / "render_interpretability.py", self.repo)]
        sources.extend(("case_array", self.base / "jobs" / f"seed_{seed}" / f"outer_{case['fold']}" / model / "cases" / f"{Path(case['image']).stem}.npz", self.out)
                       for seed in M.SEEDS for case in self.cases for model in M.MODELS)
        table(directory / "source_registry.csv", [{"kind": kind, "path": str(path.relative_to(root)), "sha256": M.sha(path), "size_bytes": path.stat().st_size}
                                                  for kind, path, root in sources])
        js(directory / "figures_complete.json", {
            "status": "complete", "case_count": 20, "seed_count": 5, "model_count": 2, "plate_count": 20, "source_npz_count": 200,
            "dpi": 300, "cam_target": "archived_oof_predicted_class",
            "assets": [{"path": str((directory / name).relative_to(self.out)), "sha256": M.sha(directory / name), "size_bytes": (directory / name).stat().st_size} for name in filenames],
        })

    def dispatch(self, argv, **_kwargs):
        name = Path(argv[1]).name
        if name == "summarize_interpretability.py":
            self.summary()
        elif name == "render_interpretability.py":
            self.figures()
        else:
            self.fail(f"Unexpected subprocess: {argv}")

    def test_completed_stages_are_verified_and_idempotent(self):
        self.summary()
        self.figures()
        with patch.object(M.subprocess, "run") as run:
            result = M.finalize(self.repo, self.out)
            final = self.base / "interpretability_complete.json"
            digest = M.sha(final)
            again = M.finalize(self.repo, self.out)
            run.assert_not_called()
        self.assertEqual(result, again)
        self.assertEqual(digest, M.sha(final))
        self.assertIn("figures_complete_sha256", result)
        self.assertEqual(result["recovery_script_sha256"], M.sha(SCRIPT))

    def test_existing_summary_render_recovery_preserves_partial(self):
        self.summary()
        put(self.base / "figures/partial.txt", "retain me")
        with patch.object(M.subprocess, "run", side_effect=self.dispatch) as run:
            M.finalize(self.repo, self.out)
            self.assertEqual([Path(call.args[0][1]).name for call in run.call_args_list], ["render_interpretability.py"])
        archives = list(self.base.glob("figures.incomplete_archive_*"))
        self.assertEqual(len(archives), 1)
        self.assertEqual((archives[0] / "partial.txt").read_text(), "retain me")

    def test_both_partial_stages_preserved_and_rebuilt(self):
        put(self.base / "summary/partial.txt", "summary bytes")
        put(self.base / "figures/partial.txt", "figure bytes")
        with patch.object(M.subprocess, "run", side_effect=self.dispatch) as run:
            M.finalize(self.repo, self.out)
            self.assertEqual(run.call_count, 2)
        self.assertEqual(len(list(self.base.glob("summary.incomplete_archive_*"))), 1)
        self.assertEqual(len(list(self.base.glob("figures.incomplete_archive_*"))), 1)

    def test_corrupt_completed_summary_is_not_overwritten(self):
        self.summary()
        put(self.base / "summary/model_summary.csv", "corrupt")
        with patch.object(M.subprocess, "run") as run, self.assertRaisesRegex(ValueError, "hash mismatch"):
            M.finalize(self.repo, self.out)
        run.assert_not_called()
        self.assertFalse(list(self.base.glob("summary.incomplete_archive_*")))

    def test_missing_job_stops_before_stage_mutation(self):
        path = self.base / f"jobs/seed_{M.SEEDS[-1]}/outer_4/lpt_full/complete.json"
        path.rename(path.with_suffix(".held"))
        put(self.base / "summary/partial.txt")
        with patch.object(M.subprocess, "run") as run, self.assertRaises(FileNotFoundError):
            M.finalize(self.repo, self.out)
        run.assert_not_called()
        self.assertTrue((self.base / "summary/partial.txt").exists())

    def test_changed_frozen_code_fails_closed(self):
        put(self.repo / M.EXPERIMENT / "run_interpretability.py", "changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            M.finalize(self.repo, self.out)

    def test_wrong_summary_input_registry_fails_even_with_updated_output_hash(self):
        self.summary()
        directory = self.base / "summary"
        rows = M.read_csv(directory / "input_registry.csv")
        table(directory / "input_registry.csv", rows[:-1])
        marker = M.read_json(directory / "summary_complete.json")
        marker["assets"]["input_registry.csv"] = M.sha(directory / "input_registry.csv")
        js(directory / "summary_complete.json", marker)
        with self.assertRaisesRegex(ValueError, "input registry differs"):
            M.finalize(self.repo, self.out)

    def test_legacy_total_marker_preserved_and_enriched(self):
        self.summary()
        self.figures()
        old = {"status": "complete", "jobs": 50, "images_per_model_per_seed": 20,
               "source_registry_sha256": self.registry_hash,
               "summary_complete_sha256": M.sha(self.base / "summary/summary_complete.json")}
        js(self.base / "interpretability_complete.json", old)
        M.finalize(self.repo, self.out)
        archive = list(self.base.glob("interpretability_complete.json.legacy_archive_*"))
        self.assertEqual(len(archive), 1)
        self.assertEqual(M.read_json(archive[0]), old)

    def test_verify_only_never_writes_completion(self):
        self.summary()
        self.figures()
        result = M.finalize(self.repo, self.out, True)
        self.assertEqual(result["status"], "verified_only")
        self.assertFalse((self.base / "interpretability_complete.json").exists())

    def test_corrupt_figure_output_fails_without_archive(self):
        self.summary()
        self.figures()
        put(self.base / f"figures/qualitative_seed_{M.SEEDS[0]}_neither.png", "corrupt")
        with self.assertRaisesRegex(ValueError, "Figure asset differs"):
            M.finalize(self.repo, self.out)
        self.assertFalse(list(self.base.glob("figures.incomplete_archive_*")))


if __name__ == "__main__":
    unittest.main()
