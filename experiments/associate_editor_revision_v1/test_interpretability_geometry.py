"""Synthetic-only regression tests; runnable with unittest or pytest."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

try:
    from .interpretability_geometry import (
        mask_bbox, mask_status, select_cases, translate_mask, translations,
    )
except ImportError:
    from interpretability_geometry import (
        mask_bbox, mask_status, select_cases, translate_mask, translations,
    )


class TranslationTests(unittest.TestCase):
    def test_area_shape_and_no_wrap(self):
        mask = np.zeros((9, 12), dtype=np.uint8)
        mask[2:5, 3] = 1
        mask[4, 3:7] = 1
        positions = translations(mask, 10000, 20261002, "synthetic")
        self.assertEqual(len(positions), (9 - 3 + 1) * (12 - 4 + 1) - 1)
        self.assertNotIn((0, 0), positions)
        self.assertEqual(len(positions), len(set(positions)))
        original = set(map(tuple, np.argwhere(mask)))
        for dy, dx in positions:
            translated = translate_mask(mask, dy, dx)
            self.assertEqual(translated.shape, mask.shape)
            self.assertEqual(translated.dtype, mask.dtype)
            self.assertEqual(int(translated.sum()), int(mask.sum()))
            self.assertEqual(
                set(map(tuple, np.argwhere(translated))),
                {(y + dy, x + dx) for y, x in original},
            )
        with self.assertRaises(ValueError):
            translate_mask(mask, -3, 0)
        with self.assertRaises(ValueError):
            translate_mask(mask, 0, 6)

    def test_empty_full_and_bbox_full_sparse(self):
        empty = np.zeros((5, 7), dtype=bool)
        self.assertIsNone(mask_bbox(empty))
        self.assertEqual(mask_status(empty), "empty")
        self.assertEqual(translations(empty), [])
        full = np.ones((5, 7), dtype=bool)
        self.assertEqual(mask_status(full), "immovable")
        self.assertEqual(translations(full), [])
        sparse = empty.copy()
        sparse[0, 0] = sparse[-1, -1] = True
        self.assertEqual(mask_status(sparse), "immovable")
        self.assertEqual(translations(sparse), [])

    def test_full_width_or_height_is_movable(self):
        width = np.zeros((5, 7), dtype=bool)
        width[2:4, :] = True
        self.assertEqual(mask_status(width), "movable")
        self.assertEqual(translations(width), [(-2, 0), (-1, 0), (1, 0)])
        height = np.zeros((5, 7), dtype=bool)
        height[:, 2:6] = True
        self.assertEqual(mask_status(height), "movable")
        self.assertEqual(translations(height), [(0, -2), (0, -1), (0, 1)])

    def test_reproducible_per_image_sampling(self):
        mask = np.zeros((15, 16), dtype=bool)
        mask[3:5, 5:8] = True
        first = translations(mask, 30, 20261002, "image-A")
        translations(mask, 30, 20261002, "image-B")
        self.assertEqual(first, translations(mask, 30, 20261002, "image-A"))
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(first), 30)
        self.assertNotEqual(first, translations(mask, 30, 20261002, "image-B"))
        self.assertNotEqual(first, translations(mask, 30, 20261003, "image-A"))

    def test_invalid_masks_and_parameters(self):
        for mask in [np.array([0, 1]), np.zeros((0, 2)), np.full((2, 2), .5)]:
            with self.assertRaises(ValueError):
                translations(mask)
        with self.assertRaises(ValueError):
            translations(np.ones((2, 2)), max_placements=0)
        with self.assertRaises(ValueError):
            translate_mask(np.ones((2, 2)), .5, 0)


class SelectionTests(unittest.TestCase):
    @staticmethod
    def manifest() -> pd.DataFrame:
        rows = []
        for label in range(4):
            for group in range(8):
                for image in range(2):
                    rows.append({
                        "image": f"image_{label}_{group}_{image}",
                        "label": label,
                        "similarity_group": f"group_{label}_{group}",
                        "fold": group % 5,
                        "prob_0": float(label + image),
                        "correct": False,
                    })
        # One group shared across labels must never be reused in selection.
        rows.append({"image": "mixed_0", "label": 0, "similarity_group": "mixed",
                     "fold": 4, "prob_0": .9, "correct": True})
        rows.append({"image": "mixed_1", "label": 1, "similarity_group": "mixed",
                     "fold": 4, "prob_0": .1, "correct": False})
        return pd.DataFrame(rows)

    def test_counts_global_groups_and_input_order(self):
        frame = self.manifest()
        chosen = select_cases(frame)
        self.assertEqual(chosen.columns.tolist(), [
            "case_rank", "image", "label", "similarity_group", "fold",
        ])
        self.assertEqual(chosen.case_rank.tolist(), list(range(1, 21)))
        self.assertEqual(chosen.groupby("label").size().tolist(), [5, 5, 5, 5])
        self.assertEqual(chosen.similarity_group.nunique(), 20)
        pd.testing.assert_frame_equal(chosen, select_cases(frame.sample(frac=1, random_state=4)))

    def test_predictions_cannot_change_selection(self):
        frame = self.manifest()
        original = select_cases(frame)
        frame["prob_0"] = np.arange(len(frame))
        frame["correct"] = ~frame["correct"]
        frame["macro_f1"] = 1.
        pd.testing.assert_frame_equal(original, select_cases(frame))
        pd.testing.assert_frame_equal(original, select_cases(frame.drop(
            columns=["prob_0", "correct", "macro_f1"])))

    def test_mixed_groups_feasible_and_impossible(self):
        # Flexible class 0 cannot consume class 1's only group in the final set.
        frame = pd.DataFrame([
            ["a", 0, "shared", 0], ["b", 1, "shared", 0],
            ["c", 0, "only_zero", 1],
        ], columns=["image", "label", "similarity_group", "fold"])
        for seed in range(10):
            chosen = select_cases(frame, seed=seed, per_class=1)
            self.assertEqual(chosen.similarity_group.tolist(), ["only_zero", "shared"])
        with self.assertRaises(ValueError):
            select_cases(frame.iloc[:2], per_class=1)

    def test_manifest_validation(self):
        frame = self.manifest()
        with self.assertRaises(ValueError):
            select_cases(pd.concat([frame, frame.iloc[:1]]))
        with self.assertRaises(ValueError):
            select_cases(frame.drop(columns="fold"))
        with self.assertRaises(ValueError):
            select_cases(frame, per_class=20)
        broken = frame.copy()
        broken.loc[1, "fold"] = 4
        with self.assertRaises(ValueError):
            select_cases(broken)


if __name__ == "__main__":
    unittest.main()
