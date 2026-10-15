from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from common import LABELS, subset, validate_assignment
from summarize import confusion, crossed_bootstrap, metric_array


class DesignTests(unittest.TestCase):
    def frame(self):
        records = []
        for role in ('fit', 'stop', 'test'):
            for label in range(4):
                for g in range(30):
                    # Include multi-image and mixed-label components.
                    for k in range(2 if g % 7 == 0 else 1):
                        cls = (label + k) % 4 if g == 0 else label
                        group = f'{role}_{label}_{g}'
                        records.append(dict(image=f'{group}_{k}.jpg', label=cls, label_name=LABELS[cls],
                                            similarity_group=group, outer_role='test' if role == 'test' else 'train',
                                            direct_role=role, fold=0 if role == 'test' else 1))
        return pd.DataFrame(records)

    def test_nested_and_deterministic(self):
        f = self.frame()
        a, b, c, d = [subset(f, 20260925, 0, size) for size in (25, 50, 75, 100)]
        for x in (a, b, c, d):
            validate_assignment(x, f)
        sets = [set(x.loc[x.outer_role.eq('train'), 'image']) for x in (a, b, c, d)]
        self.assertTrue(sets[0] < sets[1] < sets[2] < sets[3])
        self.assertEqual(sets[3], set(f.loc[f.outer_role.eq('train'), 'image']))
        changed = f.copy()
        changed.loc[changed.outer_role.eq('train'), 'direct_role'] = 'old_role_ignored'
        self.assertEqual(a.direct_role.tolist(), subset(changed, 20260925, 0, 25).direct_role.tolist())
        self.assertTrue(a.equals(subset(f, 20260925, 0, 25)))
        self.assertFalse(a.equals(subset(f, 20260926, 0, 25)))

    def test_no_test_label_in_subset_choice(self):
        f = self.frame()
        a = subset(f, 20260925, 0, 50)
        changed = f.copy()
        changed.loc[changed.outer_role.eq('test'), 'label'] = 3
        b = subset(changed, 20260925, 0, 50)
        self.assertEqual(a.outer_role.tolist(), b.outer_role.tolist())
        self.assertEqual(a.direct_role.tolist(), b.direct_role.tolist())

    def test_group_split_detected(self):
        f = self.frame()
        a = subset(f, 20260925, 0, 75)
        multi = a.loc[a.direct_role.eq('fit')].groupby('similarity_group').size()
        group = multi.loc[multi.gt(1)].index[0]
        i = a.loc[a.similarity_group.eq(group)].index[0]
        a.loc[i, 'direct_role'] = 'stop'
        with self.assertRaises(AssertionError):
            validate_assignment(a, f)

    def test_metrics_match_sklearn(self):
        from sklearn.metrics import f1_score, precision_score, recall_score
        y = np.tile(np.arange(4), 15)
        predicted = np.random.default_rng(91).integers(4, size=len(y))
        m = metric_array(confusion(y, predicted))
        self.assertAlmostEqual(m[0], f1_score(y, predicted, average='macro'))
        self.assertAlmostEqual(m[1], precision_score(y, predicted, average=None)[2])
        self.assertAlmostEqual(m[2], recall_score(y, predicted, average=None)[2])
        self.assertAlmostEqual(m[3], f1_score(y, predicted, average=None)[2])

    def test_undefined_precision_is_nan(self):
        m = np.ones((4, 4))
        m[:, 2] = 0
        result = metric_array(m)
        self.assertTrue(np.isnan(result[1]))
        self.assertEqual(result[2], 0)

    def test_identical_cells_have_identical_paired_draws(self):
        one = np.zeros((5, 20, 4, 4), int)
        for seed in range(5):
            for g in range(20):
                one[seed, g, g % 4, (g + int(seed == 0)) % 4] = 1
        draws = crossed_bootstrap(np.stack([one, one]), repetitions=40)
        np.testing.assert_equal(draws[:, 0], draws[:, 1])
        self.assertEqual(draws.shape, (40, 2, 4))

    def test_seed_metrics_not_seed_pooled_confusion(self):
        matrices = np.array([np.eye(4) * 10, [[1, 30, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]])
        expected = metric_array(matrices).mean(axis=0)[0]
        wrong = metric_array(matrices.sum(axis=0))[0]
        self.assertNotAlmostEqual(expected, wrong)


if __name__ == '__main__':
    unittest.main()
