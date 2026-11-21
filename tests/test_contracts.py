"""L2 synthetic contracts; run as part of the explicitly GPU-hosted suite."""

import random
import unittest

import numpy as np

from neuroroute.contracts import CHANNELS, partition, regions_checked
from neuroroute.metrics import (
    EarlyStopping,
    classification,
    js_divergence,
    routing_summary,
    topk_switch,
)
from neuroroute.missingness import fixed_mask, sample_configured_mask, sample_mask


class ContractTests(unittest.TestCase):
    def test_pdf_four_branch_distribution_and_replay(self):
        state = random.getstate()
        config = {
            "selection": "pdf_four_branch",
            "branch_probabilities": [0.25] * 4,
            "other_count_probabilities": [0.5, 0.5],
        }
        identities = [
            dict(seed=42, fold=9, epoch=epoch, session="A01T", trial_id=trial)
            for epoch in range(1, 11)
            for trial in range(1, 289)
        ]
        masks = [sample_configured_mask(config, **key) for key in identities]
        self.assertEqual(
            masks,
            list(reversed([sample_configured_mask(config, **key) for key in reversed(identities)])),
        )
        counts = [0, 0, 0, 0]
        for mask in masks:
            missing = {CHANNELS[i] for i, v in enumerate(mask) if not v}
            if not missing:
                counts[0] += 1
            elif missing == {"C3"}:
                counts[1] += 1
            elif missing == {"C4"}:
                counts[2] += 1
            else:
                counts[3] += 1
                self.assertIn(len(missing), (1, 2))
                self.assertFalse(missing & {"C3", "C4"})
        self.assertTrue(all(0.20 < c / len(masks) < 0.30 for c in counts))
        self.assertEqual(state, random.getstate())

    def test_loso_subject_isolation(self):
        for fold in range(1, 10):
            p = partition(fold)
            self.assertEqual(list(map(len, p.values())), [7, 1, 1])
            self.assertEqual(p["test"], [fold])
            self.assertEqual(set(sum(p.values(), [])), set(range(1, 10)))
            self.assertFalse(set(p["train"]) & set(p["validation"]))
        for bad in (0, 10, True, 1.0):
            with self.assertRaises(ValueError):
                partition(bad)

    def test_mask_replay_coverage_and_global_rng(self):
        state = random.getstate()
        keys = [
            dict(seed=42, fold=9, epoch=e, session="A01T", trial_id=t)
            for e in range(1, 41)
            for t in range(1, 289)
        ]
        first = [
            sample_mask([0.5, 0.25, 0.25], **k) for k in keys
        ]  # Synthetic test distribution, NOT an approved experiment.
        self.assertEqual(
            first, list(reversed([sample_mask([0.5, 0.25, 0.25], **k) for k in reversed(keys)]))
        )
        self.assertEqual(state, random.getstate())
        patterns = set()
        counts = [0, 0, 0]
        for mask in first:
            self.assertFalse(not mask[7] and not mask[11])
            missing = tuple(i for i, v in enumerate(mask) if not v)
            patterns.add(missing)
            counts[len(missing)] += 1
        self.assertEqual(len(patterns), 253)
        self.assertTrue(0.45 < counts[0] / len(first) < 0.55)
        for missing in (["C3", "C4"], ["C3", "C4", "Cz"], CHANNELS, ["unknown"]):
            with self.assertRaises(ValueError):
                fixed_mask(missing, development=True)

    def test_regions_cannot_hide_missing_channels(self):
        # Artificial groups solely for shape checks; not physiological regions.
        groups = {str(g): CHANNELS[g::4] for g in range(4)}
        self.assertEqual(sorted(sum(regions_checked(groups), [])), list(range(22)))
        groups["0"] = []
        with self.assertRaises(ValueError):
            regions_checked(groups)

    def test_metrics_and_selection_counterexample(self):
        y = np.tile(np.arange(4), 2)
        complete = classification(y, y)["macro_f1"]
        c3 = classification(y, np.zeros(8, dtype=int))["macro_f1"]
        c4 = classification(y, (y + 1) % 4)["macro_f1"]
        self.assertAlmostEqual((complete + c3 + c4) / 3, 11 / 30)
        pooled = classification(
            np.tile(y, 3), np.concatenate((y, np.zeros(8, dtype=int), (y + 1) % 4))
        )["macro_f1"]
        self.assertAlmostEqual(pooled, 37 / 90)
        with self.assertRaises(ValueError):
            classification([0, 1], [0, 1])
        with self.assertRaises(ValueError):
            classification([0.0, 1.0, 2.0, 3.0], [0, 1, 2, 3])

    def test_earliest_best_and_stop(self):
        e = EarlyStopping(10, 0)
        self.assertEqual(e.step(0.4, 1), (True, False))
        for epoch in range(2, 11):
            self.assertEqual(e.step(0.4, epoch), (False, False))
        self.assertEqual(e.step(0.4, 11), (False, True))
        self.assertEqual(e.best_epoch, 1)
        with self.assertRaises(ValueError):
            e.step(0.5, 12)

    def test_routing_math(self):
        p = np.ones((2, 7)) / 7
        r = routing_summary(p, np.array([[0, 1], [2, 3]]))
        self.assertAlmostEqual(sum(r["importance"]), 1)
        self.assertAlmostEqual(sum(r["load"]), 2)
        self.assertAlmostEqual(r["normalized_entropy"], 1)
        np.testing.assert_allclose(js_divergence([[1.0, 0.0]], [[0.0, 1.0]]), np.log(2))
        np.testing.assert_allclose(js_divergence(p, p), 0)
        np.testing.assert_allclose(
            topk_switch([[0, 1], [0, 1], [0, 1]], [[0, 1], [1, 2], [2, 3]]), [0, 0.5, 1]
        )
