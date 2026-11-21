"""6.2：人工事件/标签序列检查；不需要真实GDF，不产生实验数据。"""

import unittest

import numpy as np

from neuroroute.data import audit_trials


class EventAlignmentTests(unittest.TestCase):
    def events(self, kind="T"):
        rows = []
        for i in range(4):
            rows.extend([[i * 2000, 0, 768], [i * 2000 + 500, 0, 769 + i if kind == "T" else 783]])
        rows.append([2000, 0, 1023])
        return np.array(sorted(rows, key=lambda r: r[0]), dtype=np.int64)

    def test_full_alignment_before_artifact_filter(self):
        records = audit_trials(self.events(), np.arange(4), "T", 8000, "synthetic")
        self.assertEqual([r["trial_id"] for r in records], [1, 2, 3, 4])
        self.assertEqual([r["label"] for r in records], [0, 1, 2, 3])
        self.assertEqual([r["artifact"] for r in records], [False, True, False, False])
        self.assertEqual([r["trial_id"] for r in records if not r["artifact"]], [1, 3, 4])

    def test_evaluation_unknown_cue_uses_official_labels(self):
        records = audit_trials(self.events("E"), np.array([3, 1, 0, 2]), "E", 8000, "synthetic")
        self.assertEqual([r["label"] for r in records], [3, 1, 0, 2])

    def test_label_shift_rejected(self):
        with self.assertRaises(ValueError):
            audit_trials(self.events(), np.array([1, 2, 3, 0]), "T", 8000, "synthetic")

    def test_missing_duplicate_and_mistimed_cue(self):
        events = self.events()
        cases = [events[events[:, 2] != 769], np.vstack([events, [500, 0, 769]])]
        shifted = events.copy()
        shifted[shifted[:, 2] == 769, 0] += 1
        cases.append(shifted)
        for value in cases:
            with self.subTest(value=value.tolist()):
                with self.assertRaises(ValueError):
                    audit_trials(value, np.arange(4), "T", 8000, "synthetic")

    def test_run_boundary_and_truncated_trial(self):
        with self.assertRaises(ValueError):
            audit_trials(
                np.vstack([self.events(), [1400, 0, 32766]]), np.arange(4), "T", 8000, "synthetic"
            )
        with self.assertRaises(ValueError):
            audit_trials(self.events(), np.arange(4), "T", 7400, "synthetic")

    def test_trial_label_count(self):
        with self.assertRaises(ValueError):
            audit_trials(self.events(), np.arange(3), "T", 8000, "synthetic")
