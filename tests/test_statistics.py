"""6.7：手算统计反例；仅临时合成数据，不是论文结果。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from neuroroute.analysis import align_expert_profiles, paired_subject_statistics
from neuroroute.contracts import write_json


class StatisticalTests(unittest.TestCase):
    def plan(self):
        return {
            "approved": True,
            "approval_note": "SYNTHETIC_TEST_ONLY",
            "test": "exact_paired_sign_flip",
            "metric": "macro_f1",
            "condition": "C3_C4",
            "bootstrap_replicates": 100,
            "confidence": 0.95,
            "seed": 42,
            "comparisons": [["B0", "B1"]],
        }

    def fixture(self, root):
        paths = []
        for variant in ("B0", "B1"):
            for fold in range(1, 10):
                for seed in (42, 123, 3407):
                    path = root / f"{variant}_{fold}_{seed}.json"
                    write_json(
                        path,
                        {
                            "status": "final_evaluation_completed",
                            "variant": variant,
                            "fold_id": fold,
                            "seed": seed,
                            "cache_manifest_sha256": "synthetic",
                            "backbone_sha256": "synthetic",
                            "config_sha256": "a" * 64,
                            "source_commit": "SYNTHETIC_NOT_REAL",
                            "conditions": {"C3_C4": {"macro_f1": 0.2 if variant == "B0" else 0.4}},
                        },
                    )
                    paths.append(path)
        return paths

    def test_nine_subjects_not_27_and_exact_sign_flip(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            paths = self.fixture(root)
            report = paired_subject_statistics(paths, self.plan(), root / "stats.json")
            result = report["comparisons"][0]
            self.assertEqual(result["subjects"], 9)
            self.assertEqual(len(result["candidate_subject_means"]), 9)
            self.assertAlmostEqual(result["mean_difference"], 0.2)
            self.assertEqual(result["p_exact"], 2 / 512)
            self.assertEqual(result["p_holm"], 2 / 512)

    def test_missing_seed_duplicate_and_unapproved_plan(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            paths = self.fixture(root)
            with self.assertRaises(ValueError):
                paired_subject_statistics(paths[:-1], self.plan(), root / "missing.json")
            with self.assertRaises(ValueError):
                paired_subject_statistics(paths + [paths[0]], self.plan(), root / "duplicate.json")
            plan = self.plan()
            plan["approved"] = False
            with self.assertRaises(ValueError):
                paired_subject_statistics(paths, plan, root / "unapproved.json")

    def test_permutation_alignment_and_zero_profile(self):
        result = align_expert_profiles([[1, 0], [0, 1]], [[0, 1], [1, 0]])
        self.assertEqual(result["matched_right_indices"], [1, 0])
        with self.assertRaises(ValueError):
            align_expert_profiles([[0, 1], [0, 1]], [[0, 1], [0, 1]])
