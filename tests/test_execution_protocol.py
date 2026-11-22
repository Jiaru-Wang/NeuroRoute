"""Approved plans and statistical behavior, synthetic CPU fixtures only."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuroroute.analysis import paired_subject_statistics
from neuroroute.contracts import read_json, validate_config, write_json
from neuroroute.planning import generate_ablations, plan_runs

ROOT = Path(__file__).resolve().parents[1]


class ExecutionProtocolTests(unittest.TestCase):
    def configs(self):
        config = read_json(ROOT / "configs/protocol.review.json")
        config.update(approved=True, approval_note="synthetic approval fixture")
        plan = read_json(ROOT / "configs/ablations.review.json")
        plan.update(approved=True, approval_note="synthetic ablation fixture")
        return config, plan

    def test_budget_reuses_base_and_excludes_ablation_mechanisms(self):
        config, plan = self.configs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main = plan_runs(config, root / "main.json")
            jobs = generate_ablations(config, plan, root / "ablations")
            self.assertEqual(len(main), 216)
            self.assertEqual(len(jobs), 9)
            self.assertEqual(len(main) + len(jobs) * 27, 459)
            for job in jobs:
                candidate = read_json(root / "ablations" / job["config"])
                validate_config(candidate)
                self.assertFalse(candidate["final_test"]["mechanisms"]["approved"])
                axis = candidate["ablation"]["axis"]
                if axis == "random_region_seed":
                    self.assertEqual([len(v) for v in candidate["regions"].values()], [6, 6, 4, 6])
                else:
                    section = "training" if axis == "balance_lambda" else "model"
                    self.assertNotEqual(candidate[section][axis], config[section][axis])

    def test_six_comparison_holm_seed_means_bootstrap_and_effect_direction(self):
        plan = read_json(ROOT / "configs/statistics.review.json")
        plan.update(approved=True, approval_note="synthetic statistics fixture")
        names = sorted({name for pair in plan["comparisons"] for name in pair})
        rank = {name: i + 1 for i, name in enumerate(names)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name in names:
                for fold in range(1, 10):
                    for seed, offset in zip((42, 123, 3407), (-0.02, 0, 0.02)):
                        path = root / f"{name}-{fold}-{seed}.json"
                        write_json(
                            path,
                            dict(
                                status="final_evaluation_completed",
                                variant=name,
                                fold_id=fold,
                                seed=seed,
                                cache_manifest_sha256="fixture",
                                backbone_sha256="fixture",
                                config_sha256="a" * 64,
                                source_commit="fixture",
                                conditions={
                                    "C3_C4": {"macro_f1": 0.3 + rank[name] * fold * 0.001 + offset}
                                },
                            ),
                        )
                        paths.append(path)
            report = paired_subject_statistics(paths, plan, root / "report.json")
            for result in report["comparisons"]:
                delta = (rank[result["candidate"]] - rank[result["reference"]]) * 0.001
                difference = np.arange(1, 10) * delta
                self.assertEqual(result["subjects"], 9)
                self.assertAlmostEqual(result["mean_difference"], difference.mean())
                self.assertAlmostEqual(
                    result["mean_difference_percentage_points"], difference.mean() * 100
                )
                self.assertAlmostEqual(
                    result["paired_cohen_dz"], difference.mean() / difference.std(ddof=1)
                )
                self.assertEqual(result["p_exact"], 2 / 512)
                self.assertEqual(result["p_holm"], 12 / 512)
                rng = np.random.default_rng(42)
                boot = difference[rng.integers(0, 9, (10000, 9))].mean(1)
                np.testing.assert_allclose(
                    result["percentile_bootstrap_ci"], np.quantile(boot, [0.025, 0.975])
                )
