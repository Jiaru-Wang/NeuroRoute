"""CPU signal-processing tests, not neural model or real-data acceptance."""

import copy
import json
import unittest
from pathlib import Path

import mne
import numpy as np

from neuroroute.contracts import validate_config
from neuroroute.data import check_fir_physical_range, continuous_fir


class ContinuousFirTests(unittest.TestCase):
    def test_finite_overflow_is_rejected_not_clipped(self):
        x = np.array([[-100.0, 0.0, 100.0, np.nan]])
        check_fir_physical_range(x, [-100.0], [100.0])
        x[0, 1] = -1600.0
        with self.assertRaisesRegex(ValueError, "sample=1"):
            check_fir_physical_range(x, [-100.0], [100.0])
        self.assertEqual(x[0, 1], -1600.0)

    def signal(self):
        t = np.arange(10000) / 250
        return np.array([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 60 * t)])

    def test_frequency_response_and_zero_phase(self):
        x = self.signal()
        before = x.copy()
        y, audit = continuous_fir(x, [0], 0.3, 40.0)
        middle = slice(3000, -3000)
        np.testing.assert_allclose(y[0, middle], x[0, middle], atol=0.01)
        self.assertLess(np.max(np.abs(y[1, middle])), 0.01)
        np.testing.assert_array_equal(x, before)
        self.assertEqual(audit["segments"], [[0, 10000]])
        json.dumps(audit, allow_nan=False)

    def test_run_boundary_and_channel_isolation(self):
        x = np.concatenate([self.signal(), self.signal()], axis=1)
        y, _ = continuous_fir(x, [0, 10000], 0.3, 40.0)
        changed = x.copy()
        changed[:, 10000:] *= 100
        other, _ = continuous_fir(changed, [0, 10000], 0.3, 40.0)
        np.testing.assert_array_equal(y[:, :10000], other[:, :10000])
        changed = x.copy()
        changed[1] *= 100
        other, _ = continuous_fir(changed, [0, 10000], 0.3, 40.0)
        np.testing.assert_array_equal(y[0], other[0])

    def test_missing_gap_is_not_filtered_across(self):
        x = np.concatenate([self.signal(), np.full((2, 100), np.nan), self.signal()], axis=1)
        y, audit = continuous_fir(x, [0], 0.3, 40.0)
        self.assertEqual(audit["segments"], [[0, 10000], [10100, 20100]])
        self.assertEqual(audit["nonfinite_time_samples"], 100)
        self.assertTrue(np.isnan(y[:, 10000:10100]).all())
        reference = mne.filter.filter_data(
            x[:, :10000], 250, 0.3, 40, method="fir", phase="zero", verbose="ERROR"
        )
        np.testing.assert_allclose(y[:, :10000], reference, atol=1e-12)

    def test_short_segments_and_invalid_boundaries_fail(self):
        for x, boundaries in [
            (self.signal()[:, :100], [0]),
            (self.signal(), [-1]),
            (self.signal(), [10001]),
            (self.signal(), [0.5]),
        ]:
            with self.subTest(boundaries=boundaries, shape=x.shape):
                with self.assertRaises(ValueError):
                    continuous_fir(x, boundaries, 0.3, 40.0)

    def test_review_template_remains_unapproved(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs/protocol.review.json").read_text())
        with self.assertRaisesRegex(ValueError, "not approved"):
            validate_config(config)
        # Synthetic approval tests validation only, never authorizes experiments.
        config = copy.deepcopy(config)
        config.update(approved=True, approval_note="synthetic test only")
        p = config["preprocessing"]
        p.update(
            filter="continuous_fir_zero",
            filter_order=None,
            gdf_eeg_names=[f"EEG{i}" for i in range(22)],
            gdf_eog_names=[f"EOG{i}" for i in range(3)],
        )
        validate_config(config)
        p["filter_order"] = 5
        with self.assertRaises(ValueError):
            validate_config(config)
