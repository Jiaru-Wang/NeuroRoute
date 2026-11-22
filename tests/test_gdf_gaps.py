"""Synthetic GDF encoding checks; no real EEG or model execution."""

import unittest

import numpy as np

from neuroroute.data import continuous_fir, decode_gdf_gaps


class GdfGapTests(unittest.TestCase):
    def fixture(self):
        codes = np.zeros((22, 10000))
        codes[:, 5000:5100] = -32768
        codes[0, 100] = -2048
        header = {
            "units": np.full(22, 1e-6),
            "cal": np.full(22, 0.048828125),
            "offsets": np.zeros(22),
            "digital_min": np.full(22, -2047),
            "digital_max": np.full(22, 2047),
        }
        trials = [{"trial_id": 1, "start_sample": 5200, "cue_sample": 5700}]
        return codes * 0.048828125, header, np.arange(22), [0, 5000], trials

    def test_gap_preserves_positions_and_endpoint(self):
        args = self.fixture()
        before = args[0].copy()
        x, audit = decode_gdf_gaps(*args)
        self.assertEqual(x.shape, before.shape)
        self.assertTrue(np.isnan(x[:, 5000:5100]).all())
        np.testing.assert_array_equal(x[:, :5000], before[:, :5000])
        np.testing.assert_array_equal(x[:, 5100:], before[:, 5100:])
        np.testing.assert_array_equal(args[0], before)
        self.assertEqual(x[0, 100], -100.0)
        self.assertEqual(audit["gap_time_samples"], 100)
        self.assertEqual(audit["lower_endpoint_channel_samples"], 1)

    def test_unknown_codes_fail(self):
        for code in (-2049, 2048, -30000):
            args = self.fixture()
            args[0][0, 200] = code * 0.048828125
            with self.subTest(code=code), self.assertRaises(ValueError):
                decode_gdf_gaps(*args)

    def test_decoded_gap_isolates_fir_and_keeps_trial_window(self):
        args = self.fixture()
        signal, _ = decode_gdf_gaps(*args)
        filtered, audit = continuous_fir(signal, args[3], 0.3, 40)
        self.assertEqual(audit["segments"], [[0, 5000], [5100, 10000]])
        self.assertTrue(np.isfinite(filtered[:, 5700:6700]).all())
        self.assertEqual(filtered[:, 5700:6700].shape, (22, 1000))
        args[0][:, 5100:] = 1000 * 0.048828125
        changed, _ = decode_gdf_gaps(*args)
        changed, _ = continuous_fir(changed, args[3], 0.3, 40)
        np.testing.assert_array_equal(filtered[:, :5000], changed[:, :5000])
        self.assertTrue(np.isnan(changed[:, 5000:5100]).all())

    def test_misaligned_partial_or_wrong_length_gap_fails(self):
        for index in ((0, 5000), (0, 5100), (0, 4000)):
            args = self.fixture()
            args[0][index] = 0 if index[1] == 5000 else -1600
            with self.subTest(index=index), self.assertRaises(ValueError):
                decode_gdf_gaps(*args)

    def test_gap_overlapping_trial_fails(self):
        args = self.fixture()
        args[-1][0].update(start_sample=4900, cue_sample=5400)
        with self.assertRaisesRegex(ValueError, "Trial 1"):
            decode_gdf_gaps(*args)

    def test_calibration_and_noninteger_decode_fail(self):
        args = self.fixture()
        args[1]["digital_min"][0] = -4095
        with self.assertRaises(ValueError):
            decode_gdf_gaps(*args)
        args = self.fixture()
        args[0][0, 200] = 0.01
        with self.assertRaises(ValueError):
            decode_gdf_gaps(*args)
