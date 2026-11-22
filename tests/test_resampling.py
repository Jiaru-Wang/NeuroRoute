"""CPU resampling and official-artifact identity tests, no neural execution."""

import copy
import unittest

import mne
import numpy as np

from neuroroute.data import mark_artifact_selection, resample_window


class ResamplingTests(unittest.TestCase):
    def test_matches_lecture_epochs_resample_with_padding(self):
        x = np.random.default_rng(42).normal(size=(22, 1000))
        x[:, 0] += 10
        x[:, -1] -= 20
        before = x.copy()
        epochs = mne.EpochsArray(
            x[None],
            mne.create_info(22, 250, "eeg"),
            tmin=0,
            baseline=None,
            verbose="ERROR",
        )
        reference = epochs.resample(200, npad="auto", n_jobs=1).get_data()[0]
        actual = resample_window(x, "mne_fft_auto")
        np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(x, before)
        self.assertEqual(actual.shape, (22, 800))
        patches = actual.reshape(22, 4, 200).astype(np.float32)
        self.assertEqual(patches.shape, (22, 4, 200))
        self.assertTrue(np.isfinite(patches).all())

    def test_constant_signal_channel_independence_and_trial_order(self):
        x = np.repeat(np.arange(22, dtype=float)[:, None], 1000, axis=1)
        np.testing.assert_allclose(resample_window(x, "mne_fft_auto"), x[:, :800], atol=1e-12)
        trials = [x, x + 10]
        forward = [resample_window(t, "mne_fft_auto") for t in trials]
        reverse = [resample_window(t, "mne_fft_auto") for t in reversed(trials)]
        for a, b in zip(forward, reversed(reverse)):
            np.testing.assert_array_equal(a, b)
        changed = x.copy()
        changed[0] = np.arange(1000)
        np.testing.assert_array_equal(resample_window(changed, "mne_fft_auto")[1:], forward[0][1:])

    def test_invalid_input_and_unknown_method_fail(self):
        for shape in [(22, 1001), (25, 1000), (22, 999)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                resample_window(np.zeros(shape), "mne_fft_auto")
        x = np.zeros((22, 1000))
        x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            resample_window(x, "mne_fft_auto")
        with self.assertRaises(ValueError):
            resample_window(np.zeros((22, 1000)), "unknown")

    def test_official_flags_keep_trial_identity_and_labels(self):
        rows = [
            dict(trial_id=1, label=3, cue_sample=500, artifact=True),
            dict(trial_id=2, label=1, cue_sample=2500, artifact=False),
            dict(trial_id=3, label=0, cue_sample=4500, artifact=True),
            dict(trial_id=4, label=2, cue_sample=6500, artifact=False),
        ]
        before = copy.deepcopy(rows)
        audit = mark_artifact_selection(rows, "drop")
        selected = [r for r in audit if r["selected"]]
        self.assertEqual(
            [(r["trial_id"], r["label"], r["cue_sample"]) for r in selected],
            [(2, 1, 2500), (4, 2, 6500)],
        )
        self.assertEqual(
            [r["exclusion_reason"] for r in audit], ["official_1023", None, "official_1023", None]
        )
        self.assertEqual(rows, before)
        self.assertTrue(all(r["selected"] for r in mark_artifact_selection(rows, "keep")))
        with self.assertRaises(ValueError):
            mark_artifact_selection(rows, "automatic")
