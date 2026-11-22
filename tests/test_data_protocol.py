"""CPU synthetic data tests; no neural computation or real EEG."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuroroute.contracts import CHANNELS, canonical_hash, digest, write_json
from neuroroute.data import TrialDataset, fit_statistics


class ApprovedDataProtocolTests(unittest.TestCase):
    def test_all_folds_identity_channel_statistics_and_heldout_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preprocessing = {"artifact_policy": "drop"}
            records = []
            channel = np.arange(22, dtype=np.float32)[None, :, None, None]
            for sid in range(1, 10):
                for kind in "TE":
                    session = f"A{sid:02d}{kind}"
                    path = root / f"{session}.npz"
                    x = np.broadcast_to(sid * (channel + 1) + channel * 10, (4, 22, 4, 200)).copy()
                    x[:, 0] = 7  # Exercise the standard-deviation floor.
                    np.savez_compressed(
                        path,
                        x=x,
                        y=np.arange(4, dtype=np.int64),
                        trial_ids=np.array([1, 4, 7, 288], dtype=np.int64),
                        artifacts=np.zeros(4, dtype=bool),
                    )
                    records.append(
                        dict(
                            session=session,
                            subject_id=sid,
                            file=path.name,
                            sha256=digest(path),
                            count=4,
                            trial_ids=[1, 4, 7, 288],
                            class_counts=[1] * 4,
                        )
                    )
            write_json(
                root / "manifest.json",
                dict(
                    schema="neuroroute_cache_v1",
                    channel_order=CHANNELS,
                    preprocessing_sha256=canonical_hash(preprocessing),
                    sessions=records,
                ),
            )
            for fold in range(1, 10):
                with self.subTest(fold=fold):
                    validation = 9 if fold == 1 else fold - 1
                    subjects = sorted(set(range(1, 10)) - {fold, validation})
                    sets = {
                        role: TrialDataset(root, fold, role, preprocessing)
                        for role in ("train", "validation", "test")
                    }
                    self.assertEqual(sets["train"].subject_ids, subjects)
                    self.assertEqual(sets["validation"].subject_ids, [validation])
                    self.assertEqual(sets["test"].subject_ids, [fold])
                    for role, dataset in sets.items():
                        self.assertEqual(
                            {r["session"] for r in dataset.rows},
                            {f"A{s:02d}{k}" for s in dataset.subject_ids for k in "TE"},
                        )
                        self.assertEqual({r["trial_id"] for r in dataset.rows}, {1, 4, 7, 288})
                        if role != "train":
                            with self.assertRaises(ValueError):
                                fit_statistics(dataset, "train_channel_zscore")
                    a, b, c = [sets[r].keys for r in ("train", "validation", "test")]
                    self.assertFalse(a & b or a & c or b & c)
                    self.assertEqual(len(a | b | c), 72)
                    stats = fit_statistics(sets["train"], "train_channel_zscore")
                    expected_mean = np.mean(subjects) * (np.arange(22) + 1) + np.arange(22) * 10
                    expected_std = np.std(subjects, ddof=0) * (np.arange(22) + 1)
                    expected_mean[0], expected_std[0] = 7, 1e-6
                    np.testing.assert_allclose(stats["mean"], expected_mean)
                    np.testing.assert_allclose(stats["std"], expected_std)
                    saved = {}
                    for sid in (validation, fold):
                        for kind in "TE":
                            path = root / f"A{sid:02d}{kind}.npz"
                            saved[path] = path.read_bytes()
                            path.write_bytes(b"held-out EEG must not be opened by training")
                    try:
                        self.assertEqual(
                            stats,
                            fit_statistics(
                                TrialDataset(root, fold, "train", preprocessing),
                                "train_channel_zscore",
                            ),
                        )
                        for role in ("validation", "test"):
                            with self.assertRaises(ValueError):
                                TrialDataset(root, fold, role, preprocessing)
                    finally:
                        for path, content in saved.items():
                            path.write_bytes(content)
