"""6.2 synthetic cache/LOSO/fitted-statistics tests; no real EEG."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuroroute.contracts import CHANNELS, canonical_hash, digest, write_json
from neuroroute.data import TrialDataset, fit_statistics


class DataTests(unittest.TestCase):
    def test_partition_coverage_and_training_only_statistics(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            preprocessing = {"artifact_policy": "drop"}
            records = []
            for sid in range(1, 10):
                for kind in "TE":
                    session = f"A{sid:02d}{kind}"
                    path = root / f"{session}.npz"
                    np.savez_compressed(
                        path,
                        x=np.full((8, 22, 4, 200), sid, dtype=np.float32),
                        y=np.tile(np.arange(4, dtype=np.int64), 2),
                        trial_ids=np.arange(1, 9, dtype=np.int64),
                        artifacts=np.zeros(8, dtype=bool),
                    )
                    records.append(
                        {
                            "session": session,
                            "subject_id": sid,
                            "file": path.name,
                            "sha256": digest(path),
                            "count": 8,
                            "trial_ids": list(range(1, 9)),
                            "class_counts": [2, 2, 2, 2],
                        }
                    )
            write_json(
                root / "manifest.json",
                {
                    "schema": "neuroroute_cache_v1",
                    "channel_order": CHANNELS,
                    "preprocessing_sha256": canonical_hash(preprocessing),
                    "sessions": records,
                },
            )
            train = TrialDataset(root, 9, "train", preprocessing)
            val = TrialDataset(root, 9, "validation", preprocessing)
            test = TrialDataset(root, 9, "test", preprocessing)
            self.assertEqual((len(train), len(val), len(test)), (112, 16, 16))
            self.assertTrue(train.keys.isdisjoint(val.keys) and train.keys.isdisjoint(test.keys))
            self.assertEqual(train.subject_ids, [1, 2, 3, 4, 5, 6, 7])
            stats = fit_statistics(train, "train_channel_zscore")
            np.testing.assert_allclose(stats["mean"], 4)
            np.testing.assert_allclose(stats["std"], 2)
            with self.assertRaises(ValueError):
                fit_statistics(val, "train_channel_zscore")
            # Training must not open validation EEG: corruption there affects only validation.
            (root / "A08T.npz").write_bytes(b"synthetic corruption")
            unchanged = fit_statistics(
                TrialDataset(root, 9, "train", preprocessing), "train_channel_zscore"
            )
            self.assertEqual(stats, unchanged)
            with self.assertRaises(ValueError):
                TrialDataset(root, 9, "validation", preprocessing)
            (root / "A01T.npz").write_bytes(b"synthetic corruption")
            with self.assertRaises(ValueError):
                TrialDataset(root, 9, "train", preprocessing)
