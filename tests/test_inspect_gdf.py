"""CPU metadata regression tests; no neural model execution or real EEG required."""

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import numpy as np

from neuroroute.cli import main
from neuroroute.data import audit_trials


class MetadataJsonTests(unittest.TestCase):
    def test_inspect_cli_preserves_numeric_metadata(self):
        raw = MagicMock()
        raw.__enter__.return_value = raw
        raw.ch_names = ["EEG-C3", "EEG-C4"]
        raw.get_channel_types.return_value = ["eeg", "eeg"]
        raw.info = {"sfreq": np.float64(250)}
        raw.n_times = np.int64(12345)
        raw.annotations.description = np.array(["769", "768", "769"])
        output = io.StringIO()
        with (
            patch("mne.io.read_raw_gdf", return_value=raw),
            patch("sys.argv", ["neuroroute", "inspect-gdf", "synthetic.gdf"]),
            redirect_stdout(output),
        ):
            main()
        result = json.loads(output.getvalue())
        self.assertEqual(
            result,
            {
                "channels": ["EEG-C3", "EEG-C4"],
                "types": ["eeg", "eeg"],
                "sfreq": 250.0,
                "samples": 12345,
                "annotation_descriptions": ["768", "769"],
            },
        )
        self.assertIs(type(result["samples"]), int)
        self.assertIs(type(result["sfreq"]), float)

    def test_last_trial_boundary_serializes_as_number(self):
        events = np.array([[0, 0, 768], [500, 0, 769]], dtype=np.int64)
        rows = audit_trials(events, np.array([0]), "T", np.int64(1500), "synthetic")
        decoded = json.loads(json.dumps(rows, allow_nan=False))
        self.assertEqual(decoded[0]["end_sample"], 1500)
        self.assertIs(type(decoded[0]["end_sample"]), int)
