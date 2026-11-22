"""Data approval authorizes cache creation, never model training."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from neuroroute.cli import main
from neuroroute.contracts import validate_config, validate_prepare_config


class PrepareApprovalTests(unittest.TestCase):
    def config(self):
        root = Path(__file__).resolve().parents[1]
        full = json.loads((root / "configs/protocol.review.json").read_text())
        return dict(
            schema="neuroroute_data_v1",
            approval_scope="data_preparation",
            approved=True,
            approval_note="Explicit owner data approval",
            channel_order=full["channel_order"],
            preprocessing=full["preprocessing"],
        )

    def test_data_approval_does_not_authorize_full_protocol(self):
        config = self.config()
        self.assertEqual(validate_prepare_config(config), config)
        with self.assertRaises(ValueError):
            validate_config(config)
        for field, value in [
            ("approved", False),
            ("approval_note", ""),
            ("approval_scope", "training"),
        ]:
            bad = copy.deepcopy(config)
            bad[field] = value
            with self.assertRaises(ValueError):
                validate_prepare_config(bad)

    def test_invalid_preprocessing_and_unapproved_full_config_rejected(self):
        for field, value in [
            ("gdf_eeg_names", None),
            ("reference", "CAR"),
            ("uv_divisor", 0),
            ("resample", "unknown"),
        ]:
            bad = self.config()
            bad["preprocessing"][field] = value
            with self.assertRaises(ValueError):
                validate_prepare_config(bad)
        full = json.loads(
            (Path(__file__).resolve().parents[1] / "configs/protocol.review.json").read_text()
        )
        with self.assertRaises(ValueError):
            validate_prepare_config(full)

    def test_prepare_cli_accepts_data_scope_and_blocks_unapproved_before_io(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            config = self.config()
            path.write_text(json.dumps(config))
            argv = [
                "neuroroute",
                "prepare",
                "--config",
                str(path),
                "--raw-dir",
                "raw",
                "--labels-dir",
                "labels",
                "--output",
                str(Path(directory) / "cache"),
            ]
            with (
                patch("sys.argv", argv),
                patch("neuroroute.cli.repo_state", return_value="fixture"),
                patch("neuroroute.data.prepare", return_value={"sessions": []}) as prepare,
                redirect_stdout(io.StringIO()),
            ):
                main()
                self.assertEqual(prepare.call_args.args[3], config)
            config["approved"] = False
            path.write_text(json.dumps(config))
            with patch("sys.argv", argv), patch("neuroroute.data.prepare") as prepare:
                with self.assertRaises(ValueError):
                    main()
                prepare.assert_not_called()
