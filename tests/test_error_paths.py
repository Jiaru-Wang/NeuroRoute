"""6.2/6.4：合成异常路径，随 GPU 测试入口执行；不读真实 EEG。"""

import unittest
from unittest.mock import patch

import torch
from test_gpu import ProgrammedModel, ToyData
from torch.utils.data import DataLoader

from neuroroute.engine import evaluate
from neuroroute.models import cuda_device


class EvaluationErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = cuda_device()

    def setUp(self):
        self.model = ProgrammedModel().to(self.device).train()
        self.stats = {"mean": [0.0] * 22, "std": [1.0] * 22}
        self.conditions = {"complete": []}

    def check(self, data):
        return evaluate(self.model, data, self.device, self.stats, self.conditions, 3)

    def test_dropped_tail(self):
        with patch(
            "neuroroute.engine.loader",
            side_effect=lambda d, b, s, t: DataLoader(d, batch_size=b, drop_last=True),
        ):
            with self.assertRaises(ValueError):
                self.check(ToyData())
        self.assertTrue(self.model.training)

    def test_duplicate_identity(self):
        class Duplicate(ToyData):
            def __getitem__(self, i):
                return super().__getitem__(0 if i == 1 else i)

        with self.assertRaises(ValueError):
            self.check(Duplicate())

    def test_wrong_partition(self):
        class Wrong(ToyData):
            def __getitem__(self, i):
                row = super().__getitem__(i)
                row["subject_id"] = 9
                return row

        with self.assertRaises(ValueError):
            self.check(Wrong())

    def test_invalid_input(self):
        class Invalid(ToyData):
            def __getitem__(self, i):
                row = super().__getitem__(i)
                row["x"][0, 0, 0] = float("nan")
                return row

        with self.assertRaises(ValueError):
            self.check(Invalid())

    def test_invalid_output_and_mode_restoration(self):
        class Invalid(ProgrammedModel):
            def forward(self, x, m):
                return torch.full((len(x), 4), float("nan"), device=x.device), {}

        self.model = Invalid().to(self.device).train()
        with self.assertRaises(ValueError):
            self.check(ToyData())
        self.assertTrue(self.model.training)
        self.assertIsNone(self.model.dummy.grad)

    def test_outer_test_not_implicitly_allowed(self):
        data = ToyData()
        data.role = "test"
        with self.assertRaises(ValueError):
            self.check(data)
