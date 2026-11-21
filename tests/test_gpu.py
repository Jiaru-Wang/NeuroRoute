"""L2/L3 GPU synthetic tests. No EEG, outer-test partition or trained results.

No skip-to-success and no CPU fallback. Real checkpoint/data are tested by smoke.
Artificial region membership in these tests is NOT a scientific region choice.
"""

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from neuroroute.checkpoints import cpu_copy, load_checkpoint, save_checkpoint
from neuroroute.contracts import CHANNELS, VARIANTS, digest
from neuroroute.engine import evaluate, train_epoch
from neuroroute.models import (
    Downstream,
    EEGModel,
    FrozenEncoder,
    cuda_device,
    importance_loss,
    linear_cost,
)

REGIONS = {str(g): CHANNELS[g::4] for g in range(4)}


class ToyData(torch.utils.data.Dataset):
    role = "validation"
    fold = 9
    subject_ids = [8]

    def __init__(self, training=False):
        self.subject = 1 if training else 8
        self.role = "train" if training else "validation"
        self.subject_ids = list(range(1, 8)) if training else [8]
        self.session = f"A{self.subject:02d}T"
        self.keys = {(self.session, i + 1) for i in range(8)}

    def __len__(self):
        return 8

    def __getitem__(self, i):
        x = torch.zeros(22, 4, 200, dtype=torch.float32)
        x[0, 0, 0] = i % 4
        return {
            "x": x,
            "y": i % 4,
            "session": self.session,
            "trial_id": i + 1,
            "subject_id": self.subject,
            "artifact": False,
        }


class ProgrammedModel(nn.Module):
    """Hand-computable teaching predictions; never an experimental model."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, m):
        y = x[:, 0, 0, 0].long()
        predicted = torch.where(
            ~m[:, 7], torch.zeros_like(y), torch.where(~m[:, 11], (y + 1) % 4, y)
        )
        return torch.nn.functional.one_hot(predicted, 4).float() * 2 + self.dummy * 0, {}


class GPUModelTests(unittest.TestCase):
    def test_hand_calculated_shared_fusion_and_intervention(self):
        model = Downstream("spatial_egs", REGIONS).to(self.device)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.classifier.weight[:, :4] = torch.eye(4, device=self.device)
            model.e0.net[2].bias.fill_(2.0)
            for i, expert in enumerate(model.experts):
                expert.net[2].bias.fill_(i + 1.0)
        h = torch.zeros(1, 200, device=self.device)
        mask = torch.ones(1, 22, dtype=torch.bool, device=self.device)
        indices = torch.tensor([[0, 2]], device=self.device)
        weights = torch.tensor([[0.25, 0.75]], device=self.device)
        output, _ = model(h, mask, routing=(indices, weights))
        torch.testing.assert_close(output, torch.full((1, 4), 2.25, device=self.device))
        removed, _ = model(h, mask, routing=(indices, weights), remove_shared=True)
        torch.testing.assert_close(removed, torch.full((1, 4), 1.25, device=self.device))
        natural, trace = model(h, mask)
        replay, _ = model(h, mask, routing=(trace["indices"], trace["weights"]))
        torch.testing.assert_close(natural, replay, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            model(h, mask, routing=(indices, torch.ones_like(weights)))

    @classmethod
    def setUpClass(cls):
        cls.device = cuda_device()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(42)

    def test_parameter_budget_and_information_slots(self):
        expected = {
            "B0": 804,
            "B1": 804,
            "matched_mlp": 442906,
            "vanilla_moe": 443116,
            "mask_moe": 443116,
            "shared_h": 442987,
            "shared_hm": 442987,
            "spatial_egs": 442987,
        }
        h = torch.randn(4, 200, device=self.device)
        full = torch.ones(4, 22, dtype=torch.bool, device=self.device)
        missing = full.clone()
        missing[:, 7] = False
        for variant in VARIANTS:
            model = Downstream(variant, REGIONS).to(self.device)
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected[variant])
            out, trace = model(h, missing)
            self.assertEqual(tuple(out.shape), (4, 4))
            self.assertTrue(bool(torch.isfinite(out).all()))
            if variant in ("vanilla_moe", "shared_h"):
                torch.testing.assert_close(
                    trace["probabilities"], model(h, full)[1]["probabilities"], rtol=0, atol=0
                )
            if trace:
                torch.testing.assert_close(
                    trace["weights"].sum(1), torch.ones(4, device=self.device)
                )
                self.assertLess(
                    linear_cost(model)["downstream_active_parameters_per_sample"], expected[variant]
                )

    def test_sparse_dense_output_and_gradients(self):
        for variant in ("spatial_egs", "vanilla_moe"):
            sparse = Downstream(variant, REGIONS).to(self.device)
            dense = copy.deepcopy(sparse)
            h = torch.randn(5, 200, device=self.device)
            m = torch.ones(5, 22, device=self.device, dtype=torch.bool)
            m[::2, 7] = False
            a, ta = sparse(h, m)
            b, tb = dense(h, m, dense=True)
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
            (a.square().mean() + 0.01 * importance_loss(ta["probabilities"])).backward()
            (b.square().mean() + 0.01 * importance_loss(tb["probabilities"])).backward()
            for (name, p), (other, q) in zip(sparse.named_parameters(), dense.named_parameters()):
                self.assertEqual(name, other)
                pg = p.grad if p.grad is not None else torch.zeros_like(p)
                qg = q.grad if q.grad is not None else torch.zeros_like(q)
                torch.testing.assert_close(pg, qg, rtol=2e-4, atol=2e-6)
            self.assertGreater(float(sparse.router[0].weight.grad.norm()), 0)

    def test_checkpoint_next_step_and_metadata_transaction(self):
        model = Downstream("spatial_egs", REGIONS).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        x = torch.randn(4, 200, device=self.device)
        m = torch.ones(4, 22, dtype=torch.bool, device=self.device)
        y = torch.arange(4, device=self.device)

        def step(net, optimizer):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(net(x, m)[0], y)
            loss.backward()
            optimizer.step()

        step(model, opt)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "synthetic.pth"
            save_checkpoint(path, model, opt, {"fixture": "synthetic_gpu", "fold": 9})
            restored = Downstream("spatial_egs", REGIONS).to(self.device)
            restored_opt = torch.optim.AdamW(restored.parameters(), lr=3e-4)
            load_checkpoint(path, restored, {"fixture": "synthetic_gpu", "fold": 9}, restored_opt)
            step(model, opt)
            step(restored, restored_opt)
            for a, b in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b)
            before = cpu_copy(restored.state_dict())
            with self.assertRaises(ValueError):
                load_checkpoint(path, restored, {"fold": 8}, restored_opt)
            for k, v in restored.state_dict().items():
                self.assertTrue(torch.equal(v.cpu(), before[k]))
            with self.assertRaises(ValueError):
                save_checkpoint(path, model, opt, {"fixture": "duplicate"})

    def test_official_mask_and_frozen_adapter(self):
        from neuroroute.vendor.cbramod import CBraMod

        # Full official architecture, randomly initialized synthetic fixture, NOT pretrained acceptance.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "random_official_fixture.pth"
            original = CBraMod()
            torch.save(original.state_dict(), path)
            del original
            encoder = FrozenEncoder(path, digest(path)).to(self.device)
            encoder.train(True)
            self.assertFalse(encoder.training)
            self.assertFalse(any(p.requires_grad for p in encoder.parameters()))
            x = torch.randn(2, 22, 4, 200, device=self.device)
            m = torch.ones(2, 22, device=self.device, dtype=torch.bool)
            m[:, 7] = False
            before = x.clone()
            a = encoder(x, m)
            altered = x.clone()
            altered[:, 7] = 9876
            b = encoder(altered, m)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertTrue(torch.equal(x, before))
            self.assertEqual(tuple(a.shape), (2, 200))
            self.assertFalse(a.requires_grad)
            full = torch.ones_like(m)
            with torch.no_grad():
                reference = encoder.backbone(x).mean(2).mean(1)
            torch.testing.assert_close(encoder(x, full), reference)
            joint = full.clone()
            joint[:, 7] = False
            joint[:, 11] = False
            changed = x.clone()
            changed[:, [7, 11]] = 1234.0
            torch.testing.assert_close(encoder(x, joint), encoder(changed, joint), rtol=0, atol=0)
            with self.assertRaises(ValueError):
                encoder(x, torch.zeros_like(m))

    def test_evaluator_partition_aggregation_and_modes(self):
        data = ToyData()
        model = ProgrammedModel().to(self.device).train()
        conditions = {"complete": [], "C3_only": ["C3"], "C4_only": ["C4"]}
        stats = {"mean": [0.0] * 22, "std": [1.0] * 22}
        a = evaluate(model, data, self.device, stats, conditions, 3)
        b = evaluate(model, data, self.device, stats, conditions, 8)
        self.assertAlmostEqual(a["selection_score"], 11 / 30)
        self.assertEqual(a["selection_score"], b["selection_score"])
        self.assertTrue(model.training)
        self.assertIsNone(model.dummy.grad)
        for name in conditions:
            self.assertEqual(a["conditions"][name]["batch_sizes"], [3, 3, 2])
            self.assertAlmostEqual(
                a["conditions"][name]["loss"], b["conditions"][name]["loss"], places=6
            )
            self.assertEqual(
                a["conditions"][name]["predictions"], b["conditions"][name]["predictions"]
            )

        class Broken(ProgrammedModel):
            def forward(self, x, m):
                raise RuntimeError("synthetic model failure")

        broken = Broken().to(self.device).train()
        with self.assertRaises(RuntimeError):
            evaluate(broken, data, self.device, stats, conditions, 3)
        self.assertTrue(broken.training)
        with self.assertRaises(ValueError):
            evaluate(model, data, self.device, stats, {"forbidden": ["C3", "C4"]}, 3)

    def test_shared_train_engine_synthetic_epoch(self):
        class ToyEncoder(nn.Module):
            def train(self, mode=True):
                return super().train(False)

            def forward(self, x, m):
                return (x.mean(2) * m[..., None]).sum(1) / m.sum(1, keepdim=True)

        data = ToyData(training=True)
        model = EEGModel(ToyEncoder().eval(), Downstream("spatial_egs", REGIONS)).to(self.device)
        opt = torch.optim.AdamW(model.downstream.parameters(), lr=3e-4)
        cfg = {
            "training": {"batch_size": 3, "balance_lambda": 0.01},
            "augmentation": {
                "selection": "uniform_legal_sets",
                "count_probabilities": [0.5, 0.25, 0.25],
            },
        }
        result = train_epoch(
            model,
            data,
            self.device,
            opt,
            {"mean": [0.0] * 22, "std": [1.0] * 22},
            cfg,
            42,
            1,
            progress_every=0,
        )
        self.assertEqual(result["samples"], 8)
        self.assertEqual(result["batch_sizes"], [3, 3, 2])
        self.assertFalse(model.encoder.training)
