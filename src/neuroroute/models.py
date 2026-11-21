"""Lectures 2.1/3.1/3.2/6.3/6.5: frozen encoder and eight comparisons.

x: float32 [B,22,4,200]; M: bool [B,22], True=available.
h/z: [B,200]; logits: [B,4]. Only downstream parameters train.
No fitted statistics or partition selection here. Tests: tests/test_gpu.py.
"""

from pathlib import Path

import torch
from torch import nn

from .contracts import VARIANTS, digest, regions_checked, require


def cuda_device():
    require(torch.cuda.is_available(), "CUDA unavailable; CPU/MPS fallback is prohibited")
    require(
        torch.cuda.device_count() == 1,
        "Expose exactly one authorized GPU using CUDA_VISIBLE_DEVICES",
    )
    torch.cuda.init()
    device = torch.device("cuda:0")
    probe = torch.ones(1, device=device)
    torch.cuda.synchronize(device)
    require(probe.item() == 1, "CUDA probe failed")
    del probe
    return device


def check_mask(m, batch):
    require(m.dtype == torch.bool and m.shape == (batch, 22), "availability must be bool [B,22]")
    require(bool(m.any(1).all()), "Every trial needs at least one electrode")


class FrozenEncoder(nn.Module):
    """Official strict weights loaded before replacing reconstruction projection."""

    def __init__(self, checkpoint, expected_sha):
        super().__init__()
        from .contracts import read_json

        vendor = Path(__file__).parent / "vendor"
        for name, record in read_json(vendor / "UPSTREAM.json")["files"].items():
            require(
                digest(vendor / name) == record["vendored_sha256"],
                f"Vendored backbone source changed: {name}",
            )
        from .vendor.cbramod import CBraMod

        require(
            isinstance(expected_sha, str) and len(expected_sha) == 64,
            "Approved checkpoint SHA256 required",
        )
        require(digest(checkpoint) == expected_sha, "Checkpoint SHA256 mismatch")
        self.weights_sha256 = expected_sha
        self.backbone = CBraMod()
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        require(
            isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values()),
            "Expected official plain state_dict; inspect other formats rather than guessing",
        )
        expected = self.backbone.state_dict()
        require(set(state) == set(expected), "Official checkpoint parameter keys mismatch")
        for name, value in state.items():
            require(
                value.shape == expected[name].shape
                and value.dtype == expected[name].dtype
                and bool(torch.isfinite(value).all()),
                f"Invalid official tensor: {name}",
            )
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.proj_out = nn.Identity()
        self.backbone.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, x, available):
        require(
            x.dtype == torch.float32 and x.ndim == 4 and x.shape[1:] == (22, 4, 200),
            "Invalid EEG shape/dtype",
        )
        check_mask(available, len(x))
        require(
            x.device == available.device and bool(torch.isfinite(x).all()),
            "Invalid device/nonfinite EEG",
        )
        self.backbone.eval()
        features = self.backbone(x, mask=(~available).unsqueeze(-1).expand(-1, -1, 4))
        require(
            features.shape == x.shape and bool(torch.isfinite(features).all()),
            "Invalid CBraMod output",
        )
        electrodes = features.mean(2)
        return (electrodes * available.unsqueeze(-1)).sum(1) / available.sum(1, keepdim=True)


class Expert(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(200, hidden), nn.GELU(), nn.Linear(hidden, 200))

    def forward(self, h):
        return self.net(h)


class Downstream(nn.Module):
    """Sparse implementation with dense reference; padded Router inputs per PDF 6.5.

    Ordinary MoE uses residual h + weighted experts; this fusion definition must
    be included in the approved comparison protocol. E0 removal is an intervention.
    """

    def __init__(self, variant, regions, *, shared_k=2, vanilla_k=3, eta=0.5):
        super().__init__()
        require(variant in VARIANTS, "Unknown baseline")
        require(0 <= eta <= 1, "eta outside [0,1]")
        self.variant, self.eta = variant, eta
        groups = regions_checked(regions)
        coverage = torch.zeros(22, 4)
        for g, indices in enumerate(groups):
            coverage[indices, g] = 1 / len(indices)
        self.register_buffer("coverage", coverage)
        self.classifier = nn.Linear(200, 4)
        self.shared = variant in ("shared_h", "shared_hm", "spatial_egs")
        self.routed = variant not in ("B0", "B1", "matched_mlp")
        if variant == "matched_mlp":
            self.body = Expert(1102)
        if self.routed:
            n = 7 if self.shared else 8
            self.k = shared_k if self.shared else vanilla_k
            require(type(self.k) is int and 1 <= self.k <= n, "Invalid Top-K")
            self.router = nn.Sequential(nn.Linear(226, 128), nn.GELU(), nn.Linear(128, n))
            self.experts = nn.ModuleList(Expert() for _ in range(n))
            if self.shared:
                self.e0 = Expert()

    def forward(self, h, available, *, dense=False, routing=None, remove_shared=False):
        require(
            h.ndim == 2 and h.shape[1] == 200 and h.dtype == torch.float32,
            "h must be float32 [B,200]",
        )
        check_mask(available, len(h))
        require(
            h.device == available.device and bool(torch.isfinite(h).all()),
            "Nonfinite/device mismatch",
        )
        trace = {}
        if not self.routed:
            z = self.body(h) if self.variant == "matched_mlp" else h
            return self.classifier(z), trace
        m = available.to(h.dtype)
        mask_input = (
            m if self.variant in ("mask_moe", "shared_hm", "spatial_egs") else torch.zeros_like(m)
        )
        a = m @ self.coverage if self.variant == "spatial_egs" else h.new_zeros(len(h), 4)
        p = self.router(torch.cat((h, mask_input, a), 1)).softmax(1)
        selected_p, indices = p.topk(self.k, dim=1)
        weights = selected_p / selected_p.sum(1, keepdim=True)
        if routing is not None:
            indices, weights = routing
            require(
                indices.shape == weights.shape == (len(h), self.k), "Invalid intervention shape"
            )
            require(
                indices.dtype == torch.long
                and indices.device == h.device
                and weights.device == h.device,
                "Intervention dtype/device mismatch",
            )
            require(bool(((indices >= 0) & (indices < len(self.experts))).all()), "Unknown expert")
            require(
                all(len(set(row)) == self.k for row in indices.tolist()),
                "Duplicate intervention expert",
            )
            require(
                bool(torch.isfinite(weights).all()) and bool((weights >= 0).all()),
                "Invalid weights",
            )
            require(
                torch.allclose(weights.sum(1), torch.ones(len(h), device=h.device)),
                "Weights must sum to one",
            )
        if dense:
            all_outputs = torch.stack([expert(h) for expert in self.experts], 1)
            chosen = all_outputs.gather(1, indices[..., None].expand(-1, -1, 200))
            mix = (chosen * weights[..., None]).sum(1)
        else:
            mix = torch.zeros_like(h)
            # Only assigned examples execute each expert. index_add preserves gradients.
            for e, expert in enumerate(self.experts):
                rows, slots = torch.where(indices == e)
                if rows.numel():
                    out = expert(h.index_select(0, rows)) * weights[rows, slots, None]
                    mix = mix.index_add(0, rows, out)
        if self.shared:
            general = torch.zeros_like(h) if remove_shared else self.e0(h)
            z = h + self.eta * general + (1 - self.eta) * mix
        else:
            z = h + mix
        trace = {"probabilities": p, "indices": indices, "weights": weights}
        return self.classifier(z), trace


class EEGModel(nn.Module):
    def __init__(self, encoder, downstream):
        super().__init__()
        self.encoder, self.downstream = encoder, downstream

    def forward(self, x, available, **kwargs):
        return self.downstream(self.encoder(x, available), available, **kwargs)


def importance_loss(probabilities):
    """Lecture 3.2 NR-9; population variance, hard Load is not optimized."""
    mean = probabilities.mean(0)
    return mean.var(unbiased=False) / (mean.mean().square() + 1e-12)


def linear_cost(head):
    """6.5: downstream-only theoretical parameters/MACs; no speed claim."""

    def count(m):
        return sum(p.numel() for p in m.parameters())

    def macs(m):
        return sum(x.in_features * x.out_features for x in m.modules() if isinstance(x, nn.Linear))

    total = count(head)
    if head.routed:
        active = count(head.router) + count(head.classifier) + head.k * count(head.experts[0])
        linear_macs = macs(head.router) + macs(head.classifier) + head.k * macs(head.experts[0])
        if head.shared:
            active += count(head.e0)
            linear_macs += macs(head.e0)
    else:
        active, linear_macs = total, macs(head)
    return {
        "downstream_total_parameters": total,
        "downstream_trainable_parameters": total,
        "downstream_active_parameters_per_sample": active,
        "downstream_linear_macs_per_sample": linear_macs,
        "scope": "excludes shared frozen backbone, nonlinearities, gather/scatter and pooling",
    }
