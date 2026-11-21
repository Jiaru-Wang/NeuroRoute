"""Lecture 6.5: synchronized GPU latency and memory on training examples.

Uses a trained downstream checkpoint; never opens outer-test EEG.
"""

from pathlib import Path

import numpy as np
import torch

from .checkpoints import load_checkpoint
from .contracts import canonical_hash, digest, read_json, repo_state, require, write_json
from .data import TrialDataset
from .engine import evaluation_mode, loader, normalized_batch
from .experiments import model_for, setup
from .models import linear_cost


def benchmark(
    config,
    project,
    cache,
    backbone_path,
    backbone_sha,
    summary_path,
    output,
    *,
    warmup,
    repeats,
    batch_size,
):
    require(
        type(warmup) is int
        and warmup >= 1
        and type(repeats) is int
        and repeats >= 10
        and type(batch_size) is int
        and batch_size > 0,
        "Explicit benchmark warmup/repeats/batch size required",
    )
    summary_path = Path(summary_path).resolve()
    s = read_json(summary_path)
    require(
        s["status"] == "development_training_completed"
        and s["config_sha256"] == canonical_hash(config)
        and s["source_commit"] == repo_state(project),
        "Training source mismatch",
    )
    require(s["backbone_sha256"] == backbone_sha, "Backbone mismatch")
    cp = summary_path.parent / s["best_checkpoint"]
    require(digest(cp) == s["best_checkpoint_sha256"], "Head hash mismatch")
    device = setup(config, s["seed"])
    model = model_for(config, s["variant"], backbone_path, backbone_sha, device)
    meta = load_checkpoint(
        cp,
        model.downstream,
        {
            "scope": "development_training",
            "config_sha256": s["config_sha256"],
            "variant": s["variant"],
            "fold_id": s["fold_id"],
            "seed": s["seed"],
            "backbone_sha256": backbone_sha,
        },
    )
    ds = TrialDataset(cache, s["fold_id"], "train", config["preprocessing"])
    require(
        ds.manifest_sha256 == s["cache_manifest_sha256"] and len(ds) >= batch_size,
        "Dataset/batch mismatch",
    )
    x, _ = normalized_batch(
        next(iter(loader(ds, batch_size, s["seed"], False))), device, meta["normalization"]
    )
    mask = torch.ones(batch_size, 22, device=device, dtype=torch.bool)
    report = {
        "scope": "real_training_batch_gpu_benchmark",
        "summary_sha256": digest(summary_path),
        "batch_size": batch_size,
        "warmup": warmup,
        "repeats": repeats,
        "gpu_name": torch.cuda.get_device_name(0),
        "condition": "complete",
        "cost": linear_cost(model.downstream),
        "total_parameters_including_frozen_backbone": sum(p.numel() for p in model.parameters()),
        "measurements": {},
    }
    with evaluation_mode(model):
        h = model.encoder(x, mask)
        _, batch_trace = model.downstream(h, mask)
        if batch_trace:
            selected = set(batch_trace["indices"].reshape(-1).tolist())
            active = sum(p.numel() for p in model.downstream.router.parameters()) + sum(
                p.numel() for p in model.downstream.classifier.parameters()
            )
            active += sum(
                p.numel() for i in selected for p in model.downstream.experts[i].parameters()
            )
            if model.downstream.shared:
                active += sum(p.numel() for p in model.downstream.e0.parameters())
            report["observed_batch_candidate_experts"] = sorted(selected)
        else:
            active = sum(p.numel() for p in model.downstream.parameters())
        report["observed_batch_unique_downstream_active_parameters"] = active
        for name, call in (
            ("full_sparse", lambda: model(x, mask)),
            ("downstream_sparse", lambda: model.downstream(h, mask)),
            ("downstream_dense_reference", lambda: model.downstream(h, mask, dense=True)),
        ):
            for _ in range(warmup):
                call()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            times = []
            for _ in range(repeats):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                call()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            report["measurements"][name] = {
                "median_batch_ms": float(np.median(times)),
                "p05_batch_ms": float(np.quantile(times, 0.05)),
                "p95_batch_ms": float(np.quantile(times, 0.95)),
                "samples_per_second_at_median": float(batch_size * 1000 / np.median(times)),
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
            }
    report["limitations"] = (
        "GPU event timings exclude CPU data loading; memory includes resident models/input/cached pooled features; batch routing and background GPU activity affect speed. No speedup claim from Top-K alone."
    )
    write_json(output, report)
    return report
