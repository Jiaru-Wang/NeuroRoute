"""Lecture 6.4: atomic downstream checkpoint, immutable names and metadata.

Frozen CBraMod weights are referenced by SHA256, never duplicated in each run.
"""

import copy
import json
import os
import tempfile
from pathlib import Path

import torch

from .contracts import require


def cpu_copy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: cpu_copy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(cpu_copy(v) for v in x)
    return x


def optimizer_checked(head, optimizer):
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    expected = [p for p in head.parameters() if p.requires_grad]
    require(
        len(actual) == len(set(map(id, actual))) and set(map(id, actual)) == set(map(id, expected)),
        "Optimizer must contain exactly trainable downstream parameters",
    )


def save_checkpoint(path, head, optimizer, metadata):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), "Checkpoint already exists")
    optimizer_checked(head, optimizer)
    state = cpu_copy(head.state_dict())
    require(all(bool(torch.isfinite(v).all()) for v in state.values()), "Nonfinite checkpoint")
    meta = json.loads(json.dumps(metadata, allow_nan=False))
    payload = {
        "format": "neuroroute_downstream_v1",
        "model": state,
        "optimizer": cpu_copy(optimizer.state_dict()),
        "metadata": meta,
    }
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as f:
            temp = Path(f.name)
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.link(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def load_checkpoint(path, head, expected, optimizer=None):
    require(isinstance(expected, dict) and expected, "Expected metadata is mandatory")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(
        isinstance(payload, dict) and payload.get("format") == "neuroroute_downstream_v1",
        "Unsupported checkpoint",
    )
    metadata = payload.get("metadata")
    require(
        isinstance(metadata, dict)
        and all(k in metadata and metadata[k] == v for k, v in expected.items()),
        "Checkpoint metadata mismatch",
    )
    current, saved = head.state_dict(), payload.get("model")
    require(isinstance(saved, dict) and set(current) == set(saved), "Model keys mismatch")
    for key, v in saved.items():
        require(
            isinstance(v, torch.Tensor)
            and v.shape == current[key].shape
            and v.dtype == current[key].dtype
            and bool(torch.isfinite(v).all()),
            f"Invalid checkpoint tensor {key}",
        )
    if optimizer is not None:
        optimizer_checked(head, optimizer)
        require(isinstance(payload.get("optimizer"), dict), "Invalid optimizer state")
    before = cpu_copy(current)
    old_opt = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    try:
        head.load_state_dict(saved, strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
    except BaseException:
        head.load_state_dict(before, strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(old_opt)
        raise
    return metadata
