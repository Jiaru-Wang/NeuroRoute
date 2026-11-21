"""Lectures 5.1/6.4: GPU epochs, whole-partition development evaluation.

Training metrics use predictions during updates; they are not final-model metrics.
An exception invalidates an epoch; previous optimizer steps are not rolled back.
"""

from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoints import optimizer_checked
from .contracts import partition, require
from .metrics import classification, routing_summary
from .missingness import fixed_mask, sample_configured_mask
from .models import importance_loss


@contextmanager
def evaluation_mode(model):
    states = {m: m.training for m in model.modules()}
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, state in states.items():
            module.training = state


def loader(dataset, batch_size, seed, training):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=False,
        num_workers=0,
        generator=generator,
        pin_memory=True,
    )


def normalized_batch(batch, device, statistics):
    x, y = batch["x"].to(device), batch["y"].to(device)
    require(x.dtype == torch.float32 and x.shape == (len(y), 22, 4, 200), "Invalid x schema")
    require(
        y.dtype == torch.int64 and y.ndim == 1 and bool(((y >= 0) & (y < 4)).all()),
        "Invalid y schema",
    )
    mean = torch.tensor(statistics["mean"], dtype=x.dtype, device=device)[None, :, None, None]
    std = torch.tensor(statistics["std"], dtype=x.dtype, device=device)[None, :, None, None]
    require(
        mean.shape == std.shape == (1, 22, 1, 1)
        and bool(torch.isfinite(mean).all())
        and bool(torch.isfinite(std).all())
        and bool((std > 0).all()),
        "Invalid fitted statistics",
    )
    x = (x - mean) / std
    require(bool(torch.isfinite(x).all()), "Nonfinite input")
    return x, y


def identities(batch, dataset=None):
    n = len(batch["y"])
    require(
        all(len(batch[k]) == n for k in ("session", "trial_id", "subject_id")),
        "Identity shape mismatch",
    )
    keys = [(s, int(i)) for s, i in zip(batch["session"], batch["trial_id"])]
    if dataset is not None:
        allowed = partition(dataset.fold)[dataset.role]
        require(dataset.subject_ids == allowed, "Partition subject declaration mismatch")
        for (session, trial), subject in zip(keys, batch["subject_id"]):
            sid = int(subject)
            require(
                sid in allowed
                and session in (f"A{sid:02d}T", f"A{sid:02d}E")
                and 1 <= trial <= 288,
                "Trial/subject/partition mismatch",
            )
    return keys


def append_predictions(output, batch, logits, trace, mask):
    scores = logits.detach().cpu()
    predicted = scores.argmax(1).tolist()
    for j, key in enumerate(identities(batch)):
        row = {
            "session": key[0],
            "trial_id": key[1],
            "subject_id": int(batch["subject_id"][j]),
            "y_true": int(batch["y"][j]),
            "y_pred": predicted[j],
            "logits": scores[j].tolist(),
            "available": mask[j].detach().cpu().tolist(),
        }
        if trace:
            row.update({k: v[j].detach().cpu().tolist() for k, v in trace.items()})
        output.append(row)


def finish(rows, dataset, loss_sum, batch_sizes):
    keys = {(r["session"], r["trial_id"]) for r in rows}
    require(
        len(rows) == len(keys) == len(dataset) and keys == dataset.keys,
        "Incomplete/duplicate partition coverage",
    )
    metrics = classification([r["y_true"] for r in rows], [r["y_pred"] for r in rows])
    result = {
        **metrics,
        "loss": loss_sum / len(rows),
        "batch_sizes": batch_sizes,
        "predictions": rows,
    }
    if rows and "probabilities" in rows[0]:
        result["routing"] = routing_summary(
            [r["probabilities"] for r in rows], [r["indices"] for r in rows]
        )
    return result


def train_epoch(
    model, dataset, device, optimizer, statistics, config, seed, epoch, progress_every=20
):
    require(
        dataset.role == "train" and device.type == "cuda",
        "Training requires train partition and CUDA",
    )
    optimizer_checked(model.downstream, optimizer)
    require(all(not p.requires_grad for p in model.encoder.parameters()), "Encoder is not frozen")
    model.train()
    require(not model.encoder.training, "Encoder must remain in eval mode")
    rows, seen, batch_sizes = [], set(), []
    loss_sum, balance_sum = 0.0, 0.0
    for step, batch in enumerate(
        loader(dataset, config["training"]["batch_size"], seed + epoch - 1, True), 1
    ):
        keys = identities(batch, dataset)
        require(
            len(set(keys)) == len(keys) and seen.isdisjoint(keys) and set(keys) <= dataset.keys,
            "Invalid training identities",
        )
        seen.update(keys)
        masks = []
        for session, trial in keys:
            masks.append(
                fixed_mask([])
                if model.downstream.variant == "B0"
                else sample_configured_mask(
                    config["augmentation"],
                    seed=seed,
                    fold=dataset.fold,
                    epoch=epoch,
                    session=session,
                    trial_id=trial,
                )
            )
        available = torch.tensor(masks, dtype=torch.bool, device=device)
        require(
            not bool(((~available[:, 7]) & (~available[:, 11])).any()), "Forbidden training mask"
        )
        x, y = normalized_batch(batch, device, statistics)
        optimizer.zero_grad(set_to_none=True)
        logits, trace = model(x, available)
        require(
            logits.shape == (len(y), 4) and bool(torch.isfinite(logits).all()), "Invalid logits"
        )
        ce = torch.nn.functional.cross_entropy(logits, y)
        imp = importance_loss(trace["probabilities"]) if trace else logits.new_zeros(())
        objective = ce + config["training"]["balance_lambda"] * imp
        require(bool(torch.isfinite(objective)), "Nonfinite objective")
        objective.backward()
        for name, p in model.downstream.named_parameters():
            if p.grad is not None:
                require(bool(torch.isfinite(p.grad).all()), f"Nonfinite gradient {name}")
            elif not name.startswith("experts."):
                raise ValueError(f"Missing downstream gradient: {name}")
        require(
            all(p.grad is None for p in model.encoder.parameters()), "Backbone gradient detected"
        )
        optimizer.step()
        require(
            all(bool(torch.isfinite(p).all()) for p in model.downstream.parameters()),
            "Nonfinite downstream parameters",
        )
        n = len(y)
        loss_sum += ce.detach().item() * n
        balance_sum += imp.detach().item() * n
        batch_sizes.append(n)
        append_predictions(rows, batch, logits, trace, available)
        if progress_every and step % progress_every == 0:
            print(f"train epoch={epoch} batches={step} samples={len(rows)}", flush=True)
    result = finish(rows, dataset, loss_sum, batch_sizes)
    result["mean_batch_importance_loss"] = balance_sum / len(rows)
    return result


def evaluate(
    model,
    dataset,
    device,
    statistics,
    conditions,
    batch_size,
    *,
    allow_test=False,
    intervention=None,
):
    require(device.type == "cuda", "Evaluation requires CUDA")
    require(
        dataset.role == "validation" or (dataset.role == "test" and allow_test),
        "Outer test requires locked evaluation entry",
    )
    require(isinstance(conditions, dict) and conditions, "Conditions required")
    base_masks = {
        name: fixed_mask(missing, development=dataset.role != "test")
        for name, missing in conditions.items()
    }
    rows = {name: [] for name in conditions}
    sums = {name: 0.0 for name in conditions}
    sizes, seen = [], set()
    with evaluation_mode(model):
        for batch in loader(dataset, batch_size, 0, False):
            keys = identities(batch, dataset)
            require(
                len(set(keys)) == len(keys) and seen.isdisjoint(keys) and set(keys) <= dataset.keys,
                "Invalid evaluation identity",
            )
            seen.update(keys)
            x, y = normalized_batch(batch, device, statistics)
            for name, base in base_masks.items():
                available = (
                    torch.tensor(base, dtype=torch.bool, device=device).expand(len(y), -1).clone()
                )
                if intervention is None:
                    logits, trace = model(x, available)
                else:
                    logits, trace = intervention(model, x, available)
                require(
                    logits.shape == (len(y), 4) and bool(torch.isfinite(logits).all()),
                    "Invalid evaluation logits",
                )
                loss = torch.nn.functional.cross_entropy(logits, y)
                sums[name] += loss.item() * len(y)
                append_predictions(rows[name], batch, logits, trace, available)
            sizes.append(len(y))
    results = {name: finish(rows[name], dataset, sums[name], sizes) for name in conditions}
    return {
        "conditions": results,
        "selection_score": float(np.mean([r["macro_f1"] for r in results.values()])),
        "role": dataset.role,
        "subject_ids": dataset.subject_ids,
        "fold_id": dataset.fold,
    }


def compact_evaluation(result):
    return {
        **{k: v for k, v in result.items() if k != "conditions"},
        "conditions": {
            k: {a: b for a, b in v.items() if a != "predictions"}
            for k, v in result["conditions"].items()
        },
    }
