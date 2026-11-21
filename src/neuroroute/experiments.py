"""Lectures 6.1/6.3/6.4: explicit GPU acceptance, training and locked test.

No work starts on import. All runs use exclusive timestamped directories.
No automatic training resume: restoring a head is not restoring full RNG state.
"""

import importlib.metadata
import json
import os
import platform
import random
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .checkpoints import cpu_copy, load_checkpoint, save_checkpoint
from .contracts import (
    VARIANTS,
    canonical_hash,
    digest,
    partition,
    read_json,
    repo_state,
    require,
    validate_config,
    write_json,
)
from .data import TrialDataset, fit_statistics
from .engine import compact_evaluation, evaluate, loader, normalized_batch, train_epoch
from .metrics import EarlyStopping
from .models import Downstream, EEGModel, FrozenEncoder, cuda_device, linear_cost


def setup(config, seed):
    validate_config(config)
    require(seed in config["seeds"], "Seed is not in approved plan")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(
        visible.startswith(("GPU-", "MIG-")) and "," not in visible,
        "Set CUDA_VISIBLE_DEVICES to exactly one authorized GPU/MIG UUID",
    )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return cuda_device()


def model_for(config, variant, checkpoint, weights_sha, device):
    require(variant in VARIANTS, "Unknown variant")
    encoder = FrozenEncoder(checkpoint, weights_sha)
    m = config["model"]
    downstream = Downstream(
        variant, config["regions"], shared_k=m["shared_k"], vanilla_k=m["vanilla_k"], eta=m["eta"]
    )
    return EEGModel(encoder, downstream).to(device)


def optimizer_for(model, config):
    t = config["training"]
    return torch.optim.AdamW(
        model.downstream.parameters(),
        lr=t["lr"],
        weight_decay=t["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def metadata(config, project, variant, fold, seed, weights_sha, manifest_sha):
    return {
        "schema": "neuroroute_run_v1",
        "source_commit": repo_state(project),
        "config_sha256": canonical_hash(config),
        "variant": variant,
        "fold_id": fold,
        "seed": seed,
        "backbone_sha256": weights_sha,
        "cache_manifest_sha256": manifest_sha,
        "partition": partition(fold),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "dependencies": {
                n: importlib.metadata.version(n)
                for n in ("torch", "numpy", "scipy", "mne", "matplotlib")
            },
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0),
        },
    }


def new_run(output, prefix):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = Path(output).resolve() / f"{prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def frozen_snapshot(model):
    return cpu_copy(model.encoder.state_dict())


def check_frozen(model, before):
    for key, value in model.encoder.state_dict().items():
        require(torch.equal(value.detach().cpu(), before[key]), f"Frozen backbone changed: {key}")
    require(
        all(p.grad is None for p in model.encoder.parameters()), "Frozen backbone received gradient"
    )


def smoke(config, project, cache, checkpoint, weights_sha, variant, fold, seed, output):
    """Small real-train GPU update plus complete legal development evaluation.

    The synthetic hidden-value counterfactual does not use outer-test data.
    This is pipeline acceptance, never an efficacy result.
    """
    device = setup(config, seed)
    train = TrialDataset(cache, fold, "train", config["preprocessing"])
    val = TrialDataset(cache, fold, "validation", config["preprocessing"])
    stats = fit_statistics(train, config["preprocessing"]["normalization"])
    meta = metadata(config, project, variant, fold, seed, weights_sha, train.manifest_sha256)
    run = new_run(output, f"smoke_{variant}_fold{fold}_seed{seed}")
    print("Run directory:", run, flush=True)
    write_json(run / "config.json", config)
    write_json(run / "metadata.json", meta)
    try:
        model = model_for(config, variant, checkpoint, weights_sha, device)
        before = frozen_snapshot(model)
        batch = next(iter(loader(train, min(4, config["training"]["batch_size"]), seed, False)))
        x, y = normalized_batch(batch, device, stats)
        original = x.clone()
        mask = torch.ones(len(y), 22, dtype=torch.bool, device=device)
        mask[:, 7] = False
        model.eval()
        with torch.no_grad():
            logits, _ = model(x, mask)
            changed = x.clone()
            changed[:, 7] = 12345.0
            alternate, _ = model(changed, mask)
            torch.testing.assert_close(logits, alternate, rtol=0, atol=0)
        optimizer = optimizer_for(model, config)
        head_before = cpu_copy(model.downstream.state_dict())
        model.train()
        scores, trace = model(x, mask)
        loss = torch.nn.functional.cross_entropy(scores, y)
        if trace:
            from .models import importance_loss

            loss = loss + config["training"]["balance_lambda"] * importance_loss(
                trace["probabilities"]
            )
        loss.backward()
        gradients = {
            name: float(p.grad.norm())
            for name, p in model.downstream.named_parameters()
            if p.grad is not None
        }
        require(
            gradients
            and all(np.isfinite(v) for v in gradients.values())
            and gradients["classifier.weight"] > 0,
            "Ineffective/nonfinite classifier gradient",
        )
        route_gradient_expected = model.downstream.routed and (
            (model.downstream.k > 1 and (not model.downstream.shared or model.downstream.eta < 1))
            or config["training"]["balance_lambda"] > 0
        )
        if route_gradient_expected:
            require(
                any(v > 0 for k, v in gradients.items() if k.startswith("router.")),
                "Router has no effective gradient",
            )
        if model.downstream.shared and model.downstream.eta > 0:
            require(
                any(v > 0 for k, v in gradients.items() if k.startswith("e0.")),
                "E0 has no effective gradient",
            )
        optimizer.step()
        require(
            any(
                not torch.equal(p.cpu(), head_before[k])
                for k, p in model.downstream.state_dict().items()
            ),
            "No head update",
        )
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            expected, _ = model(x, mask)
        cp = run / "smoke_downstream.pth"
        save_checkpoint(cp, model.downstream, optimizer, {**meta, "scope": "gpu_pipeline_smoke"})
        restored = model_for(config, variant, checkpoint, weights_sha, device)
        restored_opt = optimizer_for(restored, config)
        load_checkpoint(
            cp, restored.downstream, {**meta, "scope": "gpu_pipeline_smoke"}, restored_opt
        )
        restored.eval()
        with torch.no_grad():
            actual, _ = restored(x, mask)
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)
        # Verify optimizer continuation on identical batch; do not reuse the smoke head for training.
        for candidate, opt in ((model, optimizer), (restored, restored_opt)):
            candidate.train()
            opt.zero_grad(set_to_none=True)
            out, _ = candidate(x, mask)
            torch.nn.functional.cross_entropy(out, y).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        for a, b in zip(model.downstream.parameters(), restored.downstream.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        result = evaluate(
            model,
            val,
            device,
            stats,
            config["selection"]["conditions"],
            config["training"]["batch_size"],
        )
        check_frozen(model, before)
        check_frozen(restored, before)
        require(torch.equal(x, original), "Input mutated")
        write_json(run / "validation_predictions.json", result)
        report = {
            **meta,
            "scope": "L4_pipeline_smoke",
            "status": "passed",
            "training_trials_used": len(y),
            "validation_trials": len(val),
            "outer_test_loaded": False,
            "loss": float(loss.detach()),
            "gradient_norms": gradients,
            "counterfactual_passed": True,
            "checkpoint_roundtrip_passed": True,
            "router_nonzero_gradient_expected": route_gradient_expected,
            "optimizer_next_step_passed": True,
            "frozen_state_unchanged": True,
            "input_unchanged": True,
            "gpu_unit_report_sha256": None,
            "validation": compact_evaluation(result),
        }
        write_json(run / "acceptance.json", report)
        print("GPU_PIPELINE_SMOKE_PASSED:", run / "acceptance.json", flush=True)
        return run
    except BaseException as error:
        write_json(
            run / "failure.json",
            {
                "status": "failed",
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def train(
    config,
    project,
    cache,
    checkpoint,
    weights_sha,
    variant,
    fold,
    seed,
    output,
    acceptance,
    unit_report,
):
    device = setup(config, seed)
    dataset = TrialDataset(cache, fold, "train", config["preprocessing"])
    val = TrialDataset(cache, fold, "validation", config["preprocessing"])
    stats = fit_statistics(dataset, config["preprocessing"]["normalization"])
    meta = metadata(config, project, variant, fold, seed, weights_sha, dataset.manifest_sha256)
    proof = read_json(acceptance)
    require(
        proof.get("status") == "passed" and proof.get("scope") == "L4_pipeline_smoke",
        "Matching real GPU smoke acceptance required",
    )
    for k in (
        "source_commit",
        "config_sha256",
        "variant",
        "fold_id",
        "backbone_sha256",
        "cache_manifest_sha256",
    ):
        require(proof.get(k) == meta[k], f"GPU acceptance mismatch: {k}")
    units = read_json(unit_report)
    require(
        units.get("status") == "passed"
        and units.get("device_type") == "cuda"
        and units.get("source_commit") == meta["source_commit"],
        "GPU unit report for this code revision required",
    )
    run = new_run(output, f"train_{variant}_fold{fold}_seed{seed}")
    print("Run directory:", run, flush=True)
    write_json(run / "config.json", config)
    write_json(run / "metadata.json", meta)
    try:
        model = model_for(config, variant, checkpoint, weights_sha, device)
        optimizer = optimizer_for(model, config)
        frozen = frozen_snapshot(model)
        stopper = EarlyStopping(config["training"]["patience"], config["training"]["min_delta"])
        base = {**meta, "scope": "development_training", "normalization": stats}
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        best_path, best_predictions = None, None
        history = []
        for epoch in range(1, config["training"]["max_epochs"] + 1):
            tr = train_epoch(model, dataset, device, optimizer, stats, config, seed, epoch)
            optimizer.zero_grad(set_to_none=True)
            head_after = cpu_copy(model.downstream.state_dict())
            va = evaluate(
                model,
                val,
                device,
                stats,
                config["selection"]["conditions"],
                config["training"]["batch_size"],
            )
            require(all(p.grad is None for p in model.parameters()), "Evaluation created gradients")
            require(
                all(
                    torch.equal(v.cpu(), head_after[k])
                    for k, v in model.downstream.state_dict().items()
                ),
                "Evaluation mutated downstream state",
            )
            check_frozen(model, frozen)
            improved, stop = stopper.step(va["selection_score"], epoch)
            write_json(run / f"train_epoch_{epoch:03d}.json", tr)
            write_json(run / f"validation_epoch_{epoch:03d}.json", va)
            row = {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "train_macro_f1": tr["macro_f1"],
                "selection_score": va["selection_score"],
                "is_best": improved,
                "bad_epochs": stopper.bad,
                "validation": compact_evaluation(va),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            with (run / "epochs.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            print(
                f"Epoch {epoch}: train_loss={tr['loss']:.6f} val_mean_macro_f1={va['selection_score']:.6f} best={improved} bad_epochs={stopper.bad}",
                flush=True,
            )
            if improved:
                best_path = run / f"best_epoch_{epoch:03d}.pth"
                best_predictions = run / f"validation_epoch_{epoch:03d}.json"
                save_checkpoint(
                    best_path,
                    model.downstream,
                    optimizer,
                    {**base, "epoch": epoch, "selection_score": va["selection_score"]},
                )
            if stop:
                break
        require(best_path is not None, "No completed epoch")
        restored = load_checkpoint(best_path, model.downstream, base)
        require(restored["epoch"] == stopper.best_epoch, "Best epoch mismatch")
        check = evaluate(
            model,
            val,
            device,
            stats,
            config["selection"]["conditions"],
            config["training"]["batch_size"],
        )
        saved = read_json(best_predictions)
        max_diff = 0.0
        for condition in config["selection"]["conditions"]:
            a, b = saved["conditions"][condition], check["conditions"][condition]
            require(len(a["predictions"]) == len(b["predictions"]), "Reload coverage mismatch")
            for left, right in zip(a["predictions"], b["predictions"]):
                require(
                    all(
                        left[k] == right[k]
                        for k in (
                            "session",
                            "trial_id",
                            "subject_id",
                            "y_true",
                            "y_pred",
                            "available",
                        )
                    ),
                    "Reload identity/prediction mismatch",
                )
                x, y = np.asarray(left["logits"]), np.asarray(right["logits"])
                require(np.allclose(x, y, rtol=1e-5, atol=1e-6), "Reload logits mismatch")
                max_diff = max(max_diff, float(np.abs(x - y).max()))
            require(
                a["macro_f1"] == b["macro_f1"] and abs(a["loss"] - b["loss"]) <= 1e-6,
                "Reload metrics mismatch",
            )
        check_frozen(model, frozen)
        require(repo_state(project) == meta["source_commit"], "Code changed during run")
        summary = {
            **meta,
            "status": "development_training_completed",
            "completed_epochs": epoch,
            "best_epoch": stopper.best_epoch,
            "stop_reason": "early_stopping" if stopper.stopped else "max_epochs",
            "best_selection_score": stopper.best,
            "best_checkpoint": best_path.name,
            "best_checkpoint_sha256": digest(best_path),
            "best_validation_predictions": best_predictions.name,
            "best_validation_predictions_sha256": digest(best_predictions),
            "checkpoint_reload_max_diff": max_diff,
            "outer_test_loaded": False,
            "cost": linear_cost(model.downstream),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "acceptance_sha256": digest(acceptance),
            "unit_report_sha256": digest(unit_report),
        }
        write_json(run / "summary.json", summary)
        print("DEVELOPMENT_TRAINING_COMPLETED:", run / "summary.json", flush=True)
        return run
    except BaseException as error:
        write_json(
            run / "failure.json",
            {
                "status": "failed",
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def lock_test(config, summaries, output, approval_note):
    """Explicit owner action only; this function loads no EEG or model."""
    validate_config(config)
    require(approval_note.strip(), "Record explicit approval to open final evaluation")
    conditions = config["final_test"]["conditions"]
    from .missingness import fixed_mask

    require(
        isinstance(conditions, dict)
        and all(name in conditions for name in ("complete", "C3_only", "C4_only", "C3_C4")),
        "Specify final conditions",
    )
    require(
        conditions["complete"] == []
        and conditions["C3_only"] == ["C3"]
        and conditions["C4_only"] == ["C4"]
        and set(conditions["C3_C4"]) == {"C3", "C4"},
        "Final anchor conditions mismatch",
    )
    for missing in conditions.values():
        fixed_mask(missing)
    entries = []
    for path in summaries:
        path = Path(path).resolve()
        s = read_json(path)
        require(
            s["status"] == "development_training_completed"
            and s["config_sha256"] == canonical_hash(config),
            "Unfinished/mismatched training run",
        )
        cp = path.parent / s["best_checkpoint"]
        require(digest(cp) == s["best_checkpoint_sha256"], "Checkpoint changed")
        entries.append(
            {
                "summary": str(path),
                "summary_sha256": digest(path),
                "checkpoint_sha256": digest(cp),
                "variant": s["variant"],
                "fold_id": s["fold_id"],
                "seed": s["seed"],
            }
        )
    require(
        entries and len({(e["variant"], e["fold_id"], e["seed"]) for e in entries}) == len(entries),
        "Duplicate/empty final plan",
    )
    lock = {
        "scope": "locked_final_test",
        "config_sha256": canonical_hash(config),
        "conditions": conditions,
        "approval_note": approval_note,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runs": entries,
    }
    write_json(output, lock)
    return lock


def final_evaluation(
    config, project, cache, checkpoint, weights_sha, summary_path, lock_path, output
):
    summary_path = Path(summary_path).resolve()
    summary = read_json(summary_path)
    lock = read_json(lock_path)
    require(
        lock.get("scope") == "locked_final_test"
        and lock.get("config_sha256") == canonical_hash(config),
        "Final protocol lock missing/mismatch",
    )
    entries = [e for e in lock["runs"] if e["summary_sha256"] == digest(summary_path)]
    require(len(entries) == 1, "Training summary is not uniquely authorized by final lock")
    s = summary
    device = setup(config, s["seed"])
    require(
        repo_state(project) == s["source_commit"],
        "Final evaluation must use the trained code revision",
    )
    require(
        s["backbone_sha256"] == weights_sha and s["config_sha256"] == canonical_hash(config),
        "Source/config mismatch",
    )
    cp = summary_path.parent / s["best_checkpoint"]
    require(
        digest(cp) == entries[0]["checkpoint_sha256"] == s["best_checkpoint_sha256"],
        "Checkpoint hash mismatch",
    )
    model = model_for(config, s["variant"], checkpoint, weights_sha, device)
    expected = {
        k: s[k]
        for k in (
            "source_commit",
            "config_sha256",
            "variant",
            "fold_id",
            "seed",
            "backbone_sha256",
            "cache_manifest_sha256",
        )
    }
    meta = load_checkpoint(cp, model.downstream, {**expected, "scope": "development_training"})
    test = TrialDataset(cache, s["fold_id"], "test", config["preprocessing"])
    require(test.manifest_sha256 == s["cache_manifest_sha256"], "Data manifest changed")
    require(
        meta["normalization"]["subjects"] == partition(s["fold_id"])["train"],
        "Normalization fitted on wrong subjects",
    )
    run = new_run(output, f"final_{s['variant']}_fold{s['fold_id']}_seed{s['seed']}")
    print("Run directory:", run, flush=True)
    try:
        frozen = frozen_snapshot(model)
        head = cpu_copy(model.downstream.state_dict())
        result = evaluate(
            model,
            test,
            device,
            meta["normalization"],
            lock["conditions"],
            config["training"]["batch_size"],
            allow_test=True,
        )
        check_frozen(model, frozen)
        require(
            all(torch.equal(v.cpu(), head[k]) for k, v in model.downstream.state_dict().items()),
            "Final evaluation changed head",
        )
        result.pop("selection_score", None)
        result.update(
            {
                **expected,
                "status": "final_evaluation_completed",
                "checkpoint_sha256": digest(cp),
                "lock_sha256": digest(lock_path),
            }
        )
        write_json(run / "predictions.json", result)
        write_json(
            run / "summary.json",
            {**compact_evaluation(result), "predictions_sha256": digest(run / "predictions.json")},
        )
        print("FINAL_EVALUATION_COMPLETED:", run / "summary.json", flush=True)
        return run
    except BaseException as error:
        write_json(
            run / "failure.json",
            {
                "status": "failed",
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
