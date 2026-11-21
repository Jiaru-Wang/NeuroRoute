"""Lecture 6.6: frozen-path interventions on explicitly locked final data."""

from pathlib import Path

import numpy as np
import torch

from .checkpoints import cpu_copy, load_checkpoint
from .contracts import canonical_hash, digest, partition, read_json, repo_state, require, write_json
from .data import TrialDataset
from .engine import compact_evaluation, evaluate, evaluation_mode, loader, normalized_batch
from .experiments import check_frozen, frozen_snapshot, model_for, new_run, setup
from .missingness import fixed_mask


def locked_context(config, project, cache, backbone_path, backbone_sha, summary_path, lock_path):
    summary_path = Path(summary_path).resolve()
    summary, lock = read_json(summary_path), read_json(lock_path)
    require(
        lock.get("scope") == "locked_final_test"
        and lock["config_sha256"] == canonical_hash(config),
        "Protocol not locked",
    )
    matches = [e for e in lock["runs"] if e["summary_sha256"] == digest(summary_path)]
    require(len(matches) == 1, "Run not authorized in final lock")
    require(
        summary["source_commit"] == repo_state(project)
        and summary["config_sha256"] == canonical_hash(config),
        "Code/config changed",
    )
    require(summary["backbone_sha256"] == backbone_sha, "Backbone changed")
    path = summary_path.parent / summary["best_checkpoint"]
    require(
        digest(path) == summary["best_checkpoint_sha256"] == matches[0]["checkpoint_sha256"],
        "Downstream checkpoint changed",
    )
    device = setup(config, summary["seed"])
    model = model_for(config, summary["variant"], backbone_path, backbone_sha, device)
    expected = {
        k: summary[k]
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
    metadata = load_checkpoint(
        path, model.downstream, {**expected, "scope": "development_training"}
    )
    require(
        metadata["normalization"]["subjects"] == partition(summary["fold_id"])["train"],
        "Normalization leakage",
    )
    dataset = TrialDataset(cache, summary["fold_id"], "test", config["preprocessing"])
    require(dataset.manifest_sha256 == summary["cache_manifest_sha256"], "Cache changed")
    return model, dataset, device, metadata["normalization"], lock, summary


def mechanism_run(
    config, project, cache, backbone_path, backbone_sha, summary_path, lock_path, output
):
    plan = config["final_test"]["mechanisms"]
    require(
        plan["approved"] is True and bool(plan["approval_note"]),
        "Mechanism plan must be approved before locking final protocol",
    )
    model, data, device, stats, lock, summary = locked_context(
        config, project, cache, backbone_path, backbone_sha, summary_path, lock_path
    )
    require(
        model.downstream.routed, "Routing interventions are not applicable to linear/MLP baselines"
    )
    kinds = plan["interventions"]
    supported = {
        "fixed_uniform",
        "uniform_selected",
        "reverse_selected_weights",
        "cyclic_expert_ids",
        "remove_shared",
        "donor_route",
        "donor_rank_weights",
    }
    require(
        kinds and len(kinds) == len(set(kinds)) and set(kinds) <= supported,
        "Unknown/duplicate intervention",
    )
    donor_name = plan["donor_condition"]
    require(donor_name in lock["conditions"], "Donor condition must be prespecified in lock")
    donor = fixed_mask(lock["conditions"][donor_name])
    original = frozen_snapshot(model)
    downstream = cpu_copy(model.downstream.state_dict())
    run = new_run(
        output, f"mechanisms_{summary['variant']}_fold{summary['fold_id']}_seed{summary['seed']}"
    )
    baseline = evaluate(
        model,
        data,
        device,
        stats,
        lock["conditions"],
        config["training"]["batch_size"],
        allow_test=True,
    )
    write_json(run / "baseline.json", baseline)
    effects = {}
    for kind in kinds:
        if kind == "remove_shared" and not model.downstream.shared:
            effects[kind] = {"status": "not_applicable", "reason": "Model has no always-active E0"}
            continue

        def intervention(m, x, available, kind=kind):
            head = m.downstream
            h = m.encoder(x, available)
            _, natural = head(h, available)
            indices, weights = natural["indices"], natural["weights"]
            if kind == "remove_shared":
                return head(h, available, remove_shared=True)
            if kind == "fixed_uniform":
                fixed = plan["fixed_expert_ids"][summary["variant"]]
                require(len(fixed) == head.k, "Fixed route must match K")
                indices = torch.tensor(fixed, device=device, dtype=torch.long).expand(len(x), -1)
                weights = torch.full_like(weights, 1 / head.k)
            elif kind == "uniform_selected":
                weights = torch.full_like(weights, 1 / head.k)
            elif kind == "reverse_selected_weights":
                weights = weights.flip(1)
            elif kind == "cyclic_expert_ids":
                offset = plan["expert_id_offset"]
                require(
                    type(offset) is int and 0 < offset < len(head.experts),
                    "Specify nonzero expert-ID permutation offset",
                )
                indices = (indices + offset) % len(head.experts)
            elif kind in ("donor_route", "donor_rank_weights"):
                donor_mask = torch.tensor(donor, dtype=torch.bool, device=device).expand(len(x), -1)
                _, route = head(m.encoder(x, donor_mask), donor_mask)
                weights = route["weights"]
                if kind == "donor_route":
                    indices = route["indices"]
            return head(h, available, routing=(indices, weights))

        result = evaluate(
            model,
            data,
            device,
            stats,
            lock["conditions"],
            config["training"]["batch_size"],
            allow_test=True,
            intervention=intervention,
        )
        write_json(run / f"{kind}.json", result)
        effects[kind] = {
            name: {
                metric: result["conditions"][name][metric] - baseline["conditions"][name][metric]
                for metric in ("accuracy", "macro_f1", "balanced_accuracy")
            }
            for name in lock["conditions"]
        }
    # Dense all-expert computation here is an analysis, not the sparse inference benchmark.
    similarity = {}
    with evaluation_mode(model):
        for name, missing in lock["conditions"].items():
            vectors = [[] for _ in model.downstream.experts]
            for batch in loader(data, config["training"]["batch_size"], 0, False):
                x, _ = normalized_batch(batch, device, stats)
                m = torch.tensor(fixed_mask(missing), dtype=torch.bool, device=device).expand(
                    len(x), -1
                )
                h = model.encoder(x, m)
                for i, expert in enumerate(model.downstream.experts):
                    vectors[i].append(expert(h).cpu().numpy())
            v = np.stack([np.concatenate(parts).reshape(-1) for parts in vectors]).astype(float)
            norms = np.linalg.norm(v, axis=1)
            valid = norms > 0
            cos = (v @ v.T) / np.maximum(norms[:, None] * norms[None, :], 1e-300)
            similarity[name] = {
                "cosine": cos.tolist(),
                "nonzero_experts": valid.tolist(),
                "scope": "candidate outputs on same masked trials; flattened sample/features",
            }
    check_frozen(model, original)
    require(
        all(torch.equal(v.cpu(), downstream[k]) for k, v in model.downstream.state_dict().items()),
        "Mechanism analysis modified weights",
    )
    report = {
        "scope": "locked_computational_interventions",
        "summary_sha256": digest(summary_path),
        "lock_sha256": digest(lock_path),
        "baseline": compact_evaluation(baseline),
        "effects_intervention_minus_baseline": effects,
        "expert_output_similarity": similarity,
        "donor_rank_weights_semantics": "donor sorted Top-K weights assigned to target sorted Top-K slots; not expert-identity matching",
        "interpretation": "computational-path evidence; no neurophysiological causal claim",
    }
    write_json(run / "report.json", report)
    return run
