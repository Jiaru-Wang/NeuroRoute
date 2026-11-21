"""Lectures 6.5/6.7: explicitly approved ablation and run manifests.

Generation only; never launches processes or selects using test outcomes.
"""

import random
from copy import deepcopy
from pathlib import Path

from .contracts import (
    CHANNELS,
    VARIANTS,
    canonical_hash,
    regions_checked,
    require,
    validate_config,
    write_json,
)


def plan_runs(config, output):
    validate_config(config)
    jobs = [
        {"variant": v, "fold_id": f, "seed": s, "config_sha256": canonical_hash(config)}
        for v in VARIANTS
        for f in range(1, 10)
        for s in config["seeds"]
    ]
    write_json(output, {"scope": "planned_not_executed", "jobs": jobs, "job_count": len(jobs)})
    return jobs


def generate_ablations(config, plan, output):
    validate_config(config)
    require(
        plan["approved"] is True and bool(plan["approval_note"]), "Ablation grid must be approved"
    )
    require(
        all(
            isinstance(plan[k], list) and plan[k]
            for k in ("random_region_seeds", "shared_k", "balance_lambda", "eta")
        ),
        "Declare all ablation axes",
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for axis, values in (
        ("random_region_seed", plan["random_region_seeds"]),
        ("shared_k", plan["shared_k"]),
        ("balance_lambda", plan["balance_lambda"]),
        ("eta", plan["eta"]),
    ):
        require(len(values) == len(set(values)), "Duplicate ablation value")
        for index, value in enumerate(values):
            candidate = deepcopy(config)
            if axis == "random_region_seed":
                require(type(value) is int and value >= 0, "Invalid region seed")
                shuffled = list(CHANNELS)
                random.Random(value).shuffle(shuffled)
                offset = 0
                for name, group in config["regions"].items():
                    candidate["regions"][name] = shuffled[offset : offset + len(group)]
                    offset += len(group)
                regions_checked(candidate["regions"])
            elif axis == "balance_lambda":
                candidate["training"][axis] = value
            else:
                candidate["model"][axis] = value
            candidate["approval_note"] += (
                f"; approved ablation plan: {plan['approval_note']}; {axis}={value}"
            )
            candidate["ablation"] = {
                "axis": axis,
                "value": value,
                "base_config_sha256": canonical_hash(config),
            }
            validate_config(candidate)
            name = f"{axis}_{index:03d}.json"
            write_json(output / name, candidate)
            jobs.append(
                {
                    "config": name,
                    "sha256": canonical_hash(candidate),
                    "variant": "spatial_egs",
                    "retrain_required": True,
                }
            )
    write_json(
        output / "manifest.json",
        {
            "scope": "planned_not_executed",
            "jobs": jobs,
            "structural_ablations": "shared_h, shared_hm, mask_moe and vanilla_moe are in the main comparison plan",
        },
    )
    return jobs
