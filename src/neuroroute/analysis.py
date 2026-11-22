"""Lectures 6.5/6.6/6.7: real-output analysis, paired statistics and figures.

No fabricated results. Subject is the statistical unit after seed averaging.
Interventions are computational manipulations, not neurophysiological causation.
"""

import itertools
import math
from pathlib import Path

import numpy as np

from .contracts import digest, read_json, require, write_json
from .metrics import js_divergence, routing_summary, topk_switch


def analyze_predictions(path, output):
    data = read_json(path)
    require(
        data.get("status") == "final_evaluation_completed",
        "Use a genuine completed final prediction file",
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    conditions = data["conditions"]
    report = {
        "source_sha256": digest(path),
        "fold_id": data["fold_id"],
        "seed": data["seed"],
        "variant": data["variant"],
        "conditions": {},
        "paired_equal_missing_count": {},
    }
    reference = conditions["complete"]
    for name, result in conditions.items():
        rows = result["predictions"]
        require(len(rows) == result["samples"], "Prediction count mismatch")
        entry = {k: result[k] for k in ("accuracy", "macro_f1", "balanced_accuracy", "loss")}
        entry["drop_from_complete"] = {
            k: reference[k] - result[k] for k in ("accuracy", "macro_f1", "balanced_accuracy")
        }
        if "probabilities" in rows[0]:
            entry["routing"] = routing_summary(
                [r["probabilities"] for r in rows], [r["indices"] for r in rows]
            )
        report["conditions"][name] = entry
    for left, right in itertools.combinations(conditions, 2):
        a, b = conditions[left]["predictions"], conditions[right]["predictions"]

        def keyed(rows):
            return {(r["session"], r["trial_id"]): r for r in rows}

        aa, bb = keyed(a), keyed(b)
        require(
            len(aa) == len(a) and len(bb) == len(b) and aa.keys() == bb.keys(),
            "Paired trial identities differ",
        )
        keys = sorted(aa)
        require(
            all(
                aa[k]["y_true"] == bb[k]["y_true"] and aa[k]["subject_id"] == bb[k]["subject_id"]
                for k in keys
            ),
            "Paired labels/subjects differ",
        )
        if not all(sum(aa[k]["available"]) == sum(bb[k]["available"]) for k in keys):
            continue
        if "probabilities" not in a[0]:
            continue
        js = js_divergence(
            [aa[k]["probabilities"] for k in keys], [bb[k]["probabilities"] for k in keys]
        )
        switch = topk_switch([aa[k]["indices"] for k in keys], [bb[k]["indices"] for k in keys])
        report["paired_equal_missing_count"][f"{left}__{right}"] = {
            "mean_js": float(js.mean()),
            "mean_topk_switch": float(switch.mean()),
            "trials": [
                {
                    "session": k[0],
                    "trial_id": k[1],
                    "js": float(js[i]),
                    "topk_switch": float(switch[i]),
                }
                for i, k in enumerate(keys)
            ],
        }
    write_json(output / "routing_report.json", report)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(conditions)
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.3), 4))
    ax.bar(names, [conditions[n]["macro_f1"] for n in names])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1")
    ax.set_title(
        f"Actual final evaluation: {data['variant']}, fold {data['fold_id']}, seed {data['seed']}"
    )
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output / "macro_f1.png", dpi=180)
    plt.close(fig)
    if all("routing" in report["conditions"][n] for n in names):
        matrix = np.array([report["conditions"][n]["routing"]["load"] for n in names])
        fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.5)))
        im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1)
        ax.set_yticks(range(len(names)), names)
        ax.set_xticks(range(matrix.shape[1]), [f"E{i + 1}" for i in range(matrix.shape[1])])
        ax.set_xlabel("Candidate expert index (not a physiological region)")
        fig.colorbar(im, ax=ax, label="Selection fraction")
        fig.tight_layout()
        fig.savefig(output / "expert_load.png", dpi=180)
        plt.close(fig)
    return report


def paired_subject_statistics(paths, plan, output):
    """Predeclared exact sign-flip of 9 subject differences, percentile bootstrap.

    The sign-flip null assumes exchangeability/symmetry of paired differences.
    This is one supported statistical plan, not a universal default.
    """
    require(
        plan["approved"] is True and bool(plan["approval_note"]),
        "Statistical plan needs explicit approval",
    )
    require(plan["test"] == "exact_paired_sign_flip", "Unsupported statistical test")
    metric, condition = plan["metric"], plan["condition"]
    require(metric in ("accuracy", "macro_f1", "balanced_accuracy"), "Invalid metric")
    require(
        type(plan["bootstrap_replicates"]) is int and plan["bootstrap_replicates"] >= 100,
        "Specify bootstrap budget",
    )
    require(0 < plan["confidence"] < 1 and type(plan["seed"]) is int, "Invalid bootstrap plan")
    records = {}
    provenance = []
    # Optional predeclared aliases distinguish retrained ablations of the same class.
    groups = plan.get("groups")
    if groups is not None:
        require(isinstance(groups, dict) and groups, "Declare nonempty statistical groups")
        pairs = [(v["variant"], v["config_sha256"]) for v in groups.values()]
        require(len(pairs) == len(set(pairs)), "Ambiguous group assignments")
        require(
            all(len(v["config_sha256"]) == 64 for v in groups.values()),
            "Pin every ablation configuration hash before final analysis",
        )
    for path in paths:
        r = read_json(path)
        require(r.get("status") == "final_evaluation_completed", "Incomplete/nonfinal result")
        group = r["variant"]
        if groups is not None:
            matches = [
                name
                for name, v in groups.items()
                if v["variant"] == r["variant"] and v["config_sha256"] == r["config_sha256"]
            ]
            require(
                len(matches) == 1,
                "Result is not uniquely assigned to a predeclared statistical group",
            )
            group = matches[0]
        key = (group, r["fold_id"], r["seed"])
        require(key not in records, "Duplicate model/fold/seed")
        require(condition in r["conditions"], "Required condition absent")
        value = r["conditions"][condition][metric]
        require(math.isfinite(value) and 0 <= value <= 1, "Invalid observed metric")
        records[key] = r
        provenance.append({"file": str(Path(path)), "sha256": digest(path)})
    seeds = [42, 123, 3407]
    comparisons = []
    require(
        plan["comparisons"]
        and len({tuple(p) for p in plan["comparisons"]}) == len(plan["comparisons"]),
        "Empty/duplicate comparisons",
    )
    for reference, candidate in plan["comparisons"]:
        require(reference != candidate, "Cannot compare a model to itself")
        values = {}
        for name in (reference, candidate):
            expected = {(name, f, s) for f in range(1, 10) for s in seeds}
            require(
                expected <= records.keys(),
                f"Missing fold/seed for {name}; no imputation or silent omission",
            )
            require({k for k in records if k[0] == name} == expected, "Unexpected fold/seed")
            values[name] = np.array(
                [
                    np.mean([records[(name, f, s)]["conditions"][condition][metric] for s in seeds])
                    for f in range(1, 10)
                ]
            )
        for f in range(1, 10):
            for s in seeds:
                a, b = records[(reference, f, s)], records[(candidate, f, s)]
                require(
                    a["cache_manifest_sha256"] == b["cache_manifest_sha256"]
                    and a["backbone_sha256"] == b["backbone_sha256"],
                    "Incomparable data/backbone",
                )
                require(
                    (groups is not None or a["config_sha256"] == b["config_sha256"])
                    and a["source_commit"] == b["source_commit"],
                    "Comparison requires common code and either common config or explicitly pinned ablation groups",
                )
        difference = values[candidate] - values[reference]
        observed = float(difference.mean())
        signs = np.asarray(list(itertools.product([-1, 1], repeat=9)))
        null = (signs * difference).mean(1)
        pvalue = float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
        rng = np.random.default_rng(plan["seed"])
        boot = difference[rng.integers(0, 9, size=(plan["bootstrap_replicates"], 9))].mean(1)
        alpha = (1 - plan["confidence"]) / 2
        sd = float(difference.std(ddof=1))
        comparisons.append(
            {
                "reference": reference,
                "candidate": candidate,
                "subjects": 9,
                "seeds_averaged_per_subject": 3,
                "reference_subject_means": values[reference].tolist(),
                "candidate_subject_means": values[candidate].tolist(),
                "mean_difference": observed,
                "mean_difference_percentage_points": observed * 100,
                "paired_cohen_dz": observed / sd if sd > 0 else None,
                "percentile_bootstrap_ci": np.quantile(boot, [alpha, 1 - alpha]).tolist(),
                "percentile_bootstrap_ci_percentage_points": (
                    np.quantile(boot, [alpha, 1 - alpha]) * 100
                ).tolist(),
                "p_exact": pvalue,
                "wins": int((difference > 0).sum()),
                "ties": int((difference == 0).sum()),
                "losses": int((difference < 0).sum()),
            }
        )
    # Holm family-wise adjustment across every comparison declared in this plan.
    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i]["p_exact"])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * comparisons[i]["p_exact"]))
        comparisons[i]["p_holm"] = running
    report = {
        "scope": "subject_level_confirmatory_analysis",
        "plan": plan,
        "comparisons": comparisons,
        "sources": provenance,
    }
    write_json(output, report)
    return report


def align_expert_profiles(left, right):
    """6.6 exploratory correspondence; rows=conditions, columns=experts.

    Uses cosine cost and Hungarian matching. Alignment is not proof of equivalent
    semantics; prefer training/development profiles for prespecified matching.
    """
    from scipy.optimize import linear_sum_assignment

    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    require(
        a.ndim == 2 and a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
        "Profile matrices must align by condition",
    )
    require(
        (np.linalg.norm(a, axis=0) > 0).all() and (np.linalg.norm(b, axis=0) > 0).all(),
        "Unidentifiable inactive expert profile",
    )
    a = a / np.linalg.norm(a, axis=0)
    b = b / np.linalg.norm(b, axis=0)
    cost = 1 - a.T @ b
    i, j = linear_sum_assignment(cost)
    return {
        "left_indices": i.tolist(),
        "matched_right_indices": j.tolist(),
        "cosine_costs": cost[i, j].tolist(),
        "interpretation": "exploratory permutation alignment, not physiological identification",
    }


def plot_training(run, output):
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run)
    summary = read_json(run / "summary.json")
    require(summary["status"] == "development_training_completed", "Run is incomplete")
    rows = [json.loads(line) for line in (run / "epochs.jsonl").read_text().splitlines()]
    require(
        [r["epoch"] for r in rows] == list(range(1, summary["completed_epochs"] + 1)),
        "Epoch log coverage mismatch",
    )
    path = Path(output)
    require(not path.exists(), "Refusing to overwrite plot")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows])
    axes[0].set_ylabel("Training CE (during updates)")
    axes[1].plot([r["epoch"] for r in rows], [r["selection_score"] for r in rows])
    axes[1].set_ylabel("Development mean condition Macro-F1")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.axvline(summary["best_epoch"], linestyle="--", color="gray")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
