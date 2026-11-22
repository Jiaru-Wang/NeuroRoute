"""Lectures 1.3/6.1: explicit contracts, configuration and provenance.

No EEG, gradient or fitted statistics. Unknown scientific choices fail closed.
"""

import hashlib
import json
import math
import subprocess
from pathlib import Path

# Explicit canonical 22-channel order; actual GDF names must be audited separately.
CHANNELS = "Fz FC3 FC1 FCz FC2 FC4 C5 C3 C1 Cz C2 C4 C6 CP3 CP1 CPz CP2 CP4 P1 Pz P2 POz".split()
VARIANTS = (
    "B0",
    "B1",
    "matched_mlp",
    "vanilla_moe",
    "mask_moe",
    "shared_h",
    "shared_hm",
    "spatial_egs",
)
UPSTREAM_COMMIT = "b9e961003214326972c567eff390e75b0287e32a"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Exclusive creation: an existing experimental artifact is never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def partition(fold):
    """PDF 6.4 candidate: deterministic seven/one/one; not nested CV."""
    require(type(fold) is int and 1 <= fold <= 9, "fold must be integer 1..9")
    val = fold - 1 if fold > 1 else 9
    return {
        "train": [s for s in range(1, 10) if s not in (val, fold)],
        "validation": [val],
        "test": [fold],
    }


def regions_checked(regions):
    require(isinstance(regions, dict) and len(regions) == 4, "Four named regions must be approved")
    require(
        all(isinstance(group, list) and group for group in regions.values()),
        "Regions must be nonempty lists",
    )
    flat = [c for group in regions.values() for c in group]
    require(
        len(flat) == 22 and sorted(flat) == sorted(CHANNELS),
        "Regions must partition all 22 electrodes exactly once",
    )
    return [[CHANNELS.index(c) for c in group] for group in regions.values()]


def number(value, name, low=0, high=None, strict_low=False):
    require(type(value) in (float, int) and math.isfinite(value), f"{name}: finite number required")
    require(value > low if strict_low else value >= low, f"{name}: out of range")
    require(high is None or value <= high, f"{name}: out of range")


def validate_preprocessing(c):
    """Shared numeric/channel checks; this alone grants no approval."""
    require(c["channel_order"] == CHANNELS, "Channel order changed")
    p = c["preprocessing"]
    require(p["artifact_policy"] in ("drop", "keep"), "Select artifact policy")
    require(
        p["reference"] == "original",
        "Only original acquisition reference supported; cross-channel rereferencing requires separate missingness audit",
    )
    require(
        p["filter"] in ("none", "trial_butter_causal", "continuous_fir_zero"),
        "Select supported filter explicitly",
    )
    if p["filter"] != "none":
        require(0 < p["low_hz"] < p["high_hz"] < 125, "Invalid filter frequencies")
        if p["filter"] == "trial_butter_causal":
            require(
                type(p["filter_order"]) is int and p["filter_order"] > 0, "Invalid filter order"
            )
        else:
            require(p["filter_order"] is None, "FIR uses automatic length, not IIR order")
    require(p["resample"] in ("fft", "polyphase", "mne_fft_auto"), "Select resampling method")
    number(p["uv_divisor"], "uv_divisor", strict_low=True)
    require(p["normalization"] in ("none", "train_channel_zscore"), "Select normalization")
    require(
        isinstance(p["gdf_eeg_names"], list)
        and len(p["gdf_eeg_names"]) == len(set(p["gdf_eeg_names"])) == 22,
        "Supply 22 audited GDF EEG names in canonical order",
    )
    require(
        isinstance(p["gdf_eog_names"], list)
        and len(p["gdf_eog_names"]) == len(set(p["gdf_eog_names"])) == 3
        and not set(p["gdf_eog_names"]) & set(p["gdf_eeg_names"]),
        "Supply three separate EOG names",
    )


def validate_prepare_config(c):
    """Accept data-only approval for cache creation, never for training."""
    if c.get("schema") == "neuroroute_v1":
        return validate_config(c)
    require(c.get("schema") == "neuroroute_data_v1", "Unsupported data schema")
    require(c.get("approval_scope") == "data_preparation", "Data approval scope required")
    require(
        c.get("approved") is True and bool(c.get("approval_note")),
        "Data preparation is not approved",
    )
    validate_preprocessing(c)
    return c


def validate_config(c):
    """Validate approved choices; no automatic completion of null fields."""
    require(c.get("schema") == "neuroroute_v1", "Unsupported schema")
    require(
        c.get("approved") is True and bool(c.get("approval_note")),
        "Protocol is not approved; resolve docs/PROTOCOL_DECISIONS.md",
    )
    require(
        c["split"] == "deterministic_7_1_1",
        "This implementation supports the PDF 7/1/1 candidate only; nested CV requires a separate reviewed protocol",
    )
    require(
        c["device"] == "cuda" and c["precision"] == "float32",
        "GPU float32 required; no CPU fallback",
    )
    require(c["channel_order"] == CHANNELS, "Channel order changed")
    regions_checked(c["regions"])
    validate_preprocessing(c)
    a = c["augmentation"]
    require(
        a["selection"] in ("uniform_legal_sets", "pdf_four_branch"),
        "Explicitly select an implemented sampling protocol",
    )
    specifications = (
        [("count_probabilities", 3)]
        if a["selection"] == "uniform_legal_sets"
        else [("branch_probabilities", 4), ("other_count_probabilities", 2)]
    )
    for key, length in specifications:
        require(isinstance(a[key], list) and len(a[key]) == length, f"Specify {key}")
        for v in a[key]:
            number(v, key, high=1)
        require(math.isclose(sum(a[key]), 1, abs_tol=1e-12), f"{key} must sum to one")
    require(
        c["selection"]["metric"] == "mean_condition_macro_f1",
        "Explicitly approve three-condition mean",
    )
    require(
        c["selection"]["conditions"] == {"complete": [], "C3_only": ["C3"], "C4_only": ["C4"]},
        "Development conditions must be complete/C3/C4 only",
    )
    t = c["training"]
    for key in ("batch_size", "max_epochs", "patience"):
        require(type(t[key]) is int and t[key] > 0, f"Invalid {key}")
    for key in ("lr", "weight_decay", "balance_lambda", "min_delta"):
        number(t[key], key, strict_low=(key == "lr"))
    require(t["optimizer"] == "AdamW", "Only explicitly selected AdamW is implemented")
    require(c["model"]["activation"] == "gelu", "PDF model uses GELU")
    require(
        c["model"]["ordinary_fusion"] == "residual_weighted_sum",
        "Approve ordinary MoE residual fusion explicitly",
    )
    number(c["model"]["eta"], "eta", high=1)
    for key, maximum in (("shared_k", 7), ("vanilla_k", 8)):
        require(type(c["model"][key]) is int and 1 <= c["model"][key] <= maximum, "Invalid Top-K")
    require(
        c["seeds"] == [42, 123, 3407], "Changing seed plan requires a separately reviewed protocol"
    )
    return c


def repo_state(root):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    require(
        not git("status", "--porcelain"),
        "Commit or explicitly resolve working-tree changes before an experiment",
    )
    return git("rev-parse", "HEAD")
