"""Lectures 1.3/6.4: local-RNG legal masks, one Trial/Epoch view.

Returns Python availability booleans [22]; no EEG or learned statistics.
The 0/1/2 distribution is configured, never inferred from course examples.
"""

import hashlib
import itertools
import json
import random
import re

from .contracts import CHANNELS, require


def fixed_mask(missing, *, development=False):
    require(isinstance(missing, list) and len(missing) == len(set(missing)), "Invalid missing list")
    require(set(missing) <= set(CHANNELS) and len(missing) < 22, "Unknown/all-missing electrodes")
    require(
        not development or not {"C3", "C4"} <= set(missing),
        "Held-out C3/C4 joint missingness in development",
    )
    return [c not in missing for c in CHANNELS]


def sample_mask(probabilities, *, seed, fold, epoch, session, trial_id, view_id=0):
    require(
        all(type(v) is int for v in (seed, fold, epoch, trial_id, view_id)),
        "Integer identity fields required",
    )
    require(
        seed >= 0 and 1 <= fold <= 9 and epoch >= 1 and trial_id >= 1 and view_id >= 0,
        "Identity out of range",
    )
    require(isinstance(session, str) and re.fullmatch(r"A0[1-9][TE]", session), "Invalid session")
    require(
        len(probabilities) == 3
        and all(0 <= p <= 1 for p in probabilities)
        and abs(sum(probabilities) - 1) < 1e-12,
        "Invalid sampling probabilities",
    )
    key = json.dumps(
        [seed, fold, epoch, session, trial_id, view_id], separators=(",", ":")
    ).encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))
    u = rng.random()
    count = 0 if u < probabilities[0] else (1 if u < probabilities[0] + probabilities[1] else 2)
    legal = [x for x in itertools.combinations(CHANNELS, count) if not {"C3", "C4"} <= set(x)]
    return fixed_mask(list(rng.choice(legal)), development=True)


def sample_configured_mask(config, **identity):
    """PDF p300 four branches OR separately approved uniform-legal-set design."""
    if config["selection"] == "uniform_legal_sets":
        return sample_mask(config["count_probabilities"], **identity)
    require(config["selection"] == "pdf_four_branch", "Unknown sampler")
    # Validate identity using the common sampler without touching global RNG.
    sample_mask([1.0, 0.0, 0.0], **identity)
    fields = [identity[k] for k in ("seed", "fold", "epoch", "session", "trial_id")] + [
        identity.get("view_id", 0)
    ]
    rng = random.Random(
        int.from_bytes(
            hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).digest(), "big"
        )
    )

    def choose(probabilities):
        require(
            probabilities
            and all(0 <= p <= 1 for p in probabilities)
            and abs(sum(probabilities) - 1) < 1e-12,
            "Invalid probabilities",
        )
        u = rng.random()
        cumulative = 0.0
        for i, p in enumerate(probabilities):
            cumulative += p
            if u < cumulative:
                return i
        return len(probabilities) - 1

    branches = config["branch_probabilities"]
    other = config["other_count_probabilities"]
    require(
        len(branches) == 4 and len(other) == 2,
        "Expected four branches and one/two other-electrode counts",
    )
    branch = choose(branches)
    if branch == 0:
        missing = []
    elif branch == 1:
        missing = ["C3"]
    elif branch == 2:
        missing = ["C4"]
    else:
        missing = rng.sample([c for c in CHANNELS if c not in ("C3", "C4")], 1 + choose(other))
    return fixed_mask(missing, development=True)
