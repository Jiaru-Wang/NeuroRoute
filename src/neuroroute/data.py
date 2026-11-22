"""Lectures 6.2/1.3: audited GDF -> immutable session caches -> LOSO.

CPU file/signal processing is explicit; neural training remains GPU-only.
Cache x float32 [N,22,4,200], y int64 [N], original trial IDs [N].
No fitted normalization during cache creation. Fold statistics use train only.
"""

import importlib.metadata
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import CHANNELS, canonical_hash, digest, partition, read_json, require, write_json


def inspect_gdf(path):
    import mne

    with mne.io.read_raw_gdf(path, preload=False, verbose="ERROR") as raw:
        return {
            "channels": raw.ch_names,
            "types": raw.get_channel_types(),
            "sfreq": float(raw.info["sfreq"]),
            "samples": int(raw.n_times),
            "annotation_descriptions": sorted(set(raw.annotations.description.tolist())),
        }


def audit_trials(events, labels, kind, n_times, session):
    """6.2：先对齐全部原始 Trial 与独立标签，再允许伪迹筛选。"""
    require(kind in ("T", "E"), "Unknown session kind")
    starts = events[events[:, 2] == 768, 0]
    require(
        len(starts) == len(labels) and np.all(np.diff(starts) > 0),
        f"{session}: trial/label alignment failure",
    )
    audit = []
    for i, start in enumerate(starts):
        next_start = int(starts[i + 1]) if i + 1 < len(starts) else int(n_times)
        boundaries = events[
            (events[:, 2] == 32766) & (events[:, 0] > start) & (events[:, 0] < next_start), 0
        ]
        end = min(next_start, int(boundaries[0])) if len(boundaries) else next_start
        local = events[(events[:, 0] >= start) & (events[:, 0] < end)]
        cues = local[np.isin(local[:, 2], [769, 770, 771, 772, 783])]
        require(len(cues) == 1, f"{session}/{i + 1}: expected one cue")
        cue, code = int(cues[0, 0]), int(cues[0, 2])
        require(cue - start == 500, f"{session}/{i + 1}: cue not 2 seconds after trial start")
        require(
            (code - 769 == labels[i]) if kind == "T" else (code == 783),
            f"{session}/{i + 1}: cue/official label mismatch",
        )
        require(
            cue + 1000 <= end, f"{session}/{i + 1}: four-second crop crosses trial/run boundary"
        )
        audit.append(
            {
                "trial_id": i + 1,
                "start_sample": int(start),
                "end_sample": end,
                "cue_sample": cue,
                "label": int(labels[i]),
                "artifact": bool(np.any(local[:, 2] == 1023)),
            }
        )
    return audit


def prepare(raw_dir, labels_dir, output, config, *, source_commit):
    """Approved original-reference candidate; never claims official MAT parity.

    Filtering is either absent or per-trial causal Butterworth, configured explicitly.
    The official MAT reference applies CAR; this GDF pipeline does not mix channels.
    """
    import mne
    from scipy.io import loadmat
    from scipy.signal import butter, resample, resample_poly, sosfilt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    p = config["preprocessing"]
    manifest = {
        "schema": "neuroroute_cache_v1",
        "channel_order": CHANNELS,
        "source_commit": source_commit,
        "preprocessing": p,
        "preprocessing_sha256": canonical_hash(p),
        "dependencies": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "mne")},
        "sessions": [],
    }
    for sid in range(1, 10):
        for kind in ("T", "E"):
            session = f"A{sid:02d}{kind}"
            gdf, label_path = Path(raw_dir) / f"{session}.gdf", Path(labels_dir) / f"{session}.mat"
            labels_raw = np.asarray(loadmat(label_path)["classlabel"]).reshape(-1)
            require(
                labels_raw.size == 288 and np.isin(labels_raw, [1, 2, 3, 4]).all(),
                f"{session}: expected 288 official labels in 1..4",
            )
            labels = labels_raw.astype(np.int64) - 1
            with mne.io.read_raw_gdf(
                gdf, eog=p["gdf_eog_names"], preload=True, verbose="ERROR"
            ) as raw:
                require(raw.info["sfreq"] == 250, "Expected 250 Hz")
                require(
                    set(p["gdf_eeg_names"] + p["gdf_eog_names"]) <= set(raw.ch_names),
                    "Audited channel names do not match this session",
                )
                picks = [raw.ch_names.index(n) for n in p["gdf_eeg_names"]]
                require(
                    all(raw.get_channel_types()[i] == "eeg" for i in picks),
                    "Selected non-EEG channel",
                )
                events, _ = mne.events_from_annotations(
                    raw,
                    event_id=lambda desc: int(desc) if desc.isdecimal() else None,
                    verbose="ERROR",
                )
                events[:, 0] -= raw.first_samp
                audit = audit_trials(events, labels, kind, raw.n_times, session)
                # MNE EEG is SI volts; explicit conversion to microvolts precedes approved divisor.
                signal_uv = raw.get_data(picks=picks) * 1e6
                trials, ids, ys, flags = [], [], [], []
                for row in audit:
                    i = row["trial_id"] - 1
                    start, end, cue = row["start_sample"], row["end_sample"], row["cue_sample"]
                    flagged = row["artifact"]
                    row["selected"] = not (flagged and p["artifact_policy"] == "drop")
                    if not row["selected"]:
                        continue
                    segment = signal_uv[:, int(start) : end]
                    require(
                        np.isfinite(segment).all(),
                        f"{session}/{i + 1}: nonfinite raw trial; inspect acquisition gaps",
                    )
                    if p["filter"] == "trial_butter_causal":
                        sos = butter(
                            p["filter_order"],
                            [p["low_hz"], p["high_hz"]],
                            btype="bandpass",
                            fs=250,
                            output="sos",
                        )
                        segment = sosfilt(sos, segment, axis=-1)
                    window = segment[:, cue - int(start) : cue - int(start) + 1000]
                    window = (
                        resample(window, 800, axis=-1)
                        if p["resample"] == "fft"
                        else resample_poly(window, 4, 5, axis=-1)
                    )
                    x = (window / p["uv_divisor"]).reshape(22, 4, 200).astype(np.float32)
                    require(np.isfinite(x).all(), "Nonfinite processed trial")
                    trials.append(x)
                    ids.append(i + 1)
                    ys.append(labels[i])
                    flags.append(flagged)
                require(
                    trials and set(ys) == {0, 1, 2, 3},
                    "No selected examples for one or more classes",
                )
            cache_path = output / f"{session}.npz"
            with cache_path.open("xb") as f:
                np.savez_compressed(
                    f,
                    x=np.stack(trials),
                    y=np.asarray(ys, dtype=np.int64),
                    trial_ids=np.asarray(ids, dtype=np.int64),
                    artifacts=np.asarray(flags, dtype=bool),
                )
            audit_path = output / f"{session}_audit.json"
            write_json(audit_path, audit)
            manifest["sessions"].append(
                {
                    "session": session,
                    "subject_id": sid,
                    "file": cache_path.name,
                    "sha256": digest(cache_path),
                    "audit_file": audit_path.name,
                    "audit_sha256": digest(audit_path),
                    "gdf_sha256": digest(gdf),
                    "labels_sha256": digest(label_path),
                    "count": len(ids),
                    "trial_ids": ids,
                    "class_counts": np.bincount(ys, minlength=4).tolist(),
                }
            )
    write_json(output / "manifest.json", manifest)
    return manifest


class TrialDataset(Dataset):
    def __init__(self, cache, fold, role, preprocessing):
        require(role in ("train", "validation", "test"), "Invalid role")
        cache = Path(cache).resolve()
        manifest = read_json(cache / "manifest.json")
        require(
            manifest["schema"] == "neuroroute_cache_v1" and manifest["channel_order"] == CHANNELS,
            "Cache schema mismatch",
        )
        require(
            manifest["preprocessing_sha256"] == canonical_hash(preprocessing),
            "Preprocessing differs from approved config",
        )
        records = manifest["sessions"]
        require(
            {r["session"] for r in records} == {f"A{s:02d}{k}" for s in range(1, 10) for k in "TE"}
            and len(records) == 18,
            "Expected 18 unique sessions",
        )
        self.role, self.fold, self.rows, self.arrays = role, fold, [], []
        self.manifest_sha256 = digest(cache / "manifest.json")
        self.subject_ids = partition(fold)[role]
        for r in records:
            sid = int(r["session"][1:3])
            require(r["subject_id"] == sid, "Session subject mismatch")
            if sid not in self.subject_ids:
                continue  # Do not read the other roles' EEG caches.
            path = (cache / r["file"]).resolve()
            require(
                path.is_relative_to(cache) and digest(path) == r["sha256"],
                "Cache path/hash invalid",
            )
            with np.load(path, allow_pickle=False) as z:
                x, y, ids, flags = (z[key].copy() for key in ("x", "y", "trial_ids", "artifacts"))
            n = len(y)
            require(
                n == r["count"] and x.shape == (n, 22, 4, 200) and x.dtype == np.float32,
                "Cache x mismatch",
            )
            require(
                y.dtype == ids.dtype == np.int64 and y.shape == ids.shape == flags.shape == (n,),
                "Cache identity schema mismatch",
            )
            require(
                flags.dtype == bool and np.isfinite(x).all() and np.isin(y, range(4)).all(),
                "Invalid cache values",
            )
            require(
                ids.tolist() == r["trial_ids"]
                and len(set(ids.tolist())) == n
                and np.all((ids >= 1) & (ids <= 288)),
                "Invalid trial IDs",
            )
            require(
                np.bincount(y, minlength=4).tolist() == r["class_counts"], "Class count mismatch"
            )
            require(
                preprocessing["artifact_policy"] != "drop" or not flags.any(),
                "Flagged trial in drop-artifact cache",
            )
            index = len(self.arrays)
            self.arrays.append(x)
            self.rows.extend(
                {
                    "array": index,
                    "offset": j,
                    "session": r["session"],
                    "trial_id": int(ids[j]),
                    "subject_id": sid,
                    "label": int(y[j]),
                    "artifact": bool(flags[j]),
                }
                for j in range(n)
            )
        require(
            self.rows and {r["label"] for r in self.rows} == set(range(4)),
            "Partition lacks classes",
        )
        self.keys = {(r["session"], r["trial_id"]) for r in self.rows}
        require(len(self.keys) == len(self.rows), "Duplicate trial identities")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {
            "x": torch.from_numpy(self.arrays[r["array"]][r["offset"]].copy()),
            "y": r["label"],
            "session": r["session"],
            "trial_id": r["trial_id"],
            "subject_id": r["subject_id"],
            "artifact": r["artifact"],
        }


def fit_statistics(dataset, mode):
    """Train-only per-channel population moments, streamed in float64."""
    require(dataset.role == "train", "Cannot fit normalization outside training partition")
    if mode == "none":
        return {
            "mode": mode,
            "mean": [0.0] * 22,
            "std": [1.0] * 22,
            "subjects": dataset.subject_ids,
        }
    require(mode == "train_channel_zscore", "Unknown normalization")
    count, mean, m2 = 0, np.zeros(22), np.zeros(22)
    for x in dataset.arrays:
        a = x.astype(np.float64).transpose(1, 0, 2, 3).reshape(22, -1)
        n, mu, v = a.shape[1], a.mean(1), a.var(1)
        delta = mu - mean
        total = count + n
        m2 += v * n + delta * delta * count * n / total
        mean += delta * n / total
        count = total
    std = np.maximum(np.sqrt(m2 / count), 1e-6)
    return {
        "mode": mode,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "subjects": dataset.subject_ids,
    }
