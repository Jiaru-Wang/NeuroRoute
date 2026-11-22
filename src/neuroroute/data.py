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


def check_fir_physical_range(signal_uv, lower_uv, upper_uv):
    """Reject finite overflow/sentinel values; do not guess or silently repair gaps."""
    lower_uv, upper_uv = np.asarray(lower_uv), np.asarray(upper_uv)
    require(lower_uv.shape == upper_uv.shape == (signal_uv.shape[0],), "Physical range shape")
    require(
        np.isfinite(lower_uv).all() and np.isfinite(upper_uv).all() and np.all(lower_uv < upper_uv),
        "Invalid physical ranges",
    )
    tolerance = np.maximum(np.abs(lower_uv), np.abs(upper_uv)) * 1e-10
    invalid = np.isfinite(signal_uv) & (
        (signal_uv < (lower_uv - tolerance)[:, None])
        | (signal_uv > (upper_uv + tolerance)[:, None])
    )
    if invalid.any():
        channel, sample = np.argwhere(invalid)[0]
        raise ValueError(
            f"Finite out-of-range sample before FIR: channel={channel}, sample={sample}; "
            "inspect GDF overflow/gap encoding, do not clip or impute"
        )


def decode_gdf_gaps(signal_uv, header, picks, run_starts, trials):
    """Audited IV-2a encoding only: -32768 gaps vs retained -2048 ADC endpoints.

    Replace verified gap cells with NaN in a copy, preserving every sample index.
    Unknown encodings fail; endpoints are logged, not certified artifact-free.
    """
    signal_uv = np.asarray(signal_uv, dtype=np.float64)
    require(signal_uv.ndim == 2 and signal_uv.shape[0] == 22, "Expected 22 EEG channels")
    require(np.isfinite(signal_uv).all(), "Unexpected nonfinite GDF decoding; inspect source")

    def field(name):
        value = np.asarray(header[name])[picks]
        require(value.shape == (22,) and np.isfinite(value).all(), f"Invalid GDF {name}")
        return value[:, None]

    units, cal, offsets = field("units"), field("cal"), field("offsets")
    require(np.all(units == 1e-6) and np.all(cal > 0), "Unexpected GDF scaling")
    require(
        np.all(field("digital_min") == -2047) and np.all(field("digital_max") == 2047),
        "Unreviewed GDF digital range",
    )
    decoded = (signal_uv / (units * 1e6) - offsets) / cal
    require(np.max(np.abs(decoded - np.rint(decoded))) < 1e-6, "Noninteger GDF reconstruction")
    require(np.all(np.abs(decoded) <= 32768), "Unexpected digital magnitude")
    codes = np.rint(decoded).astype(np.int64)
    require(
        np.all(((codes >= -2048) & (codes <= 2047)) | (codes == -32768)),
        "Unknown GDF out-of-range encoding",
    )
    starts = np.asarray(run_starts)
    require(
        starts.ndim == 1 and starts.size > 0 and np.issubdtype(starts.dtype, np.integer),
        "Integer run starts required",
    )
    require(starts[0] == 0 and np.all(np.diff(starts) > 0), "Unexpected run order")
    require(starts[-1] < signal_uv.shape[1], "Run outside recording")
    expected = np.zeros(signal_uv.shape[1], dtype=bool)
    intervals = []
    for start in starts[1:]:
        left, right = int(start), int(start) + 100
        require(right <= signal_uv.shape[1] and not expected[left:right].any(), "Invalid gap span")
        expected[left:right] = True
        intervals.append([left, right])
    sentinel = codes == -32768
    require(np.all(sentinel == expected[None, :]), "Gap encoding/location/length mismatch")
    endpoints = codes == -2048
    per_trial = []
    for row in trials:
        left, cue = int(row["start_sample"]), int(row["cue_sample"])
        right = cue + 1000
        require(0 <= left <= cue < right <= signal_uv.shape[1], "Invalid trial span")
        require(not expected[left:right].any(), f"Trial {row['trial_id']} intersects GDF gap")
        per_trial.append(
            {
                "trial_id": int(row["trial_id"]),
                "lower_endpoint_channel_samples": int(endpoints[:, cue:right].sum()),
            }
        )
    result = signal_uv.copy()
    result[:, expected] = np.nan
    return result, {
        "encoding": "bciciv2a_int16_gap_minus32768_endpoint_minus2048_v1",
        "gap_intervals": intervals,
        "gap_time_samples": int(expected.sum()),
        "gap_channel_samples": int(sentinel.sum()),
        "lower_endpoint_channel_samples": int(endpoints.sum()),
        "trial_endpoint_counts": per_trial,
    }


def continuous_fir(signal_uv, run_starts, low_hz, high_hz):
    """Offline 250-Hz FIR within finite run segments; never cross acquisition gaps.

    No fitted statistics, channel mixing, trial rejection or input mutation.
    Preserve missing samples so subsequent trial checks still reject them.
    """
    from mne.filter import create_filter, filter_data

    signal_uv = np.asarray(signal_uv, dtype=np.float64)
    require(signal_uv.ndim == 2 and min(signal_uv.shape) > 0, "Expected channels x samples")
    n_times = signal_uv.shape[1]
    starts = np.asarray(run_starts)
    require(
        starts.ndim == 1 and (not starts.size or np.issubdtype(starts.dtype, np.integer)),
        "Run boundaries must be integer sample indices",
    )
    require(np.all((starts >= 0) & (starts < n_times)), "Run boundary outside recording")
    require(0 < low_hz < high_hz < 125, "Invalid FIR frequencies")
    options = dict(
        sfreq=250,
        l_freq=low_hz,
        h_freq=high_hz,
        method="fir",
        phase="zero",
        filter_length="auto",
        l_trans_bandwidth="auto",
        h_trans_bandwidth="auto",
        fir_window="hamming",
        fir_design="firwin",
    )
    taps = create_filter(None, **options, verbose="WARNING")
    finite = np.isfinite(signal_uv).all(axis=0)
    edges = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    segments = []
    for left, right in zip(edges[::2], edges[1::2]):
        boundaries = sorted({int(left), int(right)} | {int(s) for s in starts if left < s < right})
        segments.extend(zip(boundaries[:-1], boundaries[1:]))
    require(bool(segments), "No finite recording segments")
    for left, right in segments:
        require(
            right - left >= len(taps),
            f"FIR segment [{left},{right}) shorter than {len(taps)} taps; inspect recording",
        )
    filtered = signal_uv.copy()
    for left, right in segments:
        filtered[:, left:right] = filter_data(
            signal_uv[:, left:right],
            **options,
            pad="reflect_limited",
            n_jobs=1,
            copy=True,
            verbose="WARNING",
        )
    return filtered, {
        "filter": "continuous_fir_zero",
        "phase": "zero",
        "fir_design": "firwin",
        "fir_window": "hamming",
        "pad": "reflect_limited",
        "sfreq": 250,
        "low_hz": float(low_hz),
        "high_hz": float(high_hz),
        "filter_length_samples": len(taps),
        "filter_coefficients_sha256": canonical_hash(taps.tolist()),
        "segments": [list(s) for s in segments],
        "nonfinite_time_samples": int((~finite).sum()),
        "run_starts": sorted(set(int(s) for s in starts)),
    }


def prepare(raw_dir, labels_dir, output, config, *, source_commit):
    """Approved original-reference candidate; never claims official MAT parity.

    Filtering is absent, per-trial causal Butterworth, or continuous segmented FIR.
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
                filter_audit = None
                if p["filter"] == "continuous_fir_zero":
                    signal_uv, encoding_audit = decode_gdf_gaps(
                        signal_uv,
                        raw._raw_extras[0],
                        picks,
                        events[events[:, 2] == 32766, 0],
                        audit,
                    )
                    signal_uv, filter_audit = continuous_fir(
                        signal_uv, events[events[:, 2] == 32766, 0], p["low_hz"], p["high_hz"]
                    )
                    filter_audit["gdf_encoding"] = encoding_audit
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
                    "filter_audit": filter_audit,
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
