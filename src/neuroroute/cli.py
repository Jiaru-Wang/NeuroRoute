"""Explicit entry points; no training, downloads, or server connection on import."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from .contracts import (
    digest,
    read_json,
    repo_state,
    require,
    validate_config,
    validate_prepare_config,
    write_json,
)


def project_root():
    return Path(__file__).resolve().parents[2]


def doctor():
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": {},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "training_started": False,
    }
    for name in ("torch", "numpy", "scipy", "mne", "matplotlib"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = "NOT_INSTALLED"
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        result["gpu_query"] = {"exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        result["compute_processes"] = {
            "exit_code": r.returncode,
            "stdout": r.stdout,
            "stderr": r.stderr,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        result["gpu_query"] = {"error": str(e)}
    return result


def fetch(url, expected_sha, output, max_bytes):
    """User-supplied official URL plus independently reviewed checksum required."""
    parsed = urllib.parse.urlsplit(url)
    require(
        parsed.scheme == "https"
        and parsed.hostname
        and not parsed.username
        and not parsed.password,
        "Use a public HTTPS URL without credentials",
    )
    require(
        len(expected_sha) == 64 and all(c in "0123456789abcdef" for c in expected_sha),
        "Reviewed SHA256 required",
    )
    require(type(max_bytes) is int and max_bytes > 0, "Positive maximum bytes required")
    output = Path(output).resolve()
    require(not output.exists() and not output.is_symlink(), "Destination exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as f:
            temp = Path(f.name)
            with urllib.request.urlopen(url, timeout=30) as response:
                require(
                    urllib.parse.urlsplit(response.url).scheme == "https", "Refuse HTTPS downgrade"
                )
                size = 0
                while True:
                    block = response.read(1048576)
                    if not block:
                        break
                    size += len(block)
                    require(size <= max_bytes, "Download exceeds declared size limit")
                    f.write(block)
            f.flush()
            os.fsync(f.fileno())
        require(digest(temp) == expected_sha, "Downloaded file hash mismatch")
        os.link(temp, output)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return {
        "saved": str(output),
        "sha256": expected_sha,
        "bytes": size,
        "source_host": parsed.hostname,
    }


def gpu_tests(output, project):
    import unittest

    import torch

    from .models import cuda_device

    commit = repo_state(project)
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith(("GPU-", "MIG-")),
        "Expose authorized GPU UUID",
    )
    device = cuda_device()
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.discover(str(project / "tests"), pattern="test_*.py")
    require(suite.countTestCases() > 0, "No tests discovered")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "status": "passed" if result.wasSuccessful() and not result.skipped else "failed",
        "scope": "gpu_unit_and_synthetic_integration",
        "source_commit": commit,
        "device_type": device.type,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": str(torch.__version__),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skips": len(result.skipped),
        "failure_details": [
            {"test": str(test), "traceback": trace} for test, trace in result.failures
        ],
        "error_details": [{"test": str(test), "traceback": trace} for test, trace in result.errors],
        "skipped_details": [
            {"test": str(test), "reason": reason} for test, reason in result.skipped
        ],
    }
    write_json(output, report)
    require(report["status"] == "passed", "GPU test suite failed; inspect preceding test output")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="NeuroRoute: explicit GPU research stages, no automatic CPU fallback"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "doctor",
        help="Read-only environment and GPU process inspection; does not initialize a model",
    )
    p = sub.add_parser("inspect-gdf")
    p.add_argument("path", type=Path)
    p = sub.add_parser("fetch")
    p.add_argument("--url", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max-bytes", required=True, type=int)
    p = sub.add_parser("check-config")
    p.add_argument("--config", required=True, type=Path)
    p = sub.add_parser("prepare")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--labels-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("gpu-tests")
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("plan")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("ablations")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    for command in ("smoke", "train", "evaluate", "mechanisms", "benchmark"):
        p = sub.add_parser(command)
        for field in ("config", "cache", "backbone", "output"):
            p.add_argument("--" + field, required=True, type=Path)
        p.add_argument("--backbone-sha256", required=True)
        if command in ("smoke", "train"):
            p.add_argument("--variant", required=True)
            p.add_argument("--fold", required=True, type=int)
            p.add_argument("--seed", required=True, type=int)
        else:
            p.add_argument("--summary", required=True, type=Path)
        if command == "train":
            p.add_argument("--acceptance", required=True, type=Path)
            p.add_argument("--unit-report", required=True, type=Path)
        if command in ("evaluate", "mechanisms"):
            p.add_argument("--lock", required=True, type=Path)
        if command == "benchmark":
            for field in ("warmup", "repeats", "batch-size"):
                p.add_argument("--" + field, required=True, type=int)
    p = sub.add_parser("lock-test")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--summaries", required=True, nargs="+", type=Path)
    p.add_argument("--approval-note", required=True)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("analyze")
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("statistics")
    p.add_argument("--summaries", required=True, nargs="+", type=Path)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("align-experts")
    p.add_argument("--profiles", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("plot-training")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = None
    if hasattr(args, "config"):
        validator = validate_prepare_config if args.command == "prepare" else validate_config
        config = validator(read_json(args.config))
    if args.command == "doctor":
        result = doctor()
    elif args.command == "inspect-gdf":
        from .data import inspect_gdf

        result = inspect_gdf(args.path)
    elif args.command == "fetch":
        result = fetch(args.url, args.sha256, args.output, args.max_bytes)
    elif args.command == "check-config":
        result = {"status": "CONFIG_CONTRACT_VALID", "training_started": False}
    elif args.command == "gpu-tests":
        result = gpu_tests(args.output, project_root())
    elif args.command == "prepare":
        from .data import prepare

        commit = repo_state(project_root())
        manifest = prepare(args.raw_dir, args.labels_dir, args.output, config, source_commit=commit)
        result = {
            "status": "CACHE_CREATED",
            "sessions": len(manifest["sessions"]),
            "manifest": str(args.output / "manifest.json"),
        }
    elif args.command == "plan":
        from .planning import plan_runs

        result = {"planned_jobs": len(plan_runs(config, args.output)), "executed": False}
    elif args.command == "ablations":
        from .planning import generate_ablations

        result = {
            "generated_configs": len(generate_ablations(config, read_json(args.plan), args.output)),
            "executed": False,
        }
    elif args.command in ("smoke", "train"):
        from .experiments import smoke, train

        common = (
            config,
            project_root(),
            args.cache,
            args.backbone,
            args.backbone_sha256,
            args.variant,
            args.fold,
            args.seed,
            args.output,
        )
        path = (
            smoke(*common)
            if args.command == "smoke"
            else train(*common, args.acceptance, args.unit_report)
        )
        result = {"run_directory": str(path)}
    elif args.command == "lock-test":
        from .experiments import lock_test

        result = lock_test(config, args.summaries, args.output, args.approval_note)
    elif args.command == "evaluate":
        from .experiments import final_evaluation

        result = {
            "run_directory": str(
                final_evaluation(
                    config,
                    project_root(),
                    args.cache,
                    args.backbone,
                    args.backbone_sha256,
                    args.summary,
                    args.lock,
                    args.output,
                )
            )
        }
    elif args.command == "mechanisms":
        from .mechanisms import mechanism_run

        result = {
            "run_directory": str(
                mechanism_run(
                    config,
                    project_root(),
                    args.cache,
                    args.backbone,
                    args.backbone_sha256,
                    args.summary,
                    args.lock,
                    args.output,
                )
            )
        }
    elif args.command == "benchmark":
        from .benchmark import benchmark

        result = benchmark(
            config,
            project_root(),
            args.cache,
            args.backbone,
            args.backbone_sha256,
            args.summary,
            args.output,
            warmup=args.warmup,
            repeats=args.repeats,
            batch_size=args.batch_size,
        )
    elif args.command == "analyze":
        from .analysis import analyze_predictions

        result = analyze_predictions(args.predictions, args.output)
    elif args.command == "statistics":
        from .analysis import paired_subject_statistics

        result = paired_subject_statistics(args.summaries, read_json(args.plan), args.output)
    elif args.command == "align-experts":
        from .analysis import align_expert_profiles

        profiles = read_json(args.profiles)
        result = align_expert_profiles(profiles["left"], profiles["right"])
        write_json(args.output, result)
    else:
        from .analysis import plot_training

        plot_training(args.run, args.output)
        result = {"saved": str(args.output)}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
