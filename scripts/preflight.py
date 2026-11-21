"""Read-only server entry; Python standard library only. No model/GPU allocation."""

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    result = {
        "project": str(root),
        "hostname": platform.node(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "uid": os.getuid() if hasattr(os, "getuid") else None,
        "cpu_count": os.cpu_count(),
        "disk_free_gib": shutil.disk_usage(root).free / 1024**3,
        "uv_path": shutil.which("uv"),
        "git_path": shutil.which("git"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "training_started": False,
        "files_modified": False,
    }
    for name in ("torch", "numpy", "scipy", "mne"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED_IN_THIS_INTERPRETER"
    commands = {
        "gpu": [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv",
        ],
        "gpu_processes": [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv",
        ],
        "commit": ["git", "rev-parse", "HEAD"],
        "working_tree": ["git", "status", "--porcelain"],
    }
    for name, command in commands.items():
        try:
            r = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=15)
            result[name] = {"exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            result[name] = {"error": str(error)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("PREFLIGHT_INSPECTION_ONLY — no training or file changes")


if __name__ == "__main__":
    main()
