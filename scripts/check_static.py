"""静态交付检查：不导入 PyTorch，不运行模型，不代替 GPU 验收。"""

import ast
import hashlib
import json
import tomllib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    sources = sorted(
        p for folder in ("src", "tests", "scripts") for p in (root / folder).rglob("*.py")
    )
    test_count = 0
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        compile(tree, str(path), "exec")
        if path.is_relative_to(root / "tests"):
            test_count += sum(
                isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
                for n in ast.walk(tree)
            )
    configs = list((root / "configs").glob("*.json"))
    for path in configs:
        json.loads(path.read_text(encoding="utf-8"))
    for path in (root / "pyproject.toml", root / "uv.lock"):
        tomllib.loads(path.read_text(encoding="utf-8"))

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    vendor = root / "src/neuroroute/vendor"
    provenance = json.loads((vendor / "UPSTREAM.json").read_text())
    for name, record in provenance["files"].items():
        if sha(vendor / name) != record["vendored_sha256"]:
            raise ValueError(f"Official-source hash mismatch: {name}")
    pdfs = list((root / "docs/source").glob("*.pdf"))
    if (
        len(pdfs) != 1
        or sha(pdfs[0]) != "2e31ad2c289c94c57739ec8df4027508aca7857b71df6a78825c8a49c1dc7887"
    ):
        raise ValueError("Original PDF missing or changed")
    print(
        json.dumps(
            {
                "scope": "static_delivery_checks_only",
                "status": "passed",
                "python_files": len(sources),
                "test_methods_discovered_by_ast": test_count,
                "json_configs": len(configs),
                "original_pdf_sha256": sha(pdfs[0]),
                "model_tests_executed": False,
                "gpu_tests_executed": False,
                "real_data_loaded": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
