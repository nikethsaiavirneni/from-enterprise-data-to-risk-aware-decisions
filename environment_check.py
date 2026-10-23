#!/usr/bin/env python
"""Inspect the current Python, ML, and GPU environment for this research project.

Run with:
    python environment_check.py

The report is written to results/environment_report.json and printed to the console.
CUDA availability is diagnostic, not mandatory: CPU fallback is supported.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "results" / "environment_report.json"

PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "xgboost": "xgboost",
    "shap": "shap",
    "matplotlib": "matplotlib",
    "torch": "torch",
    "pytest": "pytest",
}


def package_report(distribution: str, module_name: str) -> dict[str, Any]:
    """Return installation/version/import status for one dependency."""
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None

    installed = version is not None
    import_ok = False
    import_error = None

    if installed:
        try:
            importlib.import_module(module_name)
            import_ok = True
        except Exception as exc:  # Report local environment issues without hiding them.
            import_error = f"{type(exc).__name__}: {exc}"

    return {
        "installed": installed,
        "version": version,
        "import_ok": import_ok,
        "import_error": import_error,
    }


def nvidia_smi_report() -> dict[str, Any]:
    """Read NVIDIA GPU details if nvidia-smi is available."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi was not found on PATH"}

    command = [
        executable,
        "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=False
        )
        if completed.returncode != 0:
            return {
                "available": False,
                "reason": (completed.stderr or completed.stdout).strip(),
            }

        gpus = []
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 5:
                gpus.append(
                    {
                        "name": fields[0],
                        "memory_total_mb": fields[1],
                        "memory_used_mb": fields[2],
                        "utilization_percent": fields[3],
                        "temperature_c": fields[4],
                    }
                )
        return {"available": bool(gpus), "gpus": gpus}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def torch_report() -> dict[str, Any]:
    """Report PyTorch and CUDA details without assuming CUDA is available."""
    try:
        import torch  # Imported only for this diagnostic.

        report: dict[str, Any] = {
            "import_ok": True,
            "version": torch.__version__,
            "cuda_build_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_count": int(torch.cuda.device_count()),
            "gpus": [],
        }

        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                report["gpus"].append(
                    {
                        "index": index,
                        "name": properties.name,
                        "total_memory_gb": round(
                            properties.total_memory / (1024**3), 3
                        ),
                    }
                )
        return report
    except Exception as exc:
        return {
            "import_ok": False,
            "cuda_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    packages = {
        distribution: package_report(distribution, module)
        for distribution, module in PACKAGES.items()
    }
    required = ["numpy", "pandas", "scikit-learn", "xgboost", "shap", "matplotlib", "torch", "pytest"]
    missing_or_broken = [
        name
        for name in required
        if not packages[name]["installed"] or not packages[name]["import_ok"]
    ]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "packages": packages,
        "pytorch_cuda": torch_report(),
        "nvidia_smi": nvidia_smi_report(),
        "research_configuration": {
            "target_max_gpu_memory_gb": 4.0,
            "cpu_preprocessing": True,
            "cpu_fallback_supported": True,
            "note": (
                "The 4 GB figure is a training target, not an environment check "
                "failure threshold. CUDA is optional when CPU fallback is used."
            ),
        },
        "readiness": {
            "core_dependencies_ready": len(missing_or_broken) == 0,
            "missing_or_broken_packages": missing_or_broken,
            "cuda_available": bool(torch_report().get("cuda_available", False)),
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved environment report to: {REPORT_PATH}")
    if missing_or_broken:
        print("\nSTATUS: NOT READY — install/fix the listed dependencies, then rerun.")
        return 1
    print("\nSTATUS: DEPENDENCIES READY — continue to dataset inspection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
