"""Phase 1 environment check. Reports versions and GPU status; exits 1 if not ready."""
import importlib
import platform
import sys

MAX_GPU_MEMORY_GB = 4.0


def version(mod):
    try:
        return importlib.import_module(mod).__version__
    except Exception as exc:  # missing or broken
        return f"MISSING ({type(exc).__name__})"


def main():
    ok = True
    print(f"Python        : {platform.python_version()}")
    for label, mod in [("PyTorch", "torch"), ("XGBoost", "xgboost"), ("SHAP", "shap"),
                       ("pandas", "pandas"), ("numpy", "numpy"),
                       ("scikit-learn", "sklearn"), ("matplotlib", "matplotlib"),
                       ("pytest", "pytest")]:
        v = version(mod)
        ok &= not v.startswith("MISSING")
        print(f"{label:<14}: {v}")

    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"CUDA available: {cuda}")
        if cuda:
            p = torch.cuda.get_device_properties(0)
            total = p.total_memory / 1024**3
            print(f"GPU           : {p.name}")
            print(f"GPU memory    : {total:.2f} GB total; research cap {MAX_GPU_MEMORY_GB} GB")
        else:
            print("GPU           : none (code will fall back to CPU)")
    except Exception:
        ok = False

    print("\nENVIRONMENT READY" if ok else "\nENVIRONMENT NOT READY")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
