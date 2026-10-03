import sys
print(f"Python: {sys.version}")

for lib in ["sklearn", "xgboost", "lightgbm", "torch", "optuna", "numpy", "pandas"]:
    try:
        mod = __import__(lib)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  {lib}: OK ({ver})")
    except ImportError as e:
        print(f"  {lib}: IMPORT ERROR - {e}")
    except Exception as e:
        print(f"  {lib}: ERROR - {type(e).__name__}: {e}")
