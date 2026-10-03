import sys
print("Testing sklearn submodule imports...")

tests = [
    ("sklearn.linear_model.LogisticRegression", "from sklearn.linear_model import LogisticRegression"),
    ("sklearn.ensemble.RandomForestClassifier", "from sklearn.ensemble import RandomForestClassifier"),
    ("sklearn.model_selection.StratifiedShuffleSplit", "from sklearn.model_selection import StratifiedShuffleSplit"),
]

for name, code in tests:
    try:
        exec(code)
        print(f"  {name}: OK")
    except ImportError as e:
        print(f"  {name}: FAIL - {e}")
    except Exception as e:
        print(f"  {name}: FAIL - {type(e).__name__}: {e}")

print("\nTesting xgboost...")
try:
    from xgboost import XGBClassifier
    print("  XGBClassifier: OK")
except Exception as e:
    print(f"  XGBClassifier: FAIL - {e}")

print("\nTesting lightgbm...")
try:
    from lightgbm import LGBMClassifier
    print("  LGBMClassifier: OK")
except Exception as e:
    print(f"  LGBMClassifier: FAIL - {e}")

print("\nTesting optuna...")
try:
    import optuna
    study = optuna.create_study(direction="maximize")
    print(f"  optuna study creation: OK ({optuna.__version__})")
except Exception as e:
    print(f"  optuna: FAIL - {e}")
