"""Post-hoc fold-averaged AUROC for an existing nested-CV run (Phase 14B).

READ-ONLY with respect to the run folder: reads ``predictions.csv`` and
``resolved_config.json`` and writes a new JSON file elsewhere (refuses to
overwrite). Re-computes the summary with the run's own bootstrap settings and
adds ``auroc_fold_mean`` (mean of per-outer-fold AUROCs, averaged over repeats).

This is a POST-HOC methodological correction, not a re-specified endpoint: the
pre-specified primary endpoint of Phase 13E stays pooled out-of-fold AUROC.
Pooled AUROC mixes the score scales of independently fitted fold models; a
model whose scale varies between folds loses pooled AUROC that it would not
lose within any single fold.

Usage::

    python scripts/posthoc_fold_mean_auroc.py RUN_DIR OUT_JSON

Synthetic demonstration data only. Not clinical evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dialong_automl.config.schema import AppConfig  # noqa: E402
from dialong_automl.experiments.nested_cv import NestedCVRunner  # noqa: E402

KEYS = ("auroc", "auroc_ci_low", "auroc_ci_high",
        "auroc_fold_mean", "auroc_fold_mean_ci_low", "auroc_fold_mean_ci_high")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out_json", type=Path)
    a = ap.parse_args(argv)
    if a.out_json.exists():
        print(f"Refusing to overwrite {a.out_json}", file=sys.stderr)
        return 1
    if a.out_json.resolve().is_relative_to(a.run_dir.resolve()):
        print("Output must not be written inside the run folder.", file=sys.stderr)
        return 1
    cfg = AppConfig.model_validate(json.loads((a.run_dir / "resolved_config.json").read_text()))
    pred = pd.read_csv(a.run_dir / "predictions.csv")
    summary, comps = NestedCVRunner(cfg).summarise(pred)

    # Consistency check: pooled AUROC must reproduce the run's own summary.csv.
    old = pd.read_csv(a.run_dir / "summary.csv").set_index("predictor")
    max_dev = max(abs(r["auroc"] - old.loc[r["predictor"], "auroc"]) for r in summary)

    out = {
        "source_run": str(a.run_dir), "note": __doc__.split("\n\n")[1].replace("\n", " "),
        "bootstrap_resamples": cfg.nested_cv.bootstrap_resamples,
        "bootstrap_seed": cfg.nested_cv.bootstrap_seed,
        "pooled_auroc_max_abs_deviation_from_run_summary": max_dev,
        "summary": [{"predictor": r["predictor"], **{k: r[k] for k in KEYS}} for r in summary],
        "comparisons": [c for c in comps if c["metric"] in ("auroc", "auroc_fold_mean")],
    }
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for r in out["summary"]:
        print(f"{r['predictor']:28s} pooled {r['auroc']:.4f} [{r['auroc_ci_low']:.4f}, "
              f"{r['auroc_ci_high']:.4f}]  fold-mean {r['auroc_fold_mean']:.4f} "
              f"[{r['auroc_fold_mean_ci_low']:.4f}, {r['auroc_fold_mean_ci_high']:.4f}]")
    for c in out["comparisons"]:
        print(f"{c['comparison']:45s} {c['metric']:16s} diff {c['difference']:+.4f} "
              f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] p={c['p_value']:.3f} "
              f"holm={c['p_value_holm']} primary={c['primary']}")
    print(f"pooled AUROC max deviation from run summary: {max_dev:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
