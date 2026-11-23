#!/usr/bin/env python
"""Run the Phase 13E nested-CV evaluation protocol on a synthetic scenario.

Generates the configured scenario in memory (nothing under data/ is written),
builds the cohort, runs repeated nested patient-level CV for the AutoML search
and the B0-B3 baselines, and writes a NEW run folder under
``<paths.outputs_dir>/<nested_cv.output_subdir>/<run_id>`` (never overwritten).

Usage (PowerShell, from the repo root)::

    .\\.venv\\Scripts\\python.exe scripts/run_nested_cv.py `
        --config configs/research_v4_nested_cv.yaml
    .\\.venv\\Scripts\\python.exe scripts/run_nested_cv.py `
        --config configs/research_v4_nested_cv.yaml `
        --scenario legacy_v1 --patients 1000 --repeats 1 --trials 5

DISCLAIMER: synthetic demonstration data only. Results are not clinical evidence.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DiaLong-AutoML nested-CV evaluation (Phase 13E)")
    p.add_argument("--config", required=True, help="YAML config path.")
    p.add_argument("--scenario", choices=["legacy_v1", "v2_standard"], default=None)
    p.add_argument("--patients", type=int, default=None, help="Override nested_cv.n_patients.")
    p.add_argument("--dataset-seed", type=int, default=None, dest="dataset_seed")
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--outer-folds", type=int, default=None, dest="outer_folds")
    p.add_argument("--inner-folds", type=int, default=None, dest="inner_folds")
    p.add_argument("--trials", type=int, default=None, help="Optuna trials per outer fold.")
    p.add_argument("--bootstrap", type=int, default=None, help="Bootstrap resamples.")
    p.add_argument("--run-name", default=None, dest="run_name",
                   help="Folder name (default: generated run id). Must not exist.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from dialong_automl.config import load_config
    from dialong_automl.experiments.nested_cv import NestedCVRunner
    from dialong_automl.utils.logging_utils import configure_logging
    from dialong_automl.utils.seed import set_global_seed

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = _repo_root / cfg_path
    cfg = load_config(cfg_path)
    n = cfg.nested_cv
    overrides = {"scenario": args.scenario, "n_patients": args.patients,
                 "dataset_seed": args.dataset_seed, "repeats": args.repeats,
                 "outer_folds": args.outer_folds, "inner_folds": args.inner_folds,
                 "n_trials": args.trials, "bootstrap_resamples": args.bootstrap}
    cfg.nested_cv = n.model_validate({**n.model_dump(),
                                      **{k: v for k, v in overrides.items() if v is not None}})
    if cfg.nested_cv.scenario == "v2_standard" and not cfg.cohort.require_outcome_measurement:
        logging.getLogger(__name__).warning(
            "v2_standard has missing future HbA1c; set cohort.require_outcome_measurement: true "
            "or unmeasured outcomes are labelled 0.")
    configure_logging(cfg.logging, logs_dir=Path(cfg.paths.logs_dir), force=True)
    set_global_seed(cfg.reproducibility.seed,
                    deterministic_torch=cfg.reproducibility.deterministic_torch,
                    benchmark_cudnn=cfg.reproducibility.benchmark_cudnn)

    res = NestedCVRunner(cfg, repo_root=_repo_root).run(run_name=args.run_name)

    print("\nDiaLong-AutoML nested CV  (SYNTHETIC DATA - NOT CLINICAL EVIDENCE)")
    m = res.run_result.metadata
    print(f"scenario={m['scenario']} eligible={m['n_eligible_patients']} "
          f"positives={m['n_positive']} repeats={cfg.nested_cv.repeats} "
          f"outer={cfg.nested_cv.outer_folds} inner={cfg.nested_cv.inner_folds}")
    print(f"selected models per fold: {m['selected_model_counts']}")
    print(f"\n{'predictor':26s} {'AUROC [95% CI]':26s} {'AUPRC':8s} {'Brier':8s}")
    for r in res.summary:
        brier = f"{r['brier']:.4f}" if r["brier"] is not None else "n/a"
        print(f"{r['predictor']:26s} {r['auroc']:.3f} [{r['auroc_ci_low']:.3f}, "
              f"{r['auroc_ci_high']:.3f}]   {r['auprc']:.3f}    {brier}")
    print("\nPaired AUROC differences (automl_selected - comparator):")
    for c in res.comparisons:
        if c["metric"] != "auroc":
            continue
        tag = "PRIMARY " if c["primary"] else ("CEILING " if c["role"] == "ceiling" else "")
        holm = f"{c['p_value_holm']:.3f}" if c["p_value_holm"] is not None else "n/a"
        print(f"  {tag}{c['comparator']:24s} {c['difference']:+.3f} "
              f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]  p={c['p_value']:.3f} holm={holm}")
    print(f"\nRun folder: {res.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
