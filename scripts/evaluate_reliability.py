"""Phase 14F: evaluate the reliability layer against design section 6.

READ-ONLY on the run folder: reads predictions.csv, summary.csv, comparisons.csv
and writes one new JSON file elsewhere (refuses to overwrite). Patient-level
bootstrap (patients resampled with replacement; every repeat uses the same
resampled patients); point estimates are means over repeats.

Usage::

    python scripts/evaluate_reliability.py RUN_DIR OUT_JSON [--n-boot 1000] [--seed 2026]

Synthetic demonstration data only. Not clinical evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CAL, ORACLE, B3 = "automl_calibrated", "oracle_latent_truth", "b3_logreg_all"
ALPHA, SENS_ALPHA = 0.10, 0.20


def mats(pred: pd.DataFrame, predictor: str, cols: list[str]) -> dict[str, np.ndarray]:
    """(repeats, patients) matrices in a fixed patient order."""
    sub = pred[pred.predictor == predictor]
    ids = sorted(sub["patient_id"].unique())
    out = {c: [] for c in cols}
    for _, g in sub.groupby("repeat", sort=True):
        g = g.set_index("patient_id").loc[ids]
        for c in cols:
            out[c].append(g[c].to_numpy())
    return {c: np.vstack(v) for c, v in out.items()}


def ci(draws: list[float]) -> list[float | None]:
    d = np.asarray([x for x in draws if x is not None and np.isfinite(x)], dtype=float)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if d.size else [None,
                                                                                         None]


def boot(stat, n: int, n_boot: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    point = stat(np.arange(n))
    draws = [stat(rng.integers(0, n, n)) for _ in range(n_boot)]
    return {"estimate": point, "ci": ci(draws)}


def mean_over_repeats(fn, R: int):
    def stat(idx):
        vals = [fn(r, idx) for r in range(R)]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    return stat


def stratified_diff(err: np.ndarray, low: np.ndarray, high: np.ndarray, strata: np.ndarray):
    """Size-weighted mean over strata of mean(err|low) - mean(err|high)."""
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        a, b = err[m & low], err[m & high]
        if a.size and b.size:
            w = m.sum()
            num += w * (a.mean() - b.mean())
            den += w
    return num / den if den else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out_json", type=Path)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(argv)
    if a.out_json.exists():
        print(f"Refusing to overwrite {a.out_json}", file=sys.stderr)
        return 1
    pred = pd.read_csv(a.run_dir / "predictions.csv")
    cols = ["y", "score", "threshold", "set_no_onset", "set_onset", "sens_set_no_onset",
            "sens_set_onset", "range_width", "reliability_level", "flags"]
    M = mats(pred, CAL, cols)
    y, p, thr = M["y"].astype(int), M["score"].astype(float), M["threshold"].astype(float)
    R, n = y.shape
    level = M["reliability_level"].astype(str)
    sq = (p - y) ** 2
    mis = ((p >= thr).astype(int) != y).astype(float)
    # Risk deciles of the calibrated probability, per repeat (label-free).
    dec = np.vstack([pd.qcut(pd.Series(p[r]).rank(method="first"), 10, labels=False).to_numpy()
                     for r in range(R)])
    out: dict[str, object] = {"source_run": str(a.run_dir), "n_patients": n, "repeats": R,
                              "n_boot": a.n_boot, "bootstrap_seed": a.seed,
                              "synthetic_data": True}

    # ---- R1 (from the run's own summary / comparisons) ----
    summ = pd.read_csv(a.run_dir / "summary.csv").set_index("predictor")
    comp = pd.read_csv(a.run_dir / "comparisons.csv")
    r1 = {}
    for nm in ("automl_selected", CAL, B3):
        row = summ.loc[nm]
        r1[nm] = {k: [row[k], row[f"{k}_ci_low"], row[f"{k}_ci_high"]]
                  for k in ("ece", "ece_equal_mass", "cal_slope", "cal_intercept", "brier",
                            "auroc", "auroc_fold_mean")}
    fm = comp[(comp.comparator == CAL) & (comp.metric == "auroc_fold_mean")].iloc[0]
    ece_d = comp[(comp.comparator == CAL) & (comp.metric == "ece")].iloc[0]
    citl = r1[CAL]["cal_intercept"]
    r1["pass_ece_lower"] = bool(r1[CAL]["ece"][0] < r1["automl_selected"]["ece"][0])
    r1["ece_difference_raw_minus_calibrated"] = [ece_d["difference"], ece_d["ci_low"],
                                                 ece_d["ci_high"]]
    r1["pass_citl_ci_contains_0"] = bool(citl[1] <= 0 <= citl[2])
    r1["fold_mean_auroc_difference"] = float(fm["difference"])
    r1["pass_fold_mean_auroc_unchanged"] = bool(abs(fm["difference"]) <= 1e-12)
    r1["pass"] = r1["pass_ece_lower"] and r1["pass_citl_ci_contains_0"] and \
        r1["pass_fold_mean_auroc_unchanged"]
    out["R1"] = r1

    # ---- R3 coverage (alpha 0.10 and 0.20 sensitivity) ----
    def coverage(set0, set1, cls):
        h0, h1 = set0.astype(bool), set1.astype(bool)
        cov = np.where(y == 1, h1, h0)

        def fn(r, idx):
            m = y[r, idx] == cls
            return float(cov[r, idx][m].mean()) if m.any() else None
        return fn
    r3 = {}
    for tag, s0, s1, al in (("alpha_0.10", "set_no_onset", "set_onset", ALPHA),
                            ("alpha_0.20_sensitivity", "sens_set_no_onset", "sens_set_onset",
                             SENS_ALPHA)):
        res = {}
        for cls, name in ((0, "no_onset"), (1, "onset")):
            res[f"coverage_{name}"] = boot(mean_over_repeats(coverage(M[s0], M[s1], cls), R),
                                           n, a.n_boot, a.seed)
        h0, h1 = M[s0].astype(bool), M[s1].astype(bool)
        res["set_shares"] = {"no_onset": float((h0 & ~h1).mean()), "onset": float((~h0 & h1).mean()),
                             "undecided": float((h0 & h1).mean()), "empty": float((~h0 & ~h1).mean())}
        res["criterion_min"] = 1 - al - 0.03
        res["pass"] = all(res[f"coverage_{c}"]["estimate"] >= 1 - al - 0.03
                          for c in ("no_onset", "onset"))
        r3[tag] = res
    out["R3"] = r3
    out["R3"]["pass"] = r3["alpha_0.10"]["pass"]

    # ---- R6: Low vs High within risk deciles ----
    shares = {lv: float((level == lv).mean()) for lv in ("High", "Moderate", "Low")}
    r6 = {"level_shares": shares}
    by_level = {}
    for lv in ("High", "Moderate", "Low"):
        m = level == lv
        by_level[lv] = {"brier": float(sq[m].mean()) if m.any() else None,
                        "misclassification": float(mis[m].mean()) if m.any() else None,
                        "onset_rate": float(y[m].mean()) if m.any() else None}
    r6["by_level_unstratified"] = by_level
    for name, err in (("brier", sq), ("misclassification", mis)):
        r6[f"stratified_low_minus_high_{name}"] = boot(mean_over_repeats(
            lambda r, idx, e=err: stratified_diff(e[r, idx], level[r, idx] == "Low",
                                                  level[r, idx] == "High", dec[r, idx]), R),
            n, a.n_boot, a.seed)
    lo_b = r6["stratified_low_minus_high_brier"]["ci"][0]
    lo_m = r6["stratified_low_minus_high_misclassification"]["ci"][0]
    r6["pass_brier"] = bool(lo_b is not None and lo_b > 0)
    r6["pass_misclassification"] = bool(lo_m is not None and lo_m > 0)
    r6["pass"] = r6["pass_brier"] and r6["pass_misclassification"]
    codes = pd.Series(M["flags"].ravel()).fillna("").astype(str).str.split(";").explode()
    r6["flag_shares"] = {k: float(v / M["flags"].size)
                         for k, v in codes[codes != ""].value_counts().items()}
    # Reason composition among Low predictions.
    und = M["set_no_onset"].astype(bool) & M["set_onset"].astype(bool)
    low = level == "Low"
    r6["low_with_undecided_set_share"] = float(und[low].mean()) if low.any() else None
    out["R6"] = r6

    # ---- R2: width vs |p_cal - oracle| (v2 only) ----
    if (pred.predictor == ORACLE).any():
        o = mats(pred, ORACLE, ["score"])["score"].astype(float)
        gap = np.abs(p - o)
        w = M["range_width"].astype(float)

        def rho(r, idx):
            return float(spearmanr(w[r, idx], gap[r, idx]).statistic)
        r2 = boot(mean_over_repeats(rho, R), n, a.n_boot, a.seed)
        r2["pass"] = bool(r2["ci"][0] is not None and r2["ci"][0] > 0)
        r2["note"] = ("Oracle is a coarse Monte-Carlo estimate (slope 0.17-0.5); "
                      "used only as a ranking reference.")
        out["R2"] = r2
    else:
        out["R2"] = {"pass": None, "note": "no oracle in this scenario"}

    # ---- Selective prediction ----
    sel = []
    for tag, keep in (("all", np.ones_like(level, bool)), ("drop_Low", level != "Low"),
                      ("drop_Low_and_Moderate", level == "High")):
        sel.append({"retained": tag, "share": float(keep.mean()),
                    "brier": float(sq[keep].mean()) if keep.any() else None,
                    "misclassification": float(mis[keep].mean()) if keep.any() else None,
                    "onset_rate": float(y[keep].mean()) if keep.any() else None})
    b = [s["brier"] for s in sel if s["brier"] is not None]
    mc = [s["misclassification"] for s in sel if s["misclassification"] is not None]
    out["selective_prediction"] = {
        "curve": sel,
        "monotone_brier": bool(all(x >= y_ for x, y_ in zip(b, b[1:]))),
        "monotone_misclassification": bool(all(x >= y_ for x, y_ in zip(mc, mc[1:]))),
        "note": "Unstratified: retained patients also differ in risk level.",
    }

    # ---- R7 candidate: disagreement with B3 (report only) ----
    if (pred.predictor == B3).any():
        pb = mats(pred, B3, ["score"])["score"].astype(float)
        d = np.abs(p - pb)
        top = np.vstack([d[r] > np.percentile(d[r], 95) for r in range(R)])
        aug = np.where(top, "Low", level)
        r7 = {}
        for name, err in (("brier", sq), ("misclassification", mis)):
            r7[f"stratified_low_minus_high_{name}_with_R7"] = boot(mean_over_repeats(
                lambda r, idx, e=err: stratified_diff(e[r, idx], aug[r, idx] == "Low",
                                                      aug[r, idx] == "High", dec[r, idx]), R),
                n, a.n_boot, a.seed)
        r7["added_low_share"] = float((top & (level != "Low")).mean())
        r7["rule"] = ("candidate: also Low if |calibrated p - B3 p| is above its 95th percentile "
                      "(per repeat, label-free); not adopted unless it improves separation")
        out["R7_candidate"] = r7

    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
