#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired statistics for the Stage-I comparisons (median of per-pair differences, percentile bootstrap over pairs,
Wilcoxon signed-rank test). Inputs are the per-pair arrays stored by review_stage1_baselines.py and by the
unified benchmark; output: validation/review_paired_stats.json (fixed seed)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(r"D:\ISMAR2026\egomotion-registration")
V = ROOT / "validation"
BINS = ["[0.0,0.5)", "[0.5,1.0)", "[1.0,2.0)", "[2.0,10.0)"]
N_BOOT = 20000
SEED = 20261024


def paired_gain(a, b, rng):
    diff = np.asarray(a, float) - np.asarray(b, float)
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    boots = np.median(diff[idx], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    try:
        p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        p = float("nan")
    return {"n": int(len(diff)), "median_gain": round(float(np.median(diff)), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)], "wilcoxon_p": p}


def main() -> None:
    rng = np.random.default_rng(SEED)
    out: dict = {"seed": SEED, "n_boot": N_BOOT}

    d = json.loads((V / "review_stage1_baselines.json").read_text(encoding="utf-8"))
    raw_add = d["raw_add_mm"]
    raw_px = d["raw_px"]
    b4 = BINS[3]
    val = {}
    for base in ("ecd_gn_existing", "ecd_gn_plus_imu", "roi_gn_oracle_imu", "roi_gn_pred_imu"):
        val[f"{base}_minus_full_add_mm_B4"] = paired_gain(raw_add[base][b4], raw_add["full_existing"][b4], rng)
        val[f"{base}_minus_full_px_B4"] = paired_gain(raw_px[base][b4], raw_px["full_existing"][b4], rng)
    out["validation_260"] = val

    u = json.loads((V / "unified_sota_benchmark.json").read_text(encoding="utf-8"))
    cp, ou = u["raw_40pairs_coupled"], u["raw_40pairs_ours"]

    def gather(src, field, bins):
        arr = []
        for k in bins:
            arr.extend(src[k][field])
        return np.asarray(arr, float)

    test = {}
    for label, field in (("add_mm", "add_mm"), ("target_rot_deg", "obj_rot"), ("screen_px", "reproj_px")):
        test[f"all_bins_{label}"] = paired_gain(gather(cp, field, BINS), gather(ou, field, BINS), rng)
        test[f"B4_{label}"] = paired_gain(gather(cp, field, [b4]), gather(ou, field, [b4]), rng)
    out["test_40"] = test

    (V / "review_paired_stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
