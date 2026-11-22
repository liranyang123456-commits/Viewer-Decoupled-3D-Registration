#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Blocked hold-out of the measured stereo extrinsic by triangulation, not by per-pair PnP.

The previous 4.85 mm figure is the disagreement between the fitted extrinsic and an
independent PnP on each pair. This script refits on time blocks and scores the
held-out pairs by reprojection, grid shape, and scale.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"E:\EGO_Mo")
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "stereo_cal", ROOT / "tools" / "calibrate_stereo_extrinsic.py"
)
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)

DATA = ROOT / "datasets"
SESSION = DATA / "calib_intrinsics_20260922_123120"
OUT = Path(r"D:\ISMAR2026\egomotion-registration\validation\stereo_holdout_triangulation.json")
PATTERN = (11, 8)
SQUARE_MM = 3.0


def grid_mm() -> np.ndarray:
    x, y = np.meshgrid(np.arange(PATTERN[0]), np.arange(PATTERN[1]))
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float64) * SQUARE_MM


def shape_metrics(pts: np.ndarray, obj: np.ndarray) -> tuple[float, float]:
    mu_e, mu_g = pts.mean(0), obj.mean(0)
    Xe, Xg = pts - mu_e, obj - mu_g
    H = Xe.T @ Xg
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1] *= -1
        R = Vt.T @ U.T
    aligned = Xe @ R.T + mu_g
    rmse = float(np.sqrt(np.mean(np.sum((aligned - obj) ** 2, axis=1))))
    grid = pts.reshape(PATTERN[1], PATTERN[0], 3)
    pitch = np.concatenate([
        np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
        np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
    ])
    scale = float(abs(np.median(pitch) - SQUARE_MM) / SQUARE_MM * 100.0)
    return rmse, scale


def triangulate(p0, p1, K0, d0, K1, d1, R, t_mm):
    u0 = cv2.undistortPoints(p0.reshape(-1, 1, 2), K0, d0).reshape(-1, 2)
    u1 = cv2.undistortPoints(p1.reshape(-1, 1, 2), K1, d1).reshape(-1, 2)
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([R, t_mm.reshape(3, 1)])
    pts4 = cv2.triangulatePoints(P0, P1, u0.T, u1.T)
    X = (pts4[:3] / pts4[3]).T
    rv1, _ = cv2.Rodrigues(R)
    r0 = cv2.projectPoints(X, np.zeros(3), np.zeros(3), K0, d0)[0].reshape(-1, 2)
    r1 = cv2.projectPoints(X, rv1, t_mm.reshape(3, 1), K1, d1)[0].reshape(-1, 2)
    err = np.vstack([r0 - p0, r1 - p1])
    rms = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
    return X, rms


def summarize(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"n": 0}
    q25, q50, q75 = np.percentile(a, [25, 50, 75])
    return {
        "n": int(len(a)),
        "median": round(float(q50), 4),
        "iqr": round(float(q75 - q25), 4),
        "mean": round(float(a.mean()), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
    }


def run_mode(name: str, motion_px: float, spacing: float, K0, d0, K1, d1, size) -> dict:
    rows, selection = cal.detect_quasi_static(
        SESSION, motion_px, 0.08, spacing, compensate_lag=True
    )
    print(name, "pairs", len(rows), flush=True)
    if len(rows) < 20:
        return {"mode": name, "motion_px": motion_px, "pairs": len(rows), "status": "too_few"}
    t = np.asarray([r[6] for r in rows], dtype=np.float64)
    order = np.argsort(t)
    rows = [rows[i] for i in order]
    t = t[order]
    folds = np.minimum((np.arange(len(rows)) * 4) // len(rows), 3)
    obj = grid_mm()
    fold_rows = []
    shape, scale, rms, pnp_t = [], [], [], []
    for fold in range(4):
        test = np.flatnonzero(folds == fold)
        train = [r for i, r in enumerate(rows) if i not in set(test.tolist())]
        print(f"  fold {fold} train {len(train)} test {len(test)}", flush=True)
        fit_rms, R, T, _E, _F = cal.fit(train, K0, d0, K1, d1, size)
        t_mm = T.reshape(3) * 1000.0
        re, te, ep = cal.holdout_score(rows, test, R, T, K0, d0, K1, d1)
        for i in test:
            p0 = rows[i][1].reshape(-1, 2)
            p1 = rows[i][2].reshape(-1, 2)
            X, pr = triangulate(p0, p1, K0, d0, K1, d1, R, t_mm)
            sh, sc = shape_metrics(X, obj)
            shape.append(sh)
            scale.append(sc)
            rms.append(pr)
        pnp_t.extend(te.tolist())
        fold_rows.append({
            "fold": fold,
            "train": len(train),
            "test": int(len(test)),
            "fit_rms_px": round(float(fit_rms), 4),
            "holdout_pnp_translation_rmse_mm": round(float(np.sqrt(np.mean(te ** 2))), 3) if len(te) else None,
            "holdout_epipolar_p50_px": round(float(np.median(ep)), 3) if len(ep) else None,
        })
    full_rms, R, T, _E, _F = cal.fit(rows, K0, d0, K1, d1, size)
    return {
        "mode": name,
        "motion_px": motion_px,
        "pairs": len(rows),
        "selection": {k: selection[k] for k in ("pairs", "motion_px_max", "right_offset_frames") if k in selection},
        "full_fit_rms_px": round(float(full_rms), 4),
        "full_baseline_mm": round(float(np.linalg.norm(T) * 1000.0), 3),
        "full_t_mm": [round(float(x) * 1000.0, 3) for x in np.asarray(T).reshape(3)],
        "full_r_deg": [round(float(x), 3) for x in np.degrees(cv2.Rodrigues(R)[0]).reshape(3)],
        "folds": fold_rows,
        "holdout_shape_rmse_mm": summarize(shape),
        "holdout_scale_err_pct": summarize(scale),
        "holdout_reproj_rms_px": summarize(rms),
        "holdout_pnp_translation_mm": summarize(pnp_t),
        "holdout_pnp_translation_rmse_mm": round(float(np.sqrt(np.mean(np.asarray(pnp_t) ** 2))), 3),
    }


def main() -> None:
    K0, d0, size = cal.load_camera(SESSION / "camera_calibration_cam0.json")
    K1, d1, _ = cal.load_camera(SESSION / "camera_calibration_cam1.json")
    report = {
        "session": SESSION.name,
        "metric": "time-interleaved 4-fold refit; test pairs scored by triangulation against the 3 mm grid",
        "modes": [
            run_mode("quasi_static_4px_interleaved", 4.0, 0.0, K0, d0, K1, d1, size),
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["modes"][0], indent=2)[:4000])
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
