#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Part B redone with the real gyroscope and an independent image-space reference.

The original eval_multiseq_real_endoscope.py scored the "viewer update" with the ground-truth camera motion
plus synthetic Gaussian noise (0.15 deg, 0.45-1.45 mm), not with the recorded IMU. Here, for the same 13
recorded sequences:

* truth        : board pose from PnP on the endoscope's own frames (pose_gt_raw), i.e. what the overlay must
                 align with at the later frame; no IMU involved.
* viewer-only  : rotation from the recorded gyroscope (integrated over the lag, cam0 USB delay removed),
                 translation by constant-velocity extrapolation of the camera positions seen so far.
* camera-IMU rotation: (a) the single calibration used in the paper, (b) a per-session fit with 2-fold
                 block cross-validation (fit on one half of the recording, score on the other half).
* coupled / delayed / 2D homography baselines as before (homography uses the same gyro rotation).

Lag tau = 2 frames (~67 ms) is the main setting; 1, 3, 4 frames are also reported.
Writes validation/review_partB_real_gyro.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_real_viewer_overlay as E  # noqa: E402

DATA = E.DATA
OUT = Path(r"D:\ISMAR2026\egomotion-registration\validation\review_partB_real_gyro.json")
SESSIONS = [
    "traj_20260923_015507", "traj_20260923_015629", "traj_20260923_015918", "traj_20260923_020132",
    "traj_20260923_020407", "traj_20260923_023241", "traj_20260923_023422", "traj_20260924_143156",
    "traj_20260924_144019", "traj_20260926_143837", "gtraj_20261002_002134", "gtraj_20261002_003129",
    "gtraj_20261002_024309",
]
rng = np.random.default_rng(0)
R_CI_PAPER = E.R_CI.copy()


def centered_points():
    x, y = np.meshgrid(np.arange(11), np.arange(8))
    pts = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float64) * 0.003
    return pts - pts.mean(axis=0, keepdims=True)


def load_raw(session):
    z = np.load(DATA / "pose_gt_raw" / f"{session}.npz")
    delay = float(np.load(DATA / "pose_gt" / f"{session}.npz")["delay_usb"][0])
    use = np.flatnonzero((z["usable"] == 1) & (z["px"] < 2.0))
    t = z["t"].astype(np.float64)[use] - delay
    return t, z["R"].astype(np.float64)[use], z["p"].astype(np.float64)[use], delay


def gyro_vec(t_imu, w_imu, t0, t1):
    """Integral of the raw gyro vector (IMU frame) over [t0, t1] (small-angle sum)."""
    i0, i1 = np.searchsorted(t_imu, t0), np.searchsorted(t_imu, t1)
    if i1 <= i0:
        return np.zeros(3)
    ts = t_imu[i0:i1]
    dts = np.diff(ts, append=t1)
    dts[0] += ts[0] - t0
    return (w_imu[i0:i1] * np.clip(dts, 0, None)[:, None]).sum(0)


def delta_R(t_imu, w_imu, t0, t1, Rci):
    i0, i1 = np.searchsorted(t_imu, t0), np.searchsorted(t_imu, t1)
    if i1 <= i0:
        return np.eye(3)
    ts = t_imu[i0:i1]
    wc = (Rci @ w_imu[i0:i1].T).T
    dts = np.diff(ts, append=t1)
    dts[0] += ts[0] - t0
    R = np.eye(3)
    for wi, dt in zip(wc, dts):
        if dt > 0:
            R = R @ Rotation.from_rotvec(wi * dt).as_matrix()
    return R


def kabsch(G, V):
    """Rotation Q minimising sum |V - Q G|^2 (rows are vectors)."""
    H = G.T @ V
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def fit_rci(t, R, t_imu, w_imu, mask):
    G, V = [], []
    for k in range(len(t) - 1):
        if not (mask[k] and mask[k + 1]):
            continue
        if not (0.02 < t[k + 1] - t[k] < 0.06):
            continue
        rv = Rotation.from_matrix(R[k].T @ R[k + 1]).as_rotvec()
        if np.linalg.norm(rv) < np.radians(0.4):
            continue
        G.append(gyro_vec(t_imu, w_imu, t[k], t[k + 1]))
        V.append(rv)
    if len(G) < 20:
        return None, len(G)
    return kabsch(np.array(G), np.array(V)), len(G)


def stats_row(rows, key):
    a = np.array([r[key] for r in rows], dtype=np.float64)
    return round(float(np.median(a)), 3) if len(a) else None


def boot_diff(rows, ka, kb, n_boot=2000):
    cl = np.array([r["session"] for r in rows])
    a = np.array([r[ka] for r in rows])
    b = np.array([r[kb] for r in rows])
    ids = np.unique(cl)
    idx = {c: np.flatnonzero(cl == c) for c in ids}
    d = []
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        sel = np.concatenate([idx[c] for c in pick])
        d.append(np.median(a[sel]) - np.median(b[sel]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


def evaluate(tau_frames: int):
    K, dist = E.load_camera()
    Xb = centered_points()
    homog = np.c_[Xb, np.ones(len(Xb))].T
    rows_all = []
    per_session = {}
    for session in SESSIONS:
        t, R, p, delay = load_raw(session)
        t_imu, w_imu = E.load_gyro(session)
        if len(t) < 40:
            continue
        # 2-fold block cross-validation of the per-session camera-IMU rotation (10 s blocks)
        block = np.floor((t - t[0]) / 10.0).astype(int)
        fit = {}
        for fold in (0, 1):
            Rf, nfit = fit_rci(t, R, t_imu, w_imu, (block % 2) != fold)  # fit on the other half
            fit[fold] = Rf
        n_session = 0
        for k in range(1, len(t) - 1):
            dt_prev = t[k] - t[k - 1]
            if not (0.02 < dt_prev < 0.12):
                continue
            target = tau_frames / 30.0
            j = None
            for jj in range(k + 1, min(k + 8, len(t))):
                if abs((t[jj] - t[k]) - target) < 0.010:
                    j = jj
                    break
            if j is None:
                continue
            tau = t[j] - t[k]
            R0, p0, R1, p1, Rp, pp = R[k], p[k], R[j], p[j], R[k - 1], p[k - 1]
            X_true = (R1.T @ (Xb - p1).T).T
            if not (0.04 < float(np.median(X_true[:, 2])) < 0.45):
                continue
            uv_true = E.project(X_true, K, dist)
            if (uv_true[:, 0] < 10).any() or (uv_true[:, 0] > 1270).any() or (uv_true[:, 1] < 10).any() or (uv_true[:, 1] > 710).any():
                continue
            om = float(np.linalg.norm(w_imu[np.searchsorted(t_imu, t[k]): np.searchsorted(t_imu, t[j])], axis=1).mean()) \
                if np.searchsorted(t_imu, t[j]) > np.searchsorted(t_imu, t[k]) else float("nan")
            if not np.isfinite(om):
                continue
            X_del = (R0.T @ (Xb - p0).T).T
            uv_del = E.project(X_del, K, dist)
            # coupled: previous camera step, extrapolated in time
            dR_prev = R0.T @ Rp
            dt_prev_cam = R0.T @ (pp - p0)
            sc = tau / dt_prev
            Rstep = Rotation.from_rotvec(Rotation.from_matrix(dR_prev).as_rotvec() * sc).as_matrix()
            X_cpl = (Rstep @ X_del.T).T + sc * dt_prev_cam
            uv_cpl = E.project(X_cpl, K, dist)

            def viewer(Rci):
                dR = delta_R(t_imu, w_imu, t[k], t[j], Rci)
                v_pos = (p0 - pp) / dt_prev
                R_pred = R0 @ dR
                p_pred = p0 + v_pos * tau
                X = (R_pred.T @ (Xb - p_pred).T).T
                uv = E.project(X, K, dist)
                H = K @ dR.T @ np.linalg.inv(K)
                return X, uv, E.apply_H(H, uv_del), dR

            Rci_cv = fit[int(block[k] % 2)]
            X_pap, uv_pap, uv_atw_pap, dR_pap = viewer(R_CI_PAPER)
            if Rci_cv is None:
                continue
            X_cv, uv_cv, uv_atw_cv, dR_cv = viewer(Rci_cv)
            rot_true = R0.T @ R1
            row = dict(
                session=session, omega=om, tau=tau,
                delayed_px=E.px_err(uv_del, uv_true), delayed_mm=E.add_mm(X_del, X_true),
                coupled_px=E.px_err(uv_cpl, uv_true), coupled_mm=E.add_mm(X_cpl, X_true),
                atw_cv_px=E.px_err(uv_atw_cv, uv_true), atw_paper_px=E.px_err(uv_atw_pap, uv_true),
                viewer_cv_px=E.px_err(uv_cv, uv_true), viewer_cv_mm=E.add_mm(X_cv, X_true),
                viewer_paper_px=E.px_err(uv_pap, uv_true), viewer_paper_mm=E.add_mm(X_pap, X_true),
                rot_err_cv_deg=float(np.degrees(Rotation.from_matrix(rot_true.T @ dR_cv).magnitude())),
                rot_err_paper_deg=float(np.degrees(Rotation.from_matrix(rot_true.T @ dR_pap).magnitude())),
                rot_true_deg=float(np.degrees(Rotation.from_matrix(rot_true).magnitude())),
            )
            if max(row[k_] for k_ in ("delayed_px", "coupled_px", "atw_cv_px", "viewer_cv_px", "viewer_paper_px")) > 260.0:
                continue
            rows_all.append(row)
            n_session += 1
        per_session[session] = n_session
    return rows_all, per_session


def summarise(rows):
    res = {"n": len(rows), "sessions": len({r["session"] for r in rows})}
    bins = {"[0.00,0.05)": (0, 0.05), "[0.05,0.20)": (0.05, 0.20), "[0.20,0.50)": (0.2, 0.5), "[0.50,1.00)": (0.5, 1.0),
            "[1.00,2.00)": (1.0, 2.0), "[2.00,10.0)": (2.0, 10.0), "B1:[0,0.5)": (0, 0.5), "B2:[0.5,1)": (0.5, 1.0),
            "B3:[1,2)": (1.0, 2.0), "B4:[2,10)": (2.0, 10.0), "all": (0, 10)}
    for name, (lo, hi) in bins.items():
        sel = [r for r in rows if lo <= r["omega"] < hi]
        if len(sel) < 5:
            res[name] = {"n": len(sel)}
            continue
        cell = {"n": len(sel), "n_sessions": len({r["session"] for r in sel})}
        for k in ("delayed_px", "coupled_px", "atw_cv_px", "atw_paper_px", "viewer_cv_px", "viewer_paper_px",
                  "delayed_mm", "coupled_mm", "viewer_cv_mm", "viewer_paper_mm", "rot_err_cv_deg", "rot_err_paper_deg",
                  "rot_true_deg"):
            cell[k] = stats_row(sel, k)
        if len({r["session"] for r in sel}) >= 3:
            cell["delayed_minus_viewer_cv_px_ci95"] = boot_diff(sel, "delayed_px", "viewer_cv_px")
            cell["coupled_minus_viewer_cv_px_ci95"] = boot_diff(sel, "coupled_px", "viewer_cv_px")
        res[name] = cell
    return res


def main():
    out = {}
    for tf in (1, 2, 3, 4):
        rows, per_session = evaluate(tf)
        out[f"{tf}_frames"] = {"summary": summarise(rows), "pairs_per_session": per_session}
        s = out[f"{tf}_frames"]["summary"]
        print(f"tau={tf} frames: n={s['n']} sessions={s['sessions']}", flush=True)
        for name in ("B1:[0,0.5)", "B2:[0.5,1)", "B3:[1,2)", "B4:[2,10)", "all"):
            c = s.get(name, {})
            if c.get("n", 0) >= 5:
                print("  ", name, {k: c[k] for k in ("n", "delayed_px", "coupled_px", "atw_cv_px", "viewer_cv_px", "viewer_paper_px")}, flush=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
