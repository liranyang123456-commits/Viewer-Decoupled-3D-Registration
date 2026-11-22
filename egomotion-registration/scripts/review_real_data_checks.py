#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reviewer-requested checks on the recorded dual-board sessions (S1-S3).

(1) tau sweep: the display-update test of eval_real_viewer_overlay.py for several registration lags,
    with a cluster bootstrap (clusters = recording segments) for the median improvement.
(2) reference independence: the fused reference rotates with the IMU's own orientation output, so we
    compare the gyroscope increments with the *vision-only* 4K board poses (global_pose_gt.csv),
    estimate the vision noise floor from near-static pairs, and repeat the overlay test with
    vision-only poses at the native 4K timestamps.

Writes validation/review_real_checks.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_real_viewer_overlay as E  # noqa: E402

OUT = Path(r"D:\ISMAR2026\egomotion-registration\validation\review_real_checks.json")
DATA = E.DATA
rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------- helpers
def med(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.median(a)) if len(a) else float("nan")


def cluster_bootstrap_diff(vals_a, vals_b, clusters, n_boot=2000):
    """95% CI of median(a) - median(b), resampling whole clusters."""
    vals_a, vals_b = np.asarray(vals_a), np.asarray(vals_b)
    clusters = np.asarray(clusters)
    ids = np.unique(clusters)
    idx_by = {c: np.flatnonzero(clusters == c) for c in ids}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        sel = np.concatenate([idx_by[c] for c in pick])
        diffs.append(np.median(vals_a[sel]) - np.median(vals_b[sel]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


# ----------------------------------------------------------------------------- (1) tau sweep
def eval_session_pairs(session, K, dist, Xb, tau):
    """Same computation as eval_real_viewer_overlay.evaluate_session, but returns per-pair rows."""
    E.TAU = tau
    segments = E.load_fused(session)
    t_imu, w_imu = E.load_gyro(session)
    side, _ = E.choose_multiply(segments, t_imu, w_imu)
    rows = []
    homog = np.c_[Xb, np.ones(len(Xb))].T
    for si, seg in enumerate(segments):
        s = E.Series(seg)
        for t0 in s.t[:: E.STRIDE]:
            t0 = float(t0)
            if not (s.inside(t0 - E.PAST) and s.inside(t0 + tau)):
                continue
            w = E.omega_mean(t_imu, w_imu, t0, t0 + tau)
            if not np.isfinite(w):
                continue
            Rc0, pc0 = s.pose("c", t0)
            Rc1, pc1 = s.pose("c", t0 + tau)
            Rp, pp = s.pose("c", t0 - E.PAST)
            Rb0, pb0 = s.pose("b", t0)
            Rb1, pb1 = s.pose("b", t0 + tau)
            Rbp, pbp = s.pose("b", t0 - E.PAST)
            T_c0, T_c1, T_cp = E.se3(Rc0, pc0), E.se3(Rc1, pc1), E.se3(Rp, pp)
            T_b0, T_b1, T_bp = E.se3(Rb0, pb0), E.se3(Rb1, pb1), E.se3(Rbp, pbp)
            X_true = (E.invert(T_c1) @ T_b1 @ homog).T[:, :3]
            if np.median(X_true[:, 2]) < 0.03:
                continue
            uv_true = E.project(X_true, K, dist)
            X_del = (E.invert(T_c0) @ T_b0 @ homog).T[:, :3]
            Tcm0 = E.invert(T_c0) @ T_b0
            Tcmp = E.invert(T_cp) @ T_bp
            xi = E.se3_log(E.invert(Tcmp) @ Tcm0) * (tau / E.PAST)
            X_cpl = (Tcm0 @ E.se3_exp(xi) @ homog).T[:, :3]
            dR = E.delta_R(t_imu, w_imu, t0, t0 + tau)
            R_hat = Rc0 @ dR if side == "right" else dR @ Rc0
            v = (pc0 - pp) / E.PAST
            X_se3 = (E.invert(E.se3(R_hat, pc0 + v * tau)) @ T_b0 @ homog).T[:, :3]
            if min(np.median(X_del[:, 2]), np.median(X_cpl[:, 2]), np.median(X_se3[:, 2])) < 0.03:
                continue
            uv_del, uv_cpl, uv_se3 = E.project(X_del, K, dist), E.project(X_cpl, K, dist), E.project(X_se3, K, dist)
            H = K @ R_hat.T @ Rc0 @ np.linalg.inv(K)
            uv_atw = E.apply_H(H, uv_del)
            item = dict(
                session=session, seg=si, omega=w,
                delayed_px=E.px_err(uv_del, uv_true), delayed_mm=E.add_mm(X_del, X_true),
                coupled_px=E.px_err(uv_cpl, uv_true), coupled_mm=E.add_mm(X_cpl, X_true),
                atw_px=E.px_err(uv_atw, uv_true),
                viewer_px=E.px_err(uv_se3, uv_true), viewer_mm=E.add_mm(X_se3, X_true),
            )
            if max(item["delayed_px"], item["coupled_px"], item["atw_px"], item["viewer_px"]) > 200.0:
                continue
            rows.append(item)
    return rows


def tau_sweep():
    K, dist = E.load_camera()
    Xb = E.object_points()
    out = {}
    for tau in (0.033, 0.060, 0.100, 0.130):
        rows = []
        for session in E.SESSIONS:
            rows += eval_session_pairs(session, K, dist, Xb, tau)
        cl = np.array([f"{r['session']}#{r['seg']}" for r in rows])
        w = np.array([r["omega"] for r in rows])
        res = {"n_total": len(rows)}
        for name, lo, hi in (("[0.05,0.20)", 0.05, 0.20), ("[0.20,1.00)", 0.20, 1.00), ("all", 0.0, 10.0)):
            sel = (w >= lo) & (w < hi)
            if sel.sum() < 5:
                res[name] = {"n": int(sel.sum())}
                continue
            cell = {"n": int(sel.sum()), "n_clusters": int(len(np.unique(cl[sel])))}
            for k in ("delayed", "coupled", "atw", "viewer"):
                cell[f"{k}_px"] = round(med([r[f"{k}_px"] for r, m in zip(rows, sel) if m]), 3)
            for k in ("delayed", "coupled", "viewer"):
                cell[f"{k}_mm"] = round(med([r[f"{k}_mm"] for r, m in zip(rows, sel) if m]), 3)
            a = [r["delayed_px"] for r, m in zip(rows, sel) if m]
            b = [r["viewer_px"] for r, m in zip(rows, sel) if m]
            lo_ci, hi_ci = cluster_bootstrap_diff(a, b, cl[sel])
            cell["delayed_minus_viewer_px_ci95"] = [round(lo_ci, 2), round(hi_ci, 2)]
            a = [r["coupled_px"] for r, m in zip(rows, sel) if m]
            lo_ci, hi_ci = cluster_bootstrap_diff(a, b, cl[sel])
            cell["coupled_minus_viewer_px_ci95"] = [round(lo_ci, 2), round(hi_ci, 2)]
            a = [r["delayed_mm"] for r, m in zip(rows, sel) if m]
            b2 = [r["viewer_mm"] for r, m in zip(rows, sel) if m]
            lo_ci, hi_ci = cluster_bootstrap_diff(a, b2, cl[sel])
            cell["delayed_minus_viewer_mm_ci95"] = [round(lo_ci, 3), round(hi_ci, 3)]
            res[name] = cell
        out[f"{int(round(tau * 1000))}ms"] = res
        print("tau", tau, json.dumps(res["[0.20,1.00)"]), flush=True)
    return out


# ----------------------------------------------------------------------------- (2) vision-only check
def load_vision(session):
    rows = list(csv.DictReader((DATA / session / "global_pose_gt.csv").open(newline="", encoding="utf-8")))
    t = np.array([float(r["stamp"]) for r in rows])

    def fl(x):
        return float(x) if x not in ("", None) else float("nan")

    def mat(prefix):
        R = np.array([[fl(r[f"{prefix}_r{i}{j}"]) for i in range(3) for j in range(3)] for r in rows]).reshape(-1, 3, 3)
        p = np.array([[fl(r[f"{prefix}_t{a}_mm"]) for a in "xyz"] for r in rows]) / 1000.0
        return R, p

    RA, pA = mat("G_A")
    RB, pB = mat("G_B")
    # left-camera pose from the rig board pose and the fixed rig extrinsic (same one the fused GT uses)
    T_C0_A = np.asarray(json.loads((DATA.parent / "datasets" / "global_rig_state.json").read_text(encoding="utf-8"))
                        ["T_C0_A"]["matrix"], dtype=np.float64) if (DATA.parent / "datasets" / "global_rig_state.json").exists() \
        else np.asarray(json.loads((DATA / "global_rig_state.json").read_text(encoding="utf-8"))["T_C0_A"]["matrix"])
    T_A_C0 = E.invert(T_C0_A)
    RC = RA @ T_A_C0[:3, :3]
    pC = (RA @ T_A_C0[:3, 3]) + pA
    okA = np.array([r["A_found"] == "1" and fl(r["A_reproj_px"]) < 3.0 for r in rows])
    okB = np.array([r["B_found"] == "1" and fl(r["B_reproj_px"]) < 3.0 for r in rows])
    okA &= np.isfinite(RA).reshape(len(rows), -1).all(1) & np.isfinite(pA).all(1)
    okB &= np.isfinite(RB).reshape(len(rows), -1).all(1) & np.isfinite(pB).all(1)
    return dict(t=t, RC=RC, pC=pC, RB=RB, pB=pB, okA=okA, okB=okB)


def ang(R):
    return float(np.degrees(Rotation.from_matrix(R).magnitude()))


def vision_only_check():
    K, dist = E.load_camera()
    Xb = E.object_points()
    homog = np.c_[Xb, np.ones(len(Xb))].T
    out = {}
    for session in E.SESSIONS:
        V = load_vision(session)
        t_imu, w_imu = E.load_gyro(session)
        t, ok = V["t"], V["okA"] & V["okB"]
        # consecutive usable 4K frames
        idx = [i for i in range(1, len(t) - 1) if ok[i - 1] and ok[i] and ok[i + 1]
               and 0.045 < t[i + 1] - t[i] < 0.110 and 0.045 < t[i] - t[i - 1] < 0.110]
        # time offset between 4K host stamps and the IMU clock: scan, keep the best
        best = None
        for d in np.arange(0.0, 0.50, 0.01):
            errs = []
            for i in idx[::3]:
                dRv = V["RC"][i].T @ V["RC"][i + 1]
                dRg = E.delta_R(t_imu, w_imu, t[i] - d, t[i + 1] - d)
                errs.append(ang(dRv.T @ dRg))
            m = med(errs)
            if best is None or m < best[1]:
                best = (float(d), m)
        d_star = best[0]
        # side convention (right/left multiplication) as in the main evaluation
        errs_r, errs_l, w_pair, noise = [], [], [], []
        for i in idx:
            dRv = V["RC"][i].T @ V["RC"][i + 1]  # body-frame increment from vision
            dRg = E.delta_R(t_imu, w_imu, t[i] - d_star, t[i + 1] - d_star)
            errs_r.append(ang(dRv.T @ dRg))
            errs_l.append(ang(dRv.T @ (V["RC"][i].T @ dRg @ V["RC"][i])))
            w_pair.append(E.omega_mean(t_imu, w_imu, t[i] - d_star, t[i + 1] - d_star))
        errs_r, errs_l, w_pair = np.array(errs_r), np.array(errs_l), np.array(w_pair)
        use_right = med(errs_r) <= med(errs_l)
        errs = errs_r if use_right else errs_l
        # noise floor: near-static pairs (gyro says no rotation) -> vision relative rotation itself
        static = [ang(V["RC"][i].T @ V["RC"][i + 1]) for i, w in zip(idx, w_pair) if np.isfinite(w) and w < 0.02]
        # vision-only overlay test at native 4K timestamps
        rows = []
        for i in idx:
            tau = t[i + 1] - t[i]
            past = t[i] - t[i - 1]
            T = lambda R, p: E.se3(R, p)
            Tc0, Tc1, Tcp = T(V["RC"][i], V["pC"][i]), T(V["RC"][i + 1], V["pC"][i + 1]), T(V["RC"][i - 1], V["pC"][i - 1])
            Tb0, Tb1, Tbp = T(V["RB"][i], V["pB"][i]), T(V["RB"][i + 1], V["pB"][i + 1]), T(V["RB"][i - 1], V["pB"][i - 1])
            X_true = (E.invert(Tc1) @ Tb1 @ homog).T[:, :3]
            if np.median(X_true[:, 2]) < 0.03:
                continue
            uv_true = E.project(X_true, K, dist)
            X_del = (E.invert(Tc0) @ Tb0 @ homog).T[:, :3]
            Tcm0, Tcmp = E.invert(Tc0) @ Tb0, E.invert(Tcp) @ Tbp
            xi = E.se3_log(E.invert(Tcmp) @ Tcm0) * (tau / past)
            X_cpl = (Tcm0 @ E.se3_exp(xi) @ homog).T[:, :3]
            dRg = E.delta_R(t_imu, w_imu, t[i] - d_star, t[i + 1] - d_star)
            R_hat = V["RC"][i] @ dRg if use_right else dRg @ V["RC"][i]
            v = (V["pC"][i] - V["pC"][i - 1]) / past
            X_se3 = (E.invert(E.se3(R_hat, V["pC"][i] + v * tau)) @ Tb0 @ homog).T[:, :3]
            if min(np.median(X_del[:, 2]), np.median(X_cpl[:, 2]), np.median(X_se3[:, 2])) < 0.03:
                continue
            uv_del, uv_cpl, uv_se3 = E.project(X_del, K, dist), E.project(X_cpl, K, dist), E.project(X_se3, K, dist)
            H = K @ R_hat.T @ V["RC"][i] @ np.linalg.inv(K)
            uv_atw = E.apply_H(H, uv_del)
            om = E.omega_mean(t_imu, w_imu, t[i] - d_star, t[i + 1] - d_star)
            r = dict(omega=om, tau=tau, delayed=E.px_err(uv_del, uv_true), coupled=E.px_err(uv_cpl, uv_true),
                     atw=E.px_err(uv_atw, uv_true), viewer=E.px_err(uv_se3, uv_true))
            if max(r["delayed"], r["coupled"], r["atw"], r["viewer"]) < 300:
                rows.append(r)
        w_r = np.array([r["omega"] for r in rows])
        cell = {}
        for name, lo, hi in (("[0.05,0.20)", 0.05, 0.20), ("[0.20,1.00)", 0.20, 1.00), ("all", 0.0, 10.0)):
            sel = (w_r >= lo) & (w_r < hi)
            if sel.sum() >= 5:
                cell[name] = {"n": int(sel.sum())} | {k: round(med([r[k] for r, m in zip(rows, sel) if m]), 2)
                                                     for k in ("delayed", "coupled", "atw", "viewer")}
        hi_mask = np.isfinite(w_pair) & (w_pair >= 0.20)
        out[session] = {
            "pairs": len(idx), "time_offset_s": d_star, "convention": "right" if use_right else "left",
            "gyro_vs_vision_rot_err_deg": {
                "all_median": round(med(errs), 3), "all_p90": round(float(np.percentile(errs, 90)), 3),
                "omega_ge_0.2_median": round(med(errs[hi_mask]), 3) if hi_mask.sum() else None,
                "omega_ge_0.2_n": int(hi_mask.sum()),
                "static_noise_floor_median": round(med(static), 3) if static else None, "static_n": len(static),
            },
            "vision_only_overlay_px": cell,
        }
        print(session, json.dumps(out[session]), flush=True)
    return out


def main():
    res = {"tau_sweep": tau_sweep(), "vision_only": vision_only_check()}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
