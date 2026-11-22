#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real viewer-overlay test on the three gtraj sessions.

The fixed 4K camera is the global frame G. The moving viewer is the left
stereo camera C0. The target is the GP050 board B. USB gyroscope samples are
degrees per second. Display delay is 60 ms, the measured cam0 USB delivery
delay. Stage I is given the fused exposure-time model pose; this script tests
only the display-time update.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

DATA = Path(r"E:\EGO_Mo\datasets")
OUT = Path(r"D:\ISMAR2026\egomotion-registration\validation\real_viewer_overlay.json")
SESSIONS = [
    "gtraj_20261002_002134",
    "gtraj_20261002_003129",
    "gtraj_20261002_024309",
]
TAU = 0.060
PAST = 0.050
STRIDE = 8
R_CI = np.array([
    [0.007668, -0.946526, -0.322536],
    [-0.003137, 0.322521, -0.946557],
    [0.999966, 0.008270, -0.000496],
], dtype=np.float64)
BINS = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.00)]
PAPER_BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 10.0)]


def bin_name(lo: float, hi: float) -> str:
    return f"[{lo:.2f},{hi:.2f})" if hi <= 1.0 else f"[{lo:.1f},{hi:.1f})"


def stats(vals: list[float]) -> dict:
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
    if len(a) == 0:
        return {"n": 0}
    q25, q50, q75 = np.percentile(a, [25, 50, 75])
    return {
        "n": int(len(a)),
        "median": round(float(q50), 3),
        "iqr": round(float(q75 - q25), 3),
        "mean": round(float(a.mean()), 3),
        "sd": round(float(a.std(ddof=0)), 3),
    }


def skew(w: np.ndarray) -> np.ndarray:
    x, y, z = w
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    rho, theta = xi[:3], xi[3:]
    th = float(np.linalg.norm(theta))
    R = Rotation.from_rotvec(theta).as_matrix()
    W = skew(theta)
    if th < 1e-10:
        V = np.eye(3)
    else:
        V = np.eye(3) + (1 - np.cos(th)) / th**2 * W + (th - np.sin(th)) / th**3 * (W @ W)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    theta = Rotation.from_matrix(R).as_rotvec()
    th = float(np.linalg.norm(theta))
    W = skew(theta)
    if th < 1e-10:
        Vinv = np.eye(3)
    else:
        half = 0.5 * th
        cot = half / np.tan(half) if abs(np.tan(half)) > 1e-8 else 1.0
        Vinv = np.eye(3) - 0.5 * W + (1 - cot) / th**2 * (W @ W)
    return np.r_[Vinv @ t, theta]


def se3(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def invert(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def object_points() -> np.ndarray:
    x, y = np.meshgrid(np.arange(11), np.arange(8))
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float64) * 0.003


def load_camera():
    cal = json.loads((DATA / "calib_intrinsics_20260922_123120" / "camera_calibration_cam0.json").read_text(encoding="utf-8"))
    K = np.asarray(cal["camera_matrix"], np.float64)
    dist = np.asarray(cal["distortion_coefficients"], np.float64).reshape(-1)
    return K, dist


def load_fused(session: str):
    rows = []
    for path in sorted((DATA / session / "fused_gt").glob("segment_*_fused.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    rows.sort(key=lambda r: float(r["stamp"]))
    kept = [rows[0]]
    for rec in rows[1:]:
        if float(rec["stamp"]) > float(kept[-1]["stamp"]) + 1e-4:
            kept.append(rec)
    rows = kept
    ts = np.asarray([float(r["stamp"]) for r in rows], dtype=np.float64)
    def pack(prefix):
        q = np.asarray([[float(r[f"{prefix}_q{a}"]) for a in "xyzw"] for r in rows], dtype=np.float64)
        p = np.asarray([[float(r[f"{prefix}_t{a}"]) for a in "xyz"] for r in rows], dtype=np.float64)
        return Rotation.from_quat(q), p
    Rc, pc = pack("C0")
    Rb, pb = pack("B")
    cuts = np.flatnonzero(np.diff(ts) > 0.05)
    bounds = np.r_[0, cuts + 1, len(ts)]
    segments = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 30:
            continue
        segments.append({
            "t": ts[a:b],
            "Rc": Slerp(ts[a:b], Rc[a:b]),
            "pc": pc[a:b],
            "Rb": Slerp(ts[a:b], Rb[a:b]),
            "pb": pb[a:b],
        })
    return segments


def load_gyro(session: str):
    t, w = [], []
    with (DATA / session / "imu_stream.csv").open(newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            t.append(float(rec["timestamp"]))
            w.append([float(rec[f"gyro_{a}"]) for a in "xyz"])
    t = np.asarray(t, dtype=np.float64)
    w = np.deg2rad(np.asarray(w, dtype=np.float64))
    order = np.argsort(t)
    return t[order], w[order]


class Series:
    def __init__(self, seg):
        self.t = seg["t"]
        self.Rc = seg["Rc"]
        self.pc = seg["pc"]
        self.Rb = seg["Rb"]
        self.pb = seg["pb"]

    def inside(self, tq: float) -> bool:
        return bool(self.t[0] <= tq <= self.t[-1])

    def pose(self, which: str, tq: float):
        i = int(np.searchsorted(self.t, tq))
        i = min(max(i, 1), len(self.t) - 1)
        a = (tq - self.t[i - 1]) / max(self.t[i] - self.t[i - 1], 1e-9)
        a = float(np.clip(a, 0.0, 1.0))
        if which == "c":
            R = self.Rc([tq]).as_matrix()[0]
            p = (1 - a) * self.pc[i - 1] + a * self.pc[i]
        else:
            R = self.Rb([tq]).as_matrix()[0]
            p = (1 - a) * self.pb[i - 1] + a * self.pb[i]
        return R, p


def delta_R(t_imu, w_imu, t0: float, t1: float) -> np.ndarray:
    i0 = int(np.searchsorted(t_imu, t0))
    i1 = int(np.searchsorted(t_imu, t1))
    if i1 <= i0:
        i0 = max(0, min(i0, len(w_imu) - 1))
        return Rotation.from_rotvec((R_CI @ w_imu[i0]) * (t1 - t0)).as_matrix()
    ts = t_imu[i0:i1]
    wc = (R_CI @ w_imu[i0:i1].T).T
    if len(ts) == 1:
        return Rotation.from_rotvec(wc[0] * (t1 - t0)).as_matrix()
    R = np.eye(3)
    dts = np.diff(ts, append=t1)
    dts[0] += ts[0] - t0
    for wi, dt in zip(wc, dts):
        if dt <= 0:
            continue
        R = R @ Rotation.from_rotvec(wi * dt).as_matrix()
    return R


def omega_mean(t_imu, w_imu, t0: float, t1: float) -> float:
    i0 = int(np.searchsorted(t_imu, t0))
    i1 = int(np.searchsorted(t_imu, t1))
    if i1 <= i0:
        return float("nan")
    return float(np.linalg.norm(w_imu[i0:i1], axis=1).mean())


def project(X: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    uv, _ = cv2.projectPoints(X.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    return uv.reshape(-1, 2)


def add_mm(Xa: np.ndarray, Xb: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(Xa - Xb, axis=1)) * 1000.0)


def px_err(ua: np.ndarray, ub: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(ua - ub, axis=1)))


def apply_H(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    hom = np.c_[uv, np.ones(len(uv))]
    y = (H @ hom.T).T
    return y[:, :2] / y[:, 2:3]


def choose_multiply(segments, t_imu, w_imu) -> str:
    scores = {"right": [], "left": []}
    rng = np.random.default_rng(0)
    for seg in segments:
        s = Series(seg)
        if s.t[-1] - s.t[0] < TAU + 0.2:
            continue
        draw = rng.uniform(s.t[0] + 0.1, s.t[-1] - TAU, size=40)
        for t0 in draw:
            R0, _ = s.pose("c", float(t0))
            R1, _ = s.pose("c", float(t0 + TAU))
            dR = delta_R(t_imu, w_imu, float(t0), float(t0 + TAU))
            scores["right"].append(Rotation.from_matrix(R1.T @ (R0 @ dR)).magnitude())
            scores["left"].append(Rotation.from_matrix(R1.T @ (dR @ R0)).magnitude())
    med = {k: float(np.median(v)) for k, v in scores.items() if v}
    winner = min(med, key=med.get)
    return winner, {k: round(np.degrees(v), 4) for k, v in med.items()}


def evaluate_session(session: str, K, dist, Xb: np.ndarray) -> dict:
    segments = load_fused(session)
    t_imu, w_imu = load_gyro(session)
    side, side_err = choose_multiply(segments, t_imu, w_imu)
    acc = {bin_name(lo, hi): {m: [] for m in ("delayed_mm", "delayed_px", "coupled_mm", "coupled_px", "atw_px", "viewer_rot_mm", "viewer_rot_px", "viewer_se3_mm", "viewer_se3_px", "gyro_rot_deg")} for lo, hi in BINS + PAPER_BINS}
    w_all = []
    for seg in segments:
        s = Series(seg)
        times = s.t[::STRIDE]
        for t0 in times:
            t0 = float(t0)
            if not (s.inside(t0 - PAST) and s.inside(t0 + TAU)):
                continue
            w = omega_mean(t_imu, w_imu, t0, t0 + TAU)
            if not np.isfinite(w):
                continue
            w_all.append(w)
            Rc0, pc0 = s.pose("c", t0)
            Rc1, pc1 = s.pose("c", t0 + TAU)
            Rp, pp = s.pose("c", t0 - PAST)
            Rb0, pb0 = s.pose("b", t0)
            Rb1, pb1 = s.pose("b", t0 + TAU)
            Rpast, ppast = s.pose("b", t0 - PAST)
            T_c0 = se3(Rc0, pc0)
            T_c1 = se3(Rc1, pc1)
            T_cp = se3(Rp, pp)
            T_b0 = se3(Rb0, pb0)
            T_b1 = se3(Rb1, pb1)
            T_bp = se3(Rpast, ppast)
            X_true = (invert(T_c1) @ T_b1 @ np.c_[Xb, np.ones(len(Xb))].T).T[:, :3]
            if np.median(X_true[:, 2]) < 0.03:
                continue
            uv_true = project(X_true, K, dist)

            X_del = (invert(T_c0) @ T_b0 @ np.c_[Xb, np.ones(len(Xb))].T).T[:, :3]
            uv_del = project(X_del, K, dist)

            Tcm0 = invert(T_c0) @ T_b0
            Tcmp = invert(T_cp) @ T_bp
            xi = se3_log(invert(Tcmp) @ Tcm0) * (TAU / PAST)
            X_cpl = (Tcm0 @ se3_exp(xi) @ np.c_[Xb, np.ones(len(Xb))].T).T[:, :3]
            uv_cpl = project(X_cpl, K, dist)

            dR = delta_R(t_imu, w_imu, t0, t0 + TAU)
            R_hat = Rc0 @ dR if side == "right" else dR @ Rc0
            v = (pc0 - pp) / PAST
            p_rot = pc0
            p_se3 = pc0 + v * TAU
            X_rot = (invert(se3(R_hat, p_rot)) @ T_b0 @ np.c_[Xb, np.ones(len(Xb))].T).T[:, :3]
            X_se3 = (invert(se3(R_hat, p_se3)) @ T_b0 @ np.c_[Xb, np.ones(len(Xb))].T).T[:, :3]
            uv_rot = project(X_rot, K, dist)
            uv_se3 = project(X_se3, K, dist)

            Rcw0 = Rc0.T
            Rcw_hat = R_hat.T
            H = K @ Rcw_hat @ Rcw0.T @ np.linalg.inv(K)
            uv_atw = apply_H(H, uv_del)
            gyro_deg = float(np.degrees(Rotation.from_matrix(Rc1.T @ R_hat).magnitude()))
            if min(np.median(X_del[:, 2]), np.median(X_cpl[:, 2]), np.median(X_rot[:, 2]), np.median(X_se3[:, 2])) < 0.03:
                continue

            item = {
                "delayed_mm": add_mm(X_del, X_true),
                "delayed_px": px_err(uv_del, uv_true),
                "coupled_mm": add_mm(X_cpl, X_true),
                "coupled_px": px_err(uv_cpl, uv_true),
                "atw_px": px_err(uv_atw, uv_true),
                "viewer_rot_mm": add_mm(X_rot, X_true),
                "viewer_rot_px": px_err(uv_rot, uv_true),
                "viewer_se3_mm": add_mm(X_se3, X_true),
                "viewer_se3_px": px_err(uv_se3, uv_true),
                "gyro_rot_deg": gyro_deg,
            }
            if max(item["delayed_px"], item["coupled_px"], item["atw_px"], item["viewer_rot_px"], item["viewer_se3_px"]) > 200.0:
                continue
            for edges in (BINS, PAPER_BINS):
                for lo, hi in edges:
                    if lo <= w < hi:
                        slot = acc[bin_name(lo, hi)]
                        for k, v in item.items():
                            slot[k].append(v)
                        break
    w_all = np.asarray(w_all, dtype=np.float64)
    summary = {key: {m: stats(vs) for m, vs in slot.items()} for key, slot in acc.items()}
    return {
        "session": session,
        "multiply": side,
        "gyro_convention_error_deg": side_err,
        "n": int(len(w_all)),
        "omega_rad_s": {
            "median": round(float(np.median(w_all)), 4) if len(w_all) else None,
            "p90": round(float(np.percentile(w_all, 90)), 4) if len(w_all) else None,
            "max": round(float(w_all.max()), 4) if len(w_all) else None,
        },
        "bins": summary,
    }, acc


def corrected_consecutive() -> dict:
    src = json.loads(Path(r"D:\ISMAR2026\egomotion-registration\validation\consecutive_pairs_by_omega.json").read_text(encoding="utf-8"))
    rows = []
    cache = {}
    for row in src["rows"]:
        session = row["session"]
        if session not in cache:
            summary = json.loads((DATA / session / "capture_summary.json").read_text(encoding="utf-8"))
            delay = float(((summary.get("sync") or {}).get("cameras") or {}).get("cam0", {}).get("delay_usb_s") or 0.0)
            times = np.loadtxt(DATA / session / "cam0" / "times.txt") - delay
            cache[session] = (times, *load_gyro(session))
        times, t_imu, w_imu = cache[session]
        i = int(row["i"])
        if i + 1 >= len(times):
            continue
        w = omega_mean(t_imu, w_imu, float(times[i]), float(times[i + 1]))
        rows.append({
            "session": session,
            "i": i,
            "omega_rad_s": round(w, 4),
            "pi3_rot_err_deg": row["pi3_rot_err_deg"],
            "reloc3r_rot_err_deg": row["reloc3r_rot_err_deg"],
            "stored_omega_rad_s": row["omega_rad_s"],
        })
    out = {}
    for lo, hi in BINS:
        key = bin_name(lo, hi)
        sel = [r for r in rows if lo <= r["omega_rad_s"] < hi]
        out[key] = {
            "n": len(sel),
            "pi3_rot_err_deg": stats([r["pi3_rot_err_deg"] for r in sel]),
            "reloc3r_rot_err_deg": stats([r["reloc3r_rot_err_deg"] for r in sel]),
        }
    return {"n": len(rows), "omega_max": round(max(r["omega_rad_s"] for r in rows), 4), "bins": out}


def pool_raw(raws: list[dict]) -> dict:
    keys = raws[0]
    out = {}
    for key in keys:
        out[key] = {}
        metrics = keys[key]
        for m in metrics:
            vals = []
            for raw in raws:
                vals.extend(raw[key][m])
            out[key][m] = stats(vals)
    return out


def main() -> None:
    K, dist = load_camera()
    Xb = object_points()
    sessions = []
    raws = []
    for session in SESSIONS:
        print(session, flush=True)
        got, raw = evaluate_session(session, K, dist, Xb)
        sessions.append(got)
        raws.append(raw)
        print(" ", got["omega_rad_s"], got["multiply"], got["gyro_convention_error_deg"], "n", got["n"], flush=True)
        for key in ("[0.00,0.05)", "[0.20,1.00)"):
            row = got["bins"][key]
            print(f"  {key} delayed_px", row["delayed_px"], "viewer_px", row["viewer_se3_px"], "atw", row["atw_px"], flush=True)
    report = {
        "protocol": {
            "viewer": "cam0 left endoscope, 1280x720",
            "world": "fixed 4K camera frame G",
            "target": "GP050 board B, 11x8 corners, 3 mm",
            "tau_s": TAU,
            "tau_source": "60 ms, matching the measured cam0 USB delivery delay (55-59 ms). Not an overlay-write timestamp.",
            "gyro_unit": "imu_stream gyro_* converted from deg/s to rad/s",
            "stage_I": "fused exposure-time board pose, not EgoAnchorNet",
            "viewer_rot": "gyroscope body increment, camera translation held",
            "viewer_se3": "gyroscope increment plus causal translation from the previous 50 ms",
            "atw": "K R_hat R_exposure^{-1} K^{-1} applied to the exposure-time overlay",
            "coupled": "SE3 extrapolation of T_cm from the previous 50 ms",
        },
        "sessions": sessions,
        "pooled": pool_raw(raws),
        "consecutive_relabel": corrected_consecutive(),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
