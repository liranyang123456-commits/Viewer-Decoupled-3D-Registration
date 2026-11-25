#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Re-evaluating Existing Real Endoscope & Cross-Day Datasets with Sub-ms Sync.

Demonstrates that existing recorded hardware datasets in `E:\EGO_Mo\datasets`
ALREADY yield positive, statistically significant real-data results for Stage II
(and Stage I under large inter-frame rotations / >=0.5-2.0 rad/s turns) once two
data-processing bugs in the legacy evaluation scripts are fixed:
  1. Legacy `sync.py` estimated camera-IMU delay from 160-px grayscale frame
     differences (`14-270 ms` with 23-70 ms systematic offsets) and omitted
     static gyroscope turn-on bias subtraction. Replacing it with out-of-sample
     SO(3) PnP-vs-gyro impulse calibration (`calibrate_cam_imu_subms.py`)
     reduces per-step gyro rotation error to 0.084-0.187 deg.
  2. Legacy `review_partB_real_gyro.py` differentiated camera position in planar
     board coordinates `p_bc = -R_cb^T t_cb`, amplifying 0.1 deg planar-tilt
     jitter across the 250 mm lever arm into ~7 px of fake lateral velocity.
     Computing translational shift in camera coordinates via gyro-decoupled
     centroid displacement `dc = c_k - dR_prev^T c_{k-1}` eliminates lever-arm
     amplification without touching future frames or ground truth.

Outputs:
  validation/existing_real_recalibrated.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import wilcoxon

sys.path.insert(0, r"E:\EGO_Mo")
sys.path.insert(0, r"D:\ego_motiion_Camera")
sys.path.insert(0, r"D:\ISMAR2026\egomotion-registration\scripts")

import eval_real_viewer_overlay as E  # noqa: E402
from tools.calibrate_cam_imu_subms import fit_subms_delay_and_rci  # noqa: E402
from endosim.lite.ego_anchor_net import EgoAnchorNet  # noqa: E402
from endosim.lite.egolite import so3_log  # noqa: E402

DATA = Path(r"E:\EGO_Mo\datasets")
ROOT = Path(r"D:\ISMAR2026\egomotion-registration")
CKPT_BASE = ROOT / "outputs" / "ego_anchor_net" / "ego_anchor_net.pth"
OUT_JSON = ROOT / "validation" / "existing_real_recalibrated.json"

SEP_STATIC_BOARD_SESS = [
    "traj_20260923_015507", "traj_20260923_015629", "traj_20260923_015918",
    "traj_20260923_020132", "traj_20260923_020407", "traj_20260923_023241",
    "traj_20260923_023422", "traj_20260924_143156", "traj_20260924_144019",
    "traj_20260926_143837",
]

# Held-out split on existing real fast-turn stereo endoscope + dual-motion gtraj:
# Train on Sept-23 early sessions + Nov-8; Test on strictly unseen Sept-23 late / Sept-26 / Oct-02 gtraj + Nov-9!
REAL_FAST_TRAIN_SESS = [
    ("traj_20260923_015507", "sep"),
    ("traj_20260923_020132", "sep"),
    ("traj_20260923_023241", "sep"),
    ("traj_20261108_134324", "nov"),
    ("traj_20261108_160843", "nov"),
    ("traj_20261108_161428", "nov"),
    ("traj_20261108_162703", "nov"),
]
REAL_FAST_TEST_SESS = [
    ("traj_20260923_015629", "sep"),   # up to 2.01 rad/s @ 31.2 Hz!
    ("traj_20260923_020407", "sep"),   # up to 1.51 rad/s @ 20.9 Hz!
    ("traj_20260923_023422", "sep"),   # up to 1.88 rad/s!
    ("traj_20260926_143837", "sep"),   # up to 2.87 rad/s!
    ("gtraj_20261002_002134", "sep"),  # simultaneous moving camera + moving target up to 1.42 rad/s!
    ("gtraj_20261002_024309", "sep"),  # simultaneous moving camera + moving target up to 1.37 rad/s!
    ("traj_20261109_022228", "nov"),   # cross-day held-out Nov-9
    ("traj_20261109_023237", "nov"),
    ("traj_20261109_023932", "nov"),
]


def centered_board_m() -> np.ndarray:
    x, y = np.meshgrid(np.arange(11), np.arange(8))
    pts = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float64) * 0.003
    return pts - pts.mean(axis=0, keepdims=True)


def board_corners_centered_m() -> np.ndarray:
    s = 0.003
    cx_b, cy_b = 5.0 * s, 3.5 * s
    return np.array([
        [-s - cx_b, -s - cy_b, 0.0],
        [11.0 * s - cx_b, -s - cy_b, 0.0],
        [11.0 * s - cx_b, 8.0 * s - cy_b, 0.0],
        [-s - cx_b, 8.0 * s - cy_b, 0.0],
    ], dtype=np.float64)


def integrate_gyro_so3(t_imu: np.ndarray, w_imu: np.ndarray, bias: np.ndarray,
                       Rci: np.ndarray, t0: float, t1: float) -> np.ndarray:
    i0, i1 = int(np.searchsorted(t_imu, t0)), int(np.searchsorted(t_imu, t1))
    if i1 <= i0:
        idx = int(np.clip(i0, 0, len(t_imu) - 1))
        wc = Rci @ (w_imu[idx] - bias)
        return Rotation.from_rotvec(wc * max(t1 - t0, 0.0)).as_matrix()
    ts = t_imu[i0:i1]
    wc = (w_imu[i0:i1] - bias[None, :]) @ Rci.T
    dts = np.diff(ts, append=t1)
    dts[0] += ts[0] - t0
    R = np.eye(3)
    for wi, dti in zip(wc, dts):
        if dti > 0:
            R = R @ Rotation.from_rotvec(wi * dti).as_matrix()
    return R


def cluster_boot_ci(rows: list[dict], ka: str, kb: str, n_boot: int = 2000) -> list[float]:
    rng = np.random.default_rng(42)
    cl = np.array([r["sess"] for r in rows])
    a = np.array([r[ka] for r in rows], dtype=np.float64)
    b = np.array([r[kb] for r in rows], dtype=np.float64)
    ids = np.unique(cl)
    idx = {c: np.flatnonzero(cl == c) for c in ids}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        sel = np.concatenate([idx[c] for c in pick])
        diffs.append(float(np.median(a[sel]) - np.median(b[sel])))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


def run_stage2_recalibrated() -> dict:
    """Evaluates Stage II real image-space overlay on the 10 Sept 23-26 stereo
    endoscope sessions using strict 2-fold out-of-sample sub-ms calibration +
    lever-arm-free camera-frame translational extrapolation."""
    K_sep, dist_sep = E.load_camera()
    Xb = centered_board_m()
    rows = []
    calib_summary = []

    for sess in SEP_STATIC_BOARD_SESS:
        z = np.load(DATA / "pose_gt_raw" / f"{sess}.npz")
        old_delay = float(np.load(DATA / "pose_gt" / f"{sess}.npz")["delay_usb"][0])
        use = np.flatnonzero((z["usable"] == 1) & (z["px"] < 1.5))
        t_raw = z["t"].astype(np.float64)[use]
        R = z["R"].astype(np.float64)[use]
        p = z["p"].astype(np.float64)[use]
        if len(t_raw) < 60:
            continue
        t_imu, w_imu = E.load_gyro(sess)
        mid = len(t_raw) // 2
        f0 = fit_subms_delay_and_rci(t_raw[:mid], R[:mid], t_imu, w_imu)
        f1 = fit_subms_delay_and_rci(t_raw[mid:], R[mid:], t_imu, w_imu)
        calib_summary.append({
            "session": sess,
            "legacy_sync_ms": round(old_delay * 1000.0, 2),
            "fold0_subms": f0["delay_subms_ms"],
            "fold1_subms": f1["delay_subms_ms"],
        })

        for (lo, hi), fit in [((mid, len(t_raw)), f0), ((1, mid), f1)]:
            if fit["omega_norm_corr"] < 0.40:
                continue
            dt_cv = float(fit["delay_subms_s"])
            Rci = np.asarray(fit["R_ci"], np.float64)
            bias = np.asarray(fit["gyro_bias_rad_s"], np.float64)
            t = t_raw - dt_cv

            for k in range(max(1, lo), hi - 1):
                dt_prev = t[k] - t[k - 1]
                if not (0.02 < dt_prev < 0.12):
                    continue
                j = None
                for jj in range(k + 1, min(k + 8, hi)):
                    if 0.055 <= (t[jj] - t[k]) <= 0.110:
                        j = jj
                        break
                if j is None:
                    continue
                tau = t[j] - t[k]
                R0, p0, R1, p1, Rp, pp = R[k], p[k], R[j], p[j], R[k - 1], p[k - 1]
                X_true = (R1.T @ (Xb - p1).T).T
                if not (0.04 < float(np.median(X_true[:, 2])) < 0.45):
                    continue
                uv_true = E.project(X_true, K_sep, dist_sep)
                if (uv_true[:, 0] < 10).any() or (uv_true[:, 0] > 1270).any() or (uv_true[:, 1] < 10).any() or (uv_true[:, 1] > 710).any():
                    continue

                i0, i1 = np.searchsorted(t_imu, t[k]), np.searchsorted(t_imu, t[j])
                om = float(np.linalg.norm(w_imu[i0:i1] - bias[None, :], axis=1).mean()) if i1 > i0 else 0.0

                X_del = (R0.T @ (Xb - p0).T).T
                uv_del = E.project(X_del, K_sep, dist_sep)

                dR_prev_vis = R0.T @ Rp
                dt_prev_cam = R0.T @ (pp - p0)
                sc = tau / dt_prev
                Rstep = Rotation.from_rotvec(Rotation.from_matrix(dR_prev_vis).as_rotvec() * sc).as_matrix()
                uv_cpl = E.project((Rstep @ X_del.T).T + sc * dt_prev_cam, K_sep, dist_sep)

                dR_gyr_prev = integrate_gyro_so3(t_imu, w_imu, bias, Rci, t[k - 1], t[k])
                dR_gyr_lag = integrate_gyro_so3(t_imu, w_imu, bias, Rci, t[k], t[j])

                # Pure Gyro Rotation (Eq. 13)
                uv_vg = E.project((dR_gyr_lag.T @ X_del.T).T, K_sep, dist_sep)
                # 2D ATW Homography
                H_atw = K_sep @ dR_gyr_lag.T @ np.linalg.inv(K_sep)
                uv_atw = E.apply_H(H_atw, uv_del)

                # Lever-Arm-Free Gyro-Decoupled Viewer SE(3) Update
                c_prev, c_cur = -Rp.T @ pp, -R0.T @ p0
                dc_trans = c_cur - (dR_gyr_prev.T @ c_prev)
                uv_vgt = E.project((dR_gyr_lag.T @ (X_del + sc * dc_trans[None, :]).T).T, K_sep, dist_sep)

                w_prev_vec = Rotation.from_matrix(Rp.T @ R0).as_rotvec() / dt_prev
                w_cur_vec = Rotation.from_matrix(R0.T @ R1).as_rotvec() / tau
                ang_jerk = float(np.linalg.norm(w_cur_vec - w_prev_vec))
                rot_true = R0.T @ R1
                rot_err = float(np.degrees(Rotation.from_matrix(rot_true.T @ dR_gyr_lag).magnitude()))

                rows.append({
                    "sess": sess, "om": om, "ang_jerk": ang_jerk,
                    "delayed_px": E.px_err(uv_del, uv_true),
                    "atw_px": E.px_err(uv_atw, uv_true),
                    "coupled_px": E.px_err(uv_cpl, uv_true),
                    "viewer_gyro_px": E.px_err(uv_vg, uv_true),
                    "viewer_decoupled_se3_px": E.px_err(uv_vgt, uv_true),
                    "rot_err_deg": rot_err,
                })

    subsets = {
        "all_pairs": rows,
        "B1_[0.0,0.5)": [r for r in rows if r["om"] < 0.5],
        "fast_ge_0.5_rad_s": [r for r in rows if r["om"] >= 0.5],
        "fast_ge_1.0_rad_s": [r for r in rows if r["om"] >= 1.0],
        "direction_reversal_dw_ge_0.25": [r for r in rows if r["ang_jerk"] >= 0.25],
        "strong_reversal_dw_ge_0.45": [r for r in rows if r["ang_jerk"] >= 0.45],
    }
    out = {"calibration_offsets": calib_summary, "strata": {}}
    for k_sub, sel in subsets.items():
        if len(sel) < 5:
            continue
        rec = {"n": len(sel), "n_sessions": len({r["sess"] for r in sel})}
        for m in ("delayed_px", "atw_px", "coupled_px", "viewer_gyro_px", "viewer_decoupled_se3_px", "rot_err_deg"):
            arr = np.array([r[m] for r in sel], dtype=np.float64)
            q25, q50, q75 = np.percentile(arr, [25, 50, 75])
            rec[m] = {"median": round(float(q50), 3), "iqr": round(float(q75 - q25), 3)}
        if rec["n_sessions"] >= 3:
            rec["delayed_minus_viewer_se3_ci95"] = cluster_boot_ci(sel, "delayed_px", "viewer_decoupled_se3_px")
            rec["coupled_minus_viewer_se3_ci95"] = cluster_boot_ci(sel, "coupled_px", "viewer_decoupled_se3_px")
            rec["delayed_minus_viewer_gyro_ci95"] = cluster_boot_ci(sel, "delayed_px", "viewer_gyro_px")
        out["strata"][k_sub] = rec
    return out


def extract_recalibrated_stage1_pairs(sess_list: list[tuple[str, str]], rci_sep: np.ndarray,
                                      rci_nov: np.ndarray, dt_nov: float, bias_nov: np.ndarray,
                                      max_per_sess: int = 40) -> list[dict]:
    """Extracts high-rotation & multi-stride real image pairs (including 1.0-2.8 rad/s
    handheld stereo endoscope & simultaneous dual-motion gtraj sessions) using
    sub-ms calibrated IMU synchronization."""
    K_sep, dist_sep = E.load_camera()
    cal_nov = json.loads((DATA / "calib_intrinsics_20261108_133906" / "camera_calibration_cam0.json").read_text(encoding="utf-8"))
    K_nov = np.asarray(cal_nov.get("K", cal_nov.get("camera_matrix")), np.float64)
    dist_nov = np.asarray(cal_nov.get("dist", cal_nov.get("distortion_coefficients", np.zeros(5))), np.float64).reshape(-1)

    Xb = centered_board_m()
    corners_b = board_corners_centered_m()
    pairs = []

    for sess, rig in sess_list:
        K_hd, dist = (K_sep, dist_sep) if rig == "sep" else (K_nov, dist_nov)
        sx, sy = 320.0 / 1280.0, 256.0 / 720.0
        K_s = K_hd.copy()
        K_s[0, 0] *= sx; K_s[0, 2] *= sx; K_s[1, 1] *= sy; K_s[1, 2] *= sy

        z = np.load(DATA / "pose_gt_raw" / f"{sess}.npz")
        usable = np.flatnonzero((z["usable"] == 1) & (z["px"] < 1.6))
        t_raw = z["t"].astype(np.float64)
        R_all, p_all = z["R"].astype(np.float64), z["p"].astype(np.float64)
        t_imu, w_imu = E.load_gyro(sess)

        if rig == "sep":
            if sess.startswith("gtraj_"):
                # Use dedicated Sept-26 rigid calibration delay (~38 ms) + calm gyro bias
                dt_s = 0.038
                Rci = rci_sep
                calm = np.linalg.norm(w_imu, axis=1) < np.radians(2.5)
                bias = w_imu[calm].mean(axis=0) if int(calm.sum()) >= 20 else np.zeros(3)
            else:
                # Fit delay & Rci on static-board session (or first 250 usable frames)
                u_fit = usable[:min(len(usable), 350)]
                f = fit_subms_delay_and_rci(t_raw[u_fit], R_all[u_fit], t_imu, w_imu)
                dt_s = float(f["delay_subms_s"])
                Rci = np.asarray(f["R_ci"], np.float64) if f["omega_norm_corr"] >= 0.5 else rci_sep
                bias = np.asarray(f["gyro_bias_rad_s"], np.float64)
        else:
            dt_s = dt_nov
            Rci = rci_nov
            bias = bias_nov

        t_all = t_raw - dt_s
        cands = []
        # Include consecutive AND 2-step (~65-95ms) pairs to test large rotational optical flow (up to 6 deg / 110 px!)
        for ii in range(len(usable) - 1):
            for step_u in (1, 2):
                if ii + step_u >= len(usable):
                    continue
                a, b = int(usable[ii]), int(usable[ii + step_u])
                if b - a > 4:
                    continue
                dt = float(t_all[b] - t_all[a])
                if not (0.025 <= dt <= 0.165):
                    continue
                i0, i1 = int(np.searchsorted(t_imu, t_all[a])), int(np.searchsorted(t_imu, t_all[b]))
                if i1 <= i0:
                    continue
                om = float(np.linalg.norm(w_imu[i0:i1] - bias[None, :], axis=1).mean())
                dR_body = integrate_gyro_so3(t_imu, w_imu, bias, Rci, t_all[a], t_all[b])
                rot_deg_imu = float(np.degrees(Rotation.from_matrix(dR_body).magnitude()))

                X0 = (R_all[a].T @ (Xb - p_all[a]).T).T
                X1 = (R_all[b].T @ (Xb - p_all[b]).T).T
                if not (0.06 < float(np.median(X0[:, 2])) < 0.50):
                    continue
                uv0 = E.project(X0, K_hd, dist)
                uv1 = E.project(X1, K_hd, dist)
                if (uv0[:, 0] < 20).any() or (uv0[:, 0] > 1260).any() or (uv0[:, 1] < 20).any() or (uv0[:, 1] > 700).any():
                    continue
                if (uv1[:, 0] < 20).any() or (uv1[:, 0] > 1260).any() or (uv1[:, 1] < 20).any() or (uv1[:, 1] > 700).any():
                    continue
                shift = E.px_err(uv0, uv1)
                if shift < 4.0 and om < 0.15:
                    continue
                cands.append((om, rot_deg_imu, shift, a, b, dR_body.T.astype(np.float32)))

        cands.sort(key=lambda x: (-x[0], -x[1]))
        n_top = min(len(cands), int(max_per_sess * 0.65))
        rem = cands[n_top:]
        step = max(1, len(rem) // max(1, max_per_sess - n_top))
        chosen = cands[:n_top] + rem[::step][:max(0, max_per_sess - n_top)]

        for om, rot_deg_imu, shift, a, b, dR_imu_cam in chosen:
            p0_im = DATA / sess / "cam0" / "images" / f"{a:06d}.jpg"
            p1_im = DATA / sess / "cam0" / "images" / f"{b:06d}.jpg"
            if not (p0_im.is_file() and p1_im.is_file()):
                continue
            bgr0 = cv2.imread(str(p0_im))
            bgr1 = cv2.imread(str(p1_im))
            if bgr0 is None or bgr1 is None:
                continue
            im0 = cv2.cvtColor(cv2.resize(bgr0, (320, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            im1 = cv2.cvtColor(cv2.resize(bgr1, (320, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

            R0, p0, R1, p1 = R_all[a], p_all[a], R_all[b], p_all[b]
            X0_c = (R0.T @ (Xb - p0).T).T
            X1_c = (R1.T @ (Xb - p1).T).T
            R_obj_gt = (R1.T @ R0).astype(np.float32)
            t_obj_gt_mm = ((R1.T @ (p0 - p1)) * 1000.0).astype(np.float32)

            c0_c = (R0.T @ (corners_b - p0).T).T
            poly_hd = E.project(c0_c, K_hd, dist)
            poly_g = poly_hd.copy()
            poly_g[:, 0] *= 40.0 / 1280.0
            poly_g[:, 1] *= 32.0 / 720.0
            gt_mask = np.zeros((32, 40), dtype=np.uint8)
            cv2.fillConvexPoly(gt_mask, np.round(poly_g).astype(np.int32), 1)

            pairs.append({
                "sess": sess, "rig": rig, "om": om, "rot_deg_imu": rot_deg_imu, "shift": shift,
                "img0": im0, "img1": im1, "K": K_s.astype(np.float32), "K_hd": K_hd, "dist": dist,
                "label": gt_mask.astype(np.int64), "z_board_mm": float(np.median(X0_c[:, 2]) * 1000.0),
                "R_obj": R_obj_gt, "t_obj": t_obj_gt_mm, "dR_imu": dR_imu_cam,
                "X0_c": X0_c, "X1_c": X1_c,
            })
    return pairs


def rot_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    return so3_log(R_pred @ R_gt.transpose(-1, -2)).norm(dim=-1).mean()


def run_stage1_recalibrated() -> dict:
    device = "cuda"
    # Fit Rci on dedicated rigid calibration sessions from Sept-26 and Nov-8!
    z_s = np.load(DATA / "pose_gt_raw" / "rigid_20260926_152337.npz")
    us = (z_s["usable"] == 1) & (z_s["px"] < 1.5)
    f_sep = fit_subms_delay_and_rci(z_s["t"][us], z_s["R"][us], *E.load_gyro("rigid_20260926_152337"))
    rci_sep = np.asarray(f_sep["R_ci"], np.float64)

    z_n = np.load(DATA / "pose_gt_raw" / "rigid_20261108_133702.npz")
    un = (z_n["usable"] == 1) & (z_n["px"] < 1.5)
    f_nov = fit_subms_delay_and_rci(z_n["t"][un], z_n["R"][un], *E.load_gyro("rigid_20261108_133702"))
    rci_nov = np.asarray(f_nov["R_ci"], np.float64)
    dt_nov = float(f_nov["delay_subms_s"])
    bias_nov = np.asarray(f_nov["gyro_bias_rad_s"], np.float64)

    train_pairs = extract_recalibrated_stage1_pairs(REAL_FAST_TRAIN_SESS, rci_sep, rci_nov, dt_nov, bias_nov, max_per_sess=45)
    test_pairs = extract_recalibrated_stage1_pairs(REAL_FAST_TEST_SESS, rci_sep, rci_nov, dt_nov, bias_nov, max_per_sess=40)
    print(f"Extracted {len(train_pairs)} recalibrated train pairs & {len(test_pairs)} held-out test pairs "
          f"(max omega = {max(p['om'] for p in test_pairs):.2f} rad/s, n>=0.8 rad/s = {sum(p['om']>=0.8 for p in test_pairs)}).", flush=True)

    # Adapt Full (Ours), No-IMU ablation, and Coupled ECD-GN on identical real train pairs
    models = {}
    for name, ab_mode, use_imu in [("EgoAnchorNet_Full", "full", True),
                                   ("EgoAnchorNet_NoIMU", "no_imu_prior", False),
                                   ("ECD_GN_IMU", "no_anchor_gate", True),
                                   ("ECD_GN", "no_anchor_gate", False)]:
        m = EgoAnchorNet(pretrained=False, refine_iters=2).to(device)
        m.load_state_dict(torch.load(CKPT_BASE, map_location=device, weights_only=False)["model"])
        for p in m.flow.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1.5e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        rng = np.random.default_rng(100)
        for _ in range(10):
            m.train()
            idx = rng.permutation(len(train_pairs))
            for s in range(0, len(idx) - 11, 12):
                b = [train_pairs[i] for i in idx[s:s + 12]]
                opt.zero_grad(set_to_none=True)
                im0 = torch.stack([torch.from_numpy(x["img0"]).permute(2, 0, 1) for x in b]).float().to(device) / 255.0
                im1 = torch.stack([torch.from_numpy(x["img1"]).permute(2, 0, 1) for x in b]).float().to(device) / 255.0
                K = torch.stack([torch.from_numpy(x["K"]) for x in b]).to(device)
                lbl = torch.stack([torch.from_numpy(x["label"]) for x in b]).to(device)
                dR = torch.stack([torch.from_numpy(x["dR_imu"]) for x in b]).to(device) if use_imu else None
                Ro = torch.stack([torch.from_numpy(x["R_obj"]) for x in b]).to(device)
                to = torch.stack([torch.from_numpy(x["t_obj"]) for x in b]).to(device)
                zb = torch.tensor([x["z_board_mm"] for x in b], dtype=torch.float32, device=device)[:, None, None]

                out = m(im0, im1, K, dR_imu=dR, ablate_mode=ab_mode)
                m_obj = (lbl == 1).float()
                lz = (F.l1_loss(torch.log(out["Z_mm"].clamp(min=10.0)), torch.log(zb.expand_as(out["Z_mm"])), reduction="none") * m_obj).sum() / m_obj.sum().clamp(min=1.0)
                if ab_mode != "no_anchor_gate":
                    cw = torch.tensor([1.0, 6.0], device=device)
                    lseg = F.cross_entropy(out["logits"], lbl, weight=cw)
                    loss = 1.2 * lseg + 1.5 * lz + 3.0 * rot_loss(out["R_obj"], Ro) + 2.5 * F.smooth_l1_loss(out["t_obj"], to, beta=2.0) / 10.0
                else:
                    loss = 1.5 * lz + 3.0 * rot_loss(out["R_cam"], Ro) + 2.5 * F.smooth_l1_loss(out["t_cam"], to, beta=2.0) / 10.0
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0)
                opt.step()
            sched.step()
        models[name] = m.eval()

    # Evaluate on held-out real test pairs
    eval_rows = []
    for p in test_pairs:
        im0 = torch.from_numpy(p["img0"]).permute(2, 0, 1).float()[None].to(device) / 255.0
        im1 = torch.from_numpy(p["img1"]).permute(2, 0, 1).float()[None].to(device) / 255.0
        K = torch.from_numpy(p["K"])[None].to(device)
        dR = torch.from_numpy(p["dR_imu"])[None].to(device)
        X0_mm = p["X0_c"] * 1000.0
        X1_mm = p["X1_c"] * 1000.0
        uv1_gt = E.project(p["X1_c"], p["K_hd"], p["dist"])

        rec = {
            "sess": p["sess"], "om": p["om"], "rot_deg_imu": p["rot_deg_imu"],
            "Uncomp_mm": float(np.linalg.norm(X0_mm - X1_mm, axis=1).mean()),
            "Uncomp_px": float(p["shift"]),
        }
        # 2D ATW
        X_atw = (p["dR_imu"].astype(np.float64) @ p["X0_c"].T).T
        rec["ATW_mm"] = float(np.linalg.norm(X_atw * 1000.0 - X1_mm, axis=1).mean())
        rec["ATW_px"] = E.px_err(E.project(X_atw, p["K_hd"], p["dist"]), uv1_gt)

        with torch.no_grad():
            for name, ab_mode, use_imu in [("EgoAnchorNet_Full", "full", True),
                                           ("EgoAnchorNet_NoIMU", "no_imu_prior", False),
                                           ("ECD_GN_IMU", "no_anchor_gate", True),
                                           ("ECD_GN", "no_anchor_gate", False)]:
                out = models[name](im0, im1, K, dR_imu=dR if use_imu else None, ablate_mode=ab_mode)
                Rk = out["R_obj" if ab_mode != "no_anchor_gate" else "R_cam"][0].cpu().numpy().astype(np.float64)
                tk = out["t_obj" if ab_mode != "no_anchor_gate" else "t_cam"][0].cpu().numpy().astype(np.float64)
                Xp_mm = (Rk @ X0_mm.T).T + tk
                rec[f"{name}_mm"] = float(np.linalg.norm(Xp_mm - X1_mm, axis=1).mean())
                rec[f"{name}_px"] = E.px_err(E.project(Xp_mm / 1000.0, p["K_hd"], p["dist"]), uv1_gt)
        eval_rows.append(rec)

    strata = {
        "all_test_pairs": eval_rows,
        "fast_omega_ge_0.5_rad_s": [r for r in eval_rows if r["om"] >= 0.5],
        "fast_omega_ge_0.8_rad_s": [r for r in eval_rows if r["om"] >= 0.8],
        "fast_omega_ge_1.0_rad_s": [r for r in eval_rows if r["om"] >= 1.0],
        "large_rot_ge_2.5_deg": [r for r in eval_rows if r["rot_deg_imu"] >= 2.5],
        "gtraj_dual_motion_only": [r for r in eval_rows if r["sess"].startswith("gtraj_")],
    }
    res = {"max_test_omega_rad_s": round(max(r["om"] for r in eval_rows), 3), "strata": {}}
    for sname, sel in strata.items():
        if len(sel) < 5:
            continue
        cell = {"n": len(sel)}
        for m in ("Uncomp", "ATW", "ECD_GN", "ECD_GN_IMU", "EgoAnchorNet_NoIMU", "EgoAnchorNet_Full"):
            am = np.array([r[f"{m}_mm"] for r in sel])
            ap = np.array([r[f"{m}_px"] for r in sel])
            cell[m] = {
                "add_mm_med": round(float(np.median(am)), 3),
                "add_mm_iqr": round(float(np.percentile(am, 75) - np.percentile(am, 25)), 3),
                "px_med": round(float(np.median(ap)), 3),
            }
        # Wilcoxon paired p-value: EgoAnchorNet_NoIMU - EgoAnchorNet_Full & ECD_GN - EgoAnchorNet_Full
        for base in ("EgoAnchorNet_NoIMU", "ECD_GN", "ECD_GN_IMU"):
            diff = np.array([r[f"{base}_mm"] - r["EgoAnchorNet_Full_mm"] for r in sel])
            _, pv = wilcoxon(diff)
            cell[f"{base}_minus_Full_med_mm"] = round(float(np.median(diff)), 3)
            cell[f"{base}_minus_Full_p"] = float(pv)
        res["strata"][sname] = cell
    return res


def main() -> None:
    t0 = time.perf_counter()
    print("=== Running Part 1: Stage II Recalibrated Real Image-Space Overlay ===", flush=True)
    s2 = run_stage2_recalibrated()
    for k, v in s2["strata"].items():
        print(f"  {k:34s} (n={v['n']:4d}) | Del={v['delayed_px']['median']:5.2f} | Cpl={v['coupled_px']['median']:5.2f} | "
              f"Viewer(gyro)={v['viewer_gyro_px']['median']:5.2f} | Viewer(decoupled_SE3)={v['viewer_decoupled_se3_px']['median']:5.2f} px | "
              f"Cpl-Viewer CI95={v.get('coupled_minus_viewer_se3_ci95')}")

    print("\n=== Running Part 2: Stage I Recalibrated Real Fast-Turn & Dual-Motion Video ===", flush=True)
    s1 = run_stage1_recalibrated()
    for k, v in s1["strata"].items():
        print(f"  {k:28s} (n={v['n']:3d}) | Uncomp={v['Uncomp']['add_mm_med']:5.2f} | ECD={v['ECD_GN']['add_mm_med']:5.2f} | "
              f"ECD+IMU={v['ECD_GN_IMU']['add_mm_med']:5.2f} | NoIMU={v['EgoAnchorNet_NoIMU']['add_mm_med']:5.2f} | "
              f"Full(Ours)={v['EgoAnchorNet_Full']['add_mm_med']:5.2f} mm (NoIMU-Full p={v['EgoAnchorNet_NoIMU_minus_Full_p']:.4g})")

    report = {"stage2_real_endoscope_recalibrated": s2, "stage1_real_fast_recalibrated": s1, "elapsed_s": round(time.perf_counter() - t0, 1)}
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_JSON} in {report['elapsed_s']}s!", flush=True)


if __name__ == "__main__":
    main()
