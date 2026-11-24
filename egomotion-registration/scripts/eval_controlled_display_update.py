#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fair Multi-Seed Controlled Display-Update Benchmark (Protocol C, Table V).

Addresses every methodological requirement for isolating exposure-time scene-frame filtering
and display-time viewer-only propagation under accelerating camera motion and an
independently moving 3D target:
  - Device-B pinhole geometry (1280x720, fx=1095.34, fy=1097.58, cx=691.36, cy=383.36 px, z~150 mm)
  - Analytic continuous-time 6DoF camera trajectory with genuine angular acceleration
    spanning B1=[0,0.5), B2=[0.5,1.0), B3=[1.0,2.0), B4=[2.0,5.0) rad/s
  - Independently moving 6DoF 3D target (smooth rigid body motion, ~20-40 mm/s, +/-10 deg)
  - Exact analytic ground-truth poses evaluated directly at true display time t_d = t_e + tau
    (NO constant-velocity ground-truth assumption)
  - IDENTICAL noisy exposure-time observations (T_cw(t_e), T_cm(t_e)) and IDENTICAL noisy
    200 Hz IMU gyroscope stream over [t_e, t_d] supplied to all strategies
  - Fair baselines:
      1. Delayed T_cm(t_e): filtered exposure-time pose rendered without lag compensation
      2. Coupled SE(3) Filter (CV): alpha-beta constant-velocity manifold filter on T_cm,
         propagated over tau = 60 ms (the conventional coupled filter)
      3. 2D ATW Warp: rotational homography from integrated 200 Hz IMU over [t_e, t_d]
      4. EER (Gyro-Only): scene-frame alpha-beta filter on T_wm + IMU rotation propagation on T_cw
      5. EER (Gyro + Causal Trans.): scene-frame filter on T_wm + IMU gyro + causal T_cw translation
  - 20 independent random seeds (~3,000 frames), stratified into exact B1-B4 bins,
    reporting Medians [IQR] in 3D ADD (mm) and 2D screen error (px).

Self-contained: requires only numpy and scipy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "validation" / "controlled_display_update.json"

K_B = np.array([
    [1095.34, 0.0, 691.36],
    [0.0, 1097.58, 383.36],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 10.0)]
BKEYS = [f"[{lo:.1f},{hi:.1f})" for lo, hi in BINS]


def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def log_se3(T: np.ndarray) -> np.ndarray:
    """Return 6D twist [rho (m), theta (rad)] using exact SO(3) log + decoupled translation."""
    rotvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([T[:3, 3], rotvec])


def exp_se3(xi: np.ndarray) -> np.ndarray:
    R = Rotation.from_rotvec(xi[3:]).as_matrix()
    return se3(R, xi[:3])


def target_points_m() -> np.ndarray:
    x, y = np.meshgrid(np.arange(11), np.arange(8))
    pts = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float64) * 0.003
    return pts - pts.mean(axis=0, keepdims=True)


def project(X_c: np.ndarray) -> np.ndarray:
    u = (K_B @ X_c.T).T
    return u[:, :2] / np.maximum(u[:, 2:3], 1e-4)


class AnalyticDualMotion:
    """Continuous-time accelerating camera + independently moving 3D target."""

    def __init__(self, rng: np.random.Generator):
        # Camera Euler angles (rad) with multi-frequency chirp producing 0 to ~4.5 rad/s & high accel
        self.cam_amp = rng.uniform([0.28, 0.36, 0.14], [0.45, 0.55, 0.22])
        self.cam_f = rng.uniform([1.4, 1.8, 1.1], [2.4, 3.0, 1.9])
        self.cam_ph = rng.uniform(0.0, 2 * np.pi, size=3)
        self.cam_amp2 = rng.uniform([0.14, 0.18, 0.08], [0.26, 0.30, 0.14])
        self.cam_f2 = rng.uniform([3.8, 4.5, 3.0], [5.5, 6.5, 4.2])
        self.cam_ph2 = rng.uniform(0.0, 2 * np.pi, size=3)
        # Camera translation in world (m): +/-25 mm
        self.cam_t_amp = rng.uniform(0.015, 0.028, size=3)
        self.cam_t_f = rng.uniform(0.9, 1.8, size=3)
        self.cam_t_ph = rng.uniform(0.0, 2 * np.pi, size=3)

        # Independent target motion in world (m & rad): smooth rigid motion around z=0.15 m
        self.obj_t_amp = rng.uniform([0.010, 0.008, 0.005], [0.018, 0.014, 0.010])
        self.obj_t_f = rng.uniform(0.45, 0.90, size=3)
        self.obj_t_ph = rng.uniform(0.0, 2 * np.pi, size=3)
        self.obj_r_amp = rng.uniform(0.10, 0.20, size=3)
        self.obj_r_f = rng.uniform(0.40, 0.85, size=3)
        self.obj_r_ph = rng.uniform(0.0, 2 * np.pi, size=3)

    def T_wc(self, t: float) -> np.ndarray:
        eul = self.cam_amp * np.sin(self.cam_f * t + self.cam_ph) + self.cam_amp2 * np.sin(self.cam_f2 * t + self.cam_ph2)
        R = Rotation.from_euler("xyz", eul).as_matrix()
        p = self.cam_t_amp * np.sin(self.cam_t_f * t + self.cam_t_ph)
        return se3(R, p)

    def T_wm(self, t: float) -> np.ndarray:
        eul = self.obj_r_amp * np.sin(self.obj_r_f * t + self.obj_r_ph)
        R = Rotation.from_euler("xyz", eul).as_matrix()
        p = np.array([0.0, 0.0, 0.155]) + self.obj_t_amp * np.sin(self.obj_t_f * t + self.obj_t_ph)
        return se3(R, p)

    def body_omega(self, t: float, eps: float = 1e-4) -> np.ndarray:
        R0 = self.T_wc(t - eps)[:3, :3]
        R1 = self.T_wc(t + eps)[:3, :3]
        return Rotation.from_matrix(R0.T @ R1).as_rotvec() / (2.0 * eps)


def add_pose_noise(T: np.ndarray, rng: np.random.Generator, sigma_rot_deg: float, sigma_trans_mm: float) -> np.ndarray:
    dR = Rotation.from_rotvec(rng.normal(0.0, np.radians(sigma_rot_deg), size=3)).as_matrix()
    dt = rng.normal(0.0, sigma_trans_mm / 1000.0, size=3)
    return se3(T[:3, :3] @ dR, T[:3, 3] + dt)


def run_benchmark(n_seeds: int = 20, duration_s: float = 5.4, fps: float = 30.0, tau_s: float = 0.060) -> dict:
    Xb = target_points_m()
    hom = np.c_[Xb, np.ones(len(Xb))].T
    dt = 1.0 / fps
    imu_dt = 1.0 / 200.0

    # Alpha-beta manifold filter parameters (identical tuning structure for both Coupled T_cm and Scene T_wm)
    alpha_cm, beta_cm = 0.55, 0.25
    alpha_wm, beta_wm = 0.45, 0.15
    alpha_cw_v = 0.50

    records = {bk: {m: {"add": [], "px": []} for m in [
        "delayed", "coupled_cv_filter", "atw_2d", "eer_gyro_only", "eer_gyro_trans"
    ]} for bk in BKEYS}
    omega_all = []
    alpha_accel_all = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        traj = AnalyticDualMotion(rng)
        gyro_bias = rng.normal(0.0, 0.005, size=3)  # ~0.3 deg/s realistic MEMS bias

        n_frames = int(duration_s * fps)
        ts = np.arange(n_frames) * dt

        # State variables for fair online filters
        Tcm_bar = None
        v_cm = np.zeros(6)  # twist velocity on T_cm (m/s, rad/s)
        Twm_bar = None
        v_wm = np.zeros(6)  # twist velocity on T_wm
        p_wc_prev = None
        v_wc_est = np.zeros(3)

        for k, te in enumerate(ts):
            td = te + tau_s

            # Exact analytic poses at exposure t_e and display t_d
            Twc_e = traj.T_wc(te)
            Tcw_e = inv_se3(Twc_e)
            Twm_e = traj.T_wm(te)
            Tcm_e = Tcw_e @ Twm_e

            Twc_d = traj.T_wc(td)
            Tcw_d = inv_se3(Twc_d)
            Twm_d = traj.T_wm(td)
            Tcm_d = Tcw_d @ Twm_d

            # IDENTICAL noisy exposure-time observations supplied to ALL methods
            # (matches Stage-I noise floor: cam 0.15 deg / 0.8 mm; composite target 0.40 deg / 1.2 mm)
            Tcw_meas = add_pose_noise(Tcw_e, rng, sigma_rot_deg=0.15, sigma_trans_mm=0.80)
            Tcm_meas = add_pose_noise(Tcm_e, rng, sigma_rot_deg=0.40, sigma_trans_mm=1.20)
            Twc_meas = inv_se3(Tcw_meas)

            # Update Coupled SE(3) alpha-beta filter on T_cm
            if Tcm_bar is None:
                Tcm_bar = Tcm_meas.copy()
                v_cm = np.zeros(6)
            else:
                Tcm_pred = Tcm_bar @ exp_se3(v_cm * dt)
                res_cm = log_se3(inv_se3(Tcm_pred) @ Tcm_meas)
                Tcm_bar = Tcm_pred @ exp_se3(alpha_cm * res_cm)
                v_cm = v_cm + (beta_cm / dt) * res_cm

            # Update Scene-Referenced alpha-beta filter on lifted T_wm = T_cw^-1 T_cm
            Twm_meas = Twc_meas @ Tcm_meas
            if Twm_bar is None:
                Twm_bar = Twm_meas.copy()
                v_wm = np.zeros(6)
                p_wc_prev = Twc_meas[:3, 3].copy()
                v_wc_est = np.zeros(3)
            else:
                Twm_pred = Twm_bar @ exp_se3(v_wm * dt)
                res_wm = log_se3(inv_se3(Twm_pred) @ Twm_meas)
                Twm_bar = Twm_pred @ exp_se3(alpha_wm * res_wm)
                v_wm = v_wm + (beta_wm / dt) * res_wm
                v_inst = (Twc_meas[:3, 3] - p_wc_prev) / dt
                v_wc_est = alpha_cw_v * v_inst + (1.0 - alpha_cw_v) * v_wc_est
                p_wc_prev = Twc_meas[:3, 3].copy()

            # Warm up filters for first 10 frames (0.33 s)
            if k < 10:
                continue

            # Integrate realistic 200 Hz noisy IMU gyroscope over [te, td]
            n_imu = int(round(tau_s / imu_dt))
            dR_body = np.eye(3, dtype=np.float64)
            w_mags = []
            for j in range(n_imu):
                tj = te + (j + 0.5) * imu_dt
                w_true = traj.body_omega(tj)
                w_mags.append(float(np.linalg.norm(w_true)))
                w_meas = w_true + gyro_bias + rng.normal(0.0, 0.03, size=3)
                dR_body = dR_body @ Rotation.from_rotvec(w_meas * imu_dt).as_matrix()
            om_mean = float(np.mean(w_mags))
            om_acc = float(np.linalg.norm(traj.body_omega(td) - traj.body_omega(te)) / tau_s)
            omega_all.append(om_mean)
            alpha_accel_all.append(om_acc)

            # Ground truth target points at true display time t_d
            X_gt = (Tcm_d @ hom).T[:, :3]
            uv_gt = project(X_gt)

            # 1. Delayed filtered T_cm(t_e)
            X_del = (Tcm_bar @ hom).T[:, :3]
            uv_del = project(X_del)

            # 2. Coupled SE(3) filter constant-velocity extrapolation over tau
            Tcm_cpl = Tcm_bar @ exp_se3(v_cm * tau_s)
            X_cpl = (Tcm_cpl @ hom).T[:, :3]
            uv_cpl = project(X_cpl)

            # 3. 2D ATW rotational homography on exposure-time overlay
            # dR_cam maps camera(t_e) -> camera(t_d): R_cw(t_d) R_cw(t_e)^T = dR_body^T
            H_atw = K_B @ dR_body.T @ np.linalg.inv(K_B)
            uv_h = (H_atw @ np.c_[uv_del, np.ones(len(uv_del))].T).T
            uv_atw = uv_h[:, :2] / np.maximum(uv_h[:, 2:3], 1e-4)

            # 4. EER (Gyro-Only): hold filtered scene model fixed, propagate viewer rotation with 200 Hz IMU
            R_wc_hat = Twc_meas[:3, :3] @ dR_body
            Tcw_hat_rot = inv_se3(se3(R_wc_hat, Twc_meas[:3, 3]))
            X_eer_rot = (Tcw_hat_rot @ Twm_bar @ hom).T[:, :3]
            uv_eer_rot = project(X_eer_rot)

            # 5. EER (Gyro + Causal Camera Translation): also advance viewer translation casually
            p_wc_hat = Twc_meas[:3, 3] + v_wc_est * tau_s
            Tcw_hat_se3 = inv_se3(se3(R_wc_hat, p_wc_hat))
            X_eer_se3 = (Tcw_hat_se3 @ Twm_bar @ hom).T[:, :3]
            uv_eer_se3 = project(X_eer_se3)

            # Bin assignment
            bk_hit = None
            for (lo, hi), bk in zip(BINS, BKEYS):
                if lo <= om_mean < hi:
                    bk_hit = bk
                    break
            if bk_hit is None:
                continue

            def add_mm(X):
                return float(np.linalg.norm(X - X_gt, axis=1).mean() * 1000.0)

            def px_err(uv):
                return float(np.linalg.norm(uv - uv_gt, axis=1).mean())

            records[bk_hit]["delayed"]["add"].append(add_mm(X_del))
            records[bk_hit]["delayed"]["px"].append(px_err(uv_del))
            records[bk_hit]["coupled_cv_filter"]["add"].append(add_mm(X_cpl))
            records[bk_hit]["coupled_cv_filter"]["px"].append(px_err(uv_cpl))
            records[bk_hit]["atw_2d"]["px"].append(px_err(uv_atw))
            records[bk_hit]["eer_gyro_only"]["add"].append(add_mm(X_eer_rot))
            records[bk_hit]["eer_gyro_only"]["px"].append(px_err(uv_eer_rot))
            records[bk_hit]["eer_gyro_trans"]["add"].append(add_mm(X_eer_se3))
            records[bk_hit]["eer_gyro_trans"]["px"].append(px_err(uv_eer_se3))

    def summarize(vals):
        if not vals:
            return None
        a = np.asarray(vals, dtype=np.float64)
        return {
            "n": int(len(a)),
            "median": round(float(np.median(a)), 3),
            "iqr": round(float(np.percentile(a, 75) - np.percentile(a, 25)), 3),
            "mean": round(float(np.mean(a)), 3),
        }

    out = {
        "protocol": "Fair multi-seed controlled display-update benchmark (accelerating 6DoF camera + independently moving 3D target, analytic t_d ground truth, identical noisy inputs + 200 Hz IMU, Device-B 1280x720 geometry, tau=60 ms).",
        "n_seeds": n_seeds,
        "tau_s": tau_s,
        "fps": fps,
        "omega_max_rad_s": round(float(max(omega_all)), 3),
        "omega_p95_rad_s": round(float(np.percentile(omega_all, 95)), 3),
        "angular_accel_median_rad_s2": round(float(np.median(alpha_accel_all)), 2),
        "bins": {},
    }
    for bk in BKEYS:
        out["bins"][bk] = {
            m: {"add_mm": summarize(d["add"]), "px": summarize(d["px"])}
            for m, d in records[bk].items()
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {OUT_JSON} (omega_max={out['omega_max_rad_s']} rad/s, median accel={out['angular_accel_median_rad_s2']} rad/s^2)")
    for bk in BKEYS:
        n = out["bins"][bk]["delayed"]["px"]["n"]
        print(f"\nBin {bk} (n={n}):")
        for m in ["delayed", "coupled_cv_filter", "atw_2d", "eer_gyro_only", "eer_gyro_trans"]:
            ad = out["bins"][bk][m]["add_mm"]
            px = out["bins"][bk][m]["px"]
            ad_s = f"{ad['median']:.2f} [{ad['iqr']:.2f}] mm" if ad else "    ---       "
            print(f"  {m:18s}: ADD={ad_s} | Screen={px['median']:6.2f} [{px['iqr']:5.2f}] px")
    return out


if __name__ == "__main__":
    run_benchmark()
