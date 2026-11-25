#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone Sub-Millisecond Camera-IMU SO(3) Impulse Calibration Module.

Jointly estimates camera-to-IMU timestamp offset dt, camera-from-IMU rotation
R_ci in SO(3), and gyroscope turn-on bias b_g by aligning consecutive chessboard
PnP rotation increments against piecewise-linear cumulative 200 Hz gyroscope
integrals under Huber loss (requires only numpy and scipy).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def kabsch_so3(G: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Finds R_ci in SO(3) minimizing sum_k ||V_k - R_ci G_k||^2."""
    H = G.T @ V
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


class CumulativeGyroIntegrator:
    """Exact O(log N) piecewise-linear cumulative integral of 200 Hz gyroscope
    samples supporting continuous sub-millisecond query timestamps."""

    def __init__(self, t_imu: np.ndarray, w_imu_rad: np.ndarray):
        self.t = np.asarray(t_imu, dtype=np.float64)
        self.w = np.asarray(w_imu_rad, dtype=np.float64)
        dts = np.diff(self.t)
        mid_w = 0.5 * (self.w[:-1] + self.w[1:])
        self.cum = np.vstack([
            np.zeros((1, 3), dtype=np.float64),
            np.cumsum(mid_w * dts[:, None], axis=0),
        ])

    def eval_cum(self, t_query: np.ndarray) -> np.ndarray:
        tq = np.clip(t_query, self.t[0], self.t[-1] - 1e-9)
        idx = np.clip(np.searchsorted(self.t, tq, side="right") - 1, 0, len(self.t) - 2)
        dt_rem = tq - self.t[idx]
        dt_seg = np.maximum(self.t[idx + 1] - self.t[idx], 1e-9)
        alpha = dt_rem / dt_seg
        w_interp = self.w[idx] * (1.0 - 0.5 * alpha[:, None]) + self.w[idx + 1] * (0.5 * alpha[:, None])
        return self.cum[idx] + w_interp * dt_rem[:, None]

    def integrate_intervals(self, t0: np.ndarray, t1: np.ndarray, bias: np.ndarray) -> np.ndarray:
        return (self.eval_cum(t1) - self.eval_cum(t0)) - (t1 - t0)[:, None] * bias[None, :]


def fit_subms_delay_and_rci(
    t_cam_abs: np.ndarray,
    R_cam: np.ndarray,
    t_imu_abs: np.ndarray,
    w_imu: np.ndarray,
    search_range: tuple[float, float] = (-0.10, 0.45),
) -> dict:
    """Jointly estimates sub-millisecond camera timestamp delay dt, R_ci in SO(3),
    and residual gyro bias b_g from consecutive PnP frame pairs."""
    t_origin = float(min(t_cam_abs[0], t_imu_abs[0]))
    t_cam = np.asarray(t_cam_abs, dtype=np.float64) - t_origin
    t_imu = np.asarray(t_imu_abs, dtype=np.float64) - t_origin
    w_imu = np.asarray(w_imu, dtype=np.float64)

    w_norm = np.linalg.norm(w_imu, axis=1)
    calm = w_norm < np.radians(2.5)
    bias0 = w_imu[calm].mean(axis=0) if int(calm.sum()) >= 30 else np.zeros(3)

    t0_list, t1_list, v_list = [], [], []
    for k in range(len(t_cam) - 1):
        dt_f = float(t_cam[k + 1] - t_cam[k])
        if not (0.015 <= dt_f <= 0.12):
            continue
        rv = Rotation.from_matrix(R_cam[k].T @ R_cam[k + 1]).as_rotvec()
        t0_list.append(t_cam[k])
        t1_list.append(t_cam[k + 1])
        v_list.append(rv)

    t0_arr = np.asarray(t0_list, dtype=np.float64)
    t1_arr = np.asarray(t1_list, dtype=np.float64)
    V_arr = np.asarray(v_list, dtype=np.float64)
    v_ang = np.linalg.norm(V_arr, axis=1)

    act = v_ang >= np.radians(0.35)
    if int(act.sum()) < 15:
        act = v_ang >= np.percentile(v_ang, 50)

    t0_act, t1_act, V_act = t0_arr[act], t1_arr[act], V_arr[act]
    integrator = CumulativeGyroIntegrator(t_imu, w_imu)

    delays_coarse = np.arange(search_range[0], search_range[1] + 1e-9, 0.001)
    best_loss = float("inf")
    best_dt = 0.0
    best_R = np.eye(3)

    for dt in delays_coarse:
        if (t0_act[0] - dt < t_imu[0]) or (t1_act[-1] - dt > t_imu[-1]):
            continue
        G = integrator.integrate_intervals(t0_act - dt, t1_act - dt, bias0)
        if float(np.linalg.norm(G)) < 1e-6:
            continue
        R_cand = kabsch_so3(G, V_act)
        res = V_act - (G @ R_cand.T)
        rn = np.linalg.norm(res, axis=1)
        loss = float(np.where(rn < 0.015, 0.5 * rn ** 2, 0.015 * (rn - 0.0075)).mean())
        if loss < best_loss:
            best_loss = loss
            best_dt = float(dt)
            best_R = R_cand

    def residual_fn(params: np.ndarray) -> np.ndarray:
        dt = float(params[0])
        dR = Rotation.from_rotvec(params[1:4]).as_matrix()
        R_cur = dR @ best_R
        b_cur = bias0 + params[4:7]
        G_all = integrator.integrate_intervals(t0_arr - dt, t1_arr - dt, b_cur)
        return (V_arr - (G_all @ R_cur.T)).ravel()

    x0 = np.zeros(7, dtype=np.float64)
    x0[0] = best_dt
    lb = np.array([best_dt - 0.025, -0.35, -0.35, -0.35, -0.05, -0.05, -0.05])
    ub = np.array([best_dt + 0.025, +0.35, +0.35, +0.35, +0.05, +0.05, +0.05])
    sol = least_squares(
        residual_fn,
        x0,
        bounds=(lb, ub),
        loss="huber",
        f_scale=0.006,
        diff_step=np.array([2e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]),
        max_nfev=300,
    )

    dt_opt = float(sol.x[0])
    R_opt = Rotation.from_rotvec(sol.x[1:4]).as_matrix() @ best_R
    bias_opt = bias0 + sol.x[4:7]

    G_opt = integrator.integrate_intervals(t0_arr - dt_opt, t1_arr - dt_opt, bias_opt)
    V_pred = G_opt @ R_opt.T
    dts_f = (t1_arr - t0_arr)[:, None]
    w_vis = V_arr / dts_f
    w_gyr = V_pred / dts_f
    norm_corr = float(np.corrcoef(np.linalg.norm(w_vis, axis=1), np.linalg.norm(w_gyr, axis=1))[0, 1])
    step_err_deg = np.degrees(np.linalg.norm(V_arr - V_pred, axis=1))

    return {
        "delay_subms_s": round(dt_opt, 6),
        "delay_subms_ms": round(dt_opt * 1000.0, 3),
        "R_ci": R_opt.tolist(),
        "gyro_bias_rad_s": bias_opt.tolist(),
        "omega_norm_corr": round(norm_corr, 5),
        "per_step_rot_err_deg_median": round(float(np.median(step_err_deg)), 4),
    }


def integrate_gyro_so3(
    t_imu: np.ndarray,
    w_imu: np.ndarray,
    bias: np.ndarray,
    Rci: np.ndarray,
    t0: float,
    t1: float,
) -> np.ndarray:
    """Integrates 200 Hz gyroscope angular velocity over [t0, t1] on SO(3)."""
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
