#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Fig. 2 (Two-Stage Viewer-Decoupled Architecture) and Fig. 5 (Calibrated Stereo 3D Triangulation,
4K Rolling-Shutter Row-Timed vs. Frame-Level IMU Warp, and Real 4K+IMU Consecutive-Pair Degradation)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(r"D:\ISMAR2026\egomotion-registration")
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def make_fig2_architecture() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.16, 2.35), facecolor="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fc, ec, title_col="#111111"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2",
                           facecolor=fc, edgecolor=ec, linewidth=1.1)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                fontsize=8.6, fontweight="bold", color=title_col)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=7.4, color="#222222")

    def arrow(x0, y0, x1, y1, col="#333333", lw=1.2, label=""):
        arr = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=4.5,head_width=2.6",
                              color=col, linewidth=lw, connectionstyle="arc3,rad=0.0")
        ax.add_patch(arr)
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 1.5, label, ha="center", va="bottom",
                    fontsize=7.3, color=col, fontweight="bold")

    # Stage banners
    ax.text(2.0, 31.8, "Exposure Midpoint  $t_e$", fontsize=8.8, fontweight="bold", color="#1F4E79")
    ax.text(35.0, 31.8, "Stage I: Exposure-Time Scene-Fixed Registration", fontsize=8.8, fontweight="bold", color="#0D7C66")
    ax.text(72.0, 31.8, "Stage II: Display-Time Viewer Compensation  $t_d = t_e + \\tau$", fontsize=8.8, fontweight="bold", color="#C82829")

    # Input sensors
    box(1.5, 18.5, 22.0, 10.5, "Video Stream (4K / Stereo)",
        "Exposure midpoint $t_e$ (row $t_e(y)$)\nCalibrated stereo $b=46.30$ mm", "#EEF4FA", "#1F4E79", "#1F4E79")
    box(1.5, 3.5, 22.0, 10.5, "Rigidly Mounted IMU",
        "200 Hz angular rate $\\omega(t)$\nCalibrated extrinsics $R_{ci}$", "#FFF6E8", "#D97706", "#B45309")

    # Stage I boxes
    box(29.5, 18.5, 31.0, 10.5, "Viewer-Decoupled World Pose",
        "$T_{wm}(t_e) = T_{cw}(t_e)^{-1} T_{cm}(t_e)$\nFiltered in scene coordinates (decoupled from $\\omega$)",
        "#EBF8F4", "#0D7C66", "#0D7C66")
    box(29.5, 3.5, 31.0, 10.5, "Conventional Coupled Filter (Failure Mode)",
        "Directly smooths/extrapolates $T_{cm} = T_{cw} T_{wm}$\nCamera rotation $\\omega$ corrupts 3D model trajectory",
        "#FDEDEC", "#C82829", "#A91D1D")

    # Stage II box
    box(66.5, 11.0, 31.5, 18.0, "Viewer-Only Display Update",
        "Propagate viewer only over pipeline delay $\\tau$ (or row $t_e(y)$):\n"
        "$\\hat{T}_{cw}(t_d) = \\mathcal{P}\\!\\left(T_{cw}(t_e),\\, \\omega([t_e, t_d])\\right)$\n"
        "Compose overlay pose:  $\\hat{T}_{cm}(t_d) = \\hat{T}_{cw}(t_d)\\, \\bar{T}_{wm}(t_e)$",
        "#F3F7FD", "#1F4E79", "#1F4E79")

    arrow(23.5, 23.8, 29.5, 23.8, "#1F4E79", 1.3, "$T_{cm}(t_e), T_{cw}(t_e)$")
    arrow(23.5, 8.8, 29.5, 20.5, "#D97706", 1.1)
    arrow(60.5, 23.8, 66.5, 22.0, "#0D7C66", 1.4, "$\\bar{T}_{wm}(t_e)$")
    arrow(23.5, 8.8, 66.5, 15.0, "#D97706", 1.3, "$\\omega,\\, \\tau = t_d - t_e(y)$")

    fig.tight_layout(pad=0.2)
    fig.savefig(FIG / "fig2_two_stage_architecture.pdf")
    fig.savefig(FIG / "fig2_two_stage_architecture.png", dpi=240)
    plt.close(fig)


def make_fig3_add_and_sweep() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8.2,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    sr_data = json.loads((ROOT / "validation" / "stereo_and_rolling_shutter_results.json").read_text(encoding="utf-8"))
    consec = json.loads((ROOT / "validation" / "consecutive_pairs_by_omega.json").read_text(encoding="utf-8"))["summary"]["per_bin"]
    ego_res = json.loads((ROOT / "validation" / "ego_anchor_net_results.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(7.16, 2.55), facecolor="white")
    gs = GridSpec(1, 3, figure=fig, wspace=0.33, left=0.06, right=0.99, top=0.86, bottom=0.20)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # (a) Calibrated Stereo Extrinsics vs. Nominal CAD / Unfiltered Async
    st_cfgs = sr_data["stereo_calibration"]["configs"]
    labels_a = ["Nominal\n(46.5 mm)", "Async Fit\n(55.9 mm)", "Measured\n(No Lag)", "Measured\n+ Lag Comp."]
    keys_a = ["nominal_cad_46p5mm", "unfiltered_async_m25", "measured_extrinsic_no_lag_comp", "measured_extrinsic_lag_comp"]
    shape_rmse = [st_cfgs[k]["shape_rmse_med_mm"] for k in keys_a]
    add_disc = [st_cfgs[k]["stereo_vs_pnp_add_med_mm"] for k in keys_a]
    dyn_epi = [st_cfgs[k]["dyn_epipolar_med_px"] for k in keys_a]

    xa = np.arange(len(labels_a))
    w = 0.34
    b1 = ax1.bar(xa - w / 2, shape_rmse, width=w, color="#1F4E79", label="3D Grid Shape RMSE (mm)", zorder=3)
    b2 = ax1.bar(xa + w / 2, add_disc, width=w, color="#0D7C66", label="Stereo vs. PnP 3D ADD (mm)", zorder=3)
    for rect in b1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2, h + 0.15, f"{h:.2f}", ha="center", va="bottom", fontsize=6.3, color="#1F4E79")
    for rect in b2:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2, h + 0.15, f"{h:.2f}", ha="center", va="bottom", fontsize=6.3, color="#0D7C66")
    ax1.set_xticks(xa)
    ax1.set_xticklabels(labels_a, fontsize=6.8)
    ax1.set_ylabel("Metric 3D error (mm)")
    ax1.set_title("(a)  Calibrated stereo ($b=46.30$ mm)", loc="left", fontsize=8.6, fontweight="bold", pad=4)
    ax1.set_ylim(0, 8.2)
    ax1.legend(frameon=False, fontsize=6.3, loc="upper right")

    # (b) 4K Rolling-Shutter Frame-Level vs. Row-Timed IMU Compensation across Top/Mid/Bot bands
    rs_bins = sr_data["rolling_shutter_4k"]["bins"]
    bkeys = ["B1_0.0_0.5", "B2_0.5_1.0", "B3_1.0_2.0", "B4_2.0_10.0"]
    xb = np.arange(len(bkeys))

    f_top = [rs_bins[k]["frame_imu"]["top_med_px"] for k in bkeys]
    f_bot = [rs_bins[k]["frame_imu"]["bot_med_px"] for k in bkeys]
    f_mid = [rs_bins[k]["frame_imu"]["mid_med_px"] for k in bkeys]
    r_top = [rs_bins[k]["row_imu"]["top_med_px"] for k in bkeys]
    r_mid = [rs_bins[k]["row_imu"]["mid_med_px"] for k in bkeys]
    r_bot = [rs_bins[k]["row_imu"]["bot_med_px"] for k in bkeys]

    ax2.fill_between(xb, f_mid, f_top, color="#D97706", alpha=0.18, label="Frame IMU band spread")
    ax2.plot(xb, f_top, "--^", lw=1.2, ms=3.8, color="#D97706", label="Frame IMU (Top / Bot / Mid)")
    ax2.plot(xb, f_bot, "--v", lw=1.1, ms=3.5, color="#D97706")
    ax2.plot(xb, f_mid, ":o", lw=1.1, ms=3.2, color="#B45309")
    ax2.fill_between(xb, r_bot, r_top, color="#0D7C66", alpha=0.25, label="Row IMU band spread")
    ax2.plot(xb, r_top, "-o", lw=1.5, ms=4.0, color="#0D7C66", label="Row-Timed IMU (Top/Mid/Bot)")
    ax2.plot(xb, r_mid, "-s", lw=1.2, ms=3.5, color="#095C4B")
    ax2.plot(xb, r_bot, "-d", lw=1.2, ms=3.5, color="#0D7C66")

    ax2.set_xticks(xb)
    ax2.set_xticklabels(["[0, 0.5)", "[0.5, 1)", "[1, 2)", "[2, 10)"])
    ax2.set_xlabel("IMU angular velocity bin (rad/s)")
    ax2.set_ylabel("4K band registration shift (px)")
    ax2.set_title("(b)  4K rolling-shutter row-timed warp", loc="left", fontsize=8.6, fontweight="bold", pad=4)
    ax2.set_ylim(25, 66)
    ax2.legend(frameon=False, fontsize=6.0, loc="upper left")

    # (c) Real 4K + IMU consecutive-frame pose degradation vs. EgoAnchorNet
    rbins = ["[0.0,0.5)", "[0.5,1.0)", "[1.0,2.0)", "[2.0,10.0)"]
    rx = np.arange(len(rbins))
    pi3_r = [consec[b]["pi3_rot_err_deg"] for b in rbins]
    rel_r = [consec[b]["reloc3r_rot_err_deg"] for b in rbins]
    ours_r = [ego_res["ego_anchor_full"][b]["cam_rot_deg"] for b in rbins]

    ax3.plot(rx, rel_r, marker="o", lw=1.5, ms=4.3, color="#C82829", label="Reloc3r (real 4K+IMU)")
    ax3.plot(rx, pi3_r, marker="s", lw=1.5, ms=4.3, color="#1F4E79", label="Pi3 (real 4K+IMU)")
    ax3.plot(rx, ours_r, marker="^", lw=1.6, ms=4.6, color="#0D7C66", label="EgoAnchorNet (Ours, Cam)")
    ax3.set_xticks(rx)
    ax3.set_xticklabels(["[0, 0.5)", "[0.5, 1)", "[1, 2)", "[2, 10)"])
    ax3.set_xlabel("Measured IMU gyro rate (rad/s)")
    ax3.set_ylabel("Consecutive rotation error (deg)")
    ax3.set_title("(c)  Real 4K+IMU consecutive pairs", loc="left", fontsize=8.6, fontweight="bold", pad=4)
    ax3.set_ylim(0, 1.22)
    ax3.legend(frameon=False, fontsize=6.2, loc="upper left")

    for ax in (ax1, ax2, ax3):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", color="#EAEAEA", lw=0.6, zorder=0)

    fig.savefig(FIG / "fig3_add_and_real_bins.pdf")
    fig.savefig(FIG / "fig3_add_and_real_bins.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    make_fig2_architecture()
    make_fig3_add_and_sweep()
    print("generated fig2 and fig3")
