#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Fig. 2 (Two-Stage Viewer-Decoupled Architecture) and Fig. 3 (3D Model ADD & Delay Sweep)."""
from __future__ import annotations

import csv
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
        "Exposure midpoint $t_e$\nModel & background observations", "#EEF4FA", "#1F4E79", "#1F4E79")
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
        "Propagate viewer only over pipeline delay $\\tau$:\n"
        "$\\hat{T}_{cw}(t_d) = \\mathcal{P}\\!\\left(T_{cw}(t_e),\\, \\omega([t_e, t_d])\\right)$\n"
        "Compose overlay pose:  $\\hat{T}_{cm}(t_d) = \\hat{T}_{cw}(t_d)\\, \\bar{T}_{wm}(t_e)$",
        "#F3F7FD", "#1F4E79", "#1F4E79")

    arrow(23.5, 23.8, 29.5, 23.8, "#1F4E79", 1.3, "$T_{cm}(t_e), T_{cw}(t_e)$")
    arrow(23.5, 8.8, 29.5, 20.5, "#D97706", 1.1)
    arrow(60.5, 23.8, 66.5, 22.0, "#0D7C66", 1.4, "$\\bar{T}_{wm}(t_e)$")
    arrow(23.5, 8.8, 66.5, 15.0, "#D97706", 1.3, "$\\omega,\\, \\tau = t_d - t_e$")

    fig.tight_layout(pad=0.2)
    fig.savefig(FIG / "fig2_two_stage_architecture.pdf")
    fig.savefig(FIG / "fig2_two_stage_architecture.png", dpi=240)
    plt.close(fig)


def make_fig3_add_and_sweep() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    add_data = json.loads((ROOT / "outputs" / "egomotion_free_registration.json").read_text(encoding="utf-8"))
    consec = json.loads((ROOT / "validation" / "consecutive_pairs_by_omega.json").read_text(encoding="utf-8"))["summary"]["per_bin"]

    # Read theoretical sweep
    rows = list(csv.DictReader((ROOT / "outputs" / "overlay_error_sweep.csv").open(encoding="utf-8")))

    fig = plt.figure(figsize=(7.16, 2.55), facecolor="white")
    gs = GridSpec(1, 3, figure=fig, wspace=0.34, left=0.065, right=0.985, top=0.86, bottom=0.19)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # (a) 3D Model ADD (mm) across angular-velocity bins
    bins_labels = ["[0, 0.8)", "[0.8, 1.6)", "[1.6, 10)"]
    x_pos = np.arange(len(bins_labels))
    w = 0.25
    v_del = [b["add_mm"] for b in add_data["add_mm_by_yaw_rate"]["delayed"]]
    v_cpl = [b["add_mm"] for b in add_data["add_mm_by_yaw_rate"]["coupled_extrapolation"]]
    v_ego = [b["add_mm"] for b in add_data["add_mm_by_yaw_rate"]["egomotion_free"]]

    ax1.bar(x_pos - w, v_del, width=w, color="#C82829", label="Delayed $T_{cm}(t_e)$", zorder=3)
    ax1.bar(x_pos, v_cpl, width=w, color="#D97706", label="Coupled $T_{cm}$ extrap.", zorder=3)
    ax1.bar(x_pos + w, v_ego, width=w, color="#0D7C66", label="Viewer-decoupled (ours)", zorder=3)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bins_labels)
    ax1.set_xlabel("IMU angular velocity bin (rad/s)")
    ax1.set_ylabel("3D model ADD error (mm)")
    ax1.set_title("(a)  3D model ADD vs. angular velocity", loc="left", fontsize=9, fontweight="bold", pad=4)
    ax1.set_ylim(0, 17.5)
    ax1.legend(frameon=False, fontsize=6.5, loc="upper left")

    # (b) Pipeline latency & angular rate vs. screen shift
    omegas = [0.5, 1.5, 3.0]
    colors = ["#1F4E79", "#D97706", "#C82829"]
    for om, col in zip(omegas, colors):
        sub = [r for r in rows if r["source"] == "rotation"
               and abs(float(r["omega_rad_s"]) - om) < 1e-4
               and abs(float(r["depth_m"]) - 0.3) < 1e-4]
        delays = [float(r["tau_ms"]) for r in sub]
        pxs = [float(r["pixel_shift"]) for r in sub]
        ax2.plot(delays, pxs, marker="o", lw=1.5, ms=4.2, color=col, label=f"$\\|\\omega\\| = {om:.1f}$ rad/s")
    ax2.set_xlabel("Pipeline latency $\\tau$ (ms)")
    ax2.set_ylabel("Rotational overlay drift (px)")
    ax2.set_title("(b)  Latency-induced overlay drift", loc="left", fontsize=9, fontweight="bold", pad=4)
    ax2.legend(frameon=False, fontsize=6.5, loc="upper left")

    # (c) Real 4K + IMU consecutive-frame pose degradation
    rbins = ["[0.0,0.5)", "[0.5,1.0)", "[1.0,2.0)", "[2.0,10.0)"]
    rx = np.arange(len(rbins))
    pi3_r = [consec[b]["pi3_rot_err_deg"] for b in rbins]
    rel_r = [consec[b]["reloc3r_rot_err_deg"] for b in rbins]
    ax3.plot(rx, pi3_r, marker="s", lw=1.5, ms=4.5, color="#1F4E79", label="Pi3 (real 4K+IMU)")
    ax3.plot(rx, rel_r, marker="o", lw=1.5, ms=4.5, color="#C82829", label="Reloc3r (real 4K+IMU)")
    ax3.set_xticks(rx)
    ax3.set_xticklabels(["[0, 0.5)", "[0.5, 1)", "[1, 2)", "[2, 10)"])
    ax3.set_xlabel("Measured IMU gyro rate (rad/s)")
    ax3.set_ylabel("Consecutive rotation error (deg)")
    ax3.set_title("(c)  Real 4K+IMU consecutive pairs", loc="left", fontsize=9, fontweight="bold", pad=4)
    ax3.set_ylim(0, 1.20)
    ax3.legend(frameon=False, fontsize=6.5, loc="upper left")

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
