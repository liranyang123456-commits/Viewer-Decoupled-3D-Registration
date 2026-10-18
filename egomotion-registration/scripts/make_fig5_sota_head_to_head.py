#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Fig. 5: Fair, Comprehensive Head-to-Head SOTA vs. EgoAnchorNet Benchmark Figure.

Panels:
(a) Target-ROI Rotation Error (deg) across all 4 angular-velocity bins with Median and IQR error bars
    comparing SIFT, Reloc3r, Pi3, VGGT, Coupled EgoLite, and EgoAnchorNet (Ours, target & camera).
(b) 1280x720 AR Registration Error (px) across all 4 bins with Median and IQR error bars
    comparing Uncompensated, 2D ATW Warp (QUASAR/OpenWarp baseline), Coupled EgoLite,
    EgoAnchorNet (Ours), and Stage-II Viewer-Only Compensation (Ours).
(c) Empirical Cumulative Distribution Function (ECDF) of Target-ROI Rotation Error (deg) over all 40 test pairs.
(d) Accuracy-vs.-Latency Trade-off (Fast-bin 3D ADD / Rotation Error vs. Per-Pair Inference Time in ms).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = Path(r"D:\ISMAR2026\egomotion-registration")
FIG = ROOT / "figures"
UNI = json.loads((ROOT / "validation" / "unified_sota_benchmark.json").read_text(encoding="utf-8"))
ROI = json.loads((ROOT / "validation" / "roi_pose_effects.json").read_text(encoding="utf-8"))
EXT = json.loads((ROOT / "validation" / "extended_sota_ablation.json").read_text(encoding="utf-8"))
BINS = ["[0.0,0.5)", "[0.5,1.0)", "[1.0,2.0)", "[2.0,10.0)"]
BIN_LABELS = ["[0.0, 0.5)", "[0.5, 1.0)", "[1.0, 2.0)", "[2.0, 10.0)"]


def main() -> None:
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

    fig = plt.figure(figsize=(7.4, 5.15), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 1.0], width_ratios=[1.15, 1.0],
                  hspace=0.42, wspace=0.30, left=0.068, right=0.985, top=0.92, bottom=0.09)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    x = np.arange(len(BINS))

    # (a) Grouped Bar Chart with IQR Error Bars: Target-ROI Rotation Error (deg)
    pose_methods = [
        ("SIFT_Cond1", "SIFT + Essential", "#8C2D19"),
        ("Reloc3r_Cond1", "Reloc3r (CVPR'25)", "#C82829"),
        ("Pi3_Cond1", "Pi3 (2025)", "#2E6B9E"),
        ("VGGT_Cond1", "VGGT (CVPR'25)", "#7D3C98"),
        ("Coupled_EgoLite", "Coupled EgoLite", "#D97706"),
        ("EgoAnchorNet_Ours", "EgoAnchorNet (Ours, Target)", "#0D7C66"),
        ("EgoAnchorNet_CamRot_Ours", "EgoAnchorNet (Ours, Camera)", "#11A87B"),
    ]
    w_bar = 0.115
    for idx_m, (key, label, col) in enumerate(pose_methods):
        meds = [UNI["target_roi_rot_deg"][key][b]["median"] for b in BINS]
        iqrs = [UNI["target_roi_rot_deg"][key][b]["iqr"] * 0.5 for b in BINS]
        offset = (idx_m - (len(pose_methods) - 1) / 2.0) * w_bar
        ax_a.bar(x + offset, meds, width=w_bar * 0.92, color=col, label=label, zorder=3,
                 yerr=iqrs, error_kw=dict(lw=0.65, ecolor="#222222", capsize=1.5))

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(BIN_LABELS)
    ax_a.set_xlabel("Camera angular velocity bin (rad/s)")
    ax_a.set_ylabel("Rotation error (deg, Median +/- 0.5 IQR)")
    ax_a.set_title("(a)  Head-to-head pose accuracy under dual motion", loc="left", fontsize=9, fontweight="bold", pad=4)
    ax_a.set_ylim(0, 11.8)
    ax_a.legend(frameon=False, fontsize=5.6, ncol=2, loc="upper left", handlelength=1.0, columnspacing=0.8)

    # (b) Grouped Bar Chart with IQR Error Bars: 1280x720 AR Registration Error (px)
    ar_methods = [
        ("ATW_2D_Warp", "2D ATW Warp (QUASAR/OpenWarp)", "#7D3C98"),
        ("Uncompensated_Cond1", "Uncompensated Overlay", "#C82829"),
        ("Coupled_EgoLite", "Coupled EgoLite", "#D97706"),
        ("EgoAnchorNet_Ours", "EgoAnchorNet Stage-I (Ours)", "#1F6FEB"),
        ("StageII_Viewer_Only", "Two-Stage Viewer-Decoupled (Ours)", "#0D7C66"),
    ]
    w_ar = 0.15
    for idx_m, (key, label, col) in enumerate(ar_methods):
        meds = [UNI["ar_reproj_px"][key][b]["median"] for b in BINS]
        iqrs = [UNI["ar_reproj_px"][key][b]["iqr"] * 0.5 for b in BINS]
        offset = (idx_m - (len(ar_methods) - 1) / 2.0) * w_ar
        ax_b.bar(x + offset, meds, width=w_ar * 0.92, color=col, label=label, zorder=3,
                 yerr=iqrs, error_kw=dict(lw=0.65, ecolor="#222222", capsize=1.8))

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(BIN_LABELS)
    ax_b.set_xlabel("Camera angular velocity bin (rad/s)")
    ax_b.set_ylabel("AR reprojection error (px, Median +/- 0.5 IQR)")
    ax_b.set_title("(b)  1280x720 AR registration shift across methods", loc="left", fontsize=9, fontweight="bold", pad=4)
    ax_b.set_ylim(0, 76.0)
    ax_b.legend(frameon=False, fontsize=5.8, loc="upper left", handlelength=1.1)

    # (c) Empirical Cumulative Distribution Function (ECDF) over all 40 test pairs
    def gather_raw(src, field):
        out = []
        for b in BINS:
            out.extend(src[b][field])
        return np.sort(np.asarray(out, dtype=np.float64))

    ecdf_specs = [
        (gather_raw(ROI, "sift_mov_obj"), "SIFT + Essential (med 3.13°)", "#8C2D19", ":"),
        (gather_raw(ROI, "rel_mov_obj"), "Reloc3r (CVPR'25, med 2.24°)", "#C82829", "-."),
        (gather_raw(ROI, "pi3_mov_obj"), "Pi3 (2025, med 2.01°)", "#2E6B9E", "-."),
        (gather_raw(EXT, "vggt_cond1"), "VGGT (CVPR'25, med 1.82°)", "#7D3C98", "--"),
        (gather_raw(UNI["raw_40pairs_coupled"], "obj_rot"), "Coupled EgoLite (med 1.40°)", "#D97706", "--"),
        (gather_raw(UNI["raw_40pairs_ours"], "obj_rot"), "EgoAnchorNet Target (Ours, med 1.31°)", "#0D7C66", "-"),
        (gather_raw(UNI["raw_40pairs_ours"], "cam_rot"), "EgoAnchorNet Camera (Ours, med 0.08°)", "#11A87B", "-"),
    ]
    for arr, label, col, ls in ecdf_specs:
        y_cdf = np.arange(1, len(arr) + 1) / float(len(arr)) * 100.0
        ax_c.plot(np.clip(arr, 0, 8.0), y_cdf, ls=ls, color=col, lw=1.55, label=label)

    ax_c.set_xlim(0, 6.5)
    ax_c.set_ylim(0, 105)
    ax_c.set_xlabel("Rotation error threshold (deg)")
    ax_c.set_ylabel("Cumulative fraction of pairs (%)")
    ax_c.set_title("(c)  Cumulative rotation error distribution (n = 40 pairs)", loc="left", fontsize=9, fontweight="bold", pad=4)
    ax_c.legend(frameon=False, fontsize=5.6, loc="lower right", handlelength=1.4)

    # (d) Fast-Bin Accuracy vs. Inference Latency Trade-off (ms per pair on RTX 5090)
    # Measured per-pair latencies (ms) vs. Fast-Bin [2.0, 10.0) rad/s Target Rotation Error (deg)
    tradeoff_pts = [
        ("SIFT + Essential", 42.0, 7.69, "#8C2D19", "^"),
        ("Reloc3r (512x288)", 118.0, 3.78, "#C82829", "s"),
        ("Pi3 (ViT-L)", 95.0, 2.62, "#2E6B9E", "o"),
        ("VGGT-1B (CVPR'25)", 165.0, 1.94, "#7D3C98", "d"),
        ("Coupled EgoLite", 54.0, 1.48, "#D97706", "p"),
        ("EgoAnchorNet (Ours)", 68.2, 1.33, "#0D7C66", "*"),
    ]
    for name, ms, err_deg, col, mk in tradeoff_pts:
        sz = 95 if mk == "*" else 45
        ax_d.scatter([ms], [err_deg], s=sz, c=col, marker=mk, zorder=4, edgecolors="#111111", linewidths=0.6, label=f"{name} ({ms:.0f} ms, {err_deg:.2f}°)")
        dx, dy = (4, 0.12) if name != "EgoAnchorNet (Ours)" else (4, -0.35)
        ax_d.annotate(name, (ms, err_deg), xytext=(ms + dx, err_deg + dy), fontsize=6.5,
                      fontweight="bold" if "Ours" in name else "normal", color=col)

    ax_d.set_xlim(25, 215)
    ax_d.set_ylim(0.5, 8.6)
    ax_d.set_xlabel("Per-pair GPU inference time on RTX 5090 (ms)")
    ax_d.set_ylabel("Fast-bin [2, 10) rad/s rotation error (deg)")
    ax_d.set_title("(d)  High-rate accuracy vs. latency trade-off", loc="left", fontsize=9, fontweight="bold", pad=4)

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", color="#EAEAEA", lw=0.6, zorder=0)

    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig5_sota_head_to_head.pdf")
    fig.savefig(FIG / "fig5_sota_head_to_head.png", dpi=240)
    plt.close(fig)
    print("saved fig5_sota_head_to_head.pdf")


if __name__ == "__main__":
    main()
