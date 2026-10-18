#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Fig. 4: EgoAnchorNet Qualitative Static-Anchor Disentanglement & Quantitative Ablation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

sys.path.insert(0, r"D:\ego_motiion_Camera")
from endosim.lite.ego_anchor_net import EgoAnchorNet  # noqa: E402

ROOT = Path(r"D:\ISMAR2026\egomotion-registration")
FIG = ROOT / "figures"
CORPUS = Path(r"E:\EGO_Mo\datasets\dual_anchor_corpus\val")
CKPT = ROOT / "outputs" / "ego_anchor_net" / "ego_anchor_net.pth"
RES = json.loads((ROOT / "validation" / "ego_anchor_net_results.json").read_text(encoding="utf-8"))
BINS = ["[0.0,0.5)", "[0.5,1.0)", "[1.0,2.0)", "[2.0,10.0)"]


@torch.no_grad()
def main() -> None:
    device = "cuda"
    model = EgoAnchorNet(pretrained=False, refine_iters=2).to(device).eval()
    sd = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])

    # Pick a fast-bin SCARED-textured validation sample (seed 202 or 204)
    chosen_file = None
    best_w = 0.0
    for f in sorted(CORPUS.glob("s202_*.npz")) + sorted(CORPUS.glob("s204_*.npz")):
        d = np.load(f)
        w = float(d["omega_rad_s"])
        if w > 2.2 and int(d["label"].sum()) > 120:
            if w > best_w:
                best_w = w
                chosen_file = f
    if chosen_file is None:
        chosen_file = sorted(CORPUS.glob("*.npz"))[-1]
    d = np.load(chosen_file)

    img0_np = d["img0"]
    img1_np = d["img1"]
    img0 = torch.from_numpy(img0_np).permute(2, 0, 1)[None].float().to(device) / 255.0
    img1 = torch.from_numpy(img1_np).permute(2, 0, 1)[None].float().to(device) / 255.0
    K = torch.from_numpy(d["K"])[None].float().to(device)
    dR_imu = torch.from_numpy(d["dR_imu"])[None].float().to(device)
    gt_lab = d["label"]
    omega_val = float(d["omega_rad_s"])

    out = model(img0, img1, K, dR_imu=dR_imu, ablate_mode="full")
    prob_stat = out["prob"][0, 0].cpu().numpy()  # (32, 40) static anchor probability
    f_obs = out["f_obs"][0].permute(1, 2, 0).cpu().numpy()
    f_iso = out["f_scene_isolated"][0].permute(1, 2, 0).cpu().numpy()

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

    fig = plt.figure(figsize=(7.25, 4.55), facecolor="white")
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.05, 1.0], hspace=0.44, wspace=0.38,
                  left=0.065, right=0.985, top=0.92, bottom=0.10)

    # Top row: (a) Input pair with total flow f_obs, (b) Predicted Static-Anchor probability map,
    # (c) Ground-truth Static vs. Target mask, (d) Isolated target residual flow f_obj = f_obs - f_cam
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])

    # (a) Input image + subsampled total optical flow arrows
    vis_a = img0_np.copy()
    for r in range(3, 32, 4):
        for c in range(3, 40, 4):
            u0, v0 = int(c * 8 + 4), int(r * 8 + 4)
            du, dv = f_obs[r, c]
            u1, v1 = int(np.clip(u0 + du * 1.2, 0, 319)), int(np.clip(v0 + dv * 1.2, 0, 255))
            cv2.arrowedLine(vis_a, (u0, v0), (u1, v1), (255, 70, 50), 1, cv2.LINE_AA, tipLength=0.32)
    ax_a.imshow(vis_a)
    ax_a.set_title(f"(a)  Total flow $\\mathbf{{f}}_{{\\mathrm{{obs}}}}$ ({omega_val:.2f} rad/s)", loc="left", fontsize=8.6, fontweight="bold", pad=4)

    # (b) Predicted static-anchor probability map (upsampled to 256x320)
    prob_up = cv2.resize(prob_stat, (320, 256), interpolation=cv2.INTER_LINEAR)
    im_b = ax_b.imshow(prob_up, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax_b.set_title("(b)  Predicted static anchor $p_{\\mathrm{stat}}$", loc="left", fontsize=8.6, fontweight="bold", pad=4)
    ax_b.text(0.04, 0.07, "Green: static anchor\nRed: moving target", transform=ax_b.transAxes,
              fontsize=7.0, color="white", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.18", fc="#111111", ec="none", alpha=0.75))

    # (c) Ground-truth static anchor vs. moving target overlay
    gt_up = cv2.resize((1 - gt_lab).astype(np.float32), (320, 256), interpolation=cv2.INTER_NEAREST)
    ax_c.imshow(gt_up, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    inter = float(((prob_stat > 0.5) & (gt_lab == 0)).sum())
    union = float(((prob_stat > 0.5) | (gt_lab == 0)).sum())
    iou_s = 100.0 * inter / max(union, 1.0)
    ax_c.set_title(f"(c)  Ground-truth anchor (IoU {iou_s:.1f}%)", loc="left", fontsize=8.6, fontweight="bold", pad=4)

    # (d) Isolated residual object flow magnitude |f_obs - f_cam|
    iso_mag = np.linalg.norm(f_iso, axis=-1)
    iso_up = cv2.resize(iso_mag, (320, 256), interpolation=cv2.INTER_LINEAR)
    ax_d.imshow(iso_up, cmap="magma", vmin=0.0, vmax=max(float(iso_up.max()), 4.0))
    ax_d.set_title("(d)  Isolated target flow $|\\mathbf{f}_{\\mathrm{obs}}-\\mathbf{f}_{\\mathrm{cam}}|$", loc="left", fontsize=8.6, fontweight="bold", pad=4)
    ax_d.text(0.04, 0.07, "Background suppressed\nTarget motion isolated", transform=ax_d.transAxes,
              fontsize=7.0, color="white", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.18", fc="#111111", ec="none", alpha=0.75))

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_xticks([])
        ax.set_yticks([])

    # Bottom row: Quantitative ablation across 4 angular-velocity bins
    ax_e = fig.add_subplot(gs[1, 0])
    ax_f = fig.add_subplot(gs[1, 1])
    ax_g = fig.add_subplot(gs[1, 2])
    ax_h = fig.add_subplot(gs[1, 3])

    x = np.arange(len(BINS))
    x_labels = ["[0, 0.5)", "[0.5, 1)", "[1, 2)", "[2, 10)"]

    v_specs = [
        ("coupled_egolite", "Coupled EgoLite", "#C82829", "-", "s"),
        ("no_imu_prior", "w/o IMU de-rot.", "#D97706", "-.", "^"),
        ("no_closed_loop", "w/o closed-loop", "#1F4E79", ":", "d"),
        ("ego_anchor_full", "EgoAnchorNet (ours)", "#0D7C66", "--", "o"),
    ]

    metrics = [
        (ax_e, "cam_trans_mm", "Camera trans. error (mm)", "(e)  Camera translation", (0, 4.2)),
        (ax_f, "obj_trans_mm", "Target 3D trans. error (mm)", "(f)  Target 3D motion", (0, 4.8)),
        (ax_g, "model_add_mm", "3D model ADD error (mm)", "(g)  3D model ADD", (0, 4.0)),
        (ax_h, "reproj_px", "Screen reprojection (px)", "(h)  AR screen error", (0, 18.0)),
    ]

    for ax, field, ylabel, title, ylim in metrics:
        for vkey, label, col, ls, mk in v_specs:
            vals = [RES[vkey][b][field] for b in BINS]
            ax.plot(x, vals, ls=ls, marker=mk, color=col, lw=1.5, ms=4.2, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Camera angular velocity (rad/s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=8.6, fontweight="bold", pad=4)
        ax.set_ylim(*ylim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", color="#EAEAEA", lw=0.6, zorder=0)
        ax.legend(frameon=False, fontsize=5.8, loc="upper left", handlelength=1.4, labelspacing=0.2)

    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig4_ego_anchor_net.pdf")
    fig.savefig(FIG / "fig4_ego_anchor_net.png", dpi=240)
    plt.close(fig)
    print("saved fig4_ego_anchor_net.pdf")


if __name__ == "__main__":
    main()
