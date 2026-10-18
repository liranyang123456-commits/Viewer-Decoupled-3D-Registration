#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TVCG Introduction Figure: 3D AR Registration & 3-Condition Ego-Motion Ablation (with VGGT, ATW, RAFT, LoFTR)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ego_sim.core import SimConfig, _load_camera, build_sequence  # noqa: E402

ROOT = Path(r"D:\ISMAR2026\egomotion-registration")
IMG = Path(r"E:\EGO_Mo\datasets\ego_method_check")
FIG = ROOT / "figures"
PAIRS = json.loads((IMG / "pairs.json").read_text(encoding="utf-8"))
ROI = json.loads((ROOT / "validation" / "roi_pose_effects.json").read_text(encoding="utf-8"))
REL = json.loads((ROOT / "validation" / "relative_motion_effects.json").read_text(encoding="utf-8"))
EGO_REM = json.loads((ROOT / "validation" / "ego_removed_ablation.json").read_text(encoding="utf-8"))
EXT = json.loads((ROOT / "validation" / "extended_sota_ablation.json").read_text(encoding="utf-8"))
BINS = ["[0.0,0.5)", "[0.5,1.0)", "[1.0,2.0)", "[2.0,10.0)"]
FAST_N = 72


def config() -> SimConfig:
    return SimConfig(
        motion="aggressive_6dof", board_motion=True, board_motion_mode="independent",
        translation_mm=40, board_motion_mm=25, board_rotation_deg=18,
        duration_s=12.0, camera_fps=30.0, width=1280, height=720,
        render_images=False, seed=7, rotation_deg=12.0, speed=1.4, tracking_gain=1.0,
    )


def model_wireframe(cfg: SimConfig):
    s = cfg.square_mm / 1000.0
    w = (cfg.board_cols - 2) * s
    h = (cfg.board_rows - 2) * s
    z_top = 0.012
    base = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], dtype=np.float64)
    top = np.array([[0, 0, z_top], [w, 0, z_top], [w, h, z_top], [0, h, z_top]], dtype=np.float64)
    grid_pts = np.array([[c * s, r * s, 0.0] for r in range(0, cfg.board_rows - 1, 2)
                         for c in range(0, cfg.board_cols - 1, 2)], dtype=np.float64)
    return base, top, grid_pts, s, w, h


def project(pts_B, R_W_B, p_W_B, R_W_C, p_W_C, K, dist):
    pts_W = (R_W_B @ pts_B.T).T + p_W_B
    rvec, _ = cv2.Rodrigues(R_W_C.T)
    uv, _ = cv2.projectPoints(pts_W, rvec, -R_W_C.T @ p_W_C, K, dist)
    return uv.reshape(-1, 2)


def draw_ar_prism(img, base_uv, top_uv, grid_uv, color_main, color_fill, alpha_fill=0.18, lw=3):
    overlay = img.copy()
    poly_base = np.round(base_uv).astype(np.int32).reshape(-1, 1, 2)
    poly_top = np.round(top_uv).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(overlay, poly_base, color_fill, cv2.LINE_AA)
    img[:] = cv2.addWeighted(overlay, alpha_fill, img, 1.0 - alpha_fill, 0)
    cv2.polylines(img, [poly_base], True, (15, 15, 15), lw + 3, cv2.LINE_AA)
    cv2.polylines(img, [poly_base], True, color_main, lw, cv2.LINE_AA)
    cv2.polylines(img, [poly_top], True, (15, 15, 15), max(1, lw), cv2.LINE_AA)
    cv2.polylines(img, [poly_top], True, color_main, max(1, lw - 1), cv2.LINE_AA)
    for b_pt, t_pt in zip(np.round(base_uv).astype(int), np.round(top_uv).astype(int)):
        cv2.line(img, tuple(b_pt), tuple(t_pt), (15, 15, 15), lw + 1, cv2.LINE_AA)
        cv2.line(img, tuple(b_pt), tuple(t_pt), color_main, max(1, lw - 1), cv2.LINE_AA)
    for pt in np.round(grid_uv).astype(int):
        cv2.circle(img, tuple(pt), 3, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(pt), 2, color_main, -1, cv2.LINE_AA)


def draw_error_vectors(img, uv_gt, uv_est):
    for p_gt, p_est in zip(uv_gt, uv_est):
        err = float(np.linalg.norm(p_est - p_gt))
        ratio = np.clip(err / 48.0, 0.0, 1.0)
        col = (int(20 * (1 - ratio)), int(220 * (1 - ratio) + 25), 250)
        p0 = tuple(np.round(p_gt).astype(int))
        p1 = tuple(np.round(p_est).astype(int))
        cv2.arrowedLine(img, p0, p1, (10, 10, 10), 4, cv2.LINE_AA, tipLength=0.28)
        cv2.arrowedLine(img, p0, p1, col, 2, cv2.LINE_AA, tipLength=0.28)


def median(vals):
    good = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.median(good)) if good else np.nan


def series_rel(rows, bin_name, condition, fn):
    part = [r for r in rows if r["bin"] == bin_name and r["condition"] == condition]
    return median([fn(r) for r in part])


def main() -> None:
    cfg = config()
    seq = build_sequence(cfg)
    K, dist = _load_camera("cam0", cfg.width, cfg.height)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    base_B, top_B, grid_B, s, w, h = model_wireframe(cfg)
    all_grid_B = np.array([[c * s, r * s, 0.0] for r in range(cfg.board_rows - 1)
                           for c in range(cfg.board_cols - 1)], dtype=np.float64)
    poses = seq.poses
    by_n = {item["n"]: item for item in PAIRS["pairs"]}

    item = by_n[FAST_N]
    a, b = int(item["ka"]), int(item["kb"])
    raw = cv2.imread(str(IMG / "target_moving" / f"{FAST_N + 1:06d}.jpg"))

    base_gt = project(base_B, poses.R_W_B[b], poses.p_W_B[b], poses.R_W_C[b], poses.p_W_C[b], K, dist)
    top_gt = project(top_B, poses.R_W_B[b], poses.p_W_B[b], poses.R_W_C[b], poses.p_W_C[b], K, dist)
    grid_gt = project(grid_B, poses.R_W_B[b], poses.p_W_B[b], poses.R_W_C[b], poses.p_W_C[b], K, dist)
    full_gt = project(all_grid_B, poses.R_W_B[b], poses.p_W_B[b], poses.R_W_C[b], poses.p_W_C[b], K, dist)

    base_stale = project(base_B, poses.R_W_B[a], poses.p_W_B[a], poses.R_W_C[a], poses.p_W_C[a], K, dist)
    top_stale = project(top_B, poses.R_W_B[a], poses.p_W_B[a], poses.R_W_C[a], poses.p_W_C[a], K, dist)
    grid_stale = project(grid_B, poses.R_W_B[a], poses.p_W_B[a], poses.R_W_C[a], poses.p_W_C[a], K, dist)
    full_stale = project(all_grid_B, poses.R_W_B[a], poses.p_W_B[a], poses.R_W_C[a], poses.p_W_C[a], K, dist)

    base_ego = project(base_B, poses.R_W_B[a], poses.p_W_B[a], poses.R_W_C[b], poses.p_W_C[b], K, dist)
    top_ego = project(top_B, poses.R_W_B[a], poses.p_W_B[a], poses.R_W_C[b], poses.p_W_C[b], K, dist)
    grid_ego = project(grid_B, poses.R_W_B[a], poses.p_W_B[a], poses.R_W_C[b], poses.p_W_C[b], K, dist)
    full_ego = project(all_grid_B, poses.R_W_B[a], poses.p_W_B[a], poses.R_W_C[b], poses.p_W_C[b], K, dist)

    err_stale_px = float(np.linalg.norm(full_gt - full_stale, axis=1).mean())
    err_ego_px = float(np.linalg.norm(full_gt - full_ego, axis=1).mean())

    stack_pts = np.vstack((base_gt, base_stale, top_gt, top_stale))
    x0 = max(0, int(np.floor(stack_pts[:, 0].min()) - 75))
    y0 = max(0, int(np.floor(stack_pts[:, 1].min()) - 65))
    x1 = min(raw.shape[1], int(np.ceil(stack_pts[:, 0].max()) + 75))
    y1 = min(raw.shape[0], int(np.ceil(stack_pts[:, 1].max()) + 65))

    im_a = raw.copy()
    draw_ar_prism(im_a, base_gt, top_gt, grid_gt, (130, 245, 110), (80, 200, 60), alpha_fill=0.14, lw=2)
    draw_ar_prism(im_a, base_stale, top_stale, grid_stale, (45, 55, 240), (30, 35, 210), alpha_fill=0.22, lw=3)
    crop_a = cv2.cvtColor(im_a[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)

    im_b = raw.copy()
    draw_ar_prism(im_b, base_gt, top_gt, grid_gt, (130, 245, 110), (80, 200, 60), alpha_fill=0.14, lw=2)
    draw_ar_prism(im_b, base_ego, top_ego, grid_ego, (245, 185, 35), (230, 150, 20), alpha_fill=0.22, lw=3)
    crop_b = cv2.cvtColor(im_b[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)

    im_c = (raw.copy().astype(np.float32) * 0.45).astype(np.uint8)
    draw_ar_prism(im_c, base_gt, top_gt, grid_gt, (180, 230, 180), (60, 120, 60), alpha_fill=0.10, lw=1)
    draw_error_vectors(im_c, grid_gt, grid_stale)
    for p_gt, p_ego in zip(grid_gt, grid_ego):
        p0 = tuple(np.round(p_gt).astype(int))
        p1 = tuple(np.round(p_ego).astype(int))
        cv2.arrowedLine(im_c, p0, p1, (250, 200, 40), 2, cv2.LINE_AA, tipLength=0.35)
    crop_c = cv2.cvtColor(im_c[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)

    x = np.array([0.396, 0.747, 1.361, 2.587])

    reg_both, reg_target_fixed = [], []
    for bname in BINS:
        vals_both, vals_fix = [], []
        for it in [p for p in PAIRS["pairs"] if p["bin"] == bname]:
            ia, ib = int(it["ka"]), int(it["kb"])
            g_now = project(all_grid_B, poses.R_W_B[ib], poses.p_W_B[ib], poses.R_W_C[ib], poses.p_W_C[ib], K, dist)
            g_old = project(all_grid_B, poses.R_W_B[ia], poses.p_W_B[ia], poses.R_W_C[ia], poses.p_W_C[ia], K, dist)
            vals_both.append(float(np.linalg.norm(g_now - g_old, axis=1).mean()))
            gf_now = project(all_grid_B, poses.R_W_B[0], poses.p_W_B[0], poses.R_W_C[ib], poses.p_W_C[ib], K, dist)
            gf_old = project(all_grid_B, poses.R_W_B[0], poses.p_W_B[0], poses.R_W_C[ia], poses.p_W_C[ia], K, dist)
            vals_fix.append(float(np.linalg.norm(gf_now - gf_old, axis=1).mean()))
        reg_both.append(median(vals_both))
        reg_target_fixed.append(median(vals_fix))
    reg_both = np.array(reg_both)
    reg_target_fixed = np.array(reg_target_fixed)
    reg_ego_rem = np.array([median(EGO_REM[b]["reg_shift_px"]) for b in BINS])
    reg_atw = np.array([median(EXT[b]["atw_px_cond1"]) for b in BINS])

    pose_rel_both = np.array([median(ROI[b]["rel_mov_obj"]) for b in BINS])
    pose_rel_erem = np.array([median(EGO_REM[b]["rel_obj"]) for b in BINS])
    pose_pi3_both = np.array([median(ROI[b]["pi3_mov_obj"]) for b in BINS])
    pose_pi3_erem = np.array([median(EGO_REM[b]["pi3_obj"]) for b in BINS])
    pose_vggt_tfix = np.array([median(EXT[b]["vggt_cond3"]) for b in BINS])
    pose_vggt_erem = np.array([median(EXT[b]["vggt_cond2"]) for b in BINS])
    pose_sift_both = np.array([median(ROI[b]["sift_mov_obj"]) for b in BINS])
    uni = json.loads((ROOT / "validation" / "unified_sota_benchmark.json").read_text(encoding="utf-8"))
    pose_ours_obj = np.array([uni["target_roi_rot_deg"]["EgoAnchorNet_Ours"][b]["median"] for b in BINS])
    reg_ours = np.array([uni["ar_reproj_px"]["EgoAnchorNet_Ours"][b]["median"] for b in BINS])

    klt_both = np.array([100.0 * series_rel(REL["rows"], b, "target_moving", lambda r: r["klt_board"]["survival"]) for b in BINS])
    klt_erem = np.array([100.0 * median(EGO_REM[b]["klt_board"]) for b in BINS])
    raft_both = np.array([100.0 * median(EXT[b]["raft_surv_cond1"]) for b in BINS])
    raft_erem = np.array([100.0 * median(EXT[b]["raft_surv_cond2"]) for b in BINS])

    sift_both = np.array([series_rel(REL["rows"], b, "target_moving", lambda r: r["sift_kp_on_board"]) for b in BINS])
    sift_erem = np.array([median(EGO_REM[b]["sift_kp_board"]) for b in BINS])
    loftr_both = np.array([median(EXT[b]["loftr_m_cond1"]) for b in BINS])
    loftr_erem = np.array([median(EXT[b]["loftr_m_cond2"]) for b in BINS])
    orb_both = np.array([series_rel(REL["rows"], b, "target_moving", lambda r: r["orb_board_matches"]) for b in BINS])
    orb_erem = np.array([median(EGO_REM[b]["orb_board_m"]) for b in BINS])

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

    fig = plt.figure(figsize=(7.5, 6.45), facecolor="white")
    gs = GridSpec(2, 12, figure=fig, height_ratios=[1.15, 1.0], hspace=0.40, wspace=0.95,
                  left=0.055, right=0.985, top=0.90, bottom=0.08)

    ax_a = fig.add_subplot(gs[0, 0:4])
    ax_b = fig.add_subplot(gs[0, 4:8])
    ax_c = fig.add_subplot(gs[0, 8:12])

    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])
    ax_f = fig.add_subplot(gs[1, 6:9])
    ax_g = fig.add_subplot(gs[1, 9:12])

    top_specs = [
        (ax_a, crop_a, "(a)  Uncompensated AR overlay", f"Mean error: {err_stale_px:.1f} px", "#D92B2B"),
        (ax_b, crop_b, "(b)  Ego-motion removed (ours)", f"Mean error: {err_ego_px:.1f} px", "#1F6FEB"),
        (ax_c, crop_c, "(c)  Corner displacement field",
         f"{err_stale_px:.1f} px -> {err_ego_px:.1f} px (-{round(100*(1-err_ego_px/err_stale_px))}%)", "#1B7F6E"),
    ]
    for ax, img_panel, title, badge, badge_col in top_specs:
        ax.imshow(img_panel)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#222222")
            spine.set_linewidth(0.8)
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold", pad=5)
        ax.text(0.03, 0.06, badge, transform=ax.transAxes, color="white",
                fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", fc=badge_col, ec="white", lw=0.6, alpha=0.90))

    c_both = "#C82829"
    c_tfix = "#D97706"
    c_erem = "#0D7C66"
    c_pi3 = "#1F4E79"
    c_purple = "#6C3483"

    # (d) Pose Estimation in Target ROI (Pi3, Reloc3r, VGGT, SIFT, EgoAnchorNet)
    ax_d.plot(x, pose_sift_both, ls=":", marker="^", color="#7F1D1D", lw=1.2, ms=3.6, label="SIFT (Cond. 1)")
    ax_d.plot(x, pose_rel_both, ls="-", marker="s", color=c_both, lw=1.4, ms=3.8, label="Reloc3r (Cond. 1)")
    ax_d.plot(x, pose_vggt_tfix, ls="-.", marker="d", color=c_purple, lw=1.3, ms=3.8, label="VGGT (Cond. 3)")
    ax_d.plot(x, pose_pi3_both, ls="-", marker="o", color=c_pi3, lw=1.3, ms=3.6, label="Pi3 (Cond. 1)")
    ax_d.plot(x, pose_rel_erem, ls="--", marker="s", color="#159895", lw=1.35, ms=3.6, label="Reloc3r (Cond. 2)")
    ax_d.plot(x, pose_ours_obj, ls="-", marker="*", color=c_erem, lw=1.65, ms=5.2, label="EgoAnchorNet (Ours)")
    ax_d.set_ylabel("Target-ROI rotation error (deg)")
    ax_d.set_ylim(0, 8.4)

    # (e) AR/MR Registration Error (including 2D ATW Warp baseline & Ours)
    ax_e.fill_between(x, reg_ego_rem, reg_both, color="#FADBD8", alpha=0.42)
    ax_e.plot(x, reg_atw, ls=":", marker="d", color=c_purple, lw=1.4, ms=3.8, label="2D ATW warp")
    ax_e.plot(x, reg_both, ls="-", marker="s", color=c_both, lw=1.5, ms=4.0, label="1. Ego + target")
    ax_e.plot(x, reg_target_fixed, ls="-.", marker="^", color=c_tfix, lw=1.3, ms=3.8, label="3. Ego only")
    ax_e.plot(x, reg_ours, ls="-", marker="*", color="#1F6FEB", lw=1.55, ms=5.0, label="EgoAnchorNet (Ours)")
    ax_e.plot(x, reg_ego_rem, ls="--", marker="o", color=c_erem, lw=1.5, ms=4.0, label="2. Ego removed")
    ax_e.set_ylabel("AR registration error (px)")
    ax_e.set_ylim(0, 56)

    # (f) Target Tracking Survival (KLT + Dense RAFT)
    ax_f.fill_between(x, klt_both, klt_erem, color="#D4EFDF", alpha=0.38)
    ax_f.plot(x, klt_erem, ls="--", marker="o", color=c_erem, lw=1.55, ms=4.0, label="KLT (Cond. 2)")
    ax_f.plot(x, raft_erem, ls="--", marker="d", color=c_pi3, lw=1.4, ms=4.0, label="RAFT (Cond. 2)")
    ax_f.plot(x, raft_both, ls="-", marker="d", color=c_purple, lw=1.4, ms=4.0, label="RAFT (Cond. 1)")
    ax_f.plot(x, klt_both, ls="-", marker="s", color=c_both, lw=1.55, ms=4.0, label="KLT (Cond. 1)")
    ax_f.set_ylabel("Track / flow survival (%)")
    ax_f.set_ylim(0, 108)

    # (g) Feature Extraction & Matching on Target (LoFTR + SIFT + ORB)
    ax_g.plot(x, loftr_erem, ls="--", marker="d", color=c_purple, lw=1.45, ms=4.0, label="LoFTR (Cond. 2)")
    ax_g.plot(x, sift_erem, ls="--", marker="o", color=c_erem, lw=1.45, ms=4.0, label="SIFT kp (Cond. 2)")
    ax_g.plot(x, orb_erem, ls=":", marker="^", color=c_pi3, lw=1.35, ms=3.8, label="ORB m. (Cond. 2)")
    ax_g.plot(x, loftr_both, ls="-", marker="d", color="#8E44AD", lw=1.35, ms=3.8, label="LoFTR (Cond. 1)")
    ax_g.plot(x, orb_both, ls="-.", marker="^", color=c_tfix, lw=1.35, ms=3.8, label="ORB m. (Cond. 1)")
    ax_g.plot(x, sift_both, ls="-", marker="s", color=c_both, lw=1.45, ms=4.0, label="SIFT kp (Cond. 1)")
    ax_g.set_ylabel("Target features / matches")
    ax_g.set_ylim(0, 760)

    bottom_titles = [
        (ax_d, "(d)  Pose estimation"),
        (ax_e, "(e)  AR/MR registration"),
        (ax_f, "(f)  Target tracking"),
        (ax_g, "(g)  Features & matches"),
    ]
    for ax, title in bottom_titles:
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Camera angular velocity (rad/s)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", color="#EAEAEA", lw=0.6, zorder=0)
        ax.set_xlim(0.2, 2.8)
        ax.legend(frameon=False, fontsize=5.0, loc="best", handlelength=1.3, labelspacing=0.18)

    fig.legend(
        handles=[
            Line2D([0], [0], color="#42C83C", lw=2.5, label="Ground-truth 3D target mesh"),
            Line2D([0], [0], color="#F0372D", lw=2.5, label="Cond. 1: Uncompensated (camera + target moving)"),
            Line2D([0], [0], color="#23B9F5", lw=2.5, label="Cond. 2: Camera ego-motion removed"),
            Line2D([0], [0], color="#D97706", lw=2.0, ls="-.", label="Cond. 3: Target motion removed (camera ego-motion only)"),
        ],
        loc="upper center", ncol=4, frameon=False, fontsize=7.6, bbox_to_anchor=(0.5, 0.985),
    )

    FIG.mkdir(parents=True, exist_ok=True)
    png = FIG / "intro_relative_motion.png"
    pdf = FIG / "intro_relative_motion.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    print("saved extended intro figure", png)


if __name__ == "__main__":
    main()
