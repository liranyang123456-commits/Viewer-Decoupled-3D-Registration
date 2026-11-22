#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protocol R: Cross-Day Real-Video Stage I Evaluation (Train Nov 8 + SCARED -> Held-Out Test Nov 9).

Strictly session-disjoint & day-disjoint validation of EgoAnchorNet on real 1280x720 recorded video
with real 200 Hz IMU and independent image-space PnP ground truth:
  - Train real sessions (Nov 8, 7 sessions): used alongside synthetic SCARED dual_anchor_corpus
    to adapt both EgoAnchorNet (Ours) and fair coupled baselines (ECD-GN, ECD-GN+IMU).
  - Calibration R_ci: fitted strictly on Nov 8 training sessions, never refitted on Nov 9.
  - Held-out test sessions (Nov 9, 6 sessions): never seen during training or calibration.
  - Evaluates: Uncompensated, 2D ATW (real IMU), EgoAnchorNet (Zero-Shot),
    ECD-GN (Real-Adapted), ECD-GN + IMU (Real-Adapted), and EgoAnchorNet (Real-Adapted, Ours).
  - Reports: Static/Target IoU, 3D ADD (mm), 1280x720 Screen Error (px), Target Rot Error (deg),
    stratified by angular velocity AND on the High Dual-Motion subset (|bd_flow - bg_flow| >= 5 px),
    with paired bootstrap 95% CI and Wilcoxon signed-rank p-values.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from scipy.stats import wilcoxon

sys.path.insert(0, r"D:\ego_motiion_Camera")
sys.path.insert(0, r"D:\ISMAR2026\egomotion-registration\scripts")
import eval_real_viewer_overlay as E  # noqa: E402
import review_partB_real_gyro as PB  # noqa: E402
from endosim.lite.ego_anchor_net import EgoAnchorNet  # noqa: E402
from endosim.lite.egolite import so3_log  # noqa: E402

DATA = Path(r"E:\EGO_Mo\datasets")
SYN_TRAIN = DATA / "dual_anchor_corpus" / "train"
ROOT = Path(r"D:\ISMAR2026\egomotion-registration")
CKPT_BASE = ROOT / "outputs" / "ego_anchor_net" / "ego_anchor_net.pth"
OUT_DIR = ROOT / "outputs" / "ego_anchor_net"
OUT_JSON = ROOT / "validation" / "real_stage1_crossday.json"

TRAIN_SESS = [
    "traj_20261108_134324", "traj_20261108_160843", "traj_20261108_161428",
    "traj_20261108_162703", "traj_20261108_163523", "traj_20261108_163951", "traj_20261108_165541",
]
TEST_SESS = [
    "traj_20261109_012026", "traj_20261109_022228", "traj_20261109_022758",
    "traj_20261109_023237", "traj_20261109_023607", "traj_20261109_023932",
]


def load_nov_calib():
    cal = json.loads((DATA / "calib_intrinsics_20261108_133906" / "camera_calibration_cam0.json").read_text(encoding="utf-8"))
    K = np.array(cal.get("K", cal.get("camera_matrix")), dtype=np.float64)
    dist = np.array(cal.get("dist", cal.get("dist_coeffs", np.zeros(5))), dtype=np.float64).reshape(-1)
    return K, dist


def fit_nov8_rci():
    G_all, V_all = [], []
    for sess in TRAIN_SESS:
        t, R, p, _ = PB.load_raw(sess)
        t_imu, w_imu = E.load_gyro(sess)
        for k in range(len(t) - 1):
            if not (0.04 < t[k + 1] - t[k] < 0.14):
                continue
            rv = Rotation.from_matrix(R[k].T @ R[k + 1]).as_rotvec()
            if np.linalg.norm(rv) < np.radians(0.5):
                continue
            G_all.append(PB.gyro_vec(t_imu, w_imu, t[k], t[k + 1]))
            V_all.append(rv)
    return PB.kabsch(np.array(G_all), np.array(V_all))


def board_corners_centered():
    s = 0.003
    cx_b, cy_b = 5.0 * s, 3.5 * s
    return np.array([
        [-s - cx_b, -s - cy_b, 0.0],
        [11.0 * s - cx_b, -s - cy_b, 0.0],
        [11.0 * s - cx_b, 8.0 * s - cy_b, 0.0],
        [-s - cx_b, 8.0 * s - cy_b, 0.0],
    ], dtype=np.float64)


def extract_real_pairs(sessions: list[str], K_hd: np.ndarray, dist: np.ndarray,
                       Rci: np.ndarray, max_per_sess: int, min_shift_px: float,
                       compute_bg_diff: bool = False) -> list[dict]:
    Xb = PB.centered_points()
    corners_b = board_corners_centered()
    sx, sy = 320.0 / 1280.0, 256.0 / 720.0
    K_s = K_hd.copy()
    K_s[0, 0] *= sx; K_s[0, 2] *= sx; K_s[1, 1] *= sy; K_s[1, 2] *= sy

    pairs = []
    for sess in sessions:
        z = np.load(DATA / "pose_gt_raw" / f"{sess}.npz")
        delay = float(np.load(DATA / "pose_gt" / f"{sess}.npz")["delay_usb"][0])
        t_all = z["t"].astype(np.float64) - delay
        R_all, p_all = z["R"].astype(np.float64), z["p"].astype(np.float64)
        usable = (z["usable"] == 1) & (z["px"] < 2.0)
        t_imu, w_imu = E.load_gyro(sess)
        idx_u = np.flatnonzero(usable)

        cands = []
        for ii in range(len(idx_u) - 1):
            a, b = int(idx_u[ii]), int(idx_u[ii + 1])
            if b - a > 2:
                continue
            dt = t_all[b] - t_all[a]
            if not (0.04 < dt < 0.14):
                continue
            i0, i1 = np.searchsorted(t_imu, t_all[a]), np.searchsorted(t_imu, t_all[b])
            if i1 <= i0:
                continue
            om = float(np.linalg.norm(w_imu[i0:i1], axis=1).mean())
            X0 = (R_all[a].T @ (Xb - p_all[a]).T).T
            X1 = (R_all[b].T @ (Xb - p_all[b]).T).T
            z_med = float(np.median(X0[:, 2]))
            if not (0.06 < z_med < 0.55):
                continue
            uv0 = E.project(X0, K_hd, dist)
            uv1 = E.project(X1, K_hd, dist)
            if (uv0[:, 0] < 20).any() or (uv0[:, 0] > 1260).any() or (uv0[:, 1] < 20).any() or (uv0[:, 1] > 700).any():
                continue
            if (uv1[:, 0] < 20).any() or (uv1[:, 0] > 1260).any() or (uv1[:, 1] < 20).any() or (uv1[:, 1] > 700).any():
                continue
            shift = E.px_err(uv0, uv1)
            if shift < min_shift_px and om < 0.10:
                continue
            cands.append((om, shift, a, b))

        # Stratified selection: keep top high-omega + evenly spaced rest
        cands.sort(key=lambda x: (-x[0], -x[1]))
        n_top = min(len(cands), max_per_sess // 2)
        rem = cands[n_top:]
        step = max(1, len(rem) // max(1, max_per_sess - n_top))
        chosen = cands[:n_top] + rem[::step][:max(0, max_per_sess - n_top)]

        for om, shift, a, b in chosen:
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
            X0_c = (R0.T @ (Xb - p0).T).T  # m
            X1_c = (R1.T @ (Xb - p1).T).T  # m
            R_obj_gt = (R1.T @ R0).astype(np.float32)
            t_obj_gt_mm = ((R1.T @ (p0 - p1)) * 1000.0).astype(np.float32)

            # Exact board mask & metric depth on 32x40 grid
            c0_c = (R0.T @ (corners_b - p0).T).T
            poly_hd = E.project(c0_c, K_hd, dist)
            poly_g = poly_hd.copy()
            poly_g[:, 0] *= 40.0 / 1280.0
            poly_g[:, 1] *= 32.0 / 720.0
            gt_mask = np.zeros((32, 40), dtype=np.uint8)
            cv2.fillConvexPoly(gt_mask, np.round(poly_g).astype(np.int32), 1)
            gt_mask = gt_mask.astype(np.int64)
            z_board_mm = float(np.median(X0_c[:, 2]) * 1000.0)

            dR_body = PB.delta_R(t_imu, w_imu, t_all[a], t_all[b], Rci)
            dR_imu_cam = dR_body.T.astype(np.float32)

            bg_diff_px = 0.0
            if compute_bg_diff:
                g0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2GRAY)
                g1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2GRAY)
                uv0_h = E.project(X0_c, K_hd, dist)
                uv1_h = E.project(X1_c, K_hd, dist)
                bd_vec = np.median(uv1_h - uv0_h, axis=0)
                mask_bg = np.full(g0.shape, 255, dtype=np.uint8)
                x0 = max(0, int(uv0_h[:, 0].min()) - 75); x1 = min(1280, int(uv0_h[:, 0].max()) + 75)
                y0 = max(0, int(uv0_h[:, 1].min()) - 75); y1 = min(720, int(uv0_h[:, 1].max()) + 75)
                mask_bg[y0:y1, x0:x1] = 0
                pts_bg = cv2.goodFeaturesToTrack(g0, 80, 0.05, 12, mask=mask_bg)
                if pts_bg is not None and len(pts_bg) >= 10:
                    q, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts_bg, None)
                    m = st.ravel() == 1
                    if m.sum() >= 10:
                        bg_vec = np.median(q[m, 0] - pts_bg[m, 0], axis=0)
                        bg_diff_px = float(np.linalg.norm(bd_vec - bg_vec))

            pairs.append(dict(
                sess=sess, a=a, b=b, om=om, shift=shift, bg_diff_px=bg_diff_px,
                img0=im0, img1=im1, K=K_s.astype(np.float32),
                label=gt_mask, z_board_mm=z_board_mm,
                R_obj=R_obj_gt, t_obj=t_obj_gt_mm, dR_imu=dR_imu_cam,
                X0_c=X0_c, X1_c=X1_c,
            ))
    return pairs


def load_syn_train(max_n: int = 500) -> list[dict]:
    files = sorted(SYN_TRAIN.glob("*.npz"))[:max_n]
    out = []
    for f in files:
        d = np.load(f)
        out.append(dict(
            img0=d["img0"], img1=d["img1"], K=d["K"].astype(np.float32),
            label=d["label"].astype(np.int64), depth0=d["depth0"].astype(np.float32),
            R_cam=d["R_cam"].astype(np.float32), t_cam=d["t_cam"].astype(np.float32),
            R_obj=d["R_obj"].astype(np.float32), t_obj=d["t_obj"].astype(np.float32),
            dR_imu=d["dR_imu"].astype(np.float32),
        ))
    return out


def rot_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    return so3_log(R_pred @ R_gt.transpose(-1, -2)).norm(dim=-1).mean()


def adapt_model(syn_data: list[dict], real_train: list[dict], mode: str,
                device: str, epochs: int = 12, bs: int = 12, lr: float = 1.5e-4) -> EgoAnchorNet:
    model = EgoAnchorNet(pretrained=False, refine_iters=2).to(device)
    model.load_state_dict(torch.load(CKPT_BASE, map_location=device, weights_only=False)["model"])
    for p in model.flow.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(20261123 if mode == "full" else 20261124)

    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        idx_r = rng.permutation(len(real_train))
        idx_s = rng.permutation(len(syn_data))
        n_steps = min(len(idx_r), len(idx_s)) // (bs // 2)
        losses = []
        for step in range(n_steps):
            half = bs // 2
            br = [real_train[i] for i in idx_r[step * half:(step + 1) * half]]
            bsyn = [syn_data[i] for i in idx_s[step * half:(step + 1) * half]]

            opt.zero_grad(set_to_none=True)

            # 1) Real Nov 8 batch
            im0 = torch.stack([torch.from_numpy(x["img0"]).permute(2, 0, 1) for x in br]).float().to(device) / 255.0
            im1 = torch.stack([torch.from_numpy(x["img1"]).permute(2, 0, 1) for x in br]).float().to(device) / 255.0
            Kr = torch.stack([torch.from_numpy(x["K"]) for x in br]).to(device)
            lbl = torch.stack([torch.from_numpy(x["label"]) for x in br]).to(device)
            dR = torch.stack([torch.from_numpy(x["dR_imu"]) for x in br]).to(device)
            R_o_gt = torch.stack([torch.from_numpy(x["R_obj"]) for x in br]).to(device)
            t_o_gt = torch.stack([torch.from_numpy(x["t_obj"]) for x in br]).to(device)
            z_bd = torch.tensor([x["z_board_mm"] for x in br], dtype=torch.float32, device=device)[:, None, None]

            if mode == "full":
                out_r = model(im0, im1, Kr, dR_imu=dR, ablate_mode="full")
                # Class-balanced CE (board occupies ~6-10% of image)
                cw = torch.tensor([1.0, 6.0], device=device)
                l_seg = F.cross_entropy(out_r["logits"], lbl, weight=cw)
                m_obj = (lbl == 1).float()
                l_z = (F.l1_loss(torch.log(out_r["Z_mm"].clamp(min=10.0)), torch.log(z_bd.expand_as(out_r["Z_mm"])), reduction="none") * m_obj).sum() / m_obj.sum().clamp(min=1.0)
                l_ro = rot_loss(out_r["R_obj"], R_o_gt)
                l_to = F.smooth_l1_loss(out_r["t_obj"], t_o_gt, beta=2.0) / 10.0
                # Self-supervised background flow consistency on static anchor pixels
                m_stat = (lbl == 0).float()
                f_res_bg = (out_r["f_obs"] - out_r["f_cam"]).norm(dim=1)
                l_bg = (F.smooth_l1_loss(f_res_bg, torch.zeros_like(f_res_bg), reduction="none", beta=1.0) * m_stat).sum() / m_stat.sum().clamp(min=1.0)
                l_real = 1.2 * l_seg + 1.5 * l_z + 3.0 * l_ro + 2.5 * l_to + 0.15 * l_bg
            else:
                # Fair coupled ECD-GN baseline (single pose solver)
                out_r = model(im0, im1, Kr, dR_imu=None, ablate_mode="no_anchor_gate")
                m_obj = (lbl == 1).float()
                l_z = (F.l1_loss(torch.log(out_r["Z_mm"].clamp(min=10.0)), torch.log(z_bd.expand_as(out_r["Z_mm"])), reduction="none") * m_obj).sum() / m_obj.sum().clamp(min=1.0)
                l_ro = rot_loss(out_r["R_cam"], R_o_gt)
                l_to = F.smooth_l1_loss(out_r["t_cam"], t_o_gt, beta=2.0) / 10.0
                l_real = 1.5 * l_z + 3.0 * l_ro + 2.5 * l_to

            # 2) Synthetic SCARED replay batch (prevents forgetting & anchors dual-SE(3) depth/cam)
            s_im0 = torch.stack([torch.from_numpy(x["img0"]).permute(2, 0, 1) for x in bsyn]).float().to(device) / 255.0
            s_im1 = torch.stack([torch.from_numpy(x["img1"]).permute(2, 0, 1) for x in bsyn]).float().to(device) / 255.0
            Ks = torch.stack([torch.from_numpy(x["K"]) for x in bsyn]).to(device)
            s_lbl = torch.stack([torch.from_numpy(x["label"]) for x in bsyn]).to(device)
            s_z = torch.stack([torch.from_numpy(x["depth0"]) for x in bsyn]).to(device)
            s_dR = torch.stack([torch.from_numpy(x["dR_imu"]) for x in bsyn]).to(device)
            s_Rc = torch.stack([torch.from_numpy(x["R_cam"]) for x in bsyn]).to(device)
            s_tc = torch.stack([torch.from_numpy(x["t_cam"]) for x in bsyn]).to(device)
            s_Ro = torch.stack([torch.from_numpy(x["R_obj"]) for x in bsyn]).to(device)
            s_to = torch.stack([torch.from_numpy(x["t_obj"]) for x in bsyn]).to(device)

            if mode == "full":
                out_s = model(s_im0, s_im1, Ks, dR_imu=s_dR, ablate_mode="full")
                cw = torch.tensor([1.0, 5.0], device=device)
                l_syn = (
                    F.cross_entropy(out_s["logits"], s_lbl, weight=cw)
                    + F.l1_loss(torch.log(out_s["Z_mm"].clamp(min=10.0)), torch.log(s_z.clamp(min=10.0)))
                    + 2.0 * rot_loss(out_s["R_cam"], s_Rc)
                    + 1.5 * F.smooth_l1_loss(out_s["t_cam"], s_tc, beta=2.0) / 10.0
                    + 2.5 * rot_loss(out_s["R_obj"], s_Ro)
                    + 2.0 * F.smooth_l1_loss(out_s["t_obj"], s_to, beta=2.0) / 10.0
                )
            else:
                out_s = model(s_im0, s_im1, Ks, dR_imu=None, ablate_mode="no_anchor_gate")
                l_syn = (
                    F.l1_loss(torch.log(out_s["Z_mm"].clamp(min=10.0)), torch.log(s_z.clamp(min=10.0)))
                    + 2.5 * rot_loss(out_s["R_cam"], s_Ro)
                    + 2.0 * F.smooth_l1_loss(out_s["t_cam"], s_to, beta=2.0) / 10.0
                )

            loss = l_real + 0.5 * l_syn
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        if (ep + 1) in (1, 4, 8, epochs):
            print(f"  [{mode}] ep {ep+1:2d}/{epochs} loss={np.mean(losses):.3f} ({time.perf_counter()-t0:.1f}s)", flush=True)
    return model.eval()


@torch.no_grad()
def evaluate_all(test_pairs: list[dict], K_hd: np.ndarray, dist: np.ndarray,
                 model_zs: EgoAnchorNet, model_adapt: EgoAnchorNet, model_ecd: EgoAnchorNet,
                 device: str) -> list[dict]:
    rows = []
    for x in test_pairs:
        t0_t = torch.from_numpy(x["img0"]).permute(2, 0, 1)[None].float().to(device) / 255.0
        t1_t = torch.from_numpy(x["img1"]).permute(2, 0, 1)[None].float().to(device) / 255.0
        K_t = torch.from_numpy(x["K"])[None].to(device)
        dR_t = torch.from_numpy(x["dR_imu"])[None].to(device)

        X0_c, X1_gt = x["X0_c"], x["X1_c"]
        R_gt = x["R_obj"].astype(np.float64)
        uv_gt = E.project(X1_gt, K_hd, dist)
        uv_old = E.project(X0_c, K_hd, dist)

        # 2D ATW Homography with real IMU
        H_atw = K_hd @ x["dR_imu"].astype(np.float64) @ np.linalg.inv(K_hd)
        uv_atw = E.apply_H(H_atw, uv_old)

        # 1) Zero-shot EgoAnchorNet
        out_zs = model_zs(t0_t, t1_t, K_t, dR_imu=dR_t, ablate_mode="full")
        # 2) Real-adapted ECD-GN (no IMU)
        out_ecd = model_ecd(t0_t, t1_t, K_t, dR_imu=None, ablate_mode="no_anchor_gate")
        # 3) Real-adapted ECD-GN + IMU prior (fair IMU-aided coupled baseline)
        out_ecd_imu = model_ecd(t0_t, t1_t, K_t, dR_imu=dR_t, ablate_mode="no_anchor_gate")
        # 4) Real-adapted EgoAnchorNet (Ours)
        out_ours = model_adapt(t0_t, t1_t, K_t, dR_imu=dR_t, ablate_mode="full")
        # 5) Real-adapted EgoAnchorNet w/o IMU (ablation on real video)
        out_ours_noimu = model_adapt(t0_t, t1_t, K_t, dR_imu=None, ablate_mode="no_imu_prior")

        def score_out(out, use_obj: bool):
            R_e = out["R_obj" if use_obj else "R_cam"][0].cpu().numpy().astype(np.float64)
            t_e = out["t_obj" if use_obj else "t_cam"][0].cpu().numpy().astype(np.float64) / 1000.0
            X1_e = (R_e @ X0_c.T).T + t_e
            uv_e = E.project(X1_e, K_hd, dist)
            return (
                E.add_mm(X1_e, X1_gt),
                E.px_err(uv_e, uv_gt),
                float(np.degrees(Rotation.from_matrix(R_gt.T @ R_e).magnitude())),
            )

        def iou_of(out):
            pred = (out["prob"][0, 1].cpu().numpy() > 0.5).astype(np.int64)
            gt = x["label"]
            o_i = float(((pred == 1) & (gt == 1)).sum()); o_u = float(((pred == 1) | (gt == 1)).sum())
            s_i = float(((pred == 0) & (gt == 0)).sum()); s_u = float(((pred == 0) | (gt == 0)).sum())
            return (s_i / s_u * 100.0 if s_u > 0 else 0.0), (o_i / o_u * 100.0 if o_u > 0 else 0.0)

        zs_mm, zs_px, zs_rot = score_out(out_zs, True)
        ecd_mm, ecd_px, ecd_rot = score_out(out_ecd, False)
        ecdi_mm, ecdi_px, ecdi_rot = score_out(out_ecd_imu, False)
        ni_mm, ni_px, ni_rot = score_out(out_ours_noimu, True)
        ours_mm, ours_px, ours_rot = score_out(out_ours, True)
        zs_siou, zs_oiou = iou_of(out_zs)
        ours_siou, ours_oiou = iou_of(out_ours)

        rows.append(dict(
            sess=x["sess"], a=x["a"], b=x["b"], om=x["om"], shift=x["shift"], bg_diff=x["bg_diff_px"],
            uncomp_mm=E.add_mm(X0_c, X1_gt), uncomp_px=E.px_err(uv_old, uv_gt),
            atw_px=E.px_err(uv_atw, uv_gt),
            zs_mm=zs_mm, zs_px=zs_px, zs_rot=zs_rot, zs_siou=zs_siou, zs_oiou=zs_oiou,
            ecd_mm=ecd_mm, ecd_px=ecd_px, ecd_rot=ecd_rot,
            ecdi_mm=ecdi_mm, ecdi_px=ecdi_px, ecdi_rot=ecdi_rot,
            noimu_mm=ni_mm, noimu_px=ni_px, noimu_rot=ni_rot,
            ours_mm=ours_mm, ours_px=ours_px, ours_rot=ours_rot, ours_siou=ours_siou, ours_oiou=ours_oiou,
        ))
    return rows


def summarize_subset(rows: list[dict]) -> dict:
    keys = [
        "uncomp_mm", "uncomp_px", "atw_px",
        "zs_mm", "zs_px", "zs_rot", "zs_siou", "zs_oiou",
        "ecd_mm", "ecd_px", "ecd_rot",
        "ecdi_mm", "ecdi_px", "ecdi_rot",
        "noimu_mm", "noimu_px", "noimu_rot",
        "ours_mm", "ours_px", "ours_rot", "ours_siou", "ours_oiou",
    ]
    out = {"n": len(rows)}
    for k in keys:
        a = np.array([r[k] for r in rows], dtype=np.float64)
        q25, q50, q75 = np.percentile(a, [25, 50, 75])
        out[k] = {
            "median": round(float(q50), 3),
            "iqr": round(float(q75 - q25), 3),
            "mean": round(float(a.mean()), 3),
            "sd": round(float(a.std()), 3),
        }
    # Paired bootstrap & Wilcoxon for Ours vs ECD-GN and Ours vs ECD-GN+IMU
    rng = np.random.default_rng(20261123)
    for base_name, k_b, k_o in [
        ("ecd_minus_ours_mm", "ecd_mm", "ours_mm"),
        ("ecd_minus_ours_px", "ecd_px", "ours_px"),
        ("ecdi_minus_ours_mm", "ecdi_mm", "ours_mm"),
        ("ecdi_minus_ours_px", "ecdi_px", "ours_px"),
        ("noimu_minus_ours_mm", "noimu_mm", "ours_mm"),
    ]:
        diff = np.array([r[k_b] - r[k_o] for r in rows], dtype=np.float64)
        idx = rng.integers(0, len(diff), size=(10000, len(diff)))
        meds = np.median(diff[idx], axis=1)
        lo, hi = np.percentile(meds, [2.5, 97.5])
        try:
            pval = float(wilcoxon(diff).pvalue)
        except Exception:
            pval = 1.0
        out[base_name] = {
            "median_gain": round(float(np.median(diff)), 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "wilcoxon_p": pval,
        }
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K_hd, dist = load_nov_calib()
    Rci = fit_nov8_rci()

    print("Extracting Nov 8 training real pairs and Nov 9 held-out test real pairs...", flush=True)
    real_train = extract_real_pairs(TRAIN_SESS, K_hd, dist, Rci, max_per_sess=130, min_shift_px=2.0, compute_bg_diff=False)
    real_test = extract_real_pairs(TEST_SESS, K_hd, dist, Rci, max_per_sess=100, min_shift_px=3.0, compute_bg_diff=True)
    syn_train = load_syn_train(max_n=600)
    print(f"Loaded: syn_train={len(syn_train)}, real_train(Nov 8)={len(real_train)}, real_test(Nov 9 held-out)={len(real_test)}", flush=True)

    model_zs = EgoAnchorNet(pretrained=False, refine_iters=2).to(device).eval()
    model_zs.load_state_dict(torch.load(CKPT_BASE, map_location=device, weights_only=False)["model"])

    print("\nAdapting EgoAnchorNet (Ours) on synthetic + Nov 8 real sessions...", flush=True)
    model_adapt = adapt_model(syn_train, real_train, mode="full", device=device, epochs=12, bs=12, lr=1.5e-4)
    torch.save({"model": model_adapt.state_dict()}, OUT_DIR / "ego_anchor_net_real_nov8.pth")

    print("\nAdapting fair baseline ECD-GN on the exact same synthetic + Nov 8 split...", flush=True)
    model_ecd = adapt_model(syn_train, real_train, mode="no_anchor_gate", device=device, epochs=12, bs=12, lr=1.5e-4)

    print("\nEvaluating all methods on strictly held-out Nov 9 real video pairs...", flush=True)
    rows = evaluate_all(real_test, K_hd, dist, model_zs, model_adapt, model_ecd, device)

    subsets = {
        "all_nov9": rows,
        "omega_0_0.2": [r for r in rows if 0.0 <= r["om"] < 0.2],
        "omega_0.2_0.5": [r for r in rows if 0.2 <= r["om"] < 0.5],
        "omega_0.5_2.0": [r for r in rows if 0.5 <= r["om"] < 2.0],
        "high_dual_motion_ge5px": [r for r in rows if r["bg_diff"] >= 5.0],
        "high_shift_ge10px": [r for r in rows if r["uncomp_px"] >= 10.0],
    }

    summary = {
        "protocol": "Cross-day session-disjoint real-video Stage I evaluation (Train: 7 Nov-8 sessions + SCARED; Test: 6 held-out Nov-9 sessions; R_ci fitted on Nov 8 only; GT: independent image-space PnP).",
        "n_train_real": len(real_train),
        "n_test_real": len(real_test),
        "subsets": {},
    }
    for name, sel in subsets.items():
        if len(sel) < 5:
            continue
        st = summarize_subset(sel)
        summary["subsets"][name] = st
        print(
            f"  {name:24s} n={st['n']:3d} | "
            f"IoU(stat/obj)={st['ours_siou']['median']:5.1f}/{st['ours_oiou']['median']:4.1f}% (ZS {st['zs_oiou']['median']:4.1f}%) | "
            f"ADD(mm): uncomp={st['uncomp_mm']['median']:4.2f} ZS={st['zs_mm']['median']:4.2f} "
            f"ECD={st['ecd_mm']['median']:4.2f} ECD+IMU={st['ecdi_mm']['median']:4.2f} "
            f"NoIMU={st['noimu_mm']['median']:4.2f} OURS={st['ours_mm']['median']:4.2f} | "
            f"px: uncomp={st['uncomp_px']['median']:5.1f} ATW={st['atw_px']['median']:5.1f} "
            f"ECD={st['ecd_px']['median']:4.2f} ECD+IMU={st['ecdi_px']['median']:4.2f} OURS={st['ours_px']['median']:4.2f} | "
            f"gain_vs_ECD={st['ecd_minus_ours_mm']['median_gain']:+.2f}mm (p={st['ecd_minus_ours_mm']['wilcoxon_p']:.2e}) "
            f"vs_ECDI={st['ecdi_minus_ours_mm']['median_gain']:+.2f}mm (p={st['ecdi_minus_ours_mm']['wilcoxon_p']:.2e})",
            flush=True,
        )

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved:", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
