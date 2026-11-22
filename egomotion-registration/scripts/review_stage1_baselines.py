#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reviewer-requested Stage-I checks on the 260 validation pairs.

1. Harness check: re-compute the two variants already in Table III (ECD-GN and full).
2. Fairer baselines that a reviewer will ask for:
   - ECD-GN + IMU        whole-frame single SE(3) fit with the IMU rotation prior (target pose := camera pose).
   - Target-ROI GN       direct composite-motion fit on the target pixels only (no camera solve),
                         with the ground-truth ROI (oracle) or the predicted ROI, with / without IMU init.
3. Classical segmentation on de-rotated flow (no network): flow-magnitude threshold and RANSAC FOE fit.
4. Sensitivity of the full model to IMU rotation noise, extrinsic error, and time-offset-like scale error.

Everything is evaluated with the same metric code as train_eval_ego_anchor.evaluate_variants.
Results go to validation/review_stage1_baselines.json (the existing result files are not touched).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"D:\ego_motiion_Camera")
sys.path.insert(0, r"D:\ego_motiion_Camera\scripts")
from endosim.lite.ego_anchor_net import EgoAnchorNet  # noqa: E402
from train_eval_ego_anchor import DualAnchorDataset, geodesic_deg  # noqa: E402

ROOT = Path(r"D:\ISMAR2026\egomotion-registration")
CKPT = ROOT / "outputs" / "ego_anchor_net" / "ego_anchor_net.pth"
OUT = ROOT / "validation" / "review_stage1_baselines.json"
BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 10.0)]
BKEYS = [f"[{lo:.1f},{hi:.1f})" for lo, hi in BINS]
DEV = "cuda"

torch.manual_seed(0)
np.random.seed(0)


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues for a single rotation vector (3,)."""
    th = w.norm()
    if th < 1e-12:
        return torch.eye(3, device=w.device, dtype=w.dtype)
    k = w / th
    K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], device=w.device, dtype=w.dtype)
    return torch.eye(3, device=w.device, dtype=w.dtype) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    c = ((torch.trace(R) - 1.0) * 0.5).clamp(-1.0, 1.0)
    th = torch.acos(c)
    if th < 1e-9:
        return torch.zeros(3, device=R.device, dtype=R.dtype)
    v = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2 * torch.sin(th)) * th


def rand_axis() -> torch.Tensor:
    a = torch.randn(3, device=DEV)
    return a / a.norm()


def metrics(R_o, t_o, R_obj_gt, t_obj_gt, K, pts_M):
    X0 = pts_M + torch.tensor([0.0, 0.0, 150.0], device=DEV)
    X1_gt = (R_obj_gt[0] @ X0.T).T + t_obj_gt[0]
    X1_pr = (R_o[0] @ X0.T).T + t_o[0]
    add_mm = float((X1_pr - X1_gt).norm(dim=-1).mean().item())
    K_hd = K[0] * 4.0
    K_hd[2, 2] = 1.0
    uv_gt = (K_hd @ X1_gt.T).T
    uv_gt = uv_gt[:, :2] / uv_gt[:, 2:3].clamp(min=1.0)
    uv_pr = (K_hd @ X1_pr.T).T
    uv_pr = uv_pr[:, :2] / uv_pr[:, 2:3].clamp(min=1.0)
    px = float((uv_pr - uv_gt).norm(dim=-1).mean().item())
    return add_mm, px


def iou(gt: torch.Tensor, pr: torch.Tensor) -> float:
    inter = float((gt & pr).sum().item())
    union = float((gt | pr).sum().clamp(min=1).item())
    return 100.0 * inter / union


def ransac_foe_mask(f_res: np.ndarray, uu: np.ndarray, vv: np.ndarray, rng, mag_thr=0.4, ang_deg=10.0, iters=300):
    """Static = flow lines through one focus of expansion (depth-free translational model)."""
    h, w = uu.shape
    P = np.stack([uu.ravel(), vv.ravel()], 1)
    F_ = f_res.reshape(2, -1).T
    mag = np.linalg.norm(F_, axis=1)
    cand = np.flatnonzero(mag > mag_thr)
    best_inl = None
    best_n = -1
    sin_thr = np.sin(np.deg2rad(ang_deg))
    for _ in range(iters):
        if len(cand) < 2:
            break
        i, j = rng.choice(cand, 2, replace=False)
        d1, d2 = F_[i] / mag[i], F_[j] / mag[j]
        A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
        if abs(np.linalg.det(A)) < 1e-3:
            # (nearly) parallel flow lines: focus at infinity -> all flow parallel to d1
            cross = np.abs(F_[:, 0] * d1[1] - F_[:, 1] * d1[0]) / np.maximum(mag, 1e-9)
            inl = (cross < sin_thr) | (mag < mag_thr)
        else:
            s = np.linalg.solve(A, P[j] - P[i])
            e = P[i] + s[0] * d1
            rad = P - e
            rn = np.maximum(np.linalg.norm(rad, axis=1), 1e-9)
            cross = np.abs(F_[:, 0] * rad[:, 1] - F_[:, 1] * rad[:, 0]) / (np.maximum(mag, 1e-9) * rn)
            inl = (cross < sin_thr) | (mag < mag_thr)
        n = int(inl.sum())
        if n > best_n:
            best_n, best_inl = n, inl
    if best_inl is None:
        best_inl = np.ones(h * w, dtype=bool)
    return best_inl.reshape(h, w)


@torch.no_grad()
def main() -> None:
    model = EgoAnchorNet(pretrained=False, refine_iters=2).to(DEV).eval()
    model.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=False)["model"])
    ds = DualAnchorDataset("val")
    s = 3.0
    pts_M = torch.tensor([[c * s - 15.0, r * s - 10.5, 0.0] for r in range(8) for c in range(11)],
                         device=DEV, dtype=torch.float32)
    rng = np.random.default_rng(0)

    def bin_of(omega):
        for (lo, hi), k in zip(BINS, BKEYS):
            if lo <= omega < hi:
                return k
        return None

    variants = ["ecd_gn_existing", "ecd_gn_plus_imu", "roi_gn_oracle_imu", "roi_gn_oracle_noimu",
                "roi_gn_pred_imu", "full_existing"]
    rec = {v: {k: {"add": [], "px": [], "ct": [], "cr": [], "ot": [], "orr": []} for k in BKEYS} for v in variants}
    seg = {m: {k: {"s": [], "t": []} for k in BKEYS} for m in ("network", "flow_mag", "ransac_foe")}
    pairs_meta = []

    # --- thresholds for the magnitude baseline (grid-searched on val itself, which favours the baseline)
    mag_grid = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    mag_scores = {m: [] for m in mag_grid}

    cache = []
    for idx in range(len(ds)):
        it = ds[idx]
        k = bin_of(it["omega"])
        if k is None:
            continue
        img0 = it["img0"][None].to(DEV)
        img1 = it["img1"][None].to(DEV)
        K = it["K"][None].to(DEV)
        lab = it["label"][None].to(DEV)
        R_cam_gt, t_cam_gt = it["R_cam"][None].to(DEV), it["t_cam"][None].to(DEV)
        R_obj_gt, t_obj_gt = it["R_obj"][None].to(DEV), it["t_obj"][None].to(DEV)
        dR = it["dR_imu"][None].to(DEV)

        # ---- existing variants (harness check)
        model.refine_iters = 0
        out_c = model(img0, img1, K, dR_imu=None, ablate_mode="no_anchor_gate")
        R_c, t_c = out_c["R_cam"], out_c["t_cam"]
        # ---- ECD-GN + IMU
        out_ci = model(img0, img1, K, dR_imu=dR, ablate_mode="no_anchor_gate")
        model.refine_iters = 2
        out_f = model(img0, img1, K, dR_imu=dR, ablate_mode="full")

        def add_rec(name, R_o, t_o, R_c_est, t_c_est):
            add_mm, px = metrics(R_o, t_o, R_obj_gt, t_obj_gt, K, pts_M)
            r = rec[name][k]
            r["add"].append(add_mm)
            r["px"].append(px)
            r["ct"].append(float((t_c_est - t_cam_gt).norm(dim=-1)[0].item()))
            r["cr"].append(float(geodesic_deg(R_c_est, R_cam_gt)[0].item()))
            r["ot"].append(float((t_o - t_obj_gt).norm(dim=-1)[0].item()))
            r["orr"].append(float(geodesic_deg(R_o, R_obj_gt)[0].item()))

        add_rec("ecd_gn_existing", R_c, t_c, R_c, t_c)
        add_rec("ecd_gn_plus_imu", out_ci["R_cam"], out_ci["t_cam"], out_ci["R_cam"], out_ci["t_cam"])
        add_rec("full_existing", out_f["R_obj"], out_f["t_obj"], out_f["R_cam"], out_f["t_cam"])

        # ---- direct target-ROI fits on the full model's depth and flow
        Z, f_obs = out_f["Z_mm"], out_f["f_obs"]
        h, w = Z.shape[-2:]
        u, v = model._grid(h, w, DEV, f_obs.dtype)
        gt_roi = (lab == 1).float()
        pred_roi = (out_f["prob"][:, 1] > 0.5).float()
        zeros_t = torch.zeros(1, 3, device=DEV)
        eye = torch.eye(3, device=DEV)[None]
        for name, wts, R0 in (("roi_gn_oracle_imu", gt_roi, dR), ("roi_gn_oracle_noimu", gt_roi, eye),
                              ("roi_gn_pred_imu", pred_roi, dR)):
            if float(wts.sum()) < 1.0:
                wts = torch.ones_like(wts)
            R_o, t_o, _, _ = model.solve_se3(Z, u.float(), v.float(), K.float(), f_obs.float(), wts, R0, zeros_t,
                                             R_prior=None, rot_prec=0.0)
            add_rec(name, R_o, t_o, out_f["R_cam"], out_f["t_cam"])

        # ---- segmentation baselines on de-rotated flow
        gt_s, gt_m = (lab == 0), (lab == 1)
        pr_net = out_f["logits"].argmax(dim=1)
        seg["network"][k]["s"].append(iou(gt_s, pr_net == 0))
        seg["network"][k]["t"].append(iou(gt_m, pr_net == 1))
        f_rot = model.derotated_flow(u.float(), v.float(), K.float(), dR.float())  # (1,2,h,w) px
        f_res = (f_obs.float() - f_rot)[0]
        mag_map = f_res.norm(dim=0)
        cache.append((k, gt_s, gt_m, mag_map))
        f_res_np = f_res.cpu().numpy()
        inl = ransac_foe_mask(f_res_np, u.cpu().numpy(), v.cpu().numpy(), rng)
        inl_t = torch.from_numpy(inl).to(DEV)[None]
        seg["ransac_foe"][k]["s"].append(iou(gt_s, inl_t))
        seg["ransac_foe"][k]["t"].append(iou(gt_m, ~inl_t))
        pairs_meta.append({"i": idx, "bin": k})

    # magnitude-threshold baseline: pick the threshold maximising mean target IoU over all pairs
    for m in mag_grid:
        sc = [iou(gm, mg > m) for (_, gs, gm, mg) in cache]
        mag_scores[m] = float(np.mean(sc))
    best_m = max(mag_scores, key=mag_scores.get)
    for (k, gs, gm, mg) in cache:
        seg["flow_mag"][k]["s"].append(iou(gs, ~(mg > best_m)))
        seg["flow_mag"][k]["t"].append(iou(gm, mg > best_m))

    # ---- IMU sensitivity of the full model (pooled B3-B4)
    settings = [("clean", None, 0.0)]
    for deg in (0.1, 0.3, 1.0):
        settings.append((f"gyro_noise_{deg}deg", "noise", deg))
    for deg in (2.0, 5.0, 10.0):
        settings.append((f"extrinsic_{deg}deg", "extr", deg))
    for sc in (0.8, 1.2):
        settings.append((f"angle_scale_{sc}", "scale", sc))
    sens = {}
    E_axis = rand_axis()
    for name, kind, val in settings:
        acc = {"add": [], "px": [], "ct": []}
        accb4 = {"add": [], "px": [], "ct": []}
        for idx in range(len(ds)):
            it = ds[idx]
            k = bin_of(it["omega"])
            if k not in ("[1.0,2.0)", "[2.0,10.0)"):
                continue
            img0 = it["img0"][None].to(DEV)
            img1 = it["img1"][None].to(DEV)
            K = it["K"][None].to(DEV)
            R_obj_gt, t_obj_gt = it["R_obj"][None].to(DEV), it["t_obj"][None].to(DEV)
            t_cam_gt = it["t_cam"][None].to(DEV)
            dR = it["dR_imu"][None].to(DEV)
            if kind == "noise":
                eps = rand_axis() * np.deg2rad(val)
                dR = (so3_exp(eps) @ dR[0])[None]
            elif kind == "extr":
                E = so3_exp(E_axis * np.deg2rad(val))
                dR = (E @ dR[0] @ E.T)[None]
            elif kind == "scale":
                dR = so3_exp(so3_log(dR[0]) * val)[None]
            out = model(img0, img1, K, dR_imu=dR, ablate_mode="full")
            add_mm, px = metrics(out["R_obj"], out["t_obj"], R_obj_gt, t_obj_gt, K, pts_M)
            ct = float((out["t_cam"] - t_cam_gt).norm(dim=-1)[0].item())
            for a in (acc, accb4) if k == "[2.0,10.0)" else (acc,):
                a["add"].append(add_mm)
                a["px"].append(px)
                a["ct"].append(ct)
        sens[name] = {
            "B3B4": {m: round(float(np.median(v)), 3) for m, v in acc.items()} | {"n": len(acc["add"])},
            "B4": {m: round(float(np.median(v)), 3) for m, v in accb4.items()} | {"n": len(accb4["add"])},
        }
        print("sens", name, sens[name]["B3B4"], flush=True)

    # ---- summaries
    def med(v):
        return round(float(np.median(v)), 3) if len(v) else None

    out = {"harness_check": {}, "variants": {}, "segmentation": {}, "imu_sensitivity": sens,
           "flow_mag_threshold_px": best_m, "flow_mag_grid_mean_target_iou": mag_scores}
    for v in variants:
        out["variants"][v] = {k: {"n": len(rec[v][k]["add"]), "add_mm": med(rec[v][k]["add"]), "px": med(rec[v][k]["px"]),
                                  "cam_trans_mm": med(rec[v][k]["ct"]), "cam_rot_deg": med(rec[v][k]["cr"]),
                                  "obj_trans_mm": med(rec[v][k]["ot"]), "obj_rot_deg": med(rec[v][k]["orr"])}
                              for k in BKEYS}
        pooled = [x for k in ("[1.0,2.0)", "[2.0,10.0)") for x in rec[v][k]["add"]]
        out["variants"][v]["B3B4_add_mm"] = med(pooled)
        out["variants"][v]["B3B4_px"] = med([x for k in ("[1.0,2.0)", "[2.0,10.0)") for x in rec[v][k]["px"]])
    for m in seg:
        out["segmentation"][m] = {k: {"static_iou": med(seg[m][k]["s"]), "target_iou": med(seg[m][k]["t"])} for k in BKEYS}
    # raw per-pair ADD arrays for paired tests / bootstrap
    out["raw_add_mm"] = {v: {k: [round(x, 4) for x in rec[v][k]["add"]] for k in BKEYS} for v in variants}
    out["raw_px"] = {v: {k: [round(x, 4) for x in rec[v][k]["px"]] for k in BKEYS} for v in variants}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", OUT)
    for v in variants:
        print(v, {k: (out["variants"][v][k]["add_mm"], out["variants"][v][k]["px"]) for k in BKEYS})
    for m in seg:
        print("seg", m, out["segmentation"][m])


if __name__ == "__main__":
    main()
