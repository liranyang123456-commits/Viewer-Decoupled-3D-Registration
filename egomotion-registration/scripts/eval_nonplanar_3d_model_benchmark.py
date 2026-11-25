#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Non-Planar 3D Model Registration & Model-Based 6DoF Tracking Benchmark.

Addresses reviewer requirements on target geometry, model-based baselines, and
3D-model anchoring (zero planar checkerboards):
  1. Non-planar 3D target meshes inside 3D endoscopic cavities (C3VD real colon
     organ meshes + curved folded tissue tunnels textured with real SCARED frames):
       - Train split (seeds 1000..1159): BOP LineMod 3D meshes {1, 2, 4, 6, 10, 11}
         + procedural 3D lobulated anatomical nodules on train SCARED textures [0..5].
       - Held-out Test split (seeds 2000..2119, N=120 5-frame 30 Hz clips, 30 per bin
         across B1=[0,0.5), B2=[0.5,1.0), B3=[1.0,2.0), B4=[2.0,3.8) rad/s):
         100% unseen 3D target geometries {BOP-05, BOP-08, BOP-09, 3D-Lesion+Grasper}
         inside unseen C3VD organ/tunnel geometries with held-out SCARED textures [6..11].
  2. Genuine Model-Based 6DoF Tracking Baselines (all given the exact 3D target mesh
     (V_m, F_m) and initial pose T_cm(t_0), evaluated on BOTH single-step t0->t1
     registration AND 5-frame chained continuous tracking t0->t1->t2->t3->t4):
       - Uncompensated (holds T_cm(t_0))
       - Gyro-Only Propagation (2D ATW / SO(3) IMU update on T_cm)
       - KLT + 3D-Model PnP + RANSAC (classical sparse feature + 3D mesh PnP tracker)
       - Model-GN (RBOT/ICG-style coarse-to-fine 3D silhouette contour + photometric GN)
       - Model-GN + IMU Init (same 3D-model contour + photometric GN warm-started by IMU)
       - ECD-GN + IMU (coupled dense SE(3) solver adapted on the same 3D training split)
       - RAFT + 3D-Model PnP + RANSAC (dense optical flow + z-buffer 3D mesh PnP)
       - EgoAnchorNet (Model-Free Relative SE(3) Chain)
       - EgoAnchorNet + 3D-Mesh GN (Model-Anchored Ours: fuses EgoAnchorNet's
         inertial-guided dynamic confidence w_obj & decoupled warm-start with known
         3D-mesh surface & silhouette Gauss-Newton alignment)

Outputs:
  validation/nonplanar_3d_model_benchmark.json
"""
from __future__ import annotations

import json
import math
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

sys.path.insert(0, r"D:\ego_motiion_Camera")

from endosim.lite.ego_anchor_net import EgoAnchorNet
from endosim.lite.egolite import se3_exp as torch_se3_exp, skew as torch_skew, so3_log
from endosim.render.camera import PinholeCamera
from endosim.render.gpu_rasterizer import (
    MeshGPU,
    compute_normals_gpu,
    get_glctx,
    new_buffers,
    render_mesh_gpu,
    shade_background_gpu,
)
from endosim.render.rasterizer import Material, SimMesh
from endosim.scene.objects import (
    BOP_MATERIAL,
    INSTRUMENT_MATERIAL,
    TISSUE_MATERIAL,
    load_bop_object,
    make_instrument,
)
from endosim.scene.organ import ORGAN_MESHES, OrganMeshScene, available_organs
from endosim.scene.tissue import TissueTunnel

ROOT = Path(r"D:\ISMAR2026\egomotion-registration")
CKPT_BASE = ROOT / "outputs" / "ego_anchor_net" / "ego_anchor_net.pth"
OUT_JSON = ROOT / "validation" / "nonplanar_3d_model_benchmark.json"
SCARED_ROOT = Path(r"E:\World_Agent_Enoscopy\datasets\SCARED")

W_HD, H_HD = 1280, 720
W_NET, H_NET = 320, 256
STRIDE = 8
H_G, W_G = H_NET // STRIDE, W_NET // STRIDE
FPS = 30.0
DT = 1.0 / FPS

BINS = [
    ("B1_[0.0,0.5)", 0.0, 0.5),
    ("B2_[0.5,1.0)", 0.5, 1.0),
    ("B3_[1.0,2.0)", 1.0, 2.0),
    ("B4_[2.0,3.8)", 2.0, 3.8),
]


# ---------------------------------------------------------------------------
# SE(3) & Camera Helpers (millimeters)
# ---------------------------------------------------------------------------

def skew_np(w: np.ndarray) -> np.ndarray:
    x, y, z = float(w[0]), float(w[1]), float(w[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def se3_exp_np(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """xi = [rho(3), theta(3)] -> (R, t)."""
    rho = np.asarray(xi[:3], dtype=np.float64)
    theta = np.asarray(xi[3:], dtype=np.float64)
    th = float(np.linalg.norm(theta))
    R = Rotation.from_rotvec(theta).as_matrix()
    W = skew_np(theta)
    if th < 1e-9:
        V = np.eye(3)
    else:
        V = np.eye(3) + (1.0 - math.cos(th)) / (th ** 2) * W + (th - math.sin(th)) / (th ** 3) * (W @ W)
    return R, V @ rho


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def load_device_b_camera() -> PinholeCamera:
    return PinholeCamera(1282.13, 1282.99, 610.59, 358.19, W_HD, H_HD)


def scaled_K(cam: PinholeCamera, w: int, h: int) -> np.ndarray:
    sx, sy = w / float(cam.width), h / float(cam.height)
    K = cam.K.copy()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K.astype(np.float32)


# ---------------------------------------------------------------------------
# Non-Planar 3D Target Assets & Real Endoscopic Cavity Scenes
# ---------------------------------------------------------------------------

def load_scared_textures() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Disjoint train [0..5] and test [6..11] real endoscopic textures from SCARED."""
    paths = sorted(SCARED_ROOT.glob("dataset_*/keyframe_*/Left_Image.png"))
    imgs = []
    for p in paths[:12]:
        bgr = cv2.imread(str(p))
        if bgr is not None:
            rgb = cv2.cvtColor(cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            imgs.append(rgb.astype(np.float32) / 255.0)
    if len(imgs) < 12:
        rng = np.random.default_rng(42)
        while len(imgs) < 12:
            imgs.append(rng.uniform(0.35, 0.85, (512, 512, 3)).astype(np.float32))
    return imgs[:6], imgs[6:12]


def make_3d_lobulated_lesion_or_grasper(rng: np.random.Generator, kind: str = "lesion_grasper",
                                        target_diam_mm: float = 42.0) -> tuple[SimMesh, Material, str]:
    """Constructs a genuine 3D non-planar surgical target:
    either a multi-lobed exophytic anatomical tumor/nodule or a 3D articulated
    grasper head + tissue nodule assembly (zero planar faces)."""
    n_lat, n_lon = 28, 40
    lat = np.linspace(0.04 * np.pi, 0.96 * np.pi, n_lat)
    lon = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    verts = []
    colors = []
    lobes = rng.uniform(-0.22, 0.28, size=(4, 3))
    freqs = rng.integers(2, 6, size=(4, 2))
    phases = rng.uniform(0, 2 * np.pi, size=4)
    base_col = np.array([0.82, 0.34, 0.30]) if kind == "lesion" else np.array([0.76, 0.42, 0.35])

    for i, th in enumerate(lat):
        for j, ph in enumerate(lon):
            d = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th) * 0.82])
            r = 1.0
            for k in range(4):
                r += 0.16 * math.sin(freqs[k, 0] * th + phases[k]) * math.cos(freqs[k, 1] * ph - phases[k])
            v = d * r
            verts.append(v)
            vascular = 0.5 + 0.5 * math.sin(7.0 * ph + 4.0 * th) * math.cos(5.0 * th - 3.0 * ph)
            c = np.clip(base_col * (0.78 + 0.34 * vascular) + np.array([0.08, -0.04, 0.03]) * math.cos(3 * th), 0.08, 0.96)
            colors.append(c)

    # Poles
    top_idx = len(verts)
    verts.append([0.0, 0.0, 0.85])
    colors.append(base_col)
    bot_idx = len(verts)
    verts.append([0.0, 0.0, -0.85])
    colors.append(base_col * 0.8)

    faces = []
    for j in range(n_lon):
        jn = (j + 1) % n_lon
        faces.append([top_idx, j, jn])
        faces.append([bot_idx, (n_lat - 1) * n_lon + jn, (n_lat - 1) * n_lon + j])
    for i in range(n_lat - 1):
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            a, b = i * n_lon + j, i * n_lon + jn
            c_, d_ = (i + 1) * n_lon + j, (i + 1) * n_lon + jn
            faces.append([a, c_, b])
            faces.append([b, c_, d_])

    verts = np.asarray(verts, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    if kind == "lesion_grasper":
        # Attach 3D surgical grasper jaws & distal wrist gripping the 3D lesion
        inst_mesh, _ = make_instrument(rng, shaft_radius=2.8, shaft_length=28.0, jaw_length=16.0, jaw_open=22.0)
        iv = inst_mesh.verts.copy() * 0.055 + np.array([0.85, 0.15, 0.15])
        ic = np.tile(np.array([[0.58, 0.62, 0.68]]), (len(iv), 1))
        iff = inst_mesh.faces + len(verts)
        verts = np.vstack([verts, iv])
        colors = np.vstack([colors, ic])
        faces = np.vstack([faces, iff])

    verts -= verts.mean(axis=0, keepdims=True)
    ext = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    verts *= (target_diam_mm * 1.35) / max(ext, 1e-6)
    mesh = SimMesh(verts, faces, colors=colors)
    mesh.compute_normals()
    # Verify outward normals
    tri = mesh.verts[mesh.faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if float(np.mean(np.einsum("ij,ij->i", fn, tri.mean(axis=1)) > 0)) < 0.5:
        mesh.flip_winding()

    mat = Material(albedo=(1.0, 1.0, 1.0), ambient=0.10, ks=0.52, shininess=65.0, d_ref=95.0, atten_max=4.5)
    return mesh, mat, kind


def load_normalized_bop_target(obj_id: int, target_diam_mm: float = 42.0) -> tuple[SimMesh, Material, str]:
    mesh, _ = load_bop_object(obj_id, scale=1.0)
    max_span = float((mesh.verts.max(axis=0) - mesh.verts.min(axis=0)).max())
    mesh.verts = (mesh.verts - mesh.verts.mean(axis=0, keepdims=True)) * (target_diam_mm / max(max_span, 1e-6))
    # Preserve authentic BOP CAD vertex colors (no synthetic sinusoidal grid texture injection!)
    mesh.compute_normals()
    mat = Material(albedo=(1.0, 1.0, 1.0), ambient=0.11, ks=0.48, shininess=58.0, d_ref=95.0, atten_max=4.5)
    return mesh, mat, f"BOP_{obj_id:02d}"


def build_cavity_pool(device: str) -> tuple[list[dict], list[dict]]:
    """Preloads disjoint train and test 3D endoscopic cavity meshes onto GPU."""
    rng_tr = np.random.default_rng(101)
    rng_te = np.random.default_rng(202)
    organs = available_organs()

    train_cavs, test_cavs = [], []
    # Train cavities: 2 procedural folded tunnels + cecum_mold (scaled to surgical cavity)
    for k in range(3):
        tun = TissueTunnel(rng_tr, length=270.0, radius_base=84.0 + 6.0 * k, fold_amplitude=0.14, fold_count=6)
        train_cavs.append({"name": f"train_tunnel_{k}", "scene": tun, "gpu": MeshGPU(tun.mesh, device=device)})
    if "cecum_mold" in organs:
        om = OrganMeshScene(ORGAN_MESHES["cecum_mold"][0], rng_tr, decimate_faces=18000, tex_scale=0.025, scale=1.35)
        train_cavs.append({"name": "cecum_mold", "scene": om, "gpu": MeshGPU(om.mesh, device=device)})

    # Test cavities (strictly disjoint): 2 unseen folded tunnels + unseen C3VD sigmoid_mold / trans_mold
    for k in range(2):
        tun = TissueTunnel(rng_te, length=275.0, radius_base=86.0 + 5.0 * k, fold_amplitude=0.16, fold_count=7)
        test_cavs.append({"name": f"test_tunnel_{k}", "scene": tun, "gpu": MeshGPU(tun.mesh, device=device)})
    for oname in ("sigmoid_mold", "trans_mold"):
        if oname in organs:
            om = OrganMeshScene(ORGAN_MESHES[oname][0], rng_te, decimate_faces=18000, tex_scale=0.025, scale=1.45)
            test_cavs.append({"name": oname, "scene": om, "gpu": MeshGPU(om.mesh, device=device)})

    return train_cavs, test_cavs


# ---------------------------------------------------------------------------
# GPU Rendering with Motion Blur, Coaxial Specularities & Noisy 200 Hz IMU
# ---------------------------------------------------------------------------

def render_scene_frame_gpu(glctx, cam: PinholeCamera, cav_gpu: MeshGPU, cav_tex_gpu: torch.Tensor,
                           obj_gpu: MeshGPU, T_wc: np.ndarray, T_wo: np.ndarray,
                           obj_mat: Material, tissue_mat: Material, device: str = "cuda") -> dict:
    """Renders 1280x720 RGB, metric depth (mm), instance mask (1=bg, 2=3D target),
    and barycentric correspondence via nvdiffrast."""
    T_cw = inv_T(T_wc)
    T_co = T_cw @ T_wo
    bufs = new_buffers(cam.height, cam.width, device=device, track=True, track_normals=True)

    v_cav = cav_gpu.verts_with_disp(None)
    n_cav = cav_gpu.static_normals if cav_gpu.static_normals is not None else compute_normals_gpu(v_cav, cav_gpu.faces_long)
    render_mesh_gpu(glctx, bufs, cav_gpu, v_cav, n_cav, cam, T_cw, 1, tissue_mat, cav_tex_gpu, near=8.0)

    v_obj = obj_gpu.verts_with_disp(None)
    n_obj = obj_gpu.static_normals if obj_gpu.static_normals is not None else compute_normals_gpu(v_obj, obj_gpu.faces_long)
    render_mesh_gpu(glctx, bufs, obj_gpu, v_obj, n_obj, cam, T_co, 2, obj_mat, None, near=8.0)
    shade_background_gpu(bufs)
    return bufs


def apply_locked_shutter_and_sensor(rgb_f32: np.ndarray, omega_cam: np.ndarray, rng: np.random.Generator,
                                    t_exp_s: float = 0.002) -> np.ndarray:
    """Applies physically realistic directional motion blur along the camera's instantaneous
    rotational image-plane velocity (omega_y, -omega_x) + sensor readout noise."""
    om_mag = float(np.linalg.norm(omega_cam[:2]))
    blur_px = int(round(1282.0 * om_mag * t_exp_s))
    out = rgb_f32 * 255.0
    if blur_px >= 3:
        ksize = min(blur_px | 1, 25)
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        vx, vy = float(omega_cam[1]), float(-omega_cam[0])
        vn = max(math.hypot(vx, vy), 1e-6)
        vx, vy = vx / vn, vy / vn
        c = ksize // 2
        for s in np.linspace(-c, c, ksize * 2 + 1):
            ix = int(np.clip(round(c + vx * s), 0, ksize - 1))
            iy = int(np.clip(round(c + vy * s), 0, ksize - 1))
            kernel[iy, ix] += 1.0
        kernel /= max(float(kernel.sum()), 1e-6)
        out = cv2.filter2D(out, -1, kernel)
    noise = rng.normal(0.0, 2.5, size=out.shape).astype(np.float32)
    return np.clip(np.round(out + noise), 0, 255).astype(np.uint8)


def simulate_noisy_imu_increment(R_wc0: np.ndarray, R_wc1: np.ndarray, dt: float,
                                 rng: np.random.Generator, bias_body: np.ndarray,
                                 dR_extrinsic: np.ndarray) -> np.ndarray:
    """Simulates 200 Hz MEMS gyroscope preintegration over dt with white noise,
    turn-on bias, and residual camera-IMU extrinsic calibration error.
    Returns dR_imu_cam (3x3) mapping frame-0 rays to frame-1 rays (R1^T R0)."""
    dR_body_true = R_wc0.T @ R_wc1
    omega_body = Rotation.from_matrix(dR_body_true).as_rotvec() / dt
    n_sub = max(int(round(dt * 200.0)), 4)
    dt_sub = dt / n_sub
    R_acc = np.eye(3)
    for _ in range(n_sub):
        w_meas = dR_extrinsic @ omega_body + bias_body + rng.normal(0.0, 0.015, size=3)
        w_corr = dR_extrinsic.T @ w_meas  # nominal calibration applied
        R_acc = R_acc @ Rotation.from_rotvec(w_corr * dt_sub).as_matrix()
    return R_acc.T.astype(np.float32)


def generate_nonplanar_clip(glctx, cam: PinholeCamera, cav_entry: dict, tex_gpu: torch.Tensor,
                            obj_mesh: SimMesh, obj_gpu: MeshGPU, obj_mat: Material, obj_name: str,
                            omega_range: tuple[float, float], n_frames: int,
                            seed: int, device: str = "cuda") -> dict | None:
    """Generates a continuous n_frames 30 Hz clip of simultaneous 6DoF camera motion
    (within omega_range rad/s) and independent 6DoF non-planar 3D target motion."""
    rng = np.random.default_rng(seed)
    scene = cav_entry["scene"]
    arc_max = float(scene.arc[-1])
    s0 = float(rng.uniform(0.12 * arc_max, 0.30 * arc_max))
    T_wc0 = scene.axis_pose_at(s0, roll=float(rng.uniform(-25.0, 25.0)))

    tissue_mat = Material(albedo=(0.94, 0.74, 0.68), ambient=0.09, ks=0.26, shininess=38.0, d_ref=115.0, atten_max=4.2)

    # Camera angular velocity & linear velocity in camera-0 frame
    omega_mag = float(rng.uniform(omega_range[0], omega_range[1]))
    rot_axis = rng.normal(size=3)
    rot_axis[2] *= 0.45  # pitch/yaw dominant like handheld/robotic scope sweeps
    rot_axis /= max(float(np.linalg.norm(rot_axis)), 1e-8)
    omega_cam0 = rot_axis * omega_mag
    vel_cam0 = rng.uniform([-95.0, -75.0, -60.0], [95.0, 75.0, 60.0])  # mm/s

    # Place 3D non-planar target inside camera frustum ahead of cavity wall
    z0 = float(rng.uniform(118.0, 148.0))
    # Bias initial placement slightly opposite to sweep direction so target stays inside FOV across all 5 frames
    sweep_u = float(omega_cam0[1] * z0 * DT * (n_frames - 1) * 0.45)
    sweep_v = float(-omega_cam0[0] * z0 * DT * (n_frames - 1) * 0.45)
    p_co0 = np.array([sweep_u + rng.uniform(-12.0, 12.0),
                      sweep_v + rng.uniform(-8.0, 8.0),
                      z0], dtype=np.float64)
    R_co0 = Rotation.from_euler("xyz", rng.uniform([-35, -35, -60], [35, 35, 60]), degrees=True).as_matrix()
    T_wo0 = T_wc0 @ make_T(R_co0, p_co0)

    # Independent 6DoF target motion in world space (surgical organ shift / instrument manipulation)
    v_obj_w = T_wc0[:3, :3] @ rng.uniform([-85.0, -75.0, -55.0], [85.0, 75.0, 55.0])
    w_obj_local = rng.normal(size=3)
    w_obj_local /= max(float(np.linalg.norm(w_obj_local)), 1e-8)
    w_obj_local *= float(rng.uniform(0.65, 1.85))  # rad/s independent object rotation

    bias_body = rng.normal(0.0, 0.012, size=3)
    dR_ext = Rotation.from_rotvec(rng.normal(0.0, np.radians(0.35), size=3)).as_matrix()

    frames_rgb, frames_endo6ms, depths_g, labels_g = [], [], [], []
    T_wcs, T_wos, T_cms = [], [], []
    dR_imus, omegas = [], []

    for k in range(n_frames):
        tk = k * DT
        # Slightly time-varying angular acceleration so trajectory is non-constant-velocity
        mod = 1.0 + 0.15 * math.sin(2.0 * math.pi * 2.5 * tk + float(seed % 7))
        dR_c = Rotation.from_rotvec(omega_cam0 * tk * mod).as_matrix()
        dp_c = vel_cam0 * tk
        T_wc = T_wc0 @ make_T(dR_c, dp_c)

        dR_o = Rotation.from_rotvec(w_obj_local * tk).as_matrix()
        T_wo = make_T(T_wo0[:3, :3] @ dR_o, T_wo0[:3, 3] + v_obj_w * tk)
        T_cm = inv_T(T_wc) @ T_wo

        bufs = render_scene_frame_gpu(glctx, cam, cav_entry["gpu"], tex_gpu, obj_gpu,
                                      T_wc, T_wo, obj_mat, tissue_mat, device=device)
        inst = bufs["instance"]
        obj_frac = float((inst == 2).float().mean().item())
        bg_frac = float((inst == 1).float().mean().item())
        if obj_frac < 0.025 or obj_frac > 0.35 or bg_frac < 0.45:
            return None

        # Check target vertices stay comfortably inside frame
        pts_c = (T_cm[:3, :3] @ obj_mesh.verts.T).T + T_cm[:3, 3]
        if pts_c[:, 2].min() < 35.0:
            return None
        uv_chk, _ = cam.project(pts_c)
        if (uv_chk[:, 0].min() < 25 or uv_chk[:, 0].max() > W_HD - 25 or
                uv_chk[:, 1].min() < 20 or uv_chk[:, 1].max() > H_HD - 20):
            return None

        rgb_f32 = bufs["color"].cpu().numpy()
        rgb_u8 = apply_locked_shutter_and_sensor(rgb_f32, omega_cam0, rng, t_exp_s=0.002)
        rgb_endo6ms_u8 = apply_locked_shutter_and_sensor(rgb_f32, omega_cam0, rng, t_exp_s=0.0065)

        # Downsample exact depth & binary dynamic label to 32x40 network supervision grid
        depth_t = bufs["depth"]
        depth_clean = torch.where(torch.isfinite(depth_t), depth_t, torch.full_like(depth_t, 200.0))
        d_net = F.interpolate(depth_clean[None, None], size=(H_G, W_G), mode="area")[0, 0].cpu().numpy().astype(np.float32)
        lbl_f = F.interpolate((inst == 2).float()[None, None], size=(H_G, W_G), mode="area")[0, 0].cpu().numpy()
        lbl_g = (lbl_f >= 0.35).astype(np.int64)

        frames_rgb.append(rgb_u8)
        frames_endo6ms.append(rgb_endo6ms_u8)
        depths_g.append(d_net)
        labels_g.append(lbl_g)
        T_wcs.append(T_wc)
        T_wos.append(T_wo)
        T_cms.append(T_cm)

        if k > 0:
            dR_c_step = T_wcs[k - 1][:3, :3].T @ T_wcs[k][:3, :3]
            om_step = float(np.linalg.norm(Rotation.from_matrix(dR_c_step).as_rotvec()) / DT)
            omegas.append(om_step)
            dR_imus.append(simulate_noisy_imu_increment(T_wcs[k - 1][:3, :3], T_wcs[k][:3, :3], DT, rng, bias_body, dR_ext))

    return {
        "seed": seed,
        "obj_name": obj_name,
        "cavity": cav_entry["name"],
        "mesh": obj_mesh,
        "mesh_gpu": obj_gpu,
        "frames_hd": frames_rgb,
        "frames_endo6ms": frames_endo6ms,
        "depths_g": depths_g,
        "labels_g": labels_g,
        "T_wcs": T_wcs,
        "T_wos": T_wos,
        "T_cms": T_cms,
        "dR_imus": dR_imus,
        "omegas": omegas,
        "omega_01": omegas[0],
    }


# ---------------------------------------------------------------------------
# Visible 3D Mesh Sampling & Model-Based 6DoF Tracking Baselines
# ---------------------------------------------------------------------------

def sample_visible_3d_model_points(glctx, cam: PinholeCamera, mesh: SimMesh, mesh_gpu: MeshGPU,
                                   T_cm_prev: np.ndarray, max_pts: int = 320,
                                   device: str = "cuda") -> dict:
    """Renders ONLY the known 3D target mesh at pose T_cm_prev via GPU z-buffer to obtain:
      - visible 3D surface points X_m (in object frame, mm) & X_c (in camera frame, mm),
      - outward surface normals n_c in camera frame,
      - 2D projections uv_prev in 1280x720 image,
      - rendered binary mask & 3D silhouette contour vertices X_m_sil + 2D outward normals!
    """
    bufs = new_buffers(cam.height, cam.width, device=device, track=True, track_normals=True)
    v_loc = mesh_gpu.verts_with_disp(None)
    n_loc = mesh_gpu.static_normals if mesh_gpu.static_normals is not None else compute_normals_gpu(v_loc, mesh_gpu.faces_long)
    render_mesh_gpu(glctx, bufs, mesh_gpu, v_loc, n_loc, cam, T_cm_prev, 1, BOP_MATERIAL, None, near=8.0)

    mask = (bufs["instance"] == 1).cpu().numpy().astype(np.uint8)
    tri_map = bufs["tri"].cpu().numpy()
    bary_map = bufs["bary"].cpu().numpy()
    nrm_map = bufs["normal"].cpu().numpy()

    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 20:
        return {"ok": False, "mask": mask}

    # Stratified subsample across visible surface pixels
    step = max(1, len(xs) // max_pts)
    ys_s, xs_s = ys[::step][:max_pts], xs[::step][:max_pts]
    f_ids = tri_map[ys_s, xs_s]
    b0 = bary_map[ys_s, xs_s, 0]
    b1 = bary_map[ys_s, xs_s, 1]
    b2 = 1.0 - b0 - b1

    tri_v = mesh.verts[mesh.faces[f_ids]]  # (M, 3, 3)
    X_m = b0[:, None] * tri_v[:, 0] + b1[:, None] * tri_v[:, 1] + b2[:, None] * tri_v[:, 2]
    X_c = (T_cm_prev[:3, :3] @ X_m.T).T + T_cm_prev[:3, 3]
    n_c = nrm_map[ys_s, xs_s]
    uv_prev = np.stack([xs_s + 0.5, ys_s + 0.5], axis=1).astype(np.float64)

    # Viewing cosine & coaxial specular glint indicator (light is coaxial at camera origin)
    view_dir = -X_c / np.maximum(np.linalg.norm(X_c, axis=1, keepdims=True), 1e-6)
    cos_theta = np.clip(np.einsum("ij,ij->i", n_c, view_dir), 0.0, 1.0)
    # Down-weight extreme grazing rims (self-occluding) and mirror-specular centers (cos > 0.985)
    glint_pen = np.where(cos_theta > 0.985, 0.35, 1.0)
    w_geom = (np.clip((cos_theta - 0.18) / 0.82, 0.05, 1.0) ** 1.5) * glint_pen

    # Extract 3D silhouette contour points for RBOT/ICG-style contour alignment
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    rim_y, rim_x = np.nonzero((mask > 0) & (eroded == 0))
    if len(rim_x) >= 16:
        r_step = max(1, len(rim_x) // 120)
        ry, rx = rim_y[::r_step][:120], rim_x[::r_step][:120]
        rf = tri_map[ry, rx]
        rb0, rb1 = bary_map[ry, rx, 0], bary_map[ry, rx, 1]
        rb2 = 1.0 - rb0 - rb1
        rtv = mesh.verts[mesh.faces[rf]]
        X_m_sil = rb0[:, None] * rtv[:, 0] + rb1[:, None] * rtv[:, 1] + rb2[:, None] * rtv[:, 2]
    else:
        X_m_sil = X_m[:min(80, len(X_m))]

    return {
        "ok": True,
        "mask": mask,
        "X_m": X_m,
        "X_c": X_c,
        "n_c": n_c,
        "uv_prev": uv_prev,
        "w_geom": w_geom,
        "X_m_sil": X_m_sil,
        "tri_map": tri_map,
        "bary_map": bary_map,
    }


def track_klt_3d_model_pnp(img0_hd: np.ndarray, img1_hd: np.ndarray, cam: PinholeCamera,
                           mesh: SimMesh, vis: dict, T_cm_prev: np.ndarray) -> np.ndarray:
    """Baseline 3: Classical Sparse Feature KLT + Known 3D-Model PnP + RANSAC + LM."""
    if not vis["ok"]:
        return T_cm_prev.copy()
    g0 = cv2.cvtColor(img0_hd, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(img1_hd, cv2.COLOR_RGB2GRAY)

    # Combine Shi-Tomasi keypoints on rendered 3D mask + visible surface grid points
    kp = cv2.goodFeaturesToTrack(g0, maxCorners=160, qualityLevel=0.01, minDistance=5, mask=vis["mask"])
    pts0_list, Xm_list = [], []
    if kp is not None and len(kp) > 0:
        for u, v in kp.reshape(-1, 2):
            ix, iy = int(np.clip(round(u), 0, W_HD - 1)), int(np.clip(round(v), 0, H_HD - 1))
            fid = int(vis["tri_map"][iy, ix])
            if fid >= 0:
                b0, b1 = vis["bary_map"][iy, ix]
                tv = mesh.verts[mesh.faces[fid]]
                pts0_list.append([u, v])
                Xm_list.append(b0 * tv[0] + b1 * tv[1] + (1.0 - b0 - b1) * tv[2])

    # Supplement with visible mesh points
    pts0_all = np.vstack([np.array(pts0_list, dtype=np.float32).reshape(-1, 2),
                          vis["uv_prev"].astype(np.float32)])
    Xm_all = np.vstack([np.array(Xm_list, dtype=np.float64).reshape(-1, 3),
                        vis["X_m"]])

    p0_cv = pts0_all.reshape(-1, 1, 2)
    p1_cv, st_f, _ = cv2.calcOpticalFlowPyrLK(
        g0, g1, p0_cv, None, winSize=(25, 25), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    p0_back, st_b, _ = cv2.calcOpticalFlowPyrLK(
        g1, g0, p1_cv, None, winSize=(25, 25), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    fb_err = np.linalg.norm((p0_back - p0_cv).reshape(-1, 2), axis=1)
    good = (st_f.ravel() == 1) & (st_b.ravel() == 1) & (fb_err < 2.5)
    if int(good.sum()) < 8:
        return T_cm_prev.copy()

    obj_pts = Xm_all[good].astype(np.float64)
    img_pts = p1_cv.reshape(-1, 2)[good].astype(np.float64)
    rvec0, _ = cv2.Rodrigues(T_cm_prev[:3, :3])
    tvec0 = T_cm_prev[:3, 3].reshape(3, 1).copy()
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, cam.K, None, rvec=rvec0.copy(), tvec=tvec0.copy(),
            useExtrinsicGuess=True, iterationsCount=250, reprojectionError=3.5,
            confidence=0.99, flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < 6:
            return T_cm_prev.copy()
        inl = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj_pts[inl], img_pts[inl], cam.K, None, rvec, tvec)
        R_est, _ = cv2.Rodrigues(rvec)
        t_est = tvec.reshape(3)
        if not np.all(np.isfinite(t_est)) or t_est[2] < 25.0 or t_est[2] > 400.0:
            return T_cm_prev.copy()
        return make_T(R_est, t_est)
    except cv2.error:
        return T_cm_prev.copy()


def build_edge_chamfer_and_photo_pyr(img_rgb: np.ndarray) -> list[dict]:
    """Builds 3-level coarse-to-fine pyramid of luminance + Chamfer distance-to-edge
    fields (and spatial gradients) for RBOT/ICG-style 3D model contour + direct tracking."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    pyr = []
    for s in (0.25, 0.5, 1.0):
        g = cv2.resize(gray, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else gray.copy()
        g_blur = cv2.GaussianBlur(g, (5, 5), 1.1)
        gx = cv2.Sobel(g_blur, cv2.CV_32F, 1, 0, ksize=3) * 0.125
        gy = cv2.Sobel(g_blur, cv2.CV_32F, 0, 1, ksize=3) * 0.125
        mag = np.sqrt(gx * gx + gy * gy)
        thr = float(np.percentile(mag, 82))
        edges = (mag >= max(thr, 3.0)).astype(np.uint8)
        dist = cv2.distanceTransform(1 - edges, cv2.DIST_L2, 3).astype(np.float32)
        dist = np.minimum(dist, 28.0)
        dx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3) * 0.125
        dy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3) * 0.125
        pyr.append({"scale": s, "lum": g_blur, "gx": gx, "gy": gy, "dist": dist, "dx": dx, "dy": dy})
    return pyr


def bilinear_sample_np(field: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = field.shape[:2]
    x = np.clip(u - 0.5, 0.0, w - 1.001)
    y = np.clip(v - 0.5, 0.0, h - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    dx = x - x0
    dy = y - y0
    return ((1 - dx) * (1 - dy) * field[y0, x0] +
            dx * (1 - dy) * field[y0, x0 + 1] +
            (1 - dx) * dy * field[y0 + 1, x0] +
            dx * dy * field[y0 + 1, x0 + 1])


def track_rbot_icg_3d_model_gn(img0_hd: np.ndarray, img1_hd: np.ndarray, cam: PinholeCamera,
                               vis: dict, T_init: np.ndarray) -> np.ndarray:
    """Baselines 4 & 5: RBOT/ICG-Style Coarse-to-Fine 3D-Model Contour + Photometric
    Gauss-Newton Tracker (from T_cm_prev alone OR warm-started with IMU rotation)."""
    if not vis["ok"]:
        return T_init.copy()

    g0 = cv2.GaussianBlur(cv2.cvtColor(img0_hd, cv2.COLOR_RGB2GRAY).astype(np.float32), (5, 5), 1.1)
    ref_lum = bilinear_sample_np(g0, vis["uv_prev"][:, 0], vis["uv_prev"][:, 1])
    pyr1 = build_edge_chamfer_and_photo_pyr(img1_hd)

    R = T_init[:3, :3].copy()
    t = T_init[:3, 3].copy()
    X_int = vis["X_m"]
    w_int = vis["w_geom"]
    X_sil = vis["X_m_sil"]

    for lvl in pyr1:
        s = lvl["scale"]
        fx, fy, cx, cy = cam.fx * s, cam.fy * s, cam.cx * s, cam.cy * s
        h_s, w_s = lvl["lum"].shape[:2]

        for _ in range(6):
            # 1) Direct 3D-mesh surface photometric residual
            Xc = (R @ X_int.T).T + t
            Z = np.maximum(Xc[:, 2], 15.0)
            u = fx * Xc[:, 0] / Z + cx
            v = fy * Xc[:, 1] / Z + cy
            inb = (u >= 2) & (u < w_s - 2) & (v >= 2) & (v < h_s - 2)
            if int(inb.sum()) < 15:
                break

            I1 = bilinear_sample_np(lvl["lum"], u, v)
            Ix = bilinear_sample_np(lvl["gx"], u, v)
            Iy = bilinear_sample_np(lvl["gy"], u, v)
            r_p = (I1 - ref_lum) / 25.0
            huber_p = np.where(np.abs(r_p) > 1.2, 1.2 / np.maximum(np.abs(r_p), 1e-6), 1.0)
            wp = w_int * huber_p * inb.astype(np.float64)

            # Jacobian of pixel projection wrt left-perturbation delta in se(3): X' = X + rho - X x theta
            invZ = 1.0 / Z
            invZ2 = invZ * invZ
            du_dX = np.stack([fx * invZ, np.zeros_like(Z), -fx * Xc[:, 0] * invZ2], axis=1)
            dv_dX = np.stack([np.zeros_like(Z), fy * invZ, -fy * Xc[:, 1] * invZ2], axis=1)
            g_dX = Ix[:, None] * du_dX + Iy[:, None] * dv_dX  # (M, 3)
            J_p = np.hstack([g_dX, np.cross(Xc, g_dX)]) / 25.0  # (M, 6)

            # 2) 3D silhouette contour Chamfer edge-alignment residual
            Xs = (R @ X_sil.T).T + t
            Zs = np.maximum(Xs[:, 2], 15.0)
            us = fx * Xs[:, 0] / Zs + cx
            vs = fy * Xs[:, 1] / Zs + cy
            inbs = (us >= 2) & (us < w_s - 2) & (vs >= 2) & (vs < h_s - 2)
            D1 = bilinear_sample_np(lvl["dist"], us, vs)
            Dx = bilinear_sample_np(lvl["dx"], us, vs)
            Dy = bilinear_sample_np(lvl["dy"], us, vs)
            r_c = D1 / 6.0
            huber_c = np.where(np.abs(r_c) > 1.5, 1.5 / np.maximum(np.abs(r_c), 1e-6), 1.0)
            wc = 1.6 * huber_c * inbs.astype(np.float64)

            invZs = 1.0 / Zs
            invZs2 = invZs * invZs
            dus_dX = np.stack([fx * invZs, np.zeros_like(Zs), -fx * Xs[:, 0] * invZs2], axis=1)
            dvs_dX = np.stack([np.zeros_like(Zs), fy * invZs, -fy * Xs[:, 1] * invZs2], axis=1)
            d_dX = Dx[:, None] * dus_dX + Dy[:, None] * dvs_dX
            J_c = np.hstack([d_dX, np.cross(Xs, d_dX)]) / 6.0

            H = (J_p.T * wp) @ J_p + (J_c.T * wc) @ J_c + np.diag([1e-2, 1e-2, 1e-2, 5e-1, 5e-1, 5e-1])
            b = -((J_p.T * wp) @ r_p + (J_c.T * wc) @ r_c)
            try:
                delta = np.linalg.solve(H, b)
            except np.linalg.LinAlgError:
                break
            nt = float(np.linalg.norm(delta[:3]))
            nr = float(np.linalg.norm(delta[3:]))
            if nt > 12.0:
                delta[:3] *= 12.0 / nt
            if nr > 0.12:
                delta[3:] *= 0.12 / nr
            dR, dt_ = se3_exp_np(delta)
            R = dR @ R
            t = dR @ t + dt_
            if nt < 0.03 and nr < 1e-4:
                break

    if not np.all(np.isfinite(t)) or t[2] < 25.0 or t[2] > 400.0:
        return T_init.copy()
    return make_T(R, t)


def track_raft_3d_model_pnp(flow_net_320: np.ndarray, cam: PinholeCamera,
                            vis: dict, T_cm_prev: np.ndarray) -> np.ndarray:
    """Baseline 7: Dense Optical Flow (RAFT) + Known 3D-Model PnP + RANSAC + LM."""
    if not vis["ok"]:
        return T_cm_prev.copy()

    # Scale uv_prev to 320x256 flow map, sample dense RAFT flow, map back to 1280x720
    sx, sy = W_NET / float(W_HD), H_NET / float(H_HD)
    u_s = vis["uv_prev"][:, 0] * sx
    v_s = vis["uv_prev"][:, 1] * sy
    fx_flow = bilinear_sample_np(flow_net_320[0], u_s, v_s) / sx
    fy_flow = bilinear_sample_np(flow_net_320[1], u_s, v_s) / sy
    uv_target = vis["uv_prev"] + np.stack([fx_flow, fy_flow], axis=1)

    inb = ((uv_target[:, 0] >= 4) & (uv_target[:, 0] < W_HD - 4) &
           (uv_target[:, 1] >= 4) & (uv_target[:, 1] < H_HD - 4))
    if int(inb.sum()) < 10:
        return T_cm_prev.copy()

    obj_pts = vis["X_m"][inb].astype(np.float64)
    img_pts = uv_target[inb].astype(np.float64)
    rvec0, _ = cv2.Rodrigues(T_cm_prev[:3, :3])
    tvec0 = T_cm_prev[:3, 3].reshape(3, 1).copy()
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, cam.K, None, rvec=rvec0.copy(), tvec=tvec0.copy(),
            useExtrinsicGuess=True, iterationsCount=250, reprojectionError=3.5,
            confidence=0.99, flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < 8:
            return T_cm_prev.copy()
        inl = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj_pts[inl], img_pts[inl], cam.K, None, rvec, tvec)
        R_est, _ = cv2.Rodrigues(rvec)
        t_est = tvec.reshape(3)
        if not np.all(np.isfinite(t_est)) or t_est[2] < 25.0 or t_est[2] > 400.0:
            return T_cm_prev.copy()
        return make_T(R_est, t_est)
    except cv2.error:
        return T_cm_prev.copy()


def refine_egoanchor_with_3d_mesh_gn(T_warm: np.ndarray, dR_imu: np.ndarray,
                                     flow_net_320: np.ndarray, prob_obj_32x40: np.ndarray,
                                     w_obj_grid: np.ndarray, img0_hd: np.ndarray,
                                     img1_hd: np.ndarray, cam: PinholeCamera,
                                     mesh: SimMesh, vis: dict) -> np.ndarray:
    """Ours (Inertial-Derotated & Dynamic-Anchor-Gated 3D-Mesh PnP-GN):
    Applies Eq. (5) / Proposition 1 directly at full 1280x720 resolution:
      1. De-rotates img1_hd into img0_hd's orientation via H_01 = K @ dR_imu @ K^-1,
         collapsing 60-140 px fast camera whip-pans into a small ~10-20 px residual
         and eliminating rotational patch deformation before Lucas-Kanade matching.
      2. Extracts dense interior non-grazing 3D mesh points + Shi-Tomasi surface corners,
         strictly pruning colon-wall background leakage via EgoAnchorNet's inertial
         dynamic-object gate prob_obj >= 0.30 (crucial when T_prev is perturbed!).
      3. Maps validated subpixel matches back via H_01, solves EPnP-RANSAC + weighted
         IRLS Huber 3D-Mesh Gauss-Newton, and falls back seamlessly to EgoAnchorNet's
         dense w_obj solver on textureless / heavily blurred frames."""
    if not vis["ok"]:
        return T_warm.copy()

    g0 = cv2.cvtColor(img0_hd, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(img1_hd, cv2.COLOR_RGB2GRAY)

    # 1) Exact depth-independent IMU rotational homography de-rotation (Eq. 5) with Lanczos4
    rot_ang_deg = float(np.degrees(np.arccos(np.clip((np.trace(dR_imu) - 1.0) * 0.5, -1.0, 1.0))))
    if rot_ang_deg >= 0.8:
        H_01 = cam.K @ dR_imu @ np.linalg.inv(cam.K)
        g1_derot = cv2.warpPerspective(
            g1, H_01, (W_HD, H_HD),
            flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        use_derot = True
    else:
        H_01 = np.eye(3, dtype=np.float64)
        g1_derot = g1
        use_derot = False

    # 2) Full visible 3D surface mask (preserves thin 3D surgical grasper jaws!)
    kp = cv2.goodFeaturesToTrack(g0, maxCorners=180, qualityLevel=0.01, minDistance=5, mask=vis["mask"])
    pts0_list, Xm_list = [], []
    if kp is not None and len(kp) > 0:
        for u, v in kp.reshape(-1, 2):
            ix, iy = int(np.clip(round(u), 0, W_HD - 1)), int(np.clip(round(v), 0, H_HD - 1))
            fid = int(vis["tri_map"][iy, ix])
            if fid >= 0:
                b0, b1 = vis["bary_map"][iy, ix]
                tv = mesh.verts[mesh.faces[fid]]
                pts0_list.append([u, v])
                Xm_list.append(b0 * tv[0] + b1 * tv[1] + (1.0 - b0 - b1) * tv[2])

    uv0_all = np.vstack([np.array(pts0_list, dtype=np.float64).reshape(-1, 2), vis["uv_prev"]])
    Xm_all = np.vstack([np.array(Xm_list, dtype=np.float64).reshape(-1, 3), vis["X_m"]])

    # 3) EgoAnchorNet dynamic-object gate: prune static colon-wall background when mask is misaligned!
    sx, sy = W_NET / float(W_HD), H_NET / float(H_HD)
    prob_up = cv2.resize(prob_obj_32x40.astype(np.float32), (W_NET, H_NET), interpolation=cv2.INTER_LINEAR)
    p_dyn = bilinear_sample_np(prob_up, uv0_all[:, 0] * sx, uv0_all[:, 1] * sy)
    # Only prune if a substantial fraction of vis['mask'] sits on static colon wall (p_dyn < 0.20)
    bg_leak_frac = float(np.mean(p_dyn < 0.20))
    if bg_leak_frac >= 0.12 and int((p_dyn >= 0.25).sum()) >= 35:
        keep = p_dyn >= 0.25
        uv0_use, Xm_use = uv0_all[keep], Xm_all[keep]
    else:
        uv0_use, Xm_use = uv0_all, Xm_all

    # 4) Pyramidal LK on IMU-derotated full-HD pair (small residual flow, zero patch rotation!)
    p0_cv = uv0_use.astype(np.float32).reshape(-1, 1, 2)
    p1_derot, st_f, _ = cv2.calcOpticalFlowPyrLK(
        g0, g1_derot, p0_cv, None, winSize=(25, 25), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    p0_back, st_b, _ = cv2.calcOpticalFlowPyrLK(
        g1_derot, g0, p1_derot, None, winSize=(25, 25), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    fb_err = np.linalg.norm((p0_back - p0_cv).reshape(-1, 2), axis=1)
    good = (st_f.ravel() == 1) & (st_b.ravel() == 1) & (fb_err < 2.0)

    if int(good.sum()) >= 8:
        pts_d = p1_derot.reshape(-1, 2)[good].astype(np.float64)
        if use_derot:
            hom = (H_01 @ np.c_[pts_d, np.ones(len(pts_d))].T).T
            img_pts = hom[:, :2] / np.maximum(hom[:, 2:3], 1e-8)
        else:
            img_pts = pts_d
        inb = ((img_pts[:, 0] >= 2) & (img_pts[:, 0] < W_HD - 2) &
               (img_pts[:, 1] >= 2) & (img_pts[:, 1] < H_HD - 2))
        obj_pts = Xm_use[good][inb]
        img_pts = img_pts[inb]

        if len(obj_pts) >= 8:
            rvec_w, _ = cv2.Rodrigues(T_warm[:3, :3])
            tvec_w = T_warm[:3, 3].reshape(3, 1).copy()
            try:
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_pts, img_pts, cam.K, None, rvec=rvec_w.copy(), tvec=tvec_w.copy(),
                    useExtrinsicGuess=True, iterationsCount=300, reprojectionError=2.8,
                    confidence=0.995, flags=cv2.SOLVEPNP_EPNP,
                )
                if ok and inliers is not None and len(inliers) >= 6:
                    inl = inliers.ravel()
                    rvec, tvec = cv2.solvePnPRefineLM(obj_pts[inl], img_pts[inl], cam.K, None, rvec, tvec)
                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec.reshape(3)
                    if np.all(np.isfinite(t)) and 25.0 < t[2] < 400.0:
                        return make_T(R, t)
            except cv2.error:
                pass

    # 5) Textureless / heavy motion-blur fallback: w_obj-weighted dense 3D-Mesh Huber GN around T_warm
    u_s = vis["uv_prev"][:, 0] * sx
    v_s = vis["uv_prev"][:, 1] * sy
    fx_flow = bilinear_sample_np(flow_net_320[0], u_s, v_s) / sx
    fy_flow = bilinear_sample_np(flow_net_320[1], u_s, v_s) / sy
    uv_obs = vis["uv_prev"] + np.stack([fx_flow, fy_flow], axis=1)

    w_up = cv2.resize(w_obj_grid.astype(np.float32), (W_NET, H_NET), interpolation=cv2.INTER_LINEAR)
    w_dyn = bilinear_sample_np(w_up, u_s, v_s)
    w_dyn = w_dyn / max(float(np.percentile(w_dyn, 90)), 1e-6)
    w_base = np.clip(w_dyn, 0.01, 1.6) * vis["w_geom"]

    R = T_warm[:3, :3].copy()
    t = T_warm[:3, 3].copy()
    X_m = vis["X_m"]
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy

    for _ in range(6):
        Xc = (R @ X_m.T).T + t
        Z = np.maximum(Xc[:, 2], 15.0)
        u = fx * Xc[:, 0] / Z + cx
        v = fy * Xc[:, 1] / Z + cy
        eu = u - uv_obs[:, 0]
        ev = v - uv_obs[:, 1]
        en = np.sqrt(eu * eu + ev * ev)
        huber = np.where(en > 3.0, 3.0 / np.maximum(en, 1e-6), 1.0)
        w = w_base * huber

        invZ = 1.0 / Z
        invZ2 = invZ * invZ
        du_dX = np.stack([fx * invZ, np.zeros_like(Z), -fx * Xc[:, 0] * invZ2], axis=1)
        dv_dX = np.stack([np.zeros_like(Z), fy * invZ, -fy * Xc[:, 1] * invZ2], axis=1)
        Ju = np.hstack([du_dX, np.cross(Xc, du_dX)])
        Jv = np.hstack([dv_dX, np.cross(Xc, dv_dX)])

        H = (Ju.T * w) @ Ju + (Jv.T * w) @ Jv + np.diag([0.05, 0.05, 0.05, 2.0, 2.0, 2.0])
        b = -((Ju.T * w) @ eu + (Jv.T * w) @ ev)
        try:
            delta = np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            break
        nt = float(np.linalg.norm(delta[:3]))
        nr = float(np.linalg.norm(delta[3:]))
        if nt > 12.0:
            delta[:3] *= 12.0 / nt
        if nr > 0.12:
            delta[3:] *= 0.12 / nr
        dR, dt_ = se3_exp_np(delta)
        R = dR @ R
        t = dR @ t + dt_
        if nt < 0.02 and nr < 1e-4:
            break

    if not np.all(np.isfinite(t)) or t[2] < 25.0 or t[2] > 400.0:
        return T_warm.copy()
    return make_T(R, t)


# ---------------------------------------------------------------------------
# Training Adaptation on Disjoint Non-Planar 3D Objects
# ---------------------------------------------------------------------------

def extract_pair_supervision(clip: dict, k0: int, k1: int, K_net: np.ndarray) -> dict:
    im0 = cv2.resize(clip["frames_hd"][k0], (W_NET, H_NET), interpolation=cv2.INTER_AREA)
    im1 = cv2.resize(clip["frames_hd"][k1], (W_NET, H_NET), interpolation=cv2.INTER_AREA)
    T_wc0, T_wc1 = clip["T_wcs"][k0], clip["T_wcs"][k1]
    T_cm0, T_cm1 = clip["T_cms"][k0], clip["T_cms"][k1]

    R_cam = (T_wc1[:3, :3].T @ T_wc0[:3, :3]).astype(np.float32)
    t_cam = (T_wc1[:3, :3].T @ (T_wc0[:3, 3] - T_wc1[:3, 3])).astype(np.float32)

    R_obj = (T_cm1[:3, :3] @ T_cm0[:3, :3].T).astype(np.float32)
    t_obj = (T_cm1[:3, 3] - R_obj @ T_cm0[:3, 3]).astype(np.float32)

    return {
        "img0": im0,
        "img1": im1,
        "K": K_net,
        "depth0": clip["depths_g"][k0],
        "label": clip["labels_g"][k0],
        "R_cam": R_cam,
        "t_cam": t_cam,
        "R_obj": R_obj,
        "t_obj": t_obj,
        "dR_imu": clip["dR_imus"][k0],
    }


def rot_loss_t(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    return so3_log(R_pred @ R_gt.transpose(-1, -2)).norm(dim=-1).mean()


def train_nonplanar_models(train_pairs: list[dict], device: str, epochs: int = 12,
                           bs: int = 16, lr: float = 1.8e-4) -> tuple[EgoAnchorNet, EgoAnchorNet]:
    """Trains BOTH EgoAnchorNet (full) and fair coupled ECD-GN+IMU (no_anchor_gate)
    on the exact same disjoint non-planar 3D training pairs."""
    models = {}
    for mode in ("full", "coupled"):
        m = EgoAnchorNet(pretrained=False, refine_iters=2).to(device)
        m.load_state_dict(torch.load(CKPT_BASE, map_location=device, weights_only=False)["model"])
        for p in m.flow.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        rng = np.random.default_rng(7701 if mode == "full" else 7702)

        t0 = time.perf_counter()
        for ep in range(epochs):
            m.train()
            order = rng.permutation(len(train_pairs))
            losses = []
            for s in range(0, len(order) - bs + 1, bs):
                batch = [train_pairs[i] for i in order[s:s + bs]]
                opt.zero_grad(set_to_none=True)
                im0 = torch.stack([torch.from_numpy(x["img0"]).permute(2, 0, 1) for x in batch]).float().to(device) / 255.0
                im1 = torch.stack([torch.from_numpy(x["img1"]).permute(2, 0, 1) for x in batch]).float().to(device) / 255.0
                K = torch.stack([torch.from_numpy(x["K"]) for x in batch]).to(device)
                lbl = torch.stack([torch.from_numpy(x["label"]) for x in batch]).to(device)
                z = torch.stack([torch.from_numpy(x["depth0"]) for x in batch]).to(device)
                dR = torch.stack([torch.from_numpy(x["dR_imu"]) for x in batch]).to(device)
                Rc = torch.stack([torch.from_numpy(x["R_cam"]) for x in batch]).to(device)
                tc = torch.stack([torch.from_numpy(x["t_cam"]) for x in batch]).to(device)
                Ro = torch.stack([torch.from_numpy(x["R_obj"]) for x in batch]).to(device)
                to = torch.stack([torch.from_numpy(x["t_obj"]) for x in batch]).to(device)

                if mode == "full":
                    out = m(im0, im1, K, dR_imu=dR, ablate_mode="full")
                    cw = torch.tensor([1.0, 5.5], device=device)
                    loss = (
                        1.4 * F.cross_entropy(out["logits"], lbl, weight=cw)
                        + 1.2 * F.l1_loss(torch.log(out["Z_mm"].clamp(min=10.0)), torch.log(z.clamp(min=10.0)))
                        + 2.0 * rot_loss_t(out["R_cam"], Rc)
                        + 1.5 * F.smooth_l1_loss(out["t_cam"], tc, beta=2.0) / 10.0
                        + 3.0 * rot_loss_t(out["R_obj"], Ro)
                        + 2.5 * F.smooth_l1_loss(out["t_obj"], to, beta=2.0) / 10.0
                    )
                else:
                    out = m(im0, im1, K, dR_imu=dR, ablate_mode="no_anchor_gate")
                    loss = (
                        1.2 * F.l1_loss(torch.log(out["Z_mm"].clamp(min=10.0)), torch.log(z.clamp(min=10.0)))
                        + 3.0 * rot_loss_t(out["R_cam"], Ro)
                        + 2.5 * F.smooth_l1_loss(out["t_cam"], to, beta=2.0) / 10.0
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0)
                opt.step()
                losses.append(float(loss.item()))
            sched.step()
            if (ep + 1) in (1, 6, epochs):
                print(f"  [train-{mode}] ep {ep+1:2d}/{epochs} loss={np.mean(losses):.3f} ({time.perf_counter()-t0:.1f}s)", flush=True)
        models[mode] = m.eval()
    return models["full"], models["coupled"]


# ---------------------------------------------------------------------------
# Metrics & Statistical Summaries
# ---------------------------------------------------------------------------

def pose_metrics_on_3d_mesh(T_est: np.ndarray, T_gt: np.ndarray, verts_mm: np.ndarray,
                            cam: PinholeCamera) -> dict:
    # Subsample up to 500 evenly spaced mesh vertices for exact 3D ADD & 2D projection RMSE
    step = max(1, len(verts_mm) // 500)
    V = verts_mm[::step]
    X_est = (T_est[:3, :3] @ V.T).T + T_est[:3, 3]
    X_gt = (T_gt[:3, :3] @ V.T).T + T_gt[:3, 3]
    add_mm = float(np.linalg.norm(X_est - X_gt, axis=1).mean())

    uv_est, _ = cam.project(X_est)
    uv_gt, _ = cam.project(X_gt)
    px_err = float(np.linalg.norm(uv_est - uv_gt, axis=1).mean())

    dR = T_est[:3, :3].T @ T_gt[:3, :3]
    cos_a = float(np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0))
    rot_deg = float(np.degrees(np.arccos(cos_a)))
    trans_mm = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
    return {"add_mm": add_mm, "px": px_err, "rot_deg": rot_deg, "trans_mm": trans_mm}


def summarize_array(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    q25, q50, q75 = np.percentile(a, [25, 50, 75])
    return {
        "n": int(len(a)),
        "median": round(float(q50), 3),
        "iqr": round(float(q75 - q25), 3),
        "mean": round(float(a.mean()), 3),
        "sd": round(float(a.std(ddof=0)), 3),
        "add_10pct_recall": round(float(np.mean(a < 4.2) * 100.0), 1),  # 10% of 42 mm object diameter!
    }


def paired_stat(a_vals: list[float], b_vals: list[float], seed: int = 0) -> dict:
    """Paired difference (a - b): positive means b (Ours) has lower error than a."""
    a = np.asarray(a_vals, dtype=np.float64)
    b = np.asarray(b_vals, dtype=np.float64)
    diff = a - b
    rng = np.random.default_rng(seed)
    boots = [float(np.median(diff[rng.integers(0, len(diff), size=len(diff))])) for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    try:
        _, p_val = wilcoxon(diff)
    except Exception:
        p_val = 1.0
    return {
        "median_paired_gain": round(float(np.median(diff)), 3),
        "ci95": [round(float(lo), 3), round(float(hi), 3)],
        "wilcoxon_p": float(p_val),
    }


def main() -> None:
    t_start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    glctx = get_glctx()
    cam = load_device_b_camera()
    K_net = scaled_K(cam, W_NET, H_NET)
    K_net_t = torch.from_numpy(K_net[None]).to(device)

    print("1) Loading C3VD 3D cavities, SCARED endoscopic textures, and disjoint 3D non-planar targets...", flush=True)
    train_texs, test_texs = load_scared_textures()
    train_tex_gpus = [torch.from_numpy(t).to(device) for t in train_texs]
    test_tex_gpus = [torch.from_numpy(t).to(device) for t in test_texs]
    train_cavs, test_cavs = build_cavity_pool(device)

    # Disjoint train 3D targets
    rng_asset = np.random.default_rng(500)
    train_targets = []
    for bid in (1, 2, 4, 6, 10, 11):
        m, mat, name = load_normalized_bop_target(bid, target_diam_mm=42.0)
        train_targets.append((m, MeshGPU(m, device=device), mat, name))
    for k in range(2):
        m, mat, name = make_3d_lobulated_lesion_or_grasper(rng_asset, kind="lesion", target_diam_mm=42.0)
        train_targets.append((m, MeshGPU(m, device=device), mat, f"train_lesion_{k}"))

    # Strictly unseen held-out test 3D targets (zero planar checkerboards!)
    test_targets = []
    for bid in (5, 8, 9):
        m, mat, name = load_normalized_bop_target(bid, target_diam_mm=42.0)
        test_targets.append((m, MeshGPU(m, device=device), mat, name))
    m_lg, mat_lg, _ = make_3d_lobulated_lesion_or_grasper(np.random.default_rng(999), kind="lesion_grasper", target_diam_mm=42.0)
    test_targets.append((m_lg, MeshGPU(m_lg, device=device), mat_lg, "3D_Lesion_Grasper"))

    print("2) Generating non-planar 3D training pairs (strictly disjoint objects/textures/cavities)...", flush=True)
    train_pairs = []
    seed_cur = 1000
    for b_idx, (_, lo, hi) in enumerate(BINS):
        got_bin = 0
        while got_bin < 20:
            seed_cur += 1
            m, mgpu, mat, name = train_targets[seed_cur % len(train_targets)]
            cav = train_cavs[seed_cur % len(train_cavs)]
            tex = train_tex_gpus[seed_cur % len(train_tex_gpus)]
            clip = generate_nonplanar_clip(glctx, cam, cav, tex, m, mgpu, mat, name, (lo, hi), 4, seed_cur, device=device)
            if clip is None:
                continue
            for k in range(3):
                train_pairs.append(extract_pair_supervision(clip, k, k + 1, K_net))
            got_bin += 1
    print(f"   Generated {len(train_pairs)} non-planar 3D training pairs across B1..B4.", flush=True)

    print("3) Adapting EgoAnchorNet & fair coupled ECD-GN+IMU on non-planar 3D training split...", flush=True)
    ego_model, ecd_model = train_nonplanar_models(train_pairs, device=device, epochs=12, bs=16, lr=1.8e-4)

    print("4) Generating held-out unseen 3D test clips (30 clips/bin x 4 bins = 120 5-frame clips, 480 steps)...", flush=True)
    test_clips_by_bin: dict[str, list[dict]] = {b[0]: [] for b in BINS}
    seed_te = 2000
    for b_name, lo, hi in BINS:
        while len(test_clips_by_bin[b_name]) < 30:
            seed_te += 1
            t_idx = len(test_clips_by_bin[b_name]) % len(test_targets)
            m, mgpu, mat, name = test_targets[t_idx]
            cav = test_cavs[seed_te % len(test_cavs)]
            tex = test_tex_gpus[seed_te % len(test_tex_gpus)]
            clip = generate_nonplanar_clip(glctx, cam, cav, tex, m, mgpu, mat, name, (lo, hi), 5, seed_te, device=device)
            if clip is not None and (lo <= clip["omega_01"] < hi):
                clip["bin"] = b_name
                test_clips_by_bin[b_name].append(clip)
        print(f"   {b_name}: 30 unseen 3D clips ready.", flush=True)

    METHODS = [
        "Uncompensated",
        "Gyro_Only_ATW",
        "KLT_3DModel_PnP",
        "Model_GN_RBOT_ICG",
        "Model_GN_IMU_Init",
        "ECD_GN_IMU",
        "RAFT_3DModel_PnP",
        "EgoAnchorNet_Rel",
        "EgoAnchorNet_3DMesh_GN_Ours",
    ]

    def run_regime(frame_key: str, init_trans_mm: float, init_rot_deg: float, label_tag: str) -> dict:
        single_records = {m: {b[0]: [] for b in BINS} for m in METHODS}
        chain_records = {m: {b[0]: [] for b in BINS} for m in METHODS}
        per_obj_records = {m: {} for m in METHODS}
        runtimes_ms = {m: [] for m in METHODS}
        iou_vals = []
        rng_p = np.random.default_rng(90210)

        print(f"5-{label_tag}) Evaluating 9 methods on [{frame_key}] (init_perturb={init_trans_mm}mm/{init_rot_deg}deg)...", flush=True)
        for b_name, _, _ in BINS:
            for clip in test_clips_by_bin[b_name]:
                mesh = clip["mesh"]
                mgpu = clip["mesh_gpu"]
                oname = clip["obj_name"]
                frames_seq = clip[frame_key]
                for m in METHODS:
                    per_obj_records[m].setdefault(oname, [])

                T0_init = clip["T_cms"][0].copy()
                if init_trans_mm > 0.0 or init_rot_deg > 0.0:
                    d_axis = rng_p.normal(size=3)
                    d_axis /= max(float(np.linalg.norm(d_axis)), 1e-8)
                    d_rot = Rotation.from_rotvec(d_axis * np.radians(init_rot_deg)).as_matrix()
                    d_tr = rng_p.normal(size=3)
                    d_tr = d_tr / max(float(np.linalg.norm(d_tr)), 1e-8) * init_trans_mm
                    T0_init = make_T(d_rot @ T0_init[:3, :3], T0_init[:3, 3] + d_tr)

                # Maintain chained pose state T_cm_hat(t_k) for each method, initialized ONLY at t_0
                T_chain = {m: T0_init.copy() for m in METHODS}

                for step_k in range(4):
                    img0_hd = frames_seq[step_k]
                    img1_hd = frames_seq[step_k + 1]
                    T_gt_next = clip["T_cms"][step_k + 1]
                    dR_imu = clip["dR_imus"][step_k].astype(np.float64)

                    im0_s = cv2.resize(img0_hd, (W_NET, H_NET), interpolation=cv2.INTER_AREA)
                    im1_s = cv2.resize(img1_hd, (W_NET, H_NET), interpolation=cv2.INTER_AREA)
                    im0_t = torch.from_numpy(im0_s).permute(2, 0, 1).float()[None].to(device) / 255.0
                    im1_t = torch.from_numpy(im1_s).permute(2, 0, 1).float()[None].to(device) / 255.0
                    dR_t = torch.from_numpy(clip["dR_imus"][step_k][None]).to(device)

                    with torch.no_grad():
                        t_a = time.perf_counter()
                        flow_320_t = ego_model.observed_flow(im0_t, im1_t)
                        torch.cuda.synchronize()
                        dt_raft_ms = (time.perf_counter() - t_a) * 1000.0
                        flow_320 = flow_320_t[0].cpu().numpy()

                        t_a = time.perf_counter()
                        out_ego = ego_model(im0_t, im1_t, K_net_t, dR_imu=dR_t, ablate_mode="full")
                        torch.cuda.synchronize()
                        dt_ego_ms = (time.perf_counter() - t_a) * 1000.0

                        t_a = time.perf_counter()
                        out_ecd = ecd_model(im0_t, im1_t, K_net_t, dR_imu=dR_t, ablate_mode="no_anchor_gate")
                        torch.cuda.synchronize()
                        dt_ecd_ms = (time.perf_counter() - t_a) * 1000.0

                    if step_k == 0:
                        pred_m = (out_ego["prob"][0, 1].cpu().numpy() >= 0.5)
                        gt_m = (clip["labels_g"][0] == 1)
                        inter = float(np.logical_and(pred_m, gt_m).sum())
                        union = float(np.logical_or(pred_m, gt_m).sum())
                        if union > 0:
                            iou_vals.append(inter / union * 100.0)

                    R_obj_ego = out_ego["R_obj"][0].cpu().numpy().astype(np.float64)
                    t_obj_ego = out_ego["t_obj"][0].cpu().numpy().astype(np.float64)
                    w_obj_ego = out_ego["w_obj"][0].cpu().numpy().astype(np.float32)

                    R_obj_ecd = out_ecd["R_cam"][0].cpu().numpy().astype(np.float64)
                    t_obj_ecd = out_ecd["t_cam"][0].cpu().numpy().astype(np.float64)

                    # Share visible z-buffer query at step_k == 0 when all methods share T0_init!
                    vis_shared0 = sample_visible_3d_model_points(glctx, cam, mesh, mgpu, T0_init, device=device) if step_k == 0 else None

                    # --- 1. Uncompensated ---
                    T_chain["Uncompensated"] = T_chain["Uncompensated"].copy()
                    runtimes_ms["Uncompensated"].append(0.01)

                    # --- 2. Gyro-Only / 2D ATW ---
                    t_a = time.perf_counter()
                    Tc = T_chain["Gyro_Only_ATW"]
                    T_chain["Gyro_Only_ATW"] = make_T(dR_imu @ Tc[:3, :3], dR_imu @ Tc[:3, 3])
                    runtimes_ms["Gyro_Only_ATW"].append((time.perf_counter() - t_a) * 1000.0)

                    # --- 3. KLT + 3D-Model PnP + RANSAC ---
                    t_a = time.perf_counter()
                    vis_klt = vis_shared0 if vis_shared0 is not None else sample_visible_3d_model_points(glctx, cam, mesh, mgpu, T_chain["KLT_3DModel_PnP"], device=device)
                    T_chain["KLT_3DModel_PnP"] = track_klt_3d_model_pnp(img0_hd, img1_hd, cam, mesh, vis_klt, T_chain["KLT_3DModel_PnP"])
                    runtimes_ms["KLT_3DModel_PnP"].append((time.perf_counter() - t_a) * 1000.0)

                    # --- 4. Model-GN (RBOT/ICG Contour + Photometric, no IMU) ---
                    t_a = time.perf_counter()
                    vis_gn = vis_shared0 if vis_shared0 is not None else sample_visible_3d_model_points(glctx, cam, mesh, mgpu, T_chain["Model_GN_RBOT_ICG"], device=device)
                    T_chain["Model_GN_RBOT_ICG"] = track_rbot_icg_3d_model_gn(img0_hd, img1_hd, cam, vis_gn, T_chain["Model_GN_RBOT_ICG"])
                    runtimes_ms["Model_GN_RBOT_ICG"].append((time.perf_counter() - t_a) * 1000.0)

                    # --- 5. Model-GN + IMU Init ---
                    t_a = time.perf_counter()
                    Tc_mi = T_chain["Model_GN_IMU_Init"]
                    vis_mi = vis_shared0 if vis_shared0 is not None else sample_visible_3d_model_points(glctx, cam, mesh, mgpu, Tc_mi, device=device)
                    T_imu_warm = make_T(dR_imu @ Tc_mi[:3, :3], dR_imu @ Tc_mi[:3, 3])
                    T_chain["Model_GN_IMU_Init"] = track_rbot_icg_3d_model_gn(img0_hd, img1_hd, cam, vis_mi, T_imu_warm)
                    runtimes_ms["Model_GN_IMU_Init"].append((time.perf_counter() - t_a) * 1000.0)

                    # --- 6. ECD-GN + IMU (Chained) ---
                    Tc_ecd = T_chain["ECD_GN_IMU"]
                    T_chain["ECD_GN_IMU"] = make_T(R_obj_ecd @ Tc_ecd[:3, :3], R_obj_ecd @ Tc_ecd[:3, 3] + t_obj_ecd)
                    runtimes_ms["ECD_GN_IMU"].append(dt_ecd_ms)

                    # --- 7. RAFT + 3D-Model PnP + RANSAC ---
                    t_a = time.perf_counter()
                    vis_raft = vis_shared0 if vis_shared0 is not None else sample_visible_3d_model_points(glctx, cam, mesh, mgpu, T_chain["RAFT_3DModel_PnP"], device=device)
                    T_chain["RAFT_3DModel_PnP"] = track_raft_3d_model_pnp(flow_320, cam, vis_raft, T_chain["RAFT_3DModel_PnP"])
                    runtimes_ms["RAFT_3DModel_PnP"].append(dt_raft_ms + (time.perf_counter() - t_a) * 1000.0)

                    # --- 8. EgoAnchorNet (Model-Free Relative SE(3) Chain) ---
                    Tc_rel = T_chain["EgoAnchorNet_Rel"]
                    T_chain["EgoAnchorNet_Rel"] = make_T(R_obj_ego @ Tc_rel[:3, :3], R_obj_ego @ Tc_rel[:3, 3] + t_obj_ego)
                    runtimes_ms["EgoAnchorNet_Rel"].append(dt_ego_ms)

                    # --- 9. EgoAnchorNet + 3D-Mesh GN (Model-Anchored Ours) ---
                    t_a = time.perf_counter()
                    Tc_ours = T_chain["EgoAnchorNet_3DMesh_GN_Ours"]
                    prob_obj_ego = out_ego["prob"][0, 1].cpu().numpy().astype(np.float32)
                    vis_ours = vis_shared0 if vis_shared0 is not None else sample_visible_3d_model_points(glctx, cam, mesh, mgpu, Tc_ours, device=device)
                    T_warm_ours = make_T(R_obj_ego @ Tc_ours[:3, :3], R_obj_ego @ Tc_ours[:3, 3] + t_obj_ego)
                    T_chain["EgoAnchorNet_3DMesh_GN_Ours"] = refine_egoanchor_with_3d_mesh_gn(
                        T_warm_ours, dR_imu, flow_320, prob_obj_ego, w_obj_ego, img0_hd, img1_hd, cam, mesh, vis_ours
                    )
                    runtimes_ms["EgoAnchorNet_3DMesh_GN_Ours"].append(dt_ego_ms + (time.perf_counter() - t_a) * 1000.0)

                    for m in METHODS:
                        met = pose_metrics_on_3d_mesh(T_chain[m], T_gt_next, mesh.verts, cam)
                        if step_k == 0:
                            single_records[m][b_name].append(met)
                        if step_k == 3:
                            chain_records[m][b_name].append(met)
                            per_obj_records[m][oname].append(met)

        def build_table(rec_dict: dict) -> dict:
            out = {}
            for m in METHODS:
                out[m] = {"by_bin": {}, "all": {}}
                all_add, all_px, all_rot = [], [], []
                for b_name, _, _ in BINS:
                    rows = rec_dict[m][b_name]
                    adds = [r["add_mm"] for r in rows]
                    pxs = [r["px"] for r in rows]
                    rots = [r["rot_deg"] for r in rows]
                    all_add.extend(adds)
                    all_px.extend(pxs)
                    all_rot.extend(rots)
                    out[m]["by_bin"][b_name] = {
                        "add_mm": summarize_array(adds),
                        "px_1280x720": summarize_array(pxs),
                        "rot_deg": summarize_array(rots),
                    }
                out[m]["all"] = {
                    "add_mm": summarize_array(all_add),
                    "px_1280x720": summarize_array(all_px),
                    "rot_deg": summarize_array(all_rot),
                    "median_runtime_ms": round(float(np.median(runtimes_ms[m])), 2),
                }
            return out

        single_summary = build_table(single_records)
        chain_summary = build_table(chain_records)
        per_object_summary = {
            oname: {m: summarize_array([r["add_mm"] for r in per_obj_records[m][oname]]) for m in METHODS}
            for oname in sorted(per_obj_records["EgoAnchorNet_3DMesh_GN_Ours"].keys())
        }

        ours_single_all = [r["add_mm"] for b, _, _ in BINS for r in single_records["EgoAnchorNet_3DMesh_GN_Ours"][b]]
        ours_chain_all = [r["add_mm"] for b, _, _ in BINS for r in chain_records["EgoAnchorNet_3DMesh_GN_Ours"][b]]
        ours_chain_b4 = [r["add_mm"] for r in chain_records["EgoAnchorNet_3DMesh_GN_Ours"]["B4_[2.0,3.8)"]]

        paired_tests = {"single_step_all_N120": {}, "chained_t4_all_N120": {}, "chained_t4_B4_N30": {}}
        for m in METHODS:
            if m == "EgoAnchorNet_3DMesh_GN_Ours":
                continue
            m_s = [r["add_mm"] for b, _, _ in BINS for r in single_records[m][b]]
            m_c = [r["add_mm"] for b, _, _ in BINS for r in chain_records[m][b]]
            m_b4 = [r["add_mm"] for r in chain_records[m]["B4_[2.0,3.8)"]]
            paired_tests["single_step_all_N120"][f"{m}_minus_Ours_mm"] = paired_stat(m_s, ours_single_all)
            paired_tests["chained_t4_all_N120"][f"{m}_minus_Ours_mm"] = paired_stat(m_c, ours_chain_all)
            paired_tests["chained_t4_B4_N30"][f"{m}_minus_Ours_mm"] = paired_stat(m_b4, ours_chain_b4)

        return {
            "mask_iou_median_pct": round(float(np.median(iou_vals)), 2),
            "single_step_t0_to_t1": single_summary,
            "chained_5frame_t0_to_t4": chain_summary,
            "per_unseen_3d_object_chained_add_mm": per_object_summary,
            "paired_statistics": paired_tests,
        }

    reg_ideal = run_regime("frames_hd", 0.0, 0.0, "Ideal-2ms-ExactInit")
    reg_clinical = run_regime("frames_endo6ms", 3.5, 2.5, "Clinical-6.5ms-PerturbedInit")

    all_b4_omegas = [c["omega_01"] for c in test_clips_by_bin["B4_[2.0,3.8)"]]
    report = {
        "protocol": {
            "description": "Non-planar 3D model registration & 5-frame 30 Hz chained 6DoF model tracking benchmark (zero planar checkerboards).",
            "camera": "Device-B 1280x720 (fx=1282.13, fy=1282.99, cx=610.59, cy=358.19), 30 Hz + coaxial Phong specularities",
            "imu": "Simulated 200 Hz MEMS gyro with white noise (0.015 rad/s), turn-on bias (0.012 rad/s), and 0.35 deg extrinsic residual",
            "train_split": "240 pairs on BOP {1,2,4,6,10,11} + procedural 3D nodules in train cavities + SCARED train textures [0..5]",
            "test_split": "120 unseen 5-frame clips (30/bin across B1..B4, 480 steps) on strictly unseen 3D targets {BOP_05, BOP_08, BOP_09, 3D_Lesion_Grasper} (42 mm nominal diameter) in unseen C3VD/folded cavities + held-out SCARED textures [6..11]",
            "regimes": {
                "ideal_2ms_exact_init": "2.0 ms locked shutter, exact oracle initial registration T_cm(t_0)",
                "clinical_6p5ms_perturbed_init": "6.5 ms directional endoscopic shutter blur + 3.5 mm / 2.5 deg Stage-0 initial registration perturbation",
            },
            "b4_omega_rad_s": {
                "min": round(float(min(all_b4_omegas)), 3),
                "median": round(float(np.median(all_b4_omegas)), 3),
                "max": round(float(max(all_b4_omegas)), 3),
            },
        },
        "ideal_2ms_exact_init": reg_ideal,
        "clinical_6p5ms_perturbed_init": reg_clinical,
        "elapsed_s": round(time.perf_counter() - t_start, 1),
    }

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved verified dual-regime benchmark JSON to {OUT_JSON} in {report['elapsed_s']}s!\n", flush=True)

    for r_key, r_title in [("ideal_2ms_exact_init", "REGIME 1: IDEAL 2ms LOCKED SHUTTER + EXACT INIT"),
                           ("clinical_6p5ms_perturbed_init", "REGIME 2: CLINICAL 6.5ms BLUR + 3.5mm/2.5deg STAGE-0 PERTURBED INIT")]:
        cs = report[r_key]["chained_5frame_t0_to_t4"]
        ss = report[r_key]["single_step_t0_to_t1"]
        print(f"\n=== {r_title} ===")
        for m in METHODS:
            s_all = ss[m]["all"]["add_mm"]
            c_all = cs[m]["all"]["add_mm"]
            c_px = cs[m]["all"]["px_1280x720"]
            c_b4 = cs[m]["by_bin"]["B4_[2.0,3.8)"]["add_mm"]
            print(f"  {m:28s} | 1-Step: {s_all['median']:5.2f} mm | 5-Frame Chain: {c_all['median']:5.2f} [{c_all['iqr']:5.2f}] mm "
                  f"({c_px['median']:5.2f} px, R@10%={c_all['add_10pct_recall']:4.1f}%) | B4 Chain: {c_b4['median']:5.2f} mm")


if __name__ == "__main__":
    main()
