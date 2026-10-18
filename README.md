# Separating Viewer Motion from 3D Model Registration for Video See-Through Navigation

**Authors:** Ranyang Li\*, Nan Wei, Zhipeng Lin, Wufeng Liu, Chao Fan, and Junjun Pan\*  
**Target Venue:** *IEEE Transactions on Visualization and Computer Graphics (TVCG)*  
**Repository:** [https://github.com/liranyang123456-commits/Viewer-Decoupled-3D-Registration](https://github.com/liranyang123456-commits/Viewer-Decoupled-3D-Registration)  
**Companion Simulator & Hardware Capture Suite:** [https://github.com/liranyang123456-commits/EGO_Mo](https://github.com/liranyang123456-commits/EGO_Mo)

---

## Overview

Registering a 3D navigation model onto a live video see-through stream (4K camera + IMU or stereo endoscope + IMU) fails when the camera rotates rapidly, even when the physical target moves smoothly. This repository contains the counterfactual dual-motion benchmark summaries, angular-velocity-stratified evaluation protocol, figure generation scripts, and IEEE TVCG manuscript sources for **Viewer-Decoupled 3D Registration** and **EgoAnchorNet**.

### Three Core Contributions
1. **Inertial De-Rotated Static-Anchor Disentanglement and Cascaded Dual-$\mathrm{SE}(3)$ Solver (*EgoAnchorNet*):** Uses preintegrated IMU rotation $\Delta R_{\text{imu}}$ to remove depth-independent rotational optical flow before motion segmentation, reducing stationary background flow to a pure translational epipolar field (Proposition 1). Separates static scene anchors ($99.1\%$ IoU) from moving targets ($88.9\%$ IoU) and drives cascaded differentiable Gauss--Newton solvers for both $T_{cw}(t_e)$ and $T_{wm}(t_e)$.
2. **Two-Stage Viewer-Decoupled Registration and Display-Time Compensation Architecture:** Filters the scene-referenced model pose $T_{wm}(t_e) = T_{cw}(t_e)^{-1}T_{cm}(t_e)$ at the exposure midpoint $t_e$ and propagates only the viewer pose $T_{cw}$ across the measured pipeline delay $\tau = t_d - t_e$ via the rigidly mounted IMU.
3. **Counterfactual Dual-Motion Benchmark and Angular-Velocity-Stratified Verification:** Evaluates nine classical and SOTA baselines (**SIFT, ORB, KLT, LoFTR, RAFT, 2D ATW, VGGT, Reloc3r, and Pi3**) alongside **EgoAnchorNet** across four IMU angular-velocity bins ($[0.0,0.5), [0.5,1.0), [1.0,2.0), [2.0,10.0)$ rad/s), reporting both **Median [IQR]** and **Mean $\pm$ SD**.

---

## Head-to-Head Summary on the 40-Pair Dual-Motion Benchmark

| Method | Venue | Fast-Bin $[2.0, 10.0)$ rad/s Target Rot. Error ($^\circ$, Med [IQR] / Mean$\pm$SD) | All-Bin ($N=40$) Target Rot. Error ($^\circ$, Med [IQR] / Mean$\pm$SD) | Fast-Bin $1280\times720$ AR Shift (px, Med [IQR]) | Per-Pair GPU Time (ms) |
|---|---|---:|---:|---:|---:|
| SIFT + Essential | IJCV'04 | 7.69 [7.40] / 40.6$\pm$69.0 | 3.13 [8.49] / 20.5$\pm$46.0 | 42.1 [6.40] (Uncomp.) | 42 ms |
| 2D ATW Homography Warp | SIGGRAPH'25 Baseline | --- | --- | 52.4 [37.2] | $<1$ ms |
| Reloc3r | CVPR'25 | 3.78 [2.10] / 4.22$\pm$1.55 | 2.24 [1.56] / 2.56$\pm$1.78 | --- | 118 ms |
| Pi3 | arXiv'25 | 2.62 [1.48] / 2.89$\pm$1.03 | 2.01 [1.59] / 2.21$\pm$1.43 | --- | 95 ms |
| VGGT-1B | CVPR'25 | 1.94 [2.43] / 2.59$\pm$1.26 (3.20 Cond. 3) | 1.82 [1.32] / 2.29$\pm$1.46 | --- | 165 ms |
| Coupled EgoLite | Baseline | 1.48 [0.34] / 1.46$\pm$0.41 | 1.40 [0.52] / 1.41$\pm$0.53 | 15.0 [3.99] | 54 ms |
| **EgoAnchorNet (Ours, Target)** | **Ours** | **1.33 [0.64] / 1.22$\pm$0.50** | **1.31 [0.63] / 1.27$\pm$0.61** | **11.8 [12.9] (Stage-I)** | **68 ms** |
| **EgoAnchorNet (Ours, Camera)** | **Ours** | **0.12 [0.07] / 0.12$\pm$0.06** | **0.08 [0.07] / 0.09$\pm$0.05** | **10.2 [3.30] (Stage-II)** | **68 ms** |

---

## Repository Structure

- `egomotion-registration/TVCG_Viewer_Decoupled_Registration.pdf`: Compiled 11-page IEEE TVCG manuscript PDF.
- `egomotion-registration/paper/main.tex` & `refs.bib`: LaTeX source and verified bibliography.
- `egomotion-registration/figures/`: Publication-grade vector PDFs and PNG previews (`intro_relative_motion.pdf`, `fig2_two_stage_architecture.pdf`, `fig3_add_and_real_bins.pdf`, `fig4_ego_anchor_net.pdf`, `fig5_sota_head_to_head.pdf`).
- `egomotion-registration/validation/`: JSON benchmark outputs (`unified_sota_benchmark.json`, `ego_anchor_net_results.json`, `extended_sota_ablation.json`, `ego_removed_ablation.json`, `roi_pose_effects.json`, `relative_motion_effects.json`, `consecutive_pairs_by_omega.json`).
- `egomotion-registration/scripts/`: Evaluation and figure reproduction scripts.
