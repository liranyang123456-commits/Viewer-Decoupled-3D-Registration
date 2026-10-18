# Viewer-Decoupled 3D Model Registration for Video See-Through Navigation

Official benchmark data, figure generation scripts, and IEEE TVCG manuscript sources for:

> **Separating Viewer Motion from 3D Model Registration for Video See-Through Navigation**  
> Ranyang Li\*, Nan Wei, Zhipeng Lin, Wufeng Liu, Chao Fan, and Junjun Pan\*  
> GitHub: [https://github.com/liranyang123456-commits/Viewer-Decoupled-3D-Registration](https://github.com/liranyang123456-commits/Viewer-Decoupled-3D-Registration)

## Contents

- `TVCG_Viewer_Decoupled_Registration.pdf`: Compiled 12-page IEEE TVCG regular paper.
- `paper/main.tex`, `paper/refs.bib`: LaTeX manuscript and verified bibliography.
- `figures/`:
  - `intro_relative_motion.pdf` (Fig. 1: 3-Condition Counterfactual Ablation & 3D AR Overlay)
  - `fig2_two_stage_architecture.pdf` (Fig. 2: Two-Stage Viewer-Decoupled Registration Architecture with Row-Timed Readout)
  - `fig5_sota_head_to_head.pdf` (Fig. 3: Head-to-Head SOTA vs. EgoAnchorNet Benchmark, ECDF & Latency Trade-off)
  - `fig4_ego_anchor_net.pdf` (Fig. 4: EgoAnchorNet Static-Anchor Disentanglement & Component Ablation)
  - `fig3_add_and_real_bins.pdf` (Fig. 5: Calibrated 46.30 mm Stereo 3D Triangulation, 4K Rolling-Shutter Row-Timed Warp & Real 4K+IMU Sessions)
- `validation/`: Complete per-bin JSON evaluation outputs:
  - `unified_sota_benchmark.json`: 40-pair head-to-head benchmark across SIFT, ORB, KLT, RAFT, LoFTR, Pi3, Reloc3r, VGGT-1B, Coupled EgoLite, and EgoAnchorNet.
  - `ego_anchor_net_results.json`: 260-pair EgoAnchorNet component ablation across $\mathcal{B}_1\text{--}\mathcal{B}_4$.
  - `stereo_and_rolling_shutter_results.json`: Measured stereo extrinsic calibration ($b = 46.30\text{ mm}$, sub-frame lag $\Delta k = -0.518\text{ frames}$, stereo RMS $0.290\text{ px}$, 3D shape RMSE $0.087\text{ mm}$, stereo-vs-PnP ADD $0.355\text{ mm}$) and 4K ($3840\times 2160$) rolling-shutter row-timed vs. frame-level IMU compensation across Top/Middle/Bottom bands.
  - `consecutive_pairs_by_omega.json`, `extended_sota_ablation.json`, `ego_removed_ablation.json`, `roi_pose_effects.json`, `relative_motion_effects.json`.
- `scripts/`: Scripts used to reproduce the figures, calibrated stereo triangulation, and 4K rolling-shutter evaluations.
