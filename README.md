# Viewer-Decoupled 3D Registration — Benchmark Data & Evaluation Suite

This repository hosts the angular-velocity-stratified evaluation summaries and reproducible audit scripts for camera--IMU 3D model registration under simultaneous viewer and target motion.

- **Companion Simulator & Hardware Capture Tools:** [https://github.com/liranyang123456-commits/EGO_Mo](https://github.com/liranyang123456-commits/EGO_Mo)

## Contents

- `egomotion-registration/validation/`: Per-bin JSON evaluation outputs across angular-velocity tiers ($[0.0,0.5)$, $[0.5,1.0)$, $[1.0,2.0)$, $[2.0,10.0)$ rad/s):
  - `unified_sota_benchmark.json`: 40-pair dual-motion benchmark across SIFT, ORB, KLT, RAFT, LoFTR, Pi3, Reloc3r, VGGT-1B, ECD-GN, and EgoAnchorNet.
  - `ego_anchor_net_results.json`: 260-pair component ablation across angular-velocity bins.
  - `review_stage1_baselines.json` & `review_paired_stats.json`: IMU-aided/oracle-region baselines, sensitivity checks, and paired bootstrap/Wilcoxon statistics.
  - `real_stage1_crossday.json`: Cross-day session-disjoint real-video Stage-I evaluation (600 held-out $1280\times720$ pairs against independent PnP ground truth).
  - `stereo_and_rolling_shutter_results.json` & `stereo_holdout_triangulation.json`: Calibrated stereo extrinsic verification ($b = 46.30$ mm) and modeled 4K rolling-shutter readout evaluation.
  - `real_viewer_overlay.json`, `multiseq_real_endoscope.json`, `review_real_checks.json`, & `review_partB_real_gyro.json`: Stage-II display-update audits on recorded stereo-endoscope sessions.
- `egomotion-registration/scripts/`: Evaluation and statistical verification scripts (`eval_real_stage1_crossday.py`, `eval_real_viewer_overlay.py`, `eval_stereo_and_rolling_shutter.py`, `eval_stereo_holdout_triangulation.py`, `egomotion_overlay_error.py`, `review_*.py`).
- `egomotion-registration/outputs/`: Tabulated overlay-error sweep CSV and JSON summaries.
