# Viewer-Decoupled 3D Registration — Benchmark Data & Evaluation Suite

This repository hosts the angular-velocity-stratified benchmark JSON summaries, standalone controlled display-update benchmark, and statistical audit scripts for viewer-decoupled 3D model registration under simultaneous camera and target motion.

- **Companion Simulator & Hardware Capture Tools:** [https://github.com/liranyang123456-commits/EGO_Mo](https://github.com/liranyang123456-commits/EGO_Mo)

## Contents

- `egomotion-registration/validation/`: Per-bin JSON evaluation outputs across angular-velocity tiers ($[0.0,0.5)$, $[0.5,1.0)$, $[1.0,2.0)$, $[2.0,10.0)$ rad/s; observed rendered fast-bin speeds span $2.20\text{--}3.40$ rad/s on test and $2.00\text{--}4.47$ rad/s on validation):
  - `unified_sota_benchmark.json`: 40-pair dual-motion benchmark and three-condition counterfactual ablation across SIFT, ORB, KLT, RAFT, LoFTR, Pi3, Reloc3r, VGGT-1B, ECD-GN, and EgoAnchorNet.
  - `ego_anchor_net_results.json`: 260-pair component ablation across angular-velocity bins.
  - `review_stage1_baselines.json`, `review_imu_noise_fair.json`, & `review_paired_stats.json`: IMU-aided/oracle-region baselines, identical perturbed-IMU comparisons (EgoAnchorNet vs. ECD-GN+IMU), held-out PnP noise floor, and paired bootstrap/Wilcoxon statistics.
  - `real_stage1_crossday.json`: Cross-day session-disjoint real-video motion-decoupling evaluation (600 held-out $1280\times720$ Day-2 pairs against independent PnP ground truth, including zero-shot, real-adapted, and `w/o IMU` variants).
  - `controlled_display_update.json`: Fair 20-seed moving-target controlled display-update benchmark ($N=3{,}040$ frames, Device-B $1280\times720$ geometry, accelerating 6DoF camera up to $3.14$ rad/s, analytic display-time ground truth, identical noisy observations and $200$ Hz IMU).
  - `stereo_and_rolling_shutter_results.json` & `stereo_holdout_triangulation.json`: Calibrated stereo extrinsic verification ($b = 46.30$ mm, 4-fold held-out grid shape self-consistency RMSE $0.088$ mm) and modeled 4K rolling-shutter readout evaluation.
  - `real_viewer_overlay.json`, `review_real_checks.json`, & `review_partB_real_gyro.json`: Recorded stereo-endoscope lag sweeps (both pure-gyro and gyro+causal-translation variants) and independent image-space reference audits (including the negative untriggered-USB result).
- `egomotion-registration/scripts/`:
  - **Immediately runnable standalone (`numpy` + `scipy` only):** `eval_controlled_display_update.py` reproduces `validation/controlled_display_update.json` directly with zero external dataset dependencies.
  - **Evaluation & audit scripts requiring model weights/raw recordings (released upon acceptance):** `eval_real_stage1_crossday.py`, `eval_real_viewer_overlay.py`, `eval_stereo_and_rolling_shutter.py`, `eval_stereo_holdout_triangulation.py`, and `review_*.py`.
