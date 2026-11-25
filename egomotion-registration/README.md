# Viewer-Decoupled 3D Registration — Evaluation Suite

Benchmark JSON summaries, standalone controlled and non-planar 3D model tracking benchmarks, and statistical/hardware audit scripts for viewer-decoupled 3D model registration under simultaneous camera and target motion.

## Quick Standalone Reproduction

Reproduce the 20-seed moving-target controlled display-update benchmark (`Table 6`, `validation/controlled_display_update.json`) using only `numpy` and `scipy`:

```bash
python scripts/eval_controlled_display_update.py
```

## Directory Layout

- `validation/`:
  - `unified_sota_benchmark.json`: 40-pair dual-motion head-to-head benchmark and counterfactual ablation across four angular-velocity bins.
  - `ego_anchor_net_results.json`: 260-pair component ablation across $[0.0,0.5)$, $[0.5,1.0)$, $[1.0,2.0)$, and $[2.0,10.0)$ rad/s.
  - `review_stage1_baselines.json`, `review_imu_noise_fair.json`, & `review_paired_stats.json`: IMU-aided/region-restricted baselines, identical perturbed-IMU comparison on both EgoAnchorNet and ECD-GN+IMU, held-out PnP noise floor, and paired bootstrap/Wilcoxon statistics.
  - `nonplanar_3d_model_benchmark.json`: Non-planar 3D model registration and 5-frame 30 Hz chained 6DoF tracking benchmark ($N=120$ unseen clips, $480$ steps across $\mathcal{B}_1\text{--}\mathcal{B}_4$, unseen 42 mm BOP/surgical meshes in C3VD cavities, comparing KLT+PnP, RAFT+PnP, RBOT/ICG with and without IMU, ECD-GN+IMU, unanchored relative chaining, and mesh-anchored EgoAnchorNet).
  - `real_stage1_crossday.json` & `existing_real_recalibrated.json`: Cross-day real camera–IMU evaluation (600 slow Day-2 pairs + 360 recalibrated fast-turn/dual-motion pairs up to 1.39 rad/s) and 10-session independent image-space stereo-endoscope display-propagation audit ($n=7{,}409$ pairs, out-of-sample two-fold $\mathrm{SO}(3)$ impulse calibration + lever-arm-free camera-frame translation).
  - `controlled_display_update.json`: Fair 20-seed controlled moving-target display-update benchmark ($N=3{,}040$ frames, analytic $t_d$ ground truth, identical noisy inputs + $200$ Hz IMU).
  - `stereo_and_rolling_shutter_results.json` & `stereo_holdout_triangulation.json`: Calibrated stereo triangulation ($b = 46.30$ mm, 4-fold held-out grid shape self-consistency RMSE $0.088$ mm) and modeled 4K rolling-shutter readout tests.
  - `real_viewer_overlay.json`, `review_real_checks.json`, & `review_partB_real_gyro.json`: Recorded stereo-endoscope lag sweeps (pure-gyro and gyro+causal-translation) and legacy sync diagnostic outputs.
- `scripts/`: Standalone controlled benchmark (`eval_controlled_display_update.py`), non-planar 3D model tracking benchmark (`eval_nonplanar_3d_model_benchmark.py`), recalibrated real-hardware evaluation (`eval_existing_real_recalibrated.py`), and statistical audit scripts (`eval_*.py`, `review_*.py`; raw sessions and trained model code/weights to be released upon acceptance).
