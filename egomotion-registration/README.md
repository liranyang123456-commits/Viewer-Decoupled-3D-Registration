# Viewer-Decoupled 3D Registration — Evaluation Suite

Benchmark JSON summaries, standalone controlled display-update benchmark, and statistical audit scripts for viewer-decoupled 3D model registration under simultaneous camera and target motion.

## Quick Standalone Reproduction

Reproduce the 20-seed moving-target controlled display-update benchmark (`Table 5`, `validation/controlled_display_update.json`) using only `numpy` and `scipy`:

```bash
python scripts/eval_controlled_display_update.py
```

## Directory Layout

- `validation/`:
  - `unified_sota_benchmark.json`: 40-pair dual-motion head-to-head benchmark and counterfactual ablation across four angular-velocity bins.
  - `ego_anchor_net_results.json`: 260-pair component ablation across $[0.0,0.5)$, $[0.5,1.0)$, $[1.0,2.0)$, and $[2.0,10.0)$ rad/s.
  - `review_stage1_baselines.json`, `review_imu_noise_fair.json`, & `review_paired_stats.json`: IMU-aided/region-restricted baselines, identical perturbed-IMU comparison on both EgoAnchorNet and ECD-GN+IMU, held-out PnP noise floor, and paired bootstrap/Wilcoxon statistics.
  - `real_stage1_crossday.json`: Cross-day held-out real camera--IMU evaluation (600 Day-2 pairs, $1280\times720$, independent PnP ground truth) including `w/o IMU`, dual-branch, and SCARED retention checks.
  - `controlled_display_update.json`: Fair 20-seed controlled moving-target display-update benchmark ($N=3{,}040$ frames, analytic $t_d$ ground truth, identical noisy inputs + $200$ Hz IMU).
  - `stereo_and_rolling_shutter_results.json` & `stereo_holdout_triangulation.json`: Calibrated stereo triangulation ($b = 46.30$ mm, 4-fold held-out grid shape self-consistency RMSE $0.088$ mm) and modeled 4K rolling-shutter readout tests.
  - `real_viewer_overlay.json`, `review_real_checks.json`, & `review_partB_real_gyro.json`: Recorded stereo-endoscope lag sweeps (pure-gyro and gyro+causal-translation) and independent image-space reference audits.
- `scripts/`: Standalone controlled benchmark (`eval_controlled_display_update.py`) plus real-data evaluation and statistical audit scripts (`eval_*.py`, `review_*.py`; raw sessions and trained model code/weights to be released upon acceptance).
