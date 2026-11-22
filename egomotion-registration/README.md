# Viewer-Decoupled 3D Registration — Evaluation Suite

Benchmark JSON summaries and reproducible evaluation scripts for viewer-decoupled 3D model registration under simultaneous camera and target motion.

## Directory Layout

- `validation/`:
  - `unified_sota_benchmark.json`: 40-pair dual-motion head-to-head benchmark across four angular-velocity bins.
  - `ego_anchor_net_results.json`: 260-pair component ablation across $[0.0,0.5)$, $[0.5,1.0)$, $[1.0,2.0)$, and $[2.0,10.0)$ rad/s.
  - `review_stage1_baselines.json` & `review_paired_stats.json`: IMU-aided/region-restricted baselines, gyroscope-error sensitivity, and paired bootstrap/Wilcoxon statistics.
  - `real_stage1_crossday.json`: Cross-day held-out real camera--IMU evaluation (600 pairs, $1280\times720$, independent PnP ground truth) plus dual-branch and retention checks.
  - `stereo_and_rolling_shutter_results.json` & `stereo_holdout_triangulation.json`: Calibrated stereo triangulation ($b = 46.30$ mm, 4-fold held-out shape RMSE $0.088$ mm) and modeled 4K rolling-shutter readout tests.
  - `real_viewer_overlay.json`, `multiseq_real_endoscope.json`, `review_real_checks.json`, `review_partB_real_gyro.json`: Recorded stereo-endoscope lag sweeps and image-space reference audits.
- `scripts/`: Reproducible evaluation and statistical check scripts (`eval_*.py`, `review_*.py`, `egomotion_overlay_error.py`).
- `outputs/`: Tabulated CSV/JSON sweep outputs.
