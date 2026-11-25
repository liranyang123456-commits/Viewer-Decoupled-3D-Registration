# Viewer-Decoupled 3D Registration — Evaluation & Public Reproduction Suite

Benchmark JSON summaries, standalone controlled and real stereo-endoscope trajectory benchmarks, and statistical/hardware audit scripts for viewer-decoupled 3D model registration under simultaneous camera and target motion.

## Single-Command Standalone Reproduction

Requires only `numpy`, `scipy`, and `opencv-python` (no raw multi-GB video required):

```bash
python scripts/reproduce_public_benchmarks.py
```

This single command:
1. **Re-runs from scratch** the 20-seed, 3,040-frame moving-target controlled display-update benchmark (`Table 6`, `scripts/eval_controlled_display_update.py`).
2. **Re-runs from scratch** the 10-session, 7,409-pair recorded stereo-endoscope independent image-space benchmark (`Table 7(C)` and `Figure 6(c)`), executing out-of-sample two-fold sub-millisecond $\mathrm{SO}(3)$ PnP-vs.-gyroscope impulse calibration (`scripts/subms_imu_calib.py`) and lever-arm-free camera-frame translation directly on `validation/real_endoscope_10sess_pnp_imu.npz`.
3. **Verifies** benchmark summaries and paired statistical tests across `Tables 1–7` in `validation/*.json`.

## Directory Layout

- `validation/`:
  - `real_endoscope_10sess_pnp_imu.npz`: Compact 10-session recorded stereo-endoscope PnP poses + 200 Hz IMU gyroscope trajectories enabling standalone reproduction of `Table 7(C)` ($n=7{,}409$ pairs).
  - `unified_sota_benchmark.json`: 40-pair dual-motion head-to-head benchmark and counterfactual ablation across four angular-velocity bins (`Table 1`).
  - `ego_anchor_net_results.json`: 260-pair component ablation across $[0.0,0.5)$, $[0.5,1.0)$, $[1.0,2.0)$, and $[2.0,10.0)$ rad/s (`Table 2`).
  - `review_stage1_baselines.json`, `review_imu_noise_fair.json`, & `review_paired_stats.json`: IMU-aided/region-restricted baselines, identical perturbed-IMU comparison on both EgoAnchorNet and ECD-GN+IMU, held-out PnP noise floor, and paired bootstrap/Wilcoxon statistics (`Table 3`).
  - `nonplanar_3d_model_benchmark.json`: Non-planar 3D model registration and 5-frame 30 Hz chained 6DoF tracking benchmark ($N=120$ unseen clips, $480$ steps across $\mathcal{B}_1\text{--}\mathcal{B}_4$, unseen 42 mm BOP/surgical meshes in C3VD cavities, `Table 4`).
  - `real_stage1_crossday.json` & `existing_real_recalibrated.json`: Cross-day real camera–IMU evaluation (600 slow Day-2 pairs + 360 strictly disjoint recalibrated fast-turn/dual-motion pairs up to 1.39 rad/s, `Table 5`) and 10-session independent image-space stereo-endoscope display-propagation audit ($n=7{,}409$ pairs, `Table 7(C)`).
  - `controlled_display_update.json`: Fair 20-seed controlled moving-target display-update benchmark ($N=3{,}040$ frames, analytic $t_d$ ground truth, identical noisy inputs + $200$ Hz IMU, `Table 6`).
  - `stereo_and_rolling_shutter_results.json` & `stereo_holdout_triangulation.json`: Calibrated stereo triangulation ($b = 46.30$ mm, 4-fold held-out grid shape self-consistency RMSE $0.088$ mm) and modeled 4K rolling-shutter readout tests (`Figure 6(a)–(b)`).
  - `real_viewer_overlay.json`, `review_real_checks.json`, & `review_partB_real_gyro.json`: Recorded stereo-endoscope lag sweeps (`Table 7(A)–(B)`) and legacy sync diagnostic outputs.
- `scripts/`:
  - `reproduce_public_benchmarks.py`: Single-command standalone reproduction & verification suite.
  - `subms_imu_calib.py`: Standalone sub-millisecond $\mathrm{SO}(3)$ camera–IMU impulse calibration module (`numpy` + `scipy`).
  - `eval_controlled_display_update.py`: Standalone 20-seed controlled display-update benchmark.
  - `eval_nonplanar_3d_model_benchmark.py`, `eval_existing_real_recalibrated.py`, `eval_*.py`, `review_*.py`: Full evaluation and statistical audit scripts (raw image recordings and trained EgoAnchorNet module/weights to be released upon acceptance).
