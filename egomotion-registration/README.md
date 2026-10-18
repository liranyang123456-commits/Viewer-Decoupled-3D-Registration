# Viewer-Decoupled 3D Model Registration for Video See-Through Navigation

Official benchmark data, figure generation scripts, and IEEE TVCG manuscript sources for:

> **Separating Viewer Motion from 3D Model Registration for Video See-Through Navigation**  
> Ranyang Li\*, Nan Wei, Zhipeng Lin, Wufeng Liu, Chao Fan, and Junjun Pan\*  
>GitHub: [https://github.com/liranyang123456-commits/Viewer-Decoupled-3D-Registration](https://github.com/liranyang123456-commits/Viewer-Decoupled-3D-Registration)

## Contents

- `TVCG_Viewer_Decoupled_Registration.pdf`: Compiled 11-page IEEE TVCG paper.
- `paper/main.tex`, `paper/refs.bib`: LaTeX manuscript and verified bibliography.
- `figures/`:
  - `intro_relative_motion.pdf` (Fig. 1: 3-Condition Counterfactual Ablation & 3D AR Overlay)
  - `fig2_two_stage_architecture.pdf` (Fig. 2: Two-Stage Viewer-Decoupled Registration Architecture)
  - `fig5_sota_head_to_head.pdf` (Fig. 3: Head-to-Head SOTA vs. EgoAnchorNet Benchmark, ECDF & Latency Trade-off)
  - `fig4_ego_anchor_net.pdf` (Fig. 4: EgoAnchorNet Static-Anchor Disentanglement & Component Ablation)
  - `fig3_add_and_real_bins.pdf` (Fig. 5: 3D Model ADD Error, Latency Sweep & Real 4K+IMU Sessions)
- `validation/`: Complete per-bin JSON evaluation outputs (`unified_sota_benchmark.json`, `ego_anchor_net_results.json`, `extended_sota_ablation.json`, `ego_removed_ablation.json`, `roi_pose_effects.json`, `relative_motion_effects.json`, `consecutive_pairs_by_omega.json`).
- `scripts/`: Scripts used to reproduce the figures and latency sweeps.
