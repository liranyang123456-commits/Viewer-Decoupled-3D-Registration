"""Small-angle model: overlay error caused by egomotion during pose latency.

Rotation (independent of depth):
    pixel shift = f_px * ||omega|| * tau
    mm shift    = z_m * ||omega|| * tau * 1000

Translation (falls with depth):
    pixel shift = f_px * ||v_perp|| * tau / z_m
    mm shift    = ||v_perp|| * tau * 1000

This is the geometric part of registration error. It does not include
tracker failure from blur. Run from this folder:

    python scripts/egomotion_overlay_error.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import math

# 640 px wide, 70 deg horizontal FOV, a typical phone/headset camera crop.
WIDTH_PX = 640
HFOV_DEG = 70.0
F_PX = (WIDTH_PX / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))
DEPTHS_M = (0.3, 1.0, 2.0)
TAUS_MS = (10, 20, 40, 80)
# Head yaw rates and sideways speeds seen in fast AR looking / walking.
OMEGAS = (0.5, 1.5, 3.0)  # rad/s
SPEEDS = (0.2, 0.8, 1.5)  # m/s


def rotation_shift(omega: float, tau_s: float, z_m: float) -> tuple[float, float]:
    px = F_PX * omega * tau_s
    mm = z_m * omega * tau_s * 1000.0
    return px, mm


def translation_shift(speed: float, tau_s: float, z_m: float) -> tuple[float, float]:
    px = F_PX * speed * tau_s / z_m
    mm = speed * tau_s * 1000.0
    return px, mm


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "outputs" / "overlay_error_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    for tau_ms in TAUS_MS:
        tau_s = tau_ms / 1000.0
        for z_m in DEPTHS_M:
            for omega in OMEGAS:
                px, mm = rotation_shift(omega, tau_s, z_m)
                rows.append(
                    {
                        "source": "rotation",
                        "tau_ms": tau_ms,
                        "depth_m": z_m,
                        "omega_rad_s": omega,
                        "speed_m_s": 0.0,
                        "pixel_shift": round(px, 2),
                        "overlay_mm": round(mm, 2),
                    }
                )
            for speed in SPEEDS:
                px, mm = translation_shift(speed, tau_s, z_m)
                rows.append(
                    {
                        "source": "translation",
                        "tau_ms": tau_ms,
                        "depth_m": z_m,
                        "omega_rad_s": 0.0,
                        "speed_m_s": speed,
                        "pixel_shift": round(px, 2),
                        "overlay_mm": round(mm, 2),
                    }
                )
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"focal_px={F_PX:.1f}  rows={len(rows)}  wrote {out}")


if __name__ == "__main__":
    main()
