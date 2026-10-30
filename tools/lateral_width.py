"""Lateral beam width sigma(z) in the extended-model homogeneous benchmark.

Angular scattering shows up mainly in the lateral spread of the beam, so the
dose-weighted lateral RMS width per depth is the most direct observable for
the M1 closure. For every depth z:

    sigma(z)^2 = 1/2 [ sum x^2 D(x,y,z) / sum D + sum y^2 D(x,y,z) / sum D ],

evaluated on the cell centres of the full lateral domain, for the BFP Monte
Carlo reference and for first-order, MUSCL and WENO-Z reconstruction. If the
coarse-grid robustness run exists (thesis_experiments/robustness/coarse_grid),
WENO-Z on the twice coarser grid is added, compared with its own Monte Carlo
reference. The width changes with resolution if numerical lateral diffusion
contributes, and stays the same if the remaining difference is a closure
effect.

Depths are restricted to z <= 0.9 R80 of the Monte Carlo integrated depth
dose, beyond which the dose and hence the width estimate become small and
noisy.

Usage (from the repository root):
    uv run tools/lateral_width.py
Output: thesis_results/lateral_width/lateral_width.{pdf,png,json}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CASE = REPO_ROOT / "thesis_results" / "scattering_nuclear" / "homogeneous_water"
COARSE = (
    REPO_ROOT
    / "thesis_results"
    / "robustness"
    / "coarse_grid"
    / "scattering_nuclear"
    / "homogeneous_water"
    / "weno_z"
    / "results.npz"
)
OUT_DIR = REPO_ROOT / "thesis_results" / "lateral_width"
DEPTH_FRACTION = 0.9

SCHEMES = (
    ("first_order", "First order"),
    ("muscl", "MUSCL"),
    ("weno_z", "WENO-Z"),
)


def lateral_sigma(dose, x, y):
    """Dose-weighted lateral RMS width per depth (mean of x and y)."""

    profile_x = dose.sum(axis=1)
    profile_y = dose.sum(axis=0)
    total = profile_x.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        var_x = (x[:, None] ** 2 * profile_x).sum(axis=0) / total
        var_y = (y[:, None] ** 2 * profile_y).sum(axis=0) / total
    return np.sqrt(0.5 * (var_x + var_y))


def distal_r80(z, dose):
    idd = dose.sum(axis=(0, 1))
    i_peak = int(np.argmax(idd))
    target = 0.8 * float(idd[i_peak])
    for i in range(i_peak, len(idd) - 1):
        if idd[i] >= target > idd[i + 1]:
            return float(
                z[i]
                + (target - idd[i]) * (z[i + 1] - z[i]) / (idd[i + 1] - idd[i])
            )
    return float(z[-1])


def load(path):
    with np.load(path) as data:
        return {
            key: np.asarray(data[key], dtype=float)
            for key in ("x", "y", "z", "dose", "bfp_mc_dose")
        }


def summarize(z, sigma, sigma_ref, mask):
    ratio = sigma[mask] / sigma_ref[mask] - 1.0
    z_m = z[mask]
    k_end = int(np.argmax(z_m))
    k_mid = int(np.argmin(np.abs(z_m - 0.5 * z_m.max())))
    return {
        "max_abs_rel_difference_percent": 100.0 * float(np.max(np.abs(ratio))),
        "rel_difference_at_half_depth_percent": 100.0 * float(ratio[k_mid]),
        "rel_difference_at_end_percent": 100.0 * float(ratio[k_end]),
        "sigma_at_end_mm": 1000.0 * float(sigma[mask][k_end]),
    }


def main():
    runs = {}
    for key, _ in SCHEMES:
        path = CASE / key / "results.npz"
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Run the benchmark first.")
        runs[key] = load(path)

    base = runs["weno_z"]
    x, y, z = base["x"], base["y"], base["z"]
    sigma_mc = lateral_sigma(base["bfp_mc_dose"], x, y)
    r80 = distal_r80(z, base["bfp_mc_dose"])
    mask = z <= DEPTH_FRACTION * r80

    record = {
        "case": "scattering_nuclear/homogeneous_water",
        "depth_limit_mm": 1000.0 * DEPTH_FRACTION * r80,
        "mc_sigma_at_end_mm": 1000.0 * float(sigma_mc[mask][-1]),
        "schemes": {},
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    z_mm = 1000.0 * z
    axes[0].plot(
        z_mm[mask], 1000.0 * sigma_mc[mask], "k-", linewidth=2.0,
        label="BFP Monte Carlo",
    )

    for key, label in SCHEMES:
        sigma = lateral_sigma(runs[key]["dose"], x, y)
        record["schemes"][key] = summarize(z, sigma, sigma_mc, mask)
        axes[0].plot(z_mm[mask], 1000.0 * sigma[mask], label=label)
        axes[1].plot(
            z_mm[mask],
            100.0 * (sigma[mask] / sigma_mc[mask] - 1.0),
            label=label,
        )

    if COARSE.is_file():
        coarse = load(COARSE)
        zc = coarse["z"]
        sigma_c = lateral_sigma(coarse["dose"], coarse["x"], coarse["y"])
        sigma_c_mc = lateral_sigma(coarse["bfp_mc_dose"], coarse["x"], coarse["y"])
        mask_c = zc <= DEPTH_FRACTION * distal_r80(zc, coarse["bfp_mc_dose"])
        record["schemes"]["weno_z_coarse"] = summarize(
            zc, sigma_c, sigma_c_mc, mask_c
        )
        axes[1].plot(
            1000.0 * zc[mask_c],
            100.0 * (sigma_c[mask_c] / sigma_c_mc[mask_c] - 1.0),
            "--",
            color="C2",
            label="WENO-Z, coarse grid",
        )

    axes[0].set_xlabel(r"Depth $z$ [mm]")
    axes[0].set_ylabel(r"Lateral RMS width $\sigma$ [mm]")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].axhline(0.0, color="k", linewidth=0.8)
    axes[1].set_xlabel(r"Depth $z$ [mm]")
    axes[1].set_ylabel(r"$\sigma/\sigma_{\mathrm{MC}}-1$ [%]")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "lateral_width.pdf")
    fig.savefig(OUT_DIR / "lateral_width.png", dpi=300)
    plt.close(fig)
    (OUT_DIR / "lateral_width.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    print(f"Saved {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
