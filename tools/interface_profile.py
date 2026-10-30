"""Lateral dose-difference profiles across the extended-model lateral interface.

For the WENO-Z solution of the extended-model lateral-interface benchmark,
plots (D_det - D_MC) / max D_MC along x through the beam axis (central y row,
half-open cell rule) at two depths:

  - z = 60 mm, in the plateau, far upstream of both Bragg peaks,
  - z = 91 mm, at the bone-side Bragg peak,

on the benchmark grid and on the twice coarser grid of the robustness study
(each compared with its own BFP Monte Carlo reference). A discretization
error grows on the coarser grid. A deficit that is confined to the interface
and does not grow points to the M1 closure.

Usage (from the repository root):
    uv run tools/interface_profile.py
Output: thesis_results/interface_profile/interface_profile.{pdf,png,json}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "thesis_results"
CASES = (
    (
        "Benchmark grid",
        RESULTS / "scattering_nuclear/transversal_slabs/weno_z/results.npz",
    ),
    (
        "Twice coarser grid",
        RESULTS
        / "robustness/coarse_grid/scattering_nuclear/transversal_slabs"
        / "weno_z/results.npz",
    ),
)
DEPTHS_MM = (60.0, 91.0)
X_RANGE_MM = 3.0
OUT_DIR = RESULTS / "interface_profile"


def axis_index(centres):
    """Cell containing coordinate 0 (half-open rule, as in ResultMetrics)."""

    width = float(centres[1] - centres[0])
    first_face = float(centres[0]) - 0.5 * width
    index = int(np.floor((0.0 - first_face) / width + 1e-9))
    return min(max(index, 0), len(centres) - 1)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
    record = {}

    for label, path in CASES:
        if not path.is_file():
            raise SystemExit(f"Missing {path}")
        with np.load(path) as data:
            x = np.asarray(data["x"], dtype=float)
            y = np.asarray(data["y"], dtype=float)
            z = np.asarray(data["z"], dtype=float)
            det = np.asarray(data["dose"], dtype=float)
            mc = np.asarray(data["bfp_mc_dose"], dtype=float)

        iy = axis_index(y)
        diff = 100.0 * (det[:, iy, :] - mc[:, iy, :]) / mc.max()
        mask = np.abs(1000.0 * x) <= X_RANGE_MM
        record[label] = {"cell_width_mm": 1000.0 * float(x[1] - x[0])}

        for ax, depth in zip(axes, DEPTHS_MM):
            k = int(np.argmin(np.abs(1000.0 * z - depth)))
            ax.plot(
                1000.0 * x[mask],
                diff[mask, k],
                "o-" if label.startswith("Benchmark") else "s--",
                markersize=3.5,
                label=label,
            )
            record[label][f"z_{1000.0 * z[k]:.2f}_mm"] = {
                "x_mm": [round(v, 3) for v in (1000.0 * x[mask]).tolist()],
                "difference_percent_of_max": [
                    round(v, 3) for v in diff[mask, k].tolist()
                ],
            }

    for ax, depth in zip(axes, DEPTHS_MM):
        ax.axvline(0.0, color="0.4", linewidth=0.8)
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.set_title(rf"$z\approx{depth:.0f}$ mm")
        ax.set_xlabel(r"$x$ [mm]  (bone $x<0$, water $x>0$)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"$(D^{\mathrm{det}}-D^{\mathrm{MC}})/\max D^{\mathrm{MC}}$ [%]")
    axes[0].legend()
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "interface_profile.pdf")
    fig.savefig(OUT_DIR / "interface_profile.png", dpi=300)
    plt.close(fig)
    (OUT_DIR / "interface_profile.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(f"Saved {OUT_DIR}")


if __name__ == "__main__":
    main()
