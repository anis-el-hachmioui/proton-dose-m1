"""Lateral refinement at the extended-model lateral interface.

Tests whether the narrow dose deficit next to the water-bone interface in the
extended-model benchmark (Section 5.5.3) is a discretization error or an
error of the M1 closure. A discretization error decreases when the grid is
refined. A closure error does not, because the refined M1 solution converges
to the M1 solution and not to the Boltzmann-Fokker-Planck solution.

Set-up: the extended-model lateral-interface benchmark
(thesis_experiments/scattering_nuclear/transversal_slabs.py), WENO-Z, with
the domain shortened to a 12 mm x 12 mm x 64 mm section (the deficit is
already fully developed at 40 to 60 mm depth). Only x, the direction normal
to the interface, is refined: dx = 0.24, 0.12, 0.06 mm (Nx = 50, 100, 200),
with dy = 0.24 mm and dz = 0.5 mm as in the benchmark. The energy march
stops at 50 MeV, below the energy of every proton inside this section. The
BFP Monte Carlo reference (3e7 histories, ample for profiles averaged over
|y| < 0.5 mm and a depth band) is computed once on the finest grid
and restricted to the coarser grids by averaging in x, so all three levels
are compared with the same reference.

Output: thesis_results/interface_refinement/
            level_<Nx>.npz, interface_refinement.json,
            interface_refinement.{pdf,png}

Usage: uv run python -m thesis_experiments.robustness.interface_lateral_refinement
"""

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from proton_solver import Box, Material, Reference, Solver, SpatialScheme  # noqa: E402

from thesis_experiments.scattering_nuclear import (  # noqa: E402
    transversal_slabs as benchmark,
)

WIDTH_M = 0.012
DEPTH_M = 0.064
NY = 50
DZ_M = 0.0005
LEVELS_NX = (50, 100, 200)
ENERGY_MIN_MEV = 50.0
HISTORIES = 30_000_000
Y_HALF_WIDTH_MM = 0.5
NEAR_MM = 1.0
FAR_MM = (3.0, 5.0)
BANDS_MM = ((10, 20), (20, 30), (30, 40), (40, 50), (50, 62))
FIGURE_BANDS_MM = ((40, 45), (55, 62))
RESULT_DIR = Path("thesis_results/interface_refinement")


def level_config(nx: int, with_reference: bool):
    base = benchmark.build_config(SpatialScheme.WENO_Z)
    half = 0.5 * WIDTH_M
    geometry = dataclasses.replace(
        base.geometry,
        size=(WIDTH_M, WIDTH_M, DEPTH_M),
        resolution=(nx, NY, int(round(DEPTH_M / DZ_M))),
        objects=[
            Box(
                x_min=-half,
                x_max=0.0,
                y_min=-half,
                y_max=half,
                z_min=0.0,
                z_max=DEPTH_M,
                material=Material.BONE_COMPACT,
            )
        ],
        view_geometry=False,
    )
    return dataclasses.replace(
        base,
        geometry=geometry,
        numerics=dataclasses.replace(
            base.numerics,
            energy_min=ENERGY_MIN_MEV,
        ),
        references=dataclasses.replace(
            base.references,
            reference=Reference.BFP_MC if with_reference else Reference.NONE,
            bfp_mc_histories=HISTORIES,
        ),
        output=dataclasses.replace(base.output, save_results=False),
    )


def solve_level(nx: int) -> Path:
    path = RESULT_DIR / f"level_{nx}.npz"
    if path.is_file():
        print(f"Nx={nx}: found {path}, skipping", flush=True)
        return path
    with_reference = nx == max(LEVELS_NX)
    config = level_config(nx, with_reference)
    print("=" * 60)
    print(
        f"Nx={nx}, dx={1e3 * WIDTH_M / nx:.3g} mm, "
        f"Monte Carlo: {'yes' if with_reference else 'no'}",
        flush=True,
    )
    start = time.time()
    result = Solver(config).solve()
    data = dict(
        x=np.asarray(result.grid.x),
        y=np.asarray(result.grid.y),
        z=np.asarray(result.grid.z),
        dose=np.asarray(result.dose),
        runtime=result.runtime,
    )
    if result.bfp_mc_dose is not None:
        data["mc"] = np.asarray(result.bfp_mc_dose)
        data["mc_runtime"] = result.bfp_mc_runtime
    np.savez(path, **data)
    print(f"Nx={nx} done in {time.time() - start:.0f} s", flush=True)
    return path


def restrict_x(field, nx: int):
    ratio = field.shape[0] // nx
    return field.reshape(nx, ratio, *field.shape[1:]).mean(axis=1)


def band_profile(field, y_mm, z_mm, z_lo, z_hi):
    ym = np.abs(y_mm) < Y_HALF_WIDTH_MM
    zm = (z_mm >= z_lo) & (z_mm <= z_hi)
    return field[:, ym][:, :, zm].mean(axis=(1, 2))


def rms(field, mask_x, y_mm, z_mm, z_lo, z_hi):
    ym = np.abs(y_mm) < Y_HALF_WIDTH_MM
    zm = (z_mm >= z_lo) & (z_mm <= z_hi)
    values = field[mask_x][:, ym][:, :, zm]
    return float(np.sqrt(np.mean(values**2)))


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {nx: solve_level(nx) for nx in sorted(LEVELS_NX, reverse=True)}

    with np.load(paths[max(LEVELS_NX)]) as finest:
        mc_fine = finest["mc"]
    scale = float(mc_fine.max())

    record = {
        "levels_nx": list(LEVELS_NX),
        "dx_mm": [1e3 * WIDTH_M / nx for nx in LEVELS_NX],
        "histories": HISTORIES,
        "reference_max_dose": scale,
        "bands_mm": [list(b) for b in BANDS_MM],
        "levels": {},
    }

    profiles = {}
    for nx in LEVELS_NX:
        with np.load(paths[nx]) as data:
            x_mm = 1e3 * data["x"]
            y_mm = 1e3 * data["y"]
            z_mm = 1e3 * data["z"]
            dose = data["dose"]
        mc = restrict_x(mc_fine, nx)
        diff = 100.0 * (dose - mc) / scale
        near = np.abs(x_mm) <= NEAR_MM
        far = (np.abs(x_mm) >= FAR_MM[0]) & (np.abs(x_mm) <= FAR_MM[1])
        level = {"rms_near_percent": [], "rms_far_percent": [],
                 "deficit_min_percent": [], "deficit_half_width_region_mm": []}
        for z_lo, z_hi in BANDS_MM:
            level["rms_near_percent"].append(rms(diff, near, y_mm, z_mm, z_lo, z_hi))
            level["rms_far_percent"].append(rms(diff, far, y_mm, z_mm, z_lo, z_hi))
            profile = band_profile(diff, y_mm, z_mm, z_lo, z_hi)
            minimum = float(profile.min())
            inside = x_mm[profile < 0.5 * minimum]
            level["deficit_min_percent"].append(minimum)
            level["deficit_half_width_region_mm"].append(
                [float(inside.min() - 0.5e3 * WIDTH_M / nx),
                 float(inside.max() + 0.5e3 * WIDTH_M / nx)]
            )
        record["levels"][str(nx)] = level
        profiles[nx] = (
            x_mm,
            [band_profile(diff, y_mm, z_mm, lo, hi) for lo, hi in FIGURE_BANDS_MM],
        )

    (RESULT_DIR / "interface_refinement.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
    styles = {50: "s:", 100: "o--", 200: "-"}
    for nx in LEVELS_NX:
        x_mm, band_values = profiles[nx]
        mask = np.abs(x_mm) <= 2.0
        for ax, values in zip(axes, band_values):
            ax.plot(
                x_mm[mask], values[mask], styles.get(nx, "-"), markersize=3,
                label=rf"$\Delta x={1e3 * WIDTH_M / nx:.2f}$ mm",
            )
    for ax, (lo, hi) in zip(axes, FIGURE_BANDS_MM):
        ax.axvline(0.0, color="0.4", linewidth=0.8)
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.set_title(rf"${lo}\leq z\leq {hi}$ mm")
        ax.set_xlabel(r"$x$ [mm]  (bone $x<0$, water $x>0$)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"$(D^{\mathrm{det}}-D^{\mathrm{MC}})/\max D^{\mathrm{MC}}$ [%]")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "interface_refinement.pdf")
    fig.savefig(RESULT_DIR / "interface_refinement.png", dpi=300)
    plt.close(fig)

    print()
    print("RMS of (D_det - D_MC)/max D_MC [%] per depth band, |y| < 0.5 mm")
    header = "  ".join(f"{lo}-{hi} mm" for lo, hi in BANDS_MM)
    print(f"  {'':22s}{header}")
    for nx in LEVELS_NX:
        level = record["levels"][str(nx)]
        dx = 1e3 * WIDTH_M / nx
        near = "  ".join(f"{v:8.2f}" for v in level["rms_near_percent"])
        far = "  ".join(f"{v:8.2f}" for v in level["rms_far_percent"])
        print(f"  dx={dx:.2f} |x|<=1 mm    {near}")
        print(f"  dx={dx:.2f} 3<=|x|<=5 mm {far}")
    print()
    print("Deepest deficit [%] and region where it exceeds half its depth [mm]")
    for nx in LEVELS_NX:
        level = record["levels"][str(nx)]
        dx = 1e3 * WIDTH_M / nx
        parts = [
            f"{m:6.1f} ({a:+.2f}..{b:+.2f})"
            for m, (a, b) in zip(level["deficit_min_percent"],
                                 level["deficit_half_width_region_mm"])
        ]
        print(f"  dx={dx:.2f}: " + "  ".join(parts))
    print(f"\nSaved {RESULT_DIR}")


if __name__ == "__main__":
    main()
