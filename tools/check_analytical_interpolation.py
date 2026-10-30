"""Check the stopping-power interpolation of the analytical CSDA solution.

The analytical solution used to interpolate S(E) linearly between the table
nodes, while the deterministic solver and the BFP Monte Carlo use PCHIP.
This script compares, on the reference-verification problem (homogeneous
water, 100x100x400, 1e8 histories):

  1. the old (linear) and new (PCHIP) analytical solutions with each other,
  2. both analytical solutions with the BFP Monte Carlo dose.

Besides the relative L1/L2/Linf errors over all cells, it compares the
laterally integrated depth dose I(z) and its distal 80% range R80, which are
almost free of statistical noise at 1e8 histories. A systematic difference
caused by the interpolation shows up there as a range offset.

The Monte Carlo dose comes from the reference cache if it exists, otherwise
it is computed once (about 3 minutes) and cached, so the reference
verification rerun reuses it.

Usage (from the repository root):
    uv run tools/check_analytical_interpolation.py
Output: thesis_results/checks/analytical_interpolation.json
"""

from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import proton_solver.analytical as analytical  # noqa: E402
from proton_solver.bfp_mc import run_bfp_mc_reference  # noqa: E402
from proton_solver.numerics.boundary import BeamInflow  # noqa: E402
from proton_solver.numerics.grid import Grid  # noqa: E402
from proton_solver.physics.materials import MaterialMap  # noqa: E402
from proton_solver.physics.stopping_power import (  # noqa: E402
    load_stopping_power_table,
)
from thesis_experiments.reference_verification.bfp_mc_vs_analytical import (  # noqa: E402
    build_config,
    relative_errors,
)

HISTORIES = 100_000_000
OUT = REPO_ROOT / "thesis_results" / "checks" / "analytical_interpolation.json"


@lru_cache(maxsize=None)
def linear_range_table(material):
    """The previous range table: S(E) interpolated linearly (np.interp)."""

    table = load_stopping_power_table(material)
    source_energy = np.asarray(table["energy_mev"], dtype=float)
    source_stopping = np.asarray(
        table["electronic_stopping_power_mev_per_m"], dtype=float
    )
    energy = np.geomspace(
        max(float(source_energy[0]), 1.0e-12),
        float(source_energy[-1]),
        analytical.RANGE_TABLE_POINTS,
    )
    stopping = np.interp(energy, source_energy, source_stopping)
    range_values = np.zeros_like(energy)
    range_values[1:] = np.cumsum(
        0.5 * (1.0 / stopping[1:] + 1.0 / stopping[:-1]) * np.diff(energy)
    )
    return energy, stopping, range_values


def analytical_dose(config, material_map, grid, inflow, table_function):
    """Analytical dose with the given range-table function."""

    original = analytical.range_table
    analytical.range_table = table_function
    try:
        return np.asarray(
            analytical.analytical_csda_dose(
                config, material_map, grid, inflow
            ),
            dtype=float,
        )
    finally:
        analytical.range_table = original


def integrated_depth_dose(dose, grid):
    return dose.sum(axis=(0, 1)) * float(grid.dx) * float(grid.dy)


def distal_range(z, idd, level=0.8):
    """Depth on the distal falloff where I(z) drops to level * max."""

    i_peak = int(np.argmax(idd))
    target = level * float(idd[i_peak])
    for i in range(i_peak, len(idd) - 1):
        if idd[i] >= target > idd[i + 1]:
            return float(
                z[i] + (target - idd[i]) * (z[i + 1] - z[i])
                / (idd[i + 1] - idd[i])
            )
    return None


def summary(name, dose, reference, z, grid):
    idd = integrated_depth_dose(dose, grid)
    idd_ref = integrated_depth_dose(reference, grid)
    r80 = distal_range(z, idd)
    r80_ref = distal_range(z, idd_ref)
    return {
        "comparison": name,
        **relative_errors(dose, reference),
        "idd_max_abs_difference_rel": float(
            np.max(np.abs(idd - idd_ref)) / np.max(idd_ref)
        ),
        "idd_peak_rel_difference": float(idd.max() / idd_ref.max() - 1.0),
        "r80_mm": None if r80 is None else 1000.0 * r80,
        "r80_reference_mm": None if r80_ref is None else 1000.0 * r80_ref,
        "r80_difference_um": (
            None
            if r80 is None or r80_ref is None
            else 1.0e6 * (r80 - r80_ref)
        ),
    }


def main():
    config = build_config(HISTORIES)
    grid = Grid.from_geometry(config.geometry, np)
    material_map = MaterialMap.from_geometry(config.geometry, grid, np)
    inflow = BeamInflow.from_beam(config.beam, grid, np)
    z = np.asarray(grid.z, dtype=float)

    start = time.perf_counter()
    dose_pchip = analytical_dose(
        config, material_map, grid, inflow, analytical.range_table
    )
    dose_linear = analytical_dose(
        config, material_map, grid, inflow, linear_range_table
    )
    print(f"Analytical solutions: {time.perf_counter() - start:.1f} s")

    print(f"BFP Monte Carlo, {HISTORIES:,} histories (cached if available)")
    dose_mc, runtime = run_bfp_mc_reference(
        config, material_map, grid, inflow
    )
    dose_mc = np.asarray(dose_mc, dtype=float)

    records = [
        summary("linear vs PCHIP analytical", dose_linear, dose_pchip, z, grid),
        summary("MC vs linear analytical (old)", dose_mc, dose_linear, z, grid),
        summary("MC vs PCHIP analytical (new)", dose_mc, dose_pchip, z, grid),
    ]

    print()
    header = (
        f"{'comparison':32s} {'L1':>9s} {'L2':>9s} {'Linf':>9s} "
        f"{'IDD max':>9s} {'IDD peak':>9s} {'dR80 um':>9s}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        print(
            f"{r['comparison']:32s} {r['l1']:9.2e} {r['l2']:9.2e} "
            f"{r['linf']:9.2e} {r['idd_max_abs_difference_rel']:9.2e} "
            f"{r['idd_peak_rel_difference']:9.2e} "
            f"{(r['r80_difference_um'] or float('nan')):9.2f}"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "histories": HISTORIES,
                "grid_resolution": list(config.geometry.resolution),
                "mc_runtime_seconds": float(runtime),
                "results": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
