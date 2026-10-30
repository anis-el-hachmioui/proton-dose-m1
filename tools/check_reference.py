"""Independent checks of the BFP Monte Carlo reference.

Runs four checks and writes thesis_results/checks/reference_checks.json plus
one figure (thesis_results/checks/reference_checks.pdf):

1. Step limits, reduced model (exact solution available).
   The reference-verification problem (homogeneous water, 100x100x400,
   1e8 histories) is run with all BFP step limits halved and compared with
   the analytical CSDA solution, next to the run with the standard limits
   (taken from the cache). If the small systematic difference in the
   integrated depth dose is a step-length effect, it decreases.

2. Step limits, extended model.
   Extended homogeneous water at 1e7 histories with halved step limits,
   compared with the 1e8 reference of the benchmark run. Differences in the
   integrated depth dose, R80 and the lateral width must stay at the level
   of the 1e7 statistics.

3. Angular scattering: lateral beam width against Fermi-Eyges theory with
   the same transport coefficient. For small angles the projected
   mean-square angle grows at the rate Sigma_tr = 1/lambda_1, so
       sigma_x(z)^2 = sigma_0^2 + dx^2/12 + int_0^z (z - s)^2 Sigma_tr(E(s)) ds,
   with E(s) from the CSDA range of the mean beam energy. The finite
   lateral domain (|x| <= 12 mm) is accounted for by the variance of the
   truncated Gaussian. Compared up to 90% of the CSDA range.

4. Primary-proton removal and lateral escape: the Monte Carlo integrated
   depth dose against the analytical CSDA depth dose times the survival
   probability exp(-int Sigma_r ds) and the fraction of the Fermi-Eyges
   beam inside the domain. Compared in the plateau, 10 mm <= z <= 0.8 R0.
   The remaining difference contains the small detour (path length)
   effect of scattering.

Requires the extended homogeneous benchmark (results.npz with bfp_mc_dose
at 1e8 histories) to exist.

Usage (from the repository root):
    uv run tools/check_reference.py
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from proton_solver import SpatialScheme  # noqa: E402
from proton_solver.analytical import analytical_csda_dose  # noqa: E402
from proton_solver.bfp_mc import run_bfp_mc_reference  # noqa: E402
from proton_solver.bfp_mc.materials import _build_lookups  # noqa: E402
from proton_solver.numerics.boundary import BeamInflow  # noqa: E402
from proton_solver.numerics.grid import Grid  # noqa: E402
from proton_solver.physics.materials import MaterialMap  # noqa: E402
from thesis_experiments.reference_verification import (  # noqa: E402
    bfp_mc_vs_analytical as verification,
)
from thesis_experiments.scattering_nuclear import (  # noqa: E402
    homogeneous_water as extended_homogeneous,
)

OUT_DIR = REPO_ROOT / "thesis_results" / "checks"
EXTENDED_REFERENCE = (
    REPO_ROOT
    / "thesis_results"
    / "scattering_nuclear"
    / "homogeneous_water"
    / "weno_z"
    / "results.npz"
)
STEP_CHECK_HISTORIES_EXTENDED = 10_000_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup(config):
    grid = Grid.from_geometry(config.geometry, np)
    material_map = MaterialMap.from_geometry(config.geometry, grid, np)
    inflow = BeamInflow.from_beam(config.beam, grid, np)
    return grid, material_map, inflow


def halved_steps(config, histories=None):
    refs = config.references
    changes = dict(
        bfp_mc_max_step_mm=0.5 * refs.bfp_mc_max_step_mm,
        bfp_mc_max_energy_loss_mev=0.5 * refs.bfp_mc_max_energy_loss_mev,
        bfp_mc_max_scattering_optical_depth=(
            0.5 * refs.bfp_mc_max_scattering_optical_depth
        ),
    )
    if histories is not None:
        changes["bfp_mc_histories"] = int(histories)
    return dataclasses.replace(
        config, references=dataclasses.replace(refs, **changes)
    )


def with_histories(config, histories):
    return dataclasses.replace(
        config,
        references=dataclasses.replace(
            config.references, bfp_mc_histories=int(histories)
        ),
    )


def idd(dose, grid):
    return dose.sum(axis=(0, 1)) * float(grid.dx) * float(grid.dy)


def distal_level(z, curve, level):
    """Depth on the distal falloff where curve drops to level * max."""

    i_peak = int(np.argmax(curve))
    target = level * float(curve[i_peak])
    for i in range(i_peak, len(curve) - 1):
        if curve[i] >= target > curve[i + 1]:
            return float(
                z[i]
                + (target - curve[i])
                * (z[i + 1] - z[i])
                / (curve[i + 1] - curve[i])
            )
    return float("nan")


def lateral_sigma(dose, grid):
    """Dose-weighted RMS lateral position per depth, mean of x and y."""

    x = np.asarray(grid.x, dtype=float)
    y = np.asarray(grid.y, dtype=float)
    dx_profile = dose.sum(axis=1)  # (nx, nz)
    dy_profile = dose.sum(axis=0)  # (ny, nz)
    total = dx_profile.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        var_x = (x[:, None] ** 2 * dx_profile).sum(axis=0) / total
        var_y = (y[:, None] ** 2 * dy_profile).sum(axis=0) / total
    return np.sqrt(0.5 * (var_x + var_y))


def idd_comparison(dose, reference, grid):
    z = np.asarray(grid.z, dtype=float)
    a = idd(dose, grid)
    b = idd(reference, grid)
    r80 = distal_level(z, a, 0.8)
    r80_ref = distal_level(z, b, 0.8)
    dfw = r80 - distal_level(z, a, 0.2)
    dfw_ref = r80_ref - distal_level(z, b, 0.2)
    return {
        "idd_max_abs_difference_rel": float(
            np.max(np.abs(a - b)) / np.max(b)
        ),
        "idd_peak_rel_difference": float(a.max() / b.max() - 1.0),
        "r80_difference_um": 1.0e6 * (r80 - r80_ref),
        "dfw_difference_um": 1.0e6 * (dfw - dfw_ref),
    }


def truncated_rms(sigma, half_width):
    """RMS of a zero-mean Gaussian restricted to |x| <= half_width."""

    a = half_width / sigma
    mass = math.erf(a / math.sqrt(2.0))
    pdf = math.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)
    return sigma * math.sqrt(max(1.0 - 2.0 * a * pdf / mass, 0.0))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_steps_reduced():
    print("\n[1] Step limits, reduced model, 1e8 histories")
    base = verification.build_config(100_000_000)
    grid, material_map, inflow = setup(base)
    exact = np.asarray(
        analytical_csda_dose(base, material_map, grid, inflow), dtype=float
    )
    record = {}
    for label, config in (
        ("standard_steps", base),
        ("halved_steps", halved_steps(base)),
    ):
        dose, runtime = run_bfp_mc_reference(
            config, material_map, grid, inflow
        )
        dose = np.asarray(dose, dtype=float)
        record[label] = {
            "max_step_mm": config.references.bfp_mc_max_step_mm,
            "runtime_seconds": float(runtime),
            **verification.relative_errors(dose, exact),
            **idd_comparison(dose, exact, grid),
        }
        print(f"  {label}: {json.dumps(record[label])}")
    return record


def check_steps_extended(reference_mc):
    print("\n[2] Step limits, extended model, 1e7 histories vs 1e8")
    base = extended_homogeneous.build_config(SpatialScheme.WENO_Z)
    grid, material_map, inflow = setup(base)
    record = {}
    sigma_ref = lateral_sigma(reference_mc, grid)
    z = np.asarray(grid.z, dtype=float)
    r0 = distal_level(z, idd(reference_mc, grid), 0.8)
    depth_mask = z <= 0.9 * r0
    for label, config in (
        ("standard_steps", with_histories(base, STEP_CHECK_HISTORIES_EXTENDED)),
        (
            "halved_steps",
            halved_steps(base, STEP_CHECK_HISTORIES_EXTENDED),
        ),
    ):
        dose, runtime = run_bfp_mc_reference(
            config, material_map, grid, inflow
        )
        dose = np.asarray(dose, dtype=float)
        sigma = lateral_sigma(dose, grid)
        record[label] = {
            "histories": STEP_CHECK_HISTORIES_EXTENDED,
            "max_step_mm": config.references.bfp_mc_max_step_mm,
            "runtime_seconds": float(runtime),
            "max_dose_rel_difference": float(
                dose.max() / reference_mc.max() - 1.0
            ),
            "lateral_sigma_max_rel_difference": float(
                np.nanmax(
                    np.abs(sigma[depth_mask] / sigma_ref[depth_mask] - 1.0)
                )
            ),
            **idd_comparison(dose, reference_mc, grid),
        }
        print(f"  {label}: {json.dumps(record[label])}")
    return record


def check_scattering_and_removal(reference_mc):
    print("\n[3,4] Fermi-Eyges lateral width, removal and lateral escape")
    config = extended_homogeneous.build_config(SpatialScheme.WENO_Z)
    grid, material_map, inflow = setup(config)
    beam = config.beam
    z = np.asarray(grid.z, dtype=float)
    half_width = 0.5 * float(config.geometry.size[0])

    lookup_max = beam.energy + beam.cutoff_sigma * beam.energy_spread + 1e-9
    energy_grid, sigma_tr, sigma_r, range_grid, energy_of_range = (
        _build_lookups("G4_WATER", lookup_max, nuclear_on=True)
    )
    r0 = float(np.interp(beam.energy, energy_of_range, range_grid))

    # Fine depth quadrature for the Fermi-Eyges and survival integrals.
    s = np.linspace(0.0, r0, 20001)
    energy_s = np.interp(r0 - s, range_grid, energy_of_range)
    tr_s = np.interp(energy_s, energy_grid, sigma_tr)
    rem_s = np.interp(energy_s, energy_grid, sigma_r)

    sigma0_sq = float(beam.sigma[0]) ** 2 + float(grid.dx) ** 2 / 12.0
    fe_sigma = np.full_like(z, np.nan)
    fe_sigma_truncated = np.full_like(z, np.nan)
    survival = np.full_like(z, np.nan)
    inside = np.full_like(z, np.nan)
    for k, zk in enumerate(z):
        if zk >= r0:
            continue
        m = s <= zk
        var = sigma0_sq + np.trapezoid((zk - s[m]) ** 2 * tr_s[m], s[m])
        sig = math.sqrt(var)
        fe_sigma[k] = sig
        fe_sigma_truncated[k] = truncated_rms(sig, half_width)
        survival[k] = math.exp(-np.trapezoid(rem_s[m], s[m]))
        inside[k] = math.erf(half_width / (sig * math.sqrt(2.0))) ** 2

    mc_sigma = lateral_sigma(reference_mc, grid)
    sigma_mask = (z >= 5.0e-3) & (z <= 0.9 * r0)
    sigma_ratio = mc_sigma[sigma_mask] / fe_sigma_truncated[sigma_mask]

    exact_csda = np.asarray(
        analytical_csda_dose(config, material_map, grid, inflow), dtype=float
    )
    predicted_idd = idd(exact_csda, grid) * survival * inside
    mc_idd = idd(reference_mc, grid)
    plateau = (z >= 10.0e-3) & (z <= 0.8 * r0)
    idd_ratio = mc_idd[plateau] / predicted_idd[plateau]

    k_end = int(np.argmin(np.abs(z - 0.9 * r0)))
    record = {
        "csda_range_mm": 1000.0 * r0,
        "fermi_eyges": {
            "mc_over_fe_mean": float(np.mean(sigma_ratio)),
            "mc_over_fe_min": float(np.min(sigma_ratio)),
            "mc_over_fe_max": float(np.max(sigma_ratio)),
            "fe_sigma_at_0p9_r0_mm": 1000.0 * float(fe_sigma[k_end]),
            "fe_sigma_truncated_at_0p9_r0_mm": 1000.0
            * float(fe_sigma_truncated[k_end]),
            "mc_sigma_at_0p9_r0_mm": 1000.0 * float(mc_sigma[k_end]),
            "depth_range_mm": [5.0, 900.0 * r0],
        },
        "removal_and_escape": {
            "mc_over_prediction_mean": float(np.mean(idd_ratio)),
            "mc_over_prediction_min": float(np.min(idd_ratio)),
            "mc_over_prediction_max": float(np.max(idd_ratio)),
            "survival_at_0p8_r0": float(
                np.interp(0.8 * r0, z, np.nan_to_num(survival))
            ),
            "fraction_inside_domain_at_0p9_r0": float(inside[k_end]),
            "depth_range_mm": [10.0, 800.0 * r0],
        },
    }
    print(f"  {json.dumps(record)}")

    curves = {
        "z": z,
        "mc_sigma": mc_sigma,
        "fe_sigma_truncated": fe_sigma_truncated,
        "sigma_mask": sigma_mask,
        "mc_idd": mc_idd,
        "predicted_idd": predicted_idd,
        "plateau": plateau,
    }
    return record, curves


def save_figure(curves, path):
    z_mm = 1000.0 * curves["z"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))

    m = curves["sigma_mask"]
    ax = axes[0]
    ax.plot(z_mm[m], 1000.0 * curves["mc_sigma"][m], label="BFP Monte Carlo")
    ax.plot(
        z_mm[m],
        1000.0 * curves["fe_sigma_truncated"][m],
        "--",
        label="Fermi-Eyges",
    )
    ax.set_xlabel(r"Depth $z$ [mm]")
    ax.set_ylabel(r"Lateral RMS width [mm]")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    p = curves["plateau"]
    ax.plot(
        z_mm[p],
        100.0 * (curves["mc_idd"][p] / curves["predicted_idd"][p] - 1.0),
    )
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel(r"Depth $z$ [mm]")
    ax.set_ylabel(r"MC / prediction $-1$ [%]")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main():
    if not EXTENDED_REFERENCE.exists():
        raise SystemExit(
            f"Missing {EXTENDED_REFERENCE}. Run the extended homogeneous "
            "benchmark first."
        )
    reference_mc = np.asarray(
        np.load(EXTENDED_REFERENCE)["bfp_mc_dose"], dtype=float
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    results = {"steps_reduced": check_steps_reduced()}
    results["steps_extended"] = check_steps_extended(reference_mc)
    record, curves = check_scattering_and_removal(reference_mc)
    results.update(record)

    save_figure(curves, OUT_DIR / "reference_checks.pdf")
    (OUT_DIR / "reference_checks.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"\nTotal: {time.perf_counter() - start:.0f} s")
    print(f"Saved {OUT_DIR / 'reference_checks.json'}")


if __name__ == "__main__":
    main()
