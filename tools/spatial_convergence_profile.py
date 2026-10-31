"""Where along the beam axis does the space-only convergence error sit?

The space-only self-convergence study
(thesis_experiments/numerical_verification/spatial_convergence.py) converges at
high order for the dose, but the fitted order of the fluence at 60 MeV drops on
the finest grids. This tool repeats the levels of that study with identical
settings (taken from _convergence_common.py), stores the central-axis profiles,
and plots the successive-grid difference of Phi(60 MeV) against depth.

The difference converges at about fifth order wherever the solution is smooth
and at a lower order in a narrow band around the depth at which protons that
entered with 80 MeV have slowed down to 60 MeV. The electronic stopping power
of water is the ICRU90 table of Geant4 (G4EmStandardPhysics_option4), with
nodes at, among others, 60, 80 and 100 MeV; its logarithmic slope d ln S/d ln E
jumps at these nodes. The right-hand panel shows this slope and the depths
that correspond to the nodes.

Outputs, per model, the pairwise L2 orders in depth bands.

Usage (from the repository root):
    uv run tools/spatial_convergence_profile.py            # both models
    uv run tools/spatial_convergence_profile.py reduced    # one model
Runtime: as one model of spatial_convergence.py. Profiles are cached in the
output directory, so the figure can be redone without rerunning.

Output: thesis_results/numerical_verification/spatial_convergence_profile/
            profiles_<model>_nz<N>.npz
            spatial_convergence_profile.{pdf,png}
            band_orders.json, band_orders_table.tex
"""

from __future__ import annotations

import csv
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proton_solver import Material, Reference, Region, Solver  # noqa: E402
from proton_solver.convergence import _relative_errors, _restrict_z  # noqa: E402
from proton_solver.physics.stopping_power import stopping_power  # noqa: E402

from thesis_experiments.numerical_verification import (  # noqa: E402
    reduced_physics_convergence,
    scattering_nuclear_convergence,
)
from thesis_experiments.numerical_verification._convergence_common import (  # noqa: E402
    SNAPSHOT_ENERGY,
    SPATIAL_RESOLUTIONS,
    SPATIAL_STEP_DIVISOR,
    _snapshot_distance,
)

OUT_DIR =REPO_ROOT / "thesis_results/numerical_verification/spatial_convergence_profile"
STOPPING_POWER_CSV = REPO_ROOT / "data/stopping_power/G4_WATER.csv"

MODELS = {
    "reduced": (reduced_physics_convergence, "Reduced model"),
    "extended": (scattering_nuclear_convergence, "Extended model"),
}
Z_MAX_M = 0.027
# Depth bands [mm]: entrance, smooth interior, around the 80 MeV node, rest.
BANDS_MM = ((0.0, 1.0), (1.0, 18.0), (18.0, 24.0), (24.0, 27.0), (0.0, 27.0))
# ICRU90 proton energy nodes of Geant4 in the energy range of this problem.
ICRU90_NODES_MEV = (60.0, 80.0, 100.0)


def level_config(model: str, nz: int):
    module, _ = MODELS[model]
    config = module.build_config()
    return dataclasses.replace(
        config,
        geometry=dataclasses.replace(
            config.geometry,
            resolution=(*config.geometry.resolution[:2], nz),
            view_geometry=False,
        ),
        numerics=dataclasses.replace(
            config.numerics,
            max_energy_step=_snapshot_distance(config) / SPATIAL_STEP_DIVISOR,
        ),
        references=dataclasses.replace(config.references, reference=Reference.NONE),
        output=dataclasses.replace(config.output, save_results=False),
    )


def profiles(model: str, nz: int):
    """Central-axis profiles of one level (cached)."""

    path = OUT_DIR / f"profiles_{model}_nz{nz}.npz"
    if not path.exists():
        config = level_config(model, nz)
        start = time.perf_counter()
        result = Solver(config).solve(snapshot_energies=(SNAPSHOT_ENERGY,))
        print(f"[{model}] Nz={nz}: {time.perf_counter() - start:.0f} s", flush=True)
        region = Region.central_axis()  # full depth; bands are applied later
        np.savez_compressed(
            path,
            z=np.asarray(result.grid.z),
            phi60=region.extract(np.asarray(result.snapshots[SNAPSHOT_ENERGY][0]), result.grid),
            dose=region.extract(np.asarray(result.dose), result.grid),
            max_energy_step=config.numerics.max_energy_step,
        )
    data = np.load(path)
    return {key: data[key] for key in data.files}


def successive_differences(levels):
    """Pointwise and banded differences between successive levels."""

    pairs = []
    for coarse, fine in zip(levels[:-1], levels[1:]):
        nz = len(coarse["z"])
        z = coarse["z"]
        entry = {"nz": nz, "z": z}
        for key in ("phi60", "dose"):
            fine_restricted = _restrict_z(fine[key], nz)
            evaluated = z <= Z_MAX_M
            scale = np.max(np.abs(fine_restricted[evaluated]))
            entry[f"{key}_pointwise"] = np.abs(coarse[key] - fine_restricted) / scale
            entry[f"{key}_bands"] = {}
            for lo, hi in BANDS_MM:
                band = (z >= lo * 1e-3) & (z <= hi * 1e-3)
                entry[f"{key}_bands"][(lo, hi)] = _relative_errors(
                    coarse[key][band], fine_restricted[band]
                ).l2
        pairs.append(entry)
    return pairs


def depth_of_initial_energy(initial_energies_mev):
    """Depth at which a proton entering with E0 reaches the snapshot energy."""

    energies = np.geomspace(1e-3, 300.0, 20001)
    inverse = 1.0 / np.array([stopping_power(Material.WATER, float(e)) for e in energies])
    csda_range = np.concatenate(
        [[0.0], np.cumsum(0.5 * (inverse[1:] + inverse[:-1]) * np.diff(energies))]
    )
    range_of = lambda e: np.interp(e, energies, csda_range)  # noqa: E731
    return {e0: (range_of(e0) - range_of(SNAPSHOT_ENERGY)) * 1e3 for e0 in initial_energies_mev}


def stopping_power_log_slope(e_min=50.0, e_max=105.0):
    with STOPPING_POWER_CSV.open(encoding="utf-8") as file:
        rows = [row for row in csv.reader(file) if row and row[0][0].isdigit()]
    energy = np.array([float(row[0]) for row in rows])
    s = np.array([float(row[1]) for row in rows])
    keep = (energy > e_min) & (energy < e_max)
    log_e, log_s = np.log(energy[keep]), np.log(s[keep])
    return np.sqrt(energy[keep][1:] * energy[keep][:-1]), np.diff(log_s) / np.diff(log_e)


def main():
    models = sys.argv[1:] or list(MODELS)
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}; choose from {list(MODELS)}.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    node_depths = depth_of_initial_energy(ICRU90_NODES_MEV)
    results = {}
    for model in models:
        levels = [profiles(model, nz) for nz in SPATIAL_RESOLUTIONS]
        results[model] = successive_differences(levels)

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(4.6 * (len(models) + 1), 4.0))
    for ax, model in zip(axes, models):
        for pair in results[model][1:]:  # skip the coarsest pair for readability
            z = pair["z"]
            shown = z <= Z_MAX_M
            nz = pair["nz"]
            ax.semilogy(z[shown] * 1e3, pair["phi60_pointwise"][shown] + 1e-18,
                        label=rf"$N_z$ = {nz}/{2 * nz}")
        for e0, depth in node_depths.items():
            if 0.0 <= depth <= Z_MAX_M * 1e3:
                ax.axvline(depth, color="0.4", ls="--", lw=0.9)
                ax.text(depth, 0.97, f" {e0:g} MeV", transform=ax.get_xaxis_transform(),
                        va="top", ha="left", fontsize=8, color="0.3")
        ax.set_xlabel("Depth $z$ [mm]")
        ax.set_ylabel(r"$|\Phi_\ell - \mathcal{P}\Phi_{\ell+1}|\,/\,\max\Phi_{\ell+1}$")
        ax.set_title(rf"{MODELS[model][1]}: $\Phi$ at {SNAPSHOT_ENERGY:g} MeV")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")

    energy, slope = stopping_power_log_slope()
    ax = axes[-1]
    ax.plot(energy, slope, lw=1.2)
    for e0 in ICRU90_NODES_MEV:
        ax.axvline(e0, color="0.4", ls="--", lw=0.9)
    ax.set_xlabel("Energy $E$ [MeV]")
    ax.set_ylabel(r"$d\ln S/d\ln E$ of the tabulated $S$ (water)")
    ax.set_title("Stopping-power table (ICRU90 tabulation energies dashed)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spatial_convergence_profile.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "spatial_convergence_profile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------- band orders
    summary = {
        "snapshot_energy_mev": SNAPSHOT_ENERGY,
        "resolutions": list(SPATIAL_RESOLUTIONS),
        "energy_step_divisor": SPATIAL_STEP_DIVISOR,
        "depth_of_icru90_nodes_mm": {f"{e:g}": d for e, d in node_depths.items()},
        "models": {},
    }
    lines = [
        r"\begin{tabular}{ll" + "c" * (len(SPATIAL_RESOLUTIONS) - 2) + "}",
        r"\toprule",
        # Pairwise order p(L2), listed at the finer of the two grid pairs,
        # as in the ConvergenceStudy summary.
        "Model & Depth band [mm] & "
        + " & ".join(
            rf"$N_z$ {SPATIAL_RESOLUTIONS[i + 1]}/{SPATIAL_RESOLUTIONS[i + 2]}"
            for i in range(len(SPATIAL_RESOLUTIONS) - 2)
        )
        + r" \\",
        r"\midrule",
    ]
    for model in models:
        bands = {}
        print(f"\n{MODELS[model][1]}: Phi at {SNAPSHOT_ENERGY:g} MeV, pairwise L2 order per depth band")
        for band in BANDS_MM:
            errors = np.array([pair["phi60_bands"][band] for pair in results[model]])
            orders = np.log2(errors[:-1] / errors[1:])
            bands[f"{band[0]:g}-{band[1]:g}"] = {
                "l2_errors": errors.tolist(),
                "pairwise_orders": orders.tolist(),
            }
            print(f"  {band[0]:>4g}-{band[1]:<4g} mm  errors {np.array2string(errors, precision=2)}"
                  f"  orders {np.array2string(orders, precision=2)}")
            lines.append(
                f"{MODELS[model][1].split()[0]} & {band[0]:g}--{band[1]:g} & "
                + " & ".join(f"{p:.2f}" for p in orders)
                + r" \\"
            )
        summary["models"][model] = bands
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (OUT_DIR / "band_orders.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "band_orders_table.tex").write_text("\n".join(lines), encoding="utf-8")

    print("\nDepth at which a proton entering with E0 reaches "
          f"{SNAPSHOT_ENERGY:g} MeV: "
          + ", ".join(f"{e:g} MeV -> {d:.1f} mm" for e, d in node_depths.items()))
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
