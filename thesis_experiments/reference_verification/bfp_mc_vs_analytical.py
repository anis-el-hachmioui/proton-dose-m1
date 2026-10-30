from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from proton_solver import (
    Backend,
    Beam,
    Geometry,
    Material,
    Numerics,
    OutputConfig,
    Physics,
    Reference,
    References,
    Scattering,
    Solver,
    SolverConfig,
)

from proton_solver.metrics import ResultMetrics

from thesis_experiments.common import (
    central_xz,
    extent_mm,
    integrated_depth_dose,
    save_figure,
    to_numpy,
)


RESULT_DIR = Path(
    "thesis_results/reference_verification/bfp_mc_vs_analytical"
)

HISTORIES = (
    10_000,
    30_000,
    100_000,
    300_000,
    1_000_000,
    3_000_000,
    10_000_000,
    100_000_000,
)


def build_config(histories):
    """CSDA + straight-streaming limit with analytical and BFP references."""

    return SolverConfig(
        geometry=Geometry(
            size=(0.024, 0.024, 0.20),
            resolution=(100, 100, 400),
            background_material=Material.WATER,
            view_geometry=False,
        ),
        beam=Beam(
            n_protons=1e6,
            energy=150.0,
            energy_spread=1.0,
            sigma=(0.002, 0.002),
            center=(0.0, 0.0),
            angular_spread=0.0,
            cutoff_sigma=5.0,
        ),
        backend=Backend.CPU,
        solve_deterministic=False,
        physics=Physics(
            scattering=Scattering.OFF,
            nuclear=False,
        ),
        numerics=Numerics(
            energy_min=1e-4,
            energy_window_sigma=5.0,
        ),
        references=References(
            reference=Reference.BFP_MC,
            additional_references=(Reference.ANALYTICAL,),
            bfp_mc_histories=int(histories),
            bfp_mc_seed=137,
            bfp_mc_threads=12,
            bfp_mc_max_step_mm=0.25,
            bfp_mc_max_energy_loss_mev=0.25,
            bfp_mc_max_scattering_optical_depth=0.02,
            bfp_mc_stop_energy_mev=1.0,
            bfp_mc_force_rerun=False,
        ),
        output=OutputConfig(
            save_results=False,
        ),
    )


def relative_errors(numerical, reference):
    numerical = np.asarray(numerical, dtype=float)
    reference = np.asarray(reference, dtype=float)
    difference = numerical - reference

    l1_den = np.sum(np.abs(reference))
    l2_den = np.sqrt(np.sum(reference**2))
    linf_den = np.max(np.abs(reference))

    return {
        "l1": float(np.sum(np.abs(difference)) / l1_den),
        "l2": float(np.sqrt(np.sum(difference**2)) / l2_den),
        "linf": float(np.max(np.abs(difference)) / linf_den),
    }


def central_peak_depth(result, dose):
    dose = to_numpy(dose)
    x = to_numpy(result.x)
    y = to_numpy(result.y)
    z = to_numpy(result.z)

    # Same half-open cell rule as ResultMetrics.central_axis_indices.
    ix = ResultMetrics._axis_cell_index(x)
    iy = ResultMetrics._axis_cell_index(y)
    iz = int(np.argmax(dose[ix, iy, :]))

    return float(z[iz])


def save_convergence_plot(records):
    histories = np.asarray([row["histories"] for row in records], dtype=float)
    l1 = np.asarray([row["l1"] for row in records])
    l2 = np.asarray([row["l2"] for row in records])
    linf = np.asarray([row["linf"] for row in records])

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.loglog(histories, l1, "o-", label=r"$L^1$")
    ax.loglog(histories, l2, "o-", label=r"$L^2$")
    ax.loglog(histories, linf, "o-", label=r"$L^\infty$")

    expected = l2[0] * np.sqrt(histories[0] / histories)
    ax.loglog(histories, expected, "--", label=r"$N_{\mathrm{hist}}^{-1/2}$")

    ax.set_xlabel(r"Monte Carlo histories $N_{\mathrm{hist}}$")
    ax.set_ylabel("Relative error vs. analytical solution")
    ax.set_title("BFP particle Monte Carlo statistical convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    save_figure(
        fig,
        RESULT_DIR / "bfp_mc_statistical_convergence",
    )


def save_highest_history_comparison(result):
    analytical = to_numpy(result.analytical_dose)
    bfp = to_numpy(result.bfp_mc_dose)

    vmax = max(float(np.max(analytical)), float(np.max(bfp)))
    reference_max = float(np.max(analytical))
    error = 100.0 * (bfp - analytical) / reference_max
    error_limit = min(25.0, max(1.0, float(np.max(np.abs(error)))))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.2, 3.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    image_0 = axes[0].imshow(
        central_xz(result, analytical),
        origin="lower",
        aspect="auto",
        extent=extent_mm(result),
        cmap="inferno",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )

    axes[1].imshow(
        central_xz(result, bfp),
        origin="lower",
        aspect="auto",
        extent=extent_mm(result),
        cmap="inferno",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )

    image_1 = axes[2].imshow(
        central_xz(result, error),
        origin="lower",
        aspect="auto",
        extent=extent_mm(result),
        cmap="coolwarm",
        vmin=-error_limit,
        vmax=error_limit,
        interpolation="nearest",
    )

    axes[0].set_title("Analytical")
    axes[1].set_title("BFP Monte Carlo")
    axes[2].set_title("Difference")

    axes[0].set_ylabel(r"$x$ [mm]")

    for ax in axes:
        ax.set_xlabel(r"Depth $z$ [mm]")

    fig.colorbar(
        image_0,
        ax=axes[:2],
        label="Dose [Gy]",
        fraction=0.025,
        pad=0.02,
    )

    fig.colorbar(
        image_1,
        ax=axes[2],
        label="Difference [% of analytical maximum]",
        fraction=0.05,
        pad=0.04,
    )

    save_figure(
        fig,
        RESULT_DIR / "analytical_bfp_heatmaps",
    )

    z_mm = to_numpy(result.z) * 1000.0

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        z_mm,
        integrated_depth_dose(result, analytical),
        linewidth=2.0,
        label="Analytical",
    )
    ax.plot(
        z_mm,
        integrated_depth_dose(result, bfp),
        "--",
        linewidth=1.8,
        label="BFP Monte Carlo",
    )

    ax.set_xlabel(r"Depth $z$ [mm]")
    ax.set_ylabel(r"Integrated dose [Gy m$^2$]")
    ax.set_title("Analytical and BFP Monte Carlo depth-dose")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    save_figure(
        fig,
        RESULT_DIR / "analytical_bfp_IDD",
    )


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    final_result = None

    for histories in HISTORIES:
        print()
        print("=" * 72)
        print(f"BFP MC vs analytical: {histories:,} histories")
        print("=" * 72)

        result = Solver(
            build_config(histories)
        ).solve()

        analytical = to_numpy(result.analytical_dose)
        bfp = to_numpy(result.bfp_mc_dose)

        errors = relative_errors(bfp, analytical)

        analytical_peak = central_peak_depth(result, analytical)
        bfp_peak = central_peak_depth(result, bfp)

        record = {
            "histories": int(histories),
            "runtime_seconds": float(result.bfp_mc_runtime),
            **errors,
            "analytical_peak_depth_mm": 1000.0 * analytical_peak,
            "bfp_peak_depth_mm": 1000.0 * bfp_peak,
            "peak_depth_difference_mm": 1000.0 * (bfp_peak - analytical_peak),
        }

        records.append(record)
        final_result = result

        print(json.dumps(record, indent=2))

    with (RESULT_DIR / "reference_convergence.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(records, file, indent=2)
        file.write("\n")

    with (RESULT_DIR / "reference_convergence.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    save_convergence_plot(records)
    save_highest_history_comparison(final_result)

    print()
    print(f"Saved reference verification to {RESULT_DIR.resolve()}")


if __name__ == "__main__":
    main()
