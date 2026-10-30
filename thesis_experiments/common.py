from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from proton_solver import Solver, SpatialScheme
from proton_solver.metrics import ResultMetrics


# -----------------------------------------------------------------------------
# Reconstruction study definition
# -----------------------------------------------------------------------------

RECONSTRUCTIONS = {
    "first_order": ("First Order", SpatialScheme.FIRST_ORDER),
    "muscl": ("MUSCL", SpatialScheme.MUSCL),
    "weno_z": ("WENO-Z", SpatialScheme.WENO_Z),
}

# Thesis-friendly matplotlib defaults. PDF/SVG keep line work and text vectorized.
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def to_numpy(array):
    if array is None:
        return None

    if hasattr(array, "get"):
        return array.get()

    return np.asarray(array)


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


def validate_results(results):
    """Validate a First Order / MUSCL / WENO-Z reconstruction study."""

    if set(results) != set(RECONSTRUCTIONS):
        raise RuntimeError(
            "Reconstruction study must contain exactly: "
            f"{tuple(RECONSTRUCTIONS)}. Got {tuple(results)}."
        )

    baseline = results["first_order"]

    x_ref = to_numpy(baseline.x)
    y_ref = to_numpy(baseline.y)
    z_ref = to_numpy(baseline.z)
    reference_dose = to_numpy(baseline.reference_dose)

    if reference_dose is None:
        raise RuntimeError("The reconstruction study requires a reference dose.")

    for key, result in results.items():
        label = RECONSTRUCTIONS[key][0]

        np.testing.assert_allclose(
            to_numpy(result.x), x_ref, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            to_numpy(result.y), y_ref, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            to_numpy(result.z), z_ref, rtol=0.0, atol=0.0
        )

        dose = to_numpy(result.dose)

        if dose is None:
            raise RuntimeError(f"{label} did not produce a deterministic dose.")

        if dose.shape != reference_dose.shape:
            raise RuntimeError(
                f"{label} dose shape {dose.shape} differs from reference "
                f"shape {reference_dose.shape}."
            )

        if not np.isfinite(dose).all():
            raise RuntimeError(f"{label} contains non-finite dose values.")

        current_reference = to_numpy(result.reference_dose)

        if current_reference is None:
            raise RuntimeError(f"{label} has no comparison reference.")

        # The BFP cache is deliberately independent of the deterministic
        # reconstruction. Every scheme must therefore compare against exactly
        # the same kinetic reference field.
        np.testing.assert_allclose(
            current_reference,
            reference_dose,
            rtol=1.0e-12,
            atol=1.0e-14,
        )


# Backwards-compatible alias while the legacy top-level scripts still exist.
def validate_pair(first_order, weno_z):
    validate_results({
        "first_order": first_order,
        "muscl": first_order,
        "weno_z": weno_z,
    })


# -----------------------------------------------------------------------------
# Dose utilities
# -----------------------------------------------------------------------------


def central_xz(result, dose):
    # Same half-open cell rule as ResultMetrics.central_axis_indices.
    iy = ResultMetrics._axis_cell_index(to_numpy(result.y))
    return to_numpy(dose)[:, iy, :]


def extent_mm(result):
    x = to_numpy(result.x)
    z = to_numpy(result.z)

    dx = x[1] - x[0]
    dz = z[1] - z[0]

    return [
        (z[0] - 0.5 * dz) * 1000.0,
        (z[-1] + 0.5 * dz) * 1000.0,
        (x[0] - 0.5 * dx) * 1000.0,
        (x[-1] + 0.5 * dx) * 1000.0,
    ]


def relative_error_percent(result):
    dose = to_numpy(result.dose)
    reference = to_numpy(result.reference_dose)
    reference_max = float(np.max(reference))

    if reference_max <= 0.0:
        raise RuntimeError("Reference dose maximum must be positive.")

    return 100.0 * (dose - reference) / reference_max


def integrated_depth_dose(result, dose):
    dose = to_numpy(dose)

    x = to_numpy(result.x)
    y = to_numpy(result.y)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    return np.sum(dose, axis=(0, 1)) * dx * dy


# -----------------------------------------------------------------------------
# High-quality figure output
# -----------------------------------------------------------------------------


def save_figure(fig, stem):
    """Save a thesis figure as PDF, SVG, and a 600-dpi PNG preview."""

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


def _display_error_limit(results, maximum=50.0):
    """Return a shared symmetric error scale without letting FO hide WENO-Z."""

    largest = max(
        float(np.max(np.abs(relative_error_percent(result))))
        for result in results.values()
    )

    if largest <= 0.0:
        return 1.0

    return min(maximum, largest)


def save_dose_heatmap(result, title, save_stem, vmax):
    dose_xz = central_xz(result, result.dose)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    image = ax.imshow(
        dose_xz,
        origin="lower",
        aspect="auto",
        extent=extent_mm(result),
        cmap="inferno",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )

    fig.colorbar(image, ax=ax, label="Dose [Gy]")
    ax.set_xlabel(r"Depth $z$ [mm]")
    ax.set_ylabel(r"$x$ [mm]")
    ax.set_title(title)
    fig.tight_layout()

    save_figure(fig, save_stem)


def save_error_heatmap(result, title, save_stem, error_limit):
    error_xz = central_xz(result, relative_error_percent(result))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    image = ax.imshow(
        error_xz,
        origin="lower",
        aspect="auto",
        extent=extent_mm(result),
        cmap="coolwarm",
        vmin=-error_limit,
        vmax=error_limit,
        interpolation="nearest",
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Difference [% of reference maximum]",
    )

    ax.set_xlabel(r"Depth $z$ [mm]")
    ax.set_ylabel(r"$x$ [mm]")
    ax.set_title(title)
    fig.tight_layout()

    save_figure(fig, save_stem)


def save_reference_heatmap(result, result_dir, vmax):
    reference_xz = central_xz(result, result.reference_dose)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    image = ax.imshow(
        reference_xz,
        origin="lower",
        aspect="auto",
        extent=extent_mm(result),
        cmap="inferno",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )

    fig.colorbar(image, ax=ax, label="Dose [Gy]")
    ax.set_xlabel(r"Depth $z$ [mm]")
    ax.set_ylabel(r"$x$ [mm]")
    ax.set_title(result.reference_label)
    fig.tight_layout()

    save_figure(
        fig,
        Path(result_dir) / "reference_heatmap",
    )


def save_method_comparison(results, result_dir, dose_vmax, error_limit):
    """Save the main 2x3 First Order / MUSCL / WENO-Z thesis panel."""

    keys = tuple(RECONSTRUCTIONS)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.2, 6.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    dose_image = None
    error_image = None

    for column, key in enumerate(keys):
        label = RECONSTRUCTIONS[key][0]
        result = results[key]

        dose_image = axes[0, column].imshow(
            central_xz(result, result.dose),
            origin="lower",
            aspect="auto",
            extent=extent_mm(result),
            cmap="inferno",
            vmin=0.0,
            vmax=dose_vmax,
            interpolation="nearest",
        )

        error_image = axes[1, column].imshow(
            central_xz(result, relative_error_percent(result)),
            origin="lower",
            aspect="auto",
            extent=extent_mm(result),
            cmap="coolwarm",
            vmin=-error_limit,
            vmax=error_limit,
            interpolation="nearest",
        )

        axes[0, column].set_title(label)
        axes[1, column].set_xlabel(r"Depth $z$ [mm]")

    axes[0, 0].set_ylabel(r"$x$ [mm]")
    axes[1, 0].set_ylabel(r"$x$ [mm]")

    fig.colorbar(
        dose_image,
        ax=axes[0, :],
        label="Dose [Gy]",
        fraction=0.022,
        pad=0.02,
    )

    error_label = "Difference [% of reference maximum]"

    if error_limit >= 49.999:
        pass
        # error_label += " (clipped at $\\pm 50$%)"

    fig.colorbar(
        error_image,
        ax=axes[1, :],
        label=error_label,
        fraction=0.022,
        pad=0.02,
    )

    save_figure(
        fig,
        Path(result_dir) / "method_comparison",
    )


def save_comparison_idd(results, result_dir):
    baseline = results["first_order"]
    z_mm = to_numpy(baseline.z) * 1000.0

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    ax.plot(
        z_mm,
        integrated_depth_dose(baseline, baseline.reference_dose),
        "--",
        linewidth=2.0,
        label=baseline.reference_label,
    )

    for key, (label, _) in RECONSTRUCTIONS.items():
        result = results[key]
        ax.plot(
            z_mm,
            integrated_depth_dose(result, result.dose),
            linewidth=1.8,
            label=label,
        )

    ax.set_xlabel(r"Depth $z$ [mm]")
    ax.set_ylabel(r"Integrated dose [Gy m$^2$]")
    ax.set_title("Integrated depth-dose")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    save_figure(
        fig,
        Path(result_dir) / "comparison_IDD",
    )


def save_thesis_figures(results, result_dir):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    figure_dir = result_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    reference = to_numpy(results["first_order"].reference_dose)

    dose_vmax = max(
        float(np.max(reference)),
        *(
            float(np.max(to_numpy(result.dose)))
            for result in results.values()
        ),
    )

    error_limit = _display_error_limit(results)

    filename_stems = {
        "first_order": "First_order",
        "muscl": "MUSCL",
        "weno_z": "WENO_Z",
    }

    for key, result in results.items():
        label = RECONSTRUCTIONS[key][0]
        filename = filename_stems[key]

        save_dose_heatmap(
            result,
            title=label,
            save_stem=figure_dir / f"{filename}_heatmap",
            vmax=dose_vmax,
        )

        save_error_heatmap(
            result,
            title=f"{label} error",
            save_stem=figure_dir / f"{filename}_error_heatmap",
            error_limit=error_limit,
        )

    save_reference_heatmap(
        results["first_order"],
        figure_dir,
        dose_vmax,
    )

    save_method_comparison(
        results,
        figure_dir,
        dose_vmax,
        error_limit,
    )

    save_comparison_idd(
        results,
        figure_dir,
    )


# -----------------------------------------------------------------------------
# Persisted thesis data and tables
# -----------------------------------------------------------------------------


def save_run_artifacts(result, destination):
    """Copy persisted solver output and write standard thesis metrics."""

    if result.run_directory is None:
        raise RuntimeError(
            "Result has no run directory. "
            "Ensure OutputConfig(save_results=True) is enabled."
        )

    source = Path(result.run_directory)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    required_files = (
        "config.json",
        "metadata.json",
        "results.npz",
    )

    for filename in required_files:
        source_file = source / filename

        if not source_file.exists():
            raise RuntimeError(
                f"Expected persisted result file does not exist: {source_file}"
            )

        shutil.copy2(source_file, destination / filename)

    metrics = result.metrics.comparison_metrics()

    with (destination / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")

    return metrics


def _metric_value(metrics, *keys):
    value = metrics

    for key in keys:
        if value is None:
            return None
        value = value.get(key)

    return value


def _format_number(value, digits=3):
    if value is None or value == "-":
        return "--"

    return f"{float(value):.{digits}f}"


def _write_metrics_table_tex(metrics_by_scheme, path):
    rows = [
        (
            r"$\gamma$ 3\%/3 mm [\%]",
            ("gamma", "3_percent_3_mm", "pass_rate_percent"),
        ),
        (
            r"$\gamma$ 2\%/2 mm [\%]",
            ("gamma", "2_percent_2_mm", "pass_rate_percent"),
        ),
        (
            r"$\gamma$ 1\%/1 mm [\%]",
            ("gamma", "1_percent_1_mm", "pass_rate_percent"),
        ),
        (
            r"Maximum-dose difference [\%]",
            ("maximum_dose", "relative_difference_percent"),
        ),
        (
            r"Peak-depth difference [mm]",
            ("central_axis_bragg_peak", "depth_difference_mm"),
        ),
        (
            r"Runtime [s]",
            ("performance", "deterministic_runtime_seconds"),
        ),
        (
            r"Speedup vs. BFP MC [$\times$]",
            ("performance", "speedup_vs_bfp_mc_x"),
        ),
    ]

    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Metric & First Order & MUSCL & WENO-Z \\",
        r"\midrule",
    ]

    for label, key_path in rows:
        values = [
            _format_number(
                _metric_value(metrics_by_scheme[key], *key_path)
            )
            for key in RECONSTRUCTIONS
        ]

        lines.append(
            f"{label} & {values[0]} & {values[1]} & {values[2]} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    Path(path).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def save_thesis_data(results, result_dir):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_scheme = {}

    for key, result in results.items():
        metrics_by_scheme[key] = save_run_artifacts(
            result,
            result_dir / key,
        )

    with (result_dir / "comparison_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics_by_scheme, file, indent=2)
        file.write("\n")

    with (result_dir / "comparison_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "scheme",
            "gamma_3_3_percent",
            "gamma_2_2_percent",
            "gamma_1_1_percent",
            "maximum_dose_difference_percent",
            "peak_depth_difference_mm",
            "runtime_seconds",
            "speedup_vs_bfp_mc_x",
        ])

        for key in RECONSTRUCTIONS:
            metrics = metrics_by_scheme[key]
            writer.writerow([
                key,
                _metric_value(
                    metrics,
                    "gamma",
                    "3_percent_3_mm",
                    "pass_rate_percent",
                ),
                _metric_value(
                    metrics,
                    "gamma",
                    "2_percent_2_mm",
                    "pass_rate_percent",
                ),
                _metric_value(
                    metrics,
                    "gamma",
                    "1_percent_1_mm",
                    "pass_rate_percent",
                ),
                _metric_value(
                    metrics,
                    "maximum_dose",
                    "relative_difference_percent",
                ),
                _metric_value(
                    metrics,
                    "central_axis_bragg_peak",
                    "depth_difference_mm",
                ),
                _metric_value(
                    metrics,
                    "performance",
                    "deterministic_runtime_seconds",
                ),
                _metric_value(
                    metrics,
                    "performance",
                    "speedup_vs_bfp_mc_x",
                ),
            ])

    _write_metrics_table_tex(
        metrics_by_scheme,
        result_dir / "metrics_table.tex",
    )


# -----------------------------------------------------------------------------
# Standard reconstruction-study runner
# -----------------------------------------------------------------------------


def run_reconstruction_study(build_config, result_dir):
    """Run First Order, MUSCL, and WENO-Z against one cached BFP reference."""

    results = {}

    for key, (label, spatial_scheme) in RECONSTRUCTIONS.items():
        print()
        print("=" * 72)
        print(label)
        print("=" * 72)

        result = Solver(
            build_config(spatial_scheme)
        ).solve()

        result.metrics.summary()
        results[key] = result

    validate_results(results)
    save_thesis_figures(results, result_dir)
    save_thesis_data(results, result_dir)

    print()
    print("Reconstruction study complete")
    print("=" * 72)

    for key, (label, _) in RECONSTRUCTIONS.items():
        print(f"{label:12s}: {results[key].run_directory}")

    print(f"Thesis results: {Path(result_dir).resolve()}")

    return results
