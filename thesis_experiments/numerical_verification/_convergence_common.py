from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from proton_solver import (
    ConvergenceStudy,
    Observable,
    Reference,
    Refinement,
    Region,
)


# Keep vector text editable/searchable where possible and avoid Type-3 PDF fonts.
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


RESOLUTIONS = (
    40,
    80,
    160,
    320,
    640,
    1280,
    2560,
)

SNAPSHOT_ENERGY = 60.0

# Energy-only study: fixed grid of the base configuration (20 x 20 x 40),
# energy step (E_max - 60 MeV) / k = 35/k MeV, so every level lands exactly
# on the 60 MeV snapshot. The CFL limit on this grid is about 0.25 MeV, so
# all steps, from 35/160 = 0.22 MeV down, are set by the energy step.
ENERGY_STEP_DIVISORS = (
    160,
    320,
    640,
    1280,
    2560,
    5120,
)

# Space-only study: Nz refined with one fixed energy step, 35/1280 MeV,
# below the CFL limit of the finest grid (about 0.03 MeV for Nz = 640).
SPATIAL_RESOLUTIONS = (
    40,
    80,
    160,
    320,
    640,
)

SPATIAL_STEP_DIVISOR =  5120


def _slug(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(".", "p")
    )


def _fitted_order(h, errors) -> float:
    h = np.asarray(h, dtype=float)
    errors = np.asarray(errors, dtype=float)

    valid = (
        np.isfinite(h)
        & np.isfinite(errors)
        & (h > 0.0)
        & (errors > 0.0)
    )

    if np.count_nonzero(valid) < 2:
        return float("nan")

    slope, _ = np.polyfit(
        np.log(h[valid]),
        np.log(errors[valid]),
        1,
    )

    return float(slope)


def _save_figure(fig, stem: Path):
    stem.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def _display_label(observable) -> str:
    """Typeset observable name for plot titles and LaTeX tables."""
    if observable.energy is None:
        return observable.name.capitalize()
    symbol = r"$\Phi$" if observable.name == "phi" else observable.name
    return f"{symbol} at {observable.energy:g} MeV"


def save_thesis_plots(
    result,
    result_dir: Path,
    *,
    title_prefix: str,
    x_label: str = r"Spatial step $\Delta z$ [m]",
):
    figure_dir = result_dir / "figures"
    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for observable in result.observables:
        h = result.h

        l1 = result._errors(
            observable,
            "l1",
        )

        l2 = result._errors(
            observable,
            "l2",
        )

        linf = result._errors(
            observable,
            "linf",
        )

        fig, ax = plt.subplots(
            figsize=(7.2, 4.8)
        )

        ax.loglog(
            h,
            l1,
            "o-",
            linewidth=1.6,
            markersize=5,
            label=r"$L^1$",
        )

        ax.loglog(
            h,
            l2,
            "s-",
            linewidth=1.6,
            markersize=5,
            label=r"$L^2$",
        )

        ax.loglog(
            h,
            linf,
            "^-",
            linewidth=1.6,
            markersize=5,
            label=r"$L^\infty$",
        )

        if (
            result.expected_order is not None
            and len(h) > 0
            and l2[0] > 0.0
        ):
            expected = (
                l2[0]
                * (h / h[0])
                ** result.expected_order
            )

            ax.loglog(
                h,
                expected,
                "--",
                linewidth=1.5,
                label=(
                    rf"$\mathcal{{O}}(h^{{{result.expected_order:g}}})$"
                ),
            )

        fitted_l2 = _fitted_order(
            h,
            l2,
        )

        ax.set_xlabel(
            x_label
        )

        ax.set_ylabel(
            "Relative successive-grid error"
        )

        ax.set_title(
            f"{title_prefix}: {_display_label(observable)}"
            f"\nFitted $L^2$ order = {fitted_l2:.3f}"
        )

        ax.grid(
            True,
            which="both",
            alpha=0.25,
        )

        ax.legend()
        fig.tight_layout()

        _save_figure(
            fig,
            figure_dir
            / f"{_slug(observable.label)}_convergence",
        )


def save_summary_files(
    result,
    result_dir: Path,
    *,
    physics_label: str,
):
    summary = {
        "physics": physics_label,
        "expected_order": result.expected_order,
        "refinement_ratio": result.refinement_ratio,
        "resolutions": list(result.refinement.values),
        "observables": {},
    }

    tex_rows = []

    for observable in result.observables:
        norms = {}

        for norm in (
            "l1",
            "l2",
            "linf",
        ):
            errors = result._errors(
                observable,
                norm,
            )

            norms[norm] = {
                "errors": [
                    float(value)
                    for value in errors
                ],
                "fitted_order": _fitted_order(
                    result.h,
                    errors,
                ),
            }

        summary["observables"][
            observable.label
        ] = norms

        tex_rows.append(
            (
                _display_label(observable),
                norms["l1"]["fitted_order"],
                norms["l2"]["fitted_order"],
                norms["linf"]["fitted_order"],
            )
        )

    with (
        result_dir / "convergence_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Observable & $p_{L^1}$ & $p_{L^2}$ & $p_{L^\infty}$ \\",
        r"\midrule",
    ]

    for label, p1, p2, pinf in tex_rows:
        label_tex = label

        lines.append(
            f"{label_tex} & "
            f"{p1:.3f} & "
            f"{p2:.3f} & "
            f"{pinf:.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])

    (
        result_dir / "convergence_table.tex"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _run_study(
    config,
    result_dir,
    refinement,
    *,
    physics_label: str,
    title_prefix: str,
    x_label: str,
    expected_order: float | None,
):
    result_dir = Path(result_dir)

    # Re-running the verification script should replace only its own
    # generated result directory, never unrelated thesis output.
    if result_dir.exists():
        shutil.rmtree(
            result_dir
        )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    study = ConvergenceStudy(
        config=config,

        refinement=refinement,

        observables=[
            Observable.DOSE,
            Observable.PHI.at_energy(
                SNAPSHOT_ENERGY
            ),
        ],

        reference=Reference.SELF,

        region=Region.central_axis(
            z_min=0.0,
            z_max=0.027,
        ),

        expected_order=expected_order,
    )

    result = study.run()

    result.summary()

    # Keep the original ConvergenceStudy output as the machine-readable
    # numerical record.  It contains study.json, errors.csv, and the
    # library's standard diagnostic PNGs.
    result.save(
        result_dir / "data"
    )

    # Thesis-quality vector/raster versions.
    save_thesis_plots(
        result,
        result_dir,
        title_prefix=title_prefix,
        x_label=x_label,
    )

    save_summary_files(
        result,
        result_dir,
        physics_label=physics_label,
    )

    print()
    print(
        f"Thesis convergence results: {result_dir.resolve()}"
    )

    return result


def _snapshot_distance(config) -> float:
    """Energy between the upper energy E_max and the 60 MeV snapshot."""

    energy_start = (
        config.beam.energy
        + config.numerics.energy_window_sigma
        * config.beam.energy_spread
    )

    return energy_start - SNAPSHOT_ENERGY


def run_convergence_study(
    config,
    result_dir,
    *,
    physics_label: str,
    title_prefix: str,
):
    """Coupled space-energy refinement (Section 5.3)."""

    return _run_study(
        config,
        result_dir,
        Refinement.nz(
            RESOLUTIONS,
            couple_energy_step=True,
        ),
        physics_label=physics_label,
        title_prefix=title_prefix,
        x_label=r"Spatial step $\Delta z$ [m]",
        expected_order=3,
    )


def run_convergence_study_energy(
    config,
    result_dir,
    *,
    physics_label: str,
    title_prefix: str,
):
    """Energy-only refinement on the fixed grid of config."""

    config = replace(
        config,
        numerics=replace(
            config.numerics,
            max_energy_step=(
                _snapshot_distance(config)
                / ENERGY_STEP_DIVISORS[0]
            ),
        ),
    )

    return _run_study(
        config,
        result_dir,
        Refinement.energy(
            ENERGY_STEP_DIVISORS,
        ),
        physics_label=physics_label,
        title_prefix=f"{title_prefix}, energy only",
        x_label=r"Energy step $\Delta\tau$ [MeV]",
        expected_order=3,
    )


def run_convergence_study_spatial(
    config,
    result_dir,
    *,
    physics_label: str,
    title_prefix: str,
):
    """Space-only refinement in z with one fixed energy step."""

    config = replace(
        config,
        numerics=replace(
            config.numerics,
            max_energy_step=(
                _snapshot_distance(config)
                / SPATIAL_STEP_DIVISOR
            ),
        ),
    )

    return _run_study(
        config,
        result_dir,
        Refinement.nz(
            SPATIAL_RESOLUTIONS,
            couple_energy_step=False,
        ),
        physics_label=physics_label,
        title_prefix=f"{title_prefix}, space only",
        x_label=r"Spatial step $\Delta z$ [m]",
        expected_order=None,
    )