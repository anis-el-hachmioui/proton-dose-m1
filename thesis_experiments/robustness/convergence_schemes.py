"""Self-convergence studies of first-order and MUSCL reconstruction.

The reduced- and extended-model convergence studies of
thesis_experiments/numerical_verification, repeated with first-order and
MUSCL reconstruction instead of WENO-Z. Everything else, including the
refinement path and the error measure, is unchanged. (The plots keep the
third-order guide line of the shared plotting code; the fitted orders in the
summary files are what matter.)

Output: thesis_results/numerical_verification/<model>_convergence_<scheme>/

Usage: uv run python -m thesis_experiments.robustness.convergence_schemes
"""

import dataclasses
from pathlib import Path

from proton_solver import SpatialScheme

from thesis_experiments.numerical_verification import (
    reduced_physics_convergence as reduced,
    scattering_nuclear_convergence as extended,
)
from thesis_experiments.numerical_verification._convergence_common import (
    run_convergence_study,
)

STUDIES = (
    (reduced, "reduced_physics_convergence", "CSDA + streaming", "Reduced model"),
    (
        extended,
        "scattering_nuclear_convergence",
        "CSDA + streaming + Wentzel scattering + nuclear attenuation",
        "Extended model",
    ),
)

SCHEMES = (
    ("first_order", "first order", SpatialScheme.FIRST_ORDER),
    ("muscl", "MUSCL", SpatialScheme.MUSCL),
)


def main():
    for module, name, physics_label, title in STUDIES:
        for key, scheme_label, scheme in SCHEMES:
            result_dir = Path(
                f"thesis_results/numerical_verification/{name}_{key}"
            )
            if (result_dir / "data" / "study.json").exists():
                print(f"Skipping {result_dir} (done)")
                continue
            base = module.build_config()
            config = dataclasses.replace(
                base,
                numerics=dataclasses.replace(
                    base.numerics, spatial_scheme=scheme
                ),
            )
            print("=" * 72)
            print(f"{title}, {scheme_label}")
            print("=" * 72)
            run_convergence_study(
                config,
                result_dir,
                physics_label=physics_label,
                title_prefix=f"{title}, {scheme_label}",
            )


if __name__ == "__main__":
    main()
