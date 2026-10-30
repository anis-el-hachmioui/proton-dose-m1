"""Space-only self-convergence for both models (question of D. Lathouwers).

Same test problem as Section 5.3; only Nz is refined (40, ..., 640, fixed
20 x 20 transverse grid) with one fixed energy step of 35/1280 MeV, below the
CFL limit of the finest grid.

Usage: uv run python -m thesis_experiments.numerical_verification.spatial_convergence
"""

from pathlib import Path

from thesis_experiments.numerical_verification import (
    reduced_physics_convergence as reduced,
    scattering_nuclear_convergence as extended,
)
from thesis_experiments.numerical_verification._convergence_common import (
    run_convergence_study_spatial,
)


def main():
    run_convergence_study_spatial(
        reduced.build_config(),
        Path("thesis_results/numerical_verification/reduced_physics_spatial_convergence"),
        physics_label="CSDA + streaming",
        title_prefix="Reduced model",
    )

    run_convergence_study_spatial(
        extended.build_config(),
        Path("thesis_results/numerical_verification/scattering_nuclear_spatial_convergence"),
        physics_label="CSDA + streaming + Wentzel scattering + nuclear attenuation",
        title_prefix="Extended model",
    )


if __name__ == "__main__":
    main()