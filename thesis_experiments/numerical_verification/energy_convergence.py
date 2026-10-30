"""Energy-only self-convergence for both models (question of D. Lathouwers).

Same test problem as Section 5.3, on its fixed 20 x 20 x 40 grid; only the
energy step is refined (35/k MeV, k = 160, ..., 5120). Expected order 3.

Usage: uv run python -m thesis_experiments.numerical_verification.energy_convergence
"""

from pathlib import Path

from thesis_experiments.numerical_verification import (
    reduced_physics_convergence as reduced,
    scattering_nuclear_convergence as extended,
)
from thesis_experiments.numerical_verification._convergence_common import (
    run_convergence_study_energy,
)


def main():
    run_convergence_study_energy(
        reduced.build_config(),
        Path("thesis_results/numerical_verification/reduced_physics_energy_convergence"),
        physics_label="CSDA + streaming",
        title_prefix="Reduced model",
    )

    run_convergence_study_energy(
        extended.build_config(),
        Path("thesis_results/numerical_verification/scattering_nuclear_energy_convergence"),
        physics_label="CSDA + streaming + Wentzel scattering + nuclear attenuation",
        title_prefix="Extended model",
    )


if __name__ == "__main__":
    main()