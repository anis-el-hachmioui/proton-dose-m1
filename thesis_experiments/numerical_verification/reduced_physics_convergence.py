from pathlib import Path

from proton_solver import (
    SolverConfig,
    Geometry,
    Beam,
    Physics,
    Numerics,
    Backend,
    Material,
    SpatialScheme,
    EnergyScheme,
    WaveSpeeds,
    Scattering,
)

from thesis_experiments.numerical_verification._convergence_common import (
    run_convergence_study,
)


RESULT_DIR = Path(
    "thesis_results/numerical_verification/"
    "reduced_physics_convergence"
)


def build_config():
    """Coupled space-energy convergence for CSDA + streaming."""

    return SolverConfig(
        geometry=Geometry(
            size=(0.03, 0.03, 0.03),
            resolution=(20, 20, 40),
            background_material=Material.WATER,
            view_geometry=False,
        ),

        beam=Beam(
            n_protons=1e5,
            energy=70.0,
            energy_spread=5.0,

            # Effectively uniform in x and y, making this a clean
            # one-dimensional verification problem in z.
            sigma=(0.10, 0.10),

            center=(0.0, 0.0),
            angular_spread=0.0,
            cutoff_sigma=5.0,
        ),

        backend=Backend.CPU,

        physics=Physics(
            scattering=Scattering.OFF,
            nuclear=False,
        ),

        numerics=Numerics(
            spatial_scheme=SpatialScheme.WENO_Z,
            energy_scheme=EnergyScheme.IF_SSPRK3,
            wave_speeds=WaveSpeeds.EXACT,

            # Refinement.nz(..., couple_energy_step=True) scales this
            # as 1/Nz, so Delta z and the energy-step cap are refined
            # together.
            max_energy_step=2.0,
            cfl=0.9,

            energy_min=1e-4,
            energy_window_sigma=5.0,

            moving_active_domain=False,
        ),
    )


def main():
    run_convergence_study(
        build_config(),
        RESULT_DIR,
        physics_label=(
            "CSDA + streaming"
        ),
        title_prefix=(
            "Reduced model"
        ),
    )


if __name__ == "__main__":
    main()
