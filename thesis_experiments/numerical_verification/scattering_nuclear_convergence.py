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
    "scattering_nuclear_convergence"
)


def build_config():
    """
    Coupled space-energy convergence for the final reduced BFP model:
    CSDA + streaming + Wentzel angular scattering + primary nuclear
    attenuation, integrated with IF-SSPRK3.
    """

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

            # Same beam and geometry as the reduced-physics study so
            # the only changed ingredients are the local source terms.
            sigma=(0.10, 0.10),

            center=(0.0, 0.0),
            angular_spread=0.0,
            cutoff_sigma=5.0,
        ),

        backend=Backend.CPU,

        physics=Physics(
            scattering=Scattering.WENTZEL,
            nuclear=True,
        ),

        numerics=Numerics(
            spatial_scheme=SpatialScheme.WENO_Z,
            energy_scheme=EnergyScheme.IF_SSPRK3,
            wave_speeds=WaveSpeeds.EXACT,

            # Refined together with Nz by the existing ConvergenceStudy.
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
            "CSDA + streaming + Wentzel scattering + nuclear attenuation"
        ),
        title_prefix=(
            "Extended model"
        ),
    )


if __name__ == "__main__":
    main()
