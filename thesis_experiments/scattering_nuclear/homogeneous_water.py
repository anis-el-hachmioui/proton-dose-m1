from pathlib import Path

from proton_solver import (
    Backend,
    Beam,
    EnergyScheme,
    Geometry,
    Material,
    Numerics,
    OutputConfig,
    Physics,
    Reference,
    References,
    Scattering,
    SolverConfig,
    WaveSpeeds,
)

from thesis_experiments.common import run_reconstruction_study


RESULT_DIR = Path(
    "thesis_results/scattering_nuclear/homogeneous_water"
)


def build_config(spatial_scheme):
    """Homogeneous water with Wentzel scattering and nuclear attenuation."""

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
        physics=Physics(
            scattering=Scattering.WENTZEL,
            nuclear=True,
        ),
        numerics=Numerics(
            spatial_scheme=spatial_scheme,
            energy_scheme=EnergyScheme.IF_SSPRK3,
            wave_speeds=WaveSpeeds.EXACT,
            max_energy_step=0.25,
            cfl=0.9,
            energy_min=1e-4,
            energy_window_sigma=5.0,
            moving_active_domain=False,
        ),
        references=References(
            reference=Reference.BFP_MC,
            bfp_mc_histories=100_000_000,
            bfp_mc_seed=137,
            bfp_mc_threads=12,
            bfp_mc_max_step_mm=0.25,
            bfp_mc_max_energy_loss_mev=0.25,
            bfp_mc_max_scattering_optical_depth=0.02,
            bfp_mc_stop_energy_mev=1.0,
            bfp_mc_force_rerun=False,
        ),
        output=OutputConfig(
            save_results=True,
        ),
    )


def main():
    run_reconstruction_study(
        build_config,
        RESULT_DIR,
    )


if __name__ == "__main__":
    main()
