"""Example: bone slab across the beam, extended model, against Monte Carlo.

A 50 MeV pencil beam crosses a 5 mm slab of compact bone in water. The
extended model (Wentzel-VI angular scattering and primary-proton nuclear
removal) is compared with the BFP Monte Carlo solver, which solves the same
transport equation with the same material data. A few minutes on a laptop.

Usage: uv run python examples/longitudinal_slabs.py
"""

import os

from proton_solver import (
    Backend,
    Beam,
    Box,
    EnergyScheme,
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
    SpatialScheme,
    WaveSpeeds,
)


def main():
    config = SolverConfig(
        geometry=Geometry(
            size=(0.024, 0.024, 0.03),
            resolution=(48, 48, 120),         # 0.5 x 0.5 x 0.25 mm cells
            background_material=Material.WATER,
            objects=[
                Box(                          # bone slab, 10 to 15 mm deep
                    x_min=-0.012, x_max=0.012,
                    y_min=-0.012, y_max=0.012,
                    z_min=0.010, z_max=0.015,
                    material=Material.BONE_COMPACT,
                ),
            ],
            view_geometry=False,
        ),
        beam=Beam(
            n_protons=1e6,
            energy=50.0,
            energy_spread=1.0,
            sigma=(0.002, 0.002),
            center=(0.0, 0.0),
            angular_spread=0.0,
            cutoff_sigma=5.0,
        ),
        backend=Backend.CPU,
        physics=Physics(
            scattering=Scattering.WENTZEL,    # extended model
            nuclear=True,
        ),
        numerics=Numerics(
            spatial_scheme=SpatialScheme.WENO_Z,
            energy_scheme=EnergyScheme.IF_SSPRK3,
            wave_speeds=WaveSpeeds.EXACT,
            max_energy_step=0.25,
            cfl=0.9,
        ),
        references=References(
            reference=Reference.BFP_MC,
            bfp_mc_histories=1_000_000,       # 1e8 in the thesis benchmarks
            bfp_mc_threads=os.cpu_count() or 1,
        ),
        output=OutputConfig(save_results=True),
    )

    result = Solver(config).solve()
    result.metrics.summary()
    result.plots.all()


if __name__ == "__main__":
    main()
