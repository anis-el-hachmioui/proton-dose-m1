"""Example: lateral water-bone interface, extended model, against Monte Carlo.

A 50 MeV pencil beam travels along a water-bone interface: compact bone fills
the half x < 0 over the full depth. The two halves of the beam reach
different depths, and protons scatter across the interface. The extended
model is compared with the BFP Monte Carlo solver. A few minutes on a laptop.

The beam axis lies on the interface, where the M1 solution deviates most
from the Monte Carlo (thesis Section 5.5.3 and Appendix D.3). The
central-axis peak-dose difference of about 10% printed here is that local
effect, together with the coarse grid, and not the typical error.

Usage: uv run python examples/transversal_slabs.py
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
            # An even number of cells in x puts the interface x = 0 on a
            # cell face.
            resolution=(48, 48, 120),
            background_material=Material.WATER,
            objects=[
                Box(                          # left half: compact bone
                    x_min=-0.012, x_max=0.0,
                    y_min=-0.012, y_max=0.012,
                    z_min=0.0, z_max=0.03,
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
            scattering=Scattering.WENTZEL,
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
            bfp_mc_histories=1_000_000,
            bfp_mc_threads=os.cpu_count() or 1,
        ),
        output=OutputConfig(save_results=True),
    )

    result = Solver(config).solve()
    result.metrics.summary()
    result.plots.all()


if __name__ == "__main__":
    main()
