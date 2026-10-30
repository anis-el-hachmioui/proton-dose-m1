"""Example: proton pencil beam in homogeneous water, reduced model.

The reduced model (continuous energy loss and streaming only) has an exact
solution, so the deterministic dose is compared with the analytical CSDA
dose. Runs in a few minutes on a laptop. The 1 mm cells along the beam are
coarse for a 150 MeV Bragg peak: the peak dose comes out about 8% low and
the peak one cell shallower. Refine the z resolution to reduce this (see the
coarse-grid study in Appendix D.2 of the thesis).

Usage: uv run python examples/homogeneous.py
"""

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
    Solver,
    SolverConfig,
    SpatialScheme,
    WaveSpeeds,
)


def main():
    config = SolverConfig(
        geometry=Geometry(
            size=(0.024, 0.024, 0.20),        # x, y, z extent [m]
            resolution=(48, 48, 200),         # 0.5 x 0.5 x 1 mm cells
            background_material=Material.WATER,
            view_geometry=False,              # True opens a 3D preview
        ),
        beam=Beam(
            n_protons=1e6,                    # dose normalization
            energy=150.0,                     # mean energy [MeV]
            energy_spread=1.0,                # sigma_E [MeV]
            sigma=(0.002, 0.002),             # transverse sigma [m]
            center=(0.0, 0.0),
            angular_spread=0.0,
            cutoff_sigma=5.0,
        ),
        backend=Backend.CPU,
        physics=Physics(
            scattering=Scattering.OFF,        # reduced model
            nuclear=False,
        ),
        numerics=Numerics(
            spatial_scheme=SpatialScheme.WENO_Z,
            energy_scheme=EnergyScheme.IF_SSPRK3,
            wave_speeds=WaveSpeeds.EXACT,
            max_energy_step=0.25,
            cfl=0.9,
        ),
        references=References(
            reference=Reference.ANALYTICAL,
        ),
        output=OutputConfig(
            save_results=True,                # writes to results/<run>/
        ),
    )

    result = Solver(config).solve()
    result.metrics.summary()
    result.plots.all()


if __name__ == "__main__":
    main()
