"""Example: self-convergence study of the deterministic scheme.

A smooth, essentially one-dimensional problem (70 MeV beam, much wider than
the 30 mm water cube) is solved on successively refined grids. Each level is
compared with the next finer one (successive self-convergence), so no exact
solution is needed. Three refinement modes are available:

    Refinement.nz(values, couple_energy_step=True)   space and energy together
    Refinement.nz(values, couple_energy_step=False)  space only, fixed energy step
    Refinement.energy(values)                         energy only, fixed grid

For space-only and energy-only refinement the energy step must stay below
the CFL limit on every level; each level prints the number of CFL-limited
steps, which should then be zero. The settings used in the thesis are in
thesis_experiments/numerical_verification/_convergence_common.py.

Usage: uv run python examples/convergence_study.py
"""

from proton_solver import (
    Backend,
    Beam,
    ConvergenceStudy,
    EnergyScheme,
    Geometry,
    Material,
    Numerics,
    Observable,
    Physics,
    Reference,
    Refinement,
    Region,
    Scattering,
    SolverConfig,
    SpatialScheme,
    WaveSpeeds,
)


def main():
    config = SolverConfig(
        geometry=Geometry(
            size=(0.03, 0.03, 0.03),
            resolution=(20, 20, 40),          # Nz is set by the refinement
            background_material=Material.WATER,
        ),
        beam=Beam(
            n_protons=1e5,
            energy=70.0,
            energy_spread=5.0,
            sigma=(0.10, 0.10),               # wide beam: nearly 1D in z
            center=(0.0, 0.0),
            angular_spread=0.0,
            cutoff_sigma=5.0,
        ),
        backend=Backend.CPU,
        physics=Physics(scattering=Scattering.OFF, nuclear=False),
        numerics=Numerics(
            spatial_scheme=SpatialScheme.WENO_Z,
            energy_scheme=EnergyScheme.IF_SSPRK3,
            wave_speeds=WaveSpeeds.EXACT,
            max_energy_step=2.0,              # scaled as 1/Nz below
            cfl=0.9,
        ),
    )

    study = ConvergenceStudy(
        config=config,
        refinement=Refinement.nz(
            [40, 80, 160, 320],
            couple_energy_step=True,
        ),
        observables=[
            Observable.DOSE,
            Observable.PHI.at_energy(60.0),
        ],
        reference=Reference.SELF,
        region=Region.central_axis(z_min=0.0, z_max=0.027),
        expected_order=3,
    )

    result = study.run()
    result.summary()
    result.plot()
    result.save()                             # writes to results/


if __name__ == "__main__":
    main()
