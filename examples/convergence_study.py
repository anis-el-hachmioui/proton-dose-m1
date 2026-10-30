from proton_solver import (
    SolverConfig,
    Geometry,
    Beam,
    Physics,
    Numerics,
    ConvergenceStudy,
    Refinement,
    Region,
    Observable,
    Backend,
    Material,
    SpatialScheme,
    EnergyScheme,
    Reference,
)


def main():

    config = SolverConfig(
        geometry=Geometry(
            size=(0.03, 0.03, 0.03),
            resolution=(11, 11, 40),
            background_material=Material.WATER,
        ),

        beam=Beam(
            n_protons=1e5,
            energy=70.0,
            energy_spread=5.0,
            sigma=(0.10, 0.10),
            center=(0.0, 0.0),
            angular_spread=0.0,
        ),

        backend=Backend.AUTO,

        physics=Physics(
            scattering=False,
            nuclear=False,
        ),

        numerics=Numerics(
            spatial_scheme=SpatialScheme.WENO_Z,
            energy_scheme=EnergyScheme.SSPRK3,
            max_energy_step=0.25,
        ),
    )

    study = ConvergenceStudy(
        config=config,

        refinement=Refinement.nz(
            [40, 80, 160, 320, 640],
            couple_energy_step=True,
        ),

        observables=[
            Observable.DOSE,
            Observable.PHI.at_energy(60.0),
        ],

        reference=Reference.SELF,

        region=Region.central_axis(
            z_min=0.0,
            z_max=0.027,
        ),

        expected_order=3,
    )

    result = study.run()

    result.summary()
    result.plot()
    result.save()


if __name__ == "__main__":
    main()