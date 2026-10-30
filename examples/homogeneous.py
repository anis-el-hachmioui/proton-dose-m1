from proton_solver import (
    Solver,
    SolverConfig,
    Geometry,
    Beam,
    Physics,
    Numerics,
    References,
    OutputConfig,
    Backend,
    Material,
    SpatialScheme,
    EnergyScheme,
    WaveSpeeds,
    Reference,
)


def main():

    geometry = Geometry(
        size=(0.024/2, 0.024/2, 0.20),
        resolution=(20,20,80),
        background_material=Material.WATER,
        view_geometry=True,
    )

    beam = Beam(
        n_protons=1e5,
        energy=150.0,
        energy_spread=1.0,
        sigma=(0.002, 0.002),
        center=(0.0, 0.0),
        angular_spread=0.0,
        cutoff_sigma=5.0,
    )

    config = SolverConfig(
        geometry=geometry,
        beam=beam,
        backend=Backend.CPU,

        physics=Physics(
            scattering=False,
            nuclear=False,
        ),

        numerics=Numerics(
            spatial_scheme=SpatialScheme.WENO_Z,
            energy_scheme=EnergyScheme.SSPRK3,
            wave_speeds=WaveSpeeds.EXACT,
            max_energy_step=0.25,
            cfl=0.9,
            energy_min=1e-4,
            energy_window_sigma=5.0,
            active_domain=True,
            active_relative_tolerance=1e-14,
        ),
        # references=References(
        #     reference=Reference.TOPAS,
        #     topas_histories=10_000,
        # ),

        references=References(
            reference=Reference.ANALYTICAL,
        ),

        output=OutputConfig(
            save_results=True,
        ),
    )

    result = Solver(config).solve()

    result.metrics.summary()
    result.plots.all()


if __name__ == "__main__":
    main()