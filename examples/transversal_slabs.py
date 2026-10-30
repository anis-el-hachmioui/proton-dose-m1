from proton_solver import (
    Solver,
    SolverConfig,
    Geometry,
    Box,
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
        size=(0.024, 0.024, 0.025),
        resolution=(60, 61, 200),
        background_material=Material.WATER,
        objects=[
            Box(
                # Left half: compact bone
                x_min=-0.012,
                x_max=0.0,
                y_min=-0.012,
                y_max=0.012,
                z_min=0.0,
                z_max=0.025,
                material=Material.BONE_COMPACT,
            ),
        ],
        view_geometry=True,
    )

    beam = Beam(
        n_protons=1e5,
        energy=50.0,
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
            scattering=True,
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
        ),

        references=References(
            reference=Reference.TOPAS,
            topas_histories=1_000_000,
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