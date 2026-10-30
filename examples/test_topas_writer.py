from pathlib import Path
import numpy as np

from proton_solver import (
    SolverConfig,
    Geometry,
    Beam,
    Physics,
    Numerics,
    References,
    Backend,
    Material,
    SpatialScheme,
    EnergyScheme,
    WaveSpeeds,
    Reference,
)
from proton_solver.topas.parser import (
    parse_topas_dose,
)
from proton_solver.topas.runner import run_topas
from proton_solver.topas.writer import write_topas_case


def main():

    geometry = Geometry(
        size=(0.04, 0.04, 0.05),
        resolution=(21, 21, 161),
        background_material=Material.WATER,
        view_geometry=False,
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
        ),

        references=References(
            reference=Reference.TOPAS,
            topas_histories=10_000,
        ),
    )

    case = write_topas_case(
        config,
        Path("topas_test"),
    )

    dose_file = run_topas(case)

    topas_dose = parse_topas_dose(
    dose_file,
    config,
)

    print(
        "TOPAS dose shape:",
        topas_dose.shape,
    )

    print(
        "TOPAS dose finite:",
        np.all(np.isfinite(topas_dose)),
    )

    print(
        "TOPAS minimum dose:",
        np.min(topas_dose),
    )

    print(
        "TOPAS maximum dose:",
        np.max(topas_dose),
    )

    print(
        "TOPAS dose:",
        dose_file,
    )

    print("Generated TOPAS case")
    print("Parameters:", case.parameter_file)
    print("PreInit:", case.preinit_macro)
    print("Sequence:", case.sequence_macro)


if __name__ == "__main__":
    main()