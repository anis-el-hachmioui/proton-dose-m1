"""Example: export a configuration as a TOPAS case and run it.

Requires a working TOPAS installation (not needed for anything else in this
repository). The solver configuration is written as a TOPAS parameter file,
TOPAS is run, and its dose is read back on the solver grid. TOPAS simulates
full physics (Geant4), so its dose differs from the BFP model by the model
error as well as by the numerical error; the thesis therefore uses the BFP
Monte Carlo solver as reference instead.

Usage: uv run python examples/topas_case.py
"""

from pathlib import Path

import numpy as np

from proton_solver import (
    Backend,
    Beam,
    EnergyScheme,
    Geometry,
    Material,
    Numerics,
    Physics,
    Reference,
    References,
    Scattering,
    SolverConfig,
    SpatialScheme,
    WaveSpeeds,
)
from proton_solver.topas.parser import parse_topas_dose
from proton_solver.topas.runner import run_topas
from proton_solver.topas.writer import write_topas_case


def main():
    config = SolverConfig(
        geometry=Geometry(
            size=(0.04, 0.04, 0.05),
            resolution=(21, 21, 161),
            background_material=Material.WATER,
        ),
        beam=Beam(
            n_protons=1e5,
            energy=50.0,
            energy_spread=1.0,
            sigma=(0.002, 0.002),
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
        ),
        references=References(
            reference=Reference.TOPAS,
            topas_histories=10_000,
        ),
    )

    case = write_topas_case(config, Path("topas_test"))
    print("Parameters:", case.parameter_file)

    dose_file = run_topas(case)
    dose = parse_topas_dose(dose_file, config)

    print("TOPAS dose file:", dose_file)
    print("Shape:", dose.shape, " finite:", bool(np.all(np.isfinite(dose))))
    print("Maximum dose [Gy]:", float(np.max(dose)))


if __name__ == "__main__":
    main()
