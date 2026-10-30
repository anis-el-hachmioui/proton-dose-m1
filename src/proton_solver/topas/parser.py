from pathlib import Path

import numpy as np

from ..config import SolverConfig


def parse_topas_dose(
    dose_file: Path,
    config: SolverConfig,
):
    """Read a TOPAS dose CSV and return dose on the solver grid."""

    dose_file = Path(dose_file)

    if not dose_file.exists():
        raise FileNotFoundError(
            f"TOPAS dose file was not found: {dose_file}"
        )

    data = np.loadtxt(
        dose_file,
        delimiter=",",
        comments="#",
    )

    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(
            "Expected TOPAS dose CSV with four columns: "
            "ix, iy, iz, dose."
        )

    nx, ny, nz = (
        config.geometry.resolution
    )

    expected_voxels = (
        nx * ny * nz
    )

    if data.shape[0] != expected_voxels:
        raise ValueError(
            "TOPAS dose grid does not match the solver grid: "
            f"expected {expected_voxels} voxels, "
            f"found {data.shape[0]}."
        )

    indices = data[:, :3].astype(int)
    values = data[:, 3]

    ix = indices[:, 0]
    iy = indices[:, 1]
    iz = indices[:, 2]

    if (
        np.any(ix < 0)
        or np.any(ix >= nx)
        or np.any(iy < 0)
        or np.any(iy >= ny)
        or np.any(iz < 0)
        or np.any(iz >= nz)
    ):
        raise ValueError(
            "TOPAS dose CSV contains voxel indices "
            "outside the solver grid."
        )

    dose = np.zeros(
        (nx, ny, nz),
        dtype=float,
    )

    dose[
        ix,
        iy,
        iz,
    ] = values

    # TOPAS dose corresponds to topas_histories simulated protons.
    # Rescale it to the physical proton count used by the solver.
    scale = (
        config.beam.n_protons
        / config.references.topas_histories
    )

    dose *= scale

    return dose