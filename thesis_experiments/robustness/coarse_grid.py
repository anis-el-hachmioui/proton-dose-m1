"""WENO-Z benchmarks on coarser grids.

All six benchmarks are repeated on three coarser grids, all other settings
unchanged:

    coarse_grid       50x50x200   0.48 x 0.48 x 1 mm   1e7 histories
    coarse_grid_1mm   24x24x200   1 mm cubes           3e6 histories
    coarse_grid_2mm   12x12x100   2 mm cubes           3e6 histories

The voxels are 8, 35 and 280 times larger than on the benchmark grid, so these
history numbers give about the statistical uncertainty per voxel of the
benchmark reference or better. The reduced-model cases also store the exact
analytical solution (additional reference in their configuration). The change
of the WENO-Z errors with the grid shows how much of the remaining difference
is discretization error.

Runs whose results already exist are skipped.

Usage: uv run python -m thesis_experiments.robustness.coarse_grid
"""

import dataclasses

from thesis_experiments.robustness._common import run_study

GRIDS = (
    ("coarse_grid", (50, 50, 200), 10_000_000),
    ("coarse_grid_1mm", (24, 24, 200), 3_000_000),
    ("coarse_grid_2mm", (12, 12, 100), 3_000_000),
)


def coarsen_to(resolution, histories):
    def coarsen(config):
        return dataclasses.replace(
            config,
            geometry=dataclasses.replace(
                config.geometry, resolution=resolution
            ),
            references=dataclasses.replace(
                config.references, bfp_mc_histories=histories
            ),
        )

    return coarsen


def main():
    for study, resolution, histories in GRIDS:
        run_study(study, coarsen_to(resolution, histories))


if __name__ == "__main__":
    main()