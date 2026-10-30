import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..beam import Beam
from .grid import Grid


def _gaussian_cell_average(centers, spacing, mean, sigma):
    """Return exact finite-volume cell averages of a 1D Gaussian."""

    scale = math.sqrt(2.0) * sigma
    lower = (centers - 0.5 * spacing - mean) / scale
    upper = (centers + 0.5 * spacing - mean) / scale

    mass = np.fromiter(
        (
            0.5 * (math.erf(b) - math.erf(a))
            for a, b in zip(lower, upper)
        ),
        dtype=float,
        count=len(centers),
    )

    return mass / spacing


def _transverse_profile(beam: Beam, grid: Grid):
    """Construct the normalized finite-volume transverse Gaussian beam profile."""
    x = grid.x.get() if hasattr(grid.x, "get") else np.asarray(grid.x)
    y = grid.y.get() if hasattr(grid.y, "get") else np.asarray(grid.y)

    gx = _gaussian_cell_average(
        x, grid.dx, beam.center[0], beam.sigma[0]
    )
    gy = _gaussian_cell_average(
        y, grid.dy, beam.center[1], beam.sigma[1]
    )

    profile = gx[:, None] * gy[None, :]

    radius_squared = (
        ((x[:, None] - beam.center[0]) / beam.sigma[0]) ** 2
        + ((y[None, :] - beam.center[1]) / beam.sigma[1]) ** 2
    )

    inside = radius_squared <= beam.cutoff_sigma**2
    
    profile = np.where(inside, profile, 0.0)

    discrete_mass = profile.sum() * grid.dx * grid.dy

    if not math.isfinite(discrete_mass) or discrete_mass <= 0:
        raise ValueError(
            "Beam profile has zero or non-finite mass on the domain."
        )

    return profile / discrete_mass


@dataclass(frozen=True)
class BeamInflow:
    beam: Beam
    spatial: Any

    @classmethod
    def from_beam(cls, beam: Beam, grid: Grid, xp):
        """Precompute the transverse beam distribution."""

        spatial = beam.n_protons * _transverse_profile(
            beam,
            grid,
        )

        return cls(
            beam=beam,
            spatial=xp.asarray(spatial),
        )

    def state(self, energy: float, xp):
        """Return the M1 inflow state at a given kinetic energy."""

        if self.beam.energy_spread == 0:
            raise ValueError(
                "Zero energy spread is not supported by the Gaussian inflow."
            )

        energy_density = (
            math.exp(
                -(energy - self.beam.energy) ** 2
                / (2.0 * self.beam.energy_spread**2)
            )
            / (
                math.sqrt(2.0 * math.pi)
                * self.beam.energy_spread
            )
        )

        phi = self.spatial * energy_density
        reduced_flux = math.exp(-self.beam.angular_spread**2)

        U = xp.empty(
            (4, *phi.shape),
            dtype=phi.dtype,
        )

        U[0] = phi
        U[1] = 0.0
        U[2] = 0.0
        U[3] = reduced_flux * phi

        return U
    

def fill_ghost_cells(
    U: Any,
    inflow: BeamInflow,
    energy: float,
    ghost_width: int,
    xp,
    boundary_state: Any | None = None,
):
    """
    Extend the M1 state with physical boundary ghost cells.

    For WENO-Z, the z-min ghost-cell averages are obtained from
    a quartic polynomial constrained by

        p(0) = U_in

    and the first four interior finite-volume cell averages.

    This gives a high-order smooth extension through the inflow
    boundary instead of filling all ghost cells with the same
    boundary value.
    """

    if type(ghost_width) is not int or ghost_width <= 0:
        raise ValueError(
            "ghost_width must be a positive integer."
        )

    ng = ghost_width
    _, nx, ny, nz = U.shape

    U_ext = xp.empty(
        (
            4,
            nx + 2 * ng,
            ny + 2 * ng,
            nz + 2 * ng,
        ),
        dtype=U.dtype,
    )

    # --------------------------------------------------------
    # Physical domain
    # --------------------------------------------------------

    U_ext[
        :,
        ng:ng + nx,
        ng:ng + ny,
        ng:ng + nz,
    ] = U

    # --------------------------------------------------------
    # z-min: proton inflow
    # --------------------------------------------------------

    U_in = (
        inflow.state(energy, xp)
        if boundary_state is None
        else boundary_state
    )

    if U_in.shape != (4, nx, ny):
        raise ValueError(
            f"Boundary state has shape {U_in.shape}, "
            f"expected {(4, nx, ny)}."
        )

    if ng == 1:
        # First-order reconstruction.
        U_ext[
            :,
            ng:ng + nx,
            ng:ng + ny,
            ng - 1,
        ] = U_in

    elif ng == 2:
        # WENO3 requires two ghost cells.
        #
        # Construct a degree-2 polynomial p(z) satisfying
        #
        #     p(0) = U_in
        #
        # and whose cell averages over
        #
        #     [0,h], [h,2h]
        #
        # equal U0 and U1.
        #
        # Integrating that polynomial over the two ghost
        # cells gives the formulas below.

        if nz < 2:
            raise ValueError(
                "WENO3 inflow extrapolation requires "
                "at least two z cells."
            )

        U0 = U[:, :, :, 0]
        U1 = U[:, :, :, 1]

        # Nearest ghost cell: [-h, 0]
        G1 = (
            3.0 * U_in
            - 2.5 * U0
            + 0.5 * U1
        )

        # Second ghost cell: [-2h, -h]
        G2 = (
            9.0 * U_in
            - 10.5 * U0
            + 2.5 * U1
        )

        U_ext[
            :,
            ng:ng + nx,
            ng:ng + ny,
            ng - 1,
        ] = G1

        U_ext[
            :,
            ng:ng + nx,
            ng:ng + ny,
            ng - 2,
        ] = G2

    elif ng == 3:
        # WENO-Z5 requires three ghost cells.
        #
        # Construct a degree-4 polynomial p(z) satisfying
        #
        #     p(0) = U_in
        #
        # and whose cell averages over
        #
        #     [0,h], [h,2h], [2h,3h], [3h,4h]
        #
        # equal U0, U1, U2, U3.
        #
        # Integrating that polynomial over the three ghost
        # cells gives the formulas below.

        if nz < 4:
            raise ValueError(
                "WENO-Z inflow extrapolation requires "
                "at least four z cells."
            )

        U0 = U[:, :, :, 0]
        U1 = U[:, :, :, 1]
        U2 = U[:, :, :, 2]
        U3 = U[:, :, :, 3]

        # Nearest ghost cell: [-h, 0]
        G1 = (
            5.0 * U_in
            - (77.0 / 12.0) * U0
            + (43.0 / 12.0) * U1
            - (17.0 / 12.0) * U2
            + 0.25 * U3
        )

        # Second ghost cell: [-2h, -h]
        G2 = (
            25.0 * U_in
            - (505.0 / 12.0) * U0
            + (335.0 / 12.0) * U1
            - (145.0 / 12.0) * U2
            + (9.0 / 4.0) * U3
        )

        # Third ghost cell: [-3h, -2h]
        G3 = (
            75.0 * U_in
            - (545.0 / 4.0) * U0
            + (395.0 / 4.0) * U1
            - (181.0 / 4.0) * U2
            + (35.0 / 4.0) * U3
        )

        U_ext[
            :,
            ng:ng + nx,
            ng:ng + ny,
            ng - 1,
        ] = G1

        U_ext[
            :,
            ng:ng + nx,
            ng:ng + ny,
            ng - 2,
        ] = G2

        U_ext[
            :,
            ng:ng + nx,
            ng:ng + ny,
            ng - 3,
        ] = G3

    else:
        raise NotImplementedError(
            f"Unsupported ghost width: {ng}"
        )

    # --------------------------------------------------------
    # z-max: zero-gradient outflow
    # --------------------------------------------------------

    U_ext[
        :,
        ng:ng + nx,
        ng:ng + ny,
        ng + nz:,
    ] = U_ext[
        :,
        ng:ng + nx,
        ng:ng + ny,
        ng + nz - 1:ng + nz,
    ]

    # --------------------------------------------------------
    # x boundaries: zero gradient
    # --------------------------------------------------------

    U_ext[
        :,
        :ng,
        :,
        :,
    ] = U_ext[
        :,
        ng:ng + 1,
        :,
        :,
    ]

    U_ext[
        :,
        ng + nx:,
        :,
        :,
    ] = U_ext[
        :,
        ng + nx - 1:ng + nx,
        :,
        :,
    ]

    # --------------------------------------------------------
    # y boundaries: zero gradient
    # --------------------------------------------------------

    U_ext[
        :,
        :,
        :ng,
        :,
    ] = U_ext[
        :,
        :,
        ng:ng + 1,
        :,
    ]

    U_ext[
        :,
        :,
        ng + ny:,
        :,
    ] = U_ext[
        :,
        :,
        ng + ny - 1:ng + ny,
        :,
    ]

    return U_ext