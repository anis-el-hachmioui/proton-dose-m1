from typing import Any

import numpy as np

from ..enums import SpatialScheme, WaveSpeeds
from .boundary import BeamInflow, fill_ghost_cells
from .flux import hll_flux
from .grid import Grid
from .reconstruction_methods import (
    first_order,
    muscl,
    weno3,
    weno_z,
)


def _get_reconstruction(scheme: SpatialScheme):
    """Return the reconstruction module for the requested spatial scheme."""

    if scheme is SpatialScheme.FIRST_ORDER:
        return first_order

    if scheme is SpatialScheme.MUSCL:
        return muscl

    if scheme is SpatialScheme.WENO3:
        return weno3

    if scheme is SpatialScheme.WENO_Z:
        return weno_z
    
    raise ValueError(f"Unknown spatial scheme: {scheme}")


def _as_int(value) -> int:
    """Convert a NumPy/CuPy scalar to a Python int."""

    if hasattr(value, "item"):
        value = value.item()

    return int(value)


def _as_float(value) -> float:
    """Convert a NumPy/CuPy scalar to a Python float."""

    if hasattr(value, "item"):
        value = value.item()

    return float(value)


def find_active_box(
    U: Any,
    boundary_state: Any,
    padding: int,
    relative_tolerance: float,
    xp,
):
    """
    Return the smallest padded physical-cell box containing active flux.

    Activity is determined from phi and includes the current z-min inflow.
    Upper bounds are exclusive. None is returned when both the state
    and the inflow are identically zero.
    """

    phi = U[0]
    phi_in = boundary_state[0]

    scale = max(
        _as_float(xp.max(phi)),
        _as_float(xp.max(phi_in)),
    )

    if scale == 0.0:
        return None

    threshold = relative_tolerance * scale

    active = phi > threshold
    active[:, :, 0] |= phi_in > threshold

    ix = xp.flatnonzero(xp.any(active, axis=(1, 2)))
    iy = xp.flatnonzero(xp.any(active, axis=(0, 2)))
    iz = xp.flatnonzero(xp.any(active, axis=(0, 1)))

    if ix.size == 0 or iy.size == 0 or iz.size == 0:
        return None

    nx, ny, nz = U.shape[1:]

    i0 = max(0, _as_int(ix[0]) - padding)
    i1 = min(nx, _as_int(ix[-1]) + 1 + padding)

    j0 = max(0, _as_int(iy[0]) - padding)
    j1 = min(ny, _as_int(iy[-1]) + 1 + padding)

    k0 = max(0, _as_int(iz[0]) - padding)
    k1 = min(nz, _as_int(iz[-1]) + 1 + padding)

    return i0, i1, j0, j1, k0, k1

def spatial_operator(
    U: Any,
    energy: float,
    grid: Grid,
    inflow: BeamInflow,
    spatial_scheme: SpatialScheme,
    wave_speed_mode: WaveSpeeds,
    xp,
    active_domain: bool = False,
    active_relative_tolerance: float = 1.0e-14,
    boundary_state: Any | None = None,
):
    """Return the negative divergence of the M1 spatial flux."""

    reconstruction = _get_reconstruction(spatial_scheme)
    ng = reconstruction.GHOST_WIDTH

    resolved_boundary_state = (inflow.state(energy, xp) if boundary_state is None else boundary_state)

    if active_domain:
        box = find_active_box(
            U,
            resolved_boundary_state,
            padding=ng,
            relative_tolerance=active_relative_tolerance,
            xp=xp,
        )

        if box is None:
            return xp.zeros_like(U)

        i0, i1, j0, j1, k0, k1 = box

        nx, ny, nz = U.shape[1:]

        # The homogeneous beam examples rapidly span the full transverse
        # domain while remaining localized in z.  Avoid padding the entire
        # 200-cell z domain only to slice most of it away immediately after.
        # This produces exactly the same U_work as slicing the fully padded
        # state, but allocates/copies only the active z window plus its halo.
        full_transverse = (
            xp is np
            and i0 == 0
            and i1 == nx
            and j0 == 0
            and j1 == ny
        )

        if full_transverse and k0 == 0 and k1 + ng <= nz:
            U_truncated = U[:, :, :, :k1 + ng]
            U_ext = fill_ghost_cells(
                U_truncated,
                inflow,
                energy,
                ng,
                xp,
                boundary_state=resolved_boundary_state,
            )
            U_work = U_ext[:, :, :, :k1 + 2 * ng]

        elif full_transverse and k0 >= ng and k1 + ng <= nz:
            U_core = U[:, :, :, k0 - ng:k1 + ng]
            U_work = np.pad(
                U_core,
                (
                    (0, 0),
                    (ng, ng),
                    (ng, ng),
                    (0, 0),
                ),
                mode="edge",
            )

        else:
            U_ext = fill_ghost_cells(
                U,
                inflow,
                energy,
                ng,
                xp,
                boundary_state=resolved_boundary_state,
            )
            U_work = U_ext[
                :,
                i0:i1 + 2 * ng,
                j0:j1 + 2 * ng,
                k0:k1 + 2 * ng,
            ]

    else:
        U_ext = fill_ghost_cells(
            U,
            inflow,
            energy,
            ng,
            xp,
            boundary_state=resolved_boundary_state,
        )
        U_work = U_ext

        i0, i1 = 0, U.shape[1]
        j0, j1 = 0, U.shape[2]
        k0, k1 = 0, U.shape[3]

    # Interface reconstruction
    U_left_x, U_right_x = reconstruction.reconstruct_x(U_work, xp)
    U_left_y, U_right_y = reconstruction.reconstruct_y(U_work, xp)
    U_left_z, U_right_z = reconstruction.reconstruct_z(U_work, xp)

    # HLL interface fluxes
    F_x = hll_flux(
        U_left_x,
        U_right_x,
        axis=0,
        wave_speed_mode=wave_speed_mode,
        xp=xp,
    )

    F_y = hll_flux(
        U_left_y,
        U_right_y,
        axis=1,
        wave_speed_mode=wave_speed_mode,
        xp=xp,
    )

    F_z = hll_flux(
        U_left_z,
        U_right_z,
        axis=2,
        wave_speed_mode=wave_speed_mode,
        xp=xp,
    )

    # Negative flux divergence
    rhs_work = -(
        (F_x[:, 1:] - F_x[:, :-1]) / grid.dx
        + (F_y[:, :, 1:] - F_y[:, :, :-1]) / grid.dy
        + (F_z[:, :, :, 1:] - F_z[:, :, :, :-1]) / grid.dz
    )

    if not active_domain:
        return rhs_work

    rhs = xp.zeros_like(U)

    rhs[:, i0:i1, j0:j1, k0:k1] = rhs_work

    return rhs