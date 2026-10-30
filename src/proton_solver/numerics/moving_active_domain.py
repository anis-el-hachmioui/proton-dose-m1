"""Compact moving active-domain helpers.

The global geometry remains unchanged, but the transported M1 state is kept on
only a small rectangular box surrounding the significant proton population.
The box may move and resize after every energy step.  A guard region separates
the significant state from the artificial local boundaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..physics.materials import MaterialMap
from .boundary import BeamInflow
from .grid import Grid


Box = tuple[int, int, int, int, int, int]


def _as_int(value) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _as_float(value) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def initial_box(
    inflow: BeamInflow,
    grid: Grid,
    guard: int,
    xp,
) -> Box:
    """Return a compact entrance box around the non-zero beam support."""

    active = inflow.spatial > 0.0
    ix = xp.flatnonzero(xp.any(active, axis=1))
    iy = xp.flatnonzero(xp.any(active, axis=0))

    if ix.size == 0 or iy.size == 0:
        raise ValueError("Beam inflow has empty transverse support.")

    i0 = max(0, _as_int(ix[0]) - guard)
    i1 = min(grid.nx, _as_int(ix[-1]) + 1 + guard)
    j0 = max(0, _as_int(iy[0]) - guard)
    j1 = min(grid.ny, _as_int(iy[-1]) + 1 + guard)

    # One physical entrance cell plus enough halo for the RK/WENO domain of
    # dependence.  The box will grow/move in z as soon as transport begins.
    k0 = 0
    k1 = min(grid.nz, 1 + guard)

    return i0, i1, j0, j1, k0, k1


def make_local_problem(
    full_grid: Grid,
    full_inflow: BeamInflow,
    full_material_map: MaterialMap,
    box: Box,
    xp,
):
    """Create lightweight grid/inflow/material views for one active box."""

    i0, i1, j0, j1, k0, k1 = box

    grid = Grid(
        x=full_grid.x[i0:i1],
        y=full_grid.y[j0:j1],
        z=full_grid.z[k0:k1],
        dx=full_grid.dx,
        dy=full_grid.dy,
        dz=full_grid.dz,
        shape=(i1 - i0, j1 - j0, k1 - k0),
    )

    # The physical beam enters only at global z=0.  Once the moving box has
    # left the entrance plane, its artificial z-min boundary has zero inflow.
    if k0 == 0:
        spatial = full_inflow.spatial[i0:i1, j0:j1]
    else:
        spatial = xp.zeros(
            (i1 - i0, j1 - j0),
            dtype=full_inflow.spatial.dtype,
        )

    inflow = BeamInflow(
        beam=full_inflow.beam,
        spatial=spatial,
    )

    material_map = MaterialMap(
        ids=full_material_map.ids[i0:i1, j0:j1, k0:k1],
        materials=full_material_map.materials,
    )

    return grid, inflow, material_map


def significant_box(
    U: Any,
    current_box: Box,
    local_inflow: BeamInflow,
    energy: float,
    relative_tolerance: float,
    xp,
) -> Box | None:
    """Return the global bounding box of significant current flux."""

    i0, i1, j0, j1, k0, k1 = current_box
    phi = U[0]

    phi_in = None
    if k0 == 0:
        phi_in = local_inflow.state(energy, xp)[0]

    scale = _as_float(xp.max(phi))
    if phi_in is not None:
        scale = max(scale, _as_float(xp.max(phi_in)))

    if scale == 0.0:
        return None

    threshold = relative_tolerance * scale
    active = phi > threshold

    if phi_in is not None:
        active[:, :, 0] |= phi_in > threshold

    ix = xp.flatnonzero(xp.any(active, axis=(1, 2)))
    iy = xp.flatnonzero(xp.any(active, axis=(0, 2)))
    iz = xp.flatnonzero(xp.any(active, axis=(0, 1)))

    if ix.size == 0 or iy.size == 0 or iz.size == 0:
        return None

    return (
        i0 + _as_int(ix[0]),
        i0 + _as_int(ix[-1]) + 1,
        j0 + _as_int(iy[0]),
        j0 + _as_int(iy[-1]) + 1,
        k0 + _as_int(iz[0]),
        k0 + _as_int(iz[-1]) + 1,
    )


def desired_box(
    U: Any,
    current_box: Box,
    local_inflow: BeamInflow,
    full_inflow: BeamInflow,
    full_grid: Grid,
    energy: float,
    relative_tolerance: float,
    guard: int,
    xp,
) -> Box:
    """Return the next guarded moving box in global cell coordinates."""

    active = significant_box(
        U,
        current_box,
        local_inflow,
        energy,
        relative_tolerance,
        xp,
    )

    if active is None:
        return initial_box(
            full_inflow,
            full_grid,
            guard,
            xp,
        )

    i0, i1, j0, j1, k0, k1 = active

    return (
        max(0, i0 - guard),
        min(full_grid.nx, i1 + guard),
        max(0, j0 - guard),
        min(full_grid.ny, j1 + guard),
        max(0, k0 - guard),
        min(full_grid.nz, k1 + guard),
    )


def rebox(U: Any, old_box: Box, new_box: Box, xp):
    """Move/resize a compact state while preserving the global overlap."""

    if old_box == new_box:
        return U

    ni0, ni1, nj0, nj1, nk0, nk1 = new_box
    oi0, oi1, oj0, oj1, ok0, ok1 = old_box

    U_new = xp.zeros(
        (
            4,
            ni1 - ni0,
            nj1 - nj0,
            nk1 - nk0,
        ),
        dtype=U.dtype,
    )

    gi0, gi1 = max(oi0, ni0), min(oi1, ni1)
    gj0, gj1 = max(oj0, nj0), min(oj1, nj1)
    gk0, gk1 = max(ok0, nk0), min(ok1, nk1)

    if gi1 > gi0 and gj1 > gj0 and gk1 > gk0:
        U_new[
            :,
            gi0 - ni0:gi1 - ni0,
            gj0 - nj0:gj1 - nj0,
            gk0 - nk0:gk1 - nk0,
        ] = U[
            :,
            gi0 - oi0:gi1 - oi0,
            gj0 - oj0:gj1 - oj0,
            gk0 - ok0:gk1 - ok0,
        ]

    return U_new


def embed_state(U: Any, box: Box, full_shape: tuple[int, int, int], xp):
    """Materialize a compact state on the full grid (snapshots/final output)."""

    full = xp.zeros(
        (4, *full_shape),
        dtype=U.dtype,
    )
    i0, i1, j0, j1, k0, k1 = box
    full[:, i0:i1, j0:j1, k0:k1] = U
    return full


def box_slices(box: Box):
    i0, i1, j0, j1, k0, k1 = box
    return (
        slice(i0, i1),
        slice(j0, j1),
        slice(k0, k1),
    )
