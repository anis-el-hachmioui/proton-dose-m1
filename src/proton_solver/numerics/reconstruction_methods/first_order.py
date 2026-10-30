import numpy as np
from numba import njit, prange

from ..realizability import project_realizable


GHOST_WIDTH = 1

REALIZABILITY_EPS = 1.0e-10


# ============================================================
# Numba CPU kernels
#
# Same structure and compilation settings as the WENO-Z and MUSCL
# CPU kernels (parallel, fastmath, fused realizability projection).
# The interface states are the adjacent cell averages.
# ============================================================


@njit(inline="always")
def _project_realizable_cell(
    U,
    i,
    j,
    k,
):
    """Project one reconstructed state onto the V5 realizable set."""

    phi = U[0, i, j, k]

    if phi < 0.0:
        phi = 0.0

    Jx = U[1, i, j, k]
    Jy = U[2, i, j, k]
    Jz = U[3, i, j, k]

    J_norm = np.sqrt(
        Jx * Jx
        + Jy * Jy
        + Jz * Jz
    )

    J_max = (
        (1.0 - REALIZABILITY_EPS)
        * phi
    )

    if J_norm > J_max:
        scale = J_max / J_norm
    else:
        scale = 1.0

    U[0, i, j, k] = phi
    U[1, i, j, k] = Jx * scale
    U[2, i, j, k] = Jy * scale
    U[3, i, j, k] = Jz * scale


@njit(
    cache=True,
    parallel=True,
    fastmath=True,
)
def _reconstruct_x_numba(U_ext):
    """Numba first-order reconstruction at x-interfaces."""

    ng = GHOST_WIDTH

    nx = U_ext.shape[1] - 2 * ng
    ny = U_ext.shape[2] - 2 * ng
    nz = U_ext.shape[3] - 2 * ng

    U_left = np.empty(
        (4, nx + 1, ny, nz),
        dtype=U_ext.dtype,
    )

    U_right = np.empty_like(U_left)

    for mi in prange(4 * (nx + 1)):
        m = mi // (nx + 1)
        i = mi - m * (nx + 1)

        for j in range(ny):
            jj = j + ng

            for k in range(nz):
                kk = k + ng

                U_left[m, i, j, k] = U_ext[m, i + ng - 1, jj, kk]
                U_right[m, i, j, k] = U_ext[m, i + ng, jj, kk]

    for i in prange(nx + 1):
        for j in range(ny):
            for k in range(nz):
                _project_realizable_cell(
                    U_left, i, j, k
                )

                _project_realizable_cell(
                    U_right, i, j, k
                )

    return U_left, U_right


@njit(
    cache=True,
    parallel=True,
    fastmath=True,
)
def _reconstruct_y_numba(U_ext):
    """Numba first-order reconstruction at y-interfaces."""

    ng = GHOST_WIDTH

    nx = U_ext.shape[1] - 2 * ng
    ny = U_ext.shape[2] - 2 * ng
    nz = U_ext.shape[3] - 2 * ng

    U_left = np.empty(
        (4, nx, ny + 1, nz),
        dtype=U_ext.dtype,
    )

    U_right = np.empty_like(U_left)

    for mi in prange(4 * nx):
        m = mi // nx
        i = mi - m * nx

        ii = i + ng

        for j in range(ny + 1):
            for k in range(nz):
                kk = k + ng

                U_left[m, i, j, k] = U_ext[m, ii, j + ng - 1, kk]
                U_right[m, i, j, k] = U_ext[m, ii, j + ng, kk]

    for i in prange(nx):
        for j in range(ny + 1):
            for k in range(nz):
                _project_realizable_cell(
                    U_left, i, j, k
                )

                _project_realizable_cell(
                    U_right, i, j, k
                )

    return U_left, U_right


@njit(
    cache=True,
    parallel=True,
    fastmath=True,
)
def _reconstruct_z_numba(U_ext):
    """Numba first-order reconstruction at z-interfaces."""

    ng = GHOST_WIDTH

    nx = U_ext.shape[1] - 2 * ng
    ny = U_ext.shape[2] - 2 * ng
    nz = U_ext.shape[3] - 2 * ng

    U_left = np.empty(
        (4, nx, ny, nz + 1),
        dtype=U_ext.dtype,
    )

    U_right = np.empty_like(U_left)

    for mi in prange(4 * nx):
        m = mi // nx
        i = mi - m * nx

        ii = i + ng

        for j in range(ny):
            jj = j + ng

            for k in range(nz + 1):
                U_left[m, i, j, k] = U_ext[m, ii, jj, k + ng - 1]
                U_right[m, i, j, k] = U_ext[m, ii, jj, k + ng]

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz + 1):
                _project_realizable_cell(
                    U_left, i, j, k
                )

                _project_realizable_cell(
                    U_right, i, j, k
                )

    return U_left, U_right


# ============================================================
# Vectorized implementation (GPU / generic array backends)
# ============================================================


def reconstruct_x(U_ext, xp):
    """Reconstruct first-order states at x-interfaces."""

    if xp is np:
        return _reconstruct_x_numba(U_ext)

    return _reconstruct_x_vectorized(U_ext, xp)


def _reconstruct_x_vectorized(U_ext, xp):
    """Vectorized first-order reconstruction at x-interfaces."""

    ng = GHOST_WIDTH
    nx = U_ext.shape[1] - 2 * ng
    ny = U_ext.shape[2] - 2 * ng
    nz = U_ext.shape[3] - 2 * ng

    U_left = U_ext[
        :,
        ng - 1:ng + nx,
        ng:ng + ny,
        ng:ng + nz,
    ]

    U_right = U_ext[
        :,
        ng:ng + nx + 1,
        ng:ng + ny,
        ng:ng + nz,
    ]

    return (
        project_realizable(U_left, xp),
        project_realizable(U_right, xp),
    )


def reconstruct_y(U_ext, xp):
    """Reconstruct first-order states at y-interfaces."""

    if xp is np:
        return _reconstruct_y_numba(U_ext)

    return _reconstruct_y_vectorized(U_ext, xp)


def _reconstruct_y_vectorized(U_ext, xp):
    """Vectorized first-order reconstruction at y-interfaces."""

    ng = GHOST_WIDTH
    nx = U_ext.shape[1] - 2 * ng
    ny = U_ext.shape[2] - 2 * ng
    nz = U_ext.shape[3] - 2 * ng

    U_left = U_ext[
        :,
        ng:ng + nx,
        ng - 1:ng + ny,
        ng:ng + nz,
    ]

    U_right = U_ext[
        :,
        ng:ng + nx,
        ng:ng + ny + 1,
        ng:ng + nz,
    ]

    return (
        project_realizable(U_left, xp),
        project_realizable(U_right, xp),
    )


def reconstruct_z(U_ext, xp):
    """Reconstruct first-order states at z-interfaces."""

    if xp is np:
        return _reconstruct_z_numba(U_ext)

    return _reconstruct_z_vectorized(U_ext, xp)


def _reconstruct_z_vectorized(U_ext, xp):
    """Vectorized first-order reconstruction at z-interfaces."""

    ng = GHOST_WIDTH
    nx = U_ext.shape[1] - 2 * ng
    ny = U_ext.shape[2] - 2 * ng
    nz = U_ext.shape[3] - 2 * ng

    U_left = U_ext[
        :,
        ng:ng + nx,
        ng:ng + ny,
        ng - 1:ng + nz,
    ]

    U_right = U_ext[
        :,
        ng:ng + nx,
        ng:ng + ny,
        ng:ng + nz + 1,
    ]

    return (
        project_realizable(U_left, xp),
        project_realizable(U_right, xp),
    )