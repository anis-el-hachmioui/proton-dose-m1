import numpy as np
from numba import njit, prange

from ..realizability import project_realizable


GHOST_WIDTH = 2

WENO_EPS = 1.0e-12
WENO_P = 2
REALIZABILITY_EPS = 1.0e-10


@njit(inline="always")
def _weno3_left_scalar(
    v0,
    v1,
    v2,
):
    """Scalar WENO3-Z left reconstruction for the CPU kernel."""

    # Second-order candidate polynomials
    p0 = (
        -0.5 * v0
        + 1.5 * v1
    )

    p1 = (
        0.5 * v1
        + 0.5 * v2
    )

    # Smoothness indicators
    beta0 = (v1 - v0) * (v1 - v0)
    beta1 = (v2 - v1) * (v2 - v1)

    tau3 = abs(beta1 - beta0)

    # WENO_P = 2
    r0 = tau3 / (beta0 + WENO_EPS)
    r1 = tau3 / (beta1 + WENO_EPS)

    alpha0 = (
        (1.0 / 3.0)
        * (1.0 + r0 * r0)
    )

    alpha1 = (
        (2.0 / 3.0)
        * (1.0 + r1 * r1)
    )

    alpha_sum = alpha0 + alpha1

    w0 = alpha0 / alpha_sum
    w1 = alpha1 / alpha_sum

    return (
        w0 * p0
        + w1 * p1
    )


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
)
def _reconstruct_x_numba(U_ext):
    """Numba WENO3 reconstruction at x-interfaces."""

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

                U_left[m, i, j, k] = (
                    _weno3_left_scalar(
                        U_ext[m, i, jj, kk],
                        U_ext[m, i + 1, jj, kk],
                        U_ext[m, i + 2, jj, kk],
                    )
                )

                U_right[m, i, j, k] = (
                    _weno3_left_scalar(
                        U_ext[m, i + 3, jj, kk],
                        U_ext[m, i + 2, jj, kk],
                        U_ext[m, i + 1, jj, kk],
                    )
                )

    for i in prange(nx + 1):
        for j in range(ny):
            for k in range(nz):
                _project_realizable_cell(
                    U_left,
                    i,
                    j,
                    k,
                )

                _project_realizable_cell(
                    U_right,
                    i,
                    j,
                    k,
                )

    return U_left, U_right


@njit(
    cache=True,
    parallel=True,
)
def _reconstruct_y_numba(U_ext):
    """Numba WENO3 reconstruction at y-interfaces."""

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

                U_left[m, i, j, k] = (
                    _weno3_left_scalar(
                        U_ext[m, ii, j, kk],
                        U_ext[m, ii, j + 1, kk],
                        U_ext[m, ii, j + 2, kk],
                    )
                )

                U_right[m, i, j, k] = (
                    _weno3_left_scalar(
                        U_ext[m, ii, j + 3, kk],
                        U_ext[m, ii, j + 2, kk],
                        U_ext[m, ii, j + 1, kk],
                    )
                )

    for i in prange(nx):
        for j in range(ny + 1):
            for k in range(nz):
                _project_realizable_cell(
                    U_left,
                    i,
                    j,
                    k,
                )

                _project_realizable_cell(
                    U_right,
                    i,
                    j,
                    k,
                )

    return U_left, U_right


@njit(
    cache=True,
    parallel=True,
)
def _reconstruct_z_numba(U_ext):
    """Numba WENO3 reconstruction at z-interfaces."""

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
                U_left[m, i, j, k] = (
                    _weno3_left_scalar(
                        U_ext[m, ii, jj, k],
                        U_ext[m, ii, jj, k + 1],
                        U_ext[m, ii, jj, k + 2],
                    )
                )

                U_right[m, i, j, k] = (
                    _weno3_left_scalar(
                        U_ext[m, ii, jj, k + 3],
                        U_ext[m, ii, jj, k + 2],
                        U_ext[m, ii, jj, k + 1],
                    )
                )

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz + 1):
                _project_realizable_cell(
                    U_left,
                    i,
                    j,
                    k,
                )

                _project_realizable_cell(
                    U_right,
                    i,
                    j,
                    k,
                )

    return U_left, U_right


def _weights(
    beta0,
    beta1,
    xp,
):
    """Return nonlinear WENO3-Z weights."""

    tau3 = xp.abs(beta1 - beta0)

    alpha0 = (
        (1.0 / 3.0)
        * (
            1.0
            + (
                tau3
                / (beta0 + WENO_EPS)
            ) ** WENO_P
        )
    )

    alpha1 = (
        (2.0 / 3.0)
        * (
            1.0
            + (
                tau3
                / (beta1 + WENO_EPS)
            ) ** WENO_P
        )
    )

    alpha_sum = alpha0 + alpha1

    return (
        alpha0 / alpha_sum,
        alpha1 / alpha_sum,
    )


def _reconstruct_left(
    v0,
    v1,
    v2,
    xp,
):
    """Reconstruct the left interface state with WENO3-Z."""

    p0 = (
        -0.5 * v0
        + 1.5 * v1
    )

    p1 = (
        0.5 * v1
        + 0.5 * v2
    )

    beta0 = (v1 - v0) ** 2
    beta1 = (v2 - v1) ** 2

    w0, w1 = _weights(
        beta0,
        beta1,
        xp,
    )

    return (
        w0 * p0
        + w1 * p1
    )


def _reconstruct_x_vectorized(U_ext, xp):
    """Backend-generic WENO3 reconstruction at x-interfaces."""

    ng = GHOST_WIDTH

    nx = U_ext.shape[1] - 2 * ng
    ny = U_ext.shape[2] - 2 * ng
    nz = U_ext.shape[3] - 2 * ng

    v = U_ext[
        :,
        :,
        ng:ng + ny,
        ng:ng + nz,
    ]

    U_left = _reconstruct_left(
        v[:, 0:nx + 1],
        v[:, 1:nx + 2],
        v[:, 2:nx + 3],
        xp,
    )

    U_right = _reconstruct_left(
        v[:, 3:nx + 4],
        v[:, 2:nx + 3],
        v[:, 1:nx + 2],
        xp,
    )

    return (
        project_realizable(U_left, xp),
        project_realizable(U_right, xp),
    )


def _reconstruct_y_vectorized(U_ext, xp):
    """Backend-generic WENO3 reconstruction at y-interfaces."""

    ng = GHOST_WIDTH

    nx = U_ext.shape[1] - 2 * ng
    ny = U_ext.shape[2] - 2 * ng
    nz = U_ext.shape[3] - 2 * ng

    v = U_ext[
        :,
        ng:ng + nx,
        :,
        ng:ng + nz,
    ]

    U_left = _reconstruct_left(
        v[:, :, 0:ny + 1],
        v[:, :, 1:ny + 2],
        v[:, :, 2:ny + 3],
        xp,
    )

    U_right = _reconstruct_left(
        v[:, :, 3:ny + 4],
        v[:, :, 2:ny + 3],
        v[:, :, 1:ny + 2],
        xp,
    )

    return (
        project_realizable(U_left, xp),
        project_realizable(U_right, xp),
    )


def _reconstruct_z_vectorized(U_ext, xp):
    """Backend-generic WENO3 reconstruction at z-interfaces."""

    ng = GHOST_WIDTH

    nx = U_ext.shape[1] - 2 * ng
    ny = U_ext.shape[2] - 2 * ng
    nz = U_ext.shape[3] - 2 * ng

    v = U_ext[
        :,
        ng:ng + nx,
        ng:ng + ny,
        :,
    ]

    U_left = _reconstruct_left(
        v[:, :, :, 0:nz + 1],
        v[:, :, :, 1:nz + 2],
        v[:, :, :, 2:nz + 3],
        xp,
    )

    U_right = _reconstruct_left(
        v[:, :, :, 3:nz + 4],
        v[:, :, :, 2:nz + 3],
        v[:, :, :, 1:nz + 2],
        xp,
    )

    return (
        project_realizable(U_left, xp),
        project_realizable(U_right, xp),
    )


def reconstruct_x(U_ext, xp):
    """Reconstruct WENO3 states at x-interfaces."""

    if xp is np:
        return _reconstruct_x_numba(U_ext)

    return _reconstruct_x_vectorized(
        U_ext,
        xp,
    )


def reconstruct_y(U_ext, xp):
    """Reconstruct WENO3 states at y-interfaces."""

    if xp is np:
        return _reconstruct_y_numba(U_ext)

    return _reconstruct_y_vectorized(
        U_ext,
        xp,
    )


def reconstruct_z(U_ext, xp):
    """Reconstruct WENO3 states at z-interfaces."""

    if xp is np:
        return _reconstruct_z_numba(U_ext)

    return _reconstruct_z_vectorized(
        U_ext,
        xp,
    )
