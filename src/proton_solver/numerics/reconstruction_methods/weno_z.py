import numpy as np
from numba import njit, prange

from ..realizability import project_realizable

GHOST_WIDTH = 3

WENO_EPS = 1.0e-12
WENO_P = 2
REALIZABILITY_EPS = 1.0e-10

@njit(inline="always")
def _weno_z_left_scalar(
    v0,
    v1,
    v2,
    v3,
    v4,
):
    """Scalar WENO-Z5 left reconstruction for the CPU kernel."""

    p0 = (
        v0 / 3.0
        - 7.0 * v1 / 6.0
        + 11.0 * v2 / 6.0
    )

    p1 = (
        -v1 / 6.0
        + 5.0 * v2 / 6.0
        + v3 / 3.0
    )

    p2 = (
        v2 / 3.0
        + 5.0 * v3 / 6.0
        - v4 / 6.0
    )

    beta0 = (
        13.0 / 12.0
        * (v0 - 2.0 * v1 + v2) ** 2
        + 0.25
        * (v0 - 4.0 * v1 + 3.0 * v2) ** 2
    )

    beta1 = (
        13.0 / 12.0
        * (v1 - 2.0 * v2 + v3) ** 2
        + 0.25
        * (v1 - v3) ** 2
    )

    beta2 = (
        13.0 / 12.0
        * (v2 - 2.0 * v3 + v4) ** 2
        + 0.25
        * (3.0 * v2 - 4.0 * v3 + v4) ** 2
    )

    tau5 = abs(beta2 - beta0)

    # WENO_P is fixed at 2 in V5.
    # Multiplication is cheaper than generic ** p.
    r0 = tau5 / (beta0 + WENO_EPS)
    r1 = tau5 / (beta1 + WENO_EPS)
    r2 = tau5 / (beta2 + WENO_EPS)

    alpha0 = 0.1 * (1.0 + r0 * r0)
    alpha1 = 0.6 * (1.0 + r1 * r1)
    alpha2 = 0.3 * (1.0 + r2 * r2)

    alpha_sum = (
        alpha0
        + alpha1
        + alpha2
    )

    w0 = alpha0 / alpha_sum
    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum

    return (
        w0 * p0
        + w1 * p1
        + w2 * p2
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
        # This branch implies J_norm > 0.
        # Divide by J_norm exactly as V5 project_realizable().
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
    """Numba WENO-Z reconstruction at x-interfaces."""

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
                    _weno_z_left_scalar(
                        U_ext[m, i, jj, kk],
                        U_ext[m, i + 1, jj, kk],
                        U_ext[m, i + 2, jj, kk],
                        U_ext[m, i + 3, jj, kk],
                        U_ext[m, i + 4, jj, kk],
                    )
                )

                U_right[m, i, j, k] = (
                    _weno_z_left_scalar(
                        U_ext[m, i + 5, jj, kk],
                        U_ext[m, i + 4, jj, kk],
                        U_ext[m, i + 3, jj, kk],
                        U_ext[m, i + 2, jj, kk],
                        U_ext[m, i + 1, jj, kk],
                    )
                )

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
    """Numba WENO-Z reconstruction at y-interfaces."""

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
                    _weno_z_left_scalar(
                        U_ext[m, ii, j, kk],
                        U_ext[m, ii, j + 1, kk],
                        U_ext[m, ii, j + 2, kk],
                        U_ext[m, ii, j + 3, kk],
                        U_ext[m, ii, j + 4, kk],
                    )
                )

                U_right[m, i, j, k] = (
                    _weno_z_left_scalar(
                        U_ext[m, ii, j + 5, kk],
                        U_ext[m, ii, j + 4, kk],
                        U_ext[m, ii, j + 3, kk],
                        U_ext[m, ii, j + 2, kk],
                        U_ext[m, ii, j + 1, kk],
                    )
                )

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
    """Numba WENO-Z reconstruction at z-interfaces."""

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
                    _weno_z_left_scalar(
                        U_ext[m, ii, jj, k],
                        U_ext[m, ii, jj, k + 1],
                        U_ext[m, ii, jj, k + 2],
                        U_ext[m, ii, jj, k + 3],
                        U_ext[m, ii, jj, k + 4],
                    )
                )

                U_right[m, i, j, k] = (
                    _weno_z_left_scalar(
                        U_ext[m, ii, jj, k + 5],
                        U_ext[m, ii, jj, k + 4],
                        U_ext[m, ii, jj, k + 3],
                        U_ext[m, ii, jj, k + 2],
                        U_ext[m, ii, jj, k + 1],
                    )
                )

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

def _weights(beta0, beta1, beta2, xp):
    """Return nonlinear WENO-Z weights."""

    d0 = 0.1
    d1 = 0.6
    d2 = 0.3

    tau5 = xp.abs(beta2 - beta0)

    alpha0 = d0 * (
        1.0
        + (tau5 / (beta0 + WENO_EPS)) ** WENO_P
    )

    alpha1 = d1 * (
        1.0
        + (tau5 / (beta1 + WENO_EPS)) ** WENO_P
    )

    alpha2 = d2 * (
        1.0
        + (tau5 / (beta2 + WENO_EPS)) ** WENO_P
    )

    alpha_sum = alpha0 + alpha1 + alpha2

    return (
        alpha0 / alpha_sum,
        alpha1 / alpha_sum,
        alpha2 / alpha_sum,
    )


def _reconstruct_left(v0, v1, v2, v3, v4, xp):
    """Reconstruct the left state of an interface with WENO-Z5."""

    # Third-order candidate polynomials
    p0 = (
        v0 / 3.0
        - 7.0 * v1 / 6.0
        + 11.0 * v2 / 6.0
    )

    p1 = (
        -v1 / 6.0
        + 5.0 * v2 / 6.0
        + v3 / 3.0
    )

    p2 = (
        v2 / 3.0
        + 5.0 * v3 / 6.0
        - v4 / 6.0
    )

    # Jiang-Shu smoothness indicators
    beta0 = (
        13.0 / 12.0
        * (v0 - 2.0 * v1 + v2) ** 2
        + 0.25
        * (v0 - 4.0 * v1 + 3.0 * v2) ** 2
    )

    beta1 = (
        13.0 / 12.0
        * (v1 - 2.0 * v2 + v3) ** 2
        + 0.25
        * (v1 - v3) ** 2
    )

    beta2 = (
        13.0 / 12.0
        * (v2 - 2.0 * v3 + v4) ** 2
        + 0.25
        * (3.0 * v2 - 4.0 * v3 + v4) ** 2
    )

    w0, w1, w2 = _weights(
        beta0,
        beta1,
        beta2,
        xp,
    )

    return (
        w0 * p0
        + w1 * p1
        + w2 * p2
    )


def _reconstruct_x_vectorized(U_ext, xp):
    """Reconstruct WENO-Z states at x-interfaces."""

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
        v[:, 3:nx + 4],
        v[:, 4:nx + 5],
        xp,
    )

    U_right = _reconstruct_left(
        v[:, 5:nx + 6],
        v[:, 4:nx + 5],
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
    """Reconstruct WENO-Z states at y-interfaces."""

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
        v[:, :, 3:ny + 4],
        v[:, :, 4:ny + 5],
        xp,
    )

    U_right = _reconstruct_left(
        v[:, :, 5:ny + 6],
        v[:, :, 4:ny + 5],
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
    """Reconstruct WENO-Z states at z-interfaces."""

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
        v[:, :, :, 3:nz + 4],
        v[:, :, :, 4:nz + 5],
        xp,
    )

    U_right = _reconstruct_left(
        v[:, :, :, 5:nz + 6],
        v[:, :, :, 4:nz + 5],
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
    """Reconstruct WENO-Z states at x-interfaces."""

    if xp is np:
        return _reconstruct_x_numba(U_ext)

    return _reconstruct_x_vectorized(U_ext, xp)


def reconstruct_y(U_ext, xp):
    """Reconstruct WENO-Z states at y-interfaces."""

    if xp is np:
        return _reconstruct_y_numba(U_ext)

    return _reconstruct_y_vectorized(U_ext, xp)


def reconstruct_z(U_ext, xp):
    """Reconstruct WENO-Z states at z-interfaces."""

    if xp is np:
        return _reconstruct_z_numba(U_ext)

    return _reconstruct_z_vectorized(U_ext, xp)