import numpy as np
from numba import njit, prange

from ..realizability import project_realizable


GHOST_WIDTH = 2

REALIZABILITY_EPS = 1.0e-10


# ============================================================
# Numba CPU kernels
#
# Same structure and compilation settings as the WENO-Z CPU
# kernels (parallel over component x first index, fastmath, fused
# realizability projection), so that runtimes of the reconstruction
# schemes are directly comparable. The arithmetic is identical to
# the vectorized implementation below.
# ============================================================


@njit(inline="always")
def _minmod_scalar(a, b):
    """Scalar two-argument minmod limiter."""

    if a * b > 0.0:
        if a > 0.0:
            sign_a = 1.0
        else:
            sign_a = -1.0

        return sign_a * min(abs(a), abs(b))

    return 0.0


@njit(inline="always")
def _muscl_faces_scalar(vm, v0, vp):
    """Return (minus-face, plus-face) MUSCL values of the centre cell."""

    slope = _minmod_scalar(v0 - vm, vp - v0)

    return v0 - 0.5 * slope, v0 + 0.5 * slope


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
    """Numba MUSCL reconstruction at x-interfaces."""

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

        # Interface i lies between extended cells c = i + 1
        # (left state) and c = i + 2 (right state).
        for j in range(ny):
            jj = j + ng

            for k in range(nz):
                kk = k + ng

                _, plus_face = _muscl_faces_scalar(
                    U_ext[m, i, jj, kk],
                    U_ext[m, i + 1, jj, kk],
                    U_ext[m, i + 2, jj, kk],
                )

                minus_face, _ = _muscl_faces_scalar(
                    U_ext[m, i + 1, jj, kk],
                    U_ext[m, i + 2, jj, kk],
                    U_ext[m, i + 3, jj, kk],
                )

                U_left[m, i, j, k] = plus_face
                U_right[m, i, j, k] = minus_face

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
    """Numba MUSCL reconstruction at y-interfaces."""

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

                _, plus_face = _muscl_faces_scalar(
                    U_ext[m, ii, j, kk],
                    U_ext[m, ii, j + 1, kk],
                    U_ext[m, ii, j + 2, kk],
                )

                minus_face, _ = _muscl_faces_scalar(
                    U_ext[m, ii, j + 1, kk],
                    U_ext[m, ii, j + 2, kk],
                    U_ext[m, ii, j + 3, kk],
                )

                U_left[m, i, j, k] = plus_face
                U_right[m, i, j, k] = minus_face

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
    """Numba MUSCL reconstruction at z-interfaces."""

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
                _, plus_face = _muscl_faces_scalar(
                    U_ext[m, ii, jj, k],
                    U_ext[m, ii, jj, k + 1],
                    U_ext[m, ii, jj, k + 2],
                )

                minus_face, _ = _muscl_faces_scalar(
                    U_ext[m, ii, jj, k + 1],
                    U_ext[m, ii, jj, k + 2],
                    U_ext[m, ii, jj, k + 3],
                )

                U_left[m, i, j, k] = plus_face
                U_right[m, i, j, k] = minus_face

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


def _minmod(a, b, xp):
    """Componentwise two-argument minmod slope limiter."""

    same_sign = a * b > 0.0

    return xp.where(
        same_sign,
        xp.sign(a) * xp.minimum(xp.abs(a), xp.abs(b)),
        0.0,
    )


def _limited_slopes(v, axis, xp):
    """Return minmod-limited slopes for cells with both neighbours present."""

    slices_previous = [slice(None)] * v.ndim
    slices_center = [slice(None)] * v.ndim
    slices_next = [slice(None)] * v.ndim

    slices_previous[axis] = slice(0, -2)
    slices_center[axis] = slice(1, -1)
    slices_next[axis] = slice(2, None)

    previous = v[tuple(slices_previous)]
    center = v[tuple(slices_center)]
    next_value = v[tuple(slices_next)]

    backward = center - previous
    forward = next_value - center

    return _minmod(backward, forward, xp)


def reconstruct_x(U_ext, xp):
    """Reconstruct second-order MUSCL states at x-interfaces."""

    if xp is np:
        return _reconstruct_x_numba(U_ext)

    return _reconstruct_x_vectorized(U_ext, xp)


def _reconstruct_x_vectorized(U_ext, xp):
    """Vectorized MUSCL reconstruction at x-interfaces."""

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

    slopes = _limited_slopes(v, axis=1, xp=xp)

    U_left = (
        v[:, 1:nx + 2]
        + 0.5 * slopes[:, 0:nx + 1]
    )

    U_right = (
        v[:, 2:nx + 3]
        - 0.5 * slopes[:, 1:nx + 2]
    )

    return (
        project_realizable(U_left, xp),
        project_realizable(U_right, xp),
    )


def reconstruct_y(U_ext, xp):
    """Reconstruct second-order MUSCL states at y-interfaces."""

    if xp is np:
        return _reconstruct_y_numba(U_ext)

    return _reconstruct_y_vectorized(U_ext, xp)


def _reconstruct_y_vectorized(U_ext, xp):
    """Vectorized MUSCL reconstruction at y-interfaces."""

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

    slopes = _limited_slopes(v, axis=2, xp=xp)

    U_left = (
        v[:, :, 1:ny + 2]
        + 0.5 * slopes[:, :, 0:ny + 1]
    )

    U_right = (
        v[:, :, 2:ny + 3]
        - 0.5 * slopes[:, :, 1:ny + 2]
    )

    return (
        project_realizable(U_left, xp),
        project_realizable(U_right, xp),
    )


def reconstruct_z(U_ext, xp):
    """Reconstruct second-order MUSCL states at z-interfaces."""

    if xp is np:
        return _reconstruct_z_numba(U_ext)

    return _reconstruct_z_vectorized(U_ext, xp)


def _reconstruct_z_vectorized(U_ext, xp):
    """Vectorized MUSCL reconstruction at z-interfaces."""

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

    slopes = _limited_slopes(v, axis=3, xp=xp)

    U_left = (
        v[:, :, :, 1:nz + 2]
        + 0.5 * slopes[:, :, :, 0:nz + 1]
    )

    U_right = (
        v[:, :, :, 2:nz + 3]
        - 0.5 * slopes[:, :, :, 1:nz + 2]
    )

    return (
        project_realizable(U_left, xp),
        project_realizable(U_right, xp),
    )
