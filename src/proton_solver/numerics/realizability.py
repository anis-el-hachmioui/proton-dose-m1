from typing import Any

import numpy as np
from numba import njit, prange


@njit(
    cache=True,
    parallel=True,
)
def _project_realizable_numba(
    U,
    eps,
):
    """Parallel CPU projection onto the M1 realizable set."""

    U_projected = U.copy()

    phi = U_projected[0].ravel()
    Jx = U_projected[1].ravel()
    Jy = U_projected[2].ravel()
    Jz = U_projected[3].ravel()

    for i in prange(phi.size):
        phi_i = phi[i]

        if phi_i < 0.0:
            phi_i = 0.0

        Jx_i = Jx[i]
        Jy_i = Jy[i]
        Jz_i = Jz[i]

        J_norm = np.sqrt(
            Jx_i * Jx_i
            + Jy_i * Jy_i
            + Jz_i * Jz_i
        )

        J_max = (1.0 - eps) * phi_i

        if J_norm > J_max:
            # This branch necessarily has J_norm > 0
            # unless J_max is negative, which cannot occur
            # after clipping phi.
            scale = J_max / J_norm
        else:
            scale = 1.0

        phi[i] = phi_i
        Jx[i] = Jx_i * scale
        Jy[i] = Jy_i * scale
        Jz[i] = Jz_i * scale

    return U_projected

def _project_realizable_vectorized(
    U: Any,
    xp,
    eps: float = 1.0e-10,
):
    """
    Project an M1 state onto the realizable set.

    Enforces
        phi >= 0
        ||J|| <= (1 - eps) * phi
    while preserving the direction of J.
    """

    U_projected = U.copy()

    phi = xp.maximum(U_projected[0], 0.0)

    Jx = U_projected[1]
    Jy = U_projected[2]
    Jz = U_projected[3]

    J_norm = xp.sqrt(
        Jx**2
        + Jy**2
        + Jz**2
    )

    J_max = (1.0 - eps) * phi

    denominator = xp.where(
        J_norm > 0.0,
        J_norm,
        1.0,
    )

    scale = xp.where(
        J_norm > J_max,
        J_max / denominator,
        1.0,
    )

    U_projected[0] = phi
    U_projected[1] = Jx * scale
    U_projected[2] = Jy * scale
    U_projected[3] = Jz * scale

    return U_projected


def project_realizable(
    U: Any,
    xp,
    eps: float = 1.0e-10,
):
    """
    Project an M1 state onto the realizable set.

    Enforces
        phi >= 0
        ||J|| <= (1 - eps) * phi
    while preserving the direction of J.
    """

    if xp is np:
        return _project_realizable_numba(
            U,
            eps,
        )

    return _project_realizable_vectorized(
        U,
        xp,
        eps,
    )
@njit(
    cache=True,
    parallel=True,
)
def _project_realizable_inplace_numba(
    U,
    eps,
):
    """In-place CPU projection for freshly allocated RK stage states."""

    phi = U[0].ravel()
    Jx = U[1].ravel()
    Jy = U[2].ravel()
    Jz = U[3].ravel()

    for i in prange(phi.size):
        phi_i = phi[i]

        if phi_i < 0.0:
            phi_i = 0.0

        Jx_i = Jx[i]
        Jy_i = Jy[i]
        Jz_i = Jz[i]

        J_norm = np.sqrt(
            Jx_i * Jx_i
            + Jy_i * Jy_i
            + Jz_i * Jz_i
        )

        J_max = (1.0 - eps) * phi_i

        if J_norm > J_max:
            scale = J_max / J_norm
            Jx_i *= scale
            Jy_i *= scale
            Jz_i *= scale

        phi[i] = phi_i
        Jx[i] = Jx_i
        Jy[i] = Jy_i
        Jz[i] = Jz_i

    return U


def project_realizable_inplace(
    U: Any,
    xp,
    eps: float = 1.0e-10,
):
    """Project a writable state in place and return the same array."""

    if xp is np:
        return _project_realizable_inplace_numba(
            U,
            eps,
        )

    # GPU/backend-generic path.  U is assumed writable.
    phi = xp.maximum(U[0], 0.0)
    Jx = U[1]
    Jy = U[2]
    Jz = U[3]

    J_norm = xp.sqrt(
        Jx**2
        + Jy**2
        + Jz**2
    )
    J_max = (1.0 - eps) * phi
    denominator = xp.where(J_norm > 0.0, J_norm, 1.0)
    scale = xp.where(
        J_norm > J_max,
        J_max / denominator,
        1.0,
    )

    U[0] = phi
    U[1] = Jx * scale
    U[2] = Jy * scale
    U[3] = Jz * scale
    return U
