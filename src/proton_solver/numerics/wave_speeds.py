import math
from typing import Any

import numpy as np
from numba import njit, prange

from ..enums import WaveSpeeds


EPS = 1.0e-10

@njit(
    cache=True,
    parallel=True,
)
def _eigenvalues_numba(U, axis):
    """CPU implementation of the M1 eigenvalues."""

    phi = U[0]
    Jx = U[1]
    Jy = U[2]
    Jz = U[3]

    lambda_min = np.empty_like(phi)
    lambda_max = np.empty_like(phi)

    phi_flat = phi.ravel()
    Jx_flat = Jx.ravel()
    Jy_flat = Jy.ravel()
    Jz_flat = Jz.ravel()

    lambda_min_flat = lambda_min.ravel()
    lambda_max_flat = lambda_max.ravel()

    isotropic_speed = 1.0 / np.sqrt(3.0)

    for i in prange(phi_flat.size):
        phi_i = phi_flat[i]

        if phi_i < EPS:
            phi_i = EPS

        Jx_i = Jx_flat[i]
        Jy_i = Jy_flat[i]
        Jz_i = Jz_flat[i]

        J_norm = np.sqrt(
            Jx_i * Jx_i
            + Jy_i * Jy_i
            + Jz_i * Jz_i
        )

        if J_norm < EPS:
            lambda_min_flat[i] = -isotropic_speed
            lambda_max_flat[i] = isotropic_speed
            continue

        f = J_norm / phi_i

        if f < 0.0:
            f = 0.0
        elif f > 1.0 - EPS:
            f = 1.0 - EPS

        if axis == 0:
            J_normal = Jx_i
        elif axis == 1:
            J_normal = Jy_i
        else:
            J_normal = Jz_i

        mu = J_normal / J_norm

        chi = (
            5.0
            - 2.0 * np.sqrt(
                4.0 - 3.0 * f * f
            )
        ) / 3.0

        denominator = 5.0 - 3.0 * chi

        discriminant = (
            4.0 * f * f * mu * mu
            - denominator
            * (
                mu * mu * (3.0 * chi - 1.0)
                - 2.0 * (1.0 - chi)
            )
        )

        if discriminant < 0.0:
            discriminant = 0.0

        root = np.sqrt(discriminant)

        lambda_min_flat[i] = (
            2.0 * f * mu - root
        ) / denominator

        lambda_max_flat[i] = (
            2.0 * f * mu + root
        ) / denominator

    return lambda_min, lambda_max


def _eigenvalues_vectorized(U: Any, axis: int, xp):
    """Return the minimum and maximum M1 eigenvalues along an axis."""

    phi = U[0]
    Jx = U[1]
    Jy = U[2]
    Jz = U[3]

    J_norm = xp.sqrt(
        Jx**2
        + Jy**2
        + Jz**2
    )

    phi_safe = xp.maximum(phi, EPS)

    f = xp.clip(
        J_norm / phi_safe,
        0.0,
        1.0 - EPS,
    )

    J_normal = U[axis + 1]

    mu = xp.where(
        J_norm > EPS,
        J_normal / xp.maximum(J_norm, EPS),
        0.0,
    )

    chi = (
        5.0
        - 2.0 * xp.sqrt(4.0 - 3.0 * f**2)
    ) / 3.0

    denominator = 5.0 - 3.0 * chi

    discriminant = (
        4.0 * f**2 * mu**2
        - denominator
        * (
            mu**2 * (3.0 * chi - 1.0)
            - 2.0 * (1.0 - chi)
        )
    )

    root = xp.sqrt(
        xp.maximum(discriminant, 0.0)
    )

    lambda_min = (
        2.0 * f * mu - root
    ) / denominator

    lambda_max = (
        2.0 * f * mu + root
    ) / denominator

    # Isotropic limit: f -> 0
    isotropic = J_norm < EPS
    isotropic_speed = 1.0 / math.sqrt(3.0)

    lambda_min = xp.where(
        isotropic,
        -isotropic_speed,
        lambda_min,
    )

    lambda_max = xp.where(
        isotropic,
        isotropic_speed,
        lambda_max,
    )

    return lambda_min, lambda_max



def eigenvalues(U: Any, axis: int, xp):
    """Return the minimum and maximum M1 eigenvalues along an axis."""

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")

    if xp is np:
        return _eigenvalues_numba(U, axis)

    return _eigenvalues_vectorized(U, axis, xp)


def hll_wave_speeds(
    U_left: Any,
    U_right: Any,
    axis: int,
    mode: WaveSpeeds,
    xp,
):
    """Return left and right signal speeds for the HLL flux."""

    if mode is WaveSpeeds.UNIT:
        return -1.0, 1.0

    if mode is not WaveSpeeds.EXACT:
        raise ValueError(f"Unknown wave-speed mode: {mode}")

    lambda_left_min, lambda_left_max = eigenvalues(
        U_left, axis, xp
    )

    lambda_right_min, lambda_right_max = eigenvalues(
        U_right, axis, xp
    )

    S_left = xp.minimum(
        lambda_left_min,
        lambda_right_min,
    )

    S_right = xp.maximum(
        lambda_left_max,
        lambda_right_max,
    )

    # HLL fan must include zero
    S_left = xp.minimum(S_left, 0.0)
    S_right = xp.maximum(S_right, 0.0)

    return S_left, S_right