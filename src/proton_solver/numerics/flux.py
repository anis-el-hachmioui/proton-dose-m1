from typing import Any

import numpy as np
from numba import njit, prange

from .wave_speeds import EPS, hll_wave_speeds
from ..enums import WaveSpeeds


@njit(inline="always")
def _closure_scalar(
    phi,
    Jx,
    Jy,
    Jz,
):
    """Scalar M1 closure for the fused CPU HLL kernel."""

    J_norm = np.sqrt(
        Jx * Jx
        + Jy * Jy
        + Jz * Jz
    )

    if phi > 0.0:
        f = J_norm / phi

        if f < 0.0:
            f = 0.0
        elif f > 1.0:
            f = 1.0
    else:
        f = 0.0

    chi = (
        5.0
        - 2.0 * np.sqrt(
            4.0 - 3.0 * f * f
        )
    ) / 3.0

    if J_norm > 0.0:
        nx = Jx / J_norm
        ny = Jy / J_norm
        nz = Jz / J_norm
    else:
        nx = 0.0
        ny = 0.0
        nz = 0.0

    isotropic = 0.5 * (1.0 - chi)
    directed = 0.5 * (3.0 * chi - 1.0)

    return (
        J_norm,
        nx,
        ny,
        nz,
        isotropic,
        directed,
    )

@njit(inline="always")
def _eigenvalues_scalar(
    phi,
    Jx,
    Jy,
    Jz,
    J_norm,
    axis,
):
    """Scalar M1 eigenvalues for the fused CPU HLL kernel."""

    isotropic_speed = 1.0 / np.sqrt(3.0)

    if phi < EPS:
        phi = EPS

    if J_norm < EPS:
        return (
            -isotropic_speed,
            isotropic_speed,
        )

    f = J_norm / phi

    if f < 0.0:
        f = 0.0
    elif f > 1.0 - EPS:
        f = 1.0 - EPS

    if axis == 0:
        J_normal = Jx
    elif axis == 1:
        J_normal = Jy
    else:
        J_normal = Jz

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

    lambda_min = (
        2.0 * f * mu - root
    ) / denominator

    lambda_max = (
        2.0 * f * mu + root
    ) / denominator

    return (
        lambda_min,
        lambda_max,
    )

@njit(
    cache=True,
    parallel=True,
    fastmath=True,
)
def _hll_flux_numba(
    U_left,
    U_right,
    axis,
    wave_speed_mode,
):
    """Fused CPU implementation of M1 closure, wave speeds, and HLL flux."""

    F = np.empty_like(U_left)

    n = U_left.size // 4

    U_left_flat = U_left.reshape((4, n))
    U_right_flat = U_right.reshape((4, n))
    F_flat = F.reshape((4, n))

    for i in prange(n):
        # Left state
        phi_left = U_left_flat[0, i]
        Jx_left = U_left_flat[1, i]
        Jy_left = U_left_flat[2, i]
        Jz_left = U_left_flat[3, i]

        (
            J_norm_left,
            nx_left,
            ny_left,
            nz_left,
            isotropic_left,
            directed_left,
        ) = _closure_scalar(
            phi_left,
            Jx_left,
            Jy_left,
            Jz_left,
        )

        # Right state
        phi_right = U_right_flat[0, i]
        Jx_right = U_right_flat[1, i]
        Jy_right = U_right_flat[2, i]
        Jz_right = U_right_flat[3, i]

        (
            J_norm_right,
            nx_right,
            ny_right,
            nz_right,
            isotropic_right,
            directed_right,
        ) = _closure_scalar(
            phi_right,
            Jx_right,
            Jy_right,
            Jz_right,
        )

        # Physical fluxes
        if axis == 0:
            F_left_0 = Jx_left
            F_left_1 = phi_left * (
                isotropic_left
                + directed_left * nx_left * nx_left
            )
            F_left_2 = (
                phi_left
                * directed_left
                * nx_left
                * ny_left
            )
            F_left_3 = (
                phi_left
                * directed_left
                * nx_left
                * nz_left
            )

            F_right_0 = Jx_right
            F_right_1 = phi_right * (
                isotropic_right
                + directed_right * nx_right * nx_right
            )
            F_right_2 = (
                phi_right
                * directed_right
                * nx_right
                * ny_right
            )
            F_right_3 = (
                phi_right
                * directed_right
                * nx_right
                * nz_right
            )

        elif axis == 1:
            F_left_0 = Jy_left
            F_left_1 = (
                phi_left
                * directed_left
                * nx_left
                * ny_left
            )
            F_left_2 = phi_left * (
                isotropic_left
                + directed_left * ny_left * ny_left
            )
            F_left_3 = (
                phi_left
                * directed_left
                * ny_left
                * nz_left
            )

            F_right_0 = Jy_right
            F_right_1 = (
                phi_right
                * directed_right
                * nx_right
                * ny_right
            )
            F_right_2 = phi_right * (
                isotropic_right
                + directed_right * ny_right * ny_right
            )
            F_right_3 = (
                phi_right
                * directed_right
                * ny_right
                * nz_right
            )

        else:
            F_left_0 = Jz_left
            F_left_1 = (
                phi_left
                * directed_left
                * nx_left
                * nz_left
            )
            F_left_2 = (
                phi_left
                * directed_left
                * ny_left
                * nz_left
            )
            F_left_3 = phi_left * (
                isotropic_left
                + directed_left * nz_left * nz_left
            )

            F_right_0 = Jz_right
            F_right_1 = (
                phi_right
                * directed_right
                * nx_right
                * nz_right
            )
            F_right_2 = (
                phi_right
                * directed_right
                * ny_right
                * nz_right
            )
            F_right_3 = phi_right * (
                isotropic_right
                + directed_right * nz_right * nz_right
            )

        # Wave speeds
        if wave_speed_mode == 0:
            (
                lambda_left_min,
                lambda_left_max,
            ) = _eigenvalues_scalar(
                phi_left,
                Jx_left,
                Jy_left,
                Jz_left,
                J_norm_left,
                axis,
            )

            (
                lambda_right_min,
                lambda_right_max,
            ) = _eigenvalues_scalar(
                phi_right,
                Jx_right,
                Jy_right,
                Jz_right,
                J_norm_right,
                axis,
            )

            S_left = min(
                lambda_left_min,
                lambda_right_min,
            )

            S_right = max(
                lambda_left_max,
                lambda_right_max,
            )

            # HLL fan must include zero.
            if S_left > 0.0:
                S_left = 0.0

            if S_right < 0.0:
                S_right = 0.0

        else:
            S_left = -1.0
            S_right = 1.0

        # HLL numerical flux
        denominator = S_right - S_left
        S_product = S_left * S_right

        F_flat[0, i] = (
            S_right * F_left_0
            - S_left * F_right_0
            + S_product * (phi_right - phi_left)
        ) / denominator

        F_flat[1, i] = (
            S_right * F_left_1
            - S_left * F_right_1
            + S_product * (Jx_right - Jx_left)
        ) / denominator

        F_flat[2, i] = (
            S_right * F_left_2
            - S_left * F_right_2
            + S_product * (Jy_right - Jy_left)
        ) / denominator

        F_flat[3, i] = (
            S_right * F_left_3
            - S_left * F_right_3
            + S_product * (Jz_right - Jz_left)
        ) / denominator

    return F


def _closure_terms(U: Any, xp):
    """Compute the quantities required by the M1 closure."""

    phi = U[0]
    Jx = U[1]
    Jy = U[2]
    Jz = U[3]

    J_norm = xp.sqrt(
        Jx**2
        + Jy**2
        + Jz**2
    )

    # Reduced flux f = |J| / phi
    phi_safe = xp.where(phi > 0.0, phi, 1.0)

    f = xp.where(
        phi > 0.0,
        J_norm / phi_safe,
        0.0,
    )

    f = xp.clip(f, 0.0, 1.0)

    # Levermore Eddington factor
    chi = (
        5.0
        - 2.0 * xp.sqrt(4.0 - 3.0 * f**2)
    ) / 3.0

    # Direction of the flux
    J_safe = xp.where(J_norm > 0.0, J_norm, 1.0)

    nx = xp.where(J_norm > 0.0, Jx / J_safe, 0.0)
    ny = xp.where(J_norm > 0.0, Jy / J_safe, 0.0)
    nz = xp.where(J_norm > 0.0, Jz / J_safe, 0.0)

    isotropic = 0.5 * (1.0 - chi)
    directed = 0.5 * (3.0 * chi - 1.0)

    return (
        phi,
        nx,
        ny,
        nz,
        isotropic,
        directed,
    )


def physical_flux(U: Any, axis: int, xp):
    """Return the physical M1 flux in the requested spatial direction."""

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")

    (
        phi,
        nx,
        ny,
        nz,
        isotropic,
        directed,
    ) = _closure_terms(U, xp)

    if axis == 0:
        return xp.stack((
            U[1],
            phi * (isotropic + directed * nx * nx),
            phi * directed * nx * ny,
            phi * directed * nx * nz,
        ))

    if axis == 1:
        return xp.stack((
            U[2],
            phi * directed * nx * ny,
            phi * (isotropic + directed * ny * ny),
            phi * directed * ny * nz,
        ))

    return xp.stack((
        U[3],
        phi * directed * nx * nz,
        phi * directed * ny * nz,
        phi * (isotropic + directed * nz * nz),
    ))


def _hll_flux_vectorized(
    U_left: Any,
    U_right: Any,
    axis: int,
    wave_speed_mode: WaveSpeeds,
    xp,
):
    """Return the HLL numerical flux at cell interfaces."""

    F_left = physical_flux(U_left, axis, xp)
    F_right = physical_flux(U_right, axis, xp)

    S_left, S_right = hll_wave_speeds(
        U_left,
        U_right,
        axis,
        wave_speed_mode,
        xp,
    )

    # Convert scalar UNIT speeds to backend arrays as well
    S_left = xp.asarray(S_left)
    S_right = xp.asarray(S_right)

    denominator = S_right - S_left

    F_hll = (
        S_right * F_left
        - S_left * F_right
        + S_left * S_right * (U_right - U_left)
    ) / denominator

    # Upwind limits
    return xp.where(
        (S_left >= 0.0)[None, ...],
        F_left,
        xp.where(
            (S_right <= 0.0)[None, ...],
            F_right,
            F_hll,
        ),
    )


def hll_flux(
    U_left: Any,
    U_right: Any,
    axis: int,
    wave_speed_mode: WaveSpeeds,
    xp,
):
    """Return the HLL numerical flux at cell interfaces."""

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")

    if wave_speed_mode is WaveSpeeds.EXACT:
        mode_code = 0

    elif wave_speed_mode is WaveSpeeds.UNIT:
        mode_code = 1

    else:
        raise ValueError(
            f"Unknown wave-speed mode: {wave_speed_mode}"
        )

    # Fast CPU path
    if xp is np:
        return _hll_flux_numba(
            U_left,
            U_right,
            axis,
            mode_code,
        )

    # Backend-generic GPU/CuPy path
    return _hll_flux_vectorized(
        U_left,
        U_right,
        axis,
        wave_speed_mode,
        xp,
    )
