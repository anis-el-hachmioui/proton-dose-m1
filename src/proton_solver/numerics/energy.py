from dataclasses import replace
import math
from typing import Any

import numpy as np

from ..config import SolverConfig
from ..enums import EnergyScheme, WaveSpeeds, Scattering
from ..physics.materials import MaterialMap, density, density_field
from ..physics.stopping_power import (
    stopping_power,
    stopping_power_field,
)
from ..physics.scattering import (
    angular_scattering_substep_inplace,
    scattering_optical_depth,
)
from ..physics.nuclear import (
    nuclear_attenuation_substep_inplace,
    nuclear_optical_depth,
)
from .boundary import BeamInflow
from .grid import Grid
from .moving_active_domain import (
    box_slices,
    desired_box,
    embed_state,
    initial_box,
    make_local_problem,
    rebox,
)
from .realizability import project_realizable_inplace
from .spatial import spatial_operator
from .wave_speeds import eigenvalues


MEV_TO_JOULE = 1.602176634e-13


def _stopping_power_value_or_field(
    material_map: MaterialMap,
    energy: float,
    xp,
):
    """Return a scalar for homogeneous media, otherwise a voxel field."""

    if len(material_map.materials) == 1:
        return stopping_power(
            material_map.materials[0],
            energy,
        )

    return stopping_power_field(
        material_map,
        energy,
        xp,
    )


def _boundary_stopping_power(
    material_map: MaterialMap,
    energy: float,
    xp,
):
    """Return stopping power on the interior side of the z-min boundary."""

    if len(material_map.materials) == 1:
        return stopping_power(
            material_map.materials[0],
            energy,
        )

    values = xp.asarray([
        stopping_power(material, energy)
        for material in material_map.materials
    ])

    return values[
        material_map.ids[:, :, 0]
    ]


def _conserved_inflow_state(
    inflow: BeamInflow,
    material_map: MaterialMap,
    energy: float,
    xp,
):
    """
    Return the prescribed inflow in the marched variable

        W = S(E) U.
    """

    S_boundary = _boundary_stopping_power(
        material_map,
        energy,
        xp,
    )

    return (
        S_boundary
        * inflow.state(energy, xp)
    )


def _ssprk3_stage_inflow_states(
    inflow: BeamInflow,
    material_map: MaterialMap,
    energy: float,
    dE: float,
    xp,
):
    """
    Return RK-consistent inflow states for SSPRK3 stages 1 and 2.

    Define local pseudo-time tau such that kinetic energy is

        E(tau) = energy - tau.

    The marched variable is

        W = S(E) U,

    so define the prescribed boundary trace

        B(tau) = S(E(tau)) U_in(E(tau)).

    SSPRK3 requires

        B^(1) =
            B(0) + dE B'(0),

        B^(2) =
            B(0)
            + 1/2 dE B'(0)
            + 1/4 dE^2 B''(0).

    The derivatives are represented by a cubic interpolant through

        tau / dE = 0, 1/3, 2/3, 1,

    giving fourth-order accurate stage boundary values without
    differentiating the stopping-power interpolation explicitly.
    """

    if dE <= 0.0:
        raise ValueError(
            "dE must be positive."
        )

    B_0 = _conserved_inflow_state(
        inflow,
        material_map,
        energy,
        xp,
    )

    B_13 = _conserved_inflow_state(
        inflow,
        material_map,
        energy - dE / 3.0,
        xp,
    )

    B_23 = _conserved_inflow_state(
        inflow,
        material_map,
        energy - 2.0 * dE / 3.0,
        xp,
    )

    B_1 = _conserved_inflow_state(
        inflow,
        material_map,
        energy - dE,
        xp,
    )

    # SSPRK3 stage 1:
    #
    # B^(1) = B(0) + dE B'(0) + O(dE^4)
    B_stage_1 = 0.5 * (
        -9.0 * B_0
        + 18.0 * B_13
        - 9.0 * B_23
        + 2.0 * B_1
    )

    # SSPRK3 stage 2:
    #
    # B^(2) =
    #     B(0)
    #     + 1/2 dE B'(0)
    #     + 1/4 dE^2 B''(0)
    #     + O(dE^4)
    B_stage_2 = 0.25 * (
        11.0 * B_0
        - 27.0 * B_13
        + 27.0 * B_23
        - 7.0 * B_1
    )

    # Convert the RK stage values of W back to U.
    S_stage_1 = _boundary_stopping_power(
        material_map,
        energy - dE,
        xp,
    )

    S_stage_2 = _boundary_stopping_power(
        material_map,
        energy - 0.5 * dE,
        xp,
    )

    U_stage_1 = (
        B_stage_1
        / S_stage_1
    )

    U_stage_2 = (
        B_stage_2
        / S_stage_2
    )

    return (
        U_stage_1,
        U_stage_2,
    )



def _source_optical_depths(
    material_map: MaterialMap,
    energy: float,
    dE: float,
    config: SolverConfig,
    xp,
):
    """Return scattering and nuclear optical depths over one energy interval."""

    if dE <= 0.0:
        zero = 0.0
        return zero, zero

    if config.physics.scattering is Scattering.OFF:
        scattering_depth = 0.0
    else:
        scattering_depth = scattering_optical_depth(
            material_map,
            energy,
            dE,
            config.physics.scattering,
            xp,
            config.physics.coefficient_interpolation,
        )

    if config.physics.nuclear:
        nuclear_depth = nuclear_optical_depth(
            material_map,
            energy,
            dE,
            xp,
            config.physics.coefficient_interpolation,
        )
    else:
        nuclear_depth = 0.0

    return scattering_depth, nuclear_depth


def _apply_source_factor(
    W: Any,
    scattering_depth,
    nuclear_depth,
    xp,
    *,
    inverse: bool = False,
):
    """Apply the exact combined scattering/removal integrating factor.

    Nuclear attenuation damps every moment, while angular scattering damps
    only the current moments.  The two diagonal source operators commute, so
    their exact flow is

        Phi -> exp(-tau_n) Phi,
        J   -> exp(-(tau_n + tau_s)) J.

    Setting ``inverse=True`` applies the inverse factor used by the
    integrating-factor SSPRK stages.
    """

    sign = 1.0 if inverse else -1.0

    phi_exponent = sign * xp.minimum(
        nuclear_depth,
        700.0,
    )

    current_exponent = sign * xp.minimum(
        nuclear_depth + scattering_depth,
        700.0,
    )

    W_new = W.copy()
    W_new[0] *= xp.exp(phi_exponent)
    W_new[1:] *= xp.exp(current_exponent)

    return W_new


def _boundary_depth(depth):
    """Restrict a full 3-D optical-depth field to the z-min boundary."""

    if getattr(depth, "ndim", 0) == 3:
        return depth[:, :, 0]

    return depth


def _transformed_conserved_inflow_state(
    inflow: BeamInflow,
    material_map: MaterialMap,
    energy: float,
    scattering_depth,
    nuclear_depth,
    xp,
):
    """Return C(tau)^(-1) S(E) U_in(E) at z-min."""

    B = _conserved_inflow_state(
        inflow,
        material_map,
        energy,
        xp,
    )

    return _apply_source_factor(
        B,
        _boundary_depth(scattering_depth),
        _boundary_depth(nuclear_depth),
        xp,
        inverse=True,
    )


def _if_ssprk3_stage_inflow_states(
    inflow: BeamInflow,
    material_map: MaterialMap,
    energy: float,
    dE: float,
    source_13,
    source_12,
    source_23,
    source_1,
    xp,
):
    """Return physical inflow states consistent with IF-SSPRK3 stages."""

    if dE <= 0.0:
        raise ValueError(
            "dE must be positive."
        )

    scattering_13, nuclear_13 = source_13
    scattering_12, nuclear_12 = source_12
    scattering_23, nuclear_23 = source_23
    scattering_1, nuclear_1 = source_1

    # Interpolate the transformed boundary trace
    #
    #     V_b(tau) = C_b(tau)^(-1) S(E(tau)) U_in(E(tau))
    #
    # cubically through tau/dE = 0, 1/3, 2/3, 1.  These are the same
    # third-order SSPRK boundary stage formulas used by the unsplit solver,
    # now applied in integrating-factor variables.
    V_0 = _conserved_inflow_state(
        inflow,
        material_map,
        energy,
        xp,
    )

    V_13 = _transformed_conserved_inflow_state(
        inflow,
        material_map,
        energy - dE / 3.0,
        scattering_13,
        nuclear_13,
        xp,
    )

    V_23 = _transformed_conserved_inflow_state(
        inflow,
        material_map,
        energy - 2.0 * dE / 3.0,
        scattering_23,
        nuclear_23,
        xp,
    )

    V_1 = _transformed_conserved_inflow_state(
        inflow,
        material_map,
        energy - dE,
        scattering_1,
        nuclear_1,
        xp,
    )

    V_stage_1 = 0.5 * (
        -9.0 * V_0
        + 18.0 * V_13
        - 9.0 * V_23
        + 2.0 * V_1
    )

    V_stage_2 = 0.25 * (
        11.0 * V_0
        - 27.0 * V_13
        + 27.0 * V_23
        - 7.0 * V_1
    )

    W_stage_1 = _apply_source_factor(
        V_stage_1,
        _boundary_depth(scattering_1),
        _boundary_depth(nuclear_1),
        xp,
    )

    W_stage_2 = _apply_source_factor(
        V_stage_2,
        _boundary_depth(scattering_12),
        _boundary_depth(nuclear_12),
        xp,
    )

    S_stage_1 = _boundary_stopping_power(
        material_map,
        energy - dE,
        xp,
    )

    S_stage_2 = _boundary_stopping_power(
        material_map,
        energy - 0.5 * dE,
        xp,
    )

    return (
        W_stage_1 / S_stage_1,
        W_stage_2 / S_stage_2,
    )



def _directionally_dilate_active_mask(active, axis: int, radius: int, xp):
    """Dilate a boolean support mask only along one coordinate axis.

    The CFL bound needs a directional wave speed only where that directional
    finite-volume operator can be non-zero.  In vacuum guard cells, the M1
    eigenvalue routine returns the isotropic limit +/-1/sqrt(3) even though a
    vacuum-vacuum interface carries exactly zero flux.  Applying that speed in
    *all* directions of every guard cell makes the multidimensional CFL bound
    unnecessarily restrictive.

    A radius equal to the reconstruction ghost width conservatively keeps all
    cells whose value can enter a reconstructed interface state.
    """

    mask = active.copy()

    for _ in range(radius):
        old = mask
        grown = old.copy()

        if axis == 0:
            grown[1:, :, :] |= old[:-1, :, :]
            grown[:-1, :, :] |= old[1:, :, :]
        elif axis == 1:
            grown[:, 1:, :] |= old[:, :-1, :]
            grown[:, :-1, :] |= old[:, 1:, :]
        elif axis == 2:
            grown[:, :, 1:] |= old[:, :, :-1]
            grown[:, :, :-1] |= old[:, :, 1:]
        else:
            raise ValueError("axis must be 0, 1, or 2.")

        mask = grown

    return mask


def _vacuum_aware_cfl_masks(
    U: Any,
    U_in: Any,
    relative_tolerance: float,
    stencil_radius: int,
    xp,
):
    """Return directional masks for the compact-domain CFL calculation.

    Activity uses the same relative threshold as the moving-domain truncation.
    The prescribed entrance beam is included at z=0.  Each directional mask is
    then dilated only in its own axis, which retains active-vacuum interfaces
    needed by the reconstruction while excluding unrelated vacuum directions.
    """

    phi = U[0]
    phi_in = U_in[0]

    scale = max(
        float(xp.max(phi)),
        float(xp.max(phi_in)),
    )

    if scale == 0.0:
        # No transported state and no beam.  Falling back to all directions is
        # conservative and avoids a zero inverse CFL bound.
        full = xp.ones_like(phi, dtype=bool)
        return full, full, full

    threshold = relative_tolerance * scale
    active = phi > threshold
    active[:, :, 0] |= phi_in > threshold

    return tuple(
        _directionally_dilate_active_mask(
            active,
            axis,
            stencil_radius,
            xp,
        )
        for axis in range(3)
    )

def compute_energy_step(
    U: Any,
    energy: float,
    material_map: MaterialMap,
    grid: Grid,
    inflow: BeamInflow,
    config: SolverConfig,
    xp,
    *,
    vacuum_relative_tolerance: float | None = None,
):
    """Compute a stable energy step from the spatial CFL condition."""

    # U is realizable by construction: every accepted RK state is projected,
    # and the scattering substep only damps J.  Avoid a redundant full-grid
    # projection here because this routine is called once per energy step.
    S = _stopping_power_value_or_field(
        material_map,
        energy,
        xp,
    )

    if config.numerics.wave_speeds is WaveSpeeds.UNIT:
        inverse_energy_step = (
            1.0 / (S * grid.dx)
            + 1.0 / (S * grid.dy)
            + 1.0 / (S * grid.dz)
        )

    elif config.numerics.wave_speeds is WaveSpeeds.EXACT:
        lambda_x_min, lambda_x_max = eigenvalues(
            U, axis=0, xp=xp
        )
        lambda_y_min, lambda_y_max = eigenvalues(
            U, axis=1, xp=xp
        )
        lambda_z_min, lambda_z_max = eigenvalues(
            U, axis=2, xp=xp
        )

        a_x = xp.maximum(
            xp.abs(lambda_x_min),
            xp.abs(lambda_x_max),
        )
        a_y = xp.maximum(
            xp.abs(lambda_y_min),
            xp.abs(lambda_y_max),
        )
        a_z = xp.maximum(
            xp.abs(lambda_z_min),
            xp.abs(lambda_z_max),
        )

        # Include the z=0 beam in the CFL condition.
        U_in = inflow.state(energy, xp)

        lambda_in_min, lambda_in_max = eigenvalues(
            U_in,
            axis=2,
            xp=xp,
        )

        a_in = xp.maximum(
            xp.abs(lambda_in_min),
            xp.abs(lambda_in_max),
        )

        a_z[:, :, 0] = xp.maximum(
            a_z[:, :, 0],
            a_in,
        )

        if vacuum_relative_tolerance is not None:
            # Match the reconstruction domain of dependence.  The current
            # schemes use ghost widths 1 (first order), 2 (MUSCL/WENO3), and
            # 3 (WENO-Z).
            scheme_name = config.numerics.spatial_scheme.name
            if scheme_name == "FIRST_ORDER":
                stencil_radius = 1
            elif scheme_name in ("MUSCL", "WENO3"):
                stencil_radius = 2
            else:
                stencil_radius = 3

            mask_x, mask_y, mask_z = _vacuum_aware_cfl_masks(
                U,
                U_in,
                vacuum_relative_tolerance,
                stencil_radius,
                xp,
            )

            # A vacuum cell can be relevant in x without being relevant in y
            # or z.  Zero only the *directional* CFL contribution, rather than
            # discarding the whole cell.  This retains active-vacuum interfaces
            # but removes the spurious isotropic speed in unrelated guard-cell
            # directions.
            a_x = xp.where(mask_x, a_x, 0.0)
            a_y = xp.where(mask_y, a_y, 0.0)
            a_z = xp.where(mask_z, a_z, 0.0)

        inverse_energy_step = (
            a_x / (S * grid.dx)
            + a_y / (S * grid.dy)
            + a_z / (S * grid.dz)
        )

    else:
        raise ValueError(
            f"Unknown wave-speed mode: "
            f"{config.numerics.wave_speeds}"
        )

    dE = (
        config.numerics.cfl
        / float(xp.max(inverse_energy_step))
    )

    return min(
        dE,
        config.numerics.max_energy_step,
    )


def forward_euler(
    U: Any,
    energy: float,
    dE: float,
    material_map: MaterialMap,
    grid: Grid,
    inflow: BeamInflow,
    config: SolverConfig,
    xp,
):
    """Advance one energy step with forward Euler."""

    # The solver passes an already-realizable accepted state.
    S_old = _stopping_power_value_or_field(
        material_map,
        energy,
        xp,
    )

    W_old = S_old * U

    rhs = spatial_operator(
        U,
        energy,
        grid,
        inflow,
        config.numerics.spatial_scheme,
        config.numerics.wave_speeds,
        xp,
        active_domain=config.numerics.active_domain,
        active_relative_tolerance=(
        config.numerics.active_relative_tolerance
        ),
    )

    # rhs is dead after this stage, so reuse its storage for W_new.
    W_new = rhs
    W_new *= dE
    W_new += W_old

    energy_new = energy - dE

    S_new = _stopping_power_value_or_field(
        material_map,
        energy_new,
        xp,
    )

    # W_new is not needed after conversion back to U.
    U_new = W_new
    U_new /= S_new
    U_new = project_realizable_inplace(U_new, xp)

    energy_deposition = (
        dE
        * U[0]
        * S_old
    )

    return U_new, energy_deposition


def ssprk3(
    U: Any,
    energy: float,
    dE: float,
    material_map: MaterialMap,
    grid: Grid,
    inflow: BeamInflow,
    config: SolverConfig,
    xp,
):
    """Advance one energy step with SSPRK3."""

    # Accepted states are already realizable.  With Strang splitting the
    # preceding half-scattering step only damps J, so it preserves this
    # invariant as well.
    boundary_stage_1, boundary_stage_2 = (
    _ssprk3_stage_inflow_states(
        inflow,
        material_map,
        energy,
        dE,
        xp,
    )
)

    # Stage 0
    S_0 = _stopping_power_value_or_field(
        material_map,
        energy,
        xp,
    )

    W_0 = S_0 * U
    q_0 = U[0] * S_0

    rhs_0 = spatial_operator(
        U,
        energy,
        grid,
        inflow,
        config.numerics.spatial_scheme,
        config.numerics.wave_speeds,
        xp,
        active_domain=config.numerics.active_domain,
        active_relative_tolerance=(
        config.numerics.active_relative_tolerance
        ),
    )

    # Stage 1. Reuse rhs_0 storage for W_1.
    W_1 = rhs_0
    W_1 *= dE
    W_1 += W_0
    energy_1 = energy - dE

    S_1 = _stopping_power_value_or_field(
        material_map,
        energy_1,
        xp,
    )

    U_1 = W_1 / S_1
    U_1 = project_realizable_inplace(U_1, xp)

    q_1 = U_1[0] * S_1

    rhs_1 = spatial_operator(
        U_1,
        energy_1,
        grid,
        inflow,
        config.numerics.spatial_scheme,
        config.numerics.wave_speeds,
        xp,
        boundary_state=boundary_stage_1,
        active_domain=config.numerics.active_domain,
        active_relative_tolerance=(
        config.numerics.active_relative_tolerance
        ),
    )

    # Stage 2. Reuse rhs_1 for the bracketed Euler update and allocate
    # only the actual W_2 output array.
    rhs_1 *= dE
    rhs_1 += W_1
    rhs_1 *= 0.25

    W_2 = 0.75 * W_0
    W_2 += rhs_1

    energy_2 = energy - 0.5 * dE

    S_2 = _stopping_power_value_or_field(
        material_map,
        energy_2,
        xp,
    )

    U_2 = W_2 / S_2
    U_2 = project_realizable_inplace(U_2, xp)

    q_2 = U_2[0] * S_2

    rhs_2 = spatial_operator(
        U_2,
        energy_2,
        grid,
        inflow,
        config.numerics.spatial_scheme,
        config.numerics.wave_speeds,
        xp,
        boundary_state=boundary_stage_2,
        active_domain=config.numerics.active_domain,
        active_relative_tolerance=(
        config.numerics.active_relative_tolerance
        ),
    )

    # Final stage. Reuse rhs_2 for the bracketed Euler update.
    rhs_2 *= dE
    rhs_2 += W_2
    rhs_2 *= (2.0 / 3.0)

    W_new = (1.0 / 3.0) * W_0
    W_new += rhs_2

    energy_new = energy - dE

    S_new = _stopping_power_value_or_field(
        material_map,
        energy_new,
        xp,
    )

    U_new = W_new
    U_new /= S_new
    U_new = project_realizable_inplace(U_new, xp)

    # Third-order quadrature for dose deposition
    energy_deposition = (
        dE
        * (
            (1.0 / 6.0) * q_0
            + (1.0 / 6.0) * q_1
            + (2.0 / 3.0) * q_2
        )
    )

    return U_new, energy_deposition




def if_ssprk3(
    U: Any,
    energy: float,
    dE: float,
    material_map: MaterialMap,
    grid: Grid,
    inflow: BeamInflow,
    config: SolverConfig,
    xp,
):
    """Advance one step with third-order integrating-factor SSPRK3.

    The marched conserved variable is ``W = S(E) U``.  Scattering and
    primary nuclear removal are diagonal local source terms in W:

        Phi : exp(-tau_n),
        J   : exp(-(tau_n + tau_s)).

    Writing ``W = C(tau) V`` removes both source terms analytically. SSPRK3 is
    then applied to the transformed streaming system, so no Strang splitting
    is used on this path.
    """

    # With no local source terms the integrating factor is the identity and
    # the existing optimized SSPRK3 implementation is exactly the desired
    # method.
    if (
        config.physics.scattering is Scattering.OFF
        and not config.physics.nuclear
    ):
        return ssprk3(
            U,
            energy,
            dE,
            material_map,
            grid,
            inflow,
            config,
            xp,
        )

    source_13 = _source_optical_depths(
        material_map,
        energy,
        dE / 3.0,
        config,
        xp,
    )

    source_12 = _source_optical_depths(
        material_map,
        energy,
        0.5 * dE,
        config,
        xp,
    )

    source_23 = _source_optical_depths(
        material_map,
        energy,
        2.0 * dE / 3.0,
        config,
        xp,
    )

    source_1 = _source_optical_depths(
        material_map,
        energy,
        dE,
        config,
        xp,
    )

    scattering_2, nuclear_2 = source_12
    scattering_1, nuclear_1 = source_1

    boundary_stage_1, boundary_stage_2 = (
        _if_ssprk3_stage_inflow_states(
            inflow,
            material_map,
            energy,
            dE,
            source_13,
            source_12,
            source_23,
            source_1,
            xp,
        )
    )

    # Stage 0: C(0) = I, hence V_0 = W_0.
    S_0 = _stopping_power_value_or_field(
        material_map,
        energy,
        xp,
    )

    V_0 = S_0 * U
    q_0 = U[0] * S_0

    rhs_0 = spatial_operator(
        U,
        energy,
        grid,
        inflow,
        config.numerics.spatial_scheme,
        config.numerics.wave_speeds,
        xp,
        active_domain=config.numerics.active_domain,
        active_relative_tolerance=(
            config.numerics.active_relative_tolerance
        ),
    )

    # SSPRK stage 1 at c=1 in transformed variables.
    V_1 = rhs_0
    V_1 *= dE
    V_1 += V_0

    energy_1 = energy - dE

    S_1 = _stopping_power_value_or_field(
        material_map,
        energy_1,
        xp,
    )

    W_1 = _apply_source_factor(
        V_1,
        scattering_1,
        nuclear_1,
        xp,
    )

    U_1 = W_1 / S_1
    U_1 = project_realizable_inplace(
        U_1,
        xp,
    )

    q_1 = U_1[0] * S_1

    rhs_1_W = spatial_operator(
        U_1,
        energy_1,
        grid,
        inflow,
        config.numerics.spatial_scheme,
        config.numerics.wave_speeds,
        xp,
        boundary_state=boundary_stage_1,
        active_domain=config.numerics.active_domain,
        active_relative_tolerance=(
            config.numerics.active_relative_tolerance
        ),
    )

    rhs_1 = _apply_source_factor(
        rhs_1_W,
        scattering_1,
        nuclear_1,
        xp,
        inverse=True,
    )

    # SSPRK stage 2 at c=1/2.
    rhs_1 *= dE
    rhs_1 += V_1
    rhs_1 *= 0.25

    V_2 = 0.75 * V_0
    V_2 += rhs_1

    energy_2 = energy - 0.5 * dE

    S_2 = _stopping_power_value_or_field(
        material_map,
        energy_2,
        xp,
    )

    W_2 = _apply_source_factor(
        V_2,
        scattering_2,
        nuclear_2,
        xp,
    )

    U_2 = W_2 / S_2
    U_2 = project_realizable_inplace(
        U_2,
        xp,
    )

    q_2 = U_2[0] * S_2

    rhs_2_W = spatial_operator(
        U_2,
        energy_2,
        grid,
        inflow,
        config.numerics.spatial_scheme,
        config.numerics.wave_speeds,
        xp,
        boundary_state=boundary_stage_2,
        active_domain=config.numerics.active_domain,
        active_relative_tolerance=(
            config.numerics.active_relative_tolerance
        ),
    )

    rhs_2 = _apply_source_factor(
        rhs_2_W,
        scattering_2,
        nuclear_2,
        xp,
        inverse=True,
    )

    # Final SSPRK combination in V, then exact local source flow to c=1.
    rhs_2 *= dE
    rhs_2 += V_2
    rhs_2 *= (2.0 / 3.0)

    V_new = (1.0 / 3.0) * V_0
    V_new += rhs_2

    W_new = _apply_source_factor(
        V_new,
        scattering_1,
        nuclear_1,
        xp,
    )

    U_new = W_new
    U_new /= S_1
    U_new = project_realizable_inplace(
        U_new,
        xp,
    )

    # Third-order SSPRK quadrature for primary collisional dose.  The stage
    # states already contain the exact local attenuation factors.
    energy_deposition = (
        dE
        * (
            (1.0 / 6.0) * q_0
            + (1.0 / 6.0) * q_1
            + (2.0 / 3.0) * q_2
        )
    )

    return U_new, energy_deposition

def advance_energy_step(
    U: Any,
    energy: float,
    dE: float,
    material_map: MaterialMap,
    grid: Grid,
    inflow: BeamInflow,
    config: SolverConfig,
    xp,
):
    """Advance one step with the configured energy integrator."""

    scheme = config.numerics.energy_scheme

    if scheme is EnergyScheme.FORWARD_EULER:
        return forward_euler(
            U, energy, dE,
            material_map, grid, inflow,
            config, xp,
        )

    if scheme in (
        EnergyScheme.SSPRK3,
        EnergyScheme.IF_SSPRK3,
    ):
        return ssprk3(
            U, energy, dE,
            material_map, grid, inflow,
            config, xp,
        )

    raise NotImplementedError(
        f"Energy scheme {scheme} has not been migrated yet."
    )


def _split_source_half_step_inplace(
    U: Any,
    scattering_depth,
    nuclear_depth,
    config: SolverConfig,
    xp,
):
    """Apply one half-step of the commuting local source operators."""

    if config.physics.nuclear:
        U = nuclear_attenuation_substep_inplace(
            U,
            0.5 * nuclear_depth,
            xp,
        )

    if config.physics.scattering is not Scattering.OFF:
        U = angular_scattering_substep_inplace(
            U,
            0.5 * scattering_depth,
            xp,
        )

    return U


def strang_energy_step(
    U: Any,
    energy: float,
    dE: float,
    material_map: MaterialMap,
    grid: Grid,
    inflow: BeamInflow,
    config: SolverConfig,
    xp,
):
    """Advance one step with Strang-split scattering/removal sources."""

    if config.physics.scattering is not Scattering.OFF:
        eta = scattering_optical_depth(
            material_map,
            energy,
            dE,
            config.physics.scattering,
            xp,
            config.physics.coefficient_interpolation,
        )
    else:
        eta = 0.0

    if config.physics.nuclear:
        tau_n = nuclear_optical_depth(
            material_map,
            energy,
            dE,
            xp,
            config.physics.coefficient_interpolation,
        )
    else:
        tau_n = 0.0

    # Scattering damps J only; nuclear removal scales Phi and J equally.
    # Their exact local source flows therefore commute.
    U_half = _split_source_half_step_inplace(
        U,
        eta,
        tau_n,
        config,
        xp,
    )

    # Full CSDA + streaming step.  Dose is evaluated at the Strang midpoint
    # with respect to the local attenuation/scattering source operators.
    U_transport, energy_deposition = advance_energy_step(
        U_half,
        energy,
        dE,
        material_map,
        grid,
        inflow,
        config,
        xp,
    )

    U_new = _split_source_half_step_inplace(
        U_transport,
        eta,
        tau_n,
        config,
        xp,
    )

    return U_new, energy_deposition

def _moving_active_energy_march(
    U: Any | None,
    material_map: MaterialMap,
    grid: Grid,
    inflow: BeamInflow,
    config: SolverConfig,
    xp,
    snapshot_energies: tuple[float, ...] = (),
):
    """March on a compact 3-D box that follows the proton population.

    The global dose stays on the full geometry, while the transported M1
    state is only allocated on the current active box.  The active support is
    truncated at a configurable Gaussian-equivalent relative level and is
    surrounded by ``active_guard_cells`` before the next SSPRK step.
    """

    if xp is not np:
        raise NotImplementedError(
            "moving_active_domain is currently implemented for the CPU "
            "NumPy/Numba backend only."
        )

    energy = (
        config.beam.energy
        + config.numerics.energy_window_sigma
        * config.beam.energy_spread
    )
    energy_min = config.numerics.energy_min

    targets = sorted(
        {
            float(target)
            for target in snapshot_energies
            if energy_min <= target <= energy
        },
        reverse=True,
    )
    target_index = 0
    snapshots = {}
    energies = [energy]

    guard = config.numerics.active_guard_cells
    relative_tolerance = math.exp(
        -0.5 * config.numerics.active_truncation_sigma**2
    )

    box = initial_box(
        inflow,
        grid,
        guard,
        xp,
    )

    i0, i1, j0, j1, k0, k1 = box
    local_shape = (i1 - i0, j1 - j0, k1 - k0)

    if U is None:
        U_active = xp.zeros(
            (4, *local_shape),
            dtype=float,
        )
    else:
        # Backward-compatible path for a supplied full-grid initial state.
        U_active = U[:, i0:i1, j0:j1, k0:k1].copy()

    dose = xp.zeros(
        grid.shape,
        dtype=U_active.dtype,
    )

    # Avoid recursively invoking the old per-spatial-operator active-domain
    # crop.  The entire local array is already compact and includes its guard.
    local_config = replace(
        config,
        numerics=replace(
            config.numerics,
            active_domain=False,
            moving_active_domain=False,
        ),
    )

    while energy > energy_min:
        local_grid, local_inflow, local_material_map = make_local_problem(
            grid,
            inflow,
            material_map,
            box,
            xp,
        )

        next_box = desired_box(
            U_active,
            box,
            local_inflow,
            inflow,
            grid,
            energy,
            relative_tolerance,
            guard,
            xp,
        )

        if next_box != box:
            U_active = rebox(
                U_active,
                box,
                next_box,
                xp,
            )
            box = next_box
            local_grid, local_inflow, local_material_map = make_local_problem(
                grid,
                inflow,
                material_map,
                box,
                xp,
            )

        dE = compute_energy_step(
            U_active,
            energy,
            local_material_map,
            local_grid,
            local_inflow,
            local_config,
            xp,
            vacuum_relative_tolerance=(
                relative_tolerance
                if config.numerics.vacuum_aware_cfl
                else None
            ),
        )

        # Land exactly on the next requested snapshot energy.
        if target_index < len(targets):
            target = targets[target_index]
            if energy > target >= energy - dE:
                dE = energy - target

        dE = min(
            dE,
            energy - energy_min,
        )

        if config.numerics.energy_scheme is EnergyScheme.IF_SSPRK3:
            U_active, energy_deposition = if_ssprk3(
                U_active,
                energy,
                dE,
                local_material_map,
                local_grid,
                local_inflow,
                local_config,
                xp,
            )
        elif (
            config.physics.scattering is not Scattering.OFF
            or config.physics.nuclear
        ):
            U_active, energy_deposition = strang_energy_step(
                U_active,
                energy,
                dE,
                local_material_map,
                local_grid,
                local_inflow,
                local_config,
                xp,
            )
        else:
            U_active, energy_deposition = advance_energy_step(
                U_active,
                energy,
                dE,
                local_material_map,
                local_grid,
                local_inflow,
                local_config,
                xp,
            )

        if len(local_material_map.materials) == 1:
            rho_local = density(local_material_map.materials[0])
        else:
            rho_local = density_field(
                local_material_map,
                xp,
            )

        sl = box_slices(box)
        dose[sl] += (
            energy_deposition
            * MEV_TO_JOULE
            / rho_local
        )

        energy -= dE
        energies.append(energy)

        if target_index < len(targets):
            target = targets[target_index]
            if np.isclose(
                energy,
                target,
                rtol=0.0,
                atol=1e-12,
            ):
                # Snapshot requests explicitly ask for a full-grid state.
                snapshots[target] = embed_state(
                    U_active,
                    box,
                    grid.shape,
                    xp,
                )
                target_index += 1

    if config.output.store_final_state:
        U_final = embed_state(
            U_active,
            box,
            grid.shape,
            xp,
        )
    else:
        U_final = None

    return (
        U_final,
        dose,
        np.asarray(energies),
        snapshots,
    )


def energy_march(
    U: Any,
    material_map: MaterialMap,
    grid: Grid,
    inflow: BeamInflow,
    config: SolverConfig,
    xp,
    snapshot_energies: tuple[float, ...] = (),
):
    """March the M1 system from high to low kinetic energy."""

    if config.numerics.moving_active_domain:
        return _moving_active_energy_march(
            U,
            material_map,
            grid,
            inflow,
            config,
            xp,
            snapshot_energies=snapshot_energies,
        )

    if U is None:
        raise ValueError(
            "A full-grid initial state is required when moving_active_domain "
            "is disabled."
        )

    energy = (
        config.beam.energy
        + config.numerics.energy_window_sigma
        * config.beam.energy_spread
    )

    energy_min = config.numerics.energy_min

    targets = sorted(
        {
            float(target)
            for target in snapshot_energies
            if energy_min <= target <= energy
        },
        reverse=True,
    )

    target_index = 0
    snapshots = {}

    energies = [energy]
    dose = xp.zeros_like(U[0])

    if len(material_map.materials) == 1:
        rho = density(material_map.materials[0])
    else:
        rho = density_field(
            material_map,
            xp,
        )

    while energy > energy_min:
        dE = compute_energy_step(
            U,
            energy,
            material_map,
            grid,
            inflow,
            config,
            xp,
        )

        # Land exactly on the next requested snapshot energy.
        if target_index < len(targets):
            target = targets[target_index]

            if energy > target >= energy - dE:
                dE = energy - target

        # Finish exactly at energy_min.
        dE = min(
            dE,
            energy - energy_min,
        )

        if config.numerics.energy_scheme is EnergyScheme.IF_SSPRK3:
            U, energy_deposition = if_ssprk3(
                U,
                energy,
                dE,
                material_map,
                grid,
                inflow,
                config,
                xp,
            )
        elif (
            config.physics.scattering is not Scattering.OFF
            or config.physics.nuclear
        ):
            U, energy_deposition = strang_energy_step(
                U,
                energy,
                dE,
                material_map,
                grid,
                inflow,
                config,
                xp,
            )
        else:
            U, energy_deposition = advance_energy_step(
                U,
                energy,
                dE,
                material_map,
                grid,
                inflow,
                config,
                xp,
            )

        dose += (
            energy_deposition
            * MEV_TO_JOULE
            / rho
        )

        energy -= dE
        energies.append(energy)

        # Store the state if we have reached a requested energy.
        if target_index < len(targets):
            target = targets[target_index]

            if np.isclose(
                energy,
                target,
                rtol=0.0,
                atol=1e-12,
            ):
                snapshots[target] = U.copy()
                target_index += 1

    return (
        U,
        dose,
        np.asarray(energies),
        snapshots,
    )