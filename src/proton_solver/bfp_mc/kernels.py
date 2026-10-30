"""Numba transport and scoring kernels for the BFP Monte Carlo reference.

This module will contain:
- deterministic per-history RNG,
- angular Brownian motion on S^2,
- homogeneous transport kernel,
- heterogeneous geometry-aware transport kernel,
- voxel/segment dose-scoring helpers.
"""

from __future__ import annotations

import math

import numpy as np
from numba import get_thread_id, njit, prange
from .materials import (
    _linear_material_value,
    _pchip_material_value,
)

# -----------------------------------------------------------------------------
# Deterministic per-history RNG
# -----------------------------------------------------------------------------


@njit(inline="always")
def _splitmix64(x):
    """Scramble an integer into a high-quality 64-bit RNG seed."""

    z = np.uint64(
        x
        + np.uint64(
            0x9E3779B97F4A7C15
        )
    )

    z = np.uint64(
        (
            z
            ^ (
                z
                >> np.uint64(30)
            )
        )
        * np.uint64(
            0xBF58476D1CE4E5B9
        )
    )

    z = np.uint64(
        (
            z
            ^ (
                z
                >> np.uint64(27)
            )
        )
        * np.uint64(
            0x94D049BB133111EB
        )
    )

    return np.uint64(
        z
        ^ (
            z
            >> np.uint64(31)
        )
    )


@njit(inline="always")
def _rng_u01(state):
    """Return updated RNG state and one U(0,1) sample."""

    x = state

    x ^= (
        x
        >> np.uint64(12)
    )

    x ^= (
        x
        << np.uint64(25)
    )

    x ^= (
        x
        >> np.uint64(27)
    )

    state = x

    value = np.uint64(
        x
        * np.uint64(
            0x2545F4914F6CDD1D
        )
    )

    u = (
        float(
            value
            >> np.uint64(11)
        )
        * (
            1.0
            / 9007199254740992.0
        )
    )

    if u < 1.0e-16:
        u = 1.0e-16

    return state, u


@njit(inline="always")
def _rng_normal2(state):
    """Return two independent N(0,1) samples."""

    state, u1 = _rng_u01(
        state
    )

    state, u2 = _rng_u01(
        state
    )

    radius = math.sqrt(
        -2.0
        * math.log(u1)
    )

    angle = (
        2.0
        * math.pi
        * u2
    )

    return (
        state,
        radius
        * math.cos(angle),
        radius
        * math.sin(angle),
    )


@njit(inline="always")
def _sample_cdf(
    state,
    cdf,
):
    """Sample an integer index from a cumulative distribution."""

    state, u = _rng_u01(
        state
    )

    lo = 0
    hi = cdf.size - 1

    while lo < hi:
        mid = (
            lo + hi
        ) // 2

        if u <= cdf[mid]:
            hi = mid
        else:
            lo = mid + 1

    return state, lo


# -----------------------------------------------------------------------------
# Fast homogeneous lookup interpolation
# -----------------------------------------------------------------------------


@njit(inline="always")
def _interp_uniform(
    x,
    xmax,
    values,
):
    """Linearly interpolate a table stored on a uniform [0, xmax] grid."""

    if x <= 0.0:
        return values[0]

    if x >= xmax:
        return values[-1]

    position = (
        x
        * (len(values) - 1)
        / xmax
    )

    i = int(position)

    fraction = (
        position - i
    )

    return (
        values[i]
        + fraction
        * (
            values[i + 1]
            - values[i]
        )
    )


# -----------------------------------------------------------------------------
# Angular Brownian diffusion on S^2
# -----------------------------------------------------------------------------


@njit(inline="always")
def _scatter_direction(
    ux,
    uy,
    uz,
    tau,
    g1,
    g2,
):
    """Apply one geodesic angular-Brownian increment.

    tau = Sigma_tr * ds.

    The two Gaussian samples define a random displacement in the tangent
    plane of the current direction. The tangent displacement is mapped back
    onto the unit sphere through the exponential/geodesic map.
    """

    if tau <= 0.0:
        return (
            ux,
            uy,
            uz,
        )

    transverse = math.sqrt(
        ux * ux
        + uy * uy
    )

    # Construct an orthonormal basis of the tangent plane.
    if transverse > 1.0e-14:
        e1x = -uy / transverse
        e1y = ux / transverse
        e1z = 0.0

        e2x = (
            -uz
            * ux
            / transverse
        )

        e2y = (
            -uz
            * uy
            / transverse
        )

        e2z = transverse

    else:
        # Direction is effectively parallel to z.
        e1x = 1.0
        e1y = 0.0
        e1z = 0.0

        e2x = 0.0
        e2y = 1.0
        e2z = 0.0

    scale = math.sqrt(
        tau
    )

    a = scale * g1
    b = scale * g2

    theta = math.sqrt(
        a * a
        + b * b
    )

    if theta < 1.0e-14:
        return (
            ux,
            uy,
            uz,
        )

    sinc = (
        math.sin(theta)
        / theta
    )

    cosine = math.cos(
        theta
    )

    vx = (
        a * e1x
        + b * e2x
    )

    vy = (
        a * e1y
        + b * e2y
    )

    vz = (
        a * e1z
        + b * e2z
    )

    nx = (
        cosine * ux
        + sinc * vx
    )

    ny = (
        cosine * uy
        + sinc * vy
    )

    nz = (
        cosine * uz
        + sinc * vz
    )

    inverse_norm = (
        1.0
        / math.sqrt(
            nx * nx
            + ny * ny
            + nz * nz
        )
    )

    return (
        nx * inverse_norm,
        ny * inverse_norm,
        nz * inverse_norm,
    )


# -----------------------------------------------------------------------------
# Exact segment / voxel dose scoring for homogeneous media
# -----------------------------------------------------------------------------


@njit(inline="always")
def _score_segment_dda(
    tid,
    x,
    y,
    z,
    mx,
    my,
    mz,
    ds,
    remaining_range,
    weight,
    lx,
    ly,
    lz,
    nx,
    ny,
    nz,
    dx,
    dy,
    dz,
    energy_grid_max,
    sigma_r_lookup,
    nuclear_on,
    range_grid_max,
    energy_of_range,
    dose_threads,
):
    """Deposit the CSDA energy loss along one particle segment.

    The segment is split exactly at voxel boundaries. This avoids assigning
    the entire energy loss of a long particle step to only its midpoint voxel.
    """

    s = 0.0
    tiny = 1.0e-12

    while s < ds - 1.0e-14:

        # Move an infinitesimal distance into the current segment so a point
        # lying exactly on a voxel boundary is assigned unambiguously.
        sample_s = min(
            ds,
            s + tiny,
        )

        qx = (
            x
            + mx * sample_s
        )

        qy = (
            y
            + my * sample_s
        )

        qz = (
            z
            + mz * sample_s
        )

        ix = int(
            math.floor(
                (
                    qx
                    + 0.5 * lx
                )
                / dx
            )
        )

        iy = int(
            math.floor(
                (
                    qy
                    + 0.5 * ly
                )
                / dy
            )
        )

        iz = int(
            math.floor(
                qz / dz
            )
        )

        if (
            ix < 0
            or ix >= nx
            or iy < 0
            or iy >= ny
            or iz < 0
            or iz >= nz
        ):
            break

        px = (
            x
            + mx * s
        )

        py = (
            y
            + my * s
        )

        pz = (
            z
            + mz * s
        )

        segment_length = (
            ds - s
        )

        # -----------------------------------------------------
        # Distance to next x boundary
        # -----------------------------------------------------

        if mx > 1.0e-14:

            hx = (
                (
                    -0.5 * lx
                    + (ix + 1) * dx
                    - px
                )
                / mx
            )

            if (
                tiny < hx
                < segment_length
            ):
                segment_length = hx

        elif mx < -1.0e-14:

            hx = (
                (
                    -0.5 * lx
                    + ix * dx
                    - px
                )
                / mx
            )

            if (
                tiny < hx
                < segment_length
            ):
                segment_length = hx

        # -----------------------------------------------------
        # Distance to next y boundary
        # -----------------------------------------------------

        if my > 1.0e-14:

            hy = (
                (
                    -0.5 * ly
                    + (iy + 1) * dy
                    - py
                )
                / my
            )

            if (
                tiny < hy
                < segment_length
            ):
                segment_length = hy

        elif my < -1.0e-14:

            hy = (
                (
                    -0.5 * ly
                    + iy * dy
                    - py
                )
                / my
            )

            if (
                tiny < hy
                < segment_length
            ):
                segment_length = hy

        # -----------------------------------------------------
        # Distance to next z boundary
        # -----------------------------------------------------

        if mz > 1.0e-14:

            hz = (
                (
                    (iz + 1) * dz
                    - pz
                )
                / mz
            )

            if (
                tiny < hz
                < segment_length
            ):
                segment_length = hz

        elif mz < -1.0e-14:

            hz = (
                (
                    iz * dz
                    - pz
                )
                / mz
            )

            if (
                tiny < hz
                < segment_length
            ):
                segment_length = hz

        if segment_length <= tiny:

            segment_length = min(
                ds - s,
                1.0e-10,
            )

            if segment_length <= 0.0:
                break

        # Energy is obtained from remaining CSDA range.
        energy_before = _interp_uniform(
            remaining_range - s,
            range_grid_max,
            energy_of_range,
        )

        energy_after = _interp_uniform(
            remaining_range
            - (
                s
                + segment_length
            ),
            range_grid_max,
            energy_of_range,
        )

        deposited_energy = (
            energy_before
            - energy_after
        )

        flat_index = (
            (
                ix * ny
                + iy
            )
            * nz
            + iz
        )

        if nuclear_on:
            energy_mid = (
                0.5
                * (
                    energy_before
                    + energy_after
                )
            )

            sigma_r = _interp_uniform(
                energy_mid,
                energy_grid_max,
                sigma_r_lookup,
            )

            tau_r = (
                sigma_r
                * segment_length
            )

            if tau_r > 1.0e-12:
                survival = math.exp(
                    -tau_r
                )

                average_weight = (
                    weight
                    * (1.0 - survival)
                    / tau_r
                )
            else:
                survival = 1.0 - tau_r
                average_weight = weight

            dose_threads[
                tid,
                flat_index,
            ] += (
                average_weight
                * deposited_energy
            )

            weight *= survival

        else:
            dose_threads[
                tid,
                flat_index,
            ] += (
                weight
                * deposited_energy
            )

        s += segment_length

    return weight


# -----------------------------------------------------------------------------
# Homogeneous particle transport
# -----------------------------------------------------------------------------


@njit(
    parallel=True,
    fastmath=True,
    cache=True,
)
def transport_homogeneous(
    histories,
    seed,
    physical_protons,
    beam_energy,
    energy_sigma,
    beam_sigma_x,
    beam_sigma_y,
    beam_center_x,
    beam_center_y,
    angular_sigma,
    cutoff_sigma,
    lx,
    ly,
    lz,
    nx,
    ny,
    nz,
    max_step,
    energy_grid_max,
    sigma_tr_lookup,
    sigma_r_lookup,
    range_grid_max,
    energy_of_range,
    scattering_on,
    nuclear_on,
    stop_energy,
    dose_threads,
):
    """Transport independent proton histories through one material."""

    dx = lx / nx
    dy = ly / ny
    dz = lz / nz

    history_weight = (
        physical_protons
        / histories
    )

    for history in prange(
        histories
    ):

        tid = get_thread_id()

        # Every history receives its own deterministic RNG stream.
        state = _splitmix64(
            np.uint64(seed)
            ^ _splitmix64(
                np.uint64(
                    history + 1
                )
            )
        )

        # -----------------------------------------------------
        # Initial transverse beam position
        # -----------------------------------------------------

        while True:

            state, gx, gy = (
                _rng_normal2(
                    state
                )
            )

            if (
                gx * gx
                + gy * gy
                <= cutoff_sigma
                * cutoff_sigma
            ):
                break

        x = (
            beam_center_x
            + beam_sigma_x * gx
        )

        y = (
            beam_center_y
            + beam_sigma_y * gy
        )

        z = 0.0

        # -----------------------------------------------------
        # Initial proton energy
        # -----------------------------------------------------

        if energy_sigma > 0.0:

            while True:

                state, gaussian_energy, _ = (
                    _rng_normal2(
                        state
                    )
                )

                if (
                    abs(
                        gaussian_energy
                    )
                    <= cutoff_sigma
                ):
                    break

            energy_initial = (
                beam_energy
                + energy_sigma
                * gaussian_energy
            )

        else:

            energy_initial = (
                beam_energy
            )

        if (
            energy_initial
            <= stop_energy
        ):
            continue

        energy_initial = min(
            energy_initial,
            energy_grid_max,
        )

        # -----------------------------------------------------
        # Convert initial energy to remaining CSDA range
        # -----------------------------------------------------

        lo = 0
        hi = (
            len(
                energy_of_range
            )
            - 1
        )

        while hi - lo > 1:

            mid = (
                lo + hi
            ) // 2

            if (
                energy_of_range[mid]
                <= energy_initial
            ):
                lo = mid
            else:
                hi = mid

        energy_lo = (
            energy_of_range[lo]
        )

        energy_hi = (
            energy_of_range[hi]
        )

        if energy_hi <= energy_lo:

            fraction = 0.0

        else:

            fraction = (
                energy_initial
                - energy_lo
            ) / (
                energy_hi
                - energy_lo
            )

        remaining_range = (
            range_grid_max
            * (
                lo
                + fraction
            )
            / (
                len(
                    energy_of_range
                )
                - 1
            )
        )

        # -----------------------------------------------------
        # Initial direction
        # -----------------------------------------------------

        if angular_sigma > 0.0:

            state, ga, gb = (
                _rng_normal2(
                    state
                )
            )

            ux = (
                angular_sigma
                * ga
            )

            uy = (
                angular_sigma
                * gb
            )

            uz = 1.0

            inverse_norm = (
                1.0
                / math.sqrt(
                    ux * ux
                    + uy * uy
                    + uz * uz
                )
            )

            ux *= inverse_norm
            uy *= inverse_norm
            uz *= inverse_norm

        else:

            ux = 0.0
            uy = 0.0
            uz = 1.0

        particle_weight = history_weight

        # -----------------------------------------------------
        # Particle history
        # -----------------------------------------------------

        while (
            remaining_range > 0.0
            and 0.0 <= z < lz
        ):

            ds = min(
                max_step,
                remaining_range,
            )

            midpoint_range = (
                remaining_range
                - 0.5 * ds
            )

            new_range = (
                remaining_range
                - ds
            )

            energy_old = (
                _interp_uniform(
                    remaining_range,
                    range_grid_max,
                    energy_of_range,
                )
            )

            energy_mid = (
                _interp_uniform(
                    midpoint_range,
                    range_grid_max,
                    energy_of_range,
                )
            )

            # At very low energy, deposit the remainder locally.
            if (
                energy_old
                <= stop_energy
            ):

                ix = int(
                    math.floor(
                        (
                            x
                            + 0.5 * lx
                        )
                        / dx
                    )
                )

                iy = int(
                    math.floor(
                        (
                            y
                            + 0.5 * ly
                        )
                        / dy
                    )
                )

                iz = int(
                    math.floor(
                        z / dz
                    )
                )

                if (
                    0 <= ix < nx
                    and 0 <= iy < ny
                    and 0 <= iz < nz
                ):

                    flat_index = (
                        (
                            ix * ny
                            + iy
                        )
                        * nz
                        + iz
                    )

                    dose_threads[
                        tid,
                        flat_index,
                    ] += (
                        particle_weight
                        * energy_old
                    )

                break

            # -------------------------------------------------
            # Angular diffusion
            # -------------------------------------------------

            if scattering_on:

                sigma_tr = (
                    _interp_uniform(
                        energy_mid,
                        energy_grid_max,
                        sigma_tr_lookup,
                    )
                )

                tau = (
                    sigma_tr
                    * ds
                )

                state, g1, g2 = (
                    _rng_normal2(
                        state
                    )
                )

                (
                    new_ux,
                    new_uy,
                    new_uz,
                ) = _scatter_direction(
                    ux,
                    uy,
                    uz,
                    tau,
                    g1,
                    g2,
                )

            else:

                new_ux = ux
                new_uy = uy
                new_uz = uz

            # -------------------------------------------------
            # Midpoint/trapezoidal transport direction
            # -------------------------------------------------

            mx = (
                ux + new_ux
            )

            my = (
                uy + new_uy
            )

            mz = (
                uz + new_uz
            )

            direction_norm = (
                math.sqrt(
                    mx * mx
                    + my * my
                    + mz * mz
                )
            )

            if (
                direction_norm
                <= 1.0e-14
            ):

                mx = new_ux
                my = new_uy
                mz = new_uz

            else:

                mx /= direction_norm
                my /= direction_norm
                mz /= direction_norm

            # -------------------------------------------------
            # Dose deposition
            # -------------------------------------------------

            particle_weight = _score_segment_dda(
                tid,
                x,
                y,
                z,
                mx,
                my,
                mz,
                ds,
                remaining_range,
                particle_weight,
                lx,
                ly,
                lz,
                nx,
                ny,
                nz,
                dx,
                dy,
                dz,
                energy_grid_max,
                sigma_r_lookup,
                nuclear_on,
                range_grid_max,
                energy_of_range,
                dose_threads,
            )

            # -------------------------------------------------
            # Advance particle
            # -------------------------------------------------

            x += (
                ds * mx
            )

            y += (
                ds * my
            )

            z += (
                ds * mz
            )

            ux = new_ux
            uy = new_uy
            uz = new_uz

            remaining_range = (
                new_range
            )

            # Proton left through a lateral boundary.
            if (
                x <= -0.5 * lx
                or x >= 0.5 * lx
                or y <= -0.5 * ly
                or y >= 0.5 * ly
            ):
                break


# -----------------------------------------------------------------------------
# Geometry helpers for heterogeneous transport
# -----------------------------------------------------------------------------


@njit(inline="always")
def _voxel_index(
    value,
    minimum,
    spacing,
    n,
):
    """Return voxel index, or -1 when outside the domain."""

    index = int(
        math.floor(
            (value - minimum)
            / spacing
        )
    )

    if (
        index < 0
        or index >= n
    ):
        return -1

    return index


@njit(inline="always")
def _inside_domain(
    x,
    y,
    z,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
):
    return (
        xmin <= x < xmax
        and ymin <= y < ymax
        and zmin <= z < zmax
    )


@njit(inline="always")
def _distance_to_domain_exit(
    x,
    y,
    z,
    ox,
    oy,
    oz,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
):
    """Distance along direction Omega to the outer domain boundary."""

    distance = 1.0e300
    epsilon = 1.0e-15

    if ox > epsilon:
        distance = min(
            distance,
            (xmax - x) / ox,
        )

    elif ox < -epsilon:
        distance = min(
            distance,
            (xmin - x) / ox,
        )

    if oy > epsilon:
        distance = min(
            distance,
            (ymax - y) / oy,
        )

    elif oy < -epsilon:
        distance = min(
            distance,
            (ymin - y) / oy,
        )

    if oz > epsilon:
        distance = min(
            distance,
            (zmax - z) / oz,
        )

    elif oz < -epsilon:
        distance = min(
            distance,
            (zmin - z) / oz,
        )

    return max(
        distance,
        0.0,
    )


@njit(inline="always")
def _distance_to_next_voxel_boundary(
    x,
    y,
    z,
    ox,
    oy,
    oz,
    ix,
    iy,
    iz,
    xmin,
    ymin,
    zmin,
    dx,
    dy,
    dz,
):
    """Distance to the next Cartesian voxel face.

    Restricting a heterogeneous transport step to this distance prevents one
    step from silently using the material coefficients of the wrong voxel.
    """

    distance = 1.0e300
    epsilon = 1.0e-15

    if ox > epsilon:
        boundary = (
            xmin
            + (ix + 1) * dx
        )

        distance = min(
            distance,
            (boundary - x) / ox,
        )

    elif ox < -epsilon:
        boundary = (
            xmin
            + ix * dx
        )

        distance = min(
            distance,
            (boundary - x) / ox,
        )

    if oy > epsilon:
        boundary = (
            ymin
            + (iy + 1) * dy
        )

        distance = min(
            distance,
            (boundary - y) / oy,
        )

    elif oy < -epsilon:
        boundary = (
            ymin
            + iy * dy
        )

        distance = min(
            distance,
            (boundary - y) / oy,
        )

    if oz > epsilon:
        boundary = (
            zmin
            + (iz + 1) * dz
        )

        distance = min(
            distance,
            (boundary - z) / oz,
        )

    elif oz < -epsilon:
        boundary = (
            zmin
            + iz * dz
        )

        distance = min(
            distance,
            (boundary - z) / oz,
        )

    return max(
        distance,
        0.0,
    )

@njit(inline="always")
def _midpoint_direction(
    old_ux,
    old_uy,
    old_uz,
    new_ux,
    new_uy,
    new_uz,
):
    """Normalize the average of two particle directions."""

    mx = old_ux + new_ux
    my = old_uy + new_uy
    mz = old_uz + new_uz

    norm = math.sqrt(
        mx * mx
        + my * my
        + mz * mz
    )

    if norm <= 1.0e-14:
        return (
            old_ux,
            old_uy,
            old_uz,
        )

    return (
        mx / norm,
        my / norm,
        mz / norm,
    )


# -----------------------------------------------------------------------------
# Heterogeneous particle transport
# -----------------------------------------------------------------------------


@njit(
    parallel=True,
    fastmath=True,
    cache=True,
)
def transport_heterogeneous(
    histories,
    seed,
    history_weight,
    spatial_cdf,
    beam_energy,
    energy_sigma,
    energy_min,
    energy_max,
    max_step,
    max_energy_loss,
    max_scattering_depth,
    scattering_on,
    nuclear_on,
    material_ids,
    nx,
    ny,
    nz,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
    dx,
    dy,
    dz,
    s_energy,
    s_value,
    s_slope,
    s_len,
    t_energy,
    t_value,
    t_len,
    r_energy,
    r_value,
    r_len,
    stop_energy,
    dose_threads,
):
    """Transport particles through a voxelized heterogeneous geometry."""

    for history in prange(
        histories
    ):

        tid = get_thread_id()

        state = _splitmix64(
            np.uint64(seed)
            ^ _splitmix64(
                np.uint64(
                    history + 1
                )
            )
        )

        # -----------------------------------------------------
        # Sample the exact discrete transverse BeamInflow
        # -----------------------------------------------------

        state, flat_xy = _sample_cdf(
            state,
            spatial_cdf,
        )

        ix_initial = (
            flat_xy // ny
        )

        iy_initial = (
            flat_xy
            - ix_initial * ny
        )

        state, rx = _rng_u01(
            state
        )

        state, ry = _rng_u01(
            state
        )

        x = (
            xmin
            + (
                ix_initial
                + rx
            )
            * dx
        )

        y = (
            ymin
            + (
                iy_initial
                + ry
            )
            * dy
        )

        # Tiny positive offset prevents the source from sitting
        # numerically exactly outside z_min.
        z = (
            zmin
            + 1.0e-13
        )

        # -----------------------------------------------------
        # Sample truncated Gaussian source energy
        # -----------------------------------------------------

        if energy_sigma > 0.0:

            while True:

                state, gaussian_energy, _ = (
                    _rng_normal2(
                        state
                    )
                )

                energy = (
                    beam_energy
                    + energy_sigma
                    * gaussian_energy
                )

                if (
                    energy_min
                    <= energy
                    <= energy_max
                ):
                    break

        else:

            energy = beam_energy

            if energy > energy_max:
                energy = energy_max

        # Current matched thesis inflow:
        # perfectly collimated along +z.
        ux = 0.0
        uy = 0.0
        uz = 1.0

        survival_weight = 1.0

        # -----------------------------------------------------
        # Particle history
        # -----------------------------------------------------

        while (
            energy > energy_min
            and _inside_domain(
                x,
                y,
                z,
                xmin,
                xmax,
                ymin,
                ymax,
                zmin,
                zmax,
            )
        ):

            # Probe an infinitesimal distance in the direction of travel.
            # This resolves the material unambiguously when the particle
            # lies exactly on a voxel interface.
            probe = 1.0e-12

            sample_x = x + probe * ux
            sample_y = y + probe * uy
            sample_z = z + probe * uz

            ix = _voxel_index(
                sample_x,
                xmin,
                dx,
                nx,
            )

            iy = _voxel_index(
                sample_y,
                ymin,
                dy,
                ny,
            )

            iz = _voxel_index(
                sample_z,
                zmin,
                dz,
                nz,
            )

            if (
                ix < 0
                or iy < 0
                or iz < 0
            ):
                break

            material_id = int(
                material_ids[
                    ix,
                    iy,
                    iz,
                ]
            )

            # -------------------------------------------------
            # Local material physics
            # -------------------------------------------------

            stopping_power = (
                _pchip_material_value(
                    energy,
                    material_id,
                    s_energy,
                    s_value,
                    s_slope,
                    s_len,
                )
            )

            sigma_tr = 0.0

            if scattering_on:

                sigma_tr = (
                    _linear_material_value(
                        energy,
                        material_id,
                        t_energy,
                        t_value,
                        t_len,
                    )
                )

            if stopping_power <= 0.0:
                break

            # -------------------------------------------------
            # Construct admissible step length
            # -------------------------------------------------

            ds = max_step

            # Do not lose too much energy in one step.
            ds = min(
                ds,
                max_energy_loss
                / stopping_power,
            )

            # Do not step below the lower energy boundary.
            ds = min(
                ds,
                (
                    energy
                    - energy_min
                )
                / stopping_power,
            )

            # Keep angular Brownian increments small.
            if sigma_tr > 0.0:

                ds = min(
                    ds,
                    max_scattering_depth
                    / sigma_tr,
                )

            # Never leave the computational domain unnoticed.
            domain_distance = (
                _distance_to_domain_exit(
                    x,
                    y,
                    z,
                    ux,
                    uy,
                    uz,
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    zmin,
                    zmax,
                )
            )

            ds = min(
                ds,
                domain_distance,
            )

            # Crucially: do not cross a material/voxel interface
            # using coefficients from the old voxel.
            voxel_distance = (
                _distance_to_next_voxel_boundary(
                    x,
                    y,
                    z,
                    ux,
                    uy,
                    uz,
                    ix,
                    iy,
                    iz,
                    xmin,
                    ymin,
                    zmin,
                    dx,
                    dy,
                    dz,
                )
            )

            if voxel_distance > 1.0e-13:

                ds = min(
                    ds,
                    voxel_distance,
                )

            if ds <= 1.0e-14:
                break

            # -------------------------------------------------
            # Midpoint material / energy correction
            # -------------------------------------------------

            approximate_mid_energy = max(
                energy_min,
                energy
                - 0.5
                * stopping_power
                * ds,
            )

            midpoint_x = (
                x
                + 0.5
                * ds
                * ux
            )

            midpoint_y = (
                y
                + 0.5
                * ds
                * uy
            )

            midpoint_z = (
                z
                + 0.5
                * ds
                * uz
            )

            mid_ix = _voxel_index(
                midpoint_x,
                xmin,
                dx,
                nx,
            )

            mid_iy = _voxel_index(
                midpoint_y,
                ymin,
                dy,
                ny,
            )

            mid_iz = _voxel_index(
                midpoint_z,
                zmin,
                dz,
                nz,
            )

            if (
                mid_ix < 0
                or mid_iy < 0
                or mid_iz < 0
            ):
                break

            midpoint_material = int(
                material_ids[
                    mid_ix,
                    mid_iy,
                    mid_iz,
                ]
            )

            stopping_mid = (
                _pchip_material_value(
                    approximate_mid_energy,
                    midpoint_material,
                    s_energy,
                    s_value,
                    s_slope,
                    s_len,
                )
            )

            sigma_mid = 0.0

            if scattering_on:

                sigma_mid = (
                    _linear_material_value(
                        approximate_mid_energy,
                        midpoint_material,
                        t_energy,
                        t_value,
                        t_len,
                    )
                )

            sigma_r_mid = 0.0

            if nuclear_on:

                sigma_r_mid = (
                    _linear_material_value(
                        approximate_mid_energy,
                        midpoint_material,
                        r_energy,
                        r_value,
                        r_len,
                    )
                )

            if stopping_mid <= 0.0:
                break

            # Re-apply local controls with midpoint coefficients.
            ds = min(
                ds,
                max_energy_loss
                / stopping_mid,
            )

            ds = min(
                ds,
                (
                    energy
                    - energy_min
                )
                / stopping_mid,
            )

            if sigma_mid > 0.0:

                ds = min(
                    ds,
                    max_scattering_depth
                    / sigma_mid,
                )

            if ds <= 1.0e-14:
                break

            # -------------------------------------------------
            # Angular diffusion
            # -------------------------------------------------

            old_ux = ux
            old_uy = uy
            old_uz = uz

            if scattering_on:

                state, g1, g2 = (
                    _rng_normal2(
                        state
                    )
                )

                tau = (
                    sigma_mid
                    * ds
                )

                (
                    new_ux,
                    new_uy,
                    new_uz,
                ) = _scatter_direction(
                    old_ux,
                    old_uy,
                    old_uz,
                    tau,
                    g1,
                    g2,
                )

            else:

                new_ux = old_ux
                new_uy = old_uy
                new_uz = old_uz

            (
                mx,
                my,
                mz,
            ) = _midpoint_direction(
                old_ux,
                old_uy,
                old_uz,
                new_ux,
                new_uy,
                new_uz,
            )

            # -------------------------------------------------
            # Re-check voxel boundary using actual midpoint direction
            # -------------------------------------------------

            actual_voxel_distance = (
                _distance_to_next_voxel_boundary(
                    x,
                    y,
                    z,
                    mx,
                    my,
                    mz,
                    ix,
                    iy,
                    iz,
                    xmin,
                    ymin,
                    zmin,
                    dx,
                    dy,
                    dz,
                )
            )

            if (
                actual_voxel_distance > 1.0e-13
                and actual_voxel_distance < ds
            ):

                ds = (
                    actual_voxel_distance
                )

                # Scattering depends on ds, so recompute it using
                # the same Gaussian random increment.
                if scattering_on:

                    tau = (
                        sigma_mid
                        * ds
                    )

                    (
                        new_ux,
                        new_uy,
                        new_uz,
                    ) = _scatter_direction(
                        old_ux,
                        old_uy,
                        old_uz,
                        tau,
                        g1,
                        g2,
                    )

                    (
                        mx,
                        my,
                        mz,
                    ) = _midpoint_direction(
                        old_ux,
                        old_uy,
                        old_uz,
                        new_ux,
                        new_uy,
                        new_uz,
                    )

            # -------------------------------------------------
            # Energy loss and dose scoring
            # -------------------------------------------------

            deposited_energy = (
                stopping_mid
                * ds
            )

            maximum_deposition = (
                energy
                - energy_min
            )

            if (
                deposited_energy
                > maximum_deposition
            ):

                deposited_energy = (
                    maximum_deposition
                )

                ds = (
                    deposited_energy
                    / stopping_mid
                )

            if deposited_energy <= 0.0:
                break

            # Because ds is restricted to the current voxel,
            # deposit into that voxel directly.
            flat_index = (
                (
                    ix * ny
                    + iy
                )
                * nz
                + iz
            )

            if nuclear_on:
                tau_r = (
                    sigma_r_mid
                    * ds
                )

                if tau_r > 1.0e-12:
                    survival = math.exp(
                        -tau_r
                    )

                    average_survival = (
                        survival_weight
                        * (1.0 - survival)
                        / tau_r
                    )
                else:
                    survival = 1.0 - tau_r
                    average_survival = survival_weight

                dose_threads[
                    tid,
                    flat_index,
                ] += (
                    history_weight
                    * average_survival
                    * deposited_energy
                )

                survival_weight *= survival

            else:
                dose_threads[
                    tid,
                    flat_index,
                ] += (
                    history_weight
                    * deposited_energy
                )

            # -------------------------------------------------
            # Advance particle
            # -------------------------------------------------

            x += (
                ds * mx
            )

            y += (
                ds * my
            )

            z += (
                ds * mz
            )

            energy -= (
                deposited_energy
            )

            ux = new_ux
            uy = new_uy
            uz = new_uz

            # -------------------------------------------------
            # Residual low-energy deposition
            # -------------------------------------------------

            if (
                energy <= stop_energy
                and energy > energy_min
            ):

                if _inside_domain(
                    x,
                    y,
                    z,
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    zmin,
                    zmax,
                ):

                    residual_ix = _voxel_index(
                        x,
                        xmin,
                        dx,
                        nx,
                    )

                    residual_iy = _voxel_index(
                        y,
                        ymin,
                        dy,
                        ny,
                    )

                    residual_iz = _voxel_index(
                        z,
                        zmin,
                        dz,
                        nz,
                    )

                    if (
                        residual_ix >= 0
                        and residual_iy >= 0
                        and residual_iz >= 0
                    ):

                        residual_flat = (
                            (
                                residual_ix
                                * ny
                                + residual_iy
                            )
                            * nz
                            + residual_iz
                        )

                        dose_threads[
                            tid,
                            residual_flat,
                        ] += (
                            history_weight
                            * survival_weight
                            * (
                                energy
                                - energy_min
                            )
                        )

                break