"""Public runner for the BFP particle Monte Carlo reference."""

from __future__ import annotations

import math
import os
import time

import numba
import numpy as np

from ..enums import Scattering

from .cache import (
    load_bfp_mc_cache,
    save_bfp_mc_cache,
)

from .kernels import (
    transport_heterogeneous,
    transport_homogeneous,
)

from .materials import (
    MEV_TO_J,
    _build_lookups,
    _density_kg_m3,
    _pack_material_tables,
    _to_numpy,
)


def _configure_threads(requested_threads: int) -> int:
    """Configure the number of Numba worker threads."""

    available = os.cpu_count() or 1

    nthreads = min(
        requested_threads,
        available,
    )

    numba.set_num_threads(
        nthreads
    )

    return nthreads


def _run_homogeneous(
    config,
    material_map,
    grid,
):
    """Run the original homogeneous range-based BFP Monte Carlo."""

    refs = config.references
    beam = config.beam
    geometry = config.geometry

    nx, ny, nz = grid.shape

    lx, ly, lz = geometry.size

    material = material_map.materials[0]

    material_name = getattr(
        material,
        "value",
        str(material),
    )

    # Cover the complete truncated source-energy support.
    lookup_max = (
        beam.energy
        + beam.cutoff_sigma
        * beam.energy_spread
        + 1.0e-9
    )

    (
        energy_grid,
        sigma_tr,
        sigma_r,
        range_grid,
        energy_of_range,
    ) = _build_lookups(
        material_name,
        lookup_max,
        nuclear_on=bool(
            config.physics.nuclear
        ),
    )

    density_value = (
        _density_kg_m3(
            material_name
        )
    )

    nthreads = _configure_threads(
        refs.bfp_mc_threads
    )

    dose_threads = np.zeros(
        (
            nthreads,
            nx * ny * nz,
        ),
        dtype=np.float64,
    )

    kernel_args = (
        int(
            refs.bfp_mc_histories
        ),
        int(
            refs.bfp_mc_seed
        ),
        float(
            beam.n_protons
        ),
        float(
            beam.energy
        ),
        float(
            beam.energy_spread
        ),
        float(
            beam.sigma[0]
        ),
        float(
            beam.sigma[1]
        ),
        float(
            beam.center[0]
        ),
        float(
            beam.center[1]
        ),
        float(
            beam.angular_spread
        ),
        float(
            beam.cutoff_sigma
        ),
        float(lx),
        float(ly),
        float(lz),
        int(nx),
        int(ny),
        int(nz),
        float(
            refs.bfp_mc_max_step_mm
        )
        * 1.0e-3,
        float(
            energy_grid[-1]
        ),
        sigma_tr,
        sigma_r,
        float(
            range_grid[-1]
        ),
        energy_of_range,
        (
            config.physics.scattering
            is not Scattering.OFF
        ),
        bool(
            config.physics.nuclear
        ),
        float(
            refs.bfp_mc_stop_energy_mev
        ),
        dose_threads,
    )

    # ---------------------------------------------------------
    # JIT warm-up
    # ---------------------------------------------------------
    #
    # Do not include Numba compilation in the reported MC runtime.

    warm_args = list(
        kernel_args
    )

    warm_args[0] = 1

    transport_homogeneous(
        *warm_args
    )

    dose_threads.fill(
        0.0
    )

    # ---------------------------------------------------------
    # Timed calculation
    # ---------------------------------------------------------

    start = time.perf_counter()

    transport_homogeneous(
        *kernel_args
    )

    runtime = (
        time.perf_counter()
        - start
    )

    deposited_mev = (
        dose_threads
        .sum(axis=0)
        .reshape(
            nx,
            ny,
            nz,
        )
    )

    voxel_mass = (
        density_value
        * grid.cell_volume
    )

    dose = (
        deposited_mev
        * MEV_TO_J
        / voxel_mass
    )

    return (
        dose,
        runtime,
    )


def _energy_probability_mass(
    mean_energy: float,
    sigma_energy: float,
    energy_min: float,
    energy_max: float,
) -> float:
    """Probability mass of the source Gaussian inside the energy march."""

    if sigma_energy <= 0.0:

        if (
            energy_min
            <= mean_energy
            <= energy_max
        ):
            return 1.0

        return 0.0

    root_two = math.sqrt(
        2.0
    )

    lower = (
        energy_min
        - mean_energy
    ) / sigma_energy

    upper = (
        energy_max
        - mean_energy
    ) / sigma_energy

    return (
        0.5
        * (
            math.erf(
                upper
                / root_two
            )
            - math.erf(
                lower
                / root_two
            )
        )
    )


def _run_heterogeneous(
    config,
    material_map,
    grid,
    inflow,
):
    """Run the geometry-aware BFP Monte Carlo."""

    refs = config.references
    beam = config.beam
    numerics = config.numerics

    if beam.angular_spread != 0.0:
        raise ValueError(
            "Heterogeneous BFP Monte Carlo currently requires "
            "beam.angular_spread == 0.0."
        )

    nx, ny, nz = grid.shape

    dx = float(
        grid.dx
    )

    dy = float(
        grid.dy
    )

    dz = float(
        grid.dz
    )

    # ---------------------------------------------------------
    # Computational-domain bounds
    # ---------------------------------------------------------

    x_values = _to_numpy(
        grid.x
    )

    y_values = _to_numpy(
        grid.y
    )

    z_values = _to_numpy(
        grid.z
    )

    xmin = float(
        x_values[0]
        - 0.5 * dx
    )

    ymin = float(
        y_values[0]
        - 0.5 * dy
    )

    zmin = float(
        z_values[0]
        - 0.5 * dz
    )

    xmax = (
        xmin
        + nx * dx
    )

    ymax = (
        ymin
        + ny * dy
    )

    zmax = (
        zmin
        + nz * dz
    )

    # ---------------------------------------------------------
    # Exact voxelized material geometry
    # ---------------------------------------------------------

    material_ids = (
        _to_numpy(
            material_map.ids
        )
        .astype(
            np.int32,
            copy=False,
        )
    )

    if material_ids.shape != (
        nx,
        ny,
        nz,
    ):
        raise ValueError(
            "Material-map shape does not match the BFP MC grid: "
            f"{material_ids.shape} != {(nx, ny, nz)}."
        )

    (
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
        densities,
    ) = _pack_material_tables(
        material_map,
        nuclear_on=bool(
            config.physics.nuclear
        ),
    )

    # ---------------------------------------------------------
    # Spatial source distribution
    # ---------------------------------------------------------
    #
    # Use exactly the discrete transverse BeamInflow that was
    # constructed for the deterministic finite-volume solver.

    spatial = (
        _to_numpy(
            inflow.spatial
        )
        .astype(
            np.float64,
            copy=False,
        )
    )

    cell_weights = (
        spatial.ravel()
        * dx
        * dy
    )

    spatial_mass = float(
        cell_weights.sum()
    )

    if (
        not math.isfinite(
            spatial_mass
        )
        or spatial_mass <= 0.0
    ):
        raise ValueError(
            "BFP Monte Carlo received an empty transverse beam inflow."
        )

    probabilities = (
        cell_weights
        / spatial_mass
    )

    spatial_cdf = np.cumsum(
        probabilities
    )

    spatial_cdf[-1] = 1.0

    # ---------------------------------------------------------
    # Energy support
    # ---------------------------------------------------------

    energy_min = float(
        numerics.energy_min
    )

    if beam.energy_spread > 0.0:

        energy_max = float(
            beam.energy
            + numerics.energy_window_sigma
            * beam.energy_spread
        )

    else:

        energy_max = float(
            beam.energy
        )

    if energy_max <= energy_min:
        raise ValueError(
            "Invalid BFP Monte Carlo energy interval: "
            f"[{energy_min}, {energy_max}]."
        )

    energy_mass = (
        _energy_probability_mass(
            float(
                beam.energy
            ),
            float(
                beam.energy_spread
            ),
            energy_min,
            energy_max,
        )
    )

    if energy_mass <= 0.0:
        raise ValueError(
            "The BFP Monte Carlo source has no probability mass "
            "inside the deterministic energy interval."
        )

    histories = int(
        refs.bfp_mc_histories
    )

    # Each simulated history represents the same physical source
    # mass as the deterministic finite-volume inflow.
    history_weight = (
        spatial_mass
        * energy_mass
        / histories
    )

    # ---------------------------------------------------------
    # Thread-private dose tallies
    # ---------------------------------------------------------

    nthreads = _configure_threads(
        refs.bfp_mc_threads
    )

    dose_threads = np.zeros(
        (
            nthreads,
            nx * ny * nz,
        ),
        dtype=np.float64,
    )

    kernel_args = (
        histories,
        int(
            refs.bfp_mc_seed
        ),
        float(
            history_weight
        ),
        spatial_cdf,
        float(
            beam.energy
        ),
        float(
            beam.energy_spread
        ),
        energy_min,
        energy_max,
        float(
            refs.bfp_mc_max_step_mm
        )
        * 1.0e-3,
        float(
            refs.bfp_mc_max_energy_loss_mev
        ),
        float(
            refs.bfp_mc_max_scattering_optical_depth
        ),
        (
            config.physics.scattering
            is not Scattering.OFF
        ),
        bool(
            config.physics.nuclear
        ),
        material_ids,
        int(nx),
        int(ny),
        int(nz),
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
        float(
            refs.bfp_mc_stop_energy_mev
        ),
        dose_threads,
    )

    # ---------------------------------------------------------
    # JIT warm-up
    # ---------------------------------------------------------

    warm_args = list(
        kernel_args
    )

    warm_args[0] = 1

    transport_heterogeneous(
        *warm_args
    )

    dose_threads.fill(
        0.0
    )

    # ---------------------------------------------------------
    # Timed calculation
    # ---------------------------------------------------------

    start = time.perf_counter()

    transport_heterogeneous(
        *kernel_args
    )

    runtime = (
        time.perf_counter()
        - start
    )

    # ---------------------------------------------------------
    # Convert deposited MeV to Gy
    # ---------------------------------------------------------

    deposited_mev = (
        dose_threads
        .sum(axis=0)
        .reshape(
            nx,
            ny,
            nz,
        )
    )

    density_field = (
        densities[
            material_ids
        ]
    )

    voxel_mass = (
        density_field
        * grid.cell_volume
    )

    dose = (
        deposited_mev
        * MEV_TO_J
        / voxel_mass
    )

    return (
        dose,
        runtime,
    )


def run_bfp_mc_reference(
    config,
    material_map,
    grid,
    inflow,
):
    """Return ``(dose, runtime_seconds)`` for the matched BFP MC reference.

    Homogeneous media use the range-based BFP Monte Carlo. Heterogeneous
    media use the same voxelized MaterialMap and transverse BeamInflow as the
    deterministic finite-volume solver. Optional nuclear attenuation removes
    primary-proton weight continuously using the tabulated Sigma_r(E).
    """

    if config.physics.scattering not in (
        Scattering.OFF,
        Scattering.WENTZEL,
    ):
        raise NotImplementedError(
            "Reference.BFP_MC currently supports only "
            "Scattering.OFF and Scattering.WENTZEL."
        )

    # ---------------------------------------------------------
    # Persistent reference cache
    # ---------------------------------------------------------
    #
    # The cache key deliberately contains only inputs that affect the
    # BFP Monte Carlo reference. Deterministic settings such as First
    # Order versus WENO-Z therefore reuse the same particle reference.

    if not config.references.bfp_mc_force_rerun:
        cached = load_bfp_mc_cache(
            config,
            material_map,
            grid,
            inflow,
        )

        if cached is not None:
            dose, runtime, signature = cached

            print(
                f"Using cached BFP MC result: {signature}"
            )

            return dose, runtime

        print("BFP MC cache miss; running reference.")
    else:
        print("Forcing BFP MC rerun.")

    # Keep the established homogeneous reference completely separate
    # from the geometry-aware transport path.
    if len(
        material_map.materials
    ) == 1:
        dose, runtime = _run_homogeneous(
            config,
            material_map,
            grid,
        )
    else:
        dose, runtime = _run_heterogeneous(
            config,
            material_map,
            grid,
            inflow,
        )

    signature = save_bfp_mc_cache(
        config,
        material_map,
        grid,
        inflow,
        dose,
        runtime,
    )

    print(
        f"Cached BFP MC result: {signature}"
    )

    return dose, runtime
