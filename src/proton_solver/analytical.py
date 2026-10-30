from functools import lru_cache

import numpy as np

from .config import SolverConfig
from .numerics.boundary import BeamInflow
from .numerics.grid import Grid
from .physics.materials import MaterialMap, density
from .physics.stopping_power import _stopping_power_interpolant
from .enums import Scattering

MEV_TO_JOULE = 1.602176634e-13
RANGE_TABLE_POINTS = 200_000

# Nodes of the trapezoidal quadrature over the incident energy spectrum.
# The deposited energy per cell has kinks in the incident energy (where a
# proton starts or stops inside the cell), so the quadrature converges only
# about second order. With 2000 nodes (spacing 0.078 MeV, about 0.14 mm of
# range per node) the integrated depth dose had a relative error of 1.2e-3 at
# the peak and R80 was 10 um too short. 128000 nodes reduce this to below
# 3e-6 and 0.01 um, at about 1 s per distinct material column.
ENERGY_QUADRATURE_POINTS = 128_000


def _pchip_stopping_power(material, energy):
    """
    Evaluate the stopping power with the same shape-preserving cubic
    (PCHIP) interpolant as the deterministic solver
    (physics.stopping_power.stopping_power) and the BFP Monte Carlo.

    Vectorized version of that function: same nodes, same nodal slopes,
    same interval search and Hermite basis.
    """

    nodes, values, slopes = _stopping_power_interpolant(material)

    energy = np.clip(
        np.asarray(energy, dtype=float),
        nodes[0],
        nodes[-1],
    )

    i = np.searchsorted(nodes, energy, side="right") - 1
    i = np.clip(i, 0, len(nodes) - 2)

    h = nodes[i + 1] - nodes[i]
    t = (energy - nodes[i]) / h
    t2 = t * t
    t3 = t2 * t

    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2

    return (
        h00 * values[i]
        + h10 * h * slopes[i]
        + h01 * values[i + 1]
        + h11 * h * slopes[i + 1]
    )


@lru_cache(maxsize=None)
def range_table(material):
    """
    Build a dense CSDA range table

        R(E) = integral_0^E 1 / S(epsilon) d epsilon.

    A dense geometric energy grid gives accurate inverse
    interpolation near the proton stopping point. The stopping power is
    evaluated with the PCHIP interpolant of the deterministic solver and the
    BFP Monte Carlo, so that all three solutions use identical S(E).
    (Earlier versions interpolated S(E) linearly between table nodes.)
    """

    source_energy, _, _ = _stopping_power_interpolant(material)

    energy_min = max(
        float(source_energy[0]),
        1.0e-12,
    )

    energy_max = float(source_energy[-1])

    energy = np.geomspace(
        energy_min,
        energy_max,
        RANGE_TABLE_POINTS,
    )

    stopping = _pchip_stopping_power(material, energy)

    range_values = np.zeros_like(energy)

    range_values[1:] = np.cumsum(
        0.5
        * (
            1.0 / stopping[1:]
            + 1.0 / stopping[:-1]
        )
        * np.diff(energy)
    )

    return energy, stopping, range_values


def range_from_energy(material, energy):
    """Return the CSDA range corresponding to a proton energy."""

    energy_table, _, range_values = range_table(material)

    return np.interp(
        energy,
        energy_table,
        range_values,
        left=0.0,
        right=range_values[-1],
    )


def energy_from_range(material, range_value):
    """Return the proton energy corresponding to a CSDA range."""

    energy_table, _, range_values = range_table(material)

    return np.interp(
        range_value,
        range_values,
        energy_table,
        left=0.0,
        right=energy_table[-1],
    )


def propagate_energy(energy_in, material, distance):
    """
    Propagate proton energies through a homogeneous material.

    Uses

        R(E_out) = R(E_in) - distance.
    """

    values = np.asarray(
        energy_in,
        dtype=float,
    )

    range_in = range_from_energy(
        material,
        values,
    )

    range_out = range_in - distance

    energy_out = np.where(
        range_out > 0.0,
        energy_from_range(
            material,
            np.maximum(range_out, 0.0),
        ),
        0.0,
    )

    if np.ndim(energy_in) == 0:
        return float(energy_out)

    return energy_out


def analytical_column_dose(
    material_column,
    config: SolverConfig,
    dz: float,
    num_energy_points: int = ENERGY_QUADRATURE_POINTS,
):
    """
    Return the finite-volume analytical CSDA dose along one z-column.

    For every incident energy, the energy deposited inside each cell is

        delta_E = E_in - E_out.

    Averaging this over the incident energy spectrum gives the
    deposited energy per incident transverse fluence.
    """

    if num_energy_points < 2:
        raise ValueError(
            "num_energy_points must be at least 2."
        )

    beam = config.beam
    numerics = config.numerics

    energy_start = (
        beam.energy
        + numerics.energy_window_sigma
        * beam.energy_spread
    )

    # Same truncated energy interval used by the numerical solver
    source_energy = np.linspace(
        numerics.energy_min,
        energy_start,
        num_energy_points,
    )

    energy_spectrum = (
        np.exp(
            -(
                source_energy
                - beam.energy
            ) ** 2
            / (
                2.0
                * beam.energy_spread**2
            )
        )
        / (
            np.sqrt(2.0 * np.pi)
            * beam.energy_spread
        )
    )

    dose_factor = np.zeros(
        len(material_column),
        dtype=float,
    )

    # Energy at the entrance of the current spatial cell
    energy_at_entrance = source_energy.copy()

    for iz, material in enumerate(material_column):

        if not np.any(energy_at_entrance > 0.0):
            break

        energy_at_exit = propagate_energy(
            energy_at_entrance,
            material,
            dz,
        )

        energy_lost = np.maximum(
            energy_at_entrance
            - energy_at_exit,
            0.0,
        )

        deposited_energy = np.trapezoid(
            energy_spectrum * energy_lost,
            source_energy,
        )

        dose_factor[iz] = (
            deposited_energy
            * MEV_TO_JOULE
            / (
                density(material)
                * dz
            )
        )

        energy_at_entrance = energy_at_exit

    return dose_factor


def analytical_csda_dose(
    config: SolverConfig,
    material_map: MaterialMap,
    grid: Grid,
    inflow: BeamInflow,
    num_energy_points: int = ENERGY_QUADRATURE_POINTS,
):
    """
    Return analytical CSDA + straight-streaming dose.

    Identical material columns are cached, making homogeneous
    geometries and slab geometries inexpensive to evaluate.
    """

    ids = (
        material_map.ids.get()
        if hasattr(material_map.ids, "get")
        else np.asarray(material_map.ids)
    )

    spatial = (
        inflow.spatial.get()
        if hasattr(inflow.spatial, "get")
        else np.asarray(inflow.spatial)
    )

    dose = np.zeros(
        grid.shape,
        dtype=float,
    )

    column_cache = {}

    for ix in range(grid.nx):
        for iy in range(grid.ny):

            column = tuple(
                material_map.materials[int(material_id)]
                for material_id in ids[ix, iy, :]
            )

            if column not in column_cache:
                column_cache[column] = analytical_column_dose(
                    column,
                    config,
                    grid.dz,
                    num_energy_points,
                )

            dose[ix, iy, :] = (
                spatial[ix, iy]
                * column_cache[column]
            )

    return dose


def analytical_available(config: SolverConfig) -> bool:
    """
    Return whether the straight-streaming analytical solution applies.
    """

    return (
        config.physics.scattering is Scattering.OFF
        and not config.physics.nuclear
        and config.beam.angular_spread == 0.0
    )