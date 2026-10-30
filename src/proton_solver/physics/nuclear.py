"""Primary-proton nuclear attenuation for the reduced transport model.

The tabulated coefficient is the macroscopic Geant4 proton-inelastic removal
coefficient ``Sigma_r(E)`` [1/m].  A nuclear inelastic interaction removes the
particle from the primary-proton population; nuclear secondaries are not
transported by this reduced model.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..enums import CoefficientInterpolation, Material
from .coefficient_interpolation import interpolate_coefficient, pchip_slopes
from .materials import MaterialMap
from .stopping_power import stopping_power, stopping_power_field


DATA_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "nuclear_attenuation"
)


@lru_cache
def _load_nuclear_table(
    material: Material,
):
    """Load and validate the Geant4 primary-proton removal table."""

    path = DATA_DIR / f"{material.value}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"No nuclear-attenuation data found for {material.value}."
        )

    table = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=float,
    )

    energies = np.asarray(
        table["energy_mev"],
        dtype=float,
    )

    coefficients = np.asarray(
        table["nuclear_removal_coefficient_per_m"],
        dtype=float,
    )

    if len(energies) < 2:
        raise ValueError(
            f"Nuclear-attenuation table for {material.value} "
            "must contain at least two points."
        )

    if not np.all(np.diff(energies) > 0.0):
        raise ValueError(
            f"Nuclear-attenuation energies for {material.value} "
            "must be strictly increasing."
        )

    if (
        not np.all(np.isfinite(coefficients))
        or np.any(coefficients < 0.0)
    ):
        raise ValueError(
            f"Nuclear-removal coefficients for {material.value} "
            "must be finite and non-negative."
        )

    return energies, coefficients


@lru_cache
def _nuclear_table_slopes(
    material: Material,
):
    """PCHIP nodal slopes of the nuclear removal table."""

    energies, coefficients = _load_nuclear_table(material)
    return pchip_slopes(energies, coefficients)


def nuclear_removal_coefficient(
    material: Material,
    energy: float,
    interpolation: CoefficientInterpolation = CoefficientInterpolation.LINEAR,
) -> float:
    """Return the proton-inelastic macroscopic removal coefficient [1/m]."""

    energies, coefficients = _load_nuclear_table(material)

    if (
        energy < energies[0]
        and not np.isclose(energy, energies[0])
    ) or (
        energy > energies[-1]
        and not np.isclose(energy, energies[-1])
    ):
        raise ValueError(
            f"Energy {energy} MeV is outside the nuclear table for "
            f"{material.value}: [{energies[0]}, {energies[-1]}] MeV."
        )

    energy = float(
        np.clip(
            energy,
            energies[0],
            energies[-1],
        )
    )

    if interpolation is CoefficientInterpolation.LINEAR:
        return float(
            np.interp(
                energy,
                energies,
                coefficients,
            )
        )

    return interpolate_coefficient(
        energies,
        coefficients,
        _nuclear_table_slopes(material),
        energy,
        interpolation,
    )


def nuclear_removal_coefficient_field(
    material_map: MaterialMap,
    energy: float,
    xp,
    interpolation: CoefficientInterpolation = CoefficientInterpolation.LINEAR,
):
    """Return ``Sigma_r(E)`` in every computational cell."""

    values = xp.asarray([
        nuclear_removal_coefficient(
            material,
            energy,
            interpolation,
        )
        for material in material_map.materials
    ])

    return values[
        material_map.ids
    ]


def nuclear_attenuation_rate_field(
    material_map: MaterialMap,
    energy: float,
    xp,
    interpolation: CoefficientInterpolation = CoefficientInterpolation.LINEAR,
):
    """Return ``Sigma_r(E) / S(E)`` in every cell or as a scalar."""

    if len(material_map.materials) == 1:
        material = material_map.materials[0]

        return (
            nuclear_removal_coefficient(
                material,
                energy,
                interpolation,
            )
            / stopping_power(
                material,
                energy,
            )
        )

    sigma_r = nuclear_removal_coefficient_field(
        material_map,
        energy,
        xp,
        interpolation,
    )

    stopping = stopping_power_field(
        material_map,
        energy,
        xp,
    )

    return sigma_r / stopping


def nuclear_optical_depth(
    material_map: MaterialMap,
    energy: float,
    dE: float,
    xp,
    interpolation: CoefficientInterpolation = CoefficientInterpolation.LINEAR,
):
    """Integrate ``Sigma_r(E) / S(E)`` over one descending energy step."""

    if dE <= 0.0:
        raise ValueError(
            "dE must be positive."
        )

    midpoint = energy - 0.5 * dE
    half_width = 0.5 * dE
    offset = half_width / np.sqrt(3.0)

    return half_width * (
        nuclear_attenuation_rate_field(
            material_map,
            midpoint - offset,
            xp,
            interpolation,
        )
        + nuclear_attenuation_rate_field(
            material_map,
            midpoint + offset,
            xp,
            interpolation,
        )
    )


def nuclear_attenuation_substep_inplace(
    U: Any,
    optical_depth,
    xp,
):
    """Apply the exact primary-removal source flow to every M1 moment."""

    damping = xp.exp(
        -xp.minimum(
            optical_depth,
            700.0,
        )
    )

    U *= damping

    return U
