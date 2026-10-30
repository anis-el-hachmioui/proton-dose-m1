from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


from ..enums import CoefficientInterpolation, Material, Scattering
from .coefficient_interpolation import interpolate_coefficient, pchip_slopes
from .materials import MaterialMap, radiation_length
from .stopping_power import stopping_power, stopping_power_field


DATA_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "wentzel_transport"
)

PROTON_REST_ENERGY_MEV = 938.27208816
ROSSI_ES_MEV = 15.0

@lru_cache
def _load_transport_table(
    material: Material,
):
    """Load the Geant4 Wentzel first-transport-coefficient table."""

    path = (
        DATA_DIR
        / f"{material.value}.csv"
    )

    table = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=float,
    )

    return (
        np.asarray(
            table["energy_mev"],
            dtype=float,
        ),
        np.asarray(
            table[
                "first_transport_coefficient_per_m"
            ],
            dtype=float,
        ),
    )


@lru_cache
def _transport_table_slopes(
    material: Material,
):
    """PCHIP nodal slopes of the Wentzel transport table."""

    energies, coefficients = _load_transport_table(material)
    return pchip_slopes(energies, coefficients)


def first_transport_coefficient(
    material: Material,
    energy: float,
    interpolation: CoefficientInterpolation = CoefficientInterpolation.LINEAR,
) -> float:
    """Return the Wentzel first transport coefficient [1/m]."""

    energies, coefficients = (
        _load_transport_table(material)
    )

    # Floating-point roundoff can place a value that is mathematically on
    # a table boundary a few ulps outside the stored endpoint.  Accept such
    # values and clamp them back to the tabulated interval, while still
    # rejecting genuinely out-of-range energies.
    if (
        energy < energies[0]
        and not np.isclose(energy, energies[0])
    ) or (
        energy > energies[-1]
        and not np.isclose(energy, energies[-1])
    ):
        raise ValueError(
            f"Energy {energy} MeV is outside "
            f"the Wentzel table for {material.value}: "
            f"[{energies[0]}, {energies[-1]}] MeV."
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
        _transport_table_slopes(material),
        energy,
        interpolation,
    )

def proton_pv_mev(
    energy: float,
) -> float:
    """Return relativistic p*v for a proton, in MeV."""

    energy = max(
        float(energy),
        1.0e-8,
    )

    m = PROTON_REST_ENERGY_MEV

    return (
        energy
        * (energy + 2.0 * m)
        / (energy + m)
    )


def rossi_transport_coefficient(
    material: Material,
    energy: float,
) -> float:
    """
    Return the Fermi-Rossi transport coefficient [1/m].

        Sigma_tr(E) = (Es / (p*v))^2 / X0
    """

    X0 = radiation_length(material)
    pv = proton_pv_mev(energy)

    return (
        (ROSSI_ES_MEV / pv) ** 2
        / X0
    )

def transport_coefficient(
    material: Material,
    energy: float,
    model: Scattering,
    interpolation: CoefficientInterpolation = CoefficientInterpolation.LINEAR,
) -> float:
    """Return the transport coefficient for the selected scattering model."""

    if model is Scattering.WENTZEL:
        return first_transport_coefficient(
            material,
            energy,
            interpolation,
        )

    if model is Scattering.ROSSI:
        return rossi_transport_coefficient(
            material,
            energy,
        )

    if model is Scattering.OFF:
        return 0.0

    raise ValueError(
        f"Unknown scattering model: {model}"
    )

def first_transport_coefficient_field(
    material_map: MaterialMap,
    energy: float,
    xp,
    interpolation: CoefficientInterpolation = CoefficientInterpolation.LINEAR,
):
    """Return Sigma_tr,1(E) in every computational cell."""

    values = xp.asarray([
        first_transport_coefficient(
            material,
            energy,
            interpolation,
        )
        for material in material_map.materials
    ])

    return values[
        material_map.ids
    ]

def transport_coefficient_field(
    material_map: MaterialMap,
    energy: float,
    model: Scattering,
    xp,
    interpolation: CoefficientInterpolation = CoefficientInterpolation.LINEAR,
):
    """Return Sigma_tr(E) in every computational cell."""

    values = xp.asarray([
        transport_coefficient(
            material,
            energy,
            model,
            interpolation,
        )
        for material in material_map.materials
    ])

    return values[
        material_map.ids
    ]

def scattering_rate_field(
    material_map: MaterialMap,
    energy: float,
    model: Scattering,
    xp,
    interpolation: CoefficientInterpolation = CoefficientInterpolation.LINEAR,
):
    """Return Sigma_tr(E) / S(E) in every cell."""

    if len(material_map.materials) == 1:
        material = material_map.materials[0]
        return (
            transport_coefficient(
                material,
                energy,
                model,
                interpolation,
            )
            / stopping_power(
                material,
                energy,
            )
        )

    sigma_tr = transport_coefficient_field(
        material_map,
        energy,
        model,
        xp,
        interpolation,
    )

    stopping = stopping_power_field(
        material_map,
        energy,
        xp,
    )

    return sigma_tr / stopping

def scattering_optical_depth(
    material_map: MaterialMap,
    energy: float,
    dE: float,
    model: Scattering,
    xp,
    interpolation: CoefficientInterpolation = CoefficientInterpolation.LINEAR,
):
    """
    Integrate Sigma_tr(E) / S(E) over one energy step.

    Two-point Gauss-Legendre quadrature.
    """

    if model is Scattering.OFF:
        return xp.zeros(
            material_map.ids.shape,
            dtype=float,
        )

    midpoint = energy - 0.5 * dE
    half_width = 0.5 * dE
    offset = half_width / np.sqrt(3.0)

    return half_width * (
        scattering_rate_field(
            material_map,
            midpoint - offset,
            model,
            xp,
            interpolation,
        )
        + scattering_rate_field(
            material_map,
            midpoint + offset,
            model,
            xp,
            interpolation,
        )
    )

def angular_scattering_substep(
    U: Any,
    optical_depth,
    xp,
):
    """Apply the exact M1 angular-scattering source flow."""

    damping = xp.exp(
        -xp.minimum(
            optical_depth,
            700.0,
        )
    )

    U_new = U.copy()

    U_new[1:] *= damping

    return U_new

def angular_scattering_substep_inplace(
    U: Any,
    optical_depth,
    xp,
):
    """Apply the exact angular-scattering source flow in place."""

    damping = xp.exp(
        -xp.minimum(
            optical_depth,
            700.0,
        )
    )
    U[1:] *= damping
    return U
