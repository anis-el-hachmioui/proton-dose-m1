from functools import lru_cache
from pathlib import Path

import numpy as np

from ..enums import Material
from .materials import MaterialMap


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "stopping_power"


@lru_cache
def load_stopping_power_table(material: Material):
    """Load and cache the stopping-power table for a material."""

    path = DATA_DIR / f"{material.value}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"No stopping-power data found for {material.value}."
        )

    return np.genfromtxt(
        path,
        delimiter=",",
        names=True,
    )

def _pchip_endpoint_slope(
    h0: float,
    h1: float,
    delta0: float,
    delta1: float,
) -> float:
    """Return a shape-preserving slope at one end of a PCHIP curve."""

    slope = (
        (2.0 * h0 + h1) * delta0
        - h0 * delta1
    ) / (h0 + h1)

    # Prevent the interpolant from turning in the wrong direction.
    if np.sign(slope) != np.sign(delta0):
        return 0.0

    # Limit overshoot if the neighboring secant slopes change sign.
    if (
        np.sign(delta0) != np.sign(delta1)
        and abs(slope) > 3.0 * abs(delta0)
    ):
        return 3.0 * delta0

    return slope


@lru_cache
def _stopping_power_interpolant(material: Material):
    """Return PCHIP nodes, values, and nodal slopes for one material."""

    table = load_stopping_power_table(material)

    energies = np.asarray(
        table["energy_mev"],
        dtype=float,
    )

    values = np.asarray(
        table["electronic_stopping_power_mev_per_m"],
        dtype=float,
    )

    if len(energies) < 2:
        raise ValueError(
            f"Stopping-power table for {material.value} "
            "must contain at least two points."
        )

    h = np.diff(energies)

    if not np.all(h > 0.0):
        raise ValueError(
            f"Stopping-power energies for {material.value} "
            "must be strictly increasing."
        )

    if not np.all(np.isfinite(values)) or not np.all(values > 0.0):
        raise ValueError(
            f"Stopping-power values for {material.value} "
            "must be finite and positive."
        )

    # Secant slope on every interval.
    delta = np.diff(values) / h

    # Derivative at every table node.
    slopes = np.zeros_like(values)

    # Special trivial case.
    if len(energies) == 2:
        slopes[:] = delta[0]
        return energies, values, slopes

    # Interior PCHIP slopes.
    #
    # If two neighboring secant slopes have opposite signs,
    # use slope = 0 to preserve the local shape.
    same_sign = delta[:-1] * delta[1:] > 0.0

    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]

    interior = slopes[1:-1]

    interior[same_sign] = (
        w1[same_sign] + w2[same_sign]
    ) / (
        w1[same_sign] / delta[:-1][same_sign]
        + w2[same_sign] / delta[1:][same_sign]
    )

    # Left endpoint.
    slopes[0] = _pchip_endpoint_slope(
        h[0],
        h[1],
        delta[0],
        delta[1],
    )

    # Right endpoint.
    slopes[-1] = _pchip_endpoint_slope(
        h[-1],
        h[-2],
        delta[-1],
        delta[-2],
    )

    return energies, values, slopes


def stopping_power(
    material: Material,
    energy: float,
) -> float:
    """Return smoothly interpolated electronic stopping power."""

    energies, values, slopes = _stopping_power_interpolant(
        material
    )

    if (
        energy < energies[0]
        and not np.isclose(energy, energies[0])
    ) or (
        energy > energies[-1]
        and not np.isclose(energy, energies[-1])
    ):
        raise ValueError(
            f"Energy {energy} MeV outside stopping-power table range "
            f"[{energies[0]}, {energies[-1]}] MeV."
        )

    energy = float(
        np.clip(
            energy,
            energies[0],
            energies[-1],
        )
    )

    # Avoid searchsorted returning an interval beyond the table.
    if energy == energies[-1]:
        return float(values[-1])

    i = int(
        np.searchsorted(
            energies,
            energy,
            side="right",
        )
        - 1
    )

    i = max(
        0,
        min(i, len(energies) - 2),
    )

    # Local interval [E_i, E_{i+1}]
    h = energies[i + 1] - energies[i]

    # Normalized position in the interval.
    t = (energy - energies[i]) / h

    t2 = t * t
    t3 = t2 * t

    # Cubic Hermite basis functions.
    h00 = (
        2.0 * t3
        - 3.0 * t2
        + 1.0
    )

    h10 = (
        t3
        - 2.0 * t2
        + t
    )

    h01 = (
        -2.0 * t3
        + 3.0 * t2
    )

    h11 = (
        t3
        - t2
    )

    value = (
        h00 * values[i]
        + h10 * h * slopes[i]
        + h01 * values[i + 1]
        + h11 * h * slopes[i + 1]
    )

    return float(value)


def stopping_power_field(
    material_map: MaterialMap,
    energy: float,
    xp,
):
    """Return the stopping-power value in every computational cell."""

    values = xp.asarray([
        stopping_power(material, energy)
        for material in material_map.materials
    ])

    return values[material_map.ids]