"""Interpolation of tabulated energy-dependent coefficients.

The Wentzel transport coefficient and the nuclear removal coefficient are
tabulated at discrete energies. By default they are interpolated linearly,
which is continuous but has a kink at every table energy. The PCHIP option
uses the same monotone cubic Hermite interpolation as the stopping power,
which is continuously differentiable. It is a diagnostic option for the
convergence study and is not used by the benchmark results.
"""

from __future__ import annotations

import numpy as np

from ..enums import CoefficientInterpolation
from .stopping_power import _pchip_endpoint_slope


def pchip_slopes(energies, values):
    """Return the PCHIP nodal slopes (same construction as the stopping power)."""

    energies = np.asarray(energies, dtype=float)
    values = np.asarray(values, dtype=float)

    h = np.diff(energies)
    delta = np.diff(values) / h
    slopes = np.zeros_like(values)

    if len(energies) == 2:
        slopes[:] = delta[0]
        return slopes

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

    slopes[0] = _pchip_endpoint_slope(h[0], h[1], delta[0], delta[1])
    slopes[-1] = _pchip_endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])

    return slopes


def pchip_evaluate(energies, values, slopes, energy: float) -> float:
    """Evaluate the cubic Hermite interpolant at one energy inside the table."""

    if energy >= energies[-1]:
        return float(values[-1])

    i = int(np.searchsorted(energies, energy, side="right") - 1)
    i = max(0, min(i, len(energies) - 2))

    h = energies[i + 1] - energies[i]
    t = (energy - energies[i]) / h
    t2 = t * t
    t3 = t2 * t

    return float(
        (2.0 * t3 - 3.0 * t2 + 1.0) * values[i]
        + (t3 - 2.0 * t2 + t) * h * slopes[i]
        + (-2.0 * t3 + 3.0 * t2) * values[i + 1]
        + (t3 - t2) * h * slopes[i + 1]
    )


def interpolate_coefficient(
    energies,
    values,
    slopes,
    energy: float,
    interpolation: CoefficientInterpolation,
) -> float:
    """Interpolate a tabulated coefficient with the selected method.

    ``energy`` must already be clamped to the table range. ``slopes`` is only
    used for PCHIP.
    """

    if interpolation is CoefficientInterpolation.LINEAR:
        return float(np.interp(energy, energies, values))

    if interpolation is CoefficientInterpolation.PCHIP:
        return pchip_evaluate(energies, values, slopes, energy)

    raise ValueError(f"Unknown coefficient interpolation: {interpolation}")
