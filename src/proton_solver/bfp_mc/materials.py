"""Material data and interpolation helpers for the BFP Monte Carlo reference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit

from ..physics.materials import MaterialMap, density
from ..physics.nuclear import _load_nuclear_table
from ..physics.scattering import _load_transport_table
from ..physics.stopping_power import _stopping_power_interpolant
MEV_TO_J = 1.602176634e-13


def _to_numpy(array):
    """Return a NumPy representation of a backend array."""

    if hasattr(array, "get"):
        return array.get()

    return np.asarray(array)

def _data_root() -> Path:
    """Locate the repository data directory."""

    here = Path(__file__).resolve()

    candidates = (
        here.parents[3] / "data",
        here.parents[4] / "data",
        Path.cwd() / "data",
    )

    for path in candidates:
        if (path / "materials.csv").exists():
            return path

    searched = "\n  ".join(str(path) for path in candidates)

    raise FileNotFoundError(
        "Could not locate proton material data. "
        f"Searched:\n  {searched}"
    )


def _pchip_slopes(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    h = np.diff(x)
    delta = np.diff(y) / h

    d = np.zeros_like(y)

    if len(x) == 2:
        d[:] = delta[0]
        return d

    same = delta[:-1] * delta[1:] > 0.0

    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]

    interior = d[1:-1]
    interior[same] = (
        w1[same] + w2[same]
    ) / (
        w1[same] / delta[:-1][same]
        + w2[same] / delta[1:][same]
    )

    def endpoint(h0, h1, q0, q1):
        slope = ((2.0 * h0 + h1) * q0 - h0 * q1) / (h0 + h1)

        if np.sign(slope) != np.sign(q0):
            return 0.0

        if (
            np.sign(q0) != np.sign(q1)
            and abs(slope) > 3.0 * abs(q0)
        ):
            return 3.0 * q0

        return slope

    d[0] = endpoint(
        h[0],
        h[1],
        delta[0],
        delta[1],
    )

    d[-1] = endpoint(
        h[-1],
        h[-2],
        delta[-1],
        delta[-2],
    )

    return d


def _pchip_eval(
    x: np.ndarray,
    y: np.ndarray,
    slopes: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    query = np.asarray(
        query,
        dtype=np.float64,
    )

    idx = np.searchsorted(
        x,
        query,
        side="right",
    ) - 1

    idx = np.clip(
        idx,
        0,
        len(x) - 2,
    )

    h = x[idx + 1] - x[idx]
    t = (query - x[idx]) / h

    t2 = t * t
    t3 = t2 * t

    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2

    out = (
        h00 * y[idx]
        + h10 * h * slopes[idx]
        + h01 * y[idx + 1]
        + h11 * h * slopes[idx + 1]
    )

    out[query <= x[0]] = y[0]
    out[query >= x[-1]] = y[-1]

    return out


def _build_lookups(
    material_name: str,
    energy_max: float,
    nuclear_on: bool = False,
    n_energy: int = 65536,
    n_range: int = 131072,
):
    """Build dense homogeneous-material lookup tables."""

    root = _data_root()

    stopping_data = np.genfromtxt(
        root / "stopping_power" / f"{material_name}.csv",
        delimiter=",",
        names=True,
    )

    transport_data = np.genfromtxt(
        root / "wentzel_transport" / f"{material_name}.csv",
        delimiter=",",
        names=True,
    )

    energy_source = np.asarray(
        stopping_data["energy_mev"],
        dtype=np.float64,
    )

    stopping_source = np.asarray(
        stopping_data[
            "electronic_stopping_power_mev_per_m"
        ],
        dtype=np.float64,
    )

    slopes = _pchip_slopes(
        energy_source,
        stopping_source,
    )

    energy_min = float(energy_source[0])
    energy_max = min(
        max(float(energy_max), 1.0),
        float(energy_source[-1]),
    )

    energy_integral = np.geomspace(
        energy_min,
        energy_max,
        200000,
    )

    stopping_integral = _pchip_eval(
        energy_source,
        stopping_source,
        slopes,
        energy_integral,
    )

    range_integral = np.zeros_like(
        energy_integral
    )

    range_integral[1:] = np.cumsum(
        0.5
        * (
            1.0 / stopping_integral[1:]
            + 1.0 / stopping_integral[:-1]
        )
        * np.diff(energy_integral)
    )

    energy_grid = np.linspace(
        0.0,
        energy_max,
        n_energy,
        dtype=np.float64,
    )

    sigma_tr = np.interp(
        energy_grid,
        np.asarray(
            transport_data["energy_mev"],
            dtype=float,
        ),
        np.asarray(
            transport_data[
                "first_transport_coefficient_per_m"
            ],
            dtype=float,
        ),
        left=float(
            transport_data[
                "first_transport_coefficient_per_m"
            ][0]
        ),
        right=float(
            transport_data[
                "first_transport_coefficient_per_m"
            ][-1]
        ),
    )

    if nuclear_on:
        nuclear_data = np.genfromtxt(
            root / "nuclear_attenuation" / f"{material_name}.csv",
            delimiter=",",
            names=True,
        )

        sigma_r = np.interp(
            energy_grid,
            np.asarray(
                nuclear_data["energy_mev"],
                dtype=float,
            ),
            np.asarray(
                nuclear_data[
                    "nuclear_removal_coefficient_per_m"
                ],
                dtype=float,
            ),
            left=float(
                nuclear_data[
                    "nuclear_removal_coefficient_per_m"
                ][0]
            ),
            right=float(
                nuclear_data[
                    "nuclear_removal_coefficient_per_m"
                ][-1]
            ),
        )
    else:
        sigma_r = np.zeros_like(
            energy_grid
        )

    range_max = float(range_integral[-1])

    range_grid = np.linspace(
        0.0,
        range_max,
        n_range,
        dtype=np.float64,
    )

    energy_of_range = np.interp(
        range_grid,
        range_integral,
        energy_integral,
        left=0.0,
        right=energy_max,
    )

    return (
        energy_grid,
        sigma_tr,
        sigma_r,
        range_grid,
        energy_of_range,
    )


def _density_kg_m3(
    material_name: str,
) -> float:
    table = np.genfromtxt(
        _data_root() / "materials.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    row = table[
        table["material"]
        == material_name
    ]

    if len(row) != 1:
        raise ValueError(
            f"Unknown material {material_name!r} "
            "in materials.csv"
        )

    return float(
        row["density_kg_m3"][0]
    )


def _pack_material_tables(
    material_map: MaterialMap,
    nuclear_on: bool = False,
):
    """Pack interpolation tables for all materials in one voxel map."""

    stopping = [
        _stopping_power_interpolant(material)
        for material in material_map.materials
    ]

    scattering = [
        _load_transport_table(material)
        for material in material_map.materials
    ]

    nuclear = (
        [
            _load_nuclear_table(material)
            for material in material_map.materials
        ]
        if nuclear_on
        else None
    )

    ns_max = max(
        len(item[0])
        for item in stopping
    )

    nt_max = max(
        len(item[0])
        for item in scattering
    )

    nr_max = (
        max(
            len(item[0])
            for item in nuclear
        )
        if nuclear is not None
        else 1
    )

    n_materials = len(
        material_map.materials
    )

    s_energy = np.zeros(
        (n_materials, ns_max),
        dtype=np.float64,
    )

    s_value = np.zeros(
        (n_materials, ns_max),
        dtype=np.float64,
    )

    s_slope = np.zeros(
        (n_materials, ns_max),
        dtype=np.float64,
    )

    s_len = np.zeros(
        n_materials,
        dtype=np.int32,
    )

    t_energy = np.zeros(
        (n_materials, nt_max),
        dtype=np.float64,
    )

    t_value = np.zeros(
        (n_materials, nt_max),
        dtype=np.float64,
    )

    t_len = np.zeros(
        n_materials,
        dtype=np.int32,
    )

    r_energy = np.zeros(
        (n_materials, nr_max),
        dtype=np.float64,
    )

    r_value = np.zeros(
        (n_materials, nr_max),
        dtype=np.float64,
    )

    r_len = np.zeros(
        n_materials,
        dtype=np.int32,
    )

    densities = np.empty(
        n_materials,
        dtype=np.float64,
    )

    for i, material in enumerate(
        material_map.materials
    ):
        energy, values, slopes = stopping[i]
        n = len(energy)

        s_energy[i, :n] = energy
        s_value[i, :n] = values
        s_slope[i, :n] = slopes
        s_len[i] = n

        energy, values = scattering[i]
        n = len(energy)

        t_energy[i, :n] = energy
        t_value[i, :n] = values
        t_len[i] = n

        if nuclear is not None:
            energy, values = nuclear[i]
            n = len(energy)

            r_energy[i, :n] = energy
            r_value[i, :n] = values
            r_len[i] = n
        else:
            r_len[i] = 1

        densities[i] = density(material)

    return (
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
    )


@njit(inline="always")
def _pchip_material_value(
    energy,
    material_id,
    table_energy,
    table_value,
    table_slope,
    table_length,
):
    """Evaluate stopping power for one material inside Numba."""

    n = int(table_length[material_id])

    if energy <= table_energy[material_id, 0]:
        return table_value[material_id, 0]

    if energy >= table_energy[material_id, n - 1]:
        return table_value[material_id, n - 1]

    lo = 0
    hi = n - 1

    while hi - lo > 1:
        mid = (lo + hi) // 2

        if table_energy[material_id, mid] <= energy:
            lo = mid
        else:
            hi = mid

    x0 = table_energy[material_id, lo]
    x1 = table_energy[material_id, lo + 1]
    h = x1 - x0
    t = (energy - x0) / h

    t2 = t * t
    t3 = t2 * t

    return (
        (2.0 * t3 - 3.0 * t2 + 1.0)
        * table_value[material_id, lo]
        + (t3 - 2.0 * t2 + t)
        * h
        * table_slope[material_id, lo]
        + (-2.0 * t3 + 3.0 * t2)
        * table_value[material_id, lo + 1]
        + (t3 - t2)
        * h
        * table_slope[material_id, lo + 1]
    )


@njit(inline="always")
def _linear_material_value(
    energy,
    material_id,
    table_energy,
    table_value,
    table_length,
):
    """Evaluate Wentzel coefficient for one material inside Numba."""

    n = int(table_length[material_id])

    if energy <= table_energy[material_id, 0]:
        return table_value[material_id, 0]

    if energy >= table_energy[material_id, n - 1]:
        return table_value[material_id, n - 1]

    lo = 0
    hi = n - 1

    while hi - lo > 1:
        mid = (lo + hi) // 2

        if table_energy[material_id, mid] <= energy:
            lo = mid
        else:
            hi = mid

    x0 = table_energy[material_id, lo]
    x1 = table_energy[material_id, lo + 1]
    t = (energy - x0) / (x1 - x0)

    return (
        (1.0 - t) * table_value[material_id, lo]
        + t * table_value[material_id, lo + 1]
    )
