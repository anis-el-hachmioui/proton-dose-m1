from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .materials import _to_numpy


BFP_MC_CACHE_VERSION = 1


_REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

CACHE_ROOT = (
    _REPO_ROOT
    / "bfp_mc_cache"
)


def _enum_name(value):
    return getattr(
        value,
        "name",
        str(value),
    )


def _array_hash(array):
    array = np.ascontiguousarray(
        _to_numpy(array)
    )

    hasher = hashlib.sha256()

    hasher.update(
        str(array.dtype).encode("utf-8")
    )

    hasher.update(
        np.asarray(
            array.shape,
            dtype=np.int64,
        ).tobytes()
    )

    hasher.update(
        array.tobytes()
    )

    return hasher.hexdigest()


def _cache_configuration(
    config,
    material_map,
    grid,
    inflow,
):
    """
    Return only the inputs that affect the BFP Monte Carlo reference.

    Deterministic solver settings such as WENO-Z versus first order
    deliberately do not appear here.
    """

    beam = config.beam
    physics = config.physics
    numerics = config.numerics
    references = config.references
    geometry = config.geometry

    materials = [
        getattr(
            material,
            "value",
            str(material),
        )
        for material in material_map.materials
    ]

    return {
        "cache_version": BFP_MC_CACHE_VERSION,

        "geometry": {
            "size": [
                float(value)
                for value in geometry.size
            ],
            "resolution": [
                int(value)
                for value in geometry.resolution
            ],
            "materials": materials,
            "material_map_sha256": _array_hash(
                material_map.ids
            ),
        },

        "beam": {
            "n_protons": float(
                beam.n_protons
            ),
            "energy_mev": float(
                beam.energy
            ),
            "energy_spread_mev": float(
                beam.energy_spread
            ),
            "sigma_m": [
                float(value)
                for value in beam.sigma
            ],
            "center_m": [
                float(value)
                for value in beam.center
            ],
            "angular_spread": float(
                beam.angular_spread
            ),
            "cutoff_sigma": float(
                beam.cutoff_sigma
            ),
        },

        "physics": {
            "scattering": _enum_name(
                physics.scattering
            ),
            "nuclear": bool(
                physics.nuclear
            ),
        },

        "energy": {
            "energy_min_mev": float(
                numerics.energy_min
            ),
            "energy_window_sigma": float(
                numerics.energy_window_sigma
            ),
        },

        "bfp_mc": {
            "histories": int(
                references.bfp_mc_histories
            ),
            "seed": int(
                references.bfp_mc_seed
            ),
            "max_step_mm": float(
                references.bfp_mc_max_step_mm
            ),
            "max_energy_loss_mev": float(
                references.bfp_mc_max_energy_loss_mev
            ),
            "max_scattering_optical_depth": float(
                references.bfp_mc_max_scattering_optical_depth
            ),
            "stop_energy_mev": float(
                references.bfp_mc_stop_energy_mev
            ),
        },

        # Important for heterogeneous cases and for detecting
        # changes to the finite-volume beam discretization.
        "inflow_spatial_sha256": _array_hash(
            inflow.spatial
        ),

        "grid": {
            "x_sha256": _array_hash(
                grid.x
            ),
            "y_sha256": _array_hash(
                grid.y
            ),
            "z_sha256": _array_hash(
                grid.z
            ),
        },
    }


def _signature(configuration):
    text = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:24]


def cache_paths(
    config,
    material_map,
    grid,
    inflow,
):
    configuration = _cache_configuration(
        config,
        material_map,
        grid,
        inflow,
    )

    signature = _signature(
        configuration
    )

    directory = (
        CACHE_ROOT
        / signature
    )

    return {
        "configuration": configuration,
        "signature": signature,
        "directory": directory,
        "data": directory / "reference.npz",
        "metadata": directory / "metadata.json",
    }


def load_bfp_mc_cache(
    config,
    material_map,
    grid,
    inflow,
):
    paths = cache_paths(
        config,
        material_map,
        grid,
        inflow,
    )

    data_path = paths["data"]
    metadata_path = paths["metadata"]

    if (
        not data_path.exists()
        or not metadata_path.exists()
    ):
        return None

    try:
        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            metadata.get("configuration")
            != paths["configuration"]
        ):
            return None

        with np.load(data_path) as data:
            dose = np.asarray(
                data["dose"],
                dtype=np.float64,
            )

            runtime = float(
                data["runtime_seconds"]
            )

    except (
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ):
        return None

    if dose.shape != tuple(grid.shape):
        return None

    if not np.all(np.isfinite(dose)):
        return None

    return (
        dose,
        runtime,
        paths["signature"],
    )


def save_bfp_mc_cache(
    config,
    material_map,
    grid,
    inflow,
    dose,
    runtime,
):
    paths = cache_paths(
        config,
        material_map,
        grid,
        inflow,
    )

    directory = paths["directory"]

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dose = np.asarray(
        dose,
        dtype=np.float64,
    )

    temporary_data = (
        directory
        / "reference.tmp"
    )

    with temporary_data.open("wb") as file:
        np.savez_compressed(
            file,
            dose=dose,
            runtime_seconds=np.float64(
                runtime
            ),
        )

    temporary_data.replace(
        paths["data"]
    )

    metadata = {
        "signature": paths["signature"],
        "threads_used": int(
            config.references.bfp_mc_threads
        ),
        "configuration": paths[
            "configuration"
        ],
        "runtime_seconds": float(
            runtime
        ),
    }

    temporary_metadata = (
        directory
        / "metadata.tmp"
    )

    temporary_metadata.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_metadata.replace(
        paths["metadata"]
    )

    return paths["signature"]