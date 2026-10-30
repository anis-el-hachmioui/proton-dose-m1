from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..enums import Material
from ..geometry import Geometry
from ..numerics.grid import Grid


DATA_DIR = Path(__file__).resolve().parents[3] / "data"


@lru_cache
def _load_material_table():
    """Load and cache material properties."""

    return np.genfromtxt(
        DATA_DIR / "materials.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )


def density(material: Material) -> float:
    """Return the material density in kg/m^3."""

    table = _load_material_table()
    row = table[table["material"] == material.value]

    if len(row) == 0:
        raise ValueError(f"Unknown material: {material.value}")

    return float(row["density_kg_m3"][0])

def radiation_length(material: Material) -> float:
    """Return the material radiation length in metres."""

    table = _load_material_table()
    row = table[
        table["material"] == material.value
    ]

    if len(row) == 0:
        raise ValueError(
            f"Unknown material: {material.value}"
        )

    value = float(
        row["radiation_length_m"][0]
    )

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"Invalid radiation length for {material.value}: "
            f"{value}"
        )

    return value

@dataclass(frozen=True)
class MaterialMap:
    ids: Any
    materials: tuple[Material, ...]

    @classmethod
    def from_geometry(cls, geometry: Geometry, grid: Grid, xp):
        """Create a voxelized material map from boxes or an oriented CT."""

        if geometry.ct_scan is not None:
            # CT voxel centers coincide exactly with the solver grid.
            partition = geometry.hu_partition
            materials = tuple(dict.fromkeys((
                *partition.materials, geometry.background_material,
            )))
            classes = partition.classify(geometry.ct_scan.hu)
            # Classes index partition.materials; translate if classes repeat.
            ids = np.asarray([
                materials.index(partition.materials[i])
                for i in range(len(partition.materials))
            ], dtype=np.int32)[classes]
            if geometry.ct_scan.body_mask is not None:
                ids = ids.copy()
                ids[~geometry.ct_scan.body_mask] = materials.index(
                    geometry.background_material
                )
            return cls(ids=xp.asarray(ids, dtype=xp.int32), materials=materials)

        materials = tuple(dict.fromkeys([
            geometry.background_material,
            *(obj.material for obj in geometry.objects),
        ]))

        material_ids = {
            material: i
            for i, material in enumerate(materials)
        }

        ids = xp.full(
            grid.shape,
            material_ids[geometry.background_material],
            dtype=xp.int32,
        )

        for box in geometry.objects:
            ix = xp.where((grid.x >= box.x_min) & (grid.x < box.x_max))[0]
            iy = xp.where((grid.y >= box.y_min) & (grid.y < box.y_max))[0]
            iz = xp.where((grid.z >= box.z_min) & (grid.z < box.z_max))[0]

            ids[
                ix[:, None, None],
                iy[None, :, None],
                iz[None, None, :],
            ] = material_ids[box.material]

        return cls(
            ids=ids,
            materials=materials,
        )

    def id_for(self, material: Material) -> int:
        """Return the integer ID associated with a material."""
        return self.materials.index(material)


def density_field(material_map: MaterialMap, xp):
    """Return the material density in every computational cell."""

    values = xp.asarray([
        density(material)
        for material in material_map.materials
    ])

    return values[material_map.ids]