import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .enums import Material


@dataclass(frozen=True)
class Box:
    """Axis-aligned box using half-open intervals [min, max)."""

    x_min: float                           # m
    x_max: float                           # m
    y_min: float                           # m
    y_max: float                           # m
    z_min: float                           # m
    z_max: float                           # m

    material: Material

    def __post_init__(self):
        """Validate box dimensions."""

        # Validate that all coordinates are finite
        if not all(math.isfinite(x) for x in (
            self.x_min, self.x_max,
            self.y_min, self.y_max,
            self.z_min, self.z_max,
        )):
            raise ValueError("Box coordinates must be finite.")

        # Validate that all dimensions are positive
        if (
            self.x_min >= self.x_max
            or self.y_min >= self.y_max
            or self.z_min >= self.z_max
        ):
            raise ValueError("Box minima must be smaller than maxima.")


@dataclass(frozen=True)
class HUPartition:
    """Map HU values to the existing transport materials."""

    cutoffs: tuple[float, ...] = (-200.0, 150.0)
    materials: tuple[Material, ...] = (
        Material.LUNG,
        Material.WATER,
        Material.BONE_COMPACT,
    )

    def __post_init__(self):
        if len(self.materials) != len(self.cutoffs) + 1:
            raise ValueError("HUPartition needs one more material than cutoffs.")
        if not all(isinstance(material, Material) for material in self.materials):
            raise ValueError("HUPartition materials must be Material values.")
        if not all(math.isfinite(x) for x in self.cutoffs):
            raise ValueError("HU cutoffs must be finite.")
        if any(a >= b for a, b in zip(self.cutoffs, self.cutoffs[1:])):
            raise ValueError("HU cutoffs must be strictly increasing.")

    def classify(self, hu: np.ndarray) -> np.ndarray:
        """Return integer material indices with exact half-open HU bins."""
        hu = np.asarray(hu)
        if hu.ndim != 3 or not np.issubdtype(hu.dtype, np.number):
            raise ValueError("CT HU must be a numeric 3D array.")
        if not np.all(np.isfinite(hu)):
            raise ValueError("CT HU contains non-finite values.")
        return np.searchsorted(self.cutoffs, hu, side="right").astype(np.int32)


@dataclass(frozen=True)
class CTScan:
    """Read a CT NPZ and orient its array axes as solver (x, y, +z).

    NPZ contains `HU` (or `hu` / `arr_0`) and optionally `body_mask`.
    source_spacing_m is mandatory because the CT NPZ need not store spacing.
    """

    path: str | Path
    source_spacing_m: tuple[float, float, float]
    axis_order: tuple[int, int, int] = (0, 1, 2)
    flip_axes: tuple[bool, bool, bool] = (False, False, False)

    hu: np.ndarray = field(init=False, repr=False, compare=False)
    body_mask: np.ndarray | None = field(init=False, repr=False, compare=False)
    spacing: tuple[float, float, float] = field(init=False)

    def __post_init__(self):
        if len(self.source_spacing_m) != 3 or not all(
            math.isfinite(s) and s > 0 for s in self.source_spacing_m
        ):
            raise ValueError("CT spacing must contain three positive finite metres.")
        if tuple(sorted(self.axis_order)) != (0, 1, 2):
            raise ValueError("axis_order must be a permutation of (0, 1, 2).")
        if len(self.flip_axes) != 3 or any(type(f) is not bool for f in self.flip_axes):
            raise ValueError("flip_axes must contain three booleans.")

        with np.load(self.path, allow_pickle=False) as data:
            key = "HU" if "HU" in data else "hu" if "hu" in data else None
            if key is None and data.files == ["arr_0"]:
                key = "arr_0"  # NPZ created with np.savez(path, array)
            if key is None:
                raise ValueError("CT NPZ needs a 'HU', 'hu', or single 'arr_0' array.")
            hu = np.asarray(data[key])
            mask = np.asarray(data["body_mask"]) if "body_mask" in data else None

        if hu.ndim != 3 or not np.issubdtype(hu.dtype, np.number):
            raise ValueError("CT HU must be a numeric 3D array.")
        if not np.all(np.isfinite(hu)):
            raise ValueError("CT HU contains non-finite values.")
        if mask is not None and mask.shape != hu.shape:
            raise ValueError("body_mask shape must match the HU array.")

        hu = np.transpose(hu, self.axis_order)
        if mask is not None:
            mask = np.transpose(mask.astype(bool), self.axis_order)
        for axis, flip in enumerate(self.flip_axes):
            if flip:
                hu = np.flip(hu, axis=axis)
                if mask is not None:
                    mask = np.flip(mask, axis=axis)

        object.__setattr__(self, "hu", np.ascontiguousarray(hu))
        object.__setattr__(self, "body_mask", None if mask is None else np.ascontiguousarray(mask))
        object.__setattr__(
            self, "spacing", tuple(self.source_spacing_m[i] for i in self.axis_order)
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.hu.shape


@dataclass
class Geometry:
    size: tuple[float, float, float]                # m
    resolution: tuple[int, int, int]

    background_material: Material = Material.WATER
    objects: list[Box] = field(default_factory=list)

    view_geometry: bool = False

    ct_scan: CTScan | None = None
    hu_partition: HUPartition = field(default_factory=HUPartition)

    def __post_init__(self):
        """Validate domain dimensions, resolution, and geometry objects."""
        # Validate dimensions
        if len(self.size) != 3 or len(self.resolution) != 3:
            raise ValueError("size and resolution must contain exactly three values.")

        if not all(math.isfinite(x) and x > 0 for x in self.size):
            raise ValueError("Geometry dimensions must be finite and positive.")

        if not all(type(n) is int and n > 0 for n in self.resolution):
            raise ValueError("Geometry resolution must contain positive integers.")

        # Geometry validity is always verified
        self._verify_objects()

        if self.ct_scan is not None:
            if self.objects:
                raise ValueError("CT and Box objects cannot be combined.")
            if self.resolution != self.ct_scan.shape:
                raise ValueError("Geometry resolution must match the CT array shape.")
            if not all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
                       for a, b in zip(self.spacing, self.ct_scan.spacing)):
                raise ValueError("Geometry cell spacing must match the CT spacing.")
            if self.view_geometry:
                raise ValueError("3D Box viewer does not support CT; set view_geometry=False.")

    @classmethod
    def from_ct_scan(
        cls,
        path: str | Path,
        *,
        source_spacing_m: tuple[float, float, float],
        axis_order: tuple[int, int, int] = (0, 1, 2),
        flip_axes: tuple[bool, bool, bool] = (False, False, False),
        hu_partition: HUPartition | None = None,
        background_material: Material = Material.WATER,
    ) -> "Geometry":
        """Create a domain exactly matching an oriented CT voxel grid."""
        scan = CTScan(
            path=path,
            source_spacing_m=source_spacing_m,
            axis_order=axis_order,
            flip_axes=flip_axes,
        )
        return cls(
            size=tuple(n * dx for n, dx in zip(scan.shape, scan.spacing)),
            resolution=scan.shape,
            background_material=background_material,
            ct_scan=scan,
            hu_partition=hu_partition if hu_partition is not None else HUPartition(),
        )

    @staticmethod
    def _boxes_overlap(a: Box, b: Box) -> bool:
        """Return whether two half-open boxes overlap."""
        return (
            a.x_min < b.x_max and b.x_min < a.x_max
            and a.y_min < b.y_max and b.y_min < a.y_max
            and a.z_min < b.z_max and b.z_min < a.z_max
        )

    def _verify_objects(self):
        """Validate object placement and ensure that boxes do not overlap."""

        lx, ly, lz = self.size

        for obj in self.objects:
            if not (
                -lx / 2 <= obj.x_min < obj.x_max <= lx / 2
                and -ly / 2 <= obj.y_min < obj.y_max <= ly / 2
                and 0.0 <= obj.z_min < obj.z_max <= lz
            ):
                raise ValueError("Geometry object extends outside the domain.")

        for i, a in enumerate(self.objects):
            for b in self.objects[i + 1:]:
                if self._boxes_overlap(a, b):
                    raise ValueError("Geometry objects may not overlap.")

    @property
    def nx(self):
        return self.resolution[0]

    @property
    def ny(self):
        return self.resolution[1]

    @property
    def nz(self):
        return self.resolution[2]

    @property
    def lx(self):
        return self.size[0]

    @property
    def ly(self):
        return self.size[1]

    @property
    def lz(self):
        return self.size[2]

    @property
    def dx(self):
        return self.lx / self.nx

    @property
    def dy(self):
        return self.ly / self.ny

    @property
    def dz(self):
        return self.lz / self.nz

    @property
    def spacing(self):
        return self.dx, self.dy, self.dz

    @property
    def shape(self):
        return self.resolution
