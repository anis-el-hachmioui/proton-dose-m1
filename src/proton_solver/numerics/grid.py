from dataclasses import dataclass
from typing import Any

from ..geometry import Geometry


@dataclass(frozen=True)
class Grid:
    x: Any
    y: Any
    z: Any

    dx: float
    dy: float
    dz: float

    shape: tuple[int, int, int]

    @classmethod
    def from_geometry(cls, geometry: Geometry, xp):
        """Create a cell-centered finite-volume grid from a geometry."""

        nx, ny, nz = geometry.resolution
        lx, ly, lz = geometry.size

        dx = lx / nx
        dy = ly / ny
        dz = lz / nz

        x = (xp.arange(nx) + 0.5) * dx - 0.5 * lx
        y = (xp.arange(ny) + 0.5) * dy - 0.5 * ly
        z = (xp.arange(nz) + 0.5) * dz

        return cls(
            x=x,
            y=y,
            z=z,
            dx=dx,
            dy=dy,
            dz=dz,
            shape=(nx, ny, nz),
        )

    @property
    def nx(self):
        return self.shape[0]

    @property
    def ny(self):
        return self.shape[1]

    @property
    def nz(self):
        return self.shape[2]

    @property
    def spacing(self):
        return self.dx, self.dy, self.dz

    @property
    def cell_volume(self):
        return self.dx * self.dy * self.dz