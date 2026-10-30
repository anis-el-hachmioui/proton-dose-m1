from pathlib import Path

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .beam import Beam
from .geometry import Geometry


def _cuboid_faces(
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max,
):
    """Return the six faces of a cuboid."""

    p000 = (x_min, y_min, z_min)
    p001 = (x_min, y_min, z_max)
    p010 = (x_min, y_max, z_min)
    p011 = (x_min, y_max, z_max)

    p100 = (x_max, y_min, z_min)
    p101 = (x_max, y_min, z_max)
    p110 = (x_max, y_max, z_min)
    p111 = (x_max, y_max, z_max)

    return [
        [p000, p001, p011, p010],
        [p100, p110, p111, p101],
        [p000, p100, p101, p001],
        [p010, p011, p111, p110],
        [p000, p010, p110, p100],
        [p001, p101, p111, p011],
    ]


def _add_cuboid(
    ax,
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max,
    color,
    alpha,
):
    """Add a cuboid to a 3D axis."""

    faces = _cuboid_faces(
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
    )

    collection = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolors=color,
        linewidths=1.0,
        alpha=alpha,
    )

    ax.add_collection3d(collection)


def _geometry_figure(
    geometry: Geometry,
    beam: Beam,
    *,
    clean: bool,
):
    """Build the shared 3D geometry figure."""

    materials = tuple(dict.fromkeys([
        geometry.background_material,
        *(obj.material for obj in geometry.objects),
    ]))

    cmap = plt.get_cmap("tab10")
    colors = {
        material: cmap(i % cmap.N)
        for i, material in enumerate(materials)
    }

    scale = 1000.0  # m -> mm
    lx, ly, lz = geometry.size

    fig = plt.figure(figsize=(6, 4) if clean else (10, 8))
    ax = fig.add_subplot(111, projection="3d")

    _add_cuboid(
        ax,
        -0.5 * lx * scale,
        0.5 * lx * scale,
        -0.5 * ly * scale,
        0.5 * ly * scale,
        0.0,
        lz * scale,
        colors[geometry.background_material],
        0.15 if clean else 0.08,
    )

    for obj in geometry.objects:
        _add_cuboid(
            ax,
            obj.x_min * scale,
            obj.x_max * scale,
            obj.y_min * scale,
            obj.y_max * scale,
            obj.z_min * scale,
            obj.z_max * scale,
            colors[obj.material],
            0.65,
        )

    x0 = beam.center[0] * scale
    y0 = beam.center[1] * scale
    ax.plot(
        [x0, x0],
        [y0, y0],
        [0.0, lz * scale],
        "--",
        color="black",
        linewidth=2.0,
    )

    ax.set_xlim(-0.5 * lx * scale, 0.5 * lx * scale)
    ax.set_ylim(-0.5 * ly * scale, 0.5 * ly * scale)
    ax.set_zlim(0.0, lz * scale)
    ax.set_box_aspect((lx, ly, lz))
    ax.view_init(elev=-70, azim=-45, roll=-45)

    if clean:
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        return fig

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title("Geometry\nClose window to continue")

    legend = [
        Patch(
            facecolor=colors[material],
            edgecolor=colors[material],
            label=material.name,
        )
        for material in materials
    ]
    legend.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="Beam axis",
        )
    )
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()

    return fig


def view_geometry(
    geometry: Geometry,
    beam: Beam,
):
    """Show an interactive 3D view of the geometry and beam."""

    fig = _geometry_figure(geometry, beam, clean=False)
    plt.show(block=True)
    plt.close(fig)


def save_geometry_preview(
    geometry: Geometry,
    beam: Beam,
    path,
):
    """Save a clean vector geometry preview for the thesis."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = _geometry_figure(geometry, beam, clean=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
