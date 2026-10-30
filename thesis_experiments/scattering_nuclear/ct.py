"""CT-derived benchmark: extended model, WENO-Z, against the BFP Monte Carlo.

A section of the CORT prostate CT (Craft et al., GigaScience 2014, provided
as a HU array by E. Benschop) is cut out along one lateral beam path through
the femoral head and mapped onto the two materials of the heterogeneous
benchmarks, water (HU < 150) and compact bone (HU >= 150), so that the CT
changes only the geometry. Voxels in front of the skin are set to water
through the body mask. The 3 mm
CT voxels are refined by nearest-neighbour repetition to the cell width of
the benchmarks, so the anatomy is unchanged and only the transport grid is
finer. The deterministic solver (WENO-Z, extended model) and the BFP Monte
Carlo solver use the identical material map.

Input:  data/ct/prostate.npz   (HU array, shape (183, 183, 90), 3 mm voxels)
Output: thesis_results/scattering_nuclear/ct/
            ct_section.npz            material section (HU, body_mask)
            ct_benchmark.{pdf,png}    material map, doses, difference
            ct_idd_profiles.{pdf,png} integrated depth dose, lateral profiles
            ct_metrics.json, ct_metrics_table.tex

Usage: uv run python -m thesis_experiments.scattering_nuclear.ct
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from proton_solver import (  # noqa: E402
    Backend,
    Beam,
    EnergyScheme,
    Geometry,
    HUPartition,
    Material,
    Numerics,
    OutputConfig,
    Physics,
    Reference,
    References,
    Scattering,
    Solver,
    SolverConfig,
    SpatialScheme,
    WaveSpeeds,
)

CT_SOURCE = Path("data/ct/prostate.npz")
SOURCE_SPACING_M = 3.0e-3
# Beam along +axis 1 of the source array (left-right), through the column
# (axis 0, axis 2) = (80, 45), which crosses the femoral head.
BEAM_AXIS = 1
BEAM_COLUMN = (80, 45)
HALF_WIDTH_VOXELS = 5          # 30 mm lateral section
DEPTH_VOXELS = 50              # 150 mm depth from the skin
UPSAMPLE = int(os.environ.get("CT_UPSAMPLE", "6"))   # 3 mm -> 0.5 mm
HISTORIES = int(float(os.environ.get("CT_HISTORIES", "1e8")))
SKIN_HU = -300.0
# Two-material partition: soft tissue and fat -> water, bone -> compact bone.
# No air or lung, so the CFL step stays that of the benchmarks.
HU_PARTITION = HUPartition(
    cutoffs=(150.0,),
    materials=(Material.WATER, Material.BONE_COMPACT),
)

RESULT_DIR = Path("thesis_results/scattering_nuclear/ct")
MATERIAL_COLORS = {"WATER": "#9ecae1", "BONE_COMPACT": "#bdbdbd", "LUNG": "#fdd0a2"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hu(path: Path) -> np.ndarray:
    """Load a HU volume stored as .npy (also under a .npz name) or .npz."""

    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.ndarray):
        return data
    with data:
        key = "HU" if "HU" in data.files else data.files[0]
        return np.asarray(data[key])


def prepare_section() -> tuple[Path, dict]:
    """Cut the beam section, mark voxels in front of the skin, refine."""

    hu = load_hu(CT_SOURCE)
    if hu.ndim != 3:
        raise ValueError(f"Expected a 3D HU array, got shape {hu.shape}.")

    # Reorder to solver axes: x = axis 0, y = axis 2, z = beam axis 1.
    volume = np.transpose(hu, (0, 2, 1))
    cx, cy = BEAM_COLUMN
    column = volume[cx, cy, :]
    skin = int(np.argmax(column > SKIN_HU))
    if column[skin] <= SKIN_HU:
        raise ValueError("No body voxel found along the beam column.")

    section = volume[
        cx - HALF_WIDTH_VOXELS:cx + HALF_WIDTH_VOXELS,
        cy - HALF_WIDTH_VOXELS:cy + HALF_WIDTH_VOXELS,
        skin:skin + DEPTH_VOXELS,
    ]
    if section.shape != (2 * HALF_WIDTH_VOXELS, 2 * HALF_WIDTH_VOXELS, DEPTH_VOXELS):
        raise ValueError(f"Section leaves the CT volume: shape {section.shape}.")

    # Voxels in front of the first body voxel of their own column are
    # outside the patient and become water through the body mask.
    inside = section > SKIN_HU
    body_mask = np.cumsum(inside, axis=2) > 0

    section = np.repeat(np.repeat(np.repeat(section, UPSAMPLE, 0), UPSAMPLE, 1), UPSAMPLE, 2)
    body_mask = np.repeat(np.repeat(np.repeat(body_mask, UPSAMPLE, 0), UPSAMPLE, 1), UPSAMPLE, 2)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / "ct_section.npz"
    np.savez(path, HU=section.astype(np.int16), body_mask=body_mask)

    info = {
        "source": str(CT_SOURCE),
        "source_sha256": sha256_file(CT_SOURCE),
        "source_shape": list(hu.shape),
        "source_spacing_mm": 1e3 * SOURCE_SPACING_M,
        "beam_axis_of_source": BEAM_AXIS,
        "beam_column_source_axes_0_2": list(BEAM_COLUMN),
        "skin_index_on_beam_axis": skin,
        "section_source_voxels": [2 * HALF_WIDTH_VOXELS, 2 * HALF_WIDTH_VOXELS, DEPTH_VOXELS],
        "upsample": UPSAMPLE,
        "cell_width_mm": 1e3 * SOURCE_SPACING_M / UPSAMPLE,
        "outside_body_voxels_fraction": float(1.0 - body_mask.mean()),
    }
    return path, info


def build_config(section_path: Path) -> SolverConfig:
    spacing = SOURCE_SPACING_M / UPSAMPLE
    geometry = Geometry.from_ct_scan(
        section_path,
        source_spacing_m=(spacing, spacing, spacing),
        hu_partition=HU_PARTITION,
    )
    return SolverConfig(
        geometry=geometry,
        beam=Beam(
            n_protons=1e6,
            energy=150.0,
            energy_spread=1.0,
            sigma=(0.002, 0.002),
            center=(0.0, 0.0),
            angular_spread=0.0,
            cutoff_sigma=5.0,
        ),
        backend=Backend.CPU,
        physics=Physics(scattering=Scattering.WENTZEL, nuclear=True),
        numerics=Numerics(
            spatial_scheme=SpatialScheme.WENO_Z,
            energy_scheme=EnergyScheme.IF_SSPRK3,
            wave_speeds=WaveSpeeds.EXACT,
            max_energy_step=0.25,
            cfl=0.9,
            energy_min=1e-4,
            energy_window_sigma=5.0,
            moving_active_domain=False,
        ),
        references=References(
            reference=Reference.BFP_MC,
            bfp_mc_histories=HISTORIES,
            bfp_mc_seed=137,
            bfp_mc_threads=12,
            bfp_mc_max_step_mm=0.25,
            bfp_mc_max_energy_loss_mev=0.25,
            bfp_mc_max_scattering_optical_depth=0.02,
            bfp_mc_stop_energy_mev=1.0,
            bfp_mc_force_rerun=False,
        ),
        output=OutputConfig(save_results=True),
    )


def central_index(centres) -> int:
    """Cell that contains coordinate 0 (half-open rule)."""

    centres = np.asarray(centres)
    width = float(centres[1] - centres[0])
    index = int(np.floor((0.0 - (centres[0] - 0.5 * width)) / width + 1e-9))
    return min(max(index, 0), len(centres) - 1)


def save_figures(result, material_ids, materials):
    x = 1e3 * np.asarray(result.grid.x)
    y = 1e3 * np.asarray(result.grid.y)
    z = 1e3 * np.asarray(result.grid.z)
    det = np.asarray(result.dose)
    ref = np.asarray(result.bfp_mc_dose)
    iy = central_index(y)
    scale = float(ref.max())
    extent = [z[0] - 0.5 * (z[1] - z[0]), z[-1] + 0.5 * (z[1] - z[0]),
              x[0] - 0.5 * (x[1] - x[0]), x[-1] + 0.5 * (x[1] - x[0])]

    names = [m.name.replace("_COMPACT", "").capitalize() for m in materials]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 5.6), sharex=True, sharey=True)
    panels = (
        (material_ids[:, iy, :], "Material map", "tab10", None, None),
        (det[:, iy, :] / scale, "WENO-Z dose / max reference", "viridis", 0.0, 1.0),
        (ref[:, iy, :] / scale, "BFP Monte Carlo dose / max reference", "viridis", 0.0, 1.0),
        (100.0 * (det - ref)[:, iy, :] / scale, "Difference [% of max]", "RdBu_r", -5.0, 5.0),
    )
    for ax, (field, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        if vmin is None:
            image = ax.imshow(field, origin="lower", aspect="auto", extent=extent,
                              cmap=matplotlib.colors.ListedColormap(
                                  [MATERIAL_COLORS.get(m.name, "#fdd0a2")
                                   for m in materials]),
                              vmin=-0.5, vmax=len(names) - 0.5, interpolation="nearest")
            cbar = fig.colorbar(image, ax=ax, ticks=range(len(names)))
            cbar.ax.set_yticklabels(names)
            ax.axhline(0.0, color="k", linewidth=0.8, linestyle="--")
        else:
            image = ax.imshow(field, origin="lower", aspect="auto", extent=extent,
                              cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            fig.colorbar(image, ax=ax)
        ax.set_title(title)
    for ax in axes[1]:
        ax.set_xlabel(r"Depth $z$ [mm]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$x$ [mm]")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "ct_benchmark.pdf")
    fig.savefig(RESULT_DIR / "ct_benchmark.png", dpi=300)
    plt.close(fig)

    dx = (x[1] - x[0]) * 1e-3
    dy = (y[1] - y[0]) * 1e-3
    idd_det = det.sum(axis=(0, 1)) * dx * dy
    idd_ref = ref.sum(axis=(0, 1)) * dx * dy
    peak = int(np.argmax(idd_ref))
    depths = sorted({int(np.argmin(np.abs(z - d))) for d in (20.0, 0.5 * z[peak], z[peak])})

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    axes[0].plot(z, idd_ref / idd_ref.max(), label="BFP Monte Carlo")
    axes[0].plot(z, idd_det / idd_ref.max(), "--", label="WENO-Z")
    axes[0].set_xlabel(r"Depth $z$ [mm]")
    axes[0].set_ylabel("Integrated depth dose / max")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    for k in depths:
        line, = axes[1].plot(x, ref[:, iy, k] / scale, label=f"MC, z = {z[k]:.1f} mm")
        axes[1].plot(x, det[:, iy, k] / scale, "--", color=line.get_color(),
                     label=f"WENO-Z, z = {z[k]:.1f} mm")
    axes[1].set_xlabel(r"$x$ [mm]")
    axes[1].set_ylabel("Dose / max reference")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "ct_idd_profiles.pdf")
    fig.savefig(RESULT_DIR / "ct_idd_profiles.png", dpi=300)
    plt.close(fig)


def write_table(metrics: dict, path: Path):
    gamma = metrics["gamma"]
    rng = metrics.get("range") or {}

    def fmt(value, digits=2):
        return "--" if value is None else f"{value:.{digits}f}"

    rows = [
        (r"$\gamma$ 3\%/3 mm [\%]", fmt(gamma["3_percent_3_mm"]["pass_rate_percent"])),
        (r"$\gamma$ 2\%/2 mm [\%]", fmt(gamma["2_percent_2_mm"]["pass_rate_percent"])),
        (r"$\gamma$ 1\%/1 mm [\%]", fmt(gamma["1_percent_1_mm"]["pass_rate_percent"])),
        (r"$\delta_{\max}$ [\%]", fmt(metrics["maximum_dose"]["relative_difference_percent"])),
        (r"$\Delta R_{80}$ [mm]", fmt(rng.get("r80_difference_mm"), 3)),
        (r"$\Delta\mathrm{DFW}$ [mm]", fmt(rng.get("dfw_difference_mm"), 3)),
    ]
    lines = [r"\begin{tabular}{lr}", r"\toprule", r"Metric & WENO-Z \\", r"\midrule"]
    lines += [f"{name} & {value} \\\\" for name, value in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    section_path, info = prepare_section()
    config = build_config(section_path)
    result = Solver(config).solve()
    result.metrics.summary()

    from proton_solver.numerics.grid import Grid
    from proton_solver.physics.materials import MaterialMap

    grid = Grid.from_geometry(config.geometry, np)
    material_map = MaterialMap.from_geometry(config.geometry, grid, np)
    ids = np.asarray(material_map.ids)
    fractions = {
        material.name: float(np.mean(ids == i))
        for i, material in enumerate(material_map.materials)
    }

    metrics = result.metrics.comparison_metrics()
    record = {
        "section": info,
        "material_fractions": fractions,
        "hu_partition": {
            "cutoffs": list(HU_PARTITION.cutoffs),
            "materials": [m.name for m in HU_PARTITION.materials],
        },
        "histories": HISTORIES,
        "run_directory": str(result.run_directory),
        "metrics": metrics,
    }
    (RESULT_DIR / "ct_metrics.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8"
    )
    write_table(metrics, RESULT_DIR / "ct_metrics_table.tex")
    save_figures(result, ids, material_map.materials)
    print(json.dumps({"material_fractions": fractions,
                      "gamma_1_1": metrics["gamma"]["1_percent_1_mm"]["pass_rate_percent"],
                      "delta_max_percent": metrics["maximum_dose"]["relative_difference_percent"],
                      "range": metrics.get("range")}, indent=2, default=str))
    print(f"Saved {RESULT_DIR}")


if __name__ == "__main__":
    main()
