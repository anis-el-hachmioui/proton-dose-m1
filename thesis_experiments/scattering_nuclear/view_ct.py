"""Interactive napari viewer for the CT-derived benchmark.

Shows the full prostate CT together with the dose of the last CT benchmark
run (WENO-Z and the BFP Monte Carlo reference), placed at its true position
in the patient. Run the benchmark first:

    uv run python -m thesis_experiments.scattering_nuclear.ct

then open the viewer, without adding napari to the project:

    uv run --with "napari[all]" python -m thesis_experiments.scattering_nuclear.view_ct

Layers (toggle with the eye icon, adjust contrast and opacity on the left):
    CT [HU]                 full CT, 3 mm voxels
    Bone (HU >= 150)        material map of the transported section
    Section outline         box of the transported section
    Beam axis               central beam axis
    WENO-Z dose [Gy]        absorbed dose for the simulated number of
                            incident protons (mGy if the maximum is below
                            0.1 Gy), additive overlay
    BFP Monte Carlo dose [Gy]
                            same scale, hidden at start
    Difference [% of max]   WENO-Z minus reference, in % of the maximum
                            reference dose, hidden at start

Every dose layer has a colour bar on the right while it is visible.

Axes are those of the source array (axis 0, axis 1 = beam direction,
axis 2), in mm. The viewer opens on the slice through the beam axis, zoomed
in on the section. The slider at the bottom moves through axis 0, Home shows
the whole CT, and the 2D/3D button at the bottom left gives a volume
rendering.

This script only reads results. It does not run a solver and writes no files.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

RESULT_DIR = Path("thesis_results/scattering_nuclear/ct")
BONE_HU = 150.0


def load_hu(path: Path) -> np.ndarray:
    """Load a HU volume stored as .npy (also under a .npz name) or .npz."""

    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.ndarray):
        return data
    with data:
        key = "HU" if "HU" in data.files else data.files[0]
        return np.asarray(data[key])


def load_run(metrics_path: Path) -> dict:
    record = json.loads(metrics_path.read_text(encoding="utf-8"))
    section = record["section"]
    run = np.load(Path(record["run_directory"]) / "results.npz")
    if "bfp_mc_dose" not in run.files:
        raise ValueError("The run contains no BFP Monte Carlo dose.")

    h = float(section["cell_width_mm"])
    spacing = float(section["source_spacing_mm"])
    cx, cy = section["beam_column_source_axes_0_2"]
    nx, ny, _ = section["section_source_voxels"]
    skin = int(section["skin_index_on_beam_axis"])

    # Solver arrays are (x, y, z) = (source axis 0, source axis 2, source
    # axis 1). Transpose back to source order (axis 0, axis 1, axis 2).
    def to_source(a):
        return np.transpose(np.asarray(a, dtype=np.float32), (0, 2, 1))

    # First source voxel of the section along source axes 0, 1, 2.
    first = np.array([cx - nx // 2, skin, cy - ny // 2], dtype=float)
    # Centre of the first fine cell, in the mm frame where source voxel i
    # has its centre at i * spacing.
    translate = first * spacing - 0.5 * spacing + 0.5 * h

    # Number of incident protons of the run, for the dose unit.
    n_protons = None
    config_path = Path(record["run_directory"]) / "config.json"
    if config_path.is_file():
        beam = json.loads(config_path.read_text(encoding="utf-8")).get("beam", {})
        n_protons = beam.get("n_protons")

    return {
        "record": record,
        "n_protons": n_protons,
        "dose": to_source(run["dose"]),
        "reference": to_source(run["bfp_mc_dose"]),
        "h": h,
        "spacing": spacing,
        "translate": translate,
        "first": first,
        "beam_center_mm": np.array([cx * spacing, cy * spacing]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ct", type=Path, default=None,
                        help="HU array (default: the source of the last run)")
    parser.add_argument("--metrics", type=Path,
                        default=RESULT_DIR / "ct_metrics.json")
    parser.add_argument("--3d", dest="three_d", action="store_true",
                        help="start in 3D volume rendering")
    args = parser.parse_args()

    run = load_run(args.metrics)
    ct_path = args.ct or Path(run["record"]["section"]["source"])
    hu = load_hu(ct_path).astype(np.float32)

    h = run["h"]
    s = run["spacing"]
    t = run["translate"]
    dose = run["dose"]
    reference = run["reference"]
    scale = float(reference.max())
    diff = 100.0 * (dose - reference) / scale

    section = np.load(RESULT_DIR / "ct_section.npz")
    bone = np.transpose(section["HU"] >= BONE_HU, (0, 2, 1)).astype(np.uint8)

    import napari  # imported here so --help works without napari

    viewer = napari.Viewer(title="CT benchmark: prostate, WENO-Z vs BFP Monte Carlo",
                           ndisplay=3 if args.three_d else 2)

    viewer.add_image(hu, name="CT [HU]", scale=(s, s, s), colormap="gray",
                     contrast_limits=(-400.0, 1200.0))
    viewer.add_labels(bone, name="Bone (HU >= 150)", scale=(h, h, h),
                      translate=t, opacity=0.35, visible=False)

    # Section outline and beam axis in the mm frame.
    lo = t - 0.5 * h
    hi = lo + np.array(dose.shape) * h
    corners = np.array([[a, b, c] for a in (lo[0], hi[0])
                        for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3),
             (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    viewer.add_shapes([corners[[i, j]] for i, j in edges], shape_type="line",
                      edge_color="yellow", edge_width=0.6,
                      name="Section outline")
    c0, c2 = run["beam_center_mm"]
    viewer.add_shapes([np.array([[c0, lo[1], c2], [c0, hi[1], c2]])],
                      shape_type="line", edge_color="cyan", edge_width=0.6,
                      name="Beam axis")

    # Absolute dose in Gy, or in mGy if that gives more readable numbers.
    factor, unit = (1e3, "mGy") if scale < 0.1 else (1.0, "Gy")
    det_layer = viewer.add_image(
        factor * dose, name=f"WENO-Z dose [{unit}]", scale=(h, h, h),
        translate=t, colormap="inferno", blending="additive",
        contrast_limits=(0.0, factor * scale), opacity=0.8)
    ref_layer = viewer.add_image(
        factor * reference, name=f"BFP Monte Carlo dose [{unit}]",
        scale=(h, h, h), translate=t, colormap="inferno",
        blending="additive", contrast_limits=(0.0, factor * scale),
        opacity=0.8, visible=False)
    diff_layer = viewer.add_image(
        diff, name="Difference [% of max]", scale=(h, h, h),
        translate=t, colormap="PiYG", contrast_limits=(-5.0, 5.0),
        opacity=0.9, visible=False)

    # Colour bars (napari >= 0.5). Each is drawn only while its layer is
    # visible; the positions differ so that they never overlap.
    for layer, position in ((det_layer, "top_right"),
                            (ref_layer, "middle_right"),
                            (diff_layer, "bottom_right")):
        try:
            layer.colorbar.visible = True
            layer.colorbar.position = position
        except AttributeError:
            print("This napari version has no colour bars; update napari.")
            break
    try:
        viewer.text_overlay.visible = True
        protons = (f"{run['n_protons']:.0e} incident protons"
                   if run["n_protons"] else "the simulated incident protons")
        viewer.text_overlay.text = (
            f"Dose in {unit} for {protons}; "
            "difference in % of the maximum Monte Carlo dose"
        )
    except AttributeError:
        pass

    viewer.dims.axis_labels = ("axis 0 [mm]", "axis 1, beam [mm]", "axis 2 [mm]")
    viewer.dims.set_point(0, float(c0))
    viewer.layers.selection.active = det_layer

    # Open zoomed in on the transported section. Press Home to see all.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        viewer.reset_view()
        viewer.camera.center = tuple(0.5 * (lo + hi))
        viewer.camera.zoom *= 2.5

    metrics = run["record"]["metrics"]
    print("Run:", run["record"]["run_directory"])
    print("Cell width [mm]:", h, " histories:", run["record"]["histories"])
    print("Gamma 1%/1mm [%]:",
          metrics["gamma"]["1_percent_1_mm"]["pass_rate_percent"])
    print("Max dose difference [%]:",
          metrics["maximum_dose"]["relative_difference_percent"])
    napari.run()


if __name__ == "__main__":
    main()