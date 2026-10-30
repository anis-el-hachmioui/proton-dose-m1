from pathlib import Path
import sys


# Make repository-local modules such as thesis_experiments importable when this
# file is executed directly with:
#
#     uv run tools/generate_geometry_previews.py
#
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from proton_solver import SpatialScheme
from proton_solver.geometry_viewer import save_geometry_preview

from thesis_experiments.reduced_physics.homogeneous_water import (
    build_config as build_homogeneous_water,
)
from thesis_experiments.reduced_physics.longitudinal_slabs import (
    build_config as build_depth_wise_slab,
)
from thesis_experiments.reduced_physics.transversal_slabs import (
    build_config as build_lateral_interface,
)


OUTPUT_DIR = REPO_ROOT / "geometry_previews"


CASES = {
    "homogeneous_water": build_homogeneous_water,
    "depth_wise_slab": build_depth_wise_slab,
    "lateral_interface": build_lateral_interface,
}


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, build_config in CASES.items():
        config = build_config(
            SpatialScheme.FIRST_ORDER
        )

        output_path = OUTPUT_DIR / f"{name}.pdf"

        save_geometry_preview(
            config.geometry,
            config.beam,
            output_path,
        )

        print(f"Saved {output_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()