from pathlib import Path
import os
import shutil
import subprocess

from proton_solver.io.schneider import (
    load_schneider_calibration,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = Path(__file__).resolve().parent
BUILD_DIR = TOOL_DIR / "build"
DATA_DIR = REPO_ROOT / "data"

SCHNEIDER_CALIBRATION = (
    DATA_DIR
    / "ct_calibration"
    / "HUtoMaterialSchneider.txt"
)

# Existing materials used by the synthetic validation cases.
MATERIALS = [
    "G4_AIR",
    "G4_WATER",
    "G4_LUNG_ICRP",
    "G4_BONE_COMPACT_ICRU",
    "G4_MUSCLE_SKELETAL_ICRP",
]

# Schneider tables are exported at a reference density.
#
# For the CT solver the material composition is selected by tissue ID,
# while the actual voxel density comes from the TOPAS Schneider
# HU-to-density calibration. Per-length transport quantities can then
# be scaled from this reference density by the voxel/reference density
# ratio.
SCHNEIDER_REFERENCE_DENSITY_G_CM3 = 1.0


def run(command):
    command = [str(value) for value in command]

    print("+", " ".join(command))

    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
    )


def geant4_dir():
    if "Geant4_DIR" in os.environ:
        return Path(os.environ["Geant4_DIR"])

    path = (
        Path.home()
        / "Applications"
        / "GEANT4"
        / "geant4-install"
        / "lib"
        / "cmake"
        / "Geant4"
    )

    return path if path.exists() else None


def clean_data():
    print("Cleaning generated material data...")

    (DATA_DIR / "materials.csv").unlink(
        missing_ok=True
    )

    for directory in [
        "stopping_power",
        "nuclear_attenuation",
        "wentzel_transport",
    ]:
        shutil.rmtree(
            DATA_DIR / directory,
            ignore_errors=True,
        )


def build():
    BUILD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        "cmake",
        "-S",
        str(TOOL_DIR),
        "-B",
        str(BUILD_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
    ]

    g4_dir = geant4_dir()

    if g4_dir is not None:
        command.append(
            f"-DGeant4_DIR={g4_dir}"
        )

    run(command)

    run([
        "cmake",
        "--build",
        str(BUILD_DIR),
        "-j",
    ])


def executable(name):
    path = BUILD_DIR / name

    if path.exists():
        return path

    path = BUILD_DIR / "Release" / name

    if path.exists():
        return path

    raise FileNotFoundError(
        f"Executable not found: {name}"
    )


def custom_material_command(
    exporter,
    name,
    density_g_cm3,
    elements,
    weights,
):
    command = [
        exporter,
        "--custom",
        name,
        f"{density_g_cm3:.17g}",
    ]

    for element, weight in zip(
        elements,
        weights,
    ):
        # Zero-weight elements do not need to be added to G4Material.
        if weight <= 0.0:
            continue

        command.extend([
            element,
            f"{weight:.17g}",
        ])

    return command


def generate_standard_materials(
    material_exporter,
    wentzel_exporter,
):
    print()
    print("=" * 72)
    print("STANDARD GEANT4 MATERIALS")
    print("=" * 72)

    for material in MATERIALS:
        print(f"\n=== {material} ===")

        run([
            material_exporter,
            material,
        ])

        run([
            wentzel_exporter,
            material,
        ])


def generate_schneider_materials(
    material_exporter,
    wentzel_exporter,
):
    if not SCHNEIDER_CALIBRATION.exists():
        raise FileNotFoundError(
            "Schneider calibration file not found: "
            f"{SCHNEIDER_CALIBRATION}"
        )

    calibration = load_schneider_calibration(
        SCHNEIDER_CALIBRATION
    )

    elements = calibration["elements"]
    weights = calibration["material_weights"]

    if weights.shape[0] != 25:
        raise ValueError(
            "Expected 25 Schneider material classes, "
            f"found {weights.shape[0]}."
        )

    print()
    print("=" * 72)
    print("TOPAS SCHNEIDER MATERIALS")
    print("=" * 72)
    print(
        "Reference density: "
        f"{SCHNEIDER_REFERENCE_DENSITY_G_CM3} g/cm^3"
    )

    for tissue_id, tissue_weights in enumerate(
        weights
    ):
        name = f"SCHNEIDER_{tissue_id:02d}"

        print()
        print(
            f"=== {name} "
            f"(tissue ID {tissue_id}) ==="
        )

        material_command = (
            custom_material_command(
                material_exporter,
                name,
                SCHNEIDER_REFERENCE_DENSITY_G_CM3,
                elements,
                tissue_weights,
            )
        )

        wentzel_command = (
            custom_material_command(
                wentzel_exporter,
                name,
                SCHNEIDER_REFERENCE_DENSITY_G_CM3,
                elements,
                tissue_weights,
            )
        )

        run(material_command)
        run(wentzel_command)


def generate():
    material_exporter = executable(
        "geant4_material_export"
    )

    wentzel_exporter = executable(
        "geant4_wentzel_transport_export"
    )

    generate_standard_materials(
        material_exporter,
        wentzel_exporter,
    )

    generate_schneider_materials(
        material_exporter,
        wentzel_exporter,
    )


def main():
    clean_data()
    build()
    generate()

    print()
    print("Done.")
    print(f"Generated data: {DATA_DIR}")

    print()
    print("Expected Schneider tables:")
    print("  SCHNEIDER_00 ... SCHNEIDER_24")


if __name__ == "__main__":
    main()
