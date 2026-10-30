from dataclasses import dataclass
from pathlib import Path
from ..enums import Scattering
from ..config import SolverConfig


@dataclass(frozen=True)
class TopasCase:
    """Files defining one generated TOPAS reference case."""

    directory: Path
    parameter_file: Path
    preinit_macro: Path
    sequence_macro: Path


def write_topas_case(
    config: SolverConfig,
    directory: Path,
) -> TopasCase:
    """Generate a TOPAS case matching the solver configuration."""

    physics = config.physics

    if physics.nuclear:
        raise NotImplementedError(
            "TOPAS nuclear physics is not implemented yet."
        )

    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameter_file = directory / "parameters.txt"
    preinit_macro = directory / "preinit.mac"
    sequence_macro = directory / "sequence.mac"

    _write_parameter_file(
        config,
        parameter_file,
    )

    _write_preinit_macro(
        config,
        preinit_macro,
    )

    _write_sequence_macro(
        config,
        sequence_macro,
    )

    return TopasCase(
        directory=directory,
        parameter_file=parameter_file,
        preinit_macro=preinit_macro,
        sequence_macro=sequence_macro,
    )


def _box_geometry_parameters(
    config: SolverConfig,
) -> str:
    """Generate TOPAS geometry parameters for Box objects."""

    geometry = config.geometry
    _, _, lz = geometry.size

    blocks = []

    for i, box in enumerate(geometry.objects):
        name = f"Box{i}"

        center_x = 0.5 * (box.x_min + box.x_max)
        center_y = 0.5 * (box.y_min + box.y_max)
        center_z = 0.5 * (box.z_min + box.z_max)

        half_x = 0.5 * (box.x_max - box.x_min)
        half_y = 0.5 * (box.y_max - box.y_min)
        half_z = 0.5 * (box.z_max - box.z_min)

        # Geometry uses global coordinates with z = 0 at the phantom
        # entrance. TOPAS child translations are relative to the
        # centre of the parent Phantom.
        local_z = center_z - 0.5 * lz

        blocks.append(
            f"""
s:Ge/{name}/Type = "TsBox"
s:Ge/{name}/Parent = "Phantom"
s:Ge/{name}/Material = "{box.material.value}"

d:Ge/{name}/HLX = {half_x:.12g} m
d:Ge/{name}/HLY = {half_y:.12g} m
d:Ge/{name}/HLZ = {half_z:.12g} m

d:Ge/{name}/TransX = {center_x:.12g} m
d:Ge/{name}/TransY = {center_y:.12g} m
d:Ge/{name}/TransZ = {local_z:.12g} m

d:Ge/{name}/MaxStepSize = 1.0 mm
""".strip()
        )

    return "\n\n".join(blocks)


def _write_parameter_file(
    config: SolverConfig,
    output_path: Path,
):
    """Write the main TOPAS parameter file."""

    geometry = config.geometry
    beam = config.beam
    references = config.references

    lx, ly, lz = geometry.size
    nx, ny, nz = geometry.resolution

    sigma_x, sigma_y = beam.sigma
    center_x, center_y = beam.center

    # TOPAS expects relative energy spread in percent.
    energy_spread_percent = (
        100.0
        * beam.energy_spread
        / beam.energy
    )

    cutoff_x = (
        beam.cutoff_sigma
        * sigma_x
    )

    cutoff_y = (
        beam.cutoff_sigma
        * sigma_y
    )

    material = geometry.background_material.value

    box_parameters = _box_geometry_parameters(
        config
    )

    # Zero angular spread means an exactly parallel beam.
    if beam.angular_spread == 0.0:
        angular_parameters = """
s:So/Beam/BeamAngularDistribution = "None"
"""
    else:
        angular_parameters = f"""
s:So/Beam/BeamAngularDistribution = "Gaussian"

d:So/Beam/BeamAngularSpreadX = {beam.angular_spread:.12g} rad
d:So/Beam/BeamAngularSpreadY = {beam.angular_spread:.12g} rad

d:So/Beam/BeamAngularCutoffX = 90. deg
d:So/Beam/BeamAngularCutoffY = 90. deg
"""

    text = f"""
# ------------------------------------------------------------
# Auto-generated TOPAS reference case
# ------------------------------------------------------------

# General control
i:Ts/NumberOfThreads = 0
i:Ts/Seed = {references.topas_seed}
i:Ts/ShowHistoryCountAtInterval = 100000

b:Ts/PauseBeforeInit = "True"
b:Ts/PauseBeforeSequence = "True"
b:Ts/PauseBeforeQuit = "False"

# ------------------------------------------------------------
# World
# ------------------------------------------------------------

s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 50. cm
d:Ge/World/HLY = 50. cm
d:Ge/World/HLZ = 50. cm
b:Ge/World/Invisible = "True"

# ------------------------------------------------------------
# Phantom
# ------------------------------------------------------------

s:Ge/Phantom/Type = "TsBox"
s:Ge/Phantom/Parent = "World"
s:Ge/Phantom/Material = "{material}"

d:Ge/Phantom/HLX = {0.5 * lx:.12g} m
d:Ge/Phantom/HLY = {0.5 * ly:.12g} m
d:Ge/Phantom/HLZ = {0.5 * lz:.12g} m

d:Ge/Phantom/TransX = 0. m
d:Ge/Phantom/TransY = 0. m
d:Ge/Phantom/TransZ = {0.5 * lz:.12g} m

s:Ge/Phantom/Color = "blue"

d:Ge/Phantom/MaxStepSize = 1.0 mm

# ------------------------------------------------------------
# Embedded geometry
# ------------------------------------------------------------

{box_parameters}

# ------------------------------------------------------------
# Beam position
# ------------------------------------------------------------

s:Ge/BeamPosition/Parent = "World"
s:Ge/BeamPosition/Type = "Group"

d:Ge/BeamPosition/TransX = {center_x:.12g} m
d:Ge/BeamPosition/TransY = {center_y:.12g} m
d:Ge/BeamPosition/TransZ = 0. m

d:Ge/BeamPosition/RotX = 0. deg
d:Ge/BeamPosition/RotY = 0. deg
d:Ge/BeamPosition/RotZ = 0. deg

# ------------------------------------------------------------
# Proton beam
# ------------------------------------------------------------

s:So/Beam/Type = "Beam"
s:So/Beam/Component = "BeamPosition"
s:So/Beam/BeamParticle = "proton"

i:So/Beam/NumberOfHistoriesInRun = {references.topas_histories}

d:So/Beam/BeamEnergy = {beam.energy:.12g} MeV
u:So/Beam/BeamEnergySpread = {energy_spread_percent:.12g}

s:So/Beam/BeamPositionDistribution = "Gaussian"
s:So/Beam/BeamPositionCutoffShape = "Ellipse"

d:So/Beam/BeamPositionSpreadX = {sigma_x:.12g} m
d:So/Beam/BeamPositionSpreadY = {sigma_y:.12g} m

d:So/Beam/BeamPositionCutoffX = {cutoff_x:.12g} m
d:So/Beam/BeamPositionCutoffY = {cutoff_y:.12g} m

{angular_parameters}

# ------------------------------------------------------------
# Physics
# ------------------------------------------------------------

sv:Ph/Default/Modules = 1 "g4em-standard_opt4"

d:Ph/Default/CutForElectron = 100. m
d:Ph/Default/CutForPositron = 100. m
d:Ph/Default/CutForGamma = 100. m

# ------------------------------------------------------------
# Dose scorer
# ------------------------------------------------------------

s:Sc/DoseAtPhantom/Quantity = "DoseToMedium"
s:Sc/DoseAtPhantom/Component = "Phantom"

i:Sc/DoseAtPhantom/XBins = {nx}
i:Sc/DoseAtPhantom/YBins = {ny}
i:Sc/DoseAtPhantom/ZBins = {nz}

s:Sc/DoseAtPhantom/IfOutputFileAlreadyExists = "Overwrite"
s:Sc/DoseAtPhantom/OutputFile = "DoseAtPhantom"
s:Sc/DoseAtPhantom/OutputType = "CSV"
b:Sc/DoseAtPhantom/OutputToConsole = "False"

b:Sc/DoseAtPhantom/PropagateToChildren = "True"

sv:Sc/DoseAtPhantom/Report = 1 "Sum"
"""

    output_path.write_text(
        text.strip() + "\n",
        encoding="utf-8",
    )


def _write_preinit_macro(
    config: SolverConfig,
    output_path: Path,
):
    """Write commands that must run before Geant4 physics initialization."""

    geometry = config.geometry

    materials = tuple(
        dict.fromkeys(
            [
                geometry.background_material,
                *(obj.material for obj in geometry.objects),
            ]
        )
    )

    commands = [
        "/process/eLoss/fluct false",
    ]

    # if config.physics.scattering is not Scattering.OFF:
    #     commands.append(
    #         "/process/msc/MuHadLateralDisplacement false"
    #     )

    for material in materials:
        commands.append(
            f"/material/g4/printMaterial {material.value}"
        )

    output_path.write_text(
        "\n".join(commands) + "\n",
        encoding="utf-8",
    )


def _write_sequence_macro(
    config: SolverConfig,
    output_path: Path,
):
    """Match TOPAS scattering physics to the deterministic model."""

    commands = []

    # With scattering disabled, remove multiple scattering entirely.
    if config.physics.scattering is Scattering.OFF:
        commands.append(
            "/process/inactivate msc proton"
        )

    # Explicit hard Coulomb scattering is not represented by the
    # deterministic first-transport-coefficient source.
    commands.append(
        "/process/inactivate CoulombScat proton"
    )

    # Nuclear stopping is outside the current deterministic model.
    commands.append(
        "/process/inactivate nuclearStopping proton"
    )

    output_path.write_text(
        "\n".join(commands) + "\n",
        encoding="utf-8",
    )