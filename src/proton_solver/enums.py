from enum import Enum, auto


class Backend(Enum):
    AUTO = auto()
    CPU = auto()
    GPU = auto()

class Material(Enum):
    WATER = "G4_WATER"
    BONE_COMPACT = "G4_BONE_COMPACT_ICRU"
    LUNG = "G4_LUNG_ICRP"


class SpatialScheme(Enum):
    FIRST_ORDER = auto()
    MUSCL = auto()
    WENO3 = auto()
    WENO_Z = auto()


class EnergyScheme(Enum):
    FORWARD_EULER = auto()
    BACKWARD_EULER = auto()
    TRAPEZOIDAL = auto()
    SSPRK2 = auto()
    SSPRK3 = auto()
    IF_SSPRK3 = auto()


class WaveSpeeds(Enum):
    EXACT = auto()
    UNIT = auto()
    
class Reference(Enum):
    NONE = auto()
    ANALYTICAL = auto()
    TOPAS = auto()
    BFP_MC = auto()
    SELF = auto()

class CoefficientInterpolation(Enum):
    """Interpolation of the tabulated scattering and removal coefficients."""

    LINEAR = auto()
    PCHIP = auto()

class Scattering(Enum):
    OFF = auto()
    ROSSI = auto()
    WENTZEL = auto()