import math
from dataclasses import dataclass, field

from .beam import Beam
from .geometry import Geometry
from .enums import (
    Backend,
    SpatialScheme,
    EnergyScheme,
    WaveSpeeds,
    Reference,
    Scattering,
    CoefficientInterpolation,
)

@dataclass(frozen=True)
class Physics:
    scattering: Scattering = Scattering.OFF
    nuclear: bool = False
    # Interpolation of the tabulated Wentzel transport and nuclear removal
    # coefficients in the deterministic solver. LINEAR is the default and is
    # used for all benchmark results. PCHIP (C^1) is a diagnostic option.
    coefficient_interpolation: CoefficientInterpolation = (
        CoefficientInterpolation.LINEAR
    )


@dataclass(frozen=True)
class Numerics:
    spatial_scheme: SpatialScheme = SpatialScheme.WENO_Z
    energy_scheme: EnergyScheme = EnergyScheme.SSPRK3
    wave_speeds: WaveSpeeds = WaveSpeeds.EXACT

    max_energy_step: float = 0.25
    cfl: float = 0.99
    energy_min: float = 1e-4
    energy_window_sigma: float = 5.0

    # Performance
    active_domain: bool = False
    active_relative_tolerance: float = 1.0e-14

    # Compact moving active-domain transport.  This keeps the transported
    # M1 state on a small local box while the global dose remains full-grid.
    moving_active_domain: bool = False
    # Ignore numerically empty guard-cell directions when forming the CFL
    # bound for the compact moving-domain solver. Can be disabled for A/B
    # verification against the legacy all-cell CFL calculation.
    vacuum_aware_cfl: bool = False
    active_truncation_sigma: float = 5.0
    active_guard_cells: int = 9

    def __post_init__(self):
        """Validate numerical parameters."""

        if not all(math.isfinite(x) for x in (
            self.max_energy_step,
            self.cfl,
            self.energy_min,
            self.energy_window_sigma,
            self.active_relative_tolerance,
            self.active_truncation_sigma,
        )):
            raise ValueError("Numerical parameters must be finite.")

        if self.max_energy_step <= 0:
            raise ValueError("max_energy_step must be positive.")

        if not 0 < self.cfl <= 1:
            raise ValueError("cfl must lie in (0, 1].")

        if self.energy_min < 0:
            raise ValueError("energy_min cannot be negative.")
        
        if self.energy_window_sigma <= 0:
            raise ValueError("energy_window_sigma must be positive.")
        
        if self.active_relative_tolerance < 0:
            raise ValueError("active_relative_tolerance cannot be negative.")

        if self.active_truncation_sigma <= 0:
            raise ValueError("active_truncation_sigma must be positive.")

        if type(self.vacuum_aware_cfl) is not bool:
            raise ValueError("vacuum_aware_cfl must be a bool.")

        if type(self.active_guard_cells) is not int or self.active_guard_cells < 4:
            raise ValueError("active_guard_cells must be an integer >= 4.")


@dataclass(frozen=True)
class References:
    reference: Reference = Reference.ANALYTICAL
    additional_references: tuple[Reference, ...] = ()

    topas_histories: int = 1_000_000
    topas_seed: int = 1
    topas_force_rerun: bool = False

    # Matched particle Monte Carlo for the same reduced kinetic BFP model.
    # The homogeneous path preserves the reference used in the thesis
    # validation; heterogeneous runs use the solver's voxelized MaterialMap.
    bfp_mc_histories: int = 1_000_000
    bfp_mc_seed: int = 137
    bfp_mc_force_rerun: bool = False
    bfp_mc_threads: int = 12
    bfp_mc_max_step_mm: float = 0.25
    bfp_mc_max_energy_loss_mev: float = 0.25
    bfp_mc_max_scattering_optical_depth: float = 0.02
    bfp_mc_stop_energy_mev: float = 1.0

    def __post_init__(self):

        # topas validation

        if type(self.topas_histories) is not int or self.topas_histories <= 0:
            raise ValueError(
                "topas_histories must be a positive integer."
            )

        if type(self.topas_seed) is not int or self.topas_seed < 0:
            raise ValueError(
                "topas_seed must be a non-negative integer."
            )
        
        # BFP MC validation

        if type(self.bfp_mc_histories) is not int or self.bfp_mc_histories <= 0:
            raise ValueError(
                "bfp_mc_histories must be a positive integer."
            )

        if type(self.bfp_mc_seed) is not int or self.bfp_mc_seed < 0:
            raise ValueError(
                "bfp_mc_seed must be a non-negative integer."
            )

        if type(self.bfp_mc_force_rerun) is not bool:
            raise ValueError(
                "bfp_mc_force_rerun must be a bool."
            )

        if type(self.bfp_mc_threads) is not int or self.bfp_mc_threads <= 0:
            raise ValueError(
                "bfp_mc_threads must be a positive integer."
            )

        if not all(math.isfinite(x) and x > 0.0 for x in (
            self.bfp_mc_max_step_mm,
            self.bfp_mc_max_energy_loss_mev,
            self.bfp_mc_max_scattering_optical_depth,
            self.bfp_mc_stop_energy_mev,
        )):
            raise ValueError(
                "BFP Monte Carlo step controls must be finite and positive."
            )

@dataclass(frozen=True)
class OutputConfig:
    save_results: bool = False
    results_directory: str = "results"

    # Reconstructing a full 4-component final M1 state can dominate memory on
    # very large grids. Dose is always full-grid; phi/J can be omitted.
    store_final_state: bool = True


@dataclass
class SolverConfig:
    geometry: Geometry
    beam: Beam

    physics: Physics = field(default_factory=Physics)
    numerics: Numerics = field(default_factory=Numerics)

    backend: Backend = Backend.AUTO

    solve_deterministic: bool = True

    references: References = field(default_factory=References)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self):
        """Validate constraints involving multiple configuration components."""
        self._validate_beam_geometry()
        self._validate_energy_range()

    def _validate_beam_geometry(self):
        """
        Validate that the beam is positioned consistently with the geometry.

        Checks whether the beam center lies inside the transverse (x, y)
        extent of the computational domain.
        """
        x0, y0 = self.beam.center
        lx, ly, _ = self.geometry.size

        if not -lx / 2 <= x0 < lx / 2:
            raise ValueError("Beam center lies outside the x-domain.")

        if not -ly / 2 <= y0 < ly / 2:
            raise ValueError("Beam center lies outside the y-domain.")

    def _validate_energy_range(self):
        """Validate that the requested energy interval is non-empty."""

        if self.numerics.energy_min >= self.beam.energy:
            raise ValueError(
                "energy_min must be smaller than the initial beam energy."
            )