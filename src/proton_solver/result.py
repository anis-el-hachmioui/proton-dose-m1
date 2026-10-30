from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from .enums import Backend
from .metrics import ResultMetrics
from .plots import ResultPlots


@dataclass
class Result:
    phi: Any
    J: Any
    dose: Any

    grid: Any
    energies: Any
    snapshots: dict[float, Any] = field(default_factory=dict)

    analytical_dose: Any = None
    topas_dose: Any = None
    bfp_mc_dose: Any = None

    runtime: float | None = None
    topas_runtime: float | None = None
    bfp_mc_runtime: float | None = None
    backend: Backend | None = None
    run_directory: Any = None

    @property
    def x(self):
        return self.grid.x

    @property
    def y(self):
        return self.grid.y

    @property
    def z(self):
        return self.grid.z

    # ---------------------------------------------------------
    # Primary solution
    # ---------------------------------------------------------

    @property
    def primary_dose(self):
        """Return the dose treated as the primary solution."""

        # Deterministic solution always remains primary when present.
        if self.dose is not None:
            return self.dose

        # Reference-only calculations.
        # Prefer the internal matched BFP Monte Carlo over TOPAS.
        if self.bfp_mc_dose is not None:
            return self.bfp_mc_dose

        if self.topas_dose is not None:
            return self.topas_dose

        if self.analytical_dose is not None:
            return self.analytical_dose

        return None

    @property
    def primary_label(self):
        """Return the name of the primary dose."""

        if self.dose is not None:
            return "Deterministic"

        if self.bfp_mc_dose is not None:
            return "BFP Monte Carlo"

        if self.topas_dose is not None:
            return "TOPAS"

        if self.analytical_dose is not None:
            return "Analytical"

        return None

    # ---------------------------------------------------------
    # Reference solution
    # ---------------------------------------------------------

    @property
    def reference_dose(self):
        """Return the dose used as the comparison reference."""

        # Reference-only validation:
        # when a Monte Carlo result is compared with the analytical
        # solution, the MC result is primary and analytical is reference.
        if (
            self.dose is None
            and self.analytical_dose is not None
            and (
                self.bfp_mc_dose is not None
                or self.topas_dose is not None
            )
        ):
            return self.analytical_dose

        # Normal deterministic comparisons.
        # Prefer the matched BFP MC reference over TOPAS.
        if self.bfp_mc_dose is not None:
            return self.bfp_mc_dose

        if self.topas_dose is not None:
            return self.topas_dose

        if self.analytical_dose is not None:
            return self.analytical_dose

        return None

    @property
    def reference_label(self):
        """Return the selected reference name."""

        if (
            self.dose is None
            and self.analytical_dose is not None
            and (
                self.bfp_mc_dose is not None
                or self.topas_dose is not None
            )
        ):
            return "Analytical"

        if self.bfp_mc_dose is not None:
            return "BFP Monte Carlo"

        if self.topas_dose is not None:
            return "TOPAS"

        if self.analytical_dose is not None:
            return "Analytical"

        return None
  

    @cached_property
    def metrics(self):
        """Return diagnostics and comparison metrics."""
        return ResultMetrics(self)

    @cached_property
    def plots(self):
        """Return plotting utilities for this result."""
        return ResultPlots(self)