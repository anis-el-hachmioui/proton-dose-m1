import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Beam:
    n_protons: float = 1e5
    energy: float = 50.0                            # MeV
    energy_spread: float = 1.0                      # MeV
    sigma: tuple[float, float] = (0.001, 0.001)     # m
    center: tuple[float, float] = (0.0, 0.0)        # m
    angular_spread: float = 0.0                     # rad
    cutoff_sigma: float = 5.0                       # [-]

    def __post_init__(self):
        """Validate beam parameters independent of the geometry."""

        # Validate that sigma and center contain exactly two values        
        if len(self.sigma) != 2 or len(self.center) != 2:
            raise ValueError("sigma and center must contain exactly two values.")

        # Validate that all parameters are finite numbers
        if not all(math.isfinite(x) for x in (
            self.n_protons, self.energy, self.energy_spread,
            *self.sigma, *self.center,
            self.angular_spread, self.cutoff_sigma,
        )):
            raise ValueError("Beam parameters must be finite.")

        # Validate that n_protons and energy are positive
        if self.n_protons <= 0 or self.energy <= 0:
            raise ValueError("n_protons and energy must be positive.")

        # Validate that energy_spread and angular_spread are non-negative
        if self.energy_spread < 0 or self.angular_spread < 0:
            raise ValueError("Beam spreads cannot be negative.")

        # Validate that sigma and cutoff_sigma are positive
        if min(self.sigma) <= 0 or self.cutoff_sigma <= 0:
            raise ValueError("sigma and cutoff_sigma must be positive.")