from typing import TYPE_CHECKING

import numpy as np
import pymedphys

if TYPE_CHECKING:
    from .result import Result


def _to_numpy(array):
    """Convert NumPy/CuPy arrays to NumPy."""

    if hasattr(array, "get"):
        return array.get()

    return np.asarray(array)


class ResultMetrics:
    """Diagnostics and reference metrics for a solver result."""

    def __init__(self, result: "Result"):
        self.result = result

    @property
    def phi(self):
        if self.result.phi is None:
            return None

        return _to_numpy(self.result.phi)

    @property
    def J(self):
        if self.result.J is None:
            return None

        return _to_numpy(self.result.J)

    @property
    def dose(self):
        if self.result.primary_dose is None:
            return None

        return _to_numpy(
            self.result.primary_dose
        )

    @property
    def x(self):
        return _to_numpy(self.result.x)

    @property
    def y(self):
        return _to_numpy(self.result.y)

    @property
    def z(self):
        return _to_numpy(self.result.z)

    # ---------------------------------------------------------
    # Deterministic diagnostics
    # ---------------------------------------------------------

    @property
    def all_phi_finite(self):
        if self.phi is None:
            return None

        return bool(
            np.all(np.isfinite(self.phi))
        )

    @property
    def all_J_finite(self):
        if self.J is None:
            return None

        return bool(
            np.all(np.isfinite(self.J))
        )

    @property
    def all_dose_finite(self):
        if self.dose is None:
            return None

        return bool(
            np.all(np.isfinite(self.dose))
        )

    @property
    def minimum_phi(self):
        if self.phi is None:
            return None

        return float(np.min(self.phi))

    @property
    def maximum_phi(self):
        if self.phi is None:
            return None

        return float(np.max(self.phi))

    @property
    def minimum_dose(self):
        if self.dose is None:
            return None

        return float(np.min(self.dose))

    @property
    def maximum_dose(self):
        if self.dose is None:
            return None

        return float(np.max(self.dose))

    @property
    def maximum_reduced_flux(self):
        if (
            self.phi is None
            or self.J is None
        ):
            return None

        J_norm = np.sqrt(
            np.sum(self.J**2, axis=0)
        )

        mask = self.phi > 0.0

        if not np.any(mask):
            return 0.0

        return float(
            np.max(
                J_norm[mask]
                / self.phi[mask]
            )
        )

    # ---------------------------------------------------------
    # Dose metrics
    # ---------------------------------------------------------

    @staticmethod
    def _axis_cell_index(centres) -> int:
        """Return the index of the cell that contains the coordinate 0.

        Cells are half-open intervals [c_i - h/2, c_i + h/2), the same
        convention as the material assignment. With an even number of cells
        the beam axis lies on a cell face and therefore belongs to the cell
        on its positive side. This replaces a nearest-centre search, whose
        result for two equidistant centres depended on tie-breaking.
        """

        centres = np.asarray(centres, dtype=float)

        if centres.size == 1:
            return 0

        width = float(centres[1] - centres[0])
        first_lower_face = float(centres[0]) - 0.5 * width

        index = int(
            np.floor((0.0 - first_lower_face) / width + 1e-9)
        )

        return min(max(index, 0), centres.size - 1)

    @property
    def central_axis_indices(self):
        return (
            self._axis_cell_index(self.x),
            self._axis_cell_index(self.y),
        )

    @property
    def central_axis_dose(self):
        if self.dose is None:
            return None

        ix, iy = self.central_axis_indices

        return self.dose[
            ix,
            iy,
            :,
        ]

    @property
    def peak_index(self):
        if self.central_axis_dose is None:
            return None

        return int(
            np.argmax(
                self.central_axis_dose
            )
        )

    @property
    def peak_depth(self):
        if self.peak_index is None:
            return None

        return float(
            self.z[self.peak_index]
        )

    @property
    def peak_dose(self):
        if self.peak_index is None:
            return None

        return float(
            self.central_axis_dose[
                self.peak_index
            ]
        )

    # ---------------------------------------------------------
    # Reference metrics
    # ---------------------------------------------------------

    @property
    def reference_maximum_dose(self):
        if self.result.reference_dose is None:
            return None

        reference = _to_numpy(
            self.result.reference_dose
        )

        return float(np.max(reference))


    @property
    def maximum_dose_relative_difference(self):
        reference = self.reference_maximum_dose

        if (
            reference is None
            or reference == 0.0
            or self.maximum_dose is None
        ):
            return None

        return (
            self.maximum_dose
            - reference
        ) / reference

    @property
    def reference_central_axis_dose(self):
        if self.result.reference_dose is None:
            return None

        reference = _to_numpy(
            self.result.reference_dose
        )

        ix, iy = self.central_axis_indices

        return reference[
            ix,
            iy,
            :,
        ]

    @property
    def reference_peak_index(self):
        dose = self.reference_central_axis_dose

        if dose is None:
            return None

        return int(np.argmax(dose))

    @property
    def reference_peak_depth(self):
        index = self.reference_peak_index

        if index is None:
            return None

        return float(
            self.z[index]
        )

    @property
    def reference_peak_dose(self):
        dose = self.reference_central_axis_dose
        index = self.reference_peak_index

        if (
            dose is None
            or index is None
        ):
            return None

        return float(
            dose[index]
        )

    @property
    def peak_depth_difference(self):
        if (
            self.peak_depth is None
            or self.reference_peak_depth is None
        ):
            return None

        return (
            self.peak_depth
            - self.reference_peak_depth
        )

    @property
    def peak_dose_relative_error(self):
        reference = self.reference_peak_dose

        if (
            reference is None
            or reference == 0.0
            or self.peak_dose is None
        ):
            return None

        return (
            self.peak_dose
            - reference
        ) / reference
    
    # ---------------------------------------------------------
    # Range metrics from the integrated depth dose
    # ---------------------------------------------------------

    @staticmethod
    def integrated_depth_dose(dose):
        """Laterally integrated depth dose I(z), up to the constant dx*dy."""

        return np.asarray(dose, dtype=float).sum(axis=(0, 1))

    @staticmethod
    def distal_peak_index(curve, window=10, min_fraction=0.3):
        """Index of the deepest peak of a depth-dose curve.

        A peak is a cell whose value is the largest within +-window cells
        and at least min_fraction of the global maximum. For a single Bragg
        peak this is the global maximum. When the integrated depth dose has
        two peaks (lateral interface: a shallower bone-side and a deeper
        water-side peak), the range metrics refer to the deeper one.
        """

        curve = np.asarray(curve, dtype=float)
        threshold = min_fraction * float(np.max(curve))
        n = len(curve)

        for i in range(n - 1, -1, -1):
            if curve[i] < threshold:
                continue
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            if curve[i] >= float(np.max(curve[lo:hi])):
                return i

        return int(np.argmax(curve))

    @staticmethod
    def distal_depth(z, curve, level, z_from=None):
        """Depth on the distal falloff where curve drops to level * peak.

        The peak is the maximum of the curve at depths z >= z_from (the
        whole curve if z_from is None). Linear interpolation between the
        cell centres. Returns None if the curve does not fall below the
        level inside the domain.
        """

        z = np.asarray(z, dtype=float)
        curve = np.asarray(curve, dtype=float)

        i_start = 0 if z_from is None else int(np.searchsorted(z, z_from))
        i_start = min(max(i_start, 0), len(curve) - 1)
        i_peak = i_start + int(np.argmax(curve[i_start:]))
        target = level * float(curve[i_peak])

        for i in range(i_peak, len(curve) - 1):
            if curve[i] >= target > curve[i + 1]:
                return float(
                    z[i]
                    + (target - curve[i])
                    * (z[i + 1] - z[i])
                    / (curve[i + 1] - curve[i])
                )

        return None

    # Range metrics are evaluated on the deepest Bragg peak of the reference.
    # The peak of the evaluated dose is searched from this many metres
    # upstream of it, so that a peak shifted upstream by numerical diffusion
    # is still found, while the shallower bone-side peak of the lateral
    # interface (about 65 mm upstream) is excluded.
    RANGE_SEARCH_UPSTREAM = 0.02

    def range_metrics(self, dose, z_from=None):
        """Return (R80, DFW) in metres for a dose array.

        R80: depth where the integrated depth dose falls to 80% of its peak
        on the distal side. DFW: distal falloff width R20 - R80. The peak is
        searched at z >= z_from.
        """

        if dose is None:
            return None, None

        idd = self.integrated_depth_dose(_to_numpy(dose))
        r80 = self.distal_depth(self.z, idd, 0.8, z_from)
        r20 = self.distal_depth(self.z, idd, 0.2, z_from)

        if r80 is None or r20 is None:
            return r80, None

        return r80, r20 - r80

    def range_comparison(self, reference):
        """R80 and DFW of the deterministic dose against a reference, in mm.

        Both are evaluated on the deepest Bragg peak of the reference
        (distal_peak_index). For the lateral interface this is the
        water-side peak.
        """

        reference = _to_numpy(reference)
        idd_ref = self.integrated_depth_dose(reference)
        z = np.asarray(self.z, dtype=float)
        z_from = (
            float(z[self.distal_peak_index(idd_ref)])
            - self.RANGE_SEARCH_UPSTREAM
        )

        r80, dfw = self.range_metrics(self.dose, z_from)
        r80_ref, dfw_ref = self.range_metrics(reference, z_from)

        def mm(value):
            return None if value is None else 1000.0 * value

        def diff(a, b):
            return None if a is None or b is None else 1000.0 * (a - b)

        return {
            "deterministic_r80_mm": mm(r80),
            "reference_r80_mm": mm(r80_ref),
            "r80_difference_mm": diff(r80, r80_ref),
            "deterministic_dfw_mm": mm(dfw),
            "reference_dfw_mm": mm(dfw_ref),
            "dfw_difference_mm": diff(dfw, dfw_ref),
            "peak_search_from_mm": 1000.0 * z_from,
        }

    def exact_reference_metrics(self, gamma_cutoff_percent: float = 10.0):
        """Metrics against the exact analytical solution, if it is stored.

        Available for the reduced model (CSDA + streaming), where the
        analytical solution is exact for the voxelized problem. The selected
        comparison reference (BFP Monte Carlo) is not affected.
        """

        analytical = self.result.analytical_dose

        if self.dose is None or analytical is None:
            return None

        analytical = _to_numpy(analytical)
        analytical_max = float(np.max(analytical))

        return {
            "reference": "Analytical",
            "maximum_dose": {
                "analytical_Gy": analytical_max,
                "relative_difference_percent": (
                    100.0
                    * (self.maximum_dose - analytical_max)
                    / analytical_max
                ),
            },
            "range": self.range_comparison(analytical),
            "gamma": {
                "3_percent_3_mm": self.gamma_pass_rate(
                    3.0, 3.0, gamma_cutoff_percent, analytical
                ),
                "2_percent_2_mm": self.gamma_pass_rate(
                    2.0, 2.0, gamma_cutoff_percent, analytical
                ),
                "1_percent_1_mm": self.gamma_pass_rate(
                    1.0, 1.0, gamma_cutoff_percent, analytical
                ),
            },
        }

    @property
    def speedup_vs_bfp_mc(self):
        """Return deterministic solver speed-up relative to BFP Monte Carlo."""

        deterministic_runtime = self.result.runtime
        bfp_mc_runtime = self.result.bfp_mc_runtime

        if (
            deterministic_runtime is None
            or bfp_mc_runtime is None
            or deterministic_runtime <= 0.0
            or bfp_mc_runtime <= 0.0
        ):
            return None

        return float(
            bfp_mc_runtime
            / deterministic_runtime
        )

    @property
    def speedup_vs_topas(self):
        """Return deterministic solver speed-up relative to TOPAS."""

        deterministic_runtime = self.result.runtime
        topas_runtime = self.result.topas_runtime

        if (
            deterministic_runtime is None
            or topas_runtime is None
            or deterministic_runtime <= 0.0
            or topas_runtime <= 0.0
        ):
            return None

        return float(
            topas_runtime
            / deterministic_runtime
        )
    
    def gamma_pass_rate(
        self,
        dose_percent_threshold: float,
        distance_mm_threshold: float,
        lower_percent_dose_cutoff: float = 10.0,
        reference=None,
    ):
        """Return global 3D gamma pass rate in percent.

        The reference defaults to the selected comparison reference
        (result.reference_dose). An explicit reference array can be passed,
        for example the exact analytical solution.
        """

        if reference is None:
            reference = self.result.reference_dose

        if (
            self.dose is None
            or reference is None
        ):
            return None

        dose = self.dose

        reference = _to_numpy(
            reference
        )

        axes_mm = (
            self.x * 1000.0,
            self.y * 1000.0,
            self.z * 1000.0,
        )

        reference_max = float(
            np.max(reference)
        )

        gamma = pymedphys.gamma(
            axes_reference=axes_mm,
            dose_reference=reference,
            axes_evaluation=axes_mm,
            dose_evaluation=dose,
            dose_percent_threshold=dose_percent_threshold,
            distance_mm_threshold=distance_mm_threshold,
            lower_percent_dose_cutoff=lower_percent_dose_cutoff,
            interp_fraction=10,
            max_gamma=1.1,
            local_gamma=False,
            global_normalisation=reference_max,
            skip_once_passed=True,
        )

        valid_gamma = gamma[
            ~np.isnan(gamma)
        ]

        if valid_gamma.size == 0:
            return None

        return float(
            100.0
            * np.mean(valid_gamma <= 1.0)
        )
    
    def comparison_metrics(
        self,
        gamma_cutoff_percent: float = 10.0,
    ):
        """Return standard thesis comparison metrics."""

        gamma_3_3 = self.gamma_pass_rate(
            3.0,
            3.0,
            gamma_cutoff_percent,
        )

        gamma_2_2 = self.gamma_pass_rate(
            2.0,
            2.0,
            gamma_cutoff_percent,
        )

        gamma_1_1 = self.gamma_pass_rate(
            1.0,
            1.0,
            gamma_cutoff_percent,
        )

        return {
            "reference": self.result.reference_label,

            "performance": {
                "deterministic_runtime_seconds": (
                    None
                    if self.result.runtime is None
                    else float(self.result.runtime)
                ),

                "topas_runtime_seconds": (
                    None
                    if self.result.topas_runtime is None
                    else float(self.result.topas_runtime)
                ),

                "bfp_mc_runtime_seconds": (
                    None
                    if self.result.bfp_mc_runtime is None
                    else float(self.result.bfp_mc_runtime)
                ),

                "speedup_vs_bfp_mc_x": (
                    "-"
                    if self.result.reference_label != "BFP Monte Carlo"
                    else (
                        None
                        if self.speedup_vs_bfp_mc is None
                        else float(self.speedup_vs_bfp_mc)
                    )
                ),

                "speedup_vs_topas_x": (
                    "-"
                    if self.result.reference_label != "TOPAS"
                    else (
                        None
                        if self.speedup_vs_topas is None
                        else float(self.speedup_vs_topas)
                    )
                ),
            },

            "gamma": {
                "normalization": "global_reference_maximum",
                "lower_dose_cutoff_percent": gamma_cutoff_percent,
                "interp_fraction": 10,

                "3_percent_3_mm": {
                    "pass_rate_percent": gamma_3_3,
                },

                "2_percent_2_mm": {
                    "pass_rate_percent": gamma_2_2,
                },

                "1_percent_1_mm": {
                    "pass_rate_percent": gamma_1_1,
                },
            },

            "maximum_dose": {
                "deterministic_Gy": self.maximum_dose,
                "reference_Gy": self.reference_maximum_dose,
                "relative_difference_percent": (
                    None
                    if self.maximum_dose_relative_difference is None
                    else (
                        100.0
                        * self.maximum_dose_relative_difference
                    )
                ),
            },

            "range": (
                None
                if self.result.reference_dose is None
                else self.range_comparison(
                    _to_numpy(self.result.reference_dose)
                )
            ),

            "exact_reference": (
                None
                if self.result.reference_label == "Analytical"
                else self.exact_reference_metrics(gamma_cutoff_percent)
            ),

            "central_axis_bragg_peak": {
                "deterministic_depth_mm": (
                    None
                    if self.peak_depth is None
                    else 1000.0 * self.peak_depth
                ),

                "reference_depth_mm": (
                    None
                    if self.reference_peak_depth is None
                    else 1000.0 * self.reference_peak_depth
                ),

                "depth_difference_mm": (
                    None
                    if self.peak_depth_difference is None
                    else 1000.0 * self.peak_depth_difference
                ),
            },
        }


    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self):
        """Print standard diagnostics and dose comparison metrics."""

        print("Finished")

        # Only print transport diagnostics when the
        # deterministic solver was actually executed.
        if self.result.dose is not None:
            print(
                "Energy steps:",
                len(self.result.energies) - 1,
            )

            print(
                "Final energy:",
                self.result.energies[-1],
            )

            print(
                "Runtime:",
                self.result.runtime,
            )

            if self.result.bfp_mc_runtime is not None:
                print(
                    "BFP MC runtime:",
                    self.result.bfp_mc_runtime,
                )

            if self.result.topas_runtime is not None:
                print(
                    "TOPAS runtime:",
                    self.result.topas_runtime,
                )

            print(
                "All phi finite:",
                self.all_phi_finite,
            )

            print(
                "All J finite:",
                self.all_J_finite,
            )

            print(
                "Minimum phi:",
                self.minimum_phi,
            )

            print(
                "Maximum phi:",
                self.maximum_phi,
            )

            print(
                "Maximum reduced flux:",
                self.maximum_reduced_flux,
            )

        # Dose diagnostics work for deterministic, BFP MC,
        # TOPAS, or analytical primary solutions.
        print()
        print(
            f"{self.result.primary_label} dose"
        )

        print(
            "All dose finite:",
            self.all_dose_finite,
        )

        print(
            "Minimum dose:",
            self.minimum_dose,
        )

        print(
            "Maximum dose:",
            self.maximum_dose,
        )

        print()
        print("Central-axis dose")

        print(
            f"{self.result.primary_label} peak depth:",
            self.peak_depth,
        )

        print(
            f"{self.result.primary_label} peak dose:",
            self.peak_dose,
        )

        if self.result.reference_dose is not None:
            label = self.result.reference_label

            print(
                f"{label} peak depth:",
                self.reference_peak_depth,
            )

            print(
                f"{label} peak dose:",
                self.reference_peak_dose,
            )

            print(
                "Peak depth difference:",
                self.peak_depth_difference,
            )

            print(
                "Peak dose relative error:",
                100*self.peak_dose_relative_error, " %"
            )