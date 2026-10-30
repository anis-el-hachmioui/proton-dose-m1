from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .result import Result


def _to_numpy(array):
    """Convert NumPy/CuPy arrays to NumPy."""

    if hasattr(array, "get"):
        return array.get()

    return np.asarray(array)


class ResultPlots:
    """Standard plots for a solver result."""

    def __init__(self, result: "Result"):
        self.result = result

    @property
    def dose(self):
        if self.result.primary_dose is None:
            return None

        return _to_numpy(
            self.result.primary_dose
        )

    @property
    def reference_dose(self):
        if self.result.reference_dose is None:
            return None

        return _to_numpy(
            self.result.reference_dose
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

    def dose_heatmap(
        self,
        save_path=None,
    ):
        """Plot the central XZ dose distribution."""

        if self.dose is None:
            raise ValueError(
                "No dose distribution is available for plotting."
            )

        iy = np.argmin(
            np.abs(self.y)
        )

        dose_xz = self.dose[:, iy, :]

        extent = [
            self.z[0] * 1000.0,
            self.z[-1] * 1000.0,
            self.x[0] * 1000.0,
            self.x[-1] * 1000.0,
        ]

        reference = self.reference_dose

        if reference is None:
            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            image = ax.imshow(
                dose_xz,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="inferno",
            )

            fig.colorbar(
                image,
                ax=ax,
                label="Dose [Gy]",
            )

            ax.set_xlabel(
                "Depth z [mm]"
            )

            ax.set_ylabel(
                "x [mm]"
            )

            ax.set_title(
                f"{self.result.primary_label} dose distribution"            
                )

            fig.tight_layout()

        else:
            reference_xz = reference[
                :, iy, :
            ]

            # Use the same dose scale for both solutions.
            vmax = max(
                np.max(dose_xz),
                np.max(reference_xz),
            )

            # Difference expressed relative to the global
            # maximum of the reference dose.
            reference_max = np.max(
                reference_xz
            )

            if reference_max > 0.0:
                difference_xz = (
                    100.0
                    * (dose_xz - reference_xz)
                    / reference_max
                )
            else:
                difference_xz = np.zeros_like(
                    dose_xz
                )

            difference_limit = np.max(
                np.abs(difference_xz)
            )

            if difference_limit == 0.0:
                difference_limit = 1.0

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(17, 5),
                constrained_layout=True,
            )

            # Primary solution
            image0 = axes[0].imshow(
                dose_xz,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="inferno",
                vmin=0.0,
                vmax=vmax,
            )

            axes[0].set_title(
                self.result.primary_label
            )

            # Reference solution
            image1 = axes[1].imshow(
                reference_xz,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="inferno",
                vmin=0.0,
                vmax=vmax,
            )

            axes[1].set_title(
                self.result.reference_label
            )

            # Difference
            image2 = axes[2].imshow(
                difference_xz,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="coolwarm",
                vmin=-difference_limit,
                vmax=difference_limit,
            )

            axes[2].set_title(
                f"{self.result.primary_label} - "
                f"{self.result.reference_label}"
            )

            for ax in axes:
                ax.set_xlabel(
                    "Depth z [mm]"
                )

                ax.set_ylabel(
                    "x [mm]"
                )

            fig.colorbar(
                image1,
                ax=axes[:2],
                label="Dose [Gy]",
            )

            fig.colorbar(
                image2,
                ax=axes[2],
                label=(
                    "Difference "
                    "[% of reference maximum]"
                ),
            )

        if save_path is not None:
            fig.savefig(
                save_path,
                dpi=200,
                bbox_inches="tight",
            )

        plt.show()
        plt.close(fig)

    def central_axis_dose(
        self,
        save_path=None,
    ):
        """Plot deterministic and reference central-axis depth-dose."""

        ix = np.argmin(
            np.abs(self.x)
        )

        iy = np.argmin(
            np.abs(self.y)
        )

        central_dose = self.dose[
            ix,
            iy,
            :,
        ]

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.plot(
            self.z * 1000.0,
            central_dose,
            linewidth=2,
            label=self.result.primary_label,
        )

        reference = self.reference_dose

        if reference is not None:
            central_reference = reference[
                ix,
                iy,
                :,
            ]

            ax.plot(
                self.z * 1000.0,
                central_reference,
                "--",
                linewidth=2,
                label=self.result.reference_label,
            )

            ax.legend()

        ax.set_xlabel(
            "Depth z [mm]"
        )

        ax.set_ylabel(
            "Dose [Gy]"
        )

        ax.set_title(
            "Central-axis depth-dose"
        )

        ax.grid(alpha=0.3)

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(
                save_path,
                dpi=200,
                bbox_inches="tight",
            )

        plt.show()
        plt.close(fig)

    def idd(
        self,
        save_path=None,
    ):
        """Plot integrated depth-dose."""

        idd = np.trapezoid(
            np.trapezoid(
                self.dose,
                x=self.y,
                axis=1,
            ),
            x=self.x,
            axis=0,
        )

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.plot(
            self.z * 1000.0,
            idd,
            linewidth=2,
            label=self.result.primary_label,        
            )

        reference = self.reference_dose

        if reference is not None:
            idd_reference = np.trapezoid(
                np.trapezoid(
                    reference,
                    x=self.y,
                    axis=1,
                ),
                x=self.x,
                axis=0,
            )

            ax.plot(
                self.z * 1000.0,
                idd_reference,
                "--",
                linewidth=2,
                label=self.result.reference_label,
            )

            ax.legend()

        ax.set_xlabel(
            "Depth z [mm]"
        )

        ax.set_ylabel(
            r"Integrated dose [Gy m$^2$]"
        )

        ax.set_title(
            "Integrated depth-dose"
        )

        ax.grid(alpha=0.3)

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(
                save_path,
                dpi=200,
                bbox_inches="tight",
            )

        plt.show()
        plt.close(fig)

    def all(self):
        """Show all standard plots and save them when the result was persisted."""

        directory = self.result.run_directory

        if directory is None:
            self.dose_heatmap()
            self.central_axis_dose()
            self.idd()
            return

        directory = Path(directory)

        primary = (
            self.result.primary_label
            .lower()
            .replace(" ", "_")
        )

        reference = self.result.reference_label

        if reference is not None:
            reference = (
                reference
                .lower()
                .replace(" ", "_")
            )

            prefix = (
                f"{primary}_vs_{reference}"
            )
        else:
            prefix = primary

        self.dose_heatmap(
            directory
            / f"{prefix}_heatmap.png"
        )

        self.central_axis_dose(
            directory
            / f"{prefix}_central_axis.png"
        )

        self.idd(
            directory
            / f"{prefix}_idd.png"
        )