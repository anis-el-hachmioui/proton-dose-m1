from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import json
import matplotlib.pyplot as plt
import numpy as np

from .config import SolverConfig
from .enums import Reference
from .solver import Solver


# ============================================================
# Observables
# ============================================================


@dataclass(frozen=True)
class ObservableSpec:
    name: str
    energy: float | None = None

    @property
    def label(self) -> str:
        if self.energy is None:
            return self.name

        return f"{self.name}({self.energy:g} MeV)"


class Observable(Enum):
    DOSE = "dose"
    PHI = "phi"
    JX = "jx"
    JY = "jy"
    JZ = "jz"

    def at_energy(
        self,
        energy: float,
    ) -> ObservableSpec:

        if self is Observable.DOSE:
            raise ValueError(
                "Dose is integrated over energy and cannot "
                "be evaluated at a single energy."
            )

        return ObservableSpec(
            name=self.value,
            energy=float(energy),
        )


def _observable_spec(
    observable: Observable | ObservableSpec,
) -> ObservableSpec:

    if isinstance(observable, ObservableSpec):
        return observable

    if observable is Observable.DOSE:
        return ObservableSpec(
            name="dose",
        )

    raise ValueError(
        f"{observable.value} requires an energy. "
        f"Use Observable.{observable.name}.at_energy(...)."
    )


# ============================================================
# Refinement
# ============================================================


@dataclass(frozen=True)
class Refinement:
    axis: str
    values: tuple[int, ...]
    couple_energy_step: bool = False

    @classmethod
    def nz(
        cls,
        values,
        couple_energy_step: bool = False,
    ) -> "Refinement":

        values = tuple(
            int(value)
            for value in values
        )

        if len(values) < 3:
            raise ValueError(
                "A self-convergence study requires "
                "at least three resolutions."
            )

        if any(
            value <= 0
            for value in values
        ):
            raise ValueError(
                "Grid resolutions must be positive."
            )

        if any(
            values[i + 1] <= values[i]
            for i in range(len(values) - 1)
        ):
            raise ValueError(
                "Grid resolutions must be strictly increasing."
            )

        return cls(
            axis="z",
            values=values,
            couple_energy_step=couple_energy_step,
        )

    @classmethod
    def energy(
        cls,
        values,
    ) -> "Refinement":
        """
        Energy-only refinement on a fixed spatial grid.

        values are integers k, increasing and nested (e.g. 160, 320, 640).
        Level k uses the energy-step cap

            max_energy_step * values[0] / k,

        so the cap of the base configuration belongs to the first level.
        The grid is the same on every level; no restriction is applied.
        """

        values = tuple(
            int(value)
            for value in values
        )

        if len(values) < 3:
            raise ValueError(
                "A self-convergence study requires "
                "at least three levels."
            )

        if any(
            values[i + 1] <= values[i]
            for i in range(len(values) - 1)
        ):
            raise ValueError(
                "Energy refinement values must be strictly increasing."
            )

        return cls(
            axis="energy",
            values=values,
        )

    @property
    def refinement_ratio(self) -> int:

        ratios = []

        for coarse, fine in zip(
            self.values[:-1],
            self.values[1:],
        ):
            if fine % coarse != 0:
                raise ValueError(
                    "Self-convergence requires nested grids. "
                    f"{fine} is not an integer multiple of {coarse}."
                )

            ratio = (
                fine
                // coarse
            )

            if ratio <= 1:
                raise ValueError(
                    "Refinement ratio must be greater than one."
                )

            ratios.append(ratio)

        if len(set(ratios)) != 1:
            raise ValueError(
                "Self-convergence currently requires a constant "
                "refinement ratio, for example "
                "[40, 80, 160, 320]."
            )

        return ratios[0]

    def config_for(
        self,
        config: SolverConfig,
        value: int,
    ) -> SolverConfig:

        if self.axis == "energy":
            return replace(
                config,
                geometry=replace(
                    config.geometry,
                    view_geometry=False,
                ),
                numerics=replace(
                    config.numerics,
                    max_energy_step=(
                        config.numerics.max_energy_step
                        * self.values[0]
                        / value
                    ),
                ),
            )

        if self.axis != "z":
            raise NotImplementedError(
                f"Unsupported refinement axis: {self.axis}"
            )

        nx, ny, _ = (
            config.geometry.resolution
        )

        geometry = replace(
            config.geometry,
            resolution=(
                nx,
                ny,
                value,
            ),
            view_geometry=False,
        )

        numerics = (
            config.numerics
        )

        if self.couple_energy_step:
            base_value = (
                self.values[0]
            )

            max_energy_step = (
                config.numerics.max_energy_step
                * base_value
                / value
            )

            numerics = replace(
                numerics,
                max_energy_step=max_energy_step,
            )

        return replace(
            config,
            geometry=geometry,
            numerics=numerics,
        )

    def h(
        self,
        config: SolverConfig,
        value: int,
    ) -> float:

        if self.axis == "z":
            return (
                config.geometry.size[2]
                / value
            )

        if self.axis == "energy":
            return (
                config.numerics.max_energy_step
                * self.values[0]
                / value
            )

        raise NotImplementedError(
            f"Unsupported refinement axis: {self.axis}"
        )


# ============================================================
# Regions
# ============================================================


@dataclass(frozen=True)
class Region:
    kind: str
    z_min: float | None = None
    z_max: float | None = None

    @classmethod
    def central_axis(
        cls,
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> "Region":

        if z_min is not None and z_min < 0.0:
            raise ValueError(
                "z_min cannot be negative."
            )

        if z_max is not None and z_max <= 0.0:
            raise ValueError(
                "z_max must be positive."
            )

        if (
            z_min is not None
            and z_max is not None
            and z_min >= z_max
        ):
            raise ValueError(
                "z_min must be smaller than z_max."
            )

        return cls(
            kind="central_axis",
            z_min=z_min,
            z_max=z_max,
        )

    def extract(
        self,
        values,
        grid,
    ) -> np.ndarray:

        if self.kind != "central_axis":
            raise NotImplementedError(
                f"Unsupported region: {self.kind}"
            )

        values = _to_numpy(
            values
        )

        x = _to_numpy(
            grid.x
        )

        y = _to_numpy(
            grid.y
        )

        z = _to_numpy(
            grid.z
        )

        ix = int(
            np.argmin(
                np.abs(x)
            )
        )

        iy = int(
            np.argmin(
                np.abs(y)
            )
        )

        mask = np.ones(
            len(z),
            dtype=bool,
        )

        if self.z_min is not None:
            mask &= (
                z >= self.z_min
            )

        if self.z_max is not None:
            mask &= (
                z <= self.z_max
            )

        if not np.any(mask):
            raise ValueError(
                "The selected region contains no grid cells."
            )

        return values[
            ix,
            iy,
            mask,
        ]


# ============================================================
# Error norms
# ============================================================


@dataclass(frozen=True)
class ErrorNorms:
    l1: float
    l2: float
    linf: float


def _relative_errors(
    coarse,
    fine_restricted,
) -> ErrorNorms:

    coarse = np.asarray(
        coarse,
        dtype=float,
    )

    fine_restricted = np.asarray(
        fine_restricted,
        dtype=float,
    )

    if (
        coarse.shape
        != fine_restricted.shape
    ):
        raise ValueError(
            "Coarse and restricted fine arrays "
            "must have identical shapes."
        )

    difference = (
        coarse
        - fine_restricted
    )

    reference_l1 = np.mean(
        np.abs(
            fine_restricted
        )
    )

    reference_l2 = np.sqrt(
        np.mean(
            fine_restricted**2
        )
    )

    reference_linf = np.max(
        np.abs(
            fine_restricted
        )
    )

    if (
        reference_l1 == 0.0
        or reference_l2 == 0.0
        or reference_linf == 0.0
    ):
        raise ValueError(
            "Relative self-convergence error is undefined "
            "because the fine-grid reference norm is zero."
        )

    return ErrorNorms(
        l1=float(
            np.mean(
                np.abs(
                    difference
                )
            )
            / reference_l1
        ),

        l2=float(
            np.sqrt(
                np.mean(
                    difference**2
                )
            )
            / reference_l2
        ),

        linf=float(
            np.max(
                np.abs(
                    difference
                )
            )
            / reference_linf
        ),
    )


def _successive_orders(
    errors,
    refinement_ratio: int,
) -> np.ndarray:

    errors = np.asarray(
        errors,
        dtype=float,
    )

    orders = np.full(
        len(errors),
        np.nan,
        dtype=float,
    )

    for i in range(
        1,
        len(errors),
    ):
        if (
            errors[i - 1] <= 0.0
            or errors[i] <= 0.0
        ):
            continue

        orders[i] = (
            np.log(
                errors[i - 1]
                / errors[i]
            )
            / np.log(
                refinement_ratio
            )
        )

    return orders


def _fitted_order(
    h,
    errors,
) -> float:

    h = np.asarray(
        h,
        dtype=float,
    )

    errors = np.asarray(
        errors,
        dtype=float,
    )

    valid = (
        np.isfinite(h)
        & np.isfinite(errors)
        & (h > 0.0)
        & (errors > 0.0)
    )

    if (
        np.count_nonzero(valid)
        < 2
    ):
        return float("nan")

    slope, _ = np.polyfit(
        np.log(
            h[valid]
        ),
        np.log(
            errors[valid]
        ),
        1,
    )

    return float(
        slope
    )


# ============================================================
# Finite-volume restriction
# ============================================================


def _restrict_z(
    fine,
    coarse_nz: int,
) -> np.ndarray:
    """
    Restrict fine-grid cell averages to the coarse grid.

    For refinement ratio r,

        U_coarse[i]
        =
        (1/r) sum_j U_fine[r*i + j].

    This is the exact finite-volume restriction in z.
    """

    fine = _to_numpy(
        fine
    )

    fine_nz = (
        fine.shape[-1]
    )

    if (
        fine_nz
        % coarse_nz
        != 0
    ):
        raise ValueError(
            f"Fine Nz={fine_nz} is not an integer "
            f"multiple of coarse Nz={coarse_nz}."
        )

    ratio = (
        fine_nz
        // coarse_nz
    )

    if ratio <= 1:
        raise ValueError(
            "Fine grid must be finer than coarse grid."
        )

    shape = (
        *fine.shape[:-1],
        coarse_nz,
        ratio,
    )

    return (
        fine
        .reshape(shape)
        .mean(axis=-1)
    )


# ============================================================
# Internal numerical level
# ============================================================


@dataclass
class _NumericalLevel:
    resolution: tuple[int, int, int]

    h: float
    max_energy_step: float
    runtime: float

    grid: Any

    observables: dict[
        str,
        np.ndarray,
    ]

    # Energy steps shorter than max_energy_step, other than the steps that
    # land on a snapshot energy or end at energy_min: CFL-limited steps.
    cfl_limited_steps: int = 0


# ============================================================
# Pairwise convergence result
# ============================================================


@dataclass
class ConvergenceLevel:
    resolution: tuple[int, int, int]
    reference_resolution: tuple[int, int, int]

    h: float
    max_energy_step: float
    runtime: float

    errors: dict[
        str,
        ErrorNorms,
    ]


# ============================================================
# Study result
# ============================================================


@dataclass
class ConvergenceResult:
    levels: list[ConvergenceLevel]
    observables: tuple[ObservableSpec, ...]
    expected_order: float | None
    refinement: Refinement
    region: Region

    @property
    def refinement_ratio(
        self,
    ) -> int:
        return (
            self.refinement
            .refinement_ratio
        )

    @property
    def h(
        self,
    ) -> np.ndarray:
        return np.asarray([
            level.h
            for level
            in self.levels
        ])

    def _errors(
        self,
        observable: ObservableSpec,
        norm: str,
    ) -> np.ndarray:

        return np.asarray([
            getattr(
                level.errors[
                    observable.label
                ],
                norm,
            )
            for level
            in self.levels
        ])

    def summary(
        self,
    ):
        print()
        print(
            "Self-convergence study"
        )
        print(
            "======================"
        )

        print(
            f"Refinement ratio: "
            f"{self.refinement_ratio}"
        )

        for observable in self.observables:
            print()
            print(
                observable.label
            )

            print(
                "-"
                * len(
                    observable.label
                )
            )

            l1 = self._errors(
                observable,
                "l1",
            )

            l2 = self._errors(
                observable,
                "l2",
            )

            linf = self._errors(
                observable,
                "linf",
            )

            orders = (
                _successive_orders(
                    l2,
                    self.refinement_ratio,
                )
            )

            print(
                f"{'coarse':>16}"
                f"{'fine':>16}"
                f"{'h':>14}"
                f"{'L1':>14}"
                f"{'L2':>14}"
                f"{'Linf':>14}"
                f"{'p(L2)':>10}"
            )

            for (
                level,
                error_l1,
                error_l2,
                error_linf,
                order,
            ) in zip(
                self.levels,
                l1,
                l2,
                linf,
                orders,
            ):
                coarse = (
                    "x".join(
                        str(value)
                        for value
                        in level.resolution
                    )
                )

                fine = (
                    "x".join(
                        str(value)
                        for value
                        in level.reference_resolution
                    )
                )

                order_text = (
                    "-"
                    if not np.isfinite(order)
                    else f"{order:.3f}"
                )

                print(
                    f"{coarse:>16}"
                    f"{fine:>16}"
                    f"{level.h:14.6e}"
                    f"{error_l1:14.6e}"
                    f"{error_l2:14.6e}"
                    f"{error_linf:14.6e}"
                    f"{order_text:>10}"
                )

            fitted = (
                _fitted_order(
                    self.h,
                    l2,
                )
            )

            print()
            print(
                f"Fitted L2 order: "
                f"{fitted:.4f}"
            )

            if (
                self.expected_order
                is not None
            ):
                print(
                    f"Expected order:  "
                    f"{self.expected_order:g}"
                )

    def plot(
        self,
        show: bool = True,
        directory: Path | None = None,
    ):
        if directory is not None:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        for observable in self.observables:

            l1 = self._errors(
                observable,
                "l1",
            )

            l2 = self._errors(
                observable,
                "l2",
            )

            linf = self._errors(
                observable,
                "linf",
            )

            plt.figure(
                figsize=(7, 5),
            )

            plt.loglog(
                self.h,
                l1,
                "o-",
                label="L1",
            )

            plt.loglog(
                self.h,
                l2,
                "o-",
                label="L2",
            )

            plt.loglog(
                self.h,
                linf,
                "o-",
                label="Linf",
            )

            if (
                self.expected_order
                is not None
                and len(self.h) > 0
                and l2[0] > 0.0
            ):
                expected = (
                    l2[0]
                    * (
                        self.h
                        / self.h[0]
                    )
                    ** self.expected_order
                )

                plt.loglog(
                    self.h,
                    expected,
                    "--",
                    label=(
                        f"order "
                        f"{self.expected_order:g}"
                    ),
                )

            plt.xlabel(
                "h [m]"
            )

            plt.ylabel(
                "Relative self-error"
            )

            plt.title(
                f"Self-convergence: "
                f"{observable.label}"
            )

            plt.grid(
                True,
                which="both",
            )

            plt.legend()

            plt.tight_layout()

            if directory is not None:

                filename = (
                    observable.label
                    .lower()
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("/", "_")
                )

                plt.savefig(
                    directory
                    / f"{filename}.png",
                    dpi=200,
                )

            if not show:
                plt.close()

        if show:
            plt.show()

    def save(
        self,
        directory: str | Path | None = None,
    ) -> Path:

        if directory is None:

            timestamp = (
                datetime.now()
                .strftime(
                    "%Y%m%d_%H%M%S"
                )
            )

            directory = (
                Path("results")
                / (
                    f"{timestamp}"
                    f"__self_convergence"
                )
            )

        directory = Path(
            directory
        )

        directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        metadata = {
            "reference": "self",
            "refinement_axis": (
                self.refinement.axis
            ),
            "refinement_values": list(
                self.refinement.values
            ),
            "refinement_ratio": (
                self.refinement_ratio
            ),
            "couple_energy_step": (
                self.refinement
                .couple_energy_step
            ),
            "expected_order": (
                self.expected_order
            ),
            "region": {
                "kind": (
                    self.region.kind
                ),
                "z_min": (
                    self.region.z_min
                ),
                "z_max": (
                    self.region.z_max
                ),
            },
            "observables": [
                {
                    "name": (
                        observable.name
                    ),
                    "energy": (
                        observable.energy
                    ),
                }
                for observable
                in self.observables
            ],
        }

        (
            directory
            / "study.json"
        ).write_text(
            json.dumps(
                metadata,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        with (
            directory
            / "errors.csv"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                "coarse_nx,"
                "coarse_ny,"
                "coarse_nz,"
                "fine_nx,"
                "fine_ny,"
                "fine_nz,"
                "h,"
                "max_energy_step,"
                "runtime,"
                "observable,"
                "l1,"
                "l2,"
                "linf,"
                "order_l2\n"
            )

            for observable in self.observables:

                l2 = self._errors(
                    observable,
                    "l2",
                )

                orders = (
                    _successive_orders(
                        l2,
                        self.refinement_ratio,
                    )
                )

                for (
                    level,
                    order,
                ) in zip(
                    self.levels,
                    orders,
                ):
                    (
                        nx,
                        ny,
                        nz,
                    ) = (
                        level.resolution
                    )

                    (
                        fine_nx,
                        fine_ny,
                        fine_nz,
                    ) = (
                        level
                        .reference_resolution
                    )

                    errors = (
                        level.errors[
                            observable.label
                        ]
                    )

                    file.write(
                        f"{nx},"
                        f"{ny},"
                        f"{nz},"
                        f"{fine_nx},"
                        f"{fine_ny},"
                        f"{fine_nz},"
                        f"{level.h:.16e},"
                        f"{level.max_energy_step:.16e},"
                        f"{level.runtime:.16e},"
                        f"{observable.label},"
                        f"{errors.l1:.16e},"
                        f"{errors.l2:.16e},"
                        f"{errors.linf:.16e},"
                        f"{order:.16e}\n"
                    )

        self.plot(
            show=False,
            directory=directory,
        )

        print()
        print(
            f"Saved convergence study to "
            f"{directory}"
        )

        return directory


# ============================================================
# Convergence study
# ============================================================


@dataclass
class ConvergenceStudy:
    config: SolverConfig
    refinement: Refinement

    observables: list[
        Observable | ObservableSpec
    ]

    reference: Reference
    region: Region

    expected_order: float | None = None

    def __post_init__(
        self,
    ):
        self.observables = [
            _observable_spec(
                observable
            )
            for observable
            in self.observables
        ]

        if not self.observables:
            raise ValueError(
                "At least one observable is required."
            )

        self._validate()

    def _validate(
        self,
    ):
        if (
            self.reference
            is not Reference.SELF
        ):
            raise NotImplementedError(
                "ConvergenceStudy currently uses "
                "Reference.SELF. "
                "Use reference=Reference.SELF."
            )

        # Trigger nested-grid validation.
        _ = (
            self.refinement
            .refinement_ratio
        )

        energy_start = (
            self.config.beam.energy
            + self.config.numerics.energy_window_sigma
            * self.config.beam.energy_spread
        )

        energy_min = (
            self.config
            .numerics
            .energy_min
        )

        for observable in self.observables:

            if (
                observable.energy
                is None
            ):
                continue

            if not (
                energy_min
                <= observable.energy
                <= energy_start
            ):
                raise ValueError(
                    f"Observable energy "
                    f"{observable.energy} MeV lies "
                    f"outside the energy march "
                    f"[{energy_min}, "
                    f"{energy_start}] MeV."
                )

        if (
            self.expected_order
            is not None
            and self.expected_order <= 0.0
        ):
            raise ValueError(
                "expected_order must be positive."
            )

    def _snapshot_energies(
        self,
    ) -> tuple[float, ...]:

        energies = {
            observable.energy
            for observable
            in self.observables
            if observable.energy
            is not None
        }

        return tuple(
            sorted(
                energies,
                reverse=True,
            )
        )

    def _numerical_observable(
        self,
        result,
        observable: ObservableSpec,
    ):

        if (
            observable.name
            == "dose"
        ):
            return _to_numpy(
                result.dose
            )

        energy = (
            observable.energy
        )

        if (
            energy
            not in result.snapshots
        ):
            raise RuntimeError(
                f"Missing numerical snapshot "
                f"at {energy} MeV."
            )

        U = _to_numpy(
            result.snapshots[
                energy
            ]
        )

        if (
            observable.name
            == "phi"
        ):
            return U[0]

        if (
            observable.name
            == "jx"
        ):
            return U[1]

        if (
            observable.name
            == "jy"
        ):
            return U[2]

        if (
            observable.name
            == "jz"
        ):
            return U[3]

        raise ValueError(
            f"Unknown observable: "
            f"{observable.name}"
        )

    def _solve_levels(
        self,
    ) -> list[_NumericalLevel]:

        levels = []

        snapshot_energies = (
            self._snapshot_energies()
        )

        for value in (
            self.refinement.values
        ):

            run_config = (
                self.refinement
                .config_for(
                    self.config,
                    value,
                )
            )

            # Self-convergence needs no external reference.
            run_config = replace(
                run_config,
                references=replace(
                    run_config.references,
                    reference=Reference.NONE,
                ),
                output=replace(
                    run_config.output,
                    save_results=False,
                ),
            )

            print()
            print(
                "================================"
            )

            print(
                f"Convergence level: "
                f"{run_config.geometry.resolution}"
            )

            print(
                f"max dE: "
                f"{run_config.numerics.max_energy_step:g} MeV"
            )

            print(
                "================================"
            )

            result = Solver(
                run_config
            ).solve(
                snapshot_energies=(
                    snapshot_energies
                ),
            )

            cfl_limited_steps = _count_cfl_limited_steps(
                result.energies,
                run_config.numerics.max_energy_step,
                stops=(
                    *snapshot_energies,
                    run_config.numerics.energy_min,
                ),
            )

            print(
                f"Energy steps: {len(result.energies) - 1}, "
                f"CFL-limited: {cfl_limited_steps}"
            )

            if (
                cfl_limited_steps
                and not self.refinement.couple_energy_step
            ):
                # With a fixed or energy-only refined step, every step
                # must equal max_energy_step; otherwise the study does
                # not refine what it claims to refine.
                print(
                    f"WARNING: {cfl_limited_steps} energy steps were "
                    "limited by the CFL condition. Use a smaller "
                    "max_energy_step or a coarser grid."
                )

            values = {}

            for observable in (
                self.observables
            ):
                values[
                    observable.label
                ] = (
                    self._numerical_observable(
                        result,
                        observable,
                    )
                )

            levels.append(
                _NumericalLevel(
                    resolution=(
                        run_config
                        .geometry
                        .resolution
                    ),

                    h=(
                        self.refinement.h(
                            self.config,
                            value,
                        )
                    ),

                    max_energy_step=(
                        run_config
                        .numerics
                        .max_energy_step
                    ),

                    runtime=float(
                        result.runtime
                    ),

                    grid=result.grid,

                    observables=values,

                    cfl_limited_steps=cfl_limited_steps,
                )
            )

        return levels

    def run(
        self,
    ) -> ConvergenceResult:

        numerical_levels = (
            self._solve_levels()
        )

        levels = []

        for (
            coarse_level,
            fine_level,
        ) in zip(
            numerical_levels[:-1],
            numerical_levels[1:],
        ):

            coarse_nx, coarse_ny, coarse_nz = (
                coarse_level.resolution
            )

            fine_nx, fine_ny, fine_nz = (
                fine_level.resolution
            )

            energy_only = (
                self.refinement.axis
                == "energy"
            )

            if (
                energy_only
                and coarse_level.resolution
                != fine_level.resolution
            ):
                raise ValueError(
                    "Refinement.energy requires the same "
                    "grid on every level."
                )

            if (
                coarse_nx != fine_nx
                or coarse_ny != fine_ny
            ):
                raise ValueError(
                    "Refinement.nz requires nx and ny "
                    "to remain unchanged."
                )

            if (
                fine_nz
                % coarse_nz
                != 0
            ):
                raise ValueError(
                    "Fine Nz must be an integer multiple "
                    "of coarse Nz."
                )

            errors = {}

            for observable in (
                self.observables
            ):

                coarse = (
                    coarse_level
                    .observables[
                        observable.label
                    ]
                )

                fine = (
                    fine_level
                    .observables[
                        observable.label
                    ]
                )

                # Energy-only refinement: same grid, no restriction.
                fine_restricted = (
                    np.asarray(fine)
                    if energy_only
                    else _restrict_z(
                        fine,
                        coarse_nz,
                    )
                )

                # Important:
                #
                # Restrict the complete fine solution first,
                # then apply the physical region on the coarse
                # grid. This keeps z_max consistent when the
                # number of selected fine cells is not itself
                # divisible by the refinement ratio.

                coarse_region = (
                    self.region.extract(
                        coarse,
                        coarse_level.grid,
                    )
                )

                fine_region = (
                    self.region.extract(
                        fine_restricted,
                        coarse_level.grid,
                    )
                )

                errors[
                    observable.label
                ] = (
                    _relative_errors(
                        coarse_region,
                        fine_region,
                    )
                )

            levels.append(
                ConvergenceLevel(
                    resolution=(
                        coarse_level.resolution
                    ),

                    reference_resolution=(
                        fine_level.resolution
                    ),

                    h=(
                        coarse_level.h
                    ),

                    max_energy_step=(
                        coarse_level
                        .max_energy_step
                    ),

                    runtime=(
                        coarse_level.runtime
                    ),

                    errors=errors,
                )
            )

        return ConvergenceResult(
            levels=levels,

            observables=tuple(
                self.observables
            ),

            expected_order=(
                self.expected_order
            ),

            refinement=(
                self.refinement
            ),

            region=(
                self.region
            ),
        )


# ============================================================
# Helpers
# ============================================================


def _count_cfl_limited_steps(
    energies,
    max_energy_step: float,
    stops=(),
) -> int:
    """
    Number of energy steps shorter than max_energy_step.

    Steps that end on a requested stop energy (snapshot or energy_min),
    and round-off steps next to them, are shortened on purpose and are
    not counted.
    """

    energies = np.asarray(
        energies,
        dtype=float,
    )

    steps = -np.diff(energies)

    if len(steps) == 0:
        return 0

    planned = np.zeros(
        len(steps),
        dtype=bool,
    )

    for stop in stops:
        planned |= np.isclose(
            energies[1:],
            stop,
            rtol=0.0,
            atol=1e-9,
        )

    planned[-1] = True

    tiny = steps < 1e-6 * max_energy_step

    short = steps < max_energy_step * (1.0 - 1e-6)

    return int(
        np.count_nonzero(
            short & ~planned & ~tiny
        )
    )


def _to_numpy(
    value: Any,
) -> np.ndarray:

    if hasattr(
        value,
        "get",
    ):
        return value.get()

    return np.asarray(
        value
    )