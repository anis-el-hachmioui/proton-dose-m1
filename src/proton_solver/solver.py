import time
from pathlib import Path
from tempfile import TemporaryDirectory

from .analytical import (
    analytical_available,
    analytical_csda_dose,
)
from .backend import resolve_backend
from .bfp_mc import run_bfp_mc_reference
from .config import SolverConfig
from .enums import Reference
from .geometry_viewer import view_geometry
from .persistence import save_result
from .result import Result

from .numerics.boundary import BeamInflow
from .numerics.energy import energy_march
from .numerics.grid import Grid

from .physics.materials import MaterialMap

from .topas import (
    parse_topas_dose,
    run_topas,
    write_topas_case,
)

class Solver:
    def __init__(self, config: SolverConfig):
        self.config = config
        self.backend = resolve_backend(config.backend)
 
    def solve(
        self,
        snapshot_energies: tuple[float, ...] = (),
    ) -> Result:
        """Solve the configured proton transport problem."""

        if self.config.geometry.ct_scan is not None:
            requested = (
                self.config.references.reference,
                *self.config.references.additional_references,
            )
            if Reference.TOPAS in requested:
                raise NotImplementedError(
                    "The TOPAS writer supports Box geometry only and cannot "
                    "export a CT material map. Use Reference.BFP_MC instead."
                )

        if self.config.geometry.view_geometry:
            view_geometry(
                self.config.geometry,
                self.config.beam,
            )

        xp = self.backend.xp

        # Construct computational grid
        grid = Grid.from_geometry(
            self.config.geometry,
            xp,
        )

        # Voxelize material geometry
        material_map = MaterialMap.from_geometry(
            self.config.geometry,
            grid,
            xp,
        )

        # Construct beam inflow
        inflow = BeamInflow.from_beam(
            self.config.beam,
            grid,
            xp,
        )

        # ---------------------------------------------------------
        # Optional deterministic solve
        # ---------------------------------------------------------

        phi = None
        J = None
        dose = None
        energies = None
        snapshots = {}
        runtime = None

        if self.config.solve_deterministic:
            # The moving active-domain solver deliberately avoids allocating
            # a full 4-component state.  The compact state is initialized by
            # energy_march() around the beam support.
            if self.config.numerics.moving_active_domain:
                U = None
            else:
                U = xp.zeros(
                    (4, *grid.shape),
                    dtype=float,
                )

            start = time.perf_counter()

            U, dose, energies, snapshots = energy_march(
                U,
                material_map,
                grid,
                inflow,
                self.config,
                xp,
                snapshot_energies=snapshot_energies,
            )

            runtime = time.perf_counter() - start

            if U is not None:
                phi = U[0]
                J = U[1:4]

        # ---------------------------------------------------------
        # Optional reference solutions
        # ---------------------------------------------------------

        analytical_dose = None
        topas_dose = None
        bfp_mc_dose = None

        topas_runtime = None
        bfp_mc_runtime = None

        references = (
            self.config.references.reference,
            *self.config.references.additional_references,
        )

        for reference in references:

            if reference is Reference.NONE:
                continue

            if reference is Reference.ANALYTICAL:
                if not analytical_available(self.config):
                    raise ValueError(
                        "Analytical reference is not available "
                        "for this configuration."
                    )

                analytical_dose = analytical_csda_dose(
                    self.config,
                    material_map,
                    grid,
                    inflow,
                )

            elif reference is Reference.TOPAS:
                with TemporaryDirectory(
                    prefix="proton_solver_topas_"
                ) as directory:

                    case = write_topas_case(
                        self.config,
                        Path(directory),
                    )

                    dose_file, topas_runtime = run_topas(
                        case,
                        force_rerun=(
                            self.config
                            .references
                            .topas_force_rerun
                        ),
                    )

                    topas_dose = parse_topas_dose(
                        dose_file,
                        self.config,
                    )
            elif reference is Reference.BFP_MC:
                bfp_mc_dose, bfp_mc_runtime = run_bfp_mc_reference(
                    self.config,
                    material_map,
                    grid,
                    inflow,
                )

            else:
                raise ValueError(
                    f"Unsupported solver reference: {reference}"
                )

        # ---------------------------------------------------------
        # Construct result
        # ---------------------------------------------------------

        result = Result(
            phi=phi,
            J=J,
            dose=dose,
            grid=grid,
            energies=energies,
            snapshots=snapshots,
            analytical_dose=analytical_dose,
            topas_dose=topas_dose,
            bfp_mc_dose=bfp_mc_dose,
            runtime=runtime,
            topas_runtime=topas_runtime,
            bfp_mc_runtime=bfp_mc_runtime,
            backend=self.backend.backend,
        )

        # Optional persistence
        if self.config.output.save_results:
            result.run_directory = save_result(
                result,
                self.config,
            )

        return result