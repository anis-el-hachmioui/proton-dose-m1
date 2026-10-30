# proton-dose-m1

Deterministic proton dose calculation with the $M_1$ angular moment model,
developed for the MSc thesis *Moment Methods for Proton Therapy Dose
Calculations* (A. El Hachmioui, Delft University of Technology, 2026;
supervisors D. Lathouwers, M. B. van Gijzen, E. Benschop).

The solver transports the zeroth and first angular moments of the proton
fluence through a Boltzmann-Fokker-Planck (BFP) model with continuous energy
loss, Wentzel-VI angular scattering, and primary-proton nuclear removal. It
uses a finite-volume discretization with HLL fluxes and first-order, MUSCL,
or WENO-Z reconstruction, the integrating-factor scheme IF-SSPRK3 in the
energy-loss coordinate, and a realizability projection. A particle Monte
Carlo solver of the same BFP model and an analytical CSDA solution serve as
references.

## Installation

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

The solver runs on the CPU with Numba. An experimental GPU path uses CuPy
(`uv sync --extra gpu`); it was not used for the thesis results.

## Repository layout

```
src/proton_solver/        solver package
  numerics/               grid, reconstruction, HLL flux, energy march, realizability
  physics/                stopping power, scattering, nuclear removal, materials
  bfp_mc/                 BFP Monte Carlo reference solver
  analytical.py           analytical CSDA free-streaming solution
  convergence.py          self-convergence studies
  metrics.py              gamma analysis, maximum dose, R80, distal falloff width
  topas/                  TOPAS input writer (not used for the thesis results)
thesis_experiments/       one script per result of the thesis (see below)
tools/                    checks, figures, and the export of tables for the thesis
  geant4_material_data/   generation of the material tables in data/ from Geant4
data/                     material tables (stopping power, Wentzel-VI transport
                          coefficient, nuclear removal) for five Geant4 materials
```

## Reproducing the thesis results

All commands are run from the repository root. Results are written to
`thesis_results/` (not tracked by git). Monte Carlo references are cached in
`bfp_mc_cache/`, so repeated runs reuse them.

| Thesis | Command |
|---|---|
| §5.2 Reference verification | `uv run python -m thesis_experiments.reference_verification.bfp_mc_vs_analytical` |
| §5.2 Interpolation check | `uv run tools/check_analytical_interpolation.py` |
| §5.2.3 Reference uncertainty | `uv run tools/reference_uncertainty.py` |
| §5.2.3 Extended-model reference checks | `uv run tools/check_reference.py` |
| §5.3 Convergence, WENO-Z | `uv run python -m thesis_experiments.numerical_verification.reduced_physics_convergence` and `...scattering_nuclear_convergence` |
| §5.3 Convergence, first order and MUSCL | `uv run python -m thesis_experiments.robustness.convergence_schemes` |
| §5.3 Convergence, PCHIP coefficients | `uv run python -m thesis_experiments.robustness.convergence_pchip_coefficients` |
| §5.3 Convergence, 36 mm domain | `uv run python -m thesis_experiments.robustness.convergence_long_domain` |
| §5.3.4, App. D.1 Energy-only convergence | `uv run python -m thesis_experiments.numerical_verification.energy_convergence` |
| §5.3.4, App. D.1 Space-only convergence | `uv run python -m thesis_experiments.numerical_verification.spatial_convergence` |
| App. D.1 Space-only error along the axis | `uv run tools/spatial_convergence_profile.py` |
| §5.4 Reduced-model benchmarks | `uv run python -m thesis_experiments.reduced_physics.<case>` |
| §5.5 Extended-model benchmarks | `uv run python -m thesis_experiments.scattering_nuclear.<case>` |
| §5.5.1 Lateral width | `uv run tools/lateral_width.py` |
| §5.5.3 Lateral-interface profiles | `uv run tools/interface_profile.py` |
| §5.5.4 CT-derived geometry | `uv run python -m thesis_experiments.scattering_nuclear.ct` |
| App. D.2 Coarser grids | `uv run python -m thesis_experiments.robustness.coarse_grid` |
| App. D.3 Lateral refinement | `uv run python -m thesis_experiments.robustness.interface_lateral_refinement` |
| App. D.4 Moving active domain | `uv run python -m thesis_experiments.robustness.moving_active_domain` |
| Figure 5.2 Geometry previews | `uv run tools/generate_geometry_previews.py` |
| All tables, copy for the thesis | `uv run tools/export_thesis_results_latex.py` |

`<case>` is `homogeneous_water`, `longitudinal_slabs`, or `transversal_slabs`.
Each benchmark script runs first-order, MUSCL, and WENO-Z reconstruction and
the BFP Monte Carlo reference with $10^8$ histories; the reference takes the
largest part of the runtime.

`tools/export_thesis_results_latex.py` generates the LaTeX tables and copies
the figures used in the thesis from `thesis_results/` to
`thesis_results_latex/`, which is uploaded to the `figures/` folder of the
thesis.

## CT data

The CT-derived case uses the prostate CT of the CORT dataset (Craft et al.,
*GigaScience* 3:37, 2014, doi:10.1186/2047-217X-3-37), which is included
in this repository as `data/ct/prostate.npz` (CC0; see `data/ct/README.md`), a Hounsfield-unit array of shape
(183, 183, 90) with 3 mm voxels. After running
`thesis_experiments.scattering_nuclear.ct`, the dose can be inspected on the
CT with napari:

```
uv run --with "napari[all]" python -m thesis_experiments.scattering_nuclear.view_ct
```

## Material data

The tables in `data/` were generated with Geant4
(`G4EmStandardPhysics_option4`; for water this includes the ICRU90 stopping
power) by the programs in `tools/geant4_material_data/`. Both the
deterministic solver and the BFP Monte Carlo solver read the same tables.