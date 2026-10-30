"""Extended-model self-convergence with the outflow boundary moved downstream.

Diagnostic for the non-monotone dose errors of the extended-model WENO-Z
convergence study. The study of
thesis_experiments/numerical_verification/scattering_nuclear_convergence.py
is repeated with the domain depth increased from 30 mm to 36 mm. The cell
widths are kept: Nz = 48, 96, ..., 3072 on 36 mm gives the same dz as
Nz = 40, 80, ..., 2560 on 30 mm, and the energy step is coupled in the same
way. The evaluation is unchanged (relative L2 error of the dose and of the
fluence at 60 MeV on the central axis for z <= 27 mm, after finite-volume
restriction of the finer solution). The only change is that the zero-gradient
outflow boundary lies 9 mm instead of 3 mm behind the evaluated region.

In addition to z <= 27 mm, the dose errors are reported in depth bands, to
show where any remaining irregularity is located. The central-axis profiles
of every level are saved, so the analysis can be repeated without rerunning.

Output: thesis_results/numerical_verification/
        scattering_nuclear_convergence_long_domain/
            profiles_<Nz>.npz, convergence_long_domain.json

Usage: uv run python -m thesis_experiments.robustness.convergence_long_domain
"""

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from proton_solver import Reference, Solver

from thesis_experiments.numerical_verification import (
    scattering_nuclear_convergence as extended,
)

DOMAIN_DEPTH_M = 0.036
BASE_DEPTH_M = 0.030
BASE_RESOLUTIONS = (40, 80, 160, 320, 640, 1280, 2560)
RESOLUTIONS = tuple(
    int(round(n * DOMAIN_DEPTH_M / BASE_DEPTH_M)) for n in BASE_RESOLUTIONS
)
Z_MAX_M = 0.027
BANDS_MM = ((0.0, 10.0), (10.0, 15.0), (15.0, 20.0), (20.0, 27.0))
SNAPSHOT_ENERGY = 60.0

BASE_DIR = Path(
    "thesis_results/numerical_verification/scattering_nuclear_convergence"
)
RESULT_DIR = Path(
    "thesis_results/numerical_verification/"
    "scattering_nuclear_convergence_long_domain"
)


def level_config(nz: int):
    """Configuration of one level, with the energy step coupled to dz."""

    base = extended.build_config()
    size = base.geometry.size
    nx, ny, _ = base.geometry.resolution
    return dataclasses.replace(
        base,
        geometry=dataclasses.replace(
            base.geometry,
            size=(size[0], size[1], DOMAIN_DEPTH_M),
            resolution=(nx, ny, nz),
            view_geometry=False,
        ),
        numerics=dataclasses.replace(
            base.numerics,
            max_energy_step=(
                base.numerics.max_energy_step * RESOLUTIONS[0] / nz
            ),
        ),
        references=dataclasses.replace(
            base.references,
            reference=Reference.NONE,
        ),
        output=dataclasses.replace(
            base.output,
            save_results=False,
        ),
    )


def central_index(centres) -> int:
    """Same central-axis cell as Region.central_axis (argmin |x|)."""

    return int(np.argmin(np.abs(np.asarray(centres))))


def solve_level(nz: int) -> Path:
    path = RESULT_DIR / f"profiles_{nz}.npz"
    if path.is_file():
        print(f"Level Nz={nz}: found {path}, skipping")
        return path

    config = level_config(nz)
    print("=" * 60)
    print(
        f"Level Nz={nz}, dz={1e3 * DOMAIN_DEPTH_M / nz:.6g} mm, "
        f"max dE={config.numerics.max_energy_step:g} MeV",
        flush=True,
    )
    start = time.time()
    result = Solver(config).solve(snapshot_energies=(SNAPSHOT_ENERGY,))
    runtime = time.time() - start

    x = np.asarray(result.grid.x)
    y = np.asarray(result.grid.y)
    ix, iy = central_index(x), central_index(y)
    np.savez(
        path,
        z=np.asarray(result.grid.z),
        dose=np.asarray(result.dose)[ix, iy, :],
        phi60=np.asarray(result.snapshots[SNAPSHOT_ENERGY])[0][ix, iy, :],
        runtime=runtime,
    )
    print(f"Level Nz={nz} done in {runtime:.0f} s", flush=True)
    return path


def relative_l2(coarse_path, fine_path, key, z_lo_mm, z_hi_mm):
    """Relative L2 error, fine solution restricted to the coarse grid."""

    with np.load(coarse_path) as c, np.load(fine_path) as f:
        z = c["z"]
        coarse = c[key]
        fine = f[key].reshape(len(coarse), -1).mean(axis=1)
    mask = (1e3 * z >= z_lo_mm) & (z <= z_hi_mm * 1e-3)
    return float(
        np.sqrt(np.mean((coarse[mask] - fine[mask]) ** 2))
        / np.sqrt(np.mean(fine[mask] ** 2))
    )


def fitted_order(h, errors):
    slope, _ = np.polyfit(np.log(h), np.log(errors), 1)
    return float(slope)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [solve_level(nz) for nz in RESOLUTIONS]
    h = [DOMAIN_DEPTH_M / nz for nz in RESOLUTIONS[:-1]]

    record = {
        "domain_depth_m": DOMAIN_DEPTH_M,
        "resolutions": list(RESOLUTIONS),
        "z_max_m": Z_MAX_M,
        "errors": {},
    }

    series = [("dose", 0.0, 1e3 * Z_MAX_M), ("phi60", 0.0, 1e3 * Z_MAX_M)]
    series += [("dose", lo, hi) for lo, hi in BANDS_MM]

    for key, lo, hi in series:
        errors = [
            relative_l2(c, f, key, lo, hi)
            for c, f in zip(paths[:-1], paths[1:])
        ]
        label = f"{key} z {lo:g}-{hi:g} mm"
        record["errors"][label] = {
            "errors": errors,
            "ratios": [a / b for a, b in zip(errors[:-1], errors[1:])],
            "fitted_order": fitted_order(h, errors),
        }

    base = None
    base_path = BASE_DIR / "convergence_summary.json"
    if base_path.is_file():
        observables = json.loads(base_path.read_text(encoding="utf-8"))[
            "observables"
        ]
        base = {
            "dose": observables.get("dose", {}).get("l2", {}),
            "phi60": observables.get("phi(60 MeV)", {}).get("l2", {}),
        }
        record["base_30_mm"] = base

    (RESULT_DIR / "convergence_long_domain.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    print()
    print("Relative L2 errors per pair (ratio to the next pair)")
    for label, data in record["errors"].items():
        errs = "  ".join(f"{e:.3e}" for e in data["errors"])
        rats = "  ".join(f"{r:5.2f}" for r in data["ratios"])
        print(f"\n36 mm, {label}\n  errors: {errs}\n  ratios: {rats}"
              f"\n  fitted order: {data['fitted_order']:.3f}")
    if base:
        for key in ("dose", "phi60"):
            errs = base[key].get("errors")
            if errs:
                rats = "  ".join(
                    f"{a / b:5.2f}" for a, b in zip(errs[:-1], errs[1:])
                )
                print(f"\n30 mm (original), {key} z 0-27 mm\n  errors: "
                      + "  ".join(f"{e:.3e}" for e in errs)
                      + f"\n  ratios: {rats}"
                      + f"\n  fitted order: {base[key]['fitted_order']:.3f}")


if __name__ == "__main__":
    main()
