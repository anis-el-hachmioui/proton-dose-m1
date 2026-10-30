"""Shared runner for the robustness studies of the thesis benchmarks.

Each study takes the benchmark configurations of the main experiments
(build_config in thesis_experiments/<model>/<case>.py), changes one aspect,
runs the selected spatial schemes, and stores every run like the main
benchmarks (config.json, metadata.json, results.npz, metrics.json) under
thesis_results/robustness/<study>/<model>/<case>/<scheme>/.

Runs whose metrics.json already exists are skipped, so an interrupted study
can be restarted. A summary.json collects runtime and metrics of all runs.
"""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path

from proton_solver import Solver

from thesis_experiments.common import RECONSTRUCTIONS, save_run_artifacts

CASES = {
    "reduced_physics/homogeneous_water": (
        "thesis_experiments.reduced_physics.homogeneous_water"
    ),
    "reduced_physics/longitudinal_slabs": (
        "thesis_experiments.reduced_physics.longitudinal_slabs"
    ),
    "reduced_physics/transversal_slabs": (
        "thesis_experiments.reduced_physics.transversal_slabs"
    ),
    "scattering_nuclear/homogeneous_water": (
        "thesis_experiments.scattering_nuclear.homogeneous_water"
    ),
    "scattering_nuclear/longitudinal_slabs": (
        "thesis_experiments.scattering_nuclear.longitudinal_slabs"
    ),
    "scattering_nuclear/transversal_slabs": (
        "thesis_experiments.scattering_nuclear.transversal_slabs"
    ),
}

ROOT = Path("thesis_results/robustness")


def run_study(study, modify_config, schemes=("weno_z",), cases=None):
    """Run one robustness study.

    modify_config(config) -> config applies the change of this study.
    """

    root = ROOT / study
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )

    for case_key in cases or CASES:
        module = importlib.import_module(CASES[case_key])

        for scheme_key in schemes:
            label, scheme = RECONSTRUCTIONS[scheme_key]
            destination = root / case_key / scheme_key
            run_key = f"{case_key}/{scheme_key}"

            if (destination / "metrics.json").exists():
                print(f"Skipping {study}: {run_key} (done)")
                continue

            print()
            print("=" * 72)
            print(f"{study}: {run_key}")
            print("=" * 72)

            config = modify_config(module.build_config(scheme))
            result = Solver(config).solve()
            result.metrics.summary()

            if destination.exists():
                shutil.rmtree(destination)
            metrics = save_run_artifacts(result, destination)

            summary[run_key] = {
                "resolution": list(config.geometry.resolution),
                "bfp_mc_histories": int(
                    config.references.bfp_mc_histories
                ),
                "moving_active_domain": bool(
                    config.numerics.moving_active_domain
                ),
                "deterministic_runtime_seconds": float(result.runtime),
                "bfp_mc_runtime_seconds": (
                    None
                    if result.bfp_mc_runtime is None
                    else float(result.bfp_mc_runtime)
                ),
                "metrics": metrics,
            }
            summary_path.write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )

    print(f"\n{study} complete: {root.resolve()}")
