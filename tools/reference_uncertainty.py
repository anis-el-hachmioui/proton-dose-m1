"""Batch estimate of the statistical uncertainty of the BFP Monte Carlo reference.

For each benchmark, the BFP Monte Carlo reference is run K times with
independent seeds and n_b histories per batch. The batches give a voxelwise
sample variance, from which the standard uncertainty of a reference with
N_ref histories (the number used in the thesis, read from the experiment
configuration) follows as

    u_abs = sqrt(s^2 * n_b / N_ref),       u_rel = u_abs / mean dose.

With K * n_b = N_ref (default: 10 x 1e6 = 1e7) this is the usual batch
estimate s / sqrt(K) of the mean of the K batches.

Reported per benchmark:
- relative uncertainty at the voxel of maximum dose and RMS over the peak
  region (mean dose >= 90% of the maximum),
- median and 95th percentile of the relative uncertainty in the voxels above
  50% and above 10% (gamma cutoff) of the maximum dose,
- the upward bias and the spread of the reference maximum dose. The maximum
  of a noisy dose field lies above the maximum of the exact field. With
  M_k the maximum of batch k and M the maximum of the mean of all batches,
  and scaling the bias as N^(-1/2),
      b(n_b)   = (mean_k M_k - M) / (1 - 1/sqrt(K)),
      b(N_ref) = b(n_b) * sqrt(n_b / N_ref).
  The bias grows somewhat faster than the noise, because more voxels near the
  peak compete for the maximum at higher noise. Extrapolating down from n_b
  therefore gives an upper estimate. On a synthetic field with known noise it
  overestimated the true bias by up to about 50 percent, and it is much more
  stable than half-sample variants. The run-to-run standard deviation of the
  maximum is reported as well,
- the number of histories for a target relative uncertainty in the peak
  region, N_target = N_ref * (u_peak / u_target)^2, and the Monte Carlo
  runtime this would take, from the measured runtime per history.

The batches bypass the reference cache, so the cached thesis references are
neither used nor overwritten. Seeds start at 1001 and are therefore
independent of the reference seed 137.

Usage (from the repository root):
    uv run tools/reference_uncertainty.py
    uv run tools/reference_uncertainty.py --cases extended/homogeneous_water
    uv run tools/reference_uncertainty.py --batches 10 --histories-per-batch 1000000

Output (default thesis_results/reference_uncertainty/):
    <model>__<benchmark>.json   one file per benchmark, written when it is done
    reference_uncertainty.json  all benchmarks together
    reference_uncertainty_table.tex
Benchmarks whose JSON already exists are skipped unless --force is given, so
an interrupted run can simply be restarted.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proton_solver import SpatialScheme  # noqa: E402
from proton_solver.bfp_mc.runner import (  # noqa: E402
    _run_heterogeneous,
    _run_homogeneous,
)
from proton_solver.numerics.boundary import BeamInflow  # noqa: E402
from proton_solver.numerics.grid import Grid  # noqa: E402
from proton_solver.physics.materials import MaterialMap  # noqa: E402


# Extended model first: these are the cases without an analytical check.
CASES = {
    "extended/homogeneous_water": (
        "thesis_experiments.scattering_nuclear.homogeneous_water",
        "Extended",
        "Homogeneous water",
    ),
    "extended/longitudinal_slabs": (
        "thesis_experiments.scattering_nuclear.longitudinal_slabs",
        "Extended",
        "Depth-wise slab",
    ),
    "extended/transversal_slabs": (
        "thesis_experiments.scattering_nuclear.transversal_slabs",
        "Extended",
        "Lateral interface",
    ),
    "reduced/homogeneous_water": (
        "thesis_experiments.reduced_physics.homogeneous_water",
        "Reduced",
        "Homogeneous water",
    ),
    "reduced/longitudinal_slabs": (
        "thesis_experiments.reduced_physics.longitudinal_slabs",
        "Reduced",
        "Depth-wise slab",
    ),
    "reduced/transversal_slabs": (
        "thesis_experiments.reduced_physics.transversal_slabs",
        "Reduced",
        "Lateral interface",
    ),
}

TARGET_UNCERTAINTIES = (0.01, 0.02)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def git_state():
    """Return (commit, dirty) of the repository, or (None, None)."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return commit, bool(status)
    except (OSError, subprocess.CalledProcessError):
        return None, None


def case_file(out_dir: Path, key: str) -> Path:
    return out_dir / (key.replace("/", "__") + ".json")


def percentile_or_none(values, q):
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


# ---------------------------------------------------------------------------
# One benchmark
# ---------------------------------------------------------------------------


def run_case(key, args):
    module_name, model, benchmark = CASES[key]
    module = importlib.import_module(module_name)

    # The spatial scheme does not enter the Monte Carlo reference.
    base = module.build_config(SpatialScheme.WENO_Z)

    grid = Grid.from_geometry(base.geometry, np)
    material_map = MaterialMap.from_geometry(base.geometry, grid, np)
    inflow = BeamInflow.from_beam(base.beam, grid, np)
    homogeneous = len(material_map.materials) == 1

    n_ref = int(base.references.bfp_mc_histories)
    n_b = int(args.histories_per_batch)
    k_batches = int(args.batches)

    batches = []
    batch_maxima = []
    runtimes = []
    seeds = []

    print(
        f"\n=== {key}: {k_batches} batches x {n_b:,} histories "
        f"(N_ref = {n_ref:,}, grid {tuple(base.geometry.resolution)}) ==="
    )

    for k in range(k_batches):
        seed = int(args.seed_start) + k
        references = dataclasses.replace(
            base.references,
            bfp_mc_histories=n_b,
            bfp_mc_seed=seed,
        )
        config = dataclasses.replace(base, references=references)

        if homogeneous:
            dose, runtime = _run_homogeneous(config, material_map, grid)
        else:
            dose, runtime = _run_heterogeneous(
                config, material_map, grid, inflow
            )

        dose = np.asarray(dose, dtype=np.float64)

        batches.append(dose)
        batch_maxima.append(float(dose.max()))
        runtimes.append(float(runtime))
        seeds.append(seed)

        print(
            f"  batch {k + 1:2d}/{k_batches}  seed {seed}  "
            f"{runtime:8.1f} s  max dose {dose.max():.6e} Gy"
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    stack = np.stack(batches)
    del batches
    dose_sum = stack.sum(axis=0)
    mean = dose_sum / k_batches
    variance = stack.var(axis=0, ddof=1)

    # Standard uncertainty of a reference with N_ref histories.
    u_abs = np.sqrt(variance * n_b / n_ref)

    d_max = float(mean.max())
    peak_index = np.unravel_index(int(np.argmax(mean)), mean.shape)

    positive = mean > 0.0
    u_rel = np.full_like(mean, np.nan)
    u_rel[positive] = u_abs[positive] / mean[positive]

    peak_region = mean >= 0.9 * d_max
    region_50 = mean >= 0.5 * d_max
    region_10 = mean >= 0.1 * d_max

    u_peak_voxel = float(u_rel[peak_index])
    u_peak_rms = float(np.sqrt(np.mean(u_rel[peak_region] ** 2)))

    del stack

    # Upper estimate of the bias of the maximum dose, scaled to N_ref.
    batch_maxima = np.asarray(batch_maxima)
    bias_batch = (
        (float(batch_maxima.mean()) - d_max)
        / (1.0 - 1.0 / math.sqrt(k_batches))
    )
    bias_ref = bias_batch * math.sqrt(n_b / n_ref)

    # Run-to-run spread of the maximum, from the single-batch maxima.
    spread_ref = (
        float(batch_maxima.std(ddof=1)) * math.sqrt(n_b / n_ref)
    )

    runtime_per_history = float(np.sum(runtimes)) / (k_batches * n_b)

    targets = {}
    for target in TARGET_UNCERTAINTIES:
        n_target = n_ref * (u_peak_rms / target) ** 2
        targets[f"{100 * target:g}_percent"] = {
            "histories": float(n_target),
            "runtime_seconds": float(runtime_per_history * n_target),
        }

    commit, dirty = git_state()

    record = {
        "case": key,
        "model": model,
        "benchmark": benchmark,
        "module": module_name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "grid_resolution": list(base.geometry.resolution),
        "reference_histories": n_ref,
        "batches": k_batches,
        "histories_per_batch": n_b,
        "seeds": seeds,
        "threads": int(base.references.bfp_mc_threads),
        "batch_runtimes_seconds": runtimes,
        "runtime_per_history_seconds": runtime_per_history,
        "estimated_runtime_at_reference_histories_seconds": (
            runtime_per_history * n_ref
        ),
        "maximum_dose_Gy": d_max,
        "peak_voxel_index": [int(i) for i in peak_index],
        "relative_uncertainty_percent": {
            "peak_voxel": 100.0 * u_peak_voxel,
            "peak_region_rms": 100.0 * u_peak_rms,
            "peak_region_voxels": int(peak_region.sum()),
            "above_50_percent_median": 100.0 * percentile_or_none(
                u_rel[region_50], 50
            ),
            "above_50_percent_p95": 100.0 * percentile_or_none(
                u_rel[region_50], 95
            ),
            "above_50_percent_voxels": int(region_50.sum()),
            "above_10_percent_median": 100.0 * percentile_or_none(
                u_rel[region_10], 50
            ),
            "above_10_percent_p95": 100.0 * percentile_or_none(
                u_rel[region_10], 95
            ),
            "above_10_percent_voxels": int(region_10.sum()),
        },
        "maximum_dose_statistic_percent": {
            "bias": 100.0 * bias_ref / d_max,
            "standard_deviation": 100.0 * spread_ref / d_max,
            "note": (
                "Upper estimate of the bias of max(D) for N_ref "
                "histories (N^(-1/2) extrapolation from single batches), "
                "and the run-to-run standard deviation of max(D). Positive "
                "bias means the reference maximum lies above the exact "
                "maximum."
            ),
        },
        "target_uncertainty_peak_region": targets,
    }

    return record


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def latex_power(value):
    """Format a positive number as a x 10^b with one decimal."""

    exponent = int(math.floor(math.log10(value)))
    mantissa = value / 10.0**exponent
    if round(mantissa, 1) >= 10.0:
        mantissa /= 10.0
        exponent += 1
    return f"${mantissa:.1f}\\times10^{{{exponent}}}$"


def write_latex_table(records, path: Path):
    lines = [
        "% Generated by tools/reference_uncertainty.py",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Model & Benchmark & "
        "\\shortstack{$u$ peak\\\\{}[\\%]} & "
        "\\shortstack{$u$ median\\\\$D\\geq50\\%$ [\\%]} & "
        "\\shortstack{Max. dose\\\\bias [\\%]} & "
        "\\shortstack{$N_{\\mathrm{hist}}$\\\\for $1\\%$} & "
        "\\shortstack{$N_{\\mathrm{hist}}$\\\\for $2\\%$} \\\\",
        "\\midrule",
    ]

    for record in records:
        u = record["relative_uncertainty_percent"]
        m = record["maximum_dose_statistic_percent"]
        t = record["target_uncertainty_peak_region"]
        lines.append(
            f"{record['model']} & {record['benchmark']} & "
            f"${u['peak_region_rms']:.2f}$ & "
            f"${u['above_50_percent_median']:.2f}$ & "
            f"${m['bias']:.2f}$ & "
            f"{latex_power(t['1_percent']['histories'])} & "
            f"{latex_power(t['2_percent']['histories'])} \\\\"
        )

    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="ascii")


def print_summary(records):
    print("\nReference uncertainty at N_ref histories")
    header = (
        f"{'case':30s} {'u_peak%':>8s} {'u50med%':>8s} {'u10med%':>8s} "
        f"{'bias%':>7s} {'sd_max%':>8s} {'N(1%)':>10s} {'N(2%)':>10s} "
        f"{'t(1%) s':>9s} {'t(2%) s':>9s}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        u = r["relative_uncertainty_percent"]
        m = r["maximum_dose_statistic_percent"]
        t = r["target_uncertainty_peak_region"]
        print(
            f"{r['case']:30s} {u['peak_region_rms']:8.3f} "
            f"{u['above_50_percent_median']:8.3f} "
            f"{u['above_10_percent_median']:8.3f} "
            f"{m['bias']:7.3f} {m['standard_deviation']:8.3f} "
            f"{t['1_percent']['histories']:10.2e} "
            f"{t['2_percent']['histories']:10.2e} "
            f"{t['1_percent']['runtime_seconds']:9.1f} "
            f"{t['2_percent']['runtime_seconds']:9.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=list(CASES),
        default=list(CASES),
        help="Benchmarks to evaluate (default: all six, extended first).",
    )
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument(
        "--histories-per-batch", type=int, default=1_000_000
    )
    parser.add_argument("--seed-start", type=int, default=1001)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "thesis_results" / "reference_uncertainty",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute benchmarks whose JSON already exists.",
    )
    args = parser.parse_args()

    if args.batches < 2:
        parser.error("--batches must be at least 2")

    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    for key in args.cases:
        path = case_file(args.out, key)
        if path.exists() and not args.force:
            print(f"Skipping {key}: {path.name} exists (use --force).")
            continue
        record = run_case(key, args)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"Saved {path}")

    # Collect everything that exists, in the fixed CASES order.
    records = [
        json.loads(case_file(args.out, key).read_text(encoding="utf-8"))
        for key in CASES
        if case_file(args.out, key).exists()
    ]

    (args.out / "reference_uncertainty.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    write_latex_table(records, args.out / "reference_uncertainty_table.tex")
    print_summary(records)
    print(f"\nTotal wall time: {time.perf_counter() - start:.0f} s")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()