import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


def _to_serializable(value: Any):
    """Convert config objects to stable JSON-compatible values."""

    if isinstance(value, Enum):
        return value.name

    # CT volumes are stored by content hash, not element by element.
    if isinstance(value, np.ndarray):
        return {
            "shape": [int(n) for n in value.shape],
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(
                np.ascontiguousarray(value).tobytes()
            ).hexdigest(),
        }

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return {
            key: _to_serializable(item)
            for key, item in asdict(value).items()
        }

    if isinstance(value, dict):
        return {
            str(key): _to_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _to_serializable(item)
            for item in value
        ]

    return value


def config_dict(config):
    """Return the canonical simulation configuration."""

    data = _to_serializable(config)

    # Output settings do not change the numerical solution
    data.pop("output", None)

    # Geometry viewing does not change the numerical solution
    if "geometry" in data:
        data["geometry"].pop(
            "view_geometry",
            None,
        )

    return data


def config_hash(config) -> str:
    """Return a stable SHA-256 hash of the simulation configuration."""

    payload = json.dumps(
        config_dict(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def _git_command(*args):
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        check=True,
    ).stdout


def git_provenance():
    """Return commit and working-tree provenance information."""

    try:
        commit = (
            _git_command(
                "rev-parse",
                "HEAD",
            )
            .decode()
            .strip()
        )

        repository_root = Path(
            _git_command(
                "rev-parse",
                "--show-toplevel",
            )
            .decode()
            .strip()
        )

        # Includes staged and unstaged changes relative to HEAD
        diff = _git_command(
            "diff",
            "HEAD",
            "--binary",
        )

        # Git diff does not include untracked files
        untracked_output = (
            _git_command(
                "ls-files",
                "--others",
                "--exclude-standard",
            )
            .decode()
            .splitlines()
        )

        hasher = hashlib.sha256()

        hasher.update(
            commit.encode("utf-8")
        )

        hasher.update(diff)

        for relative_path in sorted(
            untracked_output
        ):
            path = (
                repository_root
                / relative_path
            )

            hasher.update(
                relative_path.encode("utf-8")
            )

            if path.is_file():
                hasher.update(
                    path.read_bytes()
                )

        dirty = bool(
            diff
            or untracked_output
        )

        return {
            "git_commit": commit,
            "git_dirty": dirty,
            "git_state_hash": hasher.hexdigest(),
        }

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return {
            "git_commit": None,
            "git_dirty": None,
            "git_state_hash": None,
        }


def _to_numpy(array):
    """Convert NumPy/CuPy arrays to NumPy."""

    if array is None:
        return None

    if hasattr(array, "get"):
        return array.get()

    return np.asarray(array)


def save_result(
    result,
    config,
):
    """Save solver result, configuration, and provenance."""

    root = Path(
        config.output.results_directory
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_digest = config_hash(config)
    git = git_provenance()

    git_digest = (
        git["git_state_hash"][:8]
        if git["git_state_hash"] is not None
        else "nogit"
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    nx, ny, nz = config.geometry.resolution

    scheme = (
        config.numerics
        .spatial_scheme
        .name
        .lower()
    )

    run_name = (
        f"{timestamp}"
        f"__{nx}x{ny}x{nz}"
        f"__{scheme}"
        f"__{config_digest[:8]}"
        f"__{git_digest}"
    )

    run_directory = root / run_name

    run_directory.mkdir()

    # Configuration
    with (
        run_directory / "config.json"
    ).open("w") as file:
        json.dump(
            config_dict(config),
            file,
            indent=2,
            sort_keys=True,
        )

    # Metadata
    deterministic_runtime = result.runtime
    bfp_mc_runtime = result.bfp_mc_runtime
    topas_runtime = result.topas_runtime

    speedup_vs_bfp_mc = (
        None
        if (
            deterministic_runtime is None
            or bfp_mc_runtime is None
            or deterministic_runtime <= 0.0
            or bfp_mc_runtime <= 0.0
        )
        else float(
            bfp_mc_runtime
            / deterministic_runtime
        )
    )

    speedup_vs_topas = (
        None
        if (
            deterministic_runtime is None
            or topas_runtime is None
            or deterministic_runtime <= 0.0
            or topas_runtime <= 0.0
        )
        else float(
            topas_runtime
            / deterministic_runtime
        )
    )

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "config_hash": config_digest,
        **git,
        "runtime_seconds": deterministic_runtime,
        "bfp_mc_runtime_seconds": bfp_mc_runtime,
        "topas_runtime_seconds": topas_runtime,
        "speedup_vs_bfp_mc_x": (
            "-"
            if result.reference_label != "BFP Monte Carlo"
            else speedup_vs_bfp_mc
        ),
        "speedup_vs_topas_x": (
            "-"
            if result.reference_label != "TOPAS"
            else speedup_vs_topas
        ),
        "backend": (
            result.backend.name
            if result.backend is not None
            else None
        ),
    }

    with (
        run_directory / "metadata.json"
    ).open("w") as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    arrays = {
        "x": _to_numpy(result.x),
        "y": _to_numpy(result.y),
        "z": _to_numpy(result.z),
    }

    # Deterministic quantities are only present when
    # the deterministic solver was executed.
    if result.phi is not None:
        arrays["phi"] = _to_numpy(
            result.phi
        )

    if result.J is not None:
        arrays["J"] = _to_numpy(
            result.J
        )

    if result.dose is not None:
        arrays["dose"] = _to_numpy(
            result.dose
        )

    if result.energies is not None:
        arrays["energies"] = _to_numpy(
            result.energies
        )

    # Optional reference solutions.
    if result.analytical_dose is not None:
        arrays["analytical_dose"] = _to_numpy(
            result.analytical_dose
        )

    if result.topas_dose is not None:
        arrays["topas_dose"] = _to_numpy(
            result.topas_dose
        )

    if result.bfp_mc_dose is not None:
        arrays["bfp_mc_dose"] = _to_numpy(
            result.bfp_mc_dose
    )

    np.savez_compressed(
        run_directory / "results.npz",
        **arrays,
    )

    return run_directory