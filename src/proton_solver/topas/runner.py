import hashlib
import shutil
import subprocess
import time
from pathlib import Path
import json

from .writer import TopasCase


def _cache_signature(
    case: TopasCase,
) -> str:
    """Hash everything that changes the TOPAS reference result."""

    hasher = hashlib.sha256()

    for path in (
        case.parameter_file,
        case.preinit_macro,
        case.sequence_macro,
    ):
        hasher.update(
            path.read_bytes()
        )

    return hasher.hexdigest()[:16]


def run_topas(
    case: TopasCase,
    force_rerun: bool = False,
) -> tuple[Path, float | None]:
    """Run or reuse a cached TOPAS case and return the dose CSV path."""

    repo_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    expect_script = (
        repo_root
        / "tools"
        / "topas"
        / "run_topas.exp"
    )

    if not expect_script.exists():
        raise FileNotFoundError(
            "TOPAS expect script was not found: "
            f"{expect_script}"
        )

    # ------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------

    signature = _cache_signature(
        case
    )

    cache_directory = (
        repo_root
        / "topas_cache"
        / signature
    )

    cached_dose_file = (
        cache_directory
        / "DoseAtPhantom.csv"
    )

    cached_metadata_file = (
    cache_directory
    / "metadata.json"
)

    if (
        cached_dose_file.exists()
        and not force_rerun
    ):
        print(
            f"Using cached TOPAS result: {signature}"
        )

        runtime = None

        if cached_metadata_file.exists():
            metadata = json.loads(
                cached_metadata_file.read_text(
                    encoding="utf-8",
                )
            )

            runtime = metadata.get(
                "runtime_seconds"
            )

            if runtime is not None:
                runtime = float(runtime)

        return cached_dose_file, runtime

    # ------------------------------------------------------------
    # Run TOPAS
    # ------------------------------------------------------------

    parameter_file = (
        case.parameter_file.resolve()
    )

    preinit_macro = (
        case.preinit_macro.resolve()
    )

    sequence_macro = (
        case.sequence_macro.resolve()
    )

    command = [
        str(expect_script),
        str(parameter_file),
        str(preinit_macro),
        str(sequence_macro),
    ]

    print(
        "Running TOPAS command:",
        " ".join(command),
        flush=True,
    )

    start = time.perf_counter()

    process = subprocess.Popen(
        command,
        cwd=case.directory,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    output_lines = []

    assert process.stdout is not None

    for line in process.stdout:
        print(
            line,
            end="",
            flush=True,
        )

        output_lines.append(line)

    return_code = process.wait()

    runtime = (
        time.perf_counter()
        - start
    )

    output = "".join(
        output_lines
    )

    log_file = (
        case.directory
        / "topas_console_output.log"
    )

    log_file.write_text(
        output,
        encoding="utf-8",
    )

    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            command,
            output=output,
        )

    dose_file = (
        case.directory
        / "DoseAtPhantom.csv"
    )

    if not dose_file.exists():
        raise FileNotFoundError(
            "TOPAS finished successfully, "
            "but DoseAtPhantom.csv was not created."
        )

    print(
        f"TOPAS runtime: {runtime:.2f} s"
    )

    # ------------------------------------------------------------
    # Save cache
    # ------------------------------------------------------------

    cache_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        case.parameter_file,
        cache_directory / "parameters.txt",
    )

    shutil.copy2(
        case.preinit_macro,
        cache_directory / "preinit.mac",
    )

    shutil.copy2(
        case.sequence_macro,
        cache_directory / "sequence.mac",
    )

    shutil.copy2(
        dose_file,
        cached_dose_file,
    )

    shutil.copy2(
        log_file,
        cache_directory / "topas_console_output.log",
    )

    cached_metadata_file.write_text(
        json.dumps(
            {
                "runtime_seconds": float(runtime),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Cached TOPAS result: {signature}"
    )

    return cached_dose_file, float(runtime)