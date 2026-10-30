"""WENO-Z benchmarks with the moving active domain enabled.

Identical to the main benchmarks except numerics.moving_active_domain=True.
The BFP Monte Carlo reference is taken from the cache of the main runs. The
dose difference to the main WENO-Z runs and the runtimes are evaluated in
post-processing.

Usage: uv run python -m thesis_experiments.robustness.moving_active_domain
"""

import dataclasses

from thesis_experiments.robustness._common import run_study


def enable_moving_domain(config):
    return dataclasses.replace(
        config,
        numerics=dataclasses.replace(
            config.numerics, moving_active_domain=True
        ),
    )


def main():
    run_study("moving_active_domain", enable_moving_domain)


if __name__ == "__main__":
    main()
