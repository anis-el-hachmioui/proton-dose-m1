"""Extended-model self-convergence with PCHIP-interpolated coefficients.

Diagnostic for the non-monotone dose errors of the extended-model WENO-Z
convergence study. The study of
thesis_experiments/numerical_verification/scattering_nuclear_convergence.py
is repeated unchanged, except that the tabulated Wentzel transport and
nuclear removal coefficients are interpolated with PCHIP (continuously
differentiable) instead of linearly (kinks at every table energy). If the
irregularity disappears, it is caused by the linear interpolation.

The default of the solver stays LINEAR, so no other result is affected.

Output: thesis_results/numerical_verification/
        scattering_nuclear_convergence_pchip_coefficients/

Usage: uv run python -m thesis_experiments.robustness.convergence_pchip_coefficients
"""

import dataclasses
import json
from pathlib import Path

from proton_solver import CoefficientInterpolation

from thesis_experiments.numerical_verification import (
    scattering_nuclear_convergence as extended,
)
from thesis_experiments.numerical_verification._convergence_common import (
    run_convergence_study,
)

LINEAR_DIR = Path(
    "thesis_results/numerical_verification/scattering_nuclear_convergence"
)
RESULT_DIR = Path(
    "thesis_results/numerical_verification/"
    "scattering_nuclear_convergence_pchip_coefficients"
)


def _l2_errors(result_dir: Path):
    path = result_dir / "convergence_summary.json"
    if not path.is_file():
        return None
    observables = json.loads(path.read_text(encoding="utf-8"))["observables"]
    return {
        key: (data["l2"]["errors"], data["l2"]["fitted_order"])
        for key, data in observables.items()
    }


def compare():
    """Print the relative L2 errors of both interpolations side by side."""

    linear = _l2_errors(LINEAR_DIR)
    pchip = _l2_errors(RESULT_DIR)
    if pchip is None:
        print("No PCHIP results found.")
        return

    print()
    print("Relative L2 self-convergence errors, linear vs PCHIP coefficients")
    for key, (errors_p, order_p) in pchip.items():
        print(f"\n{key}")
        errors_l, order_l = (
            linear[key] if linear and key in linear else ([None] * len(errors_p), None)
        )
        for i, (el, ep) in enumerate(zip(errors_l, errors_p)):
            el_text = f"{el:.3e}" if el is not None else "   n/a   "
            print(f"  pair {i + 1}:  linear {el_text}   pchip {ep:.3e}")
        if order_l is not None:
            print(f"  fitted order:  linear {order_l:.3f}   pchip {order_p:.3f}")
        else:
            print(f"  fitted order:  pchip {order_p:.3f}")


def main():
    base = extended.build_config()
    config = dataclasses.replace(
        base,
        physics=dataclasses.replace(
            base.physics,
            coefficient_interpolation=CoefficientInterpolation.PCHIP,
        ),
    )
    run_convergence_study(
        config,
        RESULT_DIR,
        physics_label=(
            "CSDA + streaming + Wentzel scattering + nuclear attenuation, "
            "PCHIP coefficients"
        ),
        title_prefix="Extended model, PCHIP coefficients",
    )
    compare()


if __name__ == "__main__":
    main()
