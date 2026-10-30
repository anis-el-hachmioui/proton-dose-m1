from dataclasses import dataclass

from .enums import Backend


@dataclass(frozen=True)
class ResolvedBackend:
    backend: Backend
    xp: object


def resolve_backend(backend: Backend) -> ResolvedBackend:
    """Resolve the requested backend to NumPy or CuPy."""

    if backend is Backend.CPU:
        return _load_cpu_backend()

    if backend is Backend.GPU:
        return _load_gpu_backend()

    if backend is Backend.AUTO:
        try:
            return _load_gpu_backend()
        except RuntimeError:
            return _load_cpu_backend()

    raise ValueError(f"Unknown backend: {backend}")


def _load_cpu_backend() -> ResolvedBackend:
    """Load the NumPy CPU backend."""

    import numpy as np

    return ResolvedBackend(
        backend=Backend.CPU,
        xp=np,
    )


def _load_gpu_backend() -> ResolvedBackend:
    """Load and verify the CuPy GPU backend."""

    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError

        cp.zeros(1)

    except Exception as exc:
        raise RuntimeError(
            "GPU backend requested, but CUDA/CuPy is unavailable."
        ) from exc

    return ResolvedBackend(
        backend=Backend.GPU,
        xp=cp,
    )