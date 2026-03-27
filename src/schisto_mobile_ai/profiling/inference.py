"""Benchmark helpers for CPU or MPS forward-pass profiling."""

from __future__ import annotations

import time
from typing import Any

import torch


def _move_batch_to_device(batch: Any, device: str) -> Any:
    """Move tensors inside a nested structure onto the requested device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        moved = [_move_batch_to_device(item, device) for item in batch]
        return type(batch)(moved)
    if isinstance(batch, dict):
        return {key: _move_batch_to_device(value, device) for key, value in batch.items()}
    return batch


def benchmark_forward_pass(
    model: torch.nn.Module,
    batch: Any,
    *,
    device: str = "cpu",
    warmup_iters: int = 5,
    benchmark_iters: int = 20,
) -> dict[str, float | int | str]:
    """Measure simple latency and throughput for a prepared input batch."""
    if benchmark_iters <= 0:
        raise ValueError("benchmark_iters must be positive.")

    model = model.to(device)
    model.eval()
    batch = _move_batch_to_device(batch, device)

    with torch.inference_mode():
        for _ in range(warmup_iters):
            _ = model(*batch) if isinstance(batch, tuple) else model(batch)

        if device == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()
        for _ in range(benchmark_iters):
            _ = model(*batch) if isinstance(batch, tuple) else model(batch)
        if device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start

    batch_size = 1
    if isinstance(batch, torch.Tensor) and batch.ndim > 0:
        batch_size = int(batch.shape[0])
    elif isinstance(batch, tuple) and batch and isinstance(batch[0], torch.Tensor):
        batch_size = int(batch[0].shape[0])

    mean_latency_ms = 1000.0 * elapsed / benchmark_iters
    throughput_items_per_sec = (batch_size * benchmark_iters) / elapsed

    return {
        "device": device,
        "batch_size": batch_size,
        "warmup_iters": warmup_iters,
        "benchmark_iters": benchmark_iters,
        "mean_latency_ms": mean_latency_ms,
        "throughput_items_per_sec": throughput_items_per_sec,
    }

