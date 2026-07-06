"""
Campbell-diagram building blocks.

A Campbell map is amplitude on an (order × shaft-speed) plane.  The rig has no
slow run-up, but every stroke sweeps 0 → nominal → 0 RPM, so a short-time
order analysis in sliding ANGULAR windows across the whole stroke — each
window tagged by its measured mean RPM — accumulates a continuous map across
the sweep.  A vertical stripe at fixed order is a true speed-synchronous
order; a ridge drifting with RPM is a fixed-Hz resonance.

This module owns the two pure pieces: the angular-window slicer and the
(RPM-bin × order) accumulator grid.  The order-spectrum math stays in the
stage scripts (EP_order_analysis / EP_Envelope) that already own it.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import pandas as pd


def order_grid(max_order: float, step: float) -> np.ndarray:
    """Uniform order axis 0 … max_order inclusive (step-sized bins)."""
    return np.arange(0.0, float(max_order) + step / 2.0, float(step))


def angular_windows(phi: np.ndarray, window_rev: float, hop_rev: float,
                    min_samples: int = 4) -> Iterator[tuple[int, int]]:
    """Yield ``(i0, i1)`` index pairs of sliding angular windows over ``phi``.

    ``phi`` is cumulative revolutions (monotonic).  Windows are ``window_rev``
    revolutions long and advance by ``hop_rev``; a window is emitted only when
    it fits entirely inside the stroke and contains at least ``min_samples``
    samples (sparse windows can't support an FFT).
    """
    phi = np.asarray(phi, dtype=float)
    if phi.size < 2 or window_rev <= 0 or hop_rev <= 0:
        return
    total = float(phi[-1] - phi[0])
    if total < window_rev:
        return
    start = float(phi[0])
    eps = 1e-9
    while start + window_rev <= phi[0] + total + eps:
        i0 = int(np.searchsorted(phi, start, side="left"))
        i1 = int(np.searchsorted(phi, start + window_rev, side="right"))
        if i1 - i0 >= max(2, int(min_samples)):
            yield i0, i1
        start += hop_rev


class CampbellGrid:
    """Accumulates per-window order spectra into (material, RPM-bin) cells.

    ``add`` is tolerant of bad input (NaN speed, wrong-shaped spectrum,
    missing material) — such windows are dropped rather than poisoning the
    map.  ``to_long_df`` returns the tidy long form the viz layer plots:
    columns material, rpm (bin center), order, amplitude (cell mean).
    """

    def __init__(self, orders: np.ndarray, rpm_bin: float):
        self.orders = np.asarray(orders, dtype=float)
        self.rpm_bin = float(rpm_bin)
        self._sums: dict[tuple, np.ndarray] = {}
        self._counts: dict[tuple, int] = {}

    def add(self, material: Optional[str], rpm: float,
            spectrum: np.ndarray) -> None:
        if not material:
            return
        try:
            rpm = float(rpm)
        except (TypeError, ValueError):
            return
        if not np.isfinite(rpm):
            return
        spec = np.asarray(spectrum, dtype=float)
        if spec.shape != self.orders.shape or not np.all(np.isfinite(spec)):
            return
        key = (str(material), int(np.floor(rpm / self.rpm_bin)))
        if key in self._sums:
            self._sums[key] += spec
            self._counts[key] += 1
        else:
            self._sums[key] = spec.copy()
            self._counts[key] = 1

    def to_long_df(self) -> pd.DataFrame:
        rows = []
        for (material, bin_idx), total in sorted(self._sums.items()):
            mean = total / self._counts[(material, bin_idx)]
            rpm_center = (bin_idx + 0.5) * self.rpm_bin
            rows.append(pd.DataFrame({
                "material":  material,
                "rpm":       rpm_center,
                "order":     self.orders,
                "amplitude": mean,
            }))
        if not rows:
            return pd.DataFrame(
                columns=["material", "rpm", "order", "amplitude"])
        return pd.concat(rows, ignore_index=True)
