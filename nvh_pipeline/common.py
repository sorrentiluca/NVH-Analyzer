"""
Shared numerics and dataset helpers used by every stage and by the app.

Everything here is pure (no Streamlit, no matplotlib): RMS definitions,
steady-state plateau gating, anti-aliased angular resampling, material
classification, and the raw-CSV folder scanners the front end uses for
auto-detection and header QA.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Material classification
# ─────────────────────────────────────────────────────────────────────────────

def material_type(part_id: str) -> str:
    """'m*' → metal, 't*' → plastic (case-insensitive), anything else unknown."""
    if not part_id:
        return "unknown"
    first = str(part_id)[0].lower()
    if first == "m":
        return "metal"
    if first == "t":
        return "plastic"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
#  RMS definitions (exact engineering forms)
# ─────────────────────────────────────────────────────────────────────────────

def per_axis_rms(signal: np.ndarray, remove_dc: bool = False) -> float:
    """RMS = sqrt(mean(x²)).  With remove_dc the per-signal mean is subtracted
    first, giving the AC RMS (equal to the population standard deviation)."""
    s = np.asarray(signal, dtype=float)
    if s.size == 0:
        return float("nan")
    if remove_dc:
        s = s - np.mean(s)
    return float(np.sqrt(np.mean(s ** 2)))


def combined_axes_rms(signals: Iterable[np.ndarray],
                      remove_dc: bool = False) -> float:
    """Vector RMS across axes: sqrt(mean(x² + y² + z²)).

    This is the RMS of the acceleration magnitude — NOT the sum of per-axis
    RMS values (which over-counts).  remove_dc subtracts each axis' own mean
    first.  Empty input returns NaN.
    """
    signals = list(signals)
    if not signals:
        return float("nan")
    total_sq = np.zeros_like(np.asarray(signals[0], dtype=float))
    for sig in signals:
        s = np.asarray(sig, dtype=float)
        if remove_dc:
            s = s - np.mean(s)
        total_sq += s ** 2
    return float(np.sqrt(np.mean(total_sq)))


# ─────────────────────────────────────────────────────────────────────────────
#  Steady-state (plateau) gating
# ─────────────────────────────────────────────────────────────────────────────

def plateau_mask(rpm: np.ndarray, category: Optional[float],
                 frac: float = 0.90) -> np.ndarray:
    """Boolean mask of the constant-speed plateau: |RPM| ≥ frac × reference.

    ``category`` is the nominal speed class; when None (or non-positive) the
    stroke's own peak |RPM| is the reference, so the mask still isolates the
    fastest portion.  Direction sign is ignored (retract strokes gate too).
    """
    r = np.abs(np.asarray(rpm, dtype=float))
    if category is not None and float(category) > 0:
        ref = abs(float(category))
    else:
        ref = float(np.nanmax(r)) if r.size else 0.0
    if ref <= 0:
        return np.zeros(r.shape, dtype=bool)
    return r >= frac * ref


def apply_plateau(rpm: np.ndarray, category: Optional[float],
                  frac: float = 0.90,
                  min_samples: int = 16) -> tuple[np.ndarray, bool]:
    """Plateau mask with a full-stroke fallback.

    Returns ``(mask, used_plateau)``.  If fewer than ``min_samples`` samples
    survive the gate the full stroke is analysed instead (mask of all True,
    used_plateau False) — a stroke is never silently dropped.
    """
    mask = plateau_mask(rpm, category, frac)
    if int(mask.sum()) >= int(min_samples):
        return mask, True
    return np.ones(len(np.asarray(rpm)), dtype=bool), False


# ─────────────────────────────────────────────────────────────────────────────
#  Anti-aliased angular resampling  (R1)
# ─────────────────────────────────────────────────────────────────────────────

def angular_resample_antialiased(sig: np.ndarray, phi: np.ndarray,
                                 phi_uniform: np.ndarray,
                                 lp_hz: float, fs: float) -> np.ndarray:
    """Low-pass filter in the TIME domain, then interpolate onto the uniform
    angular grid.

    Plain interpolation onto an angular grid aliases any time-domain content
    above the angular Nyquist (samples_per_rev/2 orders at the slowest speed in
    the window) back into the order band.  Filtering at ``lp_hz`` — the angular
    Nyquist expressed in Hz — before resampling removes that fold-back.

    A zero-phase 4th-order Butterworth (``sosfiltfilt``) is used so the filter
    adds no phase distortion (order positions are preserved exactly).  If the
    cutoff is at/above the usable band edge the filter is skipped — nothing to
    remove.
    """
    from scipy.signal import butter, sosfiltfilt

    sig = np.asarray(sig, dtype=float)
    nyq = float(fs) / 2.0
    if lp_hz is not None and fs and 0 < lp_hz < 0.99 * nyq and sig.size > 12:
        sos = butter(4, lp_hz / nyq, btype="low", output="sos")
        try:
            sig = sosfiltfilt(sos, sig)
        except ValueError:
            pass                      # segment shorter than filter padding
    return np.interp(phi_uniform, phi, sig)


# ─────────────────────────────────────────────────────────────────────────────
#  Raw-CSV folder scanners (used by the app's Define & Setup screens)
# ─────────────────────────────────────────────────────────────────────────────

# <part>-<rpm>rpm-<trial>.csv  (case-insensitive 'rpm', trial = integer)
_NAME_RE = re.compile(
    r"^(?P<part>.+?)-(?P<rpm>\d+(?:\.\d+)?)rpm-(?P<trial>\d+)\.csv$",
    re.IGNORECASE,
)


def discover_dataset(data_dir: str) -> dict:
    """Scan a raw-data folder for the <part>-<rpm>rpm-<trial>.csv grammar.

    Returns::

        {n_files, matched, parts, rpms,
         counts: [{part, rpm, n_files}, …], unmatched: [name, …]}

    ``n_files`` counts every CSV present; ``matched`` those that follow the
    naming grammar.  Non-CSV files are ignored entirely.
    """
    out = {"n_files": 0, "matched": 0, "parts": [], "rpms": [],
           "counts": [], "unmatched": []}
    if not data_dir or not os.path.isdir(data_dir):
        return out

    counts: dict[tuple, int] = {}
    for name in sorted(os.listdir(data_dir)):
        if not name.lower().endswith(".csv"):
            continue
        out["n_files"] += 1
        m = _NAME_RE.match(name)
        if not m:
            out["unmatched"].append(name)
            continue
        out["matched"] += 1
        part = m.group("part")
        rpm = int(float(m.group("rpm")))
        counts[(part, rpm)] = counts.get((part, rpm), 0) + 1

    out["parts"] = sorted({p for p, _ in counts})
    out["rpms"] = sorted({r for _, r in counts})
    out["counts"] = [{"part": p, "rpm": r, "n_files": n}
                     for (p, r), n in sorted(counts.items())]
    return out


def _read_header(path: str) -> Optional[list]:
    """Column names of one CSV (header row only).  [] = empty file,
    None = unreadable."""
    try:
        if os.path.getsize(path) == 0:
            return []
        import pandas as pd
        return [str(c) for c in pd.read_csv(path, nrows=0).columns]
    except OSError:
        return None
    except Exception:
        # pandas raises EmptyDataError for header-less files; other parse
        # errors mean the header genuinely can't be established.
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                first = fh.readline().strip()
            return [c.strip() for c in first.split(",")] if first else []
        except Exception:
            return None


def inspect_headers(data_dir: str) -> dict:
    """Header-consistency QA over a folder of CSVs (reads only header rows).

    Files are compared BY NAME against the majority column signature, so a
    reordered file is explicitly 'handled' rather than flagged.  Statuses:

        identical   exact same columns, same order
        reordered   same column set, different order (processing unaffected)
        mismatch    missing and/or unexpected columns — needs review
        empty       no header row

    Returns ``{n_files, n_mismatched, n_reordered, reference_columns,
    all_columns, unreadable, files:[{name, columns, status,
    matches_reference, missing, extra}]}``.
    """
    result = {"n_files": 0, "n_mismatched": 0, "n_reordered": 0,
              "reference_columns": [], "all_columns": [],
              "unreadable": [], "files": []}
    if not data_dir or not os.path.isdir(data_dir):
        return result

    headers: list[tuple[str, list]] = []
    for name in sorted(os.listdir(data_dir)):
        if not name.lower().endswith(".csv"):
            continue
        path = os.path.join(data_dir, name)
        cols = _read_header(path)
        if cols is None:
            result["unreadable"].append((name, "could not read header"))
            continue
        headers.append((name, cols))

    result["n_files"] = len(headers)
    if not headers:
        return result

    # Reference = the most common non-empty signature (ties → first seen).
    # Only if every file is empty does the empty signature become reference.
    sig_counts: dict[tuple, int] = {}
    for _, cols in headers:
        sig_counts[tuple(cols)] = sig_counts.get(tuple(cols), 0) + 1
    non_empty = {s: n for s, n in sig_counts.items() if s}
    pool = non_empty or sig_counts
    first_seen = {tuple(cols): i for i, (_, cols) in
                  reversed(list(enumerate(headers)))}
    reference = list(max(pool, key=lambda s: (pool[s], -first_seen.get(s, 0))))
    ref_set = set(reference)

    union: set = set()
    for name, cols in headers:
        union |= set(cols)
        cset = set(cols)
        if not cols and reference:
            status = "empty"
        elif cols == reference:
            status = "identical"
        elif cset == ref_set:
            status = "reordered"
        else:
            status = "mismatch"
        matches = status in ("identical", "reordered")
        if status == "reordered":
            result["n_reordered"] += 1
        if not matches:
            result["n_mismatched"] += 1
        result["files"].append({
            "name": name,
            "columns": cols,
            "status": status,
            "matches_reference": matches,
            "missing": sorted(ref_set - cset),
            "extra": sorted(cset - ref_set),
        })

    result["reference_columns"] = reference
    result["all_columns"] = sorted(union)
    return result
