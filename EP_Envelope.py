"""
═══════════════════════════════════════════════════════════════════════════════
 ENVELOPE ORDER ANALYSIS  —  Metal vs Plastic  |  1600 & 2300 RPM
═══════════════════════════════════════════════════════════════════════════════

Applies envelope analysis to the high-speed actuations (1600 and 2300 RPM)
and compares the resulting order spectra between metal and plastic.

WHAT IS ENVELOPE ANALYSIS
--------------------------
  Rolling-element bearing and recirculation defects generate brief impulses
  at the ball pass frequency.  These impulses are often buried under broad
  low-frequency noise in the raw signal but modulate a high-frequency carrier
  (the accelerometer structural resonance).  Envelope analysis extracts that
  modulation:

    1. Bandpass the raw signal around the structural resonance to isolate the
       high-frequency carrier and its sidebands.
    2. Rectify via Hilbert transform to get the instantaneous amplitude
       (the envelope).
    3. Run order analysis on the envelope signal instead of the raw signal.

  Peaks in the envelope order spectrum at the ball pass order (5.35) and its
  harmonics are a direct indicator of recirculation-induced impulsive loading.

PER-ACTUATION BANDPASS AUTO-DETECTION
--------------------------------------
  Rather than using a global resonance estimate, this script detects the
  optimal bandpass band independently for EACH actuation and EACH axis:

    1. Compute the Kurtogram (spectral kurtosis map) over a range of
       center frequencies and bandwidths.
    2. Select the (f_center, bandwidth) cell with the highest kurtosis.
       Kurtosis measures impulsiveness — the band with the highest kurtosis
       contains the most bearing/defect impulse content.
    3. Apply a Butterworth bandpass filter at the selected band.
    4. Extract the envelope via Hilbert transform.

  This is the state-of-the-art approach (Antoni, 2007) and adapts
  automatically to each actuation's noise floor and resonance structure
  without any manual tuning.

NORMALIZATION
-------------
  Each axis spectrum is divided by the number of actuations before forming
  the cumulative sum, so metal and plastic are compared on a per-actuation
  basis regardless of trial count.

OUTPUTS
-------
  output_envelope/
    envelope_spectra_mean_metal.csv
    envelope_spectra_mean_plastic.csv
    bandpass_log.csv                     detected band per actuation/axis
    run_metadata.json
    plots/
      envelope_cumulative_comparison.png   primary deliverable

DEPENDENCIES
------------
  pip install pandas numpy scipy matplotlib pyarrow
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import get_window
from scipy.fft import next_fast_len, rfft, irfft


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── IO ──────────────────────────────────────────────────────────────────
    seg_output_dir: str = 'output_seg'
    data_subdir: str    = 'segmented_data'
    seg_glob: str       = '*_segments.parquet'

    output_dir: str   = 'output_envelope'
    plots_subdir: str = 'plots'

    # ── COLUMNS ─────────────────────────────────────────────────────────────
    time_col: str  = 'Time (s)'
    rpm_col: str   = 'CNT 1/Frequency (RPM)'
    angle_col: str = 'CNT 1/Angle (Degrees)'

    axis_x: Optional[str] = None
    axis_y: Optional[str] = None
    axis_z: Optional[str] = None

    # ── RPM FILTER ───────────────────────────────────────────────────────────
    only_rpms: tuple = (100, 200, 400, 800, 1600, 2300)

    # ── PER-ACTUATION KURTOGRAM SETTINGS ────────────────────────────────────
    # Minimum frequency considered for resonance detection.
    # Computed dynamically per actuation as max(bp_min_hz_abs, 25*rpm/60*1.2)
    # so the floor scales with actual speed; bp_min_hz is still the upper bound
    # used when no dynamic value is supplied (e.g. headless CLI run at fixed RPM).
    bp_min_hz: float = 1000.0
    bp_min_hz_abs: float = 50.0   # absolute floor — never search below this Hz

    # Upper search bound as a fraction of Nyquist.  Real structural resonances
    # sit well below Nyquist; the top of the band is dominated by sensor noise
    # whose Gibbs-ringing artifacts can spuriously win the kurtogram.  0.8 keeps
    # band selection physical (e.g. ≤ 40 kHz at 100 kHz sampling).
    bp_max_hz_frac: float = 0.8

    # Absolute upper bound [Hz] on the resonance band — keeps it inside the
    # accelerometer's calibrated range.  The sensor used here is flat to 9 kHz
    # (±5%) / 12 kHz (±10%) with mounted resonance ≥45 kHz, so bands above ~12 kHz
    # sit in the uncalibrated rising-noise region where the kurtogram latches onto
    # noise.  Effective cap = min(bp_max_hz, bp_max_hz_frac * fs/2).
    bp_max_hz: float = 12000.0

    # Resonance-band scope.  'per_part' = detect ONE band per specimen from its
    # highest-RPM plateau segments and demodulate all of that part's segments with
    # it (structural resonance is speed-independent → comparable by-speed spectra).
    # 'per_segment' = legacy per-segment, per-axis re-detection.
    band_scope: str = 'per_part'

    # 'reciprocating' (ballscrew rig) or 'continuous' (single steady signal —
    # allows a single vibration channel; missing axes become zero channels).
    analysis_mode: str = 'reciprocating'

    # Steady-state (plateau) gating — demodulate only the constant-speed plateau.
    plateau_gating: bool = True
    plateau_frac: float = 0.90

    # Campbell (envelope-order × measured-RPM waterfall from the ramp sweep).
    campbell: bool             = True
    campbell_window_rev: float = 1.5
    campbell_hop_rev: float    = 0.75
    campbell_rpm_bin: float    = 50.0
    campbell_order_step: float = 0.1

    # Kurtogram frequency resolution levels.  Higher = finer search but slower.
    # Each level halves the bandwidth: level 1 = fs/4, level 2 = fs/8, etc.
    # MUST match PipelineConfig.kurtogram_levels (nvh_pipeline/config.py) so a
    # standalone run and a pipeline run select bands identically.
    kurtogram_levels: int = 6

    # Minimum bandwidth to consider [Hz].  Kept at 200 (not lowered with the level
    # bump): narrower bands drop the modulation sidebands envelope demod needs and
    # collapse ball-pass recovery (golden-test verified).  This also floors the
    # effective search depth on low-fs data.
    bp_min_bw_hz: float = 200.0

    # Minimum raw kurtosis (mu4/mu2^2) to accept a band as impulsive.
    # 3.0 = Gaussian noise baseline; segments below this threshold are skipped
    # because their best band is not meaningfully impulsive (defect not detected).
    # Set to 0.0 to disable the gate and accumulate all segments regardless.
    min_kurtosis: float = 3.0

    # Normalize each segment's envelope spectrum by its combined RMS before
    # averaging, so the mean reflects spectral shape (impulsiveness) rather than
    # absolute loudness and no single loud actuation dominates.
    normalize_segments: bool = True

    # Butterworth filter order.
    bp_filter_order: int = 4

    # ── ANGLE SOURCE ────────────────────────────────────────────────────────
    angle_source: str = 'encoder'

    # ── ORDER ANALYSIS ──────────────────────────────────────────────────────
    samples_per_rev: int       = 128
    detrend: str               = 'mean'
    window: str                = 'hann'
    max_order: Optional[float] = 25.0
    order_resolution: float    = 0.02
    # 10 revolutions → 0.1-order resolution; enough to cleanly resolve BPFO
    # 5.35 without smearing into adjacent shaft orders (R2).
    min_revolutions: float     = 10.0
    min_fft_points: int        = 16

    # ── AXIS AGGREGATION (parity with EP_order_analysis) ─────────────────────
    # How per-axis envelope spectra are averaged across segments and combined into
    # the cumulative trace.  Kept identical to the order stage so the two
    # deliverables are energy-consistent:
    #   order_average:   'power' = sqrt(mean|A|²) RMS/energy-mean (default),
    #                    'magnitude' = legacy arithmetic mean(|A|).
    #   cumulative_mode: 'rss'   = sqrt(x²+y²+z²) vector magnitude (default),
    #                    'sum'    = legacy linear |x|+|y|+|z| (over-counts).
    order_average: str   = 'power'
    cumulative_mode: str = 'rss'

    # ── BALL PASS ORDER ──────────────────────────────────────────────────────
    ball_pass_order: float = 5.35
    n_bpf_harmonics: int   = 4

    # ── PLOTTING ────────────────────────────────────────────────────────────
    show_plots: bool          = False
    render_plots: bool        = True
    mark_integer_orders: bool = True
    log_y_axis: bool          = False

    color_metal:   str = '#2c7bb6'
    color_plastic: str = '#f46d43'


# ═══════════════════════════════════════════════════════════════════════════════
#  COLUMN RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

_NON_AXIS_EXTRA = {
    'segment_id', 'direction', 't_rel',
    'part', 'material_type', 'rpm_category', 'trial', 'source_file',
}
_NON_AXIS_KEYWORDS = ('time', 'rpm', 'frequency', 'angle', 'degree', 'index')


def _looks_like_axis(col_lower: str, axis: str) -> bool:
    return re.search(rf'(?<![a-z]){axis}(?![a-z])', col_lower) is not None


def resolve_axis_columns(columns: list[str], cfg: Config) -> dict:
    explicit = {'x': cfg.axis_x, 'y': cfg.axis_y, 'z': cfg.axis_z}
    known_non_axis = {cfg.time_col, cfg.rpm_col, cfg.angle_col} | _NON_AXIS_EXTRA

    candidates = [
        c for c in columns
        if c not in known_non_axis
        and not any(k in c.lower() for k in _NON_AXIS_KEYWORDS)
    ]

    # Continuous mode may carry only one/two vibration channels; a missing axis
    # then resolves to None and is skipped downstream (zero channel).
    allow_missing = getattr(cfg, 'analysis_mode', 'reciprocating') == 'continuous'

    resolved: dict = {}
    for axis in ('x', 'y', 'z'):
        if explicit[axis] is not None:
            if explicit[axis] not in columns:
                raise KeyError(
                    f"configured axis_{axis} = {explicit[axis]!r} not found. "
                    f"Available: {columns}"
                )
            resolved[axis] = explicit[axis]
            continue

        hits = [c for c in candidates if _looks_like_axis(c.lower(), axis)]
        preferred = [c for c in hits
                     if any(k in c.lower()
                            for k in ('acc', 'vib', 'accel', '(g)', ' g'))]
        pick_from = preferred or hits
        if not pick_from:
            if allow_missing:
                resolved[axis] = None
                continue
            raise KeyError(
                f"could not auto-detect {axis.upper()} axis. "
                f"Set Config.axis_{axis} explicitly. "
                f"Candidates: {candidates}"
            )
        resolved[axis] = pick_from[0]

    present = [v for v in resolved.values() if v is not None]
    if not present:
        raise KeyError(
            f"no vibration axis columns found. Set axis_x/y/z explicitly. "
            f"Columns: {columns}"
        )
    if len(set(present)) != len(present):
        raise KeyError(
            f"axis auto-detect produced duplicates {resolved}. "
            f"Set axis_x/y/z explicitly."
        )
    return resolved


# ═══════════════════════════════════════════════════════════════════════════════
#  SAMPLE RATE
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_fs(time: np.ndarray) -> float:
    if np.asarray(time).size < 2:
        raise ValueError("need >= 2 time samples to estimate sample rate")
    dt = np.median(np.diff(time))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("non-positive / non-finite time step — check time column")
    return 1.0 / dt


# ═══════════════════════════════════════════════════════════════════════════════
#  KURTOGRAM — PER-ACTUATION PER-AXIS BANDPASS DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_band_kurtogram(sig: np.ndarray, fs: float,
                           cfg: Config,
                           min_hz: float | None = None,
                           return_grid: bool = False):
    """
    FFT-domain Fast Kurtogram (Antoni 2007 concept).

    Computes one forward transform per call, then evaluates each dyadic band by
    masking spectral bins (with a raised-cosine taper at the edges to suppress
    Gibbs ringing) and reconstructing via IFFT.  O(N log N) per band instead of
    two full IIR passes.

    Returns (f_low, f_center, f_high, best_kurtosis).  With ``return_grid=True``
    also returns a 5th element: a list of per-band dicts
    ``{level, f_low_hz, f_center_hz, f_high_hz, bandwidth_hz, kurtosis}`` for the
    whole search grid (used to render the kurtogram heatmap) — the returned band
    is the grid's max-kurtosis cell.

    ``extract_envelope`` then demodulates this band with an FFT analytic-signal
    envelope (no IIR filter), so any returned band is numerically safe.

    ``min_hz`` overrides cfg.bp_min_hz for this call only (dynamic RPM floor).
    The search is capped at ``cfg.bp_max_hz_frac * nyq`` to keep band selection
    in the physical resonance range and out of the noisy near-Nyquist region.
    """
    eff_min  = min_hz if min_hz is not None else cfg.bp_min_hz
    nyq      = fs / 2.0
    f_max    = min(nyq,
                   getattr(cfg, 'bp_max_hz_frac', 1.0) * nyq,
                   getattr(cfg, 'bp_max_hz', nyq))
    f_usable = max(0.0, f_max - eff_min)

    grid: list = []
    n = len(sig)
    if n < 4:
        band = (eff_min, (eff_min + f_max) / 2, f_max, 3.0)
        return (*band, grid) if return_grid else band

    # One forward transform shared across all band evaluations.
    n_fft  = next_fast_len(n)
    spec   = rfft(sig, n=n_fft)                        # complex, shape: n_fft//2+1
    freqs  = np.arange(n_fft // 2 + 1) * (fs / n_fft) # Hz axis

    best_kurt = -np.inf
    best_band = (eff_min, (eff_min + f_max) / 2, f_max)

    for level in range(1, cfg.kurtogram_levels + 1):
        n_bands = 2 ** level
        bw      = f_usable / n_bands
        if bw < cfg.bp_min_bw_hz:
            break

        for k in range(n_bands):
            f_low = eff_min + k * bw
            f_high = f_low + bw
            f_ctr  = (f_low + f_high) / 2.0

            mask = (freqs >= f_low) & (freqs <= f_high)
            if not np.any(mask):
                continue

            n_mask = int(np.sum(mask))
            # Raised-cosine taper on the outer 10 % of bins at each edge to
            # suppress Gibbs ringing that would otherwise inflate kurtosis.
            taper_n = max(1, n_mask // 10)
            taper   = np.ones(n_mask, dtype=float)
            ramp    = 0.5 * (1 - np.cos(np.pi * np.arange(taper_n) / taper_n))
            taper[:taper_n]  = ramp
            taper[-taper_n:] = ramp[::-1]

            band_spec          = np.zeros_like(spec)
            band_spec[np.where(mask)[0]] = spec[np.where(mask)[0]] * taper
            band_sig = irfft(band_spec, n=n_fft)[:n]

            # Raw kurtosis: mu4 / mu2^2 (non-excess, matching old implementation).
            mu  = np.mean(band_sig)
            dev = band_sig - mu
            mu2 = np.mean(dev * dev)
            if mu2 < 1e-30:
                continue
            kurt = float(np.mean(dev * dev * dev * dev) / (mu2 * mu2))

            if return_grid:
                grid.append({
                    'level':        level,
                    'f_low_hz':     round(f_low, 1),
                    'f_center_hz':  round(f_ctr, 1),
                    'f_high_hz':    round(f_high, 1),
                    'bandwidth_hz': round(bw, 1),
                    'kurtosis':     round(kurt, 3),
                })

            if kurt > best_kurt:
                best_kurt = kurt
                best_band = (f_low, f_ctr, f_high)

    kurt_out = best_kurt if best_kurt > -np.inf else 3.0
    band = (best_band[0], best_band[1], best_band[2], kurt_out)
    return (*band, grid) if return_grid else band


def band_kurtosis(sig: np.ndarray, fs: float,
                  f_low: float, f_high: float) -> float:
    """Raw kurtosis (mu4/mu2²) of *sig* band-limited to [f_low, f_high].

    Single masked-IFFT reconstruction (same taper as the kurtogram) — used to
    score a *fixed* (per-part) band on each segment so the impulsiveness gate
    still applies when band selection is no longer per-segment.
    """
    n = len(sig)
    if n < 4:
        return 3.0
    n_fft = next_fast_len(n)
    spec  = rfft(sig, n=n_fft)
    freqs = np.arange(n_fft // 2 + 1) * (fs / n_fft)
    mask  = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 3.0
    idx     = np.where(mask)[0]
    n_mask  = idx.size
    taper_n = max(1, n_mask // 10)
    taper   = np.ones(n_mask, dtype=float)
    ramp    = 0.5 * (1 - np.cos(np.pi * np.arange(taper_n) / taper_n))
    taper[:taper_n]  = ramp
    taper[-taper_n:] = ramp[::-1]
    band_spec      = np.zeros_like(spec)
    band_spec[idx] = spec[idx] * taper
    band_sig = irfft(band_spec, n=n_fft)[:n]
    mu  = np.mean(band_sig)
    dev = band_sig - mu
    mu2 = np.mean(dev * dev)
    if mu2 < 1e-30:
        return 3.0
    return float(np.mean(dev * dev * dev * dev) / (mu2 * mu2))


# ═══════════════════════════════════════════════════════════════════════════════
#  ENVELOPE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_envelope(sig: np.ndarray, fs: float,
                     f_low: float, f_high: float,
                     cfg: Config) -> np.ndarray:
    """
    Band-limited analytic-signal envelope, computed entirely in the frequency
    domain.  This is the Hilbert-via-FFT demodulation: keep only the positive
    in-band spectrum (doubled), zero DC / negative / out-of-band bins, inverse
    transform, and take the magnitude.

    Unlike a Butterworth + ``filtfilt`` bandpass, this has no IIR poles and so
    is *unconditionally stable* — it cannot diverge even when the kurtogram
    selects a band whose edge approaches Nyquist (the old path exploded to
    ~1e25+ there, swamping the per-actuation mean).  A raised-cosine taper on
    the band edges (matching ``detect_band_kurtogram``) suppresses Gibbs
    ringing.  In the numerically safe regime it agrees with the old Butterworth
    envelope to ~0.998 correlation.
    """
    nyq = fs / 2.0
    f_lo = max(f_low, 0.0)
    f_hi = min(f_high, nyq)

    if f_lo >= f_hi:
        return np.abs(sig - np.mean(sig))

    n     = len(sig)
    n_fft = next_fast_len(n)
    spec  = np.fft.fft(sig, n=n_fft)
    freqs = np.fft.fftfreq(n_fft, d=1.0 / fs)

    # Analytic band filter: gain 2.0 on positive in-band bins, 0 elsewhere.
    H   = np.zeros(n_fft, dtype=float)
    pos = (freqs >= f_lo) & (freqs <= f_hi)
    H[pos] = 2.0

    # Raised-cosine taper over the outer 10 % of in-band bins (both edges).
    idx = np.where(pos)[0]
    if idx.size > 4:
        taper_n = max(1, idx.size // 10)
        ramp    = 0.5 * (1 - np.cos(np.pi * np.arange(taper_n) / taper_n))
        H[idx[:taper_n]]  *= ramp
        H[idx[-taper_n:]] *= ramp[::-1]

    analytic = np.fft.ifft(spec * H)[:n]
    return np.abs(analytic)


# ═══════════════════════════════════════════════════════════════════════════════
#  ANGULAR DOMAIN
# ═══════════════════════════════════════════════════════════════════════════════

def phi_revolutions(seg_df: pd.DataFrame, cfg: Config) -> np.ndarray:
    use_encoder = (cfg.angle_source == 'encoder'
                   and cfg.angle_col in seg_df.columns)
    if use_encoder:
        ang = seg_df[cfg.angle_col].to_numpy(dtype=float)
        phi = np.abs(ang - ang[0]) / 360.0
        # R9: warn when maximum.accumulate clamps a large fraction of steps,
        # indicating encoder wrapping, glitches, or genuine reversals being
        # silently masked.
        if phi.size > 1:
            steps = np.diff(phi)
            n_backward = int(np.sum(steps < 0))
            frac = n_backward / len(steps)
            if frac > 0.05:
                import warnings
                warnings.warn(
                    f"phi_revolutions: {n_backward}/{len(steps)} "
                    f"({frac:.1%}) of encoder steps are non-monotonic — "
                    f"check encoder column '{cfg.angle_col}' for wrapping or "
                    f"glitches (segment forced monotonic via maximum.accumulate).",
                    stacklevel=2,
                )
    else:
        t         = seg_df[cfg.time_col].to_numpy(dtype=float)
        rev_per_s = np.abs(seg_df[cfg.rpm_col].to_numpy(dtype=float)) / 60.0
        phi = np.concatenate(
            [[0.0],
             np.cumsum(0.5 * (rev_per_s[1:] + rev_per_s[:-1]) * np.diff(t))]
        )
    return np.maximum.accumulate(phi)


def envelope_order_spectrum(envelope: np.ndarray,
                             phi: np.ndarray,
                             M: int,
                             window: np.ndarray,
                             phi_uniform: np.ndarray,
                             cfg: Config,
                             lp_hz: float = None,
                             fs: float = None) -> np.ndarray:
    """Angular resample envelope + windowed FFT.

    When *lp_hz* and *fs* are provided the envelope is anti-alias filtered
    before resampling (R1).  Falls back to plain interp when omitted.
    """
    if lp_hz is not None and fs is not None:
        try:
            from nvh_pipeline.common import angular_resample_antialiased
            env_r = angular_resample_antialiased(envelope, phi, phi_uniform,
                                                  lp_hz, fs)
        except ImportError:
            env_r = np.interp(phi_uniform, phi, envelope)
    else:
        env_r = np.interp(phi_uniform, phi, envelope)

    if cfg.detrend == 'mean':
        env_r = env_r - np.mean(env_r)
    elif cfg.detrend == 'linear':
        from scipy.signal import detrend as sp_detrend
        env_r = sp_detrend(env_r, type='linear')

    spec = np.fft.rfft(env_r * window)
    amp  = np.abs(spec) * 2.0 / np.sum(window)
    if amp.size:
        amp[0] *= 0.5
    return amp


# ═══════════════════════════════════════════════════════════════════════════════
#  ACCUMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

class Accumulator:
    """
    Running average of per-axis envelope spectra across actuations.

    Mirrors ``EP_order_analysis.Accumulator`` so the envelope and order
    deliverables are combined the same, energy-correct way:
      • ``average``        — 'power' = sqrt(mean|A|²) RMS/energy-mean (default);
                             'magnitude' = legacy arithmetic mean(|A|).
      • ``cumulative_mode``— 'rss' = sqrt(x²+y²+z²) vector magnitude (default);
                             'sum' = legacy linear |x|+|y|+|z| (over-counts,
                             axis-phase dependent).
    means() divides by actuation count — normalizes for unequal trial counts.
    Default arguments keep the plain ``Accumulator(n)`` call sites working.
    """

    def __init__(self, n_orders: int, average: str = 'power',
                 cumulative_mode: str = 'rss'):
        self.sum     = {'x': np.zeros(n_orders),
                        'y': np.zeros(n_orders),
                        'z': np.zeros(n_orders)}
        self.count   = 0
        self.rev_sum = 0.0
        self.average = average
        self.cumulative_mode = cumulative_mode

    def add(self, amp_x, amp_y, amp_z, n_rev):
        if self.average == 'power':
            self.sum['x'] += np.asarray(amp_x) ** 2
            self.sum['y'] += np.asarray(amp_y) ** 2
            self.sum['z'] += np.asarray(amp_z) ** 2
        else:
            self.sum['x'] += amp_x
            self.sum['y'] += amp_y
            self.sum['z'] += amp_z
        self.count    += 1
        self.rev_sum  += n_rev

    def means(self) -> dict:
        if self.count == 0:
            return {}
        if self.average == 'power':
            mx = np.sqrt(self.sum['x'] / self.count)
            my = np.sqrt(self.sum['y'] / self.count)
            mz = np.sqrt(self.sum['z'] / self.count)
        else:
            mx = self.sum['x'] / self.count
            my = self.sum['y'] / self.count
            mz = self.sum['z'] / self.count
        if self.cumulative_mode == 'sum':
            cumulative = mx + my + mz
        else:
            cumulative = np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
        return {'x': mx, 'y': my, 'z': mz, 'cumulative': cumulative}


# ═══════════════════════════════════════════════════════════════════════════════
#  PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════

def save_mean_csv(common_orders, means: dict, path: str) -> None:
    pd.DataFrame({
        'order':      common_orders,
        'mean_x':     means['x'],
        'mean_y':     means['y'],
        'mean_z':     means['z'],
        'cumulative': means['cumulative'],
    }).to_csv(path, index=False)


def save_by_group_csv(common_orders, bucket_accs: dict, path: str,
                      max_points: int = 500) -> None:
    """Tidy long-format envelope spectrum for the interactive Review split.

    ``bucket_accs`` is keyed ``(dimension, value, material)`` where dimension is
    speed / sample / direction.  The order grid is decimated to ~``max_points``.
    Columns: order, cumulative, material, dimension, value.
    """
    n = len(common_orders)
    idx = (np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
           if n > max_points else np.arange(n))
    ords = np.asarray(common_orders)[idx]
    rows = []
    for (dim, val, mat), acc in sorted(bucket_accs.items(),
                                       key=lambda kv: str(kv[0])):
        m = acc.means()
        if not m:
            continue
        for o, c in zip(ords, m['cumulative'][idx]):
            rows.append({'order': float(o), 'cumulative': float(c),
                         'material': mat, 'dimension': dim, 'value': val})
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def _integer_order_lines(ax, max_order: float) -> None:
    for k in range(1, int(np.floor(max_order)) + 1):
        ax.axvline(k, color='grey', lw=0.5, linestyle=':', alpha=0.35, zorder=0)


def _ball_pass_order_lines(ax, cfg: Config,
                            max_order: float, y_top: float) -> None:
    bpo = cfg.ball_pass_order
    for h in range(1, cfg.n_bpf_harmonics + 1):
        order_val = bpo * h
        if order_val > max_order:
            break
        ax.axvline(
            order_val,
            color='#9b59b6', lw=1.0,
            linestyle='--', alpha=0.75, zorder=4,
        )
        label = (f'BPF×{h}\n{order_val:.2f}'
                 if h > 1 else f'BPF\n{order_val:.2f}')
        y_pos = (y_top ** 0.6) if cfg.log_y_axis else (y_top * 0.97)
        ax.text(
            order_val + 0.05, y_pos,
            label,
            fontsize=7, color='#9b59b6',
            va='top', ha='left', zorder=5,
        )


def plot_comparison(common_orders,
                    metal_means: dict,   n_metal: int,
                    plastic_means: dict, n_plastic: int,
                    bp_log: pd.DataFrame,
                    path: str, cfg: Config) -> None:
    """
    Overlay cumulative envelope order spectra for metal and plastic.
    Both curves are mean-per-actuation (normalized by actuation count).
    Subtitle shows median detected bandpass bands per material.
    A warning annotation is added when a material's median kurtosis is near
    the Gaussian floor (< 5.0), indicating the envelope may not be diagnostic.
    """
    # Summarize detected bands for subtitle
    def band_summary(material: str) -> str:
        sub = bp_log[bp_log['material'] == material]
        if sub.empty:
            return 'n/a'
        med_low = sub['f_low_hz'].median()
        med_hi  = sub['f_high_hz'].median()
        return f'{med_low:.0f}–{med_hi:.0f} Hz'

    def median_kurtosis(material: str) -> float:
        """Median kurtosis across accepted segments for this material."""
        if bp_log.empty or 'kurtosis' not in bp_log.columns:
            return float('nan')
        sub = bp_log[bp_log['material'] == material]
        return float(sub['kurtosis'].median()) if not sub.empty else float('nan')

    metal_band   = band_summary('metal')
    plastic_band = band_summary('plastic')
    metal_kurt   = median_kurtosis('metal')
    plastic_kurt = median_kurtosis('plastic')

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor('#f7f7f7')
    ax.set_facecolor('#f7f7f7')

    if cfg.mark_integer_orders:
        _integer_order_lines(ax, common_orders[-1])

    ax.plot(
        common_orders, metal_means['cumulative'],
        lw=1.3, color=cfg.color_metal,
        label=f'Metal  (n={n_metal} actuations  |  median band {metal_band})',
        zorder=3,
    )
    ax.plot(
        common_orders, plastic_means['cumulative'],
        lw=1.3, color=cfg.color_plastic,
        label=f'Plastic  (n={n_plastic} actuations  |  median band {plastic_band})',
        zorder=3,
    )

    if cfg.log_y_axis:
        ax.set_yscale('log')

    y_top = float(max(
        np.max(metal_means['cumulative']),
        np.max(plastic_means['cumulative']),
    ))
    _ball_pass_order_lines(ax, cfg, common_orders[-1], y_top)

    ax.set_xlabel('Order  (Events / Revolution)', fontsize=12)
    ylabel = ('Cumulative envelope amplitude per actuation  (X + Y + Z)'
              + ('  [log scale]' if cfg.log_y_axis else ''))
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(
        f'Envelope Order Spectrums: Metal vs Plastic',
        fontsize=12, fontweight='bold', pad=14,
    )
    ax.set_xlim(0, common_orders[-1])

    ax.legend(fontsize=10, framealpha=0.9, edgecolor='#cccccc',
              loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.25, which='both')

    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#cccccc')

    ax.tick_params(axis='both', which='both', length=0)

    # ── Impulsiveness warnings ────────────────────────────────────────────────
    # Gaussian noise has kurtosis ≈ 3.  If a material's median accepted-segment
    # kurtosis is < 5.0 (barely above Gaussian), its envelope spectrum reflects
    # broadband noise shaped by the filter — not defect-induced modulation.
    # The threshold 5.0 is heuristic; anything < 2 above the Gaussian floor is
    # considered weakly impulsive for a bearing/recirculation signature.
    _KURT_WARN_THRESHOLD = 5.0
    warn_lines = []
    for mat, kurt, color in (('Metal',   metal_kurt,   cfg.color_metal),
                              ('Plastic', plastic_kurt, cfg.color_plastic)):
        if np.isfinite(kurt) and kurt < _KURT_WARN_THRESHOLD:
            warn_lines.append(
                (f'⚠ {mat}: median kurtosis {kurt:.1f}  '
                 f'(≈ Gaussian floor {getattr(cfg, "min_kurtosis", 3.0):.0f}) '
                 f'— band not impulsive, envelope NOT diagnostic',
                 color)
            )

    if warn_lines:
        y_pos = 0.97
        for msg, color in warn_lines:
            ax.text(
                0.01, y_pos, msg,
                transform=ax.transAxes,
                fontsize=8.5, color='#333333',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.35',
                          facecolor='#fff3cd', edgecolor=color,
                          linewidth=1.4, alpha=0.92),
                zorder=6,
            )
            y_pos -= 0.09

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot  -> {os.path.abspath(path)}")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  SEGMENT WORKER  (called concurrently via ThreadPoolExecutor)
# ═══════════════════════════════════════════════════════════════════════════════

def _campbell_envelope_rows(seg, axis_cols, cfg, common_orders, fixed_band) -> list:
    """Per-window ``(rpm, cumulative-envelope-amplitude)`` over the full stroke.

    Demodulates each sliding angular window with the part's fixed band (or a band
    detected once on the full stroke when none is supplied), then takes the
    envelope order spectrum — building an envelope-order × measured-RPM map across
    the ramp sweep.
    """
    from nvh_pipeline import campbell

    if cfg.rpm_col not in seg.columns:
        return []
    phi  = phi_revolutions(seg, cfg)
    keep = (np.concatenate([[True], np.diff(phi) > 0]) if phi.size
            else np.zeros(0, dtype=bool))
    if not np.any(keep):
        return []
    phi  = phi[keep]
    rpm  = np.abs(seg[cfg.rpm_col].to_numpy(dtype=float))[keep]
    t    = seg[cfg.time_col].to_numpy(dtype=float)[keep]
    if len(t) < 4:
        return []
    fs   = estimate_fs(t)
    # Absent axes (continuous single-channel data) become zero channels.
    _zero = np.zeros(int(np.count_nonzero(keep)), dtype=float)
    sigs = {ax: (seg[axis_cols[ax]].to_numpy(dtype=float)[keep]
                 if axis_cols.get(ax) and axis_cols[ax] in seg.columns else _zero)
            for ax in ('x', 'y', 'z')}

    if fixed_band is not None:
        f_low, f_high = fixed_band[0], fixed_band[2]
    else:
        # Detect the resonance band on the first present (non-zero) axis.
        _det = next((sigs[a] for a in ('x', 'y', 'z')
                     if axis_cols.get(a) and axis_cols[a] in seg.columns), sigs['x'])
        sx = _det - np.mean(_det)
        f_low, _fc, f_high, _k = detect_band_kurtogram(sx, fs, cfg)

    max_order = float(common_orders[-1]) if len(common_orders) else cfg.samples_per_rev / 2
    c_orders = campbell.order_grid(max_order, getattr(cfg, 'campbell_order_step', 0.1))
    cum_mode = getattr(cfg, 'cumulative_mode', 'rss')

    # Continuous mode: keep under-sampled windows so the envelope Campbell still
    # renders (aliased above the raw Nyquist — flagged with a warning in Review).
    # Non-finite results are still dropped below.  Reciprocating is unchanged.
    _continuous = getattr(cfg, 'analysis_mode', 'reciprocating') == 'continuous'
    _win_min = 4 if _continuous else cfg.min_fft_points

    rows = []
    for i0, i1 in campbell.angular_windows(
            phi, getattr(cfg, 'campbell_window_rev', 1.5),
            getattr(cfg, 'campbell_hop_rev', 0.75),
            min_samples=_win_min):
        phi_w = phi[i0:i1] - phi[i0]
        n_rev_w = float(phi_w[-1]) if phi_w.size else 0.0
        M = int(np.floor(n_rev_w * cfg.samples_per_rev))
        if M < (_win_min if _continuous else cfg.min_fft_points):
            continue
        phi_uniform = np.arange(M) / cfg.samples_per_rev
        orders_fft  = np.fft.rfftfreq(M, d=1.0 / cfg.samples_per_rev)
        win         = get_window(cfg.window, M, fftbins=True)
        per_axis = []
        for ax in ('x', 'y', 'z'):
            env = extract_envelope(sigs[ax][i0:i1], fs, f_low, f_high, cfg)
            amp = envelope_order_spectrum(env, phi_w, M, win, phi_uniform, cfg)
            per_axis.append(np.interp(c_orders, orders_fft, amp,
                                      left=0.0, right=0.0))
        # Combine axes the same energy-correct way as the mean envelope spectrum.
        if cum_mode == 'sum':
            cum = per_axis[0] + per_axis[1] + per_axis[2]
        else:
            cum = np.sqrt(per_axis[0] ** 2 + per_axis[1] ** 2 + per_axis[2] ** 2)
        if np.all(np.isfinite(cum)):
            rows.append((float(np.mean(rpm[i0:i1])), cum))
    return rows


def _compute_envelope_segment(task):
    """Compute envelope order spectrum for one segment. Returns ('ok', result) or ('skip', reason)."""
    # Accept legacy 10-element tasks (no fixed_band → per-segment detection).
    task = tuple(task)
    if len(task) == 10:
        task = task + (None,)
    (seg, path_basename, seg_id, material, rpm_cat, part, direction,
     axis_cols, common_orders, cfg, fixed_band) = task

    # Campbell (envelope-order × measured-RPM) over the full stroke incl. ramps —
    # compute before plateau gating restricts ``seg``.
    campbell_rows = (_campbell_envelope_rows(seg, axis_cols, cfg, common_orders,
                                             fixed_band)
                     if getattr(cfg, 'campbell', False) else [])

    # Steady-state gating: demodulate only the constant-speed plateau (fallback to
    # the full stroke if the plateau is too short).
    used_plateau = False
    if getattr(cfg, 'plateau_gating', False) and cfg.rpm_col in seg.columns:
        from nvh_pipeline.common import plateau_mask
        pmask = plateau_mask(seg[cfg.rpm_col].to_numpy(dtype=float),
                             rpm_cat, getattr(cfg, 'plateau_frac', 0.90))
        if int(pmask.sum()) >= cfg.min_fft_points:
            seg = seg.iloc[pmask]
            used_plateau = True

    phi  = phi_revolutions(seg, cfg)
    keep = (np.concatenate([[True], np.diff(phi) > 0]) if phi.size
            else np.zeros(0, dtype=bool))
    phi  = phi[keep]
    n_rev = float(phi[-1]) if phi.size else 0.0

    if n_rev < cfg.min_revolutions:
        return ('skip', 'min_revolutions')

    M = int(np.floor(n_rev * cfg.samples_per_rev))
    if M < cfg.min_fft_points:
        return ('skip', 'min_fft_points')

    phi_uniform = np.arange(M) / cfg.samples_per_rev
    orders_fft  = np.fft.rfftfreq(M, d=1.0 / cfg.samples_per_rev)
    win         = get_window(cfg.window, M, fftbins=True)

    t  = seg[cfg.time_col].to_numpy(dtype=float)
    fs = estimate_fs(t)

    # Anti-alias cutoff for envelope→order resampling (R1).
    # Use the 5th-percentile of non-zero RPM in the kept portion so brief
    # ramp phases set the floor for the angular Nyquist in Hz.
    _env_lp_hz = None
    if cfg.rpm_col in seg.columns:
        _rpm_kept = np.abs(seg[cfg.rpm_col].to_numpy(dtype=float))[keep]
        _moving   = _rpm_kept[_rpm_kept > 5.0]
        if _moving.size:
            _rpm_min   = float(np.percentile(_moving, 5))
            _env_lp_hz = max((cfg.samples_per_rev / 2) * _rpm_min / 60.0, 10.0)

    # Envelope band-search floor. The ballscrew formula clears the rig's order-25
    # shaft harmonic, but that is meaningless for a variable-speed continuous
    # signal (and a single nominal RPM there can push the floor above Nyquist and
    # collapse the band). Continuous mode uses only the absolute floor.
    if getattr(cfg, 'analysis_mode', 'reciprocating') == 'continuous':
        _dyn_min_hz = getattr(cfg, 'bp_min_hz_abs', 50.0)
    else:
        _rpm_for_min = rpm_cat if rpm_cat is not None else 2300
        _dyn_min_hz  = max(
            getattr(cfg, 'bp_min_hz_abs', 50.0),
            25 * _rpm_for_min / 60 * 1.2,
        )

    amps     = {}
    bp_rows  = []
    max_kurt = -np.inf
    for ax_key in ('x', 'y', 'z'):
        col = axis_cols.get(ax_key)
        if col is None or col not in seg.columns:
            # Absent axis (continuous single-channel data): zero contribution.
            amps[ax_key] = np.zeros_like(common_orders, dtype=float)
            continue
        sig = seg[col].to_numpy(dtype=float)[keep]
        sig = sig - np.mean(sig)

        if fixed_band is not None:
            # Per-part fixed band: one resonance band for this specimen, applied
            # to every segment/axis.  Score its impulsiveness on this segment so
            # the low-kurtosis gate still applies.
            f_low, f_ctr, f_high = fixed_band[0], fixed_band[1], fixed_band[2]
            seg_kurt = band_kurtosis(sig, fs, f_low, f_high)
        else:
            f_low, f_ctr, f_high, seg_kurt = detect_band_kurtogram(
                sig, fs, cfg, min_hz=_dyn_min_hz)
        max_kurt = max(max_kurt, seg_kurt)

        bp_rows.append({
            'source_file': path_basename,
            'segment_id':  int(seg_id),
            'material':    material,
            'rpm_cat':     rpm_cat,
            'axis':        ax_key,
            'band_scope':  'per_part' if fixed_band is not None else 'per_segment',
            'f_low_hz':    round(f_low,    1),
            'f_center_hz': round(f_ctr,    1),
            'f_high_hz':   round(f_high,   1),
            'fs_hz':       round(fs,       1),
            'kurtosis':    round(seg_kurt, 2),
        })

        envelope = extract_envelope(sig, fs, f_low, f_high, cfg)
        amp      = envelope_order_spectrum(
            envelope, phi, M, win, phi_uniform, cfg,
            lp_hz=_env_lp_hz, fs=fs)
        amp_i = np.interp(common_orders, orders_fft, amp,
                          left=0.0, right=0.0)

        # Defense-in-depth: a non-finite spectrum (should never happen with the
        # FFT envelope) must not be allowed to poison the per-actuation mean.
        if not np.all(np.isfinite(amp_i)):
            return ('skip', 'non_finite')

        amps[ax_key] = amp_i

    # Skip segment if all axes are below the minimum impulsiveness threshold.
    min_k = getattr(cfg, 'min_kurtosis', 0.0)
    if min_k > 0 and max_kurt < min_k:
        return ('skip', 'low_kurtosis')

    # Per-segment amplitude normalization: divide all three axes by the segment's
    # combined RMS so the mean spectrum reflects envelope *shape* (impulsiveness)
    # rather than absolute actuation loudness — one loud trial no longer dominates.
    if getattr(cfg, 'normalize_segments', True):
        scale = np.sqrt(sum(float(np.mean(a * a)) for a in amps.values()))
        if scale > 1e-30:
            for k in amps:
                amps[k] = amps[k] / scale

    return ('ok', {
        'material':  material,
        'amps':      amps,
        'n_rev':     n_rev,
        'bp_rows':   bp_rows,
        'part':      part,
        'rpm_cat':   rpm_cat,
        'direction': direction,
        'used_plateau': used_plateau,
        'campbell':  campbell_rows,
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-SPECIMEN BAND SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _select_part_bands(data_dir: str, cfg: Config, axis_cols: dict,
                       max_segments_per_part: int = 8) -> dict:
    """Choose ONE resonance band per specimen (part).

    A structural resonance is a property of the rig/specimen, not the speed, so
    re-detecting it per segment lets the kurtogram wander (and latch onto noise).
    This picks the band once per part, from a bounded number of that part's
    **highest-RPM plateau** segments (best impulse SNR), and the band is then used
    to demodulate every segment of that part — making the by-speed envelope
    spectra directly comparable.

    Returns ``{part: {f_low, f_ctr, f_high, kurtosis, fs, rpm_used, n_seg}}``.
    """
    from nvh_pipeline import dataset
    from nvh_pipeline.common import plateau_mask

    cat = dataset.segment_catalog(data_dir)
    if cat is None or cat.empty:
        return {}
    if cfg.only_rpms:
        cat = cat[cat['rpm_category'].isin([int(r) for r in cfg.only_rpms])]
    if cat.empty:
        return {}

    _axis_present = [axis_cols[a] for a in ('x', 'y', 'z')
                     if axis_cols.get(a)]
    needed = _axis_present + [cfg.rpm_col, cfg.time_col]
    bands: dict = {}
    for part, g in cat.groupby('part'):
        rpm_used = int(g['rpm_category'].max())          # highest RPM = best SNR
        sel = g[g['rpm_category'] == rpm_used].head(max_segments_per_part)
        # Continuous mode: absolute floor only (see worker note above).
        if getattr(cfg, 'analysis_mode', 'reciprocating') == 'continuous':
            dyn_min = getattr(cfg, 'bp_min_hz_abs', 50.0)
        else:
            dyn_min = max(getattr(cfg, 'bp_min_hz_abs', 50.0),
                          25 * rpm_used / 60 * 1.2)
        best = None        # (kurt, f_low, f_ctr, f_high, fs, axis, grid)
        for _, row in sel.iterrows():
            try:
                seg = dataset.read_file(
                    data_dir, row['source_file'], columns=needed,
                    where=f"segment_id = {int(row['segment_id'])}")
            except Exception:
                continue
            if seg.empty:
                continue
            if getattr(cfg, 'plateau_gating', False) and cfg.rpm_col in seg.columns:
                pmask = plateau_mask(seg[cfg.rpm_col].to_numpy(dtype=float),
                                     rpm_used, getattr(cfg, 'plateau_frac', 0.90))
                if int(pmask.sum()) >= cfg.min_fft_points:
                    seg = seg.iloc[pmask]
            t = seg[cfg.time_col].to_numpy(dtype=float)
            if len(t) < 4:
                continue
            fs = estimate_fs(t)
            for ax in ('x', 'y', 'z'):
                col = axis_cols.get(ax)
                if col is None or col not in seg.columns:
                    continue
                sig = seg[col].to_numpy(dtype=float)
                sig = sig - np.mean(sig)
                fl, fc, fh, k, grid = detect_band_kurtogram(
                    sig, fs, cfg, min_hz=dyn_min, return_grid=True)
                if best is None or k > best[0]:
                    best = (k, fl, fc, fh, fs, ax, grid)
        if best is not None:
            bands[part] = {
                'f_low':    round(best[1], 1),
                'f_ctr':    round(best[2], 1),
                'f_high':   round(best[3], 1),
                'kurtosis': round(best[0], 2),
                'fs':       round(best[4], 1),
                'rpm_used': rpm_used,
                'n_seg':    int(len(sel)),
                'axis':     best[5],          # winning axis (for the kurtogram)
                'grid':     best[6],          # kurtosis grid of the winning eval
            }
    return bands


def _write_band_diagnostics(cfg: Config, part_bands: dict) -> None:
    """Emit the per-specimen band table and the representative kurtogram grid as
    CSVs (consumed by the Streamlit Envelope tab)."""
    from nvh_pipeline.common import material_type

    band_rows = []
    grid_rows = []
    for part, b in sorted(part_bands.items()):
        mat = material_type(part)
        band_rows.append({
            'part':        part,
            'material':    mat,
            'f_low_hz':    b['f_low'],
            'f_center_hz': b['f_ctr'],
            'f_high_hz':   b['f_high'],
            'kurtosis':    b['kurtosis'],
            'fs_hz':       b['fs'],
            'rpm_used':    b['rpm_used'],
            'axis':        b.get('axis'),
            'n_seg':       b['n_seg'],
        })
        for cell in b.get('grid', []):
            selected = (abs(cell['f_low_hz'] - b['f_low']) < 0.5 and
                        abs(cell['f_high_hz'] - b['f_high']) < 0.5)
            grid_rows.append({
                'part':     part,
                'material': mat,
                'axis':     b.get('axis'),
                'rpm_used': b['rpm_used'],
                **cell,
                'selected': selected,
            })

    if band_rows:
        pd.DataFrame(band_rows).to_csv(
            os.path.join(cfg.output_dir, 'part_bands.csv'), index=False)
    if grid_rows:
        pd.DataFrame(grid_rows).to_csv(
            os.path.join(cfg.output_dir, 'kurtogram_grid.csv'), index=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def run(cfg: Config) -> dict:
    data_dir  = os.path.join(cfg.seg_output_dir, cfg.data_subdir)
    plots_dir = os.path.join(cfg.output_dir, cfg.plots_subdir)
    os.makedirs(plots_dir, exist_ok=True)

    from nvh_pipeline import dataset
    files = dataset.list_segment_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"no '{cfg.seg_glob}' files in {os.path.abspath(data_dir)}"
        )

    max_order = (cfg.max_order if cfg.max_order is not None
                 else cfg.samples_per_rev / 2)
    max_order = min(max_order, cfg.samples_per_rev / 2)
    # Start from order_resolution (not 0) to exclude the DC bin, which always
    # has a residual even after mean subtraction and produces a misleading spike
    # at the left edge of the envelope spectrum.
    common_orders = np.arange(cfg.order_resolution,
                               max_order + cfg.order_resolution,
                               cfg.order_resolution)

    print(f"\nInput:            {os.path.abspath(data_dir)}  ({len(files)} files)")
    print(f"Output:           {os.path.abspath(cfg.output_dir)}")
    print(f"RPM filter:       {cfg.only_rpms}")
    print(f"Kurtogram levels: {cfg.kurtogram_levels}  "
          f"(min BW = {cfg.bp_min_bw_hz:.0f} Hz,  floor = {cfg.bp_min_hz:.0f} Hz)")
    print(f"samples/rev:      {cfg.samples_per_rev}  "
          f"(Nyquist order = {cfg.samples_per_rev/2:g})")
    print(f"order grid:       0 .. {max_order:g}  step {cfg.order_resolution:g}")
    print(f"ball pass order:  {cfg.ball_pass_order}  "
          f"(harmonics: "
          f"{', '.join(f'{cfg.ball_pass_order*h:.2f}' for h in range(1, cfg.n_bpf_harmonics+1))})\n")

    # Resolve axis columns once, then (per_part scope) choose one resonance band
    # per specimen up-front so every segment of a part demodulates the same band.
    axis_cols = resolve_axis_columns(
        dataset.file_columns(data_dir, files[0]), cfg)
    print(f"Axis channels:    X={axis_cols['x']!r}  "
          f"Y={axis_cols['y']!r}  Z={axis_cols['z']!r}\n")

    part_bands: dict = {}
    if getattr(cfg, 'band_scope', 'per_part') == 'per_part':
        part_bands = _select_part_bands(data_dir, cfg, axis_cols)
        if part_bands:
            print("Per-specimen resonance bands (fixed across speed, "
                  f"capped at {getattr(cfg, 'bp_max_hz', float('inf')):.0f} Hz):")
            for prt, b in sorted(part_bands.items()):
                print(f"  {prt:<6} {b['f_low']:.0f}-{b['f_high']:.0f} Hz "
                      f"(ctr {b['f_ctr']:.0f}, kurt {b['kurtosis']:.1f}, "
                      f"from {b['rpm_used']} RPM)")
            print()
            _write_band_diagnostics(cfg, part_bands)

    # Axis-aggregation modes (parity with EP_order_analysis) — energy-mean per
    # axis and RSS cumulative by default.
    _avg = getattr(cfg, 'order_average', 'power')
    _cum = getattr(cfg, 'cumulative_mode', 'rss')

    def _new_acc() -> Accumulator:
        return Accumulator(len(common_orders), average=_avg, cumulative_mode=_cum)

    # Keyed by material. Reciprocating data is metal/plastic; continuous data is
    # 'unknown' (a single non-comparative group). A defaultdict lets any material
    # accumulate without a KeyError, while metal/plastic stay pre-seeded so the
    # existing comparison outputs are unchanged.
    accs: dict[str, Accumulator] = defaultdict(_new_acc)
    accs['metal']   = _new_acc()
    accs['plastic'] = _new_acc()
    from nvh_pipeline import campbell as _campbell
    campbell_grid = _campbell.CampbellGrid(
        _campbell.order_grid(float(common_orders[-1]),
                             getattr(cfg, 'campbell_order_step', 0.1)),
        getattr(cfg, 'campbell_rpm_bin', 50.0))
    # Finer-grained accumulators for the interactive Review split, keyed by
    # (dimension, value, material) where dimension is speed / sample / direction.
    bucket_accs: dict[tuple, Accumulator] = {}

    skips = {
        'rpm_filtered':    0,
        'min_revolutions': 0,
        'min_fft_points':  0,
        'unknown_material': 0,
        'low_kurtosis':    0,
        'non_finite':      0,
    }

    # Log of detected bands — one row per (segment, axis)
    bp_log_rows: list[dict] = []

    n_processed = 0
    n_plateau_fallback = 0

    # ── Reduce one segment's result into the (additive) accumulators ──────────
    # Accumulators are order-independent, so reducing file-by-file gives totals
    # identical to reducing one global result list.
    def _reduce(res) -> None:
        nonlocal n_processed, n_plateau_fallback
        status, data = res
        if status == 'skip':
            skips[data] += 1
            return

        material = data['material']
        amps     = data['amps']
        n_rev    = data['n_rev']

        if not data.get('used_plateau', True):
            n_plateau_fallback += 1

        accs[material].add(amps['x'], amps['y'], amps['z'], n_rev)
        bp_log_rows.extend(data['bp_rows'])

        for rpm_w, amp_w in data.get('campbell', []):
            campbell_grid.add(material, rpm_w, amp_w)

        for dim, val in (('speed',     data['rpm_cat']),
                         ('sample',    data['part']),
                         ('direction', data['direction'])):
            if val is None:
                continue
            bkey = (dim, str(val), material)
            if bkey not in bucket_accs:
                bucket_accs[bkey] = _new_acc()
            bucket_accs[bkey].add(amps['x'], amps['y'], amps['z'], n_rev)

        n_processed += 1

    # ── Stream segments one file at a time (memory-bounded) ───────────────────
    # Each file's segments are collected, processed, reduced, then released —
    # peak memory stays near one file's segments instead of every file's at once.
    for path in files:
        if 'segment_id' not in dataset.file_columns(data_dir, path):
            continue

        tasks: list = []
        for seg_id, seg in dataset.iter_file_segments(data_dir, path):
            if cfg.only_rpms and 'rpm_category' in seg.columns:
                if int(seg['rpm_category'].iloc[0]) not in cfg.only_rpms:
                    skips['rpm_filtered'] += 1
                    continue

            material = (seg['material_type'].iloc[0]
                        if 'material_type' in seg.columns else None)
            # Reciprocating rig segments are metal or plastic; continuous mode
            # carries material 'unknown', which is a valid (non-comparative) run.
            if (cfg.analysis_mode != 'continuous'
                    and material not in ('metal', 'plastic')):
                skips['unknown_material'] += 1
                continue

            rpm_cat   = (int(seg['rpm_category'].iloc[0])
                         if 'rpm_category' in seg.columns else None)
            part      = (seg['part'].iloc[0]
                         if 'part' in seg.columns else None)
            direction = (seg['direction'].iloc[0]
                         if 'direction' in seg.columns else None)

            fb = part_bands.get(part)
            fixed_band = ((fb['f_low'], fb['f_ctr'], fb['f_high'])
                          if fb else None)
            tasks.append((seg, os.path.basename(path), seg_id,
                          material, rpm_cat, part, direction,
                          axis_cols, common_orders, cfg, fixed_band))

        if not tasks:
            continue

        # Process this file's segments in parallel (numpy/scipy release the GIL).
        n_workers = min(os.cpu_count() or 4, len(tasks))
        print(f"Processing {len(tasks)} segments from "
              f"{os.path.basename(path)} across {n_workers} threads ...")
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for res in pool.map(_compute_envelope_segment, tasks):
                _reduce(res)

    # ── Results ──────────────────────────────────────────────────────────────
    n_metal   = accs['metal'].count
    n_plastic = accs['plastic'].count
    n_total   = sum(a.count for a in accs.values())

    print(f"\nDone.  Total segments processed: {n_processed}")
    print(f"  Metal:    {n_metal}")
    print(f"  Plastic:  {n_plastic}")
    print(f"  Skips:    {skips}")

    if n_total == 0:
        raise RuntimeError(
            f"No valid segments processed. Skips: {skips}\n"
            f"Check only_rpms, min_revolutions, or angle column."
        )

    metal_means   = accs['metal'].means()
    plastic_means = accs['plastic'].means()

    # ── Save CSVs ────────────────────────────────────────────────────────────
    if metal_means:
        save_mean_csv(common_orders, metal_means,
                      os.path.join(cfg.output_dir,
                                   'envelope_spectra_mean_metal.csv'))
    if plastic_means:
        save_mean_csv(common_orders, plastic_means,
                      os.path.join(cfg.output_dir,
                                   'envelope_spectra_mean_plastic.csv'))
    # Continuous / non-comparative materials (e.g. 'unknown'): one mean spectrum.
    for _mat, _acc in accs.items():
        if _mat in ('metal', 'plastic'):
            continue
        _mm = _acc.means()
        if _mm:
            save_mean_csv(common_orders, _mm,
                          os.path.join(cfg.output_dir,
                                       'envelope_spectra_mean.csv'))

    # Tidy by-group spectra for the interactive Review split.
    save_by_group_csv(common_orders, bucket_accs,
                      os.path.join(cfg.output_dir,
                                   'envelope_spectra_by_group.csv'))

    bp_log = pd.DataFrame(bp_log_rows)
    bp_log.to_csv(
        os.path.join(cfg.output_dir, 'bandpass_log.csv'), index=False
    )

    if getattr(cfg, 'campbell', False):
        cdf = campbell_grid.to_long_df()
        if not cdf.empty:
            cdf.to_csv(os.path.join(cfg.output_dir, 'envelope_campbell.csv'),
                       index=False)
            print(f"Campbell map:     {cdf['rpm'].nunique()} RPM bins x "
                  f"{cdf['order'].nunique()} orders -> envelope_campbell.csv")

    # Console summary of detected bands
    if not bp_log.empty:
        print("\nDetected bandpass summary (median across all segments):")
        summary = (
            bp_log.groupby(['material', 'axis'])[['f_low_hz', 'f_center_hz', 'f_high_hz']]
            .median()
            .round(0)
        )
        print(summary.to_string())

    # ── Plot ─────────────────────────────────────────────────────────────────
    if metal_means and plastic_means:
        if cfg.render_plots:
            plot_comparison(
                common_orders,
                metal_means,   n_metal,
                plastic_means, n_plastic,
                bp_log,
                path = os.path.join(plots_dir,
                                    'envelope_cumulative_comparison.png'),
                cfg  = cfg,
            )
    else:
        missing = 'metal' if not metal_means else 'plastic'
        print(f"WARNING: no valid segments for {missing} — "
              f"comparison plot skipped.")

    # ── Metadata ─────────────────────────────────────────────────────────────
    bp_summary = {}
    if not bp_log.empty:
        for mat in ('metal', 'plastic'):
            sub = bp_log[bp_log['material'] == mat]
            if not sub.empty:
                bp_summary[mat] = {
                    'median_f_low_hz':    round(float(sub['f_low_hz'].median()),  1),
                    'median_f_center_hz': round(float(sub['f_center_hz'].median()), 1),
                    'median_f_high_hz':   round(float(sub['f_high_hz'].median()),  1),
                    'n_bands_detected':   len(sub),
                }

    band_scope = getattr(cfg, 'band_scope', 'per_part')
    run_meta = {
        'only_rpms':         list(cfg.only_rpms) if cfg.only_rpms else None,
        'n_metal':           n_metal,
        'n_plastic':         n_plastic,
        'n_processed':       n_processed,
        'n_plateau_fallback': n_plateau_fallback,
        'skips':             skips,
        'band_scope':        band_scope,
        'bandpass_method':   ('per-specimen kurtogram (fixed across speed)'
                              if band_scope == 'per_part'
                              else 'per-actuation-per-axis kurtogram'),
        'part_bands':        {p: {k: v for k, v in b.items() if k != 'grid'}
                              for p, b in part_bands.items()},
        'bp_max_hz':         getattr(cfg, 'bp_max_hz', None),
        'plateau_gating':    getattr(cfg, 'plateau_gating', None),
        'plateau_frac':      getattr(cfg, 'plateau_frac', None),
        'bandpass_summary':  bp_summary,
        'ball_pass_order':   cfg.ball_pass_order,
        'bpf_harmonics':     [cfg.ball_pass_order * h
                              for h in range(1, cfg.n_bpf_harmonics + 1)],
        'common_order_grid': {
            'max_order':  float(max_order),
            'resolution': cfg.order_resolution,
            'n_bins':     len(common_orders),
        },
        'config': {k: getattr(cfg, k) for k in (
            'samples_per_rev', 'detrend', 'window', 'min_revolutions',
            'angle_source', 'bp_min_hz', 'bp_min_bw_hz',
            'kurtogram_levels', 'bp_filter_order')},
    }
    with open(os.path.join(cfg.output_dir, 'run_metadata.json'), 'w') as f:
        json.dump(run_meta, f, indent=2, default=str)

    print(f"\nOutput -> {os.path.abspath(cfg.output_dir)}\n")
    return {
        'common_orders': common_orders,
        'metal_means':   metal_means,
        'plastic_means': plastic_means,
        'meta':          run_meta,
    }


def _apply_pipeline_config(cfg, _stage='envelope'):
    """Overlay shared nvh_config.json settings onto this stage's Config.

    A genuinely-absent ``nvh_pipeline`` package (a bare standalone run) is the
    ONLY reason to fall back to the built-in Config defaults — that is an
    ImportError and is silently tolerated.  Any OTHER error (a malformed
    nvh_config.json, an unknown column name, a type error) is a real
    misconfiguration and is re-raised rather than silently reverting to defaults
    and processing the wrong folder / columns without warning.
    """
    try:
        from nvh_pipeline.config import apply_to_stage
    except ImportError:
        return
    apply_to_stage(_stage, cfg)


def main():
    cfg = Config()
    _apply_pipeline_config(cfg)
    run(cfg)


if __name__ == '__main__':
    main()