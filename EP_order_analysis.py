"""
═══════════════════════════════════════════════════════════════════════════════
 ORDER ANALYSIS  —  per-actuation order spectra + averaged axis spectra
═══════════════════════════════════════════════════════════════════════════════

Companion to the segmentation script.  Consumes the per-file *_segments.parquet
files it produced and computes an ORDER SPECTRUM for every actuation, then
averages those spectra per axis and sums the axis means into a cumulative
order spectrum.

Cumulative spectra (X+Y+Z) are computed separately for metal and plastic and
overlaid on a single comparison plot.  Summing axes before comparing means
axis-labelling inconsistencies between test runs cannot bias the result.

NORMALIZATION
-------------
  Each axis accumulator divides by the number of actuations before producing
  a mean spectrum, so metal and plastic are always compared on a per-actuation
  basis regardless of how many segments each material contributed.

BALL PASS ORDER
---------------
  Ball pass frequency order (BPFO) = 5.35 and its first three harmonics
  (10.70, 16.05, 21.40) are marked on the comparison plot.

WHY ORDER ANALYSIS (and not a plain FFT)
----------------------------------------
  During an actuation the shaft RPM ramps up, plateaus, and ramps down — the
  speed is NOT constant.  A plain time-domain FFT would smear every speed-
  dependent feature across many Hz bins.  Order analysis fixes this by
  resampling the vibration signal at equal *angular* increments (equal shaft
  rotation, not equal time) before the FFT.  The resulting x-axis is "orders"
  = cycles per shaft revolution.  Order 1 is once-per-rev, order 2 is twice-
  per-rev, etc.  These features are speed-independent, which is exactly what
  lets us average actuations recorded at different RPMs.

ALGORITHM (per actuation)
-------------------------
  1. Recover the shaft angle traversed during the actuation.
       - preferred: the encoder angle column (CNT 1/Angle), referenced to the
         start of the actuation and made monotonic.
       - fallback:  integrate |RPM| over time to get revolutions.
  2. Resample each axis signal onto a uniform angle grid of `samples_per_rev`
     points per revolution (interpolation).
  3. Remove DC (or linear trend), apply a window, take the real FFT.
       - order axis  = rfftfreq(M, d = 1 / samples_per_rev)   [cycles / rev]
       - order resolution = 1 / (revolutions traversed)
       - max order (Nyquist) = samples_per_rev / 2
  4. Convert to an amplitude-correct single-sided magnitude spectrum.
  5. Interpolate onto a COMMON order grid so spectra from actuations of
     different length / RPM can be averaged together.

OUTPUTS
-------
  output_order/
    order_spectra_mean.csv               global mean (all actuations)
    order_spectra_mean_metal.csv         metal only
    order_spectra_mean_plastic.csv       plastic only
    per_actuation_peaks.csv              one row per actuation
    run_metadata.json                    config + counts + skip reasons
    plots/
      order_mean_axes.png                global mean X / Y / Z overlaid
      order_cumulative.png               global cumulative
      order_cumulative_comparison.png    metal vs plastic cumulative overlay

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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend as sp_detrend, get_window


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── IO ──────────────────────────────────────────────────────────────────
    seg_output_dir: str = 'output_seg'
    data_subdir: str    = 'segmented_data'
    seg_glob: str       = '*_segments.parquet'

    output_dir: str   = 'output_order'
    plots_subdir: str = 'plots'

    # ── COLUMNS ─────────────────────────────────────────────────────────────
    time_col: str  = 'Time (s)'
    rpm_col: str   = 'CNT 1/Frequency (RPM)'
    angle_col: str = 'CNT 1/Angle (Degrees)'

    axis_x: Optional[str] = None
    axis_y: Optional[str] = None
    axis_z: Optional[str] = None

    # ── ANGLE SOURCE ────────────────────────────────────────────────────────
    angle_source: str = 'encoder'

    # ── ORDER ANALYSIS ──────────────────────────────────────────────────────
    samples_per_rev: int       = 128
    detrend: str               = 'mean'
    window: str                = 'hann'
    max_order: Optional[float] = 25.0
    order_resolution: float    = 0.02
    # 10 revolutions gives order resolution of 0.1 — enough to cleanly resolve
    # BPFO at 5.35 away from integer orders.  1.0 (old default) gave resolution
    # of 1.0 order which cannot reliably separate 5.35 from shaft orders (R2).
    min_revolutions: float     = 10.0
    min_fft_points: int        = 16

    # ── BALL PASS ORDER ──────────────────────────────────────────────────────
    # Fundamental ball pass frequency order and number of harmonics to mark.
    ball_pass_order: float  = 5.35
    n_bpf_harmonics: int    = 4      # marks 5.35, 10.70, 16.05, 21.40

    # Half-width (in orders) of the integration band around each BPFO harmonic
    # used when computing per-actuation band energy (R2).  ±0.25 order captures
    # the main lobe of a 10-revolution segment (resolution 0.1 ord) without
    # pulling in adjacent shaft orders.
    bpfo_band_halfwidth: float = 0.25

    # ── FILTERING ───────────────────────────────────────────────────────────
    only_parts: Optional[tuple]   = None
    only_rpms: Optional[tuple]    = None
    only_direction: Optional[str] = None

    # 'reciprocating' (ballscrew rig) or 'continuous' (single steady signal —
    # allows a single vibration channel; missing axes become zero channels).
    analysis_mode: str     = 'reciprocating'

    # ── STEADY-STATE GATING / AGGREGATION (injected from PipelineConfig) ──────
    plateau_gating: bool   = True   # analyse only the constant-speed plateau
    plateau_frac: float    = 0.90   # plateau = |RPM| >= frac * rpm_category
    order_average: str     = 'power'   # 'power' = sqrt(mean|A|²); 'magnitude'
    cumulative_mode: str   = 'rss'     # 'rss' = sqrt(x²+y²+z²); 'sum'

    # ── CAMPBELL (order × measured-RPM waterfall from the ramp sweep) ─────────
    campbell: bool            = True
    campbell_window_rev: float = 1.5    # window length (rev) — order res ≈ 1/this
    campbell_hop_rev: float    = 0.75   # window advance (rev)
    campbell_rpm_bin: float    = 50.0   # RPM-axis bin width
    campbell_order_step: float = 0.1    # order-axis bin width for the map

    # group_by is locked to material_type for the comparison plot.
    group_by: tuple = ('material_type',)

    # ── PLOTTING ────────────────────────────────────────────────────────────
    show_plots: bool          = False
    render_plots: bool        = True
    mark_integer_orders: bool = True
    n_top_peaks: int          = 8

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

    # Continuous mode may carry only one or two vibration channels (e.g. a single
    # accelerometer). Then a missing axis is allowed — it resolves to None and is
    # treated as a zero channel downstream (so the RSS cumulative equals the real
    # channel and the per-axis traces for the absent axes read zero).
    allow_missing = getattr(cfg, 'analysis_mode', 'reciprocating') == 'continuous'

    resolved: dict = {}
    for axis in ('x', 'y', 'z'):
        if explicit[axis] is not None:
            if explicit[axis] not in columns:
                raise KeyError(
                    f"configured axis_{axis} = {explicit[axis]!r} not found. "
                    f"Available columns: {columns}"
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
                f"could not auto-detect the {axis.upper()} axis column. "
                f"Set Config.axis_{axis} explicitly. "
                f"Candidate columns were: {candidates}"
            )
        resolved[axis] = pick_from[0]

    present = [v for v in resolved.values() if v is not None]
    if not present:
        raise KeyError(
            f"no vibration axis columns found. Set axis_x / axis_y / axis_z "
            f"explicitly. Columns: {columns}"
        )
    if len(set(present)) != len(present):
        raise KeyError(
            f"axis auto-detect produced duplicates {resolved}. "
            f"Set axis_x / axis_y / axis_z explicitly. Columns: {columns}"
        )
    return resolved


# ═══════════════════════════════════════════════════════════════════════════════
#  ANGULAR DOMAIN
# ═══════════════════════════════════════════════════════════════════════════════

def phi_revolutions(seg_df: pd.DataFrame, cfg: Config) -> np.ndarray:
    use_encoder = cfg.angle_source == 'encoder' and cfg.angle_col in seg_df.columns
    if use_encoder:
        ang = seg_df[cfg.angle_col].to_numpy(dtype=float)
        phi = np.abs(ang - ang[0]) / 360.0
        # R9: warn when maximum.accumulate clamps a large fraction of steps,
        # which indicates encoder wrapping, glitches, or genuine reversals that
        # the monotonic enforcement is silently masking.
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
        t = seg_df[cfg.time_col].to_numpy(dtype=float)
        rev_per_s = np.abs(seg_df[cfg.rpm_col].to_numpy(dtype=float)) / 60.0
        phi = np.concatenate(
            [[0.0],
             np.cumsum(0.5 * (rev_per_s[1:] + rev_per_s[:-1]) * np.diff(t))]
        )
    return np.maximum.accumulate(phi)


def order_spectrum(sig: np.ndarray, phi: np.ndarray, M: int,
                   window: np.ndarray, phi_uniform: np.ndarray,
                   cfg: Config,
                   lp_hz: float = None, fs: float = None) -> np.ndarray:
    """Angular resample + windowed FFT.

    When *lp_hz* and *fs* are supplied the signal is anti-alias filtered in the
    time domain before resampling (R1).  Omitting them falls back to the legacy
    plain-interpolation path so callers that do not have RPM/time data are
    unaffected.
    """
    if lp_hz is not None and fs is not None:
        try:
            from nvh_pipeline.common import angular_resample_antialiased
            sig_r = angular_resample_antialiased(sig, phi, phi_uniform,
                                                  lp_hz, fs)
        except ImportError:
            sig_r = np.interp(phi_uniform, phi, sig)
    else:
        sig_r = np.interp(phi_uniform, phi, sig)

    if cfg.detrend == 'mean':
        sig_r = sig_r - np.mean(sig_r)
    elif cfg.detrend == 'linear':
        sig_r = sp_detrend(sig_r, type='linear')

    spec = np.fft.rfft(sig_r * window)
    amp  = np.abs(spec) * 2.0 / np.sum(window)
    if amp.size:
        amp[0] *= 0.5
    return amp


# ═══════════════════════════════════════════════════════════════════════════════
#  ACCUMULATION
# ═══════════════════════════════════════════════════════════════════════════════

class Accumulator:
    """
    Running average of per-axis order spectra across actuations.

    ``average`` controls how the per-axis spectra are combined across segments:
      • 'power'     — accumulate |A|² then return sqrt(mean), i.e. the RMS
                      (energy-mean) spectrum (default).  This is the energy-correct
                      average and is consistent with the RSS cumulative below.
                      For a tone of equal amplitude in every segment it equals the
                      tone amplitude (calibration unchanged vs magnitude-mean);
                      it differs only in how it weights segment-to-segment spread.
      • 'magnitude' — legacy mean(|A|), the arithmetic-mean amplitude spectrum.

    Note: averaging across actuations is *incoherent* (no phase alignment), so
    neither mode drives a random noise floor to zero — both are consistent
    estimators of their respective population spectra.

    ``cumulative_mode`` controls the combined trace:
      • 'rss' — sqrt(x²+y²+z²) vector magnitude (energy-correct, default).
      • 'sum' — legacy linear |x|+|y|+|z| (over-counts; axis-phase dependent).

    means() divides by actuation count so the result is average-per-actuation,
    normalizing for unequal trial counts.
    """

    def __init__(self, n_orders: int, average: str = 'power',
                 cumulative_mode: str = 'rss'):
        self.sum    = {'x': np.zeros(n_orders),
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
        """
        Returns mean amplitude per actuation for each axis, plus cumulative.
        Dividing by self.count normalizes for number of actuations so that
        a material with more trials is not artificially inflated.
        """
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


def group_key(meta: dict, cfg: Config) -> Optional[tuple]:
    if not cfg.group_by:
        return None
    return tuple(meta.get(k) for k in cfg.group_by)


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
    """Write a tidy long-format spectrum CSV the Review tab uses for the
    interactive Speed / Sample / Direction split.

    ``bucket_accs`` is keyed ``(dimension, value, material)``.  The order grid is
    decimated to ~``max_points`` to keep the embedded chart data light.
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
#  PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _integer_order_lines(ax, max_order: float) -> None:
    for k in range(1, int(np.floor(max_order)) + 1):
        ax.axvline(k, color='grey', lw=0.5, linestyle=':', alpha=0.35, zorder=0)


def _ball_pass_order_lines(ax, cfg: Config, max_order: float,
                            y_top: float) -> None:
    """
    Draw vertical lines at ball pass order and its harmonics.
    Labels sit just below the top of the plot area.
    """
    bpo = cfg.ball_pass_order
    for h in range(1, cfg.n_bpf_harmonics + 1):
        order_val = bpo * h
        if order_val > max_order:
            break
        ax.axvline(
            order_val,
            color='#9b59b6',   # purple — distinct from blue/orange/grey
            lw=1.0,
            linestyle='--',
            alpha=0.75,
            zorder=4,
        )
        label = (f'BPF×{h}\n{order_val:.2f}' if h > 1
                 else f'BPF\n{order_val:.2f}')
        ax.text(
            order_val + 0.05, y_top * 0.97,
            label,
            fontsize=7, color='#9b59b6',
            va='top', ha='left',
            zorder=5,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_axes(common_orders, means: dict, label: str,
              n_act: int, path: str, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor('#f7f7f7')
    ax.set_facecolor('#f7f7f7')
    if cfg.mark_integer_orders:
        _integer_order_lines(ax, common_orders[-1])
    ax.plot(common_orders, means['x'], lw=1.0, color='#1f77b4', label='mean X')
    ax.plot(common_orders, means['y'], lw=1.0, color='#2ca02c', label='mean Y')
    ax.plot(common_orders, means['z'], lw=1.0, color='#d62728', label='mean Z')
    ax.set_xlabel('Order (cycles / shaft revolution)', fontsize=11)
    ax.set_ylabel('Mean amplitude per actuation', fontsize=11)
    ax.set_title(f'Mean per-axis order spectrum  --  {label}  '
                 f'({n_act} actuations averaged)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, common_orders[-1])
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.25)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    if cfg.show_plots:
        plt.show()
    plt.close(fig)


def plot_cumulative(common_orders, means: dict, label: str,
                    n_act: int, path: str, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor('#f7f7f7')
    ax.set_facecolor('#f7f7f7')
    if cfg.mark_integer_orders:
        _integer_order_lines(ax, common_orders[-1])
    ax.plot(common_orders, means['cumulative'], lw=1.1, color='#111')
    ax.fill_between(common_orders, means['cumulative'], color='#111', alpha=0.12)
    y_top = float(np.max(means['cumulative']))
    _ball_pass_order_lines(ax, cfg, common_orders[-1], y_top)
    ax.set_xlabel('Order (cycles / shaft revolution)', fontsize=11)
    ax.set_ylabel('Cumulative amplitude per actuation (X + Y + Z)', fontsize=11)
    ax.set_title(f'Cumulative order spectrum  --  {label}  '
                 f'({n_act} actuations averaged)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, common_orders[-1])
    ax.grid(True, linestyle='--', alpha=0.25)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    if cfg.show_plots:
        plt.show()
    plt.close(fig)


def plot_comparison(common_orders,
                    metal_means: dict,   n_metal: int,
                    plastic_means: dict, n_plastic: int,
                    path: str, cfg: Config) -> None:
    """
    Overlay cumulative (X+Y+Z) order spectra for metal and plastic.

    Both curves are mean-per-actuation (already normalized by their respective
    actuation counts), so an unequal number of trials between materials does
    not bias the comparison.

    Ball pass order and harmonics are marked in purple.
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor('#f7f7f7')
    ax.set_facecolor('#f7f7f7')

    if cfg.mark_integer_orders:
        _integer_order_lines(ax, common_orders[-1])

    ax.plot(
        common_orders, metal_means['cumulative'],
        lw=1.3, color=cfg.color_metal,
        label=f'Metal  (n={n_metal} actuations)',
        zorder=3,
    )
    ax.fill_between(
        common_orders, metal_means['cumulative'],
        color=cfg.color_metal, alpha=0.10, zorder=2,
    )

    ax.plot(
        common_orders, plastic_means['cumulative'],
        lw=1.3, color=cfg.color_plastic,
        label=f'Plastic  (n={n_plastic} actuations)',
        zorder=3,
    )
    ax.fill_between(
        common_orders, plastic_means['cumulative'],
        color=cfg.color_plastic, alpha=0.10, zorder=2,
    )

    # Ball pass order lines — use the higher of the two curves for label y
    y_top = float(max(
        np.max(metal_means['cumulative']),
        np.max(plastic_means['cumulative']),
    ))
    _ball_pass_order_lines(ax, cfg, common_orders[-1], y_top)

    ax.set_xlabel('Order  (Events / Revolution)', fontsize=12)
    ax.set_ylabel('Cumulative amplitude per actuation  (X + Y + Z)', fontsize=12)
    ax.set_title(
        'Cumulative Order Spectrum — Metal vs Plastic\n'
        'Mean per actuation, normalized for trial counts\n',
        fontsize=13, fontweight='bold', pad=14,
    )
    ax.set_xlim(0, common_orders[-1])
    #ax.set_yscale('log')

    ax.legend(fontsize=11, framealpha=0.9, edgecolor='#cccccc',
              loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.25)

    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#cccccc')

    ax.tick_params(axis='both', which='both', length=0)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot  -> {os.path.abspath(path)}")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  PEAK PICKER
# ═══════════════════════════════════════════════════════════════════════════════

def top_peaks(common_orders, spectrum, n: int,
              min_order: float = 0.3) -> list[tuple]:
    mask = common_orders >= min_order
    o = common_orders[mask]
    s = spectrum[mask]
    if s.size < 3:
        return []
    is_peak = (s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:])
    idx = np.where(is_peak)[0] + 1
    if idx.size == 0:
        return []
    idx = idx[np.argsort(s[idx])[::-1][:n]]
    return [(float(o[i]), float(s[i])) for i in idx]


def bpfo_band_energy(common_orders: np.ndarray, spectrum: np.ndarray,
                     ball_pass_order: float, n_harmonics: int,
                     halfwidth: float = 0.25) -> dict:
    """Band-integrated energy around each BPFO harmonic (R2).

    Single-bin magnitude interpolation is not energy-conserving on a coarse
    order grid and is sensitive to bin alignment.  Integrating the spectrum
    over a ±halfwidth band around each harmonic gives a stable, resolution-
    independent BPFO energy estimate.

    Returns a dict with per-harmonic keys 'h1'..'hN' (trapezoidal integral)
    and 'total' (sum of all harmonics).  Values are zero when no grid points
    fall within the band.
    """
    result = {}
    total = 0.0
    for h in range(1, n_harmonics + 1):
        center = ball_pass_order * h
        lo, hi = center - halfwidth, center + halfwidth
        mask   = (common_orders >= lo) & (common_orders <= hi)
        n_pts  = int(mask.sum())
        if n_pts == 0:
            energy = 0.0
        elif n_pts == 1:
            # Single-point band: rectangle rule with full band width
            energy = float(spectrum[mask][0]) * 2.0 * halfwidth
        else:
            # np.trapezoid (NumPy >= 2.0) replaced np.trapz; the fallback must
            # not be evaluated eagerly or NumPy 2.x raises AttributeError.
            _trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
            energy = float(_trapz(spectrum[mask], common_orders[mask]))
        result[f'h{h}'] = round(energy, 8)
        total += energy
    result['total'] = round(total, 8)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  CAMPBELL  (order × measured-RPM, short-time windows over the full stroke)
# ═══════════════════════════════════════════════════════════════════════════════

def _campbell_order_rows(seg, axis_cols, cfg, common_orders) -> list:
    """Per-window ``(rpm, cumulative-amplitude-on-campbell-grid)`` for one stroke.

    Runs a short-time order analysis in sliding angular windows across the WHOLE
    stroke (ramps included) — each window tagged by its measured mean RPM — so the
    accumulated grid spans the continuous speed sweep, not just the nominal class.
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
    # Absent axes (continuous single-channel data) become zero channels.
    _zero = np.zeros(int(np.count_nonzero(keep)), dtype=float)
    sigs = {ax: (seg[axis_cols[ax]].to_numpy(dtype=float)[keep]
                 if axis_cols.get(ax) and axis_cols[ax] in seg.columns else _zero)
            for ax in ('x', 'y', 'z')}

    max_order = float(common_orders[-1]) if len(common_orders) else cfg.samples_per_rev / 2
    c_orders = campbell.order_grid(max_order, getattr(cfg, 'campbell_order_step', 0.1))
    cum_mode = getattr(cfg, 'cumulative_mode', 'rss')

    # Continuous mode: keep under-sampled windows so the Campbell still renders
    # (aliased above the raw Nyquist — surfaced with a warning in the Review tab).
    # np.interp needs >= 2 raw points, so floor at 4.  Reciprocating is unchanged.
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
            amp = order_spectrum(sigs[ax][i0:i1], phi_w, M, win, phi_uniform, cfg)
            per_axis.append(np.interp(c_orders, orders_fft, amp,
                                       left=0.0, right=0.0))
        if cum_mode == 'sum':
            cum = per_axis[0] + per_axis[1] + per_axis[2]
        else:
            cum = np.sqrt(per_axis[0] ** 2 + per_axis[1] ** 2 + per_axis[2] ** 2)
        rows.append((float(np.mean(rpm[i0:i1])), cum))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  SEGMENT WORKER  (called concurrently via ThreadPoolExecutor)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_oa_segment(task):
    """Compute order spectrum for one segment. Returns ('ok', result) or ('skip', reason)."""
    seg, meta, axis_cols, common_orders, cfg = task

    # Campbell (order × measured-RPM) runs on the FULL stroke including ramps —
    # compute it before plateau gating reassigns ``seg``.
    campbell_rows = (_campbell_order_rows(seg, axis_cols, cfg, common_orders)
                     if getattr(cfg, 'campbell', False) else [])

    # Steady-state gating: restrict the stroke to its constant-speed plateau so
    # the order anti-alias cutoff (below) is set by the plateau speed rather than
    # the slow ramp — otherwise the high-RPM order ceiling collapses.  Per-segment
    # fallback: if the plateau is too short, analyse the full stroke.
    used_plateau = False
    if getattr(cfg, 'plateau_gating', False) and cfg.rpm_col in seg.columns:
        from nvh_pipeline.common import plateau_mask
        pmask = plateau_mask(seg[cfg.rpm_col].to_numpy(dtype=float),
                             meta.get('rpm_category'),
                             getattr(cfg, 'plateau_frac', 0.90))
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
    window      = get_window(cfg.window, M, fftbins=True)

    # Anti-alias: compute LP cutoff at the angular Nyquist in Hz (R1).
    # Use the 5th-percentile of non-zero RPM over the kept samples.  With plateau
    # gating this is ~the plateau speed, so the cutoff sits at the angular Nyquist
    # in orders; without it, ramps drag rpm_min (and the cutoff) down.
    t_full  = seg[cfg.time_col].to_numpy(dtype=float)
    fs_aa   = (1.0 / float(np.median(np.diff(t_full)))
               if len(t_full) >= 2 else None)
    lp_hz   = None
    rpm_min = float('nan')
    rpm_plateau = float('nan')
    if fs_aa is not None and cfg.rpm_col in seg.columns:
        rpm_kept = np.abs(seg[cfg.rpm_col].to_numpy(dtype=float))[keep]
        moving   = rpm_kept[rpm_kept > 5.0]
        if moving.size:
            rpm_min     = float(np.percentile(moving, 5))
            rpm_plateau = float(np.median(moving))
            lp_hz       = max((cfg.samples_per_rev / 2) * rpm_min / 60.0, 10.0)

    amps = {}
    for ax_key in ('x', 'y', 'z'):
        col = axis_cols.get(ax_key)
        if col is None or col not in seg.columns:
            # Absent axis (continuous single-channel data): contribute nothing.
            amps[ax_key] = np.zeros_like(common_orders, dtype=float)
            continue
        sig = seg[col].to_numpy(dtype=float)[keep]
        amp = order_spectrum(sig, phi, M, window, phi_uniform, cfg,
                             lp_hz=lp_hz, fs=fs_aa)
        amps[ax_key] = np.interp(common_orders, orders_fft, amp,
                                  left=0.0, right=0.0)

    # Effective anti-alias order ceiling for this segment (diagnostics): how far
    # up the order axis content survives the LP.  = 64·(rpm_min/rpm_plateau).
    eff_aa_order = (lp_hz * 60.0 / rpm_plateau
                    if (lp_hz and rpm_plateau and rpm_plateau > 0)
                    else float('nan'))
    native_res = (1.0 / n_rev) if n_rev > 0 else float('nan')

    cum = amps['x'] + amps['y'] + amps['z']
    pk  = top_peaks(common_orders, cum, 1)

    hw = getattr(cfg, 'bpfo_band_halfwidth', 0.25)
    bpfo_energy = bpfo_band_energy(
        common_orders, cum,
        cfg.ball_pass_order, cfg.n_bpf_harmonics,
        halfwidth=hw,
    )

    return ('ok', {
        'meta':           meta,
        'amps':           amps,
        'n_rev':          n_rev,
        'dominant_order': round(pk[0][0], 4) if pk else float('nan'),
        'dominant_amp':   pk[0][1]            if pk else float('nan'),
        'bpfo_energy':    bpfo_energy,
        'campbell':       campbell_rows,
        'diag': {
            'part':            meta.get('part'),
            'material_type':   meta.get('material_type'),
            'rpm_category':    meta.get('rpm_category'),
            'segment_id':      meta.get('segment_id'),
            'source_file':     meta.get('source_file'),
            'used_plateau':    used_plateau,
            'n_rev':           round(n_rev, 3),
            'native_order_res': round(native_res, 4),
            'rpm_min':         round(rpm_min, 1) if rpm_min == rpm_min else None,
            'rpm_plateau':     round(rpm_plateau, 1) if rpm_plateau == rpm_plateau else None,
            'eff_aa_order':    round(eff_aa_order, 2) if eff_aa_order == eff_aa_order else None,
        },
    })


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
            f"no '{cfg.seg_glob}' files in {os.path.abspath(data_dir)} — "
            f"run the segmentation script first (or fix seg_output_dir)."
        )

    max_order = (cfg.max_order if cfg.max_order is not None
                 else cfg.samples_per_rev / 2)
    max_order = min(max_order, cfg.samples_per_rev / 2)
    common_orders = np.arange(0.0, max_order + cfg.order_resolution,
                               cfg.order_resolution)

    print(f"\nInput:            {os.path.abspath(data_dir)}  ({len(files)} files)")
    print(f"Output:           {os.path.abspath(cfg.output_dir)}")
    print(f"samples/rev:      {cfg.samples_per_rev}  "
          f"(Nyquist order = {cfg.samples_per_rev/2:g})")
    print(f"order grid:       0 .. {max_order:g}  step {cfg.order_resolution:g}")
    print(f"angle source:     {cfg.angle_source}")
    print(f"ball pass order:  {cfg.ball_pass_order}  "
          f"(harmonics: "
          f"{', '.join(f'{cfg.ball_pass_order*h:.2f}' for h in range(1, cfg.n_bpf_harmonics+1))})")
    print(f"group_by:         {cfg.group_by}\n")

    # Build accumulators with the configured averaging / cumulative modes.
    _avg = getattr(cfg, 'order_average', 'power')
    _cum = getattr(cfg, 'cumulative_mode', 'rss')
    def _new_acc() -> Accumulator:
        return Accumulator(len(common_orders), average=_avg, cumulative_mode=_cum)

    global_acc  = _new_acc()
    # Campbell (order × measured-RPM) accumulator over the ramp sweep.
    from nvh_pipeline import campbell as _campbell
    campbell_grid = _campbell.CampbellGrid(
        _campbell.order_grid(float(common_orders[-1]),
                             getattr(cfg, 'campbell_order_step', 0.1)),
        getattr(cfg, 'campbell_rpm_bin', 50.0))
    group_accs: dict[tuple, Accumulator] = {}
    # Finer-grained accumulators for the interactive Review split, keyed by
    # (dimension, value, material) where dimension is speed / sample / direction.
    bucket_accs: dict[tuple, Accumulator] = {}
    peak_rows:  list[dict] = []
    diag_rows:  list[dict] = []
    axis_cols:  Optional[dict] = None

    skips = {'min_revolutions': 0, 'min_fft_points': 0, 'filtered_out': 0}

    used_files: set = set()

    # ── Reduce one segment's result into the (additive) accumulators ──────────
    # Accumulators are order-independent, so reducing file-by-file gives totals
    # identical to reducing one global result list.
    def _reduce(res) -> None:
        status, data = res
        if status == 'skip':
            skips[data] += 1
            return

        meta  = data['meta']
        amps  = data['amps']
        n_rev = data['n_rev']

        global_acc.add(amps['x'], amps['y'], amps['z'], n_rev)
        used_files.add(meta['source_file'])

        gk = group_key(meta, cfg)
        if gk is not None:
            if gk not in group_accs:
                group_accs[gk] = _new_acc()
            group_accs[gk].add(amps['x'], amps['y'], amps['z'], n_rev)

        mat = meta['material_type']
        if mat in ('metal', 'plastic'):
            for dim, val in (('speed',     meta['rpm_category']),
                             ('sample',    meta['part']),
                             ('direction', meta['direction'])):
                if val is None:
                    continue
                bkey = (dim, str(val), mat)
                if bkey not in bucket_accs:
                    bucket_accs[bkey] = _new_acc()
                bucket_accs[bkey].add(amps['x'], amps['y'], amps['z'], n_rev)

        if 'diag' in data:
            diag_rows.append(data['diag'])

        for rpm_w, amp_w in data.get('campbell', []):
            campbell_grid.add(meta['material_type'], rpm_w, amp_w)

        bpfo_e = data.get('bpfo_energy', {})
        peak_rows.append({
            **{k: meta[k] for k in (
                'part', 'material_type', 'rpm_category', 'trial',
                'direction', 'segment_id', 'source_file')},
            'revolutions':          round(n_rev, 3),
            'dominant_order':       data['dominant_order'],
            'dominant_amp':         data['dominant_amp'],
            'bpfo_band_energy_total': bpfo_e.get('total', float('nan')),
            **{f'bpfo_band_energy_{k}': v
               for k, v in bpfo_e.items() if k != 'total'},
        })

    # ── Stream segments one file at a time (memory-bounded) ───────────────────
    # Each file's segments are collected, processed, reduced, then released —
    # peak memory stays near one file's segments instead of every file's at once.
    for path in files:
        if axis_cols is None:
            axis_cols = resolve_axis_columns(
                dataset.file_columns(data_dir, path), cfg)
            print(f"Axis channels:    X={axis_cols['x']!r}  "
                  f"Y={axis_cols['y']!r}  Z={axis_cols['z']!r}\n")

        if 'segment_id' not in dataset.file_columns(data_dir, path):
            print(f"  SKIP (no segment_id): {os.path.basename(path)}")
            continue

        tasks: list = []
        for seg_id, seg in dataset.iter_file_segments(data_dir, path):
            meta = {
                'part':          seg['part'].iloc[0]
                                 if 'part' in seg else None,
                'material_type': seg['material_type'].iloc[0]
                                 if 'material_type' in seg else None,
                'rpm_category':  int(seg['rpm_category'].iloc[0])
                                 if 'rpm_category' in seg else None,
                'trial':         int(seg['trial'].iloc[0])
                                 if 'trial' in seg else None,
                'direction':     seg['direction'].iloc[0]
                                 if 'direction' in seg else None,
                'segment_id':    int(seg_id),
                'source_file':   os.path.basename(path),
            }

            if cfg.only_parts and meta['part'] not in cfg.only_parts:
                skips['filtered_out'] += 1; continue
            if cfg.only_rpms and meta['rpm_category'] not in cfg.only_rpms:
                skips['filtered_out'] += 1; continue
            if cfg.only_direction and meta['direction'] != cfg.only_direction:
                skips['filtered_out'] += 1; continue

            tasks.append((seg, meta, axis_cols, common_orders, cfg))

        if not tasks:
            continue

        # Process this file's segments in parallel (numpy/scipy release the GIL).
        n_workers = min(os.cpu_count() or 4, len(tasks))
        print(f"Processing {len(tasks)} segments from "
              f"{os.path.basename(path)} across {n_workers} threads ...")
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for res in pool.map(_compute_oa_segment, tasks):
                _reduce(res)

    n_files_used = len(used_files)

    if global_acc.count == 0:
        raise RuntimeError(
            "no actuations survived filtering / quality gates — "
            f"skips={skips}.  Loosen min_revolutions or check angle column."
        )

    # ── global result ────────────────────────────────────────────────────────
    means = global_acc.means()
    save_mean_csv(common_orders, means,
                  os.path.join(cfg.output_dir, 'order_spectra_mean.csv'))
    plot_axes(common_orders, means, 'ALL actuations', global_acc.count,
              os.path.join(plots_dir, 'order_mean_axes.png'), cfg)
    plot_cumulative(common_orders, means, 'ALL actuations', global_acc.count,
                    os.path.join(plots_dir, 'order_cumulative.png'), cfg)

    # ── per-group results ────────────────────────────────────────────────────
    group_summaries = []
    group_means_by_material:  dict[str, dict] = {}
    group_counts_by_material: dict[str, int]  = {}

    for gk, acc in sorted(group_accs.items(), key=lambda kv: str(kv[0])):
        label  = '_'.join(str(x) for x in gk)
        gmeans = acc.means()

        save_mean_csv(common_orders, gmeans,
                      os.path.join(cfg.output_dir,
                                   f'order_spectra_mean_{label}.csv'))
        if cfg.render_plots:
            plot_axes(common_orders, gmeans, label, acc.count,
                      os.path.join(plots_dir,
                                   f'order_mean_axes_{label}.png'), cfg)
            plot_cumulative(common_orders, gmeans, label, acc.count,
                            os.path.join(plots_dir,
                                         f'order_cumulative_{label}.png'), cfg)

        group_summaries.append({
            'group':            label,
            'n_actuations':     acc.count,
            'mean_revolutions': round(acc.rev_sum / acc.count, 3),
        })

        if 'material_type' in cfg.group_by:
            mat_idx = list(cfg.group_by).index('material_type')
            mat     = gk[mat_idx]
            if mat in ('metal', 'plastic'):
                group_means_by_material[mat]  = gmeans
                group_counts_by_material[mat] = acc.count

    # ── tidy by-group spectra for the interactive Review split ───────────────
    save_by_group_csv(common_orders, bucket_accs,
                      os.path.join(cfg.output_dir,
                                   'order_spectra_by_group.csv'))

    # ── metal vs plastic comparison plot ────────────────────────────────────
    if ('metal'   in group_means_by_material and
            'plastic' in group_means_by_material):
        if cfg.render_plots:
            plot_comparison(
                common_orders,
                metal_means   = group_means_by_material['metal'],
                n_metal       = group_counts_by_material['metal'],
                plastic_means = group_means_by_material['plastic'],
                n_plastic     = group_counts_by_material['plastic'],
                path          = os.path.join(plots_dir,
                                             'order_cumulative_comparison.png'),
                cfg           = cfg,
            )
    else:
        print("WARNING: could not find both 'metal' and 'plastic' groups — "
              "comparison plot skipped.  Check material_type values in data.")

    # ── peaks csv + metadata ─────────────────────────────────────────────────
    if peak_rows:
        pd.DataFrame(peak_rows).to_csv(
            os.path.join(cfg.output_dir, 'per_actuation_peaks.csv'),
            index=False,
        )

    # ── Campbell waterfall (order × measured-RPM) ────────────────────────────
    if getattr(cfg, 'campbell', False):
        cdf = campbell_grid.to_long_df()
        if not cdf.empty:
            cdf.to_csv(os.path.join(cfg.output_dir, 'order_campbell.csv'),
                       index=False)
            print(f"Campbell map:     {cdf['rpm'].nunique()} RPM bins x "
                  f"{cdf['order'].nunique()} orders -> order_campbell.csv")

    # ── validity diagnostics ─────────────────────────────────────────────────
    # Per-segment record + a per-(RPM class, material) summary that makes the
    # order-resolution and anti-alias limits auditable rather than asserted.
    diag_summary = []
    if diag_rows:
        ddf = pd.DataFrame(diag_rows)
        ddf.to_csv(os.path.join(cfg.output_dir, 'order_diagnostics.csv'),
                   index=False)
        for (rpm_cat, mat), g in ddf.groupby(['rpm_category', 'material_type']):
            diag_summary.append({
                'rpm_category':       rpm_cat,
                'material_type':      mat,
                'n_segments':         int(len(g)),
                'n_plateau_fallback': int((~g['used_plateau']).sum()),
                'n_rev_min':          float(g['n_rev'].min()),
                'n_rev_median':       float(g['n_rev'].median()),
                'n_rev_max':          float(g['n_rev'].max()),
                'native_order_res_median': float(g['native_order_res'].median()),
                'eff_aa_order_median':     float(g['eff_aa_order'].median(skipna=True))
                                           if g['eff_aa_order'].notna().any() else None,
            })
        if diag_summary:
            pd.DataFrame(diag_summary).to_csv(
                os.path.join(cfg.output_dir, 'order_diagnostics_summary.csv'),
                index=False)
            print("\nValidity diagnostics (per RPM class / material):")
            for r in diag_summary:
                print(f"  {r['material_type']:<7} {r['rpm_category']:>5} RPM: "
                      f"{r['n_segments']:>3} seg, "
                      f"n_rev≈{r['n_rev_median']:.1f} "
                      f"(res {r['native_order_res_median']:.2f} ord), "
                      f"eff. anti-alias order "
                      f"{r['eff_aa_order_median'] if r['eff_aa_order_median'] is not None else 'n/a'}"
                      f"{'  [' + str(r['n_plateau_fallback']) + ' full-stroke fallback]' if r['n_plateau_fallback'] else ''}")

    run_meta = {
        'n_files_used':      n_files_used,
        'n_actuations':      global_acc.count,
        'mean_revolutions':  round(global_acc.rev_sum / global_acc.count, 3),
        'axis_columns':      axis_cols,
        'ball_pass_order':   cfg.ball_pass_order,
        'bpf_harmonics':     [cfg.ball_pass_order * h
                              for h in range(1, cfg.n_bpf_harmonics + 1)],
        'common_order_grid': {
            'max_order':  float(max_order),
            'resolution': cfg.order_resolution,
            'n_bins':     len(common_orders),
        },
        'skips':           skips,
        'group_by':        list(cfg.group_by),
        'group_summaries': group_summaries,
        'diagnostics_summary': diag_summary,
        'config': {k: getattr(cfg, k) for k in (
            'samples_per_rev', 'detrend', 'window', 'min_revolutions',
            'angle_source', 'only_parts', 'only_rpms', 'only_direction',
            'plateau_gating', 'plateau_frac', 'order_average',
            'cumulative_mode')},
    }
    with open(os.path.join(cfg.output_dir, 'run_metadata.json'), 'w') as f:
        json.dump(run_meta, f, indent=2, default=str)

    # ── console summary ──────────────────────────────────────────────────────
    print(f"\nActuations used:  {global_acc.count}  "
          f"(mean {run_meta['mean_revolutions']:g} rev each)")
    print(f"Skipped:          {skips}")
    print(f"\nTop cumulative-spectrum orders (global):")
    for o, a in top_peaks(common_orders, means['cumulative'], cfg.n_top_peaks):
        print(f"    order {o:7.3f}    amp {a:.5g}")
    print(f"\nOutput -> {os.path.abspath(cfg.output_dir)}\n")

    return {'common_orders': common_orders, 'means': means, 'meta': run_meta}


def _apply_pipeline_config(cfg, _stage='order'):
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