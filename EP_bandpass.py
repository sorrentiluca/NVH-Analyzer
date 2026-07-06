"""
═══════════════════════════════════════════════════════════════════════════════
 BALL PASS FREQUENCY VISUALIZATION
 Raw signal → Bandpass filtered → Envelope extraction
═══════════════════════════════════════════════════════════════════════════════

Pulls one actuation at 2300 RPM from the parquet files and produces a
three-panel plot showing the full signal processing chain:

  Panel 1 — Raw time domain signal (noisy, full spectrum)
  Panel 2 — Bandpass filtered signal (kurtogram-optimal band)
  Panel 3 — Filtered signal + envelope overlay, with ball pass period markers

All three panels share the same time axis so the transformation is directly
comparable.

USAGE
-----
  python plot_ballpass_extraction.py

  Adjust CONFIG below. PART_ID and TRIAL let you pick a specific actuation.
  Set them to None to auto-select the first available 2300 RPM actuation.

DEPENDENCIES
------------
  pip install pandas numpy scipy matplotlib pyarrow
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import glob
import os
import re
import warnings
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
from scipy.fft import next_fast_len


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── IO ──────────────────────────────────────────────────────────────────
    seg_output_dir: str = 'output_seg'
    data_subdir: str    = 'segmented_data'
    seg_glob: str       = '*_segments.parquet'

    output_dir: str  = 'output_envelope'
    output_file: str = 'ballpass_extraction.png'

    # ── WHICH ACTUATION TO PLOT ──────────────────────────────────────────────
    # Set to None to auto-select the first available 2300 RPM actuation.
    target_rpm: int         = 2300
    part_id: Optional[str]  = None   # e.g. 'T8' or 'm627' — None = any
    trial: Optional[int]    = None   # e.g. 1 — None = any
    segment_id: Optional[int] = None # None = first segment in the file
    material: Optional[str] = None   # 'metal' or 'plastic' — None = any

    # Which axis to plot (the kurtogram picks the best band for this axis)
    axis_to_plot: str = 'x'          # 'x', 'y', or 'z'

    # 'reciprocating' or 'continuous' (single-channel data → allow missing axes).
    analysis_mode: str = 'reciprocating'

    # ── COLUMNS ─────────────────────────────────────────────────────────────
    time_col: str  = 'Time (s)'
    rpm_col: str   = 'CNT 1/Frequency (RPM)'
    angle_col: str = 'CNT 1/Angle (Degrees)'

    axis_x: Optional[str] = None
    axis_y: Optional[str] = None
    axis_z: Optional[str] = None

    # ── KURTOGRAM ───────────────────────────────────────────────────────────
    bp_min_hz: float      = 1000.0
    bp_min_hz_abs: float  = 50.0   # absolute floor — never search below this Hz
    bp_min_bw_hz: float   = 200.0
    # MUST match PipelineConfig.kurtogram_levels (nvh_pipeline/config.py) so a
    # standalone run and a pipeline/FAMOS run select bands identically.
    kurtogram_levels: int = 4
    bp_filter_order: int  = 4

    # ── BALL PASS ────────────────────────────────────────────────────────────
    ball_pass_order: float = 5.35   # cycles per shaft revolution

    # ── PLOT ────────────────────────────────────────────────────────────────
    show_plot:    bool = True
    render_plots: bool = True
    figsize: tuple  = (15, 10)

    # Colors
    color_raw:      str = '#444444'
    color_filtered: str = '#2c7bb6'
    color_envelope: str = '#f46d43'
    color_bpf:      str = '#9b59b6'


# ═══════════════════════════════════════════════════════════════════════════════
#  COLUMN RESOLUTION  (same as envelope script)
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

    allow_missing = getattr(cfg, 'analysis_mode', 'reciprocating') == 'continuous'

    resolved: dict = {}
    for axis in ('x', 'y', 'z'):
        if explicit[axis] is not None:
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
                f"Could not auto-detect {axis.upper()} axis. "
                f"Set Config.axis_{axis} explicitly."
            )
        resolved[axis] = pick_from[0]

    return resolved


# ═══════════════════════════════════════════════════════════════════════════════
#  KURTOGRAM  (same algorithm as envelope script)
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_fs(time: np.ndarray) -> float:
    if np.asarray(time).size < 2:
        raise ValueError("need >= 2 time samples to estimate sample rate")
    dt = np.median(np.diff(time))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("non-positive / non-finite time step — check time column")
    return 1.0 / dt


def _bandpass_kurtosis(sig: np.ndarray, fs: float,
                        f_low: float, f_high: float,
                        order: int) -> float:
    nyq      = fs / 2.0
    f_low_n  = f_low  / nyq
    f_high_n = f_high / nyq
    if f_low_n <= 0 or f_high_n >= 1.0 or f_low_n >= f_high_n:
        return -np.inf
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            b, a     = butter(order, [f_low_n, f_high_n], btype='band')
            filtered = filtfilt(b, a, sig)
    except Exception:
        return -np.inf
    # Kurtosis (mu4/mu2^2, not excess — no -3 term)
    mu4 = np.mean((filtered - np.mean(filtered)) ** 4)
    mu2 = np.mean((filtered - np.mean(filtered)) ** 2)
    if mu2 < 1e-30:
        return -np.inf
    return float(mu4 / (mu2 ** 2))


def detect_band_kurtogram(sig: np.ndarray, fs: float,
                           cfg: Config,
                           min_hz: float | None = None) -> tuple[float, float, float]:
    """``min_hz`` overrides cfg.bp_min_hz for this call (e.g. dynamic RPM floor)."""
    eff_min  = min_hz if min_hz is not None else cfg.bp_min_hz
    nyq      = fs / 2.0
    # Cap the search below Nyquist (and the calibrated ceiling) and keep the floor
    # strictly below it, so the returned band is always valid (f_low < f_high)
    # even when the requested floor is too high for this sample rate.
    f_max    = min(nyq,
                   getattr(cfg, 'bp_max_hz_frac', 1.0) * nyq,
                   getattr(cfg, 'bp_max_hz', nyq))
    min_bw   = getattr(cfg, 'bp_min_bw_hz', 200.0)
    eff_min  = min(eff_min, max(1.0, f_max - min_bw))
    f_usable = max(0.0, f_max - eff_min)

    best_kurt = -np.inf
    best_band = (eff_min, (eff_min + f_max) / 2, f_max)

    for level in range(1, cfg.kurtogram_levels + 1):
        n_bands = 2 ** level
        bw      = f_usable / n_bands
        if bw < min_bw:
            break
        for k in range(n_bands):
            f_low  = eff_min + k * bw
            f_high = f_low + bw
            f_ctr  = (f_low + f_high) / 2.0
            kurt   = _bandpass_kurtosis(
                sig, fs, f_low, f_high, cfg.bp_filter_order
            )
            if kurt > best_kurt:
                best_kurt = kurt
                best_band = (f_low, f_ctr, f_high)

    return best_band


def extract_envelope(sig: np.ndarray, fs: float,
                     f_low: float, f_high: float,
                     cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Returns (filtered_signal, envelope). Reflect-pads before hilbert() to suppress edge distortion."""
    nyq      = fs / 2.0
    f_low_n  = np.clip(f_low  / nyq, 1e-4, 0.9999)
    f_high_n = np.clip(f_high / nyq, 1e-4, 0.9999)
    # Degenerate band (edges collapsed after clipping, e.g. a floor above Nyquist):
    # skip the bandpass and take the envelope of the raw signal instead of calling
    # butter with equal/inverted edges (which raises "Wn[0] must be less than Wn[1]").
    if f_high_n - f_low_n < 1e-3:
        filtered = np.asarray(sig, dtype=float)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            b, a     = butter(cfg.bp_filter_order, [f_low_n, f_high_n], btype='band')
            filtered = filtfilt(b, a, sig)
    n        = len(filtered)
    pad      = min(n, max(256, n // 4))
    padded   = np.pad(filtered, pad, mode='reflect')
    n_fft    = next_fast_len(len(padded))
    analytic = hilbert(padded, N=n_fft)
    envelope = np.abs(analytic[pad:pad + n])
    return filtered, envelope


# ═══════════════════════════════════════════════════════════════════════════════
#  ACTUATION FINDER
# ═══════════════════════════════════════════════════════════════════════════════

def find_actuation(cfg: Config) -> tuple[pd.DataFrame, dict]:
    """
    Search parquet files for a matching actuation.
    Returns (segment_dataframe, info_dict).
    """
    from nvh_pipeline import dataset
    data_dir = os.path.join(cfg.seg_output_dir, cfg.data_subdir)
    files    = dataset.list_segment_files(data_dir)

    if not files:
        raise FileNotFoundError(
            f"No parquet files found in {os.path.abspath(data_dir)}"
        )

    # Continuous mode has a single whole-signal actuation — no RPM/part/trial
    # selection applies, so take the first available segment.
    _continuous = getattr(cfg, 'analysis_mode', 'reciprocating') == 'continuous'

    for path in files:
        df = dataset.read_file(data_dir, path)

        # Apply filters
        if not _continuous and 'rpm_category' in df.columns:
            df = df[df['rpm_category'] == cfg.target_rpm]
        if not _continuous and cfg.part_id and 'part' in df.columns:
            df = df[df['part'] == cfg.part_id]
        if not _continuous and cfg.trial and 'trial' in df.columns:
            df = df[df['trial'] == cfg.trial]
        if not _continuous and cfg.material and 'material_type' in df.columns:
            df = df[df['material_type'] == cfg.material]

        if df.empty:
            continue

        # Pick segment
        seg_ids = df['segment_id'].unique()
        if cfg.segment_id is not None and cfg.segment_id in seg_ids:
            seg_id = cfg.segment_id
        else:
            seg_id = seg_ids[0]

        seg = df[df['segment_id'] == seg_id].copy()

        info = {
            'source_file': os.path.basename(path),
            'part':        seg['part'].iloc[0]          if 'part'          in seg else '?',
            'material':    seg['material_type'].iloc[0] if 'material_type' in seg else '?',
            'rpm_cat':     seg['rpm_category'].iloc[0]  if 'rpm_category'  in seg else '?',
            'trial':       seg['trial'].iloc[0]         if 'trial'         in seg else '?',
            'segment_id':  int(seg_id),
        }
        print(f"Selected actuation:")
        print(f"  File:       {info['source_file']}")
        print(f"  Part:       {info['part']}  ({info['material']})")
        print(f"  RPM:        {info['rpm_cat']}")
        print(f"  Trial:      {info['trial']}")
        print(f"  Segment ID: {info['segment_id']}")
        return seg, info

    raise RuntimeError(
        f"No matching actuation found for RPM={cfg.target_rpm}, "
        f"part={cfg.part_id}, trial={cfg.trial}, material={cfg.material}. "
        f"Check Config filters."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_extraction(time: np.ndarray,
                    raw: np.ndarray,
                    filtered: np.ndarray,
                    envelope: np.ndarray,
                    f_low: float, f_high: float, f_center: float,
                    kurtosis: float,
                    bp_period_s,           # float or None when rpm_cat=0
                    info: dict,
                    axis_col: str,
                    cfg: Config) -> None:

    os.makedirs(cfg.output_dir, exist_ok=True)

    t0   = time[0]
    time = time - t0    # zero-reference time axis

    fig = plt.figure(figsize=cfg.figsize)
    fig.patch.set_facecolor('#f7f7f7')

    gs = gridspec.GridSpec(
        3, 1, hspace=0.55,
        top=0.88, bottom=0.08,
        left=0.07, right=0.97,
    )

    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    for ax in axes:
        ax.set_facecolor('#f7f7f7')
        ax.grid(True, linestyle='--', alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.tick_params(length=0)

    # ── Panel 1: Raw signal ──────────────────────────────────────────────────
    axes[0].plot(time, raw, lw=0.5, color=cfg.color_raw, alpha=0.85)
    axes[0].set_ylabel('Amplitude', fontsize=10)
    axes[0].set_title(
        'Panel 1 — Raw vibration signal',
        fontsize=11, fontweight='bold', loc='left', pad=6,
    )
    axes[0].set_xlim(time[0], time[-1])
    axes[0].tick_params(labelbottom=False)

    # ── Panel 2: Bandpass filtered ───────────────────────────────────────────
    axes[1].plot(time, filtered, lw=0.6, color=cfg.color_filtered, alpha=0.9)
    axes[1].set_ylabel('Amplitude', fontsize=10)
    axes[1].set_title(
        f'Panel 2 — Bandpass filtered  '
        f'({f_low:.0f}–{f_high:.0f} Hz  |  '
        f'resonance ≈ {f_center:.0f} Hz  |  '
        f'kurtosis = {kurtosis:.1f})',
        fontsize=11, fontweight='bold', loc='left', pad=6,
    )
    axes[1].set_xlim(time[0], time[-1])
    axes[1].tick_params(labelbottom=False)

    # ── Panel 3: Filtered + envelope + ball pass markers ────────────────────
    axes[2].plot(
        time, filtered,
        lw=0.5, color=cfg.color_filtered, alpha=0.4,
        label='Filtered signal',
    )
    axes[2].plot(
        time, envelope,
        lw=1.5, color=cfg.color_envelope,
        label='Envelope  (|Hilbert|)',
        zorder=3,
    )
    axes[2].plot(
        time, -envelope,
        lw=1.5, color=cfg.color_envelope, zorder=3,
    )

    # Ball pass period markers — vertical dashed lines (skipped if rpm=0)
    if bp_period_s is not None and bp_period_s > 0:
        n_markers = int(np.floor(time[-1] / bp_period_s)) + 1
        first_marker = True
        for k in range(n_markers):
            t_mark = k * bp_period_s
            if t_mark > time[-1]:
                break
            label = f'Ball pass period  ({bp_period_s*1000:.1f} ms)' if first_marker else None
            axes[2].axvline(
                t_mark,
                color=cfg.color_bpf,
                lw=0.9, linestyle='--', alpha=0.7,
                zorder=4, label=label,
            )
            first_marker = False

    axes[2].set_ylabel('Amplitude', fontsize=10)
    axes[2].set_xlabel('Time (s)', fontsize=10)
    axes[2].set_title(
        'Panel 3 — Envelope extraction  |  ball pass period markers (purple)',
        fontsize=11, fontweight='bold', loc='left', pad=6,
    )
    axes[2].set_xlim(time[0], time[-1])
    axes[2].legend(fontsize=9, loc='upper right', framealpha=0.85,
                   edgecolor='#cccccc')

    # ── Super title ──────────────────────────────────────────────────────────
    ball_pass_hz = cfg.ball_pass_order * (info['rpm_cat'] / 60.0)
    if bp_period_s is not None:
        bpf_line = (f'Ball pass order {cfg.ball_pass_order} × '
                    f'{info["rpm_cat"]} RPM / 60 = {ball_pass_hz:.1f} Hz  '
                    f'→  period = {bp_period_s*1000:.2f} ms')
    else:
        bpf_line = f'Ball pass: N/A (RPM = 0)'
    fig.suptitle(
        f'Ball Pass Frequency Extraction  —  {info["part"]}  '
        f'({info["material"]})  |  {info["rpm_cat"]} RPM  '
        f'|  Segment {info["segment_id"]}  |  Axis {cfg.axis_to_plot.upper()}\n'
        f'{bpf_line}',
        fontsize=12, fontweight='bold', y=0.97,
    )

    out_path = os.path.join(cfg.output_dir, cfg.output_file)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved -> {os.path.abspath(out_path)}")
    if cfg.show_plot:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg = Config()
    # ImportError = nvh_pipeline genuinely absent (bare standalone run): tolerate
    # and use built-in defaults.  Any other error from apply_to_stage is a real
    # misconfiguration and must surface, not silently fall back to defaults.
    try:
        from nvh_pipeline.config import apply_to_stage
    except ImportError:
        apply_to_stage = None
    if apply_to_stage is not None:
        apply_to_stage('bandpass', cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Find actuation
    seg, info = find_actuation(cfg)

    # Resolve columns
    axis_cols = resolve_axis_columns(list(seg.columns), cfg)
    axis_col  = axis_cols.get(cfg.axis_to_plot)
    if axis_col is None:
        # Continuous single-channel data: fall back to the first present axis.
        axis_col = next((axis_cols[a] for a in ('x', 'y', 'z')
                         if axis_cols.get(a)), None)
        if axis_col is None:
            raise KeyError("no vibration axis column available to plot.")
    print(f"\nUsing axis column: {axis_col!r}")

    # Extract arrays
    time = seg[cfg.time_col].to_numpy(dtype=float)
    raw  = seg[axis_col].to_numpy(dtype=float)
    raw  = raw - np.mean(raw)   # remove DC

    fs = estimate_fs(time)
    print(f"Sample rate: {fs:.1f} Hz  (Nyquist: {fs/2:.1f} Hz)")

    # Kurtogram — optimal band for this actuation and axis.
    # Use a dynamic floor scaled to actual RPM so low-speed actuations are not
    # blocked by the 1000 Hz default (sized for 2300 RPM).
    print("\nRunning kurtogram ...")
    # Continuous mode: the ballscrew order-25 shaft floor is meaningless for a
    # variable-speed signal (and can exceed Nyquist), so use the absolute floor.
    if getattr(cfg, 'analysis_mode', 'reciprocating') == 'continuous':
        _dyn_min_hz = getattr(cfg, 'bp_min_hz_abs', 50.0)
    else:
        _rpm_bp = info['rpm_cat'] if isinstance(info['rpm_cat'], (int, float)) else cfg.target_rpm
        _dyn_min_hz = max(getattr(cfg, 'bp_min_hz_abs', 50.0), 25 * _rpm_bp / 60 * 1.2)
    f_low, f_center, f_high = detect_band_kurtogram(raw, fs, cfg, min_hz=_dyn_min_hz)
    kurtosis = _bandpass_kurtosis(raw, fs, f_low, f_high, cfg.bp_filter_order)
    print(f"Optimal band:  {f_low:.0f}–{f_high:.0f} Hz  "
          f"(center {f_center:.0f} Hz,  kurtosis = {kurtosis:.2f})")

    # Bandpass + envelope
    filtered, envelope = extract_envelope(raw, fs, f_low, f_high, cfg)

    # Ball pass period in seconds (guard: rpm_cat=0 would give 1/0)
    ball_pass_hz = cfg.ball_pass_order * (info['rpm_cat'] / 60.0)
    if ball_pass_hz > 0:
        bp_period_s = 1.0 / ball_pass_hz
        print(f"Ball pass:     order {cfg.ball_pass_order} × "
              f"{info['rpm_cat']} RPM / 60 = {ball_pass_hz:.1f} Hz  "
              f"→  period = {bp_period_s*1000:.2f} ms")
    else:
        bp_period_s = None
        print(f"Ball pass:     skipped (rpm_cat={info['rpm_cat']} — cannot compute period)")

    # Plot
    if cfg.render_plots:
        plot_extraction(
            time, raw, filtered, envelope,
            f_low, f_high, f_center, kurtosis,
            bp_period_s, info, axis_col, cfg,
        )


if __name__ == '__main__':
    main()