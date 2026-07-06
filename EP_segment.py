"""
═══════════════════════════════════════════════════════════════════════════════
 ROBUST ACTUATION SEGMENTATION  —  100 RPM to 2300 RPM
═══════════════════════════════════════════════════════════════════════════════

Segments each actuation from the moment the shaft starts moving (RPM leaves
zero) to the moment it stops (RPM returns to zero).  Works for both positive
and negative direction actuations.

FILE NAMING CONVENTION
----------------------
  <part>-<rpm>rpm-<trial>.csv

  where <part> is:
    m###  -> metal part (e.g., m635)
    T###  -> plastic part (e.g., T13, T8)

  Examples:  m635-100rpm-1.csv,  T13-2300rpm-3.csv

ALGORITHM
---------
  1. Lightly smooth the RPM signal to kill encoder sample-level jitter.
  2. Find contiguous regions where |RPM| > detect_threshold_frac × category_rpm.
     These are the "cores" — guaranteed to be in-motion, not noise.
  3. From the LEFT edge of each core: walk backward until |RPM| < zero_rpm_threshold.
  4. From the RIGHT edge of each core: walk forward until |RPM| < zero_rpm_threshold.
  5. Enforce direction alternation (POS/NEG/POS/NEG/...):
       - merge consecutive same-direction segments closer than max_merge_gap_s
         (fixes the "one actuation got split in two" failure mode)
       - for remaining same-direction consecutives, either warn or drop the weaker
  6. Discard segments shorter than min_duration_s (AFTER merging, so short
     fragments of a split actuation get a chance to be joined first).

OUTPUTS
-------
  output_seg/
    plots/
      seg_<part>_<rpm>rpm_<stem>.png       gut-check overview + overlay
    segmented_data/
      <stem>_segments.parquet              all rows within segments
      <stem>_metadata.json                 per-file segment summary + alt stats
    summary.csv                            one row per segment across every file

DOWNSTREAM USAGE
----------------
  import pandas as pd
  df = pd.read_parquet('output_seg/segmented_data/m635-100rpm-1_segments.parquet')
  for seg_id, seg_df in df.groupby('segment_id'):
      ...

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
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── IO ──────────────────────────────────────────────────────────────────
    # Folder of raw DEWESOFT CSVs.  Normally set from nvh_config.json (data_dir);
    # this relative default is only used for a bare standalone run.
    base_path: str = 'data'

    # Part identifiers to process.  Material type is auto-detected from prefix:
    # 'm*' -> metal, 'T*' -> plastic.  Swap 'T13' for 'T8' when next part starts.
    parts: tuple = ('m627', 'm625', 'm635', 'T8', 'T11', 'T13')

    rpms: tuple = (100, 200, 400, 800, 1600, 2300)

    filename_pattern: str = '{part}-{rpm}rpm-*.csv'

    # ── ANALYSIS MODE ───────────────────────────────────────────────────────
    # 'reciprocating' (default, ballscrew rig) or 'continuous' (single steady
    # rotating-machine signal → one whole-signal segment, no naming required).
    # See nvh_pipeline/config.py for the full description.
    analysis_mode: str = 'reciprocating'
    single_file: Optional[str] = None      # continuous: the one CSV to analyse
    nominal_rpm: Optional[float] = None    # continuous: steady speed (None=auto)
    signal_label: str = 'signal'           # continuous: label in place of part id

    # ── COLUMNS ─────────────────────────────────────────────────────────────
    time_col: str  = 'Time (s)'
    rpm_col: str   = 'CNT 1/Frequency (RPM)'
    angle_col: str = 'CNT 1/Angle (Degrees)'

    # None = keep all columns from the CSV in the saved parquet.
    keep_columns: Optional[tuple] = None

    # ── SEGMENTATION ────────────────────────────────────────────────────────
    smooth_ms: float = 0.05
    detect_threshold_frac: float = 0.1   # 10% of category RPM
    zero_rpm_threshold: float = 3.0       # RPM
    min_duration_s: float = 0.05

    # ── DIRECTION-ALTERNATION ENFORCEMENT ───────────────────────────────────
    # Real actuations should alternate POS/NEG/POS/NEG.  Two same-direction
    # segments in a row usually mean either:
    #   (a) one real actuation got split (RPM briefly dipped mid-ramp)
    #   (b) the OTHER-direction actuation between them was missed (too weak)
    #   (c) a spurious noise detection in the dominant direction
    #
    # Case (a) is fixed by merging close consecutives.
    # Cases (b)/(c) can't be distinguished automatically — we either warn
    # (safe default — investigate manually) or drop the weaker segment.

    enforce_alternation: bool = True

    # Same-direction segments with a gap below this are assumed to be a single
    # split actuation, and are merged into one.  Set to 0 to disable merging.
    max_merge_gap_s: float = 0.5

    # What to do with same-direction consecutives that survive merging:
    #   'warn'         keep both, print warning, flag in metadata
    #   'drop_weaker'  drop the segment with the lower peak RPM
    unresolved_action: str = 'warn'

    # ── OUTPUT ──────────────────────────────────────────────────────────────
    output_dir: str   = 'output_seg'
    plots_subdir: str = 'plots'
    data_subdir: str  = 'segmented_data'
    save_format: str  = 'parquet'   # or 'pickle'

    show_plots: bool = False
    render_plots: bool = True
    n_overlay: int   = 3
    overview_max_points: int = 100_000


# ═══════════════════════════════════════════════════════════════════════════════
#  FILENAME PARSING
# ═══════════════════════════════════════════════════════════════════════════════

_FILENAME_RE = re.compile(r'^([A-Za-z]+\d+)-(\d+)rpm-(\d+)$', re.IGNORECASE)


def material_type(part_id: str) -> str:
    p = part_id.strip().lower()
    if p.startswith('m'):
        return 'metal'
    if p.startswith('t'):
        return 'plastic'
    return 'unknown'


def parse_filename(path: str) -> Optional[dict]:
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _FILENAME_RE.match(stem)
    if not m:
        return None
    part = m.group(1)
    return {
        'part':          part,
        'material_type': material_type(part),
        'rpm_category':  int(m.group(2)),
        'trial':         int(m.group(3)),
        'source_file':   os.path.basename(path),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def discover_files(cfg: Config, part: str, rpm: int) -> list[str]:
    pattern = cfg.filename_pattern.format(part=part, rpm=rpm)
    paths   = sorted(glob.glob(os.path.join(cfg.base_path, pattern)))
    if not paths and os.path.isdir(cfg.base_path):
        stem = pattern.lower().replace('-*.csv', '').replace('*', '')
        paths = sorted(
            os.path.join(cfg.base_path, f)
            for f in os.listdir(cfg.base_path)
            if stem in f.lower() and f.lower().endswith('.csv')
        )
    return paths


def rpm_from_angle(time: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    """Derive instantaneous RPM from an encoder angle (degrees) vs time.

    RPM = (dθ/dt) [deg/s] / 360 [rev/deg] * 60 [s/min].  Used only when a speed
    reference is available as angle but not as an explicit RPM channel, so that
    downstream stages (which key on RPM) work unchanged.
    """
    time = np.asarray(time, dtype=float)
    ang  = np.asarray(angle_deg, dtype=float)
    dtheta = np.gradient(ang, time)
    return dtheta / 360.0 * 60.0


def load_file(path: str, cfg: Config) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)

    if cfg.time_col not in df.columns:
        raise KeyError(f"missing time column '{cfg.time_col}' in {path}")

    have_rpm   = cfg.rpm_col in df.columns
    have_angle = cfg.angle_col in df.columns
    # A speed reference is mandatory: either an explicit RPM channel or an
    # encoder angle we can differentiate into RPM.  (Continuous mode lets you
    # supply just one of them; reciprocating data always has RPM.)
    if not have_rpm and not have_angle:
        raise KeyError(
            f"need a speed reference in {path}: neither RPM column "
            f"'{cfg.rpm_col}' nor angle column '{cfg.angle_col}' is present"
        )

    subset = [cfg.time_col] + ([cfg.rpm_col] if have_rpm else [cfg.angle_col])
    df = df.dropna(subset=subset).reset_index(drop=True)
    time = df[cfg.time_col].values.astype(float)

    if have_rpm:
        rpm = df[cfg.rpm_col].values.astype(float)
    else:
        # Synthesize RPM from the angle channel and persist it under rpm_col so
        # the saved parquet carries the column every later stage reads.
        rpm = rpm_from_angle(time, df[cfg.angle_col].values.astype(float))
        df[cfg.rpm_col] = rpm
        print(f"    (no RPM column — derived RPM from angle '{cfg.angle_col}')")

    print(f"    loaded {len(df):,} rows, {len(df.columns)} cols  <- {os.path.basename(path)}")
    return df, time, rpm


def estimate_fs(time: np.ndarray) -> float:
    # Guard degenerate inputs: <2 samples or a non-positive/NaN time step would
    # otherwise yield NaN/inf (or a divide-by-zero) that silently poisons fs.
    if np.asarray(time).size < 2:
        raise ValueError("need >= 2 time samples to estimate sample rate")
    dt = np.median(np.diff(time))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("non-positive / non-finite time step — check time column")
    return 1.0 / dt


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def smooth_rpm(rpm: np.ndarray, fs: float, smooth_ms: float) -> np.ndarray:
    n_pts = max(3, int(round(smooth_ms * 1e-3 * fs)))
    return uniform_filter1d(np.abs(rpm), size=n_pts)


def find_cores(rpm_abs_smooth: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    above   = (rpm_abs_smooth > threshold).astype(int)
    padded  = np.concatenate([[0], above, [0]])
    changes = np.diff(padded)
    starts  = np.where(changes == 1)[0]
    ends    = np.where(changes == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def walk_to_zero(rpm_abs_smooth: np.ndarray,
                 core_start: int, core_end: int,
                 zero_threshold: float) -> tuple[int, int]:
    n = len(rpm_abs_smooth)

    seg_start = 0
    for i in range(core_start, -1, -1):
        if rpm_abs_smooth[i] < zero_threshold:
            seg_start = i
            break

    seg_end = n - 1
    for i in range(core_end, n):
        if rpm_abs_smooth[i] < zero_threshold:
            seg_end = i
            break

    return seg_start, seg_end


# ═══════════════════════════════════════════════════════════════════════════════
#  DIRECTION-ALTERNATION ENFORCEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def apply_alternation_constraint(segments: list[dict],
                                  max_merge_gap_s: float,
                                  unresolved_action: str) -> tuple[list[dict], dict]:
    """
    Enforce POS/NEG/POS/NEG alternation.

    Pass 1: merge consecutive same-direction segments whose gap is below
            max_merge_gap_s (fixes mid-ramp threshold dips).
    Pass 2: for remaining same-direction consecutives:
              'warn'        -> keep both, record in stats
              'drop_weaker' -> keep the one with higher peak RPM, drop the other

    Returns (cleaned_segments, stats_dict).
    """
    stats = {'n_merged': 0, 'n_dropped': 0, 'unresolved': []}
    if not segments:
        return [], stats

    # ── Pass 1: merge close same-direction consecutives ─────────────────────
    merged: list[dict] = [dict(segments[0])]
    for nxt in segments[1:]:
        prev = merged[-1]
        if nxt['direction'] == prev['direction']:
            gap = nxt['t0'] - prev['t1']
            if gap < max_merge_gap_s:
                prev['i1']         = nxt['i1']
                prev['t1']         = nxt['t1']
                prev['duration_s'] = prev['t1'] - prev['t0']
                prev['peak_rpm']   = max(prev['peak_rpm'], nxt['peak_rpm'])
                stats['n_merged'] += 1
                continue
        merged.append(dict(nxt))

    # ── Pass 2: handle remaining same-direction consecutives ────────────────
    if unresolved_action == 'drop_weaker':
        cleaned: list[dict] = [dict(merged[0])]
        for nxt in merged[1:]:
            prev = cleaned[-1]
            if nxt['direction'] == prev['direction']:
                stats['unresolved'].append({
                    't_prev':    prev['t0'],
                    't_next':    nxt['t0'],
                    'direction': nxt['direction'],
                    'gap_s':     nxt['t0'] - prev['t1'],
                    'action':    'dropped_weaker',
                })
                stats['n_dropped'] += 1
                # Replace prev with nxt if nxt is stronger; otherwise keep prev
                if nxt['peak_rpm'] > prev['peak_rpm']:
                    cleaned[-1] = dict(nxt)
            else:
                cleaned.append(dict(nxt))
        return cleaned, stats

    # default: 'warn'
    for i in range(1, len(merged)):
        if merged[i]['direction'] == merged[i - 1]['direction']:
            stats['unresolved'].append({
                't_prev':    merged[i - 1]['t0'],
                't_next':    merged[i]['t0'],
                'direction': merged[i]['direction'],
                'gap_s':     merged[i]['t0'] - merged[i - 1]['t1'],
                'action':    'kept_both',
            })
    return merged, stats


# ═══════════════════════════════════════════════════════════════════════════════
#  SEGMENTATION DRIVER
# ═══════════════════════════════════════════════════════════════════════════════

def segment_file(time: np.ndarray, rpm: np.ndarray,
                 category_rpm: int, cfg: Config) -> tuple[list[dict], dict]:
    fs         = estimate_fs(time)
    rpm_smooth = smooth_rpm(rpm, fs, cfg.smooth_ms)
    threshold  = cfg.detect_threshold_frac * category_rpm

    cores = find_cores(rpm_smooth, threshold)

    # Build raw segments — DO NOT apply min_duration filter yet so that
    # short fragments of a split actuation can be merged first.
    segments: list[dict] = []
    for c_start, c_end in cores:
        i0, i1 = walk_to_zero(rpm_smooth, c_start, c_end, cfg.zero_rpm_threshold)
        core_mid  = (c_start + c_end) // 2
        direction = 'POS' if rpm[core_mid] >= 0 else 'NEG'
        peak_rpm  = float(np.max(np.abs(rpm[i0:i1])))
        segments.append({
            'i0':         i0,
            'i1':         i1,
            't0':         float(time[i0]),
            't1':         float(time[i1]),
            'duration_s': float(time[i1] - time[i0]),
            'peak_rpm':   peak_rpm,
            'direction':  direction,
        })

    # Enforce alternation (merges close same-direction splits)
    if cfg.enforce_alternation:
        segments, alt_stats = apply_alternation_constraint(
            segments, cfg.max_merge_gap_s, cfg.unresolved_action
        )
    else:
        alt_stats = {'n_merged': 0, 'n_dropped': 0, 'unresolved': []}

    # NOW apply min-duration filter
    n_pre = len(segments)
    segments = [s for s in segments if s['duration_s'] >= cfg.min_duration_s]
    alt_stats['n_short_dropped'] = n_pre - len(segments)

    return segments, alt_stats


def segment_continuous(time: np.ndarray, rpm: np.ndarray,
                       cfg: Config) -> tuple[list[dict], dict]:
    """Continuous mode: treat the entire recording as ONE segment.

    A steady rotating-machine signal (e.g. a helicopter vibration file) has no
    strokes to detect, no direction to alternate and no ramp to gate out — so we
    emit a single segment spanning every sample.  Its ``direction`` is 'ALL'.
    The returned shape matches ``segment_file`` so all persistence / downstream
    code is reused unchanged.
    """
    n = len(time)
    if n < 2:
        return [], {'n_merged': 0, 'n_dropped': 0, 'unresolved': [],
                    'n_short_dropped': 0}
    seg = {
        'i0':         0,
        'i1':         n - 1,
        't0':         float(time[0]),
        't1':         float(time[n - 1]),
        'duration_s': float(time[n - 1] - time[0]),
        'peak_rpm':   float(np.max(np.abs(rpm))),
        'direction':  'ALL',
    }
    stats = {'n_merged': 0, 'n_dropped': 0, 'unresolved': [], 'n_short_dropped': 0}
    return [seg], stats


# ═══════════════════════════════════════════════════════════════════════════════
#  PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════

def save_segmented_data(df: pd.DataFrame,
                        segments: list[dict],
                        meta: dict,
                        out_path: str,
                        cfg: Config) -> None:
    if not segments:
        return

    cols = list(df.columns) if cfg.keep_columns is None else list(cfg.keep_columns)
    cols = [c for c in cols if c in df.columns]

    pieces = []
    for seg_idx, seg in enumerate(segments):
        i0, i1 = seg['i0'], seg['i1']
        sub = df.iloc[i0:i1][cols].copy()
        sub['segment_id'] = seg_idx
        sub['direction']  = seg['direction']
        sub['t_rel']      = sub[cfg.time_col].values - seg['t0']
        pieces.append(sub)

    combined = pd.concat(pieces, ignore_index=True)
    for k in ('part', 'material_type', 'rpm_category', 'trial'):
        combined[k] = meta[k]

    if cfg.save_format == 'parquet':
        combined.to_parquet(out_path, index=False)
    elif cfg.save_format == 'pickle':
        combined.to_pickle(out_path)
    else:
        raise ValueError(f"unknown save_format: {cfg.save_format}")


def save_metadata_json(segments: list[dict], alt_stats: dict,
                       meta: dict, out_path: str) -> None:
    payload = {
        **meta,
        'n_segments':           len(segments),
        'n_pos':                sum(1 for s in segments if s['direction'] == 'POS'),
        'n_neg':                sum(1 for s in segments if s['direction'] == 'NEG'),
        'alternation_stats':    alt_stats,
        'segments': [
            {
                'segment_id': i,
                'i0':         s['i0'],
                'i1':         s['i1'],
                't0':         s['t0'],
                't1':         s['t1'],
                'duration_s': s['duration_s'],
                'peak_rpm':   s['peak_rpm'],
                'direction':  s['direction'],
            }
            for i, s in enumerate(segments)
        ],
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)


def build_summary_row(seg: dict, seg_id: int, meta: dict, data_path: str) -> dict:
    return {
        'part':          meta['part'],
        'material_type': meta['material_type'],
        'rpm_category':  meta['rpm_category'],
        'trial':         meta['trial'],
        'source_file':   meta['source_file'],
        'data_path':     data_path,
        'segment_id':    seg_id,
        'direction':     seg['direction'],
        'i0':            seg['i0'],
        'i1':            seg['i1'],
        't0':            seg['t0'],
        't1':            seg['t1'],
        'duration_s':    seg['duration_s'],
        'peak_rpm':      seg['peak_rpm'],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

_DIR_COLOR = {'POS': '#1f77b4', 'NEG': '#d62728', 'ALL': '#1f77b4'}


def plot_segmentation(time: np.ndarray, rpm: np.ndarray,
                      segments: list[dict], alt_stats: dict,
                      meta: dict, filepath: str,
                      cfg: Config) -> None:
    n_seg   = len(segments)
    step    = max(1, len(time) // cfg.overview_max_points)
    part    = meta['part']
    rpm_cat = meta['rpm_category']
    mtype   = meta['material_type']
    trial   = meta['trial']

    fig, (ax_over, ax_zoom) = plt.subplots(
        2, 1, figsize=(18, 9),
        gridspec_kw={'height_ratios': [2, 1.2]}
    )

    # ── TOP: full overview ──────────────────────────────────────────────────
    ax_over.plot(time[::step], rpm[::step],
                 color='#444', lw=0.5, zorder=1, label='RPM signal')
    ax_over.axhline(0, color='k', lw=0.6, zorder=2)

    thresh = cfg.detect_threshold_frac * rpm_cat
    ax_over.axhline( thresh, color='grey', lw=0.8, linestyle=':',
                     alpha=0.6, label=f'detect threshold +/-{thresh:.0f} RPM')
    ax_over.axhline(-thresh, color='grey', lw=0.8, linestyle=':', alpha=0.6)
    ax_over.axhline( cfg.zero_rpm_threshold, color='green', lw=0.8,
                     linestyle='--', alpha=0.5,
                     label=f'zero threshold +/-{cfg.zero_rpm_threshold:.0f} RPM')
    ax_over.axhline(-cfg.zero_rpm_threshold, color='green', lw=0.8,
                     linestyle='--', alpha=0.5)

    labeled = set()
    for seg in segments:
        d   = seg['direction']
        lbl = f'{d} actuation' if d not in labeled else None
        ax_over.axvspan(seg['t0'], seg['t1'],
                        alpha=0.20, color=_DIR_COLOR[d], zorder=0, label=lbl)
        labeled.add(d)

    # Mark unresolved same-direction violations with a vertical bar
    for u in alt_stats.get('unresolved', []):
        ax_over.axvline(u['t_next'], color='orange', lw=1.2, linestyle='-.',
                        alpha=0.7, zorder=3)

    n_pos = sum(1 for s in segments if s['direction'] == 'POS')
    n_neg = sum(1 for s in segments if s['direction'] == 'NEG')
    title_extra = (
        f"   POS={n_pos} NEG={n_neg}"
        f"   merged={alt_stats.get('n_merged', 0)}"
        f"   dropped={alt_stats.get('n_dropped', 0)}"
        f"   unresolved={len(alt_stats.get('unresolved', []))}"
    )

    ax_over.set_ylabel('RPM', fontsize=11)
    ax_over.set_title(
        f'SEGMENTATION  --  {part} ({mtype})  {rpm_cat} RPM  trial {trial}   '
        f'{n_seg} actuations{title_extra}',
        fontsize=12, fontweight='bold'
    )
    ax_over.legend(fontsize=8, loc='upper right', ncol=2)
    ax_over.grid(True, linestyle='--', alpha=0.25)

    # ── BOTTOM: overlay first n_overlay segments ────────────────────────────
    n_show = min(cfg.n_overlay, n_seg)
    ax_zoom.axhline(0, color='k', lw=0.6)
    ax_zoom.axhline( cfg.zero_rpm_threshold, color='green', lw=0.8,
                     linestyle='--', alpha=0.5)
    ax_zoom.axhline(-cfg.zero_rpm_threshold, color='green', lw=0.8,
                     linestyle='--', alpha=0.5)

    cmap = plt.cm.tab10
    for k, seg in enumerate(segments[:n_show]):
        i0, i1 = seg['i0'], seg['i1']
        t_rel  = time[i0:i1] - time[i0]
        color  = cmap(k / max(n_show - 1, 1))
        ax_zoom.plot(t_rel, rpm[i0:i1], lw=1.1, color=color,
                     label=f'seg {k}  ({seg["direction"]}, '
                           f'{seg["duration_s"]:.2f}s, '
                           f'peak={seg["peak_rpm"]:.0f} RPM)')

    ax_zoom.set_xlabel('Time from actuation start (s)', fontsize=11)
    ax_zoom.set_ylabel('RPM', fontsize=11)
    ax_zoom.set_title(
        f'First {n_show} actuations overlaid  '
        f'(verify: ramp-up, plateau, ramp-down, return to zero)',
        fontsize=11
    )
    ax_zoom.legend(fontsize=9, loc='upper right')
    ax_zoom.grid(True, linestyle='--', alpha=0.25)

    fig.tight_layout()
    fig.savefig(filepath, dpi=130)
    if cfg.show_plots:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_continuous(cfg: Config, plots_dir: str, data_dir: str,
                   data_ext: str) -> dict:
    """Continuous mode driver: one steady signal → one whole-signal segment."""
    path = cfg.single_file
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"continuous mode needs a valid single_file; got {path!r}"
        )

    print(f"\nMode:         continuous (single whole-signal segment)")
    print(f"File:         {path}")
    print(f"Output:       {os.path.abspath(cfg.output_dir)}\n")

    df, time, rpm = load_file(path, cfg)

    # Report the RPM RANGE (the speed can vary a lot in a continuous recording;
    # order tracking follows the instantaneous speed, so no single nominal value
    # is needed). The median is kept only as a label / grouping key. An explicit
    # cfg.nominal_rpm still overrides the label if a caller sets one.
    rpm_abs = np.abs(rpm)
    nz = rpm_abs[rpm_abs > cfg.zero_rpm_threshold]
    if nz.size:
        rpm_min, rpm_med, rpm_max = (float(np.min(nz)), float(np.median(nz)),
                                     float(np.max(nz)))
    else:
        rpm_min = rpm_med = rpm_max = float(np.max(rpm_abs)) if rpm_abs.size else 0.0
    label_rpm = float(cfg.nominal_rpm) if cfg.nominal_rpm is not None else rpm_med
    print(f"    RPM range:   {rpm_min:.0f}–{rpm_max:.0f} "
          f"(median {rpm_med:.0f}){'' if cfg.nominal_rpm is None else ' [label overridden]'}")

    meta = {
        'part':          cfg.signal_label,
        'material_type': 'unknown',
        'rpm_category':  int(round(label_rpm)),
        'trial':         1,
        'source_file':   os.path.basename(path),
        'rpm_min':       round(rpm_min, 1),
        'rpm_median':    round(rpm_med, 1),
        'rpm_max':       round(rpm_max, 1),
    }

    segs, alt_stats = segment_continuous(time, rpm, cfg)
    if not segs:
        print("    SKIP: signal too short to segment")
        return {}

    print(f"    -> 1 whole-signal segment "
          f"({segs[0]['duration_s']:.2f}s, peak |RPM| {segs[0]['peak_rpm']:.0f})")

    stem = os.path.splitext(os.path.basename(path))[0]

    if cfg.n_overlay > 0 and cfg.render_plots:
        plot_path = os.path.join(
            plots_dir, f"seg_{meta['part']}_{meta['rpm_category']}rpm_{stem}.png")
        plot_segmentation(time, rpm, segs, alt_stats, meta, plot_path, cfg)

    data_path = os.path.join(data_dir, f"{stem}_segments{data_ext}")
    save_segmented_data(df, segs, meta, data_path, cfg)
    json_path = os.path.join(data_dir, f"{stem}_metadata.json")
    save_metadata_json(segs, alt_stats, meta, json_path)

    rel_data_path = os.path.relpath(data_path, cfg.output_dir)
    summary_rows = [build_summary_row(segs[0], 0, meta, rel_data_path)]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(cfg.output_dir, 'summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSummary  -> {summary_path}  (1 segment)")
    print(f"Data     -> {os.path.abspath(data_dir)}\n")
    return {(meta['part'], meta['rpm_category']): segs}


def run(cfg: Config) -> dict:
    plots_dir = os.path.join(cfg.output_dir, cfg.plots_subdir)
    data_dir  = os.path.join(cfg.output_dir, cfg.data_subdir)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir,  exist_ok=True)

    data_ext = '.parquet' if cfg.save_format == 'parquet' else '.pkl'

    if cfg.analysis_mode == 'continuous':
        return run_continuous(cfg, plots_dir, data_dir, data_ext)

    print(f"\nParts:        {cfg.parts}")
    print(f"RPMs:         {cfg.rpms}")
    print(f"Output:       {os.path.abspath(cfg.output_dir)}")
    print(f"Save fmt:     {cfg.save_format}")
    print(f"Alternation:  enforce={cfg.enforce_alternation}  "
          f"merge_gap={cfg.max_merge_gap_s}s  "
          f"unresolved={cfg.unresolved_action}\n")

    all_segments: dict = {}
    summary_rows: list = []
    total_merged = total_dropped = total_unresolved = 0

    for part in cfg.parts:
        mtype = material_type(part)
        print(f"-- {part}  ({mtype}) --")

        for rpm_cat in cfg.rpms:
            paths = discover_files(cfg, part, rpm_cat)
            if not paths:
                print(f"  [{part} {rpm_cat} RPM]  no files found")
                continue

            print(f"  [{part} {rpm_cat} RPM]")
            all_segs_this_combo = []

            for path in paths:
                meta = parse_filename(path)
                if meta is None:
                    print(f"    SKIP (unparseable filename): {os.path.basename(path)}")
                    continue

                df, time, rpm = load_file(path, cfg)
                segs, alt_stats = segment_file(time, rpm, meta['rpm_category'], cfg)

                n_pos = sum(1 for s in segs if s['direction'] == 'POS')
                n_neg = sum(1 for s in segs if s['direction'] == 'NEG')
                print(f"    -> {len(segs)} actuations  (POS={n_pos} NEG={n_neg})")

                if alt_stats['n_merged']:
                    print(f"       merged {alt_stats['n_merged']} same-direction "
                          f"split(s) (gap < {cfg.max_merge_gap_s}s)")
                if alt_stats['n_dropped']:
                    print(f"       dropped {alt_stats['n_dropped']} weaker "
                          f"same-direction segment(s)")
                if alt_stats.get('n_short_dropped'):
                    print(f"       dropped {alt_stats['n_short_dropped']} "
                          f"sub-{cfg.min_duration_s}s segment(s)")
                for u in alt_stats['unresolved']:
                    print(f"       WARN: two {u['direction']} segments at "
                          f"t={u['t_prev']:.2f}s and t={u['t_next']:.2f}s "
                          f"(gap {u['gap_s']:.2f}s) [{u['action']}]")

                total_merged     += alt_stats['n_merged']
                total_dropped    += alt_stats['n_dropped']
                total_unresolved += len(alt_stats['unresolved'])

                stem = os.path.splitext(os.path.basename(path))[0]

                if segs and cfg.n_overlay > 0 and cfg.render_plots:
                    plot_path = os.path.join(
                        plots_dir,
                        f"seg_{meta['part']}_{meta['rpm_category']}rpm_{stem}.png"
                    )
                    plot_segmentation(time, rpm, segs, alt_stats, meta, plot_path, cfg)

                if segs:
                    data_path = os.path.join(data_dir, f"{stem}_segments{data_ext}")
                    save_segmented_data(df, segs, meta, data_path, cfg)

                    json_path = os.path.join(data_dir, f"{stem}_metadata.json")
                    save_metadata_json(segs, alt_stats, meta, json_path)

                    rel_data_path = os.path.relpath(data_path, cfg.output_dir)
                    for seg_id, seg in enumerate(segs):
                        summary_rows.append(
                            build_summary_row(seg, seg_id, meta, rel_data_path)
                        )

                all_segs_this_combo.extend(segs)

            all_segments[(part, rpm_cat)] = all_segs_this_combo
            n_pos = sum(1 for s in all_segs_this_combo if s['direction'] == 'POS')
            n_neg = sum(1 for s in all_segs_this_combo if s['direction'] == 'NEG')
            print(f"  [{part} {rpm_cat} RPM]  total: "
                  f"{len(all_segs_this_combo)} actuations  "
                  f"(POS: {n_pos}  NEG: {n_neg})")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(cfg.output_dir, 'summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary  -> {summary_path}  ({len(summary_df)} segments)")

    print(f"\nAlternation enforcement totals:")
    print(f"  merged:     {total_merged}")
    print(f"  dropped:    {total_dropped}")
    print(f"  unresolved: {total_unresolved}")

    _print_summary(all_segments, cfg)
    print(f"\nPlots    -> {os.path.abspath(plots_dir)}")
    print(f"Data     -> {os.path.abspath(data_dir)}\n")
    return all_segments


def _print_summary(all_segments: dict, cfg: Config) -> None:
    print(f"\n{'='*78}")
    print(f"  SEGMENTATION SUMMARY")
    print(f"{'='*78}")
    print(f"  {'part':<8} {'mat':<8} {'RPM':>5} | {'n_act':>5} | "
          f"{'POS':>4} {'NEG':>4} | {'mean dur (s)':>12} | {'mean peak RPM':>14}")
    print(f"  {'-'*74}")
    for part in cfg.parts:
        mtype = material_type(part)
        for rpm_cat in cfg.rpms:
            segs = all_segments.get((part, rpm_cat), [])
            if not segs:
                print(f"  {part:<8} {mtype:<8} {rpm_cat:>5} | -- no data --")
                continue
            n_pos  = sum(1 for s in segs if s['direction'] == 'POS')
            n_neg  = sum(1 for s in segs if s['direction'] == 'NEG')
            m_dur  = np.mean([s['duration_s'] for s in segs])
            m_peak = np.mean([s['peak_rpm']   for s in segs])
            balance_flag = '  *' if n_pos != n_neg else ''
            print(f"  {part:<8} {mtype:<8} {rpm_cat:>5} | {len(segs):>5} | "
                  f"{n_pos:>4} {n_neg:>4} | {m_dur:>12.3f} | {m_peak:>14.1f}"
                  f"{balance_flag}")
    print(f"  {'-'*74}")
    print(f"  * = POS/NEG imbalance (investigate the per-file metadata.json)")
    print(f"{'='*78}\n")


def _apply_pipeline_config(cfg, _stage='segment'):
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