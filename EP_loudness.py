"""
═══════════════════════════════════════════════════════════════════════════════
 OVERALL LOUDNESS  —  Metal vs Plastic   (reads pre-segmented parquet)
═══════════════════════════════════════════════════════════════════════════════

Reads parquet files produced by EP_segment.py (no double processing).
Computes raw-signal RMS on the accelerometer within each segment, aggregated
by material_type (metal vs plastic) across whatever parts you've tested.

INPUT
-----
  output_seg/segmented_data/*_segments.parquet
  (Each parquet has: Time, RPM, X/Y/Z Accel, segment_id, direction,
                     t_rel, part, material_type, rpm_category, trial)

OUTPUT
------
  output_loudness/
    loudness_strip.png       slideshow plot (POS/NEG panels, metal vs plastic)
    rms_per_segment.csv      one row per actuation with RMS values
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── INPUT — point at EP_segment's output directory ──────────────────────
    segmented_data_dir: str = 'output_seg/segmented_data'

    # ── COLUMN NAMES (must match what EP_segment wrote into the parquet) ────
    time_col: str = 'Time (s)'

    # Accelerometer columns.  Y/Z swap is intentional — physical Y is logged
    # under "Z Accel" and vice versa.  Applies to every file.
    accel_cols: dict = field(default_factory=lambda: {
        'X': 'X Accel (m/s2)',
        'Y': 'Z Accel (m/s2)',   # physical Y <- logger "Z" column
        'Z': 'Y Accel (m/s2)',   # physical Z <- logger "Y" column
    })

    # ── PLOT STYLE ──────────────────────────────────────────────────────────
    # Keyed by material_type — set by EP_segment from the part-id prefix.
    material_colors: dict = field(default_factory=lambda: {
        'metal':   '#1f77b4',   # blue
        'plastic': '#ff7f0e',   # orange
    })

    # ── SIGNAL CONDITIONING ──────────────────────────────────────────────────
    # Remove the per-segment DC offset before computing RMS so a sensor bias
    # can't inflate the level or bias the metal-vs-plastic comparison (R3).
    # Default ON; set False to reproduce the legacy raw-RMS numbers.
    remove_dc: bool = True

    # ── OUTPUT ──────────────────────────────────────────────────────────────
    output_dir: str = 'output_loudness'
    show_plots: bool = False
    render_plots: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
#  RMS COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

# RMS helpers live in nvh_pipeline.common so the loudness scripts share one copy.
# Fall back to local definitions if the package isn't importable (bare standalone).
try:
    from nvh_pipeline.common import per_axis_rms, combined_axes_rms
except Exception:
    def per_axis_rms(signal: np.ndarray, remove_dc: bool = False) -> float:
        """RMS of a signal; remove_dc=True subtracts the mean first (AC RMS)."""
        s = signal - np.mean(signal) if remove_dc else signal
        return float(np.sqrt(np.mean(s ** 2)))

    def combined_axes_rms(signals: list[np.ndarray],
                          remove_dc: bool = False) -> float:
        """Vector RMS across axes: sqrt(mean(x² + y² + z²)); remove_dc subtracts
        each axis mean first."""
        if not signals:
            return float('nan')
        total_sq = np.zeros_like(signals[0], dtype=float)
        for sig in signals:
            s = sig - np.mean(sig) if remove_dc else sig
            total_sq += s ** 2
        return float(np.sqrt(np.mean(total_sq)))


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA COLLECTION  —  read parquet, compute RMS per segment
# ═══════════════════════════════════════════════════════════════════════════════

def load_segments(cfg: Config) -> pd.DataFrame:
    """
    One row per actuation: part, material, rpm_cat, trial, segment_id, direction,
    duration_s, rms_X, rms_Y, rms_Z, rms_combined.

    Uses a single DuckDB GROUP BY over all parquet files — no Python per-segment
    loops.  NaN samples are excluded from AVG automatically by DuckDB (same as
    the old np.isfinite filter).
    """
    from nvh_pipeline import dataset

    parquet_files = dataset.list_segment_files(cfg.segmented_data_dir)
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files under {os.path.abspath(cfg.segmented_data_dir)}.\n"
            f"Run EP_segment.py first."
        )

    print(f"Found {len(parquet_files)} parquet file(s) in "
          f"{os.path.abspath(cfg.segmented_data_dir)}")

    def _q(c: str) -> str:
        return '"' + c.replace('"', '""') + '"'

    # Per-axis RMS.  With remove_dc (default) we use the variance form
    #   sqrt(avg(c^2) - avg(c)^2) = population std = AC RMS,
    # which subtracts the per-segment DC offset inside the single GROUP BY so a
    # sensor bias cannot inflate the level (R3).  GREATEST(...,0) guards tiny
    # negative round-off on a (near-)constant channel.  DuckDB ignores NaN/NULL
    # in AVG, matching the old np.isfinite filter.
    remove_dc = getattr(cfg, 'remove_dc', True)

    def _rms_expr(qcols: list[str]) -> str:
        sq_sum = ' + '.join(f'{c} * {c}' for c in qcols)
        if not remove_dc:
            return f'SQRT(AVG({sq_sum}))'
        mean_sq = ' + '.join(f'AVG({c}) * AVG({c})' for c in qcols)
        return f'SQRT(GREATEST(AVG({sq_sum}) - ({mean_sq}), 0))'

    agg_exprs: dict[str, str] = {}
    for ax, col in cfg.accel_cols.items():
        agg_exprs[f'rms_{ax}'] = _rms_expr([_q(col)])

    # Combined RMS across the three axes (DC removed per axis when remove_dc).
    agg_exprs['rms_combined'] = _rms_expr([_q(c) for c in cfg.accel_cols.values()])

    # Duration: prefer t_rel written by EP_segment, fall back to time span.
    agg_exprs['duration_s'] = (
        f'COALESCE(MAX({_q("t_rel")}), '
        f'MAX({_q(cfg.time_col)}) - MIN({_q(cfg.time_col)}))'
    )

    df_out = dataset.reduce_segments(cfg.segmented_data_dir, agg_exprs)

    if df_out.empty:
        raise RuntimeError("No actuations were loaded — check input parquet files.")

    # Rename to match column names the rest of EP_loudness expects.
    df_out = df_out.rename(columns={
        'material_type': 'material',
        'rpm_category':  'rpm_cat',
    })
    df_out['rpm_cat']    = df_out['rpm_cat'].astype(int)
    df_out['trial']      = df_out['trial'].astype(int)
    df_out['segment_id'] = df_out['segment_id'].astype(int)

    # Drop source_file — downstream code doesn't use it.
    df_out = df_out.drop(columns=['source_file'], errors='ignore')

    print(f"\nLoaded {len(df_out)} total actuations across "
          f"{df_out['material'].nunique()} material(s) and "
          f"{df_out['rpm_cat'].nunique()} RPM categor(ies).")
    return df_out


# ═══════════════════════════════════════════════════════════════════════════════
#  TRIAL-LEVEL AGGREGATION  (R8)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_trial_means(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse segment-level rows to one row per (material, rpm_cat, direction,
    part, trial).

    Segments within a trial are correlated (same specimen, same run); treating
    them as independent inflates effective sample size and underestimates
    variability.  This function produces the statistically correct input for
    material-level comparisons: each trial contributes one value, not N segments.
    """
    trial_df = (
        df.groupby(['material', 'rpm_cat', 'direction', 'part', 'trial'],
                   as_index=False)
        .agg(trial_mean_rms=('rms_combined', 'mean'),
             n_segments=('rms_combined', 'count'))
    )
    return trial_df


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame) -> None:
    materials  = sorted(df['material'].unique())
    trial_df   = compute_trial_means(df)

    print(f"\n{'='*100}")
    print(f"  LOUDNESS SUMMARY  —  statistics are over TRIALS (one value per trial), not segments")
    print(f"{'='*100}")
    header = f"  {'RPM':>5}  {'dir':>4} | "
    for m in materials:
        header += f"{'t':>3} {'segs':>5}  {m:>9}  {'±std':>9} | "
    header += "louder"
    print(header)
    print(f"  {'-'*98}")

    for rpm_cat in sorted(df['rpm_cat'].unique()):
        for dirn in ['POS', 'NEG']:
            means = {}
            any_data = False
            line = f"  {rpm_cat:>5}  {dirn:>4} | "

            for m in materials:
                t = trial_df[(trial_df['material'] == m)
                             & (trial_df['rpm_cat'] == rpm_cat)
                             & (trial_df['direction'] == dirn)]
                if t.empty:
                    line += f"{'—':>3} {'—':>5}  {'—':>9}  {'—':>9} | "
                    means[m] = np.nan
                else:
                    n_t   = len(t)
                    n_seg = int(t['n_segments'].sum())
                    mu    = float(t['trial_mean_rms'].mean())
                    sigma = float(t['trial_mean_rms'].std()) if n_t > 1 else float('nan')
                    std_s = f'{sigma:.4g}' if np.isfinite(sigma) else '—'
                    line += f"{n_t:>3} {n_seg:>5}  {mu:>9.4g}  {std_s:>9} | "
                    means[m] = mu
                    any_data = True

            if not any_data:
                continue
            valid = {m: v for m, v in means.items() if np.isfinite(v)}
            if len(valid) >= 2:
                louder = max(valid, key=valid.get)
                line += louder
            print(line)
    print(f"{'='*100}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PLOT  —  slideshow-ready
# ═══════════════════════════════════════════════════════════════════════════════

def plot_loudness(df: pd.DataFrame, cfg: Config, outdir: str) -> None:
    """
    Two-panel strip plot:
      top    -> POSITIVE direction actuations
      bottom -> NEGATIVE direction actuations
    X = RPM category, Y = combined RMS (log), color = material.

    Dots show individual actuations (within-trial variability).
    Mean line is the mean of PER-TRIAL means (between-trial statistic, R8).
    Ratio annotation is based on trial means and shows n_trials.
    """
    rpms      = sorted(df['rpm_cat'].unique())
    materials = sorted(df['material'].unique())
    x_pos     = {r: i for i, r in enumerate(rpms)}
    n_rpm     = len(rpms)

    trial_df = compute_trial_means(df)

    if len(materials) == 1:
        mat_offset = {materials[0]: 0.0}
    else:
        spread = 0.20
        mat_offset = {m: -spread + 2 * spread * i / (len(materials) - 1)
                      for i, m in enumerate(materials)}

    jitter_w = 0.09
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(
        2, 1, figsize=(max(13, n_rpm * 2.0), 10),
        sharex=True, gridspec_kw={'hspace': 0.14}
    )

    for ax, direction in zip(axes, ['POS', 'NEG']):
        # Pre-compute trial means for ratio annotations
        trial_means_by_mat: dict[str, pd.DataFrame] = {}
        for mat in materials:
            trial_means_by_mat[mat] = trial_df[
                (trial_df['material'] == mat) & (trial_df['direction'] == direction)
            ]

        for mat in materials:
            color   = cfg.material_colors.get(mat, '#888888')
            sub_seg = df[(df['material'] == mat) & (df['direction'] == direction)]
            sub_t   = trial_means_by_mat[mat]
            if sub_seg.empty:
                continue

            # Dots: one per actuation (segment), shows within-trial spread
            xs_all, ys_all = [], []
            for rpm in rpms:
                grp = sub_seg[sub_seg['rpm_cat'] == rpm]
                if grp.empty:
                    continue
                x_base = x_pos[rpm] + mat_offset[mat]
                jitter = rng.uniform(-jitter_w, jitter_w, len(grp))
                xs_all.extend(x_base + jitter)
                ys_all.extend(grp['rms_combined'].values)

            n_trials_total = sub_t[['part', 'trial']].drop_duplicates().shape[0]
            ax.scatter(xs_all, ys_all, color=color, s=28,
                       alpha=0.55, linewidths=0, zorder=3,
                       label=f'{mat}  ({n_trials_total} trials, {len(sub_seg)} actuations)')

            # Mean line: mean of per-trial means (between-trial statistic)
            mean_x, mean_y = [], []
            for rpm in rpms:
                tg = sub_t[sub_t['rpm_cat'] == rpm]
                if tg.empty:
                    continue
                mean_x.append(x_pos[rpm] + mat_offset[mat])
                mean_y.append(float(tg['trial_mean_rms'].mean()))

            if mean_x:
                ax.plot(mean_x, mean_y, color=color, lw=2.5,
                        marker='o', markersize=8,
                        markeredgecolor='white', markeredgewidth=1.0,
                        zorder=5, label=f'{mat} trial mean')

        # Ratio annotation: based on trial means, gated on ≥1 trial per material
        if 'metal' in materials and 'plastic' in materials:
            for rpm in rpms:
                tm_m = trial_means_by_mat['metal'][
                    trial_means_by_mat['metal']['rpm_cat'] == rpm]['trial_mean_rms']
                tm_p = trial_means_by_mat['plastic'][
                    trial_means_by_mat['plastic']['rpm_cat'] == rpm]['trial_mean_rms']
                if tm_m.empty or tm_p.empty:
                    continue
                mu_m, mu_p = float(tm_m.mean()), float(tm_p.mean())
                ratio  = mu_p / mu_m
                louder = 'plastic' if ratio > 1 else 'metal'
                shown  = ratio if ratio >= 1 else 1.0 / ratio
                color  = cfg.material_colors[louder]
                ymax   = max(mu_m, mu_p)
                n_t    = min(len(tm_m), len(tm_p))
                ax.annotate(
                    f'{shown:.2f}×\n{louder}\n(n={n_t}t)',
                    xy=(x_pos[rpm], ymax),
                    xytext=(x_pos[rpm], ymax * 1.45),
                    ha='center', va='bottom', fontsize=8,
                    color=color, fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color='#bbb', lw=0.7),
                )

        title = ('POSITIVE direction  (↑)' if direction == 'POS'
                 else 'NEGATIVE direction  (↓)')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=6)
        ax.set_ylabel('Combined RMS  (m/s²)', fontsize=11)
        ax.set_yscale('log')
        ax.grid(True, axis='y', linestyle='--', alpha=0.30, which='both')
        ax.grid(True, axis='x', linestyle=':',  alpha=0.20)
        ax.set_xlim(-0.55, n_rpm - 0.45)

        h, l = ax.get_legend_handles_labels()
        seen, h2, l2 = set(), [], []
        for hi, li in zip(h, l):
            if li not in seen:
                seen.add(li); h2.append(hi); l2.append(li)
        ax.legend(h2, l2, fontsize=9.5, loc='upper left', framealpha=0.92)

    axes[1].set_xticks(list(x_pos.values()))
    axes[1].set_xticklabels([str(r) for r in rpms], fontsize=11)
    axes[1].set_xlabel('RPM', fontsize=12)

    mat_str = ' vs '.join(materials)
    fig.suptitle(
        f'Vibration Level (RMS)  —  {mat_str}\n'
        'Dots = actuations  |  line = mean of trial means  |  '
        'ratio based on trial means  (R8)',
        fontsize=13.5, fontweight='bold', y=1.005
    )
    fig.tight_layout()

    out_png = os.path.join(outdir, 'loudness_strip.png')
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    print(f"Plot saved -> {out_png}")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def run(cfg: Config) -> pd.DataFrame:
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("\nLoading pre-segmented data...")
    df = load_segments(cfg)
    print_summary(df)

    csv_path = os.path.join(cfg.output_dir, 'rms_per_segment.csv')
    df.to_csv(csv_path, index=False)
    print(f"Per-segment RMS  -> {csv_path}")

    trial_csv = os.path.join(cfg.output_dir, 'rms_trial_means.csv')
    compute_trial_means(df).to_csv(trial_csv, index=False)
    print(f"Trial-level means -> {trial_csv}")

    if cfg.render_plots:
        plot_loudness(df, cfg, cfg.output_dir)
    print(f"All outputs in: {os.path.abspath(cfg.output_dir)}\n")
    return df


def _apply_pipeline_config(cfg, _stage='loudness'):
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