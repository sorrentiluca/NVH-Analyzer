"""
═══════════════════════════════════════════════════════════════════════════════
 LOUDNESS BY SAMPLE  —  Which parts were louder or quieter?
 (variant of EP_loudness.py — same parquet input, sample-level view)
═══════════════════════════════════════════════════════════════════════════════

Reads the same pre-segmented parquets produced by EP_segment.py.
Instead of collapsing to material×RPM, this script keeps the per-part
(and per-trial) granularity so you can see which physical test specimens
were louder or quieter, and whether that varies with RPM.

OUTPUT
------
  output_loudness_samples/
    loudness_by_sample.png          heatmap:  rows = part, cols = RPM,
                                              cell = mean combined RMS
    loudness_by_sample_strip.png    strip plot per part across RPM
    rms_per_sample.csv              mean/median/std per (part, rpm, direction)
    rms_per_segment.csv             same raw per-actuation CSV as original
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── INPUT ────────────────────────────────────────────────────────────────
    segmented_data_dir: str = 'output_seg/segmented_data'

    # ── COLUMN NAMES ─────────────────────────────────────────────────────────
    time_col: str = 'Time (s)'

    # Y/Z swap matches EP_loudness.py — physical Y is logged under "Z Accel"
    accel_cols: dict = field(default_factory=lambda: {
        'X': 'X Accel (m/s2)',
        'Y': 'Z Accel (m/s2)',
        'Z': 'Y Accel (m/s2)',
    })

    # ── PLOT ─────────────────────────────────────────────────────────────────
    # One color per material_type — used in the strip plot
    material_colors: dict = field(default_factory=lambda: {
        'metal':   '#1f77b4',
        'plastic': '#ff7f0e',
    })

    # Heatmap colormap — 'YlOrRd' (low=yellow, high=red) reads intuitively
    heatmap_cmap: str = 'YlOrRd'

    # If True, log-scale the heatmap (useful when RMS spans >1 decade)
    heatmap_log: bool = True

    # Remove the per-segment DC offset before computing RMS (R3).  Matches
    # EP_loudness.py and PipelineConfig.remove_dc.  Default ON.
    remove_dc: bool = True

    # ── OUTPUT ───────────────────────────────────────────────────────────────
    output_dir: str    = 'output_loudness_samples'
    show_plots: bool   = False
    render_plots: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
#  RMS HELPERS  (identical to EP_loudness.py)
# ═══════════════════════════════════════════════════════════════════════════════

# Shared with EP_loudness via nvh_pipeline.common (falls back to local copies).
try:
    from nvh_pipeline.common import per_axis_rms, combined_axes_rms
except ImportError:
    def per_axis_rms(signal: np.ndarray, remove_dc: bool = False) -> float:
        s = signal - np.mean(signal) if remove_dc else signal
        return float(np.sqrt(np.mean(s ** 2)))

    def combined_axes_rms(signals: list[np.ndarray], remove_dc: bool = False) -> float:
        if not signals:
            return float('nan')
        total_sq = np.zeros_like(signals[0], dtype=float)
        for sig in signals:
            s = sig - np.mean(sig) if remove_dc else sig
            total_sq += s ** 2
        return float(np.sqrt(np.mean(total_sq)))


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING  (same logic as EP_loudness.py — one row per actuation)
# ═══════════════════════════════════════════════════════════════════════════════

def load_segments(cfg: Config) -> pd.DataFrame:
    # Read via the shared DuckDB-over-parquet layer (one connection, pushdown).
    from nvh_pipeline import dataset

    parquet_files = dataset.list_segment_files(cfg.segmented_data_dir)
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files under {os.path.abspath(cfg.segmented_data_dir)}.\n"
            "Run EP_segment.py first."
        )

    print(f"Found {len(parquet_files)} parquet file(s) in "
          f"{os.path.abspath(cfg.segmented_data_dir)}")

    required = ('material_type', 'rpm_category', 'part', 'trial',
                'segment_id', 'direction')
    rows = []

    for pq_path in parquet_files:
        try:
            df = dataset.read_file(cfg.segmented_data_dir, pq_path)
        except Exception as e:
            print(f"  SKIP (read error): {os.path.basename(pq_path)}  ({e})")
            continue

        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  SKIP (missing {missing}): {os.path.basename(pq_path)}")
            continue

        part     = df['part'].iloc[0]
        material = df['material_type'].iloc[0]
        rpm_cat  = int(df['rpm_category'].iloc[0])
        trial    = int(df['trial'].iloc[0])

        present = {ax: col for ax, col in cfg.accel_cols.items() if col in df.columns}
        if not present:
            print(f"  SKIP (no accel columns): {os.path.basename(pq_path)}")
            continue

        n_segs = 0
        for seg_id, seg_df in df.groupby('segment_id'):
            direction = seg_df['direction'].iloc[0]
            row = {
                'part':       part,
                'material':   material,
                'rpm_cat':    rpm_cat,
                'trial':      trial,
                'segment_id': int(seg_id),
                'direction':  direction,
            }

            if 't_rel' in seg_df.columns:
                row['duration_s'] = float(seg_df['t_rel'].max())
            elif cfg.time_col in seg_df.columns:
                row['duration_s'] = float(seg_df[cfg.time_col].max()
                                          - seg_df[cfg.time_col].min())
            else:
                row['duration_s'] = float('nan')

            sig_list = []
            for ax, col in present.items():
                sig = seg_df[col].values.astype(float)
                row[f'rms_{ax}'] = per_axis_rms(sig, remove_dc=cfg.remove_dc)
                sig_list.append(sig)
            row['rms_combined'] = combined_axes_rms(sig_list, remove_dc=cfg.remove_dc)

            rows.append(row)
            n_segs += 1

        print(f"  {os.path.basename(pq_path):50s}  "
              f"{material:7s}  {rpm_cat:>4d} RPM  trial={trial}  "
              f"-> {n_segs} actuations")

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        raise RuntimeError("No actuations loaded — check input parquet files.")

    print(f"\nLoaded {len(df_out)} actuations  |  "
          f"{df_out['part'].nunique()} part(s)  |  "
          f"{df_out['material'].nunique()} material(s)  |  "
          f"{df_out['rpm_cat'].nunique()} RPM bin(s)\n")
    return df_out


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-SAMPLE SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def build_sample_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per (part, material, rpm_cat, direction):
      n, mean_rms, median_rms, std_rms, min_rms, max_rms
    Rows sorted by mean_rms descending so loudest samples appear first.
    """
    g = (df.groupby(['part', 'material', 'rpm_cat', 'direction'])['rms_combined']
           .agg(n='count',
                mean_rms='mean',
                median_rms='median',
                std_rms='std',
                min_rms='min',
                max_rms='max')
           .reset_index()
           .sort_values('mean_rms', ascending=False))
    return g


def print_sample_summary(summary: pd.DataFrame) -> None:
    print(f"{'='*96}")
    print(f"  LOUDNESS BY SAMPLE  (combined RMS m/s²,  sorted loudest → quietest)")
    print(f"{'='*96}")
    print(f"  {'part':>12}  {'mat':>7}  {'RPM':>5}  {'dir':>4} | "
          f"{'n':>4}  {'mean':>9}  {'median':>9}  {'std':>9}  "
          f"{'min':>9}  {'max':>9}")
    print(f"  {'-'*94}")
    for _, r in summary.iterrows():
        print(f"  {r['part']:>12}  {r['material']:>7}  {int(r['rpm_cat']):>5}  "
              f"{r['direction']:>4} | "
              f"{int(r['n']):>4}  "
              f"{r['mean_rms']:>9.4g}  "
              f"{r['median_rms']:>9.4g}  "
              f"{r['std_rms']:>9.4g}  "
              f"{r['min_rms']:>9.4g}  "
              f"{r['max_rms']:>9.4g}")
    print(f"{'='*96}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — HEATMAP: part × RPM, cell = mean RMS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(df: pd.DataFrame, cfg: Config, outdir: str,
                 direction: str = 'both') -> None:
    """
    Rows    = parts (sorted by overall loudness)
    Columns = RPM categories
    Cell    = mean combined RMS, averaged over both directions (or filtered)

    One heatmap per direction if direction='split', else averaged.
    """
    directions = ['POS', 'NEG'] if direction == 'split' else [direction]
    n_panels   = len(directions)

    rpms  = sorted(df['rpm_cat'].unique())
    # Order rows by overall mean RMS so loudest part is at the top
    part_order = (df.groupby('part')['rms_combined']
                    .mean()
                    .sort_values(ascending=False)
                    .index.tolist())

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(max(8, len(rpms) * 1.6 + 3) * n_panels, max(4, len(part_order) * 0.7 + 2)),
        squeeze=False
    )

    norm_cls  = mcolors.LogNorm if cfg.heatmap_log else mcolors.Normalize

    for col_i, dirn in enumerate(directions):
        ax = axes[0][col_i]

        if dirn == 'both':
            pivot_df = df
            title_sfx = 'Both directions'
        else:
            pivot_df = df[df['direction'] == dirn]
            title_sfx = f"{'POSITIVE ↑' if dirn == 'POS' else 'NEGATIVE ↓'} direction"

        pivot = (pivot_df.groupby(['part', 'rpm_cat'])['rms_combined']
                         .mean()
                         .unstack(level='rpm_cat')
                         .reindex(index=part_order, columns=rpms))

        # Global vmin/vmax for consistent colour scale across panels
        vmin = np.nanmin(pivot.values)
        vmax = np.nanmax(pivot.values)
        if cfg.heatmap_log:
            vmin = max(vmin, 1e-9)   # log can't handle 0
        norm = norm_cls(vmin=vmin, vmax=vmax)

        im = ax.imshow(pivot.values, aspect='auto',
                       cmap=cfg.heatmap_cmap, norm=norm)

        # Annotate cells
        for ri in range(len(part_order)):
            for ci, rpm in enumerate(rpms):
                val = pivot.at[part_order[ri], rpm]
                if not np.isnan(val):
                    ax.text(ci, ri, f'{val:.3g}',
                            ha='center', va='center',
                            fontsize=7.5, color='black' if val < vmax * 0.6 else 'white')

        ax.set_xticks(range(len(rpms)))
        ax.set_xticklabels([str(r) for r in rpms], fontsize=10)
        ax.set_yticks(range(len(part_order)))
        ax.set_yticklabels(part_order, fontsize=9)
        ax.set_xlabel('RPM', fontsize=11)
        if col_i == 0:
            ax.set_ylabel('Part  (loudest → quietest)', fontsize=11)
        ax.set_title(f'Mean Combined RMS  |  {title_sfx}', fontsize=11, fontweight='bold')

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        scale_lbl = 'log ' if cfg.heatmap_log else ''
        cbar.set_label(f'Combined RMS  ({scale_lbl}m/s²)', fontsize=9)

    fig.suptitle('Vibration Loudness Heatmap by Sample & RPM',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()

    out_png = os.path.join(outdir, 'loudness_by_sample.png')
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    print(f"Heatmap saved  -> {out_png}")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — STRIP PLOT: one line per part, x = RPM, y = mean RMS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_strip_by_sample(df: pd.DataFrame, cfg: Config, outdir: str) -> None:
    """
    Two-panel (POS / NEG).  One line per part, coloured by material.
    Individual actuations shown as semi-transparent dots.
    Parts ordered in the legend loudest → quietest (by overall mean RMS).
    """
    rpms      = sorted(df['rpm_cat'].unique())
    parts     = (df.groupby('part')['rms_combined']
                   .mean()
                   .sort_values(ascending=False)
                   .index.tolist())
    x_pos     = {r: i for i, r in enumerate(rpms)}
    n_rpm     = len(rpms)
    rng       = np.random.default_rng(42)

    # Assign a shade per part within each material group
    def part_colors(parts_list, cfg):
        colors = {}
        material_parts: dict[str, list] = {}
        for p in parts_list:
            mat = df[df['part'] == p]['material'].iloc[0]
            material_parts.setdefault(mat, []).append(p)

        for mat, plist in material_parts.items():
            base = mcolors.to_rgb(cfg.material_colors.get(mat, '#888888'))
            for k, p in enumerate(plist):
                # Darken successive parts so they're distinguishable
                factor = 1.0 - 0.55 * k / max(1, len(plist) - 1)
                colors[p] = tuple(c * factor for c in base)
        return colors

    colors = part_colors(parts, cfg)

    # Slight horizontal stagger so dots from different parts don't stack
    n_parts = len(parts)
    offsets = np.linspace(-0.25, 0.25, n_parts) if n_parts > 1 else [0.0]
    part_offset = dict(zip(parts, offsets))

    fig, axes = plt.subplots(
        2, 1, figsize=(max(13, n_rpm * 2.2), 10),
        sharex=True, gridspec_kw={'hspace': 0.14}
    )

    for ax, direction in zip(axes, ['POS', 'NEG']):
        for part in parts:
            color = colors[part]
            mat   = df[df['part'] == part]['material'].iloc[0]
            sub   = df[(df['part'] == part) & (df['direction'] == direction)]
            if sub.empty:
                continue

            xs_dots, ys_dots, mean_x, mean_y = [], [], [], []
            for rpm in rpms:
                grp = sub[sub['rpm_cat'] == rpm]
                if grp.empty:
                    continue
                x_base  = x_pos[rpm] + part_offset[part]
                jitter  = rng.uniform(-0.04, 0.04, len(grp))
                xs_dots.extend(x_base + jitter)
                ys_dots.extend(grp['rms_combined'].values)
                mean_x.append(x_base)
                mean_y.append(float(grp['rms_combined'].mean()))

            ax.scatter(xs_dots, ys_dots, color=color, s=22,
                       alpha=0.45, linewidths=0, zorder=3)
            if mean_x:
                ax.plot(mean_x, mean_y, color=color, lw=2.2,
                        marker='o', markersize=7,
                        markeredgecolor='white', markeredgewidth=0.8,
                        zorder=5, label=f'{part}  [{mat}]')

        title = ('POSITIVE direction  (↑)' if direction == 'POS'
                 else 'NEGATIVE direction  (↓)')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=6)
        ax.set_ylabel('Combined RMS  (m/s²)', fontsize=11)
        ax.set_yscale('log')
        ax.grid(True, axis='y', linestyle='--', alpha=0.30, which='both')
        ax.grid(True, axis='x', linestyle=':',  alpha=0.20)
        ax.set_xlim(-0.55, n_rpm - 0.45)

        # Legend: deduplicated, parts in the defined (loudness-sorted) order
        h, l = ax.get_legend_handles_labels()
        # Re-sort by part_order
        order_map = {name: i for i, name in enumerate(
            [f'{p}  [{df[df["part"]==p]["material"].iloc[0]}]' for p in parts]
        )}
        paired = sorted(zip(l, h), key=lambda x: order_map.get(x[0], 999))
        l2, h2 = zip(*paired) if paired else ([], [])
        ax.legend(h2, l2, fontsize=9, loc='upper left',
                  framealpha=0.92, title='part  [material]', title_fontsize=8)

    axes[1].set_xticks(list(x_pos.values()))
    axes[1].set_xticklabels([str(r) for r in rpms], fontsize=11)
    axes[1].set_xlabel('RPM', fontsize=12)

    fig.suptitle(
        'Vibration Loudness by Sample & RPM\n'
        'Combined RMS per actuation  |  one dot = one actuation  |  '
        'legend ordered loudest → quietest',
        fontsize=13, fontweight='bold', y=1.005
    )
    fig.tight_layout()

    out_png = os.path.join(outdir, 'loudness_by_sample_strip.png')
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    print(f"Strip plot saved -> {out_png}")
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

    # ── Raw per-actuation CSV ─────────────────────────────────────────────────
    raw_csv = os.path.join(cfg.output_dir, 'rms_per_segment.csv')
    df.to_csv(raw_csv, index=False)
    print(f"Raw per-actuation CSV  -> {raw_csv}")

    # ── Sample summary ────────────────────────────────────────────────────────
    summary = build_sample_summary(df)
    print_sample_summary(summary)

    summary_csv = os.path.join(cfg.output_dir, 'rms_per_sample.csv')
    summary.to_csv(summary_csv, index=False)
    print(f"Per-sample summary CSV -> {summary_csv}\n")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if cfg.render_plots:
        plot_heatmap(df, cfg, cfg.output_dir, direction='both')
        plot_strip_by_sample(df, cfg, cfg.output_dir)

    print(f"\nAll outputs in: {os.path.abspath(cfg.output_dir)}\n")
    return df


def _apply_pipeline_config(cfg, _stage='loudness_by_sample'):
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
