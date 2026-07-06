"""
═══════════════════════════════════════════════════════════════════════════════
 TORQUE vs RPM CLASS  —  Metal vs Plastic Comparison
═══════════════════════════════════════════════════════════════════════════════

Reads all segmented parquet files produced by the segmentation script and
plots mean torque per RPM class, grouped by material type (metal / plastic).

Each RPM class shows two side-by-side bars:
  Blue   -> metal
  Orange -> plastic

Error bars show ±1 standard deviation of per-segment mean torques.
Torque is taken as absolute value before averaging to account for bidirectional
actuation (positive and negative directions cancel otherwise).

USAGE
-----
  python plot_torque_by_rpm.py

  Adjust CONFIG below if your paths or column names differ.

DEPENDENCIES
------------
  pip install pandas numpy matplotlib pyarrow
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

# Directory that contains all *_segments.parquet files (output_seg/segmented_data).
# Normally set from nvh_config.json; this default is for a bare standalone run.
PARQUET_DIR: str = 'output_seg/segmented_data'

# Column names — must match what was saved by the segmentation script
TORQUE_COL:   str = 'Torque (Nm)'
RPM_COL:      str = 'CNT 1/Frequency (RPM)'   # instantaneous speed (for plateau)
RPM_CAT_COL:  str = 'rpm_category'
MATERIAL_COL: str = 'material_type'   # values: 'metal' or 'plastic'
SEGMENT_COL:  str = 'segment_id'      # unique segment identifier within each file

# Steady-state (plateau) gating — analyse only the constant-speed portion of each
# stroke (|RPM| >= PLATEAU_FRAC * rpm_category), excluding ramp-up/ramp-down so the
# mean torque reflects steady drag, not the acceleration transients.  Injected
# from PipelineConfig in main().
PLATEAU_GATING: bool  = True
PLATEAU_FRAC:   float = 0.90

# RPM classes to include, in display order.  None = auto-detect from data.
RPM_ORDER: list[int] | None = [100, 200, 400, 800, 1600, 2300]

# Plot output
OUTPUT_PATH:   str  = 'torque_vs_rpm_metal_vs_plastic.png'
SHOW_PLOT:     bool = True
RENDER_PLOTS:  bool = True

# Visual
COLOR_METAL:   str   = '#2c7bb6'   # blue
COLOR_PLASTIC: str   = '#f46d43'   # orange
BAR_WIDTH:     float = 0.35
FIGSIZE:       tuple = (12, 6)


# ═══════════════════════════════════════════════════════════════════════════════
#  AGGREGATION  (memory-safe: one file at a time)
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_from_parquets(parquet_dir: str) -> pd.DataFrame:
    """
    Aggregate one file at a time to avoid loading everything into memory.

    Strategy:
      1. For each file, compute one mean |torque| per segment_id.
      2. Collect all per-segment means across all files.
      3. Final mean and std are computed across those segment means —
         so the error bars reflect actuation-to-actuation consistency,
         not sample-level noise.

    Returns one row per (rpm_category, material_type).
    """
    from nvh_pipeline import dataset

    paths = dataset.list_segment_files(parquet_dir)
    if not paths:
        raise FileNotFoundError(
            f"No *_segments.parquet files found in: {os.path.abspath(parquet_dir)}"
        )

    print(f"Found {len(paths)} parquet file(s) in {os.path.abspath(parquet_dir)}")

    # Each entry is one segment mean — these are what std is computed across
    segment_means: list[dict] = []

    cols = [TORQUE_COL, RPM_CAT_COL, MATERIAL_COL, SEGMENT_COL]
    if PLATEAU_GATING:
        cols.append(RPM_COL)
    for p in paths:
        df = dataset.read_file(parquet_dir, p, columns=cols)
        df = df.dropna(subset=[TORQUE_COL])

        # Steady-state gating: keep only the plateau (|RPM| >= frac*category).
        # Per-segment fallback — if a segment never reaches the plateau, keep its
        # full data rather than dropping the actuation entirely.
        if PLATEAU_GATING and RPM_COL in df.columns and len(df):
            plateau = (df[RPM_COL].abs()
                       >= PLATEAU_FRAC * df[RPM_CAT_COL].astype(float))
            seg_has_plateau = plateau.groupby(df[SEGMENT_COL]).transform('any')
            df = df[plateau | ~seg_has_plateau]

        # Absolute value before averaging — bidirectional actuation gives
        # signed torque; averaging signed values would cause cancellation
        df[TORQUE_COL] = df[TORQUE_COL].abs()

        # One mean per segment within this file
        seg_agg = (
            df.groupby([SEGMENT_COL, RPM_CAT_COL, MATERIAL_COL])[TORQUE_COL]
            .mean()
            .reset_index()
            .rename(columns={TORQUE_COL: 'seg_mean_torque'})
        )

        # Extend with the per-segment records without iterrows overhead.
        segment_means.extend(
            seg_agg[[RPM_CAT_COL, MATERIAL_COL, 'seg_mean_torque']].to_dict('records')
        )

        n_segs = len(seg_agg)
        print(f"  {n_segs} segments  <- {os.path.basename(p)}")
        del df, seg_agg

    # Compute mean and std across all segment means
    all_seg_means = pd.DataFrame(segment_means)
    agg = (
        all_seg_means
        .groupby([RPM_CAT_COL, MATERIAL_COL])['seg_mean_torque']
        .agg(
            mean_torque='mean',
            std_torque='std',    # std across segment means
            n_segments='count',
        )
        .reset_index()
    )

    print(f"\nTotal segments across all files: {agg['n_segments'].sum():,}\n")
    return agg[[RPM_CAT_COL, MATERIAL_COL, 'mean_torque', 'std_torque', 'n_segments']]


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot(agg: pd.DataFrame, output_path: str, show: bool = True) -> None:
    # Determine RPM order
    rpm_classes = RPM_ORDER if RPM_ORDER is not None else sorted(agg[RPM_CAT_COL].unique())
    rpm_classes = [r for r in rpm_classes if r in agg[RPM_CAT_COL].values]

    x = np.arange(len(rpm_classes))

    def get_stats(material: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
        means, stds, ns = [], [], []
        for rpm in rpm_classes:
            row = agg[(agg[RPM_CAT_COL] == rpm) & (agg[MATERIAL_COL] == material)]
            if row.empty:
                means.append(np.nan)
                stds.append(0.0)
                ns.append(0)
            else:
                means.append(float(row['mean_torque'].iloc[0]))
                stds.append(float(row['std_torque'].fillna(0).iloc[0]))
                ns.append(int(row['n_segments'].iloc[0]))
        return np.array(means), np.array(stds), ns

    metal_mean,   metal_std,   metal_n   = get_stats('metal')
    plastic_mean, plastic_std, plastic_n = get_stats('plastic')

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor('#f7f7f7')
    ax.set_facecolor('#f7f7f7')

    offset = BAR_WIDTH / 2

    bars_metal = ax.bar(
        x - offset, metal_mean, BAR_WIDTH,
        yerr=metal_std, capsize=4,
        color=COLOR_METAL, label='Metal',
        error_kw=dict(elinewidth=1.2, ecolor='#1a4f7a', capthick=1.2),
        zorder=3,
    )
    bars_plastic = ax.bar(
        x + offset, plastic_mean, BAR_WIDTH,
        yerr=plastic_std, capsize=4,
        color=COLOR_PLASTIC, label='Plastic',
        error_kw=dict(elinewidth=1.2, ecolor='#a33b1f', capthick=1.2),
        zorder=3,
    )

    # Value labels on each bar
    def label_bars(bars: plt.BarContainer, means: np.ndarray) -> None:
        y_scale = max(np.nanmax(metal_mean), np.nanmax(plastic_mean))
        for bar, val in zip(bars, means):
            if np.isnan(val):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * y_scale,
                f'{val:.2f}',
                ha='center', va='bottom',
                fontsize=8, color='#333333', fontweight='bold',
            )

    label_bars(bars_metal,   metal_mean)
    label_bars(bars_plastic, plastic_mean)

    # ── Axes formatting ──────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r} RPM' for r in rpm_classes], fontsize=11)
    ax.set_xlabel('RPM Class', fontsize=12, labelpad=8)
    ax.set_ylabel(f'Mean |{TORQUE_COL}|', fontsize=12, labelpad=8)
    ax.set_title(
        'Average Torque by RPM Class — Metal vs Plastic\n'
        'Error bars: ±1 SD across segment means',
        fontsize=13, fontweight='bold', pad=14,
    )

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis='y', which='major', linestyle='--', alpha=0.4, zorder=0)
    ax.grid(axis='y', which='minor', linestyle=':', alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#cccccc')

    ax.tick_params(axis='both', which='both', length=0)

    # Segment count annotation below each bar
    for i, rpm in enumerate(rpm_classes):
        for material, xoff, color, ns in [
            ('metal',   -offset, COLOR_METAL,   metal_n),
            ('plastic', +offset, COLOR_PLASTIC, plastic_n),
        ]:
            n = ns[i]
            ax.text(
                i + xoff, -0.04,
                f'n={n}',
                ha='center', va='top',
                fontsize=7, color=color, alpha=0.75,
                transform=ax.get_xaxis_transform(),
            )

    ax.legend(
        fontsize=11, framealpha=0.9, edgecolor='#cccccc',
        loc='upper left', frameon=True,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved -> {os.path.abspath(output_path)}")

    if show:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    global PARQUET_DIR, OUTPUT_PATH, RPM_ORDER, SHOW_PLOT, TORQUE_COL, RENDER_PLOTS
    global RPM_COL, PLATEAU_GATING, PLATEAU_FRAC
    # ImportError = nvh_pipeline genuinely absent (bare standalone run): tolerate
    # and keep the module-level defaults.  Any other error below is a real
    # misconfiguration and must surface, not silently fall back to defaults.
    try:
        from nvh_pipeline.config import active, seg_data_dir, stage_out
    except ImportError:
        active = None
    if active is not None:
        p = active()
        PARQUET_DIR  = seg_data_dir(p)
        OUTPUT_PATH  = os.path.join(stage_out(p, 'output_torque'),
                                    'torque_vs_rpm_metal_vs_plastic.png')
        RPM_ORDER    = list(p.rpms)
        SHOW_PLOT    = p.show_plots
        TORQUE_COL   = p.torque_col
        RENDER_PLOTS = p.render_plots
        RPM_COL      = p.rpm_col
        PLATEAU_GATING = getattr(p, 'plateau_gating', PLATEAU_GATING)
        PLATEAU_FRAC   = getattr(p, 'plateau_frac', PLATEAU_FRAC)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    agg = aggregate_from_parquets(PARQUET_DIR)

    print("Aggregated stats:")
    print(agg.to_string(index=False))
    print()

    # Write pre-aggregated CSV so viz.py can load it without re-scanning parquets.
    agg_csv = os.path.join(os.path.dirname(OUTPUT_PATH), "torque_agg.csv")
    agg.rename(columns={"mean_torque": "mean_torque_nm",
                        "std_torque": "std_torque_nm"}).to_csv(agg_csv, index=False)
    print(f"Aggregate CSV saved -> {os.path.abspath(agg_csv)}")

    if RENDER_PLOTS:
        plot(agg, OUTPUT_PATH, show=SHOW_PLOT)


if __name__ == '__main__':
    main()