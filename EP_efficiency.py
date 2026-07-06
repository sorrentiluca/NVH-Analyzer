
"""
═══════════════════════════════════════════════════════════════════════════════
 DRAG / SPECIFIC-ENERGY PROXY  —  Metal vs Plastic Comparison  (no-load)
═══════════════════════════════════════════════════════════════════════════════

Reads all segmented parquet files produced by the segmentation script and
computes a per-segment specific-energy (inverse-drag) proxy:

  OUTPUT (rev)  = Δrevolutions
  INPUT  (J)    = ∫ |torque(t)| × ω(t) dt       [J]  (numerical trapezoid)

  METRIC = revolutions per joule = Δrev / ∫|τ|ω dt   [rev / J]

  Higher = less energy spent per revolution (lower drag).

INTERPRET WITH CARE — THIS IS NOT A TRUE MECHANICAL EFFICIENCY
--------------------------------------------------------------
  • These are NO-LOAD runs: there is no output force, so the "output" is a bare
    revolution count, not delivered work.  The metric is therefore a relative
    DRAG indicator, not η = work_out / work_in.
  • Torque is RECTIFIED (|τ|).  During decel / holding the drive may be braking
    or back-driven; |τ|·ω still counts that as energy *in*, which inflates the
    denominator.  Metal and plastic differ in inertia / decel torque, so the
    rectification bias is not guaranteed to cancel in the comparison.
  • Use it to rank relative drag between specimens at a given RPM, not to quote
    an absolute efficiency number.
 
WHAT THE INTEGRAL DOES (and does not) CAPTURE
---------------------------------------------
  - Instantaneous power = torque × angular velocity.  Integrating |τ|·ω over
    time gives the energy magnitude moved through the drive regardless of ramp
    shape or duration, with no steady-state assumption.
  - A torque spike at low RPM contributes little energy (low ω); the same spike
    at high RPM contributes proportionally more — handled correctly by the ∫.
  - Output work (F × lead × Δrev) has F and lead constant across all tests, so
    they cancel in any relative comparison; Δrevolutions is the output proxy and
    normalizes for stroke-length differences between segments.
  - CAVEAT: because torque is rectified and there is no output load, the
    denominator counts braking / back-driven phases as energy *in*.  Treat the
    metric as a relative drag indicator between specimens at a given RPM, not as
    an absolute mechanical efficiency.
 
USAGE
-----
  python plot_efficiency.py
 
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
 
# Directory containing all *_segments.parquet files.
# Normally set from nvh_config.json; this default is for a bare standalone run.
PARQUET_DIR: str = 'output_seg/segmented_data'
 
# Column names — must match what was saved by the segmentation script
TORQUE_COL:   str = 'Torque (Nm)'
RPM_COL:      str = 'CNT 1/Frequency (RPM)'
ANGLE_COL:    str = 'CNT 1/Angle (Degrees)'
TIME_COL:     str = 'Time (s)'
RPM_CAT_COL:  str = 'rpm_category'
MATERIAL_COL: str = 'material_type'
SEGMENT_COL:  str = 'segment_id'
 
# RPM classes to include, in display order.  None = auto-detect from data.
RPM_ORDER: list[int] | None = [100, 200, 400, 800, 1600, 2300]
 
# Minimum revolutions for a segment to be considered valid.
# Guards against near-zero Δrev in the denominator.
MIN_REVOLUTIONS: float = 0.5
 
# Minimum input energy [J] — guards against near-zero energy in denominator.
MIN_INPUT_ENERGY: float = 1e-6

# Steady-state (plateau) gating — integrate power and count revolutions over only
# the constant-speed plateau (|RPM| >= PLATEAU_FRAC * rpm_category), excluding the
# acceleration/deceleration ramps.  Injected from PipelineConfig in main().
PLATEAU_GATING: bool  = True
PLATEAU_FRAC:   float = 0.90
 
# Plot output
OUTPUT_PATH:   str  = 'efficiency_proxy_metal_vs_plastic.png'
SHOW_PLOT:     bool = True
RENDER_PLOTS:  bool = True
 
# Visual
COLOR_METAL:   str   = '#2c7bb6'
COLOR_PLASTIC: str   = '#f46d43'
BAR_WIDTH:     float = 0.35
FIGSIZE:       tuple = (12, 6)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  PER-SEGMENT METRIC
# ═══════════════════════════════════════════════════════════════════════════════
 
def compute_segment_metric(seg_df: pd.DataFrame) -> float | None:
    """
    Compute output revolutions per input joule for one segment.
 
      ω(t)          = |RPM| × 2π / 60          [rad/s]
      P(t)          = |torque(t)| × ω(t)        [W]
      input_energy  = trapz(P, t)               [J]
      Δrevolutions  = |Δangle| / 360
      metric        = Δrev / input_energy       [rev / J]
 
    Returns None if the segment is invalid (too few revolutions, bad data).
    """
    time   = seg_df[TIME_COL].values.astype(float)
    torque = np.abs(seg_df[TORQUE_COL].values.astype(float))
    rpm    = np.abs(seg_df[RPM_COL].values.astype(float))
    angle  = seg_df[ANGLE_COL].values.astype(float)

    if len(time) < 2:
        return None

    # Steady-state gating: restrict to the constant-speed plateau so the metric
    # reflects steady drag, not the ramp transients.  Per-segment fallback — if
    # the plateau is too short (<2 samples) keep the full stroke.
    if PLATEAU_GATING and RPM_CAT_COL in seg_df.columns and len(seg_df):
        cat = float(seg_df[RPM_CAT_COL].iloc[0])
        if cat > 0:
            m = rpm >= PLATEAU_FRAC * cat
            if int(m.sum()) >= 2:
                time, torque, rpm, angle = time[m], torque[m], rpm[m], angle[m]

    # Angular velocity in rad/s
    omega = rpm * (2.0 * np.pi / 60.0)
 
    # Instantaneous power [W]
    power = torque * omega
 
    # Input energy via trapezoidal integration [J]
    # np.trapezoid (NumPy >= 2.0) replaced np.trapz; fall back for older NumPy.
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    input_energy = _trapz(power, time)
 
    # Output proxy: total revolutions completed
    delta_rev = np.abs(angle[-1] - angle[0]) / 360.0
 
    if delta_rev < MIN_REVOLUTIONS:
        return None
 
    if input_energy < MIN_INPUT_ENERGY:
        return None
 
    # Output / Input: revolutions delivered per joule consumed
    return float(delta_rev / input_energy)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  AGGREGATION  (memory-safe: one file at a time)
# ═══════════════════════════════════════════════════════════════════════════════
 
def aggregate_from_parquets(parquet_dir: str) -> pd.DataFrame:
    """
    Process one parquet file at a time.
    Computes the efficiency metric for every segment, collects results,
    then aggregates to mean ± std per (rpm_category, material_type).
    """
    from nvh_pipeline import dataset

    paths = dataset.list_segment_files(parquet_dir)
    if not paths:
        raise FileNotFoundError(
            f"No *_segments.parquet files found in: {os.path.abspath(parquet_dir)}"
        )

    print(f"Found {len(paths)} parquet file(s) in {os.path.abspath(parquet_dir)}")

    required_cols = [
        TIME_COL, TORQUE_COL, RPM_COL, ANGLE_COL,
        RPM_CAT_COL, MATERIAL_COL, SEGMENT_COL,
    ]

    segment_records: list[dict] = []

    for p in paths:
        df = dataset.read_file(parquet_dir, p, columns=required_cols)
        df = df.dropna(subset=[TIME_COL, TORQUE_COL, RPM_COL, ANGLE_COL])
 
        n_valid = 0
        n_dropped = 0
        for (seg_id, rpm_cat, material), seg_df in df.groupby(
            [SEGMENT_COL, RPM_CAT_COL, MATERIAL_COL]
        ):
            metric = compute_segment_metric(seg_df)
            if metric is None:
                n_dropped += 1
                continue
            segment_records.append({
                RPM_CAT_COL:  rpm_cat,
                MATERIAL_COL: material,
                'metric':     metric,
            })
            n_valid += 1
 
        print(
            f"  {n_valid} segments ok, {n_dropped} dropped  "
            f"<- {os.path.basename(p)}"
        )
        del df
 
    if not segment_records:
        raise RuntimeError(
            "No valid segments found. Check column names and MIN_REVOLUTIONS."
        )

    all_segs = pd.DataFrame(segment_records)

    # Trial-level aggregation (R8): average segments within each trial first so
    # correlated within-trial segments don't inflate the effective sample size.
    # Each (part, trial, rpm_category, material) combination counts as one trial.
    trial_col = 'trial' if 'trial' in all_segs.columns else None
    part_col  = 'part'  if 'part'  in all_segs.columns else None
    group_keys = [RPM_CAT_COL, MATERIAL_COL]
    if part_col  and part_col  in all_segs.columns: group_keys.append(part_col)
    if trial_col and trial_col in all_segs.columns: group_keys.append(trial_col)

    trial_means = (
        all_segs
        .groupby(group_keys)['metric']
        .mean()
        .reset_index()
        .rename(columns={'metric': 'trial_mean_metric'})
    )

    agg = (
        trial_means
        .groupby([RPM_CAT_COL, MATERIAL_COL])['trial_mean_metric']
        .agg(
            mean_metric='mean',
            std_metric='std',
            n_trials='count',
        )
        .reset_index()
    )
    # Keep n_segments for reference
    seg_counts = (
        all_segs.groupby([RPM_CAT_COL, MATERIAL_COL])['metric']
        .count().reset_index().rename(columns={'metric': 'n_segments'})
    )
    agg = agg.merge(seg_counts, on=[RPM_CAT_COL, MATERIAL_COL], how='left')

    print(f"\nTotal valid segments: {agg['n_segments'].sum():,}  "
          f"({agg['n_trials'].sum():,} trials)\n")
    return agg
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════
 
def plot(agg: pd.DataFrame, output_path: str, show: bool = True) -> None:
    rpm_classes = RPM_ORDER if RPM_ORDER is not None else sorted(agg[RPM_CAT_COL].unique())
    rpm_classes = [r for r in rpm_classes if r in agg[RPM_CAT_COL].values]
 
    x = np.arange(len(rpm_classes))
 
    def get_stats(material: str) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
        means, stds, n_trials_list, n_segs_list = [], [], [], []
        for rpm in rpm_classes:
            row = agg[(agg[RPM_CAT_COL] == rpm) & (agg[MATERIAL_COL] == material)]
            if row.empty:
                means.append(np.nan); stds.append(0.0)
                n_trials_list.append(0); n_segs_list.append(0)
            else:
                means.append(float(row['mean_metric'].iloc[0]))
                stds.append(float(row['std_metric'].fillna(0).iloc[0]))
                n_trials_list.append(int(row['n_trials'].iloc[0])
                                     if 'n_trials' in row.columns else 0)
                n_segs_list.append(int(row['n_segments'].iloc[0])
                                   if 'n_segments' in row.columns else 0)
        return np.array(means), np.array(stds), n_trials_list, n_segs_list
 
    metal_mean,   metal_std,   metal_n,   metal_n_seg   = get_stats('metal')
    plastic_mean, plastic_std, plastic_n, plastic_n_seg = get_stats('plastic')
 
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
 
    def label_bars(bars: plt.BarContainer, means: np.ndarray) -> None:
        y_scale = max(np.nanmax(metal_mean), np.nanmax(plastic_mean))
        for bar, val in zip(bars, means):
            if np.isnan(val):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * y_scale,
                f'{val:.4f}',
                ha='center', va='bottom',
                fontsize=8, color='#333333', fontweight='bold',
            )
 
    label_bars(bars_metal,   metal_mean)
    label_bars(bars_plastic, plastic_mean)
 
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r} RPM' for r in rpm_classes], fontsize=11)
    ax.set_xlabel('RPM Class', fontsize=12, labelpad=8)
    ax.set_ylabel('Revolutions per Joule (rev / J)', fontsize=12, labelpad=8)
    ax.set_title(
        'Drag / Specific-Energy Proxy by RPM: Metal vs Plastic  (no-load)\n'
        'Metric = Δrev / ∫|τ(t)| × ω(t) dt   (higher = less energy per rev)\n'
        'NOT a true efficiency: |τ| is rectified and there is no output load — '
        'read as a relative drag indicator',
        fontsize=11, fontweight='bold', pad=14,
    )
 
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis='y', which='major', linestyle='--', alpha=0.4, zorder=0)
    ax.grid(axis='y', which='minor', linestyle=':', alpha=0.2, zorder=0)
    # add this
    #ax.set_yscale('log')
    #ax.set_ylim(bottom=2e-1)  # adjust to match your data range
    ax.set_axisbelow(True)
 
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#cccccc')
 
    ax.tick_params(axis='both', which='both', length=0)
 
    for i, rpm in enumerate(rpm_classes):
        for material, xoff, color, n_t, n_s in [
            ('metal',   -offset, COLOR_METAL,   metal_n,   metal_n_seg),
            ('plastic', +offset, COLOR_PLASTIC, plastic_n, plastic_n_seg),
        ]:
            ax.text(
                i + xoff, -0.04,
                f'{n_t[i]}t/{n_s[i]}s',
                ha='center', va='top',
                fontsize=7, color=color, alpha=0.75,
                transform=ax.get_xaxis_transform(),
            )
 
    ax.legend(
        fontsize=11, framealpha=0.9, edgecolor='#cccccc',
        loc='upper right', frameon=True,
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
    global PARQUET_DIR, OUTPUT_PATH, RPM_ORDER, SHOW_PLOT, TORQUE_COL, RPM_COL, ANGLE_COL, TIME_COL, RENDER_PLOTS
    global PLATEAU_GATING, PLATEAU_FRAC
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
        OUTPUT_PATH  = os.path.join(stage_out(p, 'output_efficiency'),
                                    'efficiency_proxy_metal_vs_plastic.png')
        RPM_ORDER    = list(p.rpms)
        SHOW_PLOT    = p.show_plots
        TORQUE_COL   = p.torque_col
        RPM_COL      = p.rpm_col
        ANGLE_COL    = p.angle_col
        TIME_COL     = p.time_col
        RENDER_PLOTS = p.render_plots
        PLATEAU_GATING = getattr(p, 'plateau_gating', PLATEAU_GATING)
        PLATEAU_FRAC   = getattr(p, 'plateau_frac', PLATEAU_FRAC)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    agg = aggregate_from_parquets(PARQUET_DIR)

    print("Aggregated stats:")
    print(agg.to_string(index=False))
    print()

    # Write pre-aggregated CSV so viz.py can load it without re-scanning parquets.
    agg_csv = os.path.join(os.path.dirname(OUTPUT_PATH), "efficiency_agg.csv")
    agg.rename(columns={"mean_metric": "eff_proxy_mean",
                        "std_metric":  "eff_proxy_std"}).to_csv(agg_csv, index=False)
    print(f"Aggregate CSV saved -> {os.path.abspath(agg_csv)}")

    if RENDER_PLOTS:
        plot(agg, OUTPUT_PATH, show=SHOW_PLOT)
 
 
if __name__ == '__main__':
    main()
 