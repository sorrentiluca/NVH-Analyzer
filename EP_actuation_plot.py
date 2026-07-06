"""
═══════════════════════════════════════════════════════════════════════════════
 SINGLE-ACTUATION CHANNEL PLOT  —  report-quality time-series
═══════════════════════════════════════════════════════════════════════════════

Reads one parquet produced by EP_segment.py, picks a single actuation
(configurable), and plots all logged channels on a shared time axis:

  Row 1 — X / Y / Z Acceleration  (m/s²)   [3-in-1 panel]
  Row 2 — Angle                    (°)
  Row 3 — RPM
  Row 4 — Torque                   (Nm)

OUTPUT
------
  output_actuation_plot/
    actuation_plot_<part>_<rpm>rpm_<dir>_seg<N>.png
    actuation_plot_<part>_<rpm>rpm_<dir>_seg<N>.pdf   (vector, for Word)

USAGE
-----
  python EP_actuation_plot.py                  # uses defaults in Config
  # or import and call:
  #   from EP_actuation_plot import Config, run
  #   run(Config(part='M2', rpm_category=1200, direction='NEG', segment_index=0))
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG  —  edit these to pick which actuation to plot
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── INPUT ────────────────────────────────────────────────────────────────
    segmented_data_dir: str = 'output_seg/segmented_data'

    # ── ACTUATION SELECTOR ───────────────────────────────────────────────────
    # Set to None to auto-select the first available match
    part:           str | None = None   # e.g. 'M1', 'P3'
    rpm_category:   int | None = 2300   # e.g. 600, 1200, 1800
    direction:      str | None = 'POS'   # 'POS' or 'NEG'
    trial:          int | None = None   # trial number
    segment_index:  int        = 0      # 0 = first matching segment

    # ── COLUMN NAMES  (match EP_segment output) ──────────────────────────────
    time_col:   str = 't_rel'           # relative time within segment
    angle_col:  str = 'Angle (deg)'
    rpm_col:    str = 'RPM'
    torque_col: str = 'Torque (Nm)'

    # Y/Z swap: physical Y is logged under "Z Accel" and vice versa
    accel_cols: dict = field(default_factory=lambda: {
        'X': 'X Accel (m/s2)',
        'Y': 'Z Accel (m/s2)',
        'Z': 'Y Accel (m/s2)',
    })

    # ── STYLE ────────────────────────────────────────────────────────────────
    accel_colors: dict = field(default_factory=lambda: {
        'X': '#1f77b4',   # blue
        'Y': '#d62728',   # red
        'Z': '#2ca02c',   # green
    })
    rpm_color:    str = '#9467bd'   # purple
    angle_color:  str = '#8c564b'   # brown
    torque_color: str = '#e377c2'   # pink

    fig_width:  float = 11.0        # inches — A4 landscape friendly
    dpi_screen: int   = 140
    dpi_pdf:    int   = 300

    # ── OUTPUT ───────────────────────────────────────────────────────────────
    output_dir:  str  = 'output_actuation_plot'
    save_pdf:    bool = True        # also save vector PDF for Word
    show_plot:   bool = False


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD + SELECT
# ═══════════════════════════════════════════════════════════════════════════════

def load_actuation(cfg: Config) -> tuple[pd.DataFrame, dict]:
    """
    Loads parquets, applies selector filters, returns the raw time-series
    DataFrame for one segment + a metadata dict for the plot title.
    """
    from nvh_pipeline import dataset
    files = dataset.list_segment_files(cfg.segmented_data_dir)
    if not files:
        raise FileNotFoundError(
            f"No parquets under {os.path.abspath(cfg.segmented_data_dir)}.\n"
            "Run EP_segment.py first."
        )

    candidates = []
    for pq in files:
        try:
            df = dataset.read_file(cfg.segmented_data_dir, pq)
        except Exception as e:
            print(f"  SKIP: {os.path.basename(pq)}  ({e})")
            continue

        # Apply filters
        if cfg.part         and df['part'].iloc[0]          != cfg.part:           continue
        if cfg.rpm_category and int(df['rpm_category'].iloc[0]) != cfg.rpm_category: continue
        if cfg.trial        and int(df['trial'].iloc[0])    != cfg.trial:          continue

        for seg_id, seg_df in df.groupby('segment_id'):
            if cfg.direction and seg_df['direction'].iloc[0] != cfg.direction:
                continue
            candidates.append((seg_df.copy(), {
                'part':      df['part'].iloc[0],
                'material':  df['material_type'].iloc[0],
                'rpm_cat':   int(df['rpm_category'].iloc[0]),
                'trial':     int(df['trial'].iloc[0]),
                'direction': seg_df['direction'].iloc[0],
                'seg_id':    int(seg_id),
                'source':    os.path.basename(pq),
            }))

    if not candidates:
        raise RuntimeError(
            "No matching actuations found.\n"
            f"  part={cfg.part}, rpm={cfg.rpm_category}, "
            f"direction={cfg.direction}, trial={cfg.trial}\n"
            "Loosen the filters in Config."
        )

    idx = min(cfg.segment_index, len(candidates) - 1)
    seg_df, meta = candidates[idx]

    print(f"\nSelected actuation:")
    for k, v in meta.items():
        print(f"  {k:12s}: {v}")
    print(f"  rows       : {len(seg_df)}")
    return seg_df, meta


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax: plt.Axes, ylabel: str, color: str | None = None,
              ylabel_color: str = '#333333') -> None:
    """Apply consistent panel styling."""
    ax.set_ylabel(ylabel, fontsize=9.5, color=ylabel_color, labelpad=6)
    ax.tick_params(axis='y', labelsize=8.5)
    ax.tick_params(axis='x', labelsize=8.5)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.4, color='#aaaaaa')
    ax.grid(True, axis='x', linestyle=':',  linewidth=0.5, alpha=0.25, color='#aaaaaa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    if color:
        ax.spines['left'].set_color(color)
        ax.tick_params(axis='y', colors=color)
        ax.yaxis.label.set_color(color)


def plot_actuation(seg_df: pd.DataFrame, meta: dict, cfg: Config, outdir: str) -> str:
    # ── resolve time axis ────────────────────────────────────────────────────
    if cfg.time_col in seg_df.columns:
        t = seg_df[cfg.time_col].values.astype(float)
    else:
        # fallback: use raw time and zero-base it
        raw_t_cols = [c for c in seg_df.columns if 'time' in c.lower()]
        if not raw_t_cols:
            raise KeyError("No time column found. Set cfg.time_col correctly.")
        t = seg_df[raw_t_cols[0]].values.astype(float)
        t = t - t[0]

    # ── resolve channels ─────────────────────────────────────────────────────
    present_accel = {ax: col for ax, col in cfg.accel_cols.items()
                     if col in seg_df.columns}
    has_angle  = cfg.angle_col  in seg_df.columns
    has_rpm    = cfg.rpm_col    in seg_df.columns
    has_torque = cfg.torque_col in seg_df.columns

    # Build panel list: accel always first, then optional channels
    panels = ['accel']
    if has_angle:  panels.append('angle')
    if has_rpm:    panels.append('rpm')
    if has_torque: panels.append('torque')

    n_panels   = len(panels)
    # Accel gets 2× height; others equal
    height_ratios = [2.0] + [1.0] * (n_panels - 1)
    fig_height = 2.5 + sum(height_ratios) * 1.35

    # ── figure / gridspec ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(cfg.fig_width, fig_height), facecolor='white')
    gs  = gridspec.GridSpec(
        n_panels, 1,
        hspace=0.10,
        height_ratios=height_ratios,
        left=0.09, right=0.97, top=0.90, bottom=0.09,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(n_panels)]

    # hide x tick labels for all but the last panel
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    # ── PANEL 0: acceleration ────────────────────────────────────────────────
    ax_acc = axes[0]
    for ax_name, col in present_accel.items():
        color = cfg.accel_colors.get(ax_name, '#555555')
        ax_acc.plot(t, seg_df[col].values, color=color,
                    lw=0.9, alpha=0.92, label=f'{ax_name}-axis')

    ax_acc.axhline(0, color='#999999', lw=0.6, linestyle='--')
    _style_ax(ax_acc, 'Acceleration\n(m/s²)')
    ax_acc.legend(fontsize=8.5, loc='upper right', framealpha=0.85,
                  ncol=3, handlelength=1.5, columnspacing=1.0)
    # RMS annotation per axis
    rms_parts = []
    for ax_name, col in present_accel.items():
        sig  = seg_df[col].values.astype(float)
        rms  = np.sqrt(np.mean(sig ** 2))
        rms_parts.append(f'{ax_name}: {rms:.3g}')
    ax_acc.text(0.01, 0.97, 'RMS  ' + '   '.join(rms_parts),
                transform=ax_acc.transAxes, fontsize=7.5,
                va='top', ha='left', color='#444444',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    panel_idx = 1

    # ── PANEL: angle ─────────────────────────────────────────────────────────
    if has_angle:
        ax_ang = axes[panel_idx]; panel_idx += 1
        ax_ang.plot(t, seg_df[cfg.angle_col].values,
                    color=cfg.angle_color, lw=1.3, alpha=0.95)
        ax_ang.fill_between(t, seg_df[cfg.angle_col].values,
                            alpha=0.08, color=cfg.angle_color)
        _style_ax(ax_ang, 'Angle\n(°)', color=cfg.angle_color)

    # ── PANEL: RPM ───────────────────────────────────────────────────────────
    if has_rpm:
        ax_rpm = axes[panel_idx]; panel_idx += 1
        ax_rpm.plot(t, seg_df[cfg.rpm_col].values,
                    color=cfg.rpm_color, lw=1.3, alpha=0.95)
        ax_rpm.fill_between(t, seg_df[cfg.rpm_col].values,
                            alpha=0.08, color=cfg.rpm_color)
        # annotate mean RPM
        mean_rpm = seg_df[cfg.rpm_col].mean()
        ax_rpm.axhline(mean_rpm, color=cfg.rpm_color, lw=0.8,
                       linestyle='--', alpha=0.6)
        ax_rpm.text(t[-1], mean_rpm, f'  {mean_rpm:.0f} RPM avg',
                    va='center', fontsize=7.5, color=cfg.rpm_color)
        _style_ax(ax_rpm, 'Speed\n(RPM)', color=cfg.rpm_color)

    # ── PANEL: torque ─────────────────────────────────────────────────────────
    if has_torque:
        ax_tor = axes[panel_idx]; panel_idx += 1
        torque = seg_df[cfg.torque_col].values.astype(float)
        ax_tor.plot(t, torque, color=cfg.torque_color, lw=1.3, alpha=0.95)
        ax_tor.fill_between(t, torque, alpha=0.08, color=cfg.torque_color)
        ax_tor.axhline(0, color='#999999', lw=0.6, linestyle='--')
        _style_ax(ax_tor, 'Torque\n(Nm)', color=cfg.torque_color)

    # ── shared x-axis label ───────────────────────────────────────────────────
    axes[-1].set_xlabel('Time (s)', fontsize=10)
    axes[-1].tick_params(axis='x', labelsize=8.5)

    # align all y-axis labels to the same x position
    fig.align_ylabels(axes)

    # ── direction shading banner ───────────────────────────────────────────────
    dir_label = meta['direction']
    dir_color = '#1f77b4' if dir_label == 'POS' else '#d62728'
    for ax in axes:
        ax.axvspan(t[0], t[-1], alpha=0.025, color=dir_color, zorder=0)

    # ── title ─────────────────────────────────────────────────────────────────
    title_main = (
        f"Actuation Time-Series  —  "
        f"{meta['part']}  ({meta['material'].capitalize()})  |  "
        f"{meta['rpm_cat']} RPM  |  "
        f"{'Positive ↑' if dir_label == 'POS' else 'Negative ↓'}  |  "
        f"Trial {meta['trial']}  ·  Segment {meta['seg_id']}"
    )
    title_sub = (
        f"Duration: {t[-1] - t[0]:.3f} s  |  "
        f"{len(seg_df)} samples  |  "
        f"Fs ≈ {len(seg_df) / max(t[-1] - t[0], 1e-6):.0f} Hz"
    )
    fig.text(0.5, 0.965, title_main, ha='center', va='top',
             fontsize=11, fontweight='bold', color='#1a1a2e')
    fig.text(0.5, 0.945, title_sub, ha='center', va='top',
             fontsize=8.5, color='#555555')

    # ── save ─────────────────────────────────────────────────────────────────
    stem = (f"actuation_plot_{meta['part']}_{meta['rpm_cat']}rpm_"
            f"{dir_label}_seg{meta['seg_id']}")

    out_png = os.path.join(outdir, f"{stem}.png")
    fig.savefig(out_png, dpi=cfg.dpi_screen, bbox_inches='tight', facecolor='white')
    print(f"\nPNG saved -> {out_png}")

    if cfg.save_pdf:
        out_pdf = os.path.join(outdir, f"{stem}.pdf")
        fig.savefig(out_pdf, dpi=cfg.dpi_pdf, bbox_inches='tight', facecolor='white')
        print(f"PDF saved -> {out_pdf}")

    if cfg.show_plot:
        plt.show()
    plt.close(fig)
    return out_png


# ═══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def run(cfg: Config | None = None) -> str:
    if cfg is None:
        cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    seg_df, meta = load_actuation(cfg)
    out_path = plot_actuation(seg_df, meta, cfg, cfg.output_dir)
    print(f"\nAll outputs in: {os.path.abspath(cfg.output_dir)}\n")
    return out_path


def main():
    cfg = Config()
    # ImportError = nvh_pipeline genuinely absent (bare standalone run): tolerate.
    # Any other error from apply_to_stage is a real misconfiguration and must
    # surface, not silently fall back to defaults.
    try:
        from nvh_pipeline.config import apply_to_stage
    except ImportError:
        apply_to_stage = None
    if apply_to_stage is not None:
        apply_to_stage('actuation_plot', cfg)
    run(cfg)


if __name__ == '__main__':
    main()
