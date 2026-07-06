"""
Executive presentation plots — management audience.
Two figures:
  1. loudness_executive.png  — RMS per actuation, categorical RPM bins,
                               POS/NEG error bars, % louder badges
  2. loudness_scaling.png    — RMS vs RPM power-law scaling, one line per material

USAGE
-----
  Standalone:  python plot_executive.py
  From pipeline:
      from plot_executive import plot_executive, plot_scaling
      plot_executive(df, output_dir)
      plot_scaling(df, output_dir)

  df = DataFrame from EP_loudness.load_segments(cfg)
  Required columns: material, direction, rpm_cat, rms_combined
"""

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.stats import linregress

sys.path.insert(0, os.path.dirname(__file__))

# ── design tokens ─────────────────────────────────────────────────────────────
BG       = '#ffffff'
PANEL_BG = '#ffffff'
GRID     = '#e8e8e8'
TEXT     = '#1a1a1a'
SUBTEXT  = '#666666'
SPINE    = '#cccccc'
BAND     = '#f7f7f7'   # alternating bin background

# Material colors keyed by material_type (set by EP_segment from part prefix)
MAT_COLORS = {'metal': '#1a6faf', 'plastic': '#e07b00'}
MAT_LIGHT  = {'metal': '#a8cce8', 'plastic': '#f5c07a'}


def _color(material: str, light: bool = False) -> str:
    palette = MAT_LIGHT if light else MAT_COLORS
    return palette.get(material, '#cccccc' if light else '#888888')


def _pct_louder(a: float, b: float) -> str:
    pct = abs(a - b) / min(a, b) * 100
    return f'+{pct:.0f}%'


def _apply_rc():
    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Calibri', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'text.color':       TEXT,
        'axes.labelcolor':  SUBTEXT,
        'xtick.color':      SUBTEXT,
        'ytick.color':      SUBTEXT,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — LOUDNESS OVERVIEW (categorical x-axis)
# ══════════════════════════════════════════════════════════════════════════════

def plot_executive(df: pd.DataFrame, outdir: str) -> None:
    # Materials auto-detected; force metal-first ordering if both present
    mats_present = sorted(df['material'].unique())
    materials    = ([m for m in ('metal', 'plastic') if m in mats_present]
                    + [m for m in mats_present if m not in ('metal', 'plastic')])

    rpms     = sorted(df['rpm_cat'].unique())
    x_pos    = {r: i for i, r in enumerate(rpms)}
    n_rpm    = len(rpms)
    n_total  = len(df)
    rng      = np.random.default_rng(42)

    # Material offsets within each bin
    spread = 0.18
    if len(materials) == 1:
        mat_offset = {materials[0]: 0.0}
    else:
        mat_offset = {m: -spread + 2*spread*i/(len(materials)-1)
                      for i, m in enumerate(materials)}
    jitter_w = 0.09

    dir_means = (df.groupby(['material', 'rpm_cat', 'direction'])
                   .agg(mean_rms=('rms_combined', 'mean'))
                   .reset_index())
    overall_means = (df.groupby(['material', 'rpm_cat'])
                       .agg(mean_rms=('rms_combined', 'mean'))
                       .reset_index())

    _apply_rc()
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    ax.set_facecolor(PANEL_BG)

    # ── Layer 0: alternating shaded bin bands ───────────────────────────────
    for i in range(n_rpm):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, facecolor=BAND, zorder=0)

    # ── Layer 1: individual dots ────────────────────────────────────────────
    for mat in materials:
        light = _color(mat, light=True)
        sub   = df[df['material'] == mat]
        xs_all, ys_all = [], []
        for rpm in rpms:
            grp = sub[sub['rpm_cat'] == rpm]
            if grp.empty:
                continue
            x_base = x_pos[rpm] + mat_offset[mat]
            jitter = rng.uniform(-jitter_w, jitter_w, len(grp))
            xs_all.extend(x_base + jitter)
            ys_all.extend(grp['rms_combined'].values)
        ax.scatter(xs_all, ys_all, color=light, s=14,
                   alpha=0.40, linewidths=0, zorder=1)

    # ── Layer 2: mean lines (each material's trajectory) ────────────────────
    for mat in materials:
        color = _color(mat)
        xs, ys = [], []
        for rpm in rpms:
            om = overall_means[(overall_means['material'] == mat) &
                               (overall_means['rpm_cat']  == rpm)]
            if om.empty:
                continue
            xs.append(x_pos[rpm] + mat_offset[mat])
            ys.append(float(om['mean_rms'].iloc[0]))
        if xs:
            ax.plot(xs, ys, color=color, lw=2.8, zorder=4,
                    solid_capstyle='round', solid_joinstyle='round')

    # ── Layer 3: error bars (POS/NEG spread) + mean dots ────────────────────
    cap_w = 0.045
    for mat in materials:
        color = _color(mat)
        for rpm in rpms:
            dm = dir_means[(dir_means['material'] == mat) &
                           (dir_means['rpm_cat']  == rpm)]
            om = overall_means[(overall_means['material'] == mat) &
                               (overall_means['rpm_cat']  == rpm)]
            if om.empty:
                continue

            x_at   = x_pos[rpm] + mat_offset[mat]
            y_mean = float(om['mean_rms'].iloc[0])

            pos_row = dm[dm['direction'] == 'POS']
            neg_row = dm[dm['direction'] == 'NEG']
            y_pos = float(pos_row['mean_rms'].iloc[0]) if not pos_row.empty else y_mean
            y_neg = float(neg_row['mean_rms'].iloc[0]) if not neg_row.empty else y_mean
            y_hi  = max(y_pos, y_neg)
            y_lo  = min(y_pos, y_neg)

            ax.plot([x_at, x_at], [y_lo, y_hi],
                    color=color, lw=2.0, zorder=5, solid_capstyle='round')
            ax.plot([x_at - cap_w, x_at + cap_w], [y_hi, y_hi],
                    color=color, lw=1.5, zorder=5)
            ax.plot([x_at - cap_w, x_at + cap_w], [y_lo, y_lo],
                    color=color, lw=1.5, zorder=5)

            ax.scatter([x_at], [y_mean],
                       color=color, s=70, zorder=6,
                       edgecolors='white', linewidths=1.5)

    # ── Layer 4: % badges floating above the louder value ───────────────────
    if 'metal' in materials and 'plastic' in materials:
        for rpm in rpms:
            gm = overall_means[(overall_means['material'] == 'metal') &
                               (overall_means['rpm_cat']  == rpm)]
            gp = overall_means[(overall_means['material'] == 'plastic') &
                               (overall_means['rpm_cat']  == rpm)]
            if gm.empty or gp.empty:
                continue
            r_metal   = float(gm['mean_rms'].iloc[0])
            r_plastic = float(gp['mean_rms'].iloc[0])

            louder_val  = max(r_metal, r_plastic)
            quieter_val = min(r_metal, r_plastic)
            louder_mat  = 'metal' if r_metal > r_plastic else 'plastic'
            col         = _color(louder_mat)

            x_badge = x_pos[rpm]                # bin center
            y_badge = louder_val * 1.30         # above the louder dot

            ax.text(x_badge, y_badge,
                    _pct_louder(louder_val, quieter_val),
                    ha='center', va='center',
                    fontsize=9, color=col, fontweight='bold',
                    zorder=10,
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', edgecolor=col,
                              linewidth=1.2, alpha=1.0))

    # ── Axes & styling ──────────────────────────────────────────────────────
    ax.set_xlim(-0.5, n_rpm - 0.5)
    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels([f'{r:,} RPM' for r in rpms], fontsize=10.5)
    ax.set_yscale('log')

    # extra headroom for badges
    y_top_data = df['rms_combined'].max()
    ax.set_ylim(top=y_top_data * 2.2)

    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE)
        spine.set_linewidth(0.8)
    ax.tick_params(which='both', color=SPINE, labelsize=10)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f'{v:.2g} m/s\u00b2'))
    ax.grid(True, which='major', axis='y', color=GRID, lw=0.9)
    ax.grid(True, which='minor', axis='y', color=GRID, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)   # grid/bands stay behind data

    ax.set_xlabel('Operating Speed (categorical)',
                  color=SUBTEXT, fontsize=11, labelpad=10)
    ax.set_ylabel('Vibration Level',
                  color=SUBTEXT, fontsize=11, labelpad=10)

    # ── Legend ──────────────────────────────────────────────────────────────
    handles = []
    for mat in materials:
        col = _color(mat)
        handles.append(
            Line2D([0], [0], color=col, lw=2.5,
                   marker='o', markersize=8,
                   markeredgecolor='white', markeredgewidth=1.5,
                   label=f'{mat}  (mean)'))
        handles.append(
            Line2D([0], [0], color=_color(mat, light=True),
                   marker='o', markersize=6, linestyle='none',
                   alpha=0.6,
                   label=f'{mat}  (individual actuations)'))
    handles.append(
        Line2D([0], [0], color=SUBTEXT, lw=1.5,
               marker='_', markersize=10,
               label='error bar = POS / NEG direction spread'))
    ax.legend(handles=handles, fontsize=9, frameon=True,
              framealpha=0.95, edgecolor=SPINE,
              loc='upper left', ncol=1)

    # ── Title ───────────────────────────────────────────────────────────────
    if 'metal' in materials and 'plastic' in materials:
        m0 = df[df['material'] == 'metal']['rms_combined'].mean()
        m1 = df[df['material'] == 'plastic']['rms_combined'].mean()
        louder  = 'plastic' if m1 > m0 else 'metal'
        quieter = 'metal'   if louder == 'plastic' else 'plastic'
        overall_pct = _pct_louder(max(m0, m1), min(m0, m1))
        title = (f'{louder} is louder than {quieter} '
                 f'across all speeds  ({overall_pct} overall)')
    else:
        title = 'Vibration loudness by operating speed'
    ax.set_title(title, fontsize=16, fontweight='bold', color=TEXT, pad=14)

    fig.text(0.5, 0.005,
             f'Vibration level vs operating speed  \u00b7  '
             f'{n_total} total actuations across '
             f'{n_rpm} speed categories and both directions  \u00b7  '
             f'error bars show positive vs negative direction spread',
             ha='center', va='bottom',
             color=SUBTEXT, fontsize=9)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(outdir, 'loudness_executive.png')

    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f'  [1/2] saved -> {os.path.abspath(out)}')


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — RMS vs RPM SCALING (continuous log-log)
# ══════════════════════════════════════════════════════════════════════════════

def _fit_power_law(rpm_vals, rms_vals):
    log_x = np.log10(np.array(rpm_vals, dtype=float))
    log_y = np.log10(np.array(rms_vals, dtype=float))
    slope, intercept, r, *_ = linregress(log_x, log_y)
    return 10**intercept, slope, r**2


def plot_scaling(df: pd.DataFrame, outdir: str) -> None:
    mats_present = sorted(df['material'].unique())
    materials    = ([m for m in ('metal', 'plastic') if m in mats_present]
                    + [m for m in mats_present if m not in ('metal', 'plastic')])
    rng = np.random.default_rng(42)

    _apply_rc()
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    ax.set_facecolor(PANEL_BG)

    for mat in materials:
        color = _color(mat)
        light = _color(mat, light=True)
        sub   = df[df['material'] == mat].copy()
        if sub.empty:
            continue

        # Individual actuations — jitter horizontally on log scale
        jitter = sub['rpm_cat'].astype(float) * (
            1 + rng.uniform(-0.04, 0.04, len(sub)))
        ax.scatter(jitter, sub['rms_combined'],
                   color=light, s=14, alpha=0.40,
                   linewidths=0, zorder=1)

        # Per-category means at the nominal RPM
        means = (sub.groupby('rpm_cat')
                    .agg(mean_rms=('rms_combined', 'mean'))
                    .reset_index()
                    .sort_values('rpm_cat'))

        ax.plot(means['rpm_cat'], means['mean_rms'],
                color=color, lw=2.8, zorder=4,
                marker='o', markersize=9,
                markerfacecolor=color,
                markeredgecolor='white', markeredgewidth=1.8,
                label=f'{mat}  (mean per speed)')

        # Power-law fit on the category means
        if len(means) >= 3:
            a, n, r2 = _fit_power_law(means['rpm_cat'], means['mean_rms'])
            rpm_fit = np.logspace(
                np.log10(means['rpm_cat'].min() * 0.80),
                np.log10(means['rpm_cat'].max() * 1.20), 200)
            ax.plot(rpm_fit, a * rpm_fit**n,
                    color=color, lw=1.5, linestyle='--',
                    alpha=0.65, zorder=3,
                    label=f'{mat}  fit: RMS \u221d RPM^{n:.2f}  (R\u00b2={r2:.3f})')

            mid_i   = len(rpm_fit) // 2
            mid_rpm = rpm_fit[mid_i]
            mid_rms = a * mid_rpm**n
            ax.annotate(
                f'slope = {n:.2f}',
                xy=(mid_rpm, mid_rms),
                xytext=(mid_rpm * 1.12, mid_rms * 0.80),
                fontsize=9, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=0.9),
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor='white', edgecolor=color,
                          linewidth=0.8, alpha=0.92),
                zorder=8)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(df['rpm_cat'].min() * 0.70, df['rpm_cat'].max() * 1.40)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE)
        spine.set_linewidth(0.8)
    ax.tick_params(which='both', color=SPINE, labelsize=10)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f'{int(v):,} RPM'))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f'{v:.2g} m/s\u00b2'))
    ax.grid(True, which='major', color=GRID, lw=0.9)
    ax.grid(True, which='minor', color=GRID, lw=0.4, alpha=0.5)
    ax.set_xlabel('Operating Speed', color=SUBTEXT, fontsize=11, labelpad=10)
    ax.set_ylabel('Vibration Level', color=SUBTEXT, fontsize=11, labelpad=10)
    ax.legend(fontsize=9.5, frameon=True, framealpha=0.95,
              edgecolor=SPINE, loc='upper left')

    ax.set_title('How does vibration scale with speed?',
                 fontsize=16, fontweight='bold', color=TEXT, pad=14)
    fig.text(0.5, 0.005,
             'Power law fit: RMS \u221d RPM^n  \u00b7  '
             'if slopes match, materials differ only in amplitude  \u00b7  '
             'if slopes differ, materials respond differently to speed',
             ha='center', va='bottom', color=SUBTEXT, fontsize=9)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(outdir, 'loudness_scaling.png')
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches='tight')

    plt.close(fig)
    print(f'  [2/2] saved -> {os.path.abspath(out)}')


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE
# ══════════════════════════════════════════════════════════════════════════════

def main(show: bool = False):
    from EP_loudness import Config, load_segments

    cfg = Config()
    # ImportError = nvh_pipeline genuinely absent (bare standalone run): tolerate.
    # Any other error from apply_to_stage is a real misconfiguration and must
    # surface, not silently fall back to defaults.
    try:
        from nvh_pipeline.config import apply_to_stage
    except ImportError:
        apply_to_stage = None
    if apply_to_stage is not None:
        apply_to_stage('loudness', cfg)   # same input/output dir as loudness

    outdir = cfg.output_dir
    os.makedirs(outdir, exist_ok=True)

    print('\nLoading pre-segmented data...')
    df = load_segments(cfg)

    if cfg.render_plots:
        print('\nGenerating figures...')
        plot_executive(df, outdir)
        plot_scaling(df, outdir)
        print('\nDone.')
    else:
        print('\nrender_plots=False — skipping figure generation.')
    if show:
        plt.show()


if __name__ == '__main__':
    main(show=True)