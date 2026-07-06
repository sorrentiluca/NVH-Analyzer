"""
═══════════════════════════════════════════════════════════════════════════════
 HANDOVER TABLE  —  Test Sample Summary
═══════════════════════════════════════════════════════════════════════════════

Reads the same pre-segmented parquets produced by EP_segment.py and outputs
a clean summary table suitable for a project handover document.

OUTPUT
------
  output_handover/
    handover_table.png      print-ready table image (paste into Word / slides)
    handover_table.csv      same data for Excel or copy-paste
    handover_table.html     standalone HTML — open in browser, copy into Word

COLUMNS (one row per unique part)
  Part ID | Material | Trials | RPM Range | RPM Steps | Total Actuations |
  POS Actuations | NEG Actuations | Avg Actuations / Trial | Notes
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    segmented_data_dir: str = 'output_seg/segmented_data'
    output_dir:         str = 'output_handover'
    show_plots:         bool = False
    render_plots:       bool = True

    # Schaeffler brand palette — swap freely
    color_header_bg:    str = '#1B3A5C'   # dark navy
    color_header_fg:    str = '#FFFFFF'
    color_metal_bg:     str = '#DDE8F5'   # light steel blue
    color_plastic_bg:   str = '#FFF0DC'   # light amber
    color_alt_row:      str = '#F5F7FA'   # very light grey for alternating rows
    color_border:       str = '#BECAD8'

    # Optional per-part notes — edit as needed
    part_notes: dict = field(default_factory=lambda: {
        # e.g. 'M1': 'Reference baseline', 'P2': 'Epoxied variant'
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD
# ═══════════════════════════════════════════════════════════════════════════════

def load_segments(cfg: Config) -> pd.DataFrame:
    # Read via the shared DuckDB-over-parquet layer; this stage only needs the
    # metadata columns, so push those down instead of reading the full files.
    from nvh_pipeline import dataset

    files = dataset.list_segment_files(cfg.segmented_data_dir)
    if not files:
        raise FileNotFoundError(
            f"No parquets under {os.path.abspath(cfg.segmented_data_dir)}.\n"
            "Run EP_segment.py first."
        )

    required = ('material_type', 'rpm_category', 'part', 'trial',
                'segment_id', 'direction')
    rows = []

    for pq in files:
        try:
            df = dataset.read_file(cfg.segmented_data_dir, pq, columns=list(required))
        except Exception as e:
            print(f"  SKIP (read error): {os.path.basename(pq)}  ({e})")
            continue
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  SKIP (missing {missing}): {os.path.basename(pq)}")
            continue

        for seg_id, seg_df in df.groupby('segment_id'):
            rows.append({
                'part':       df['part'].iloc[0],
                'material':   df['material_type'].iloc[0],
                'rpm_cat':    int(df['rpm_category'].iloc[0]),
                'trial':      int(df['trial'].iloc[0]),
                'segment_id': int(seg_id),
                'direction':  seg_df['direction'].iloc[0],
            })

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("No actuations loaded — check parquet files.")
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
#  BUILD SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def build_summary(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    records = []
    for part, grp in raw.groupby('part'):
        material    = grp['material'].iloc[0]
        rpm_vals    = sorted(grp['rpm_cat'].unique())
        n_trials    = grp['trial'].nunique()
        n_total     = len(grp)
        n_pos       = int((grp['direction'] == 'POS').sum())
        n_neg       = int((grp['direction'] == 'NEG').sum())
        avg_per_trial = round(n_total / n_trials, 1) if n_trials else 0

        records.append({
            'Part ID':               part,
            'Material':              material.capitalize(),
            'Trials':                n_trials,
            'RPM Range':             f"{rpm_vals[0]}–{rpm_vals[-1]}",
            'RPM Steps':             len(rpm_vals),
            'Total Actuations':      n_total,
            'POS Actuations':        n_pos,
            'NEG Actuations':        n_neg,
            'Avg Acts / Trial':      avg_per_trial,
            'Notes':                 cfg.part_notes.get(part, ''),
            '_material_raw':         material,   # for row colouring — dropped before CSV/HTML
        })

    df = pd.DataFrame(records)
    # Sort: material first (metal before plastic), then part name
    order = {'metal': 0, 'plastic': 1}
    df = df.sort_values(
        by=['_material_raw', 'Part ID'],
        key=lambda col: col.map(order) if col.name == '_material_raw' else col
    ).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  PNG TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_table(df: pd.DataFrame, cfg: Config, outdir: str) -> None:
    display_cols = [c for c in df.columns if not c.startswith('_')]
    display_df   = df[display_cols]

    n_rows, n_cols = display_df.shape
    col_widths = [1.1, 1.1, 0.7, 1.1, 0.9, 1.3, 1.2, 1.2, 1.3, 1.8]
    col_widths = col_widths[:n_cols]  # trim if Notes col missing
    total_w    = sum(col_widths)

    fig_w = min(total_w * 1.05, 22)
    fig_h = (n_rows + 1) * 0.52 + 0.9    # header + rows + title

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    row_colors = []
    for _, row in df.iterrows():
        mat = row['_material_raw']
        if mat == 'metal':
            row_colors.append([cfg.color_metal_bg] * n_cols)
        elif mat == 'plastic':
            row_colors.append([cfg.color_plastic_bg] * n_cols)
        else:
            row_colors.append([cfg.color_alt_row] * n_cols)

    header_colors = [[cfg.color_header_bg] * n_cols]

    tbl = ax.table(
        cellText=display_df.values.tolist(),
        colLabels=display_cols,
        cellLoc='center',
        loc='center',
        cellColours=row_colors,
        colColours=header_colors[0],
        colWidths=[w / total_w for w in col_widths],
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.55)

    # Style header cells
    for col_i in range(n_cols):
        cell = tbl[0, col_i]
        cell.set_text_props(color=cfg.color_header_fg, fontweight='bold')
        cell.set_edgecolor(cfg.color_border)

    # Style data cells
    for row_i in range(1, n_rows + 1):
        for col_i in range(n_cols):
            cell = tbl[row_i, col_i]
            cell.set_edgecolor(cfg.color_border)
            # Right-align numeric columns
            if display_cols[col_i] in (
                'Trials', 'RPM Steps', 'Total Actuations',
                'POS Actuations', 'NEG Actuations', 'Avg Acts / Trial'
            ):
                cell.get_text().set_ha('right')
                cell.PAD = 0.06

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cfg.color_metal_bg,   edgecolor=cfg.color_border, label='Metal'),
        Patch(facecolor=cfg.color_plastic_bg, edgecolor=cfg.color_border, label='Plastic'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              fontsize=8, framealpha=0.85, bbox_to_anchor=(1.0, -0.02))

    fig.suptitle(
        'Ballscrew NVH Test Programme  —  Sample Summary',
        fontsize=13, fontweight='bold', y=0.97,
        fontfamily='DejaVu Sans'
    )
    fig.text(0.5, 0.01,
             f'Generated from {df["Part ID"].nunique()} part(s)  |  '
             f'EP_segment parquet data',
             ha='center', fontsize=7.5, color='#777777')

    out_png = os.path.join(outdir, 'handover_table.png')
    fig.savefig(out_png, dpi=160, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Table PNG  -> {out_png}")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV
# ═══════════════════════════════════════════════════════════════════════════════

def write_csv(df: pd.DataFrame, outdir: str) -> None:
    display_cols = [c for c in df.columns if not c.startswith('_')]
    out = os.path.join(outdir, 'handover_table.csv')
    df[display_cols].to_csv(out, index=False)
    print(f"Table CSV  -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML  (copy into Word or send by email)
# ═══════════════════════════════════════════════════════════════════════════════

def write_html(df: pd.DataFrame, cfg: Config, outdir: str) -> None:
    display_cols = [c for c in df.columns if not c.startswith('_')]

    numeric_cols = {
        'Trials', 'RPM Steps', 'Total Actuations',
        'POS Actuations', 'NEG Actuations', 'Avg Acts / Trial'
    }

    def row_bg(mat):
        return cfg.color_metal_bg if mat == 'metal' else (
               cfg.color_plastic_bg if mat == 'plastic' else cfg.color_alt_row)

    header_html = ''.join(
        f'<th>{col}</th>' for col in display_cols
    )

    rows_html = ''
    for _, row in df.iterrows():
        bg    = row_bg(row['_material_raw'])
        cells = ''
        for col in display_cols:
            align = 'right' if col in numeric_cols else 'left'
            cells += f'<td style="text-align:{align}">{row[col]}</td>'
        rows_html += f'<tr style="background:{bg}">{cells}</tr>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ballscrew NVH — Sample Handover</title>
<style>
  body {{
    font-family: Calibri, Arial, sans-serif;
    font-size: 13px;
    margin: 32px 40px;
    color: #1a1a2e;
    background: #fff;
  }}
  h2 {{
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 4px;
    color: {cfg.color_header_bg};
    letter-spacing: 0.02em;
  }}
  .subtitle {{
    font-size: 11px;
    color: #777;
    margin-bottom: 18px;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 16px;
  }}
  th {{
    background: {cfg.color_header_bg};
    color: {cfg.color_header_fg};
    font-weight: 600;
    padding: 8px 10px;
    text-align: center;
    border: 1px solid {cfg.color_border};
    white-space: nowrap;
  }}
  td {{
    padding: 6px 10px;
    border: 1px solid {cfg.color_border};
    vertical-align: middle;
  }}
  tr:hover td {{ filter: brightness(0.97); }}
  .legend {{
    display: flex;
    gap: 18px;
    font-size: 11px;
    color: #555;
  }}
  .swatch {{
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 1px solid {cfg.color_border};
    border-radius: 2px;
    vertical-align: middle;
    margin-right: 4px;
  }}
</style>
</head>
<body>
  <h2>Ballscrew NVH Test Programme — Sample Summary</h2>
  <p class="subtitle">
    Generated from EP_segment parquet data &nbsp;·&nbsp;
    {df['Part ID'].nunique()} part(s) &nbsp;·&nbsp;
    {int(df['Total Actuations'].sum())} total actuations
  </p>
  <table>
    <thead><tr>{header_html}</tr></thead>
    <tbody>
{rows_html}    </tbody>
  </table>
  <div class="legend">
    <span>
      <span class="swatch" style="background:{cfg.color_metal_bg}"></span>Metal
    </span>
    <span>
      <span class="swatch" style="background:{cfg.color_plastic_bg}"></span>Plastic
    </span>
  </div>
</body>
</html>
"""

    out = os.path.join(outdir, 'handover_table.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Table HTML -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSOLE PRINT
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame) -> None:
    display_cols = [c for c in df.columns if not c.startswith('_')]
    disp = df[display_cols].copy()
    print(f"\n{'='*110}")
    print("  HANDOVER TABLE — Test Sample Summary")
    print(f"{'='*110}")
    print(disp.to_string(index=False))
    print(f"{'='*110}")
    print(f"  {df['Part ID'].nunique()} parts  |  "
          f"{int(df['Total Actuations'].sum())} total actuations\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run(cfg: Config) -> pd.DataFrame:
    os.makedirs(cfg.output_dir, exist_ok=True)
    print("\nLoading segments...")
    raw     = load_segments(cfg)
    summary = build_summary(raw, cfg)
    print_summary(summary)
    write_csv(summary, cfg.output_dir)
    write_html(summary, cfg, cfg.output_dir)
    if cfg.render_plots:
        plot_table(summary, cfg, cfg.output_dir)
    print(f"\nAll outputs in: {os.path.abspath(cfg.output_dir)}\n")
    return summary


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
        apply_to_stage('sample_summary', cfg)
    run(cfg)


if __name__ == '__main__':
    main()