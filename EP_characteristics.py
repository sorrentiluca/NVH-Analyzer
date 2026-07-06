"""
═══════════════════════════════════════════════════════════════════════════════
 CHARACTERISTICS SUMMARY  —  cross-stage one-pager
═══════════════════════════════════════════════════════════════════════════════

Reads existing stage CSVs (no recompute); degrades gracefully when a stage
hasn't been run.  Produces a compact per-material summary with:

  • Vibration level (RMS, m/s²) — mean ± std across actuations
  • Plastic / metal RMS ratio   (higher = more vibration in plastic)
  • Dominant shaft order        (from order analysis)
  • Resonance band center (Hz)  (from envelope kurtogram)
  • Data-quality flags          (from dataset.DataQuality)

OUTPUT
------
  output_handover/
    characteristics.csv    one row per material (+ one combined row)
    characteristics.html   styled table for easy copy-paste into Word/email
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    output_root:   str = '.'
    output_dir:    str = 'output_handover'

    # Relative paths to upstream CSV outputs (from output_root)
    rms_csv:       str = 'output_loudness/rms_per_segment.csv'
    order_csv:     str = 'output_order/per_actuation_peaks.csv'
    bandpass_csv:  str = 'output_envelope/bandpass_log.csv'


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_read(path: str, name: str) -> pd.DataFrame | None:
    if not os.path.isfile(path):
        print(f"  characteristics: {name} not found — skipping ({path})")
        return None
    try:
        df = pd.read_csv(path)
        print(f"  characteristics: loaded {name}  ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"  characteristics: could not read {name}: {e}")
        return None


def _rms_summary(rms_df: pd.DataFrame) -> pd.DataFrame:
    """Per-material mean/std vibration level and POS/NEG split."""
    rows = []
    for mat, g in rms_df.groupby('material'):
        rows.append({
            'material':        mat,
            'n_actuations':    len(g),
            'rms_mean_m_s2':   round(g['rms_combined'].mean(), 5),
            'rms_std_m_s2':    round(g['rms_combined'].std(), 5),
            'rms_mean_POS':    round(g.loc[g['direction'] == 'POS', 'rms_combined'].mean(), 5),
            'rms_mean_NEG':    round(g.loc[g['direction'] == 'NEG', 'rms_combined'].mean(), 5),
        })
    return pd.DataFrame(rows)


def _order_summary(order_df: pd.DataFrame) -> pd.DataFrame:
    """Most common dominant order per material."""
    rows = []
    mat_col = 'material_type' if 'material_type' in order_df.columns else 'material'
    for mat, g in order_df.groupby(mat_col):
        mode_order = g['dominant_order'].mode()
        rows.append({
            'material':              mat,
            'dominant_order_mode':   round(float(mode_order.iloc[0]), 2) if len(mode_order) else float('nan'),
            'dominant_order_mean':   round(g['dominant_order'].mean(), 2),
            'dominant_amp_mean':     round(g['dominant_amp'].mean(), 6),
        })
    return pd.DataFrame(rows)


def _band_summary(band_df: pd.DataFrame) -> pd.DataFrame:
    """Mean resonance band center per material (all axes combined)."""
    rows = []
    mat_col = 'material' if 'material' in band_df.columns else 'material_type'
    for mat, g in band_df.groupby(mat_col):
        rows.append({
            'material':           mat,
            'resonance_band_mean_hz': round(g['f_center_hz'].mean(), 1),
            'resonance_band_std_hz':  round(g['f_center_hz'].std(), 1),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_characteristics(cfg: Config) -> pd.DataFrame:
    root = cfg.output_root

    rms_df   = _safe_read(os.path.join(root, cfg.rms_csv),    'rms_per_segment.csv')
    order_df = _safe_read(os.path.join(root, cfg.order_csv),  'per_actuation_peaks.csv')
    band_df  = _safe_read(os.path.join(root, cfg.bandpass_csv), 'bandpass_log.csv')

    # Build base from RMS (always available if loudness stage ran)
    if rms_df is None:
        print("  characteristics: no RMS data — cannot build summary")
        return pd.DataFrame()

    summary = _rms_summary(rms_df)

    # Merge order summary
    if order_df is not None:
        ord_sum = _order_summary(order_df)
        summary = summary.merge(ord_sum, on='material', how='left')

    # Merge band summary
    if band_df is not None:
        band_sum = _band_summary(band_df)
        summary = summary.merge(band_sum, on='material', how='left')

    # Plastic/metal RMS ratio
    mats = summary.set_index('material')
    if 'metal' in mats.index and 'plastic' in mats.index:
        ratio = mats.loc['plastic', 'rms_mean_m_s2'] / mats.loc['metal', 'rms_mean_m_s2']
        summary['plastic_metal_rms_ratio'] = summary['material'].apply(
            lambda m: round(float(ratio), 3) if m in ('metal', 'plastic') else float('nan')
        )

    return summary


def _to_html(df: pd.DataFrame) -> str:
    """Styled HTML table for the summary."""
    styles = (
        "border-collapse:collapse;font-family:Arial,sans-serif;"
        "font-size:13px;width:100%"
    )
    th_style = (
        "background:#1a1a2e;color:white;padding:7px 10px;"
        "text-align:left;border:1px solid #ccc"
    )
    td_style = "padding:6px 10px;border:1px solid #ddd"
    tr_even  = "background:#f5f7fa"

    rows_html = []
    for i, (_, row) in enumerate(df.iterrows()):
        tr_bg = f' style="{tr_even}"' if i % 2 == 1 else ''
        cells = ''.join(
            f'<td style="{td_style}">{v}</td>'
            for v in row.values
        )
        rows_html.append(f'<tr{tr_bg}>{cells}</tr>')

    headers = ''.join(
        f'<th style="{th_style}">{c}</th>' for c in df.columns
    )
    return (
        f'<table style="{styles}">'
        f'<thead><tr>{headers}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table>'
    )


def run(cfg: Config | None = None) -> None:
    if cfg is None:
        cfg = Config()

    outdir = os.path.join(cfg.output_root, cfg.output_dir)
    os.makedirs(outdir, exist_ok=True)

    print(f"\nCharacteristics summary:")
    summary = compute_characteristics(cfg)

    if summary.empty:
        print("  No data — characteristics outputs skipped.")
        return

    csv_path  = os.path.join(outdir, 'characteristics.csv')
    html_path = os.path.join(outdir, 'characteristics.html')

    summary.to_csv(csv_path, index=False)
    print(f"  -> {csv_path}")

    html = _to_html(summary)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(f'<!DOCTYPE html><html><body>\n{html}\n</body></html>\n')
    print(f"  -> {html_path}")

    # Print to console
    print(f"\n{'─'*60}")
    print(summary.to_string(index=False))
    print(f"{'─'*60}\n")


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
        apply_to_stage('characteristics', cfg)
    run(cfg)


if __name__ == '__main__':
    main()
