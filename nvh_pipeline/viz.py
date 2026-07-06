"""
viz — interactive (Altair) result views for the app's Review tab.

``build_views(manifest, out_root, seg_dir, …)`` turns a finished run's CSV
outputs into a list of :class:`StageView` objects: one per analysis module,
each carrying a plain-English purpose line, KPI takeaways, interactive
charts, optional split-by variants, honesty notes (resolution caveats, data
warnings) and the run's static PNG/CSV files.

Everything here is read-only over the run folder — no recomputation — so the
Review tab stays instant even for large campaigns.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import altair as alt

# Campbell / spectrum tables can exceed Altair's 5k default row cap; the data
# is pre-aggregated (binned) before charting so the embedded size stays small.
try:
    alt.data_transformers.disable_max_rows()
except Exception:                                     # pragma: no cover
    pass

_MATERIAL_SCALE = alt.Scale(domain=["metal", "plastic"],
                            range=["#1f77b4", "#ff7f0e"])


# ─────────────────────────────────────────────────────────────────────────────
#  View container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StageView:
    key: str
    title: str
    purpose: str = ""
    available: bool = False
    takeaways: list = field(default_factory=list)      # [{label, value, help?}]
    charts: list = field(default_factory=list)         # [(caption, alt.Chart)]
    chart_variants: list = field(default_factory=list) # [(caption, {label: c})]
    notes: list = field(default_factory=list)          # strings; ⚠/✅ prefixed
    figure_files: list = field(default_factory=list)
    table_files: list = field(default_factory=list)


def _read_csv(*parts) -> Optional[pd.DataFrame]:
    path = os.path.join(*parts)
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception:
        return None


def _read_json(*parts) -> dict:
    path = os.path.join(*parts)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _color_material(extra_domain=()):
    domain = ["metal", "plastic", *extra_domain]
    return alt.Color("material:N",
                     scale=alt.Scale(domain=domain,
                                     range=["#1f77b4", "#ff7f0e",
                                            "#2ca02c", "#9467bd"][:len(domain)]),
                     legend=alt.Legend(title="Material"))


# ─────────────────────────────────────────────────────────────────────────────
#  Campbell heatmap + honesty note
# ─────────────────────────────────────────────────────────────────────────────

def _pool_orders(cm: pd.DataFrame, max_cols: int = 150) -> pd.DataFrame:
    """Decimate the order axis with MAX pooling so narrow peaks survive
    display binning (mean pooling would wash them out)."""
    orders = np.sort(cm["order"].unique())
    if orders.size <= max_cols:
        return cm
    edges = np.linspace(orders.min(), orders.max(), max_cols + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = cm.copy()
    idx = np.clip(np.searchsorted(edges, out["order"], side="right") - 1,
                  0, max_cols - 1)
    out["order"] = centers[idx]
    return (out.groupby(["material", "rpm", "order"], as_index=False)
            ["amplitude"].max())


def _campbell_heatmap(view: StageView, out_dir: str, csv_path: str,
                      caption: str, x_title: str = "order") -> None:
    """Amplitude heatmap on a REAL linear (order × RPM) plane.

    Both axes are binned so cells tile the plane (raw per-window RPM values
    would render as pixel slivers), and every tooltip field is the binned /
    aggregated value of the hovered cell — never a raw row that may not match
    what the cell's colour shows.  Materials are faceted, never merged: metal
    and plastic amplitudes must not aggregate into one cell.
    """
    if not os.path.isfile(csv_path):
        return
    try:
        cm = pd.read_csv(csv_path)
    except Exception:
        return
    if cm.empty or not {"material", "rpm", "order", "amplitude"} <= set(cm.columns):
        return

    cm = _pool_orders(cm.dropna(subset=["rpm", "order", "amplitude"]))
    if cm.empty:
        return

    rpms = np.sort(cm["rpm"].unique())
    rpm_step = (float(np.min(np.diff(rpms))) if rpms.size > 1
                else max(float(rpms[0]) * 0.1, 1.0))
    orders = np.sort(cm["order"].unique())
    ord_step = (float(np.min(np.diff(orders))) if orders.size > 1 else 0.1)

    x_bin = alt.Bin(step=ord_step)
    y_bin = alt.Bin(step=rpm_step)
    base = alt.Chart(cm).mark_rect().encode(
        x=alt.X("order:Q", bin=x_bin, title=x_title),
        y=alt.Y("rpm:Q", bin=y_bin, title="measured RPM"),
        color=alt.Color("amplitude:Q", aggregate="max",
                        scale=alt.Scale(scheme="viridis"),
                        legend=alt.Legend(title="amplitude")),
        tooltip=[
            alt.Tooltip("order:Q", bin=x_bin, title=x_title),
            alt.Tooltip("rpm:Q", bin=y_bin, title="RPM"),
            alt.Tooltip("amplitude:Q", aggregate="max",
                        format=".3g", title="amplitude"),
        ],
    ).properties(height=260)

    if cm["material"].nunique() > 1:
        chart = base.facet(facet=alt.Facet("material:N", title=None),
                           columns=2)
    else:
        chart = base
    view.charts.append((caption, chart))


def _campbell_resolution_note(view: StageView, cm: pd.DataFrame,
                              rpm_bin: float) -> None:
    """State — honestly — how much of a waterfall this map really is.

    Driven by the ACTUAL measured-RPM bin coverage: a genuine run-up reads as
    a waterfall, a single-speed strip is flagged as a snapshot, and sparse
    clustered coverage (a few nominal classes with empty gaps) as an overview.
    """
    if cm is None or cm.empty or "rpm" not in cm.columns:
        return
    rpm = pd.to_numeric(cm["rpm"], errors="coerce").dropna()
    if rpm.empty:
        return
    bins = np.unique(np.floor(rpm / float(rpm_bin)).astype(int))
    n_bins = int(bins.size)
    if n_bins <= 1:
        view.notes.append(
            "⚠ Campbell map covers a single measured-RPM bin — this is a "
            "snapshot at one speed, not a waterfall. A run-up/coast-down "
            "recording is needed for a real speed sweep.")
        return
    span_bins = int(bins.max() - bins.min()) + 1
    coverage = n_bins / span_bins
    if coverage < 0.5 or n_bins < 8:
        view.notes.append(
            f"⚠ Campbell map has sparse speed coverage ({n_bins} "
            f"measured-RPM bins over a {span_bins}-bin span) — read it as an "
            "overview of the tested speed classes, not a continuous run-up.")
        return
    view.notes.append(
        f"Campbell map spans {n_bins} measured-RPM bins across the ramp "
        "sweep — vertical stripes are speed-synchronous orders; ridges that "
        "drift with RPM are fixed-frequency resonances.")


# ─────────────────────────────────────────────────────────────────────────────
#  Shared chart builders
# ─────────────────────────────────────────────────────────────────────────────

def _spectrum_df(out_root: str, folder: str, stem: str) -> pd.DataFrame:
    """Long-form (material, order, cumulative) from the per-material mean
    spectrum CSVs, decimated for chart weight."""
    frames = []
    for mat in ("metal", "plastic"):
        df = _read_csv(out_root, folder, f"{stem}_{mat}.csv")
        if df is None:
            continue
        frames.append(pd.DataFrame({
            "order": df["order"], "cumulative": df["cumulative"],
            "material": mat}))
    if not frames:                       # continuous / single-material runs
        df = _read_csv(out_root, folder, f"{stem}.csv")
        if df is not None:
            frames.append(pd.DataFrame({
                "order": df["order"], "cumulative": df["cumulative"],
                "material": "signal"}))
    if not frames:
        return pd.DataFrame()
    # ≤ 800 points per trace keeps the embedded data light.
    thinned = []
    for f in frames:
        if len(f) > 800:
            idx = np.unique(np.linspace(0, len(f) - 1, 800).round()
                            .astype(int))
            f = f.iloc[idx]
        thinned.append(f)
    return pd.concat(thinned, ignore_index=True)


def _spectrum_chart(df: pd.DataFrame, harmonics: list,
                    x_title: str, y_title: str) -> alt.Chart:
    line = alt.Chart(df).mark_line(interpolate="linear").encode(
        x=alt.X("order:Q", title=x_title),
        y=alt.Y("cumulative:Q", title=y_title),
        color=_color_material(
            tuple(m for m in df["material"].unique()
                  if m not in ("metal", "plastic"))),
        tooltip=[alt.Tooltip("order:Q", format=".2f"),
                 alt.Tooltip("cumulative:Q", format=".3g"),
                 alt.Tooltip("material:N")],
    ).properties(height=280)
    if harmonics:
        rules = alt.Chart(pd.DataFrame({"order": harmonics})).mark_rule(
            strokeDash=[4, 3], color="#888").encode(
            x="order:Q",
            tooltip=[alt.Tooltip("order:Q", title="ball-pass harmonic",
                                 format=".2f")])
        return line + rules
    return line


def _split_variants(bg: Optional[pd.DataFrame], harmonics: list,
                    x_title: str, y_title: str) -> dict:
    """{'By speed': chart, 'By specimen': chart, 'By direction': chart} from
    the tidy by-group spectrum CSV."""
    if bg is None or bg.empty:
        return {}
    label_for = {"speed": "By speed", "sample": "By specimen",
                 "direction": "By direction"}
    variants = {}
    for dim, sub in bg.groupby("dimension"):
        chart = alt.Chart(sub).mark_line().encode(
            x=alt.X("order:Q", title=x_title),
            y=alt.Y("cumulative:Q", title=y_title),
            color=alt.Color("value:N", legend=alt.Legend(
                title=dim.capitalize())),
            strokeDash=alt.StrokeDash("material:N",
                                      legend=alt.Legend(title="Material")),
            tooltip=[alt.Tooltip("order:Q", format=".2f"),
                     alt.Tooltip("cumulative:Q", format=".3g"),
                     alt.Tooltip("value:N", title=dim),
                     alt.Tooltip("material:N")],
        ).properties(height=280)
        variants[label_for.get(dim, dim)] = chart
    return variants


# ─────────────────────────────────────────────────────────────────────────────
#  Per-view builders
# ─────────────────────────────────────────────────────────────────────────────

def _view_data_quality(out_root: str, seg_dir: str,
                       requested_parts, requested_rpms) -> StageView:
    from . import dataset

    v = StageView("data_quality", "Data Quality",
                  purpose="Coverage audit: does the segmented data actually "
                          "contain every specimen and speed you asked for, "
                          "with balanced extend/retract strokes?")
    dq = dataset.data_quality(seg_dir, tuple(requested_parts or ()),
                              tuple(requested_rpms or ()))
    if dq.n_segments == 0:
        return v
    v.available = True
    v.takeaways = [
        {"label": "Actuations", "value": f"{dq.n_segments:,}",
         "help": "Individual strokes segmented out of the raw recordings."},
        {"label": "Parts × RPMs",
         "value": f"{len(dq.parts)} × {len(dq.rpms)}",
         "help": "Distinct specimens and speed classes present in the data."},
        {"label": "Flags", "value": str(len(dq.flags)) if dq.flags else "None",
         "help": "Coverage or balance problems that could bias comparisons."},
    ]
    if isinstance(dq.counts, pd.DataFrame) and not dq.counts.empty:
        c = dq.counts.rename(columns={"rpm_category": "rpm"})
        chart = alt.Chart(c).mark_rect().encode(
            x=alt.X("rpm:O", title="speed class (RPM)"),
            y=alt.Y("part:N", title="specimen"),
            color=alt.Color("n_segments:Q",
                            scale=alt.Scale(scheme="blues"),
                            legend=alt.Legend(title="actuations")),
            tooltip=["part:N", "rpm:O", "n_segments:Q",
                     "n_pos:Q", "n_neg:Q"],
        ).properties(height=30 + 28 * max(1, c["part"].nunique()))
        text = alt.Chart(c).mark_text(fontSize=11).encode(
            x="rpm:O", y="part:N", text="n_segments:Q",
            color=alt.value("#1a1a1e"))
        v.charts.append(("Actuation coverage — specimens × speed classes",
                         chart + text))
    for f in dq.flags:
        v.notes.append("⚠ " + f)
    if not dq.flags:
        v.notes.append("✅ Every requested specimen × speed combination has "
                       "data and stroke directions are balanced.")
    return v


def _view_segment(out_root: str, seg_dir: str) -> StageView:
    from . import dataset

    v = StageView("segment", "Segmentation",
                  purpose="Splits each raw recording into individual "
                          "actuations (strokes) so every analysis compares "
                          "like with like — one stroke at a time.")
    files = dataset.list_segment_files(seg_dir)
    if not files:
        return v
    v.available = True
    cat = dataset.segment_catalog(seg_dir)
    n_seg = int(len(cat)) if cat is not None else 0
    v.takeaways = [
        {"label": "Recordings", "value": str(len(files)),
         "help": "Raw CSV files that produced segments."},
        {"label": "Actuations", "value": f"{n_seg:,}",
         "help": "Strokes extracted across all recordings."},
    ]
    if cat is not None and not cat.empty and "direction" in cat.columns:
        per_dir = cat.groupby("direction").size()
        v.notes.append(
            "Direction split: "
            + ", ".join(f"{k} = {int(n)}" for k, n in per_dir.items())
            + ". Alternating extend/retract strokes are expected.")
    return v


def _view_loudness(out_root: str) -> StageView:
    v = StageView("loudness", "Vibration Level",
                  purpose="How much does each specimen shake overall? "
                          "One RMS value per stroke (higher = louder), "
                          "compared metal vs plastic at every speed.")
    df = _read_csv(out_root, "output_loudness", "rms_per_segment.csv")
    if df is None:
        return v
    v.available = True

    trial = _read_csv(out_root, "output_loudness", "rms_trial_means.csv")

    # Plastic / metal ratio on trial means (the statistically honest unit).
    ratio_txt = "—"
    src = trial if trial is not None else df
    val_col = "trial_mean_rms" if trial is not None else "rms_combined"
    mats = src.groupby("material")[val_col].mean()
    if {"metal", "plastic"} <= set(mats.index) and mats["metal"] > 0:
        ratio_txt = f"{mats['plastic'] / mats['metal']:.2f}×"

    v.takeaways = [
        {"label": "Actuations", "value": f"{len(df):,}",
         "help": "Strokes with a valid RMS measurement."},
        {"label": "Plastic / metal", "value": ratio_txt,
         "help": "Ratio of mean vibration level (trial means). "
                 "Above 1× means plastic shakes more."},
        {"label": "Speed classes",
         "value": str(df["rpm_cat"].nunique()) if "rpm_cat" in df else "—"},
    ]

    if {"rpm_cat", "rms_combined", "material"} <= set(df.columns):
        pts = alt.Chart(df).mark_circle(size=34, opacity=0.45).encode(
            x=alt.X("rpm_cat:O", title="speed class (RPM)"),
            y=alt.Y("rms_combined:Q", title="combined RMS (m/s²)",
                    scale=alt.Scale(type="log")),
            color=_color_material(),
            xOffset="material:N",
            tooltip=["part:N", "rpm_cat:O", "direction:N",
                     alt.Tooltip("rms_combined:Q", format=".3g")],
        ).properties(height=300)
        mean_line = alt.Chart(df).mark_line(point=True, size=2.5).encode(
            x="rpm_cat:O",
            y=alt.Y("mean(rms_combined):Q"),
            color=_color_material(),
            xOffset="material:N",
        )
        v.charts.append(
            ("Vibration level by speed — every dot is one stroke; the line "
             "is the mean", pts + mean_line))

        variants = {}
        if "part" in df.columns:
            variants["By specimen"] = alt.Chart(df).mark_boxplot().encode(
                x=alt.X("part:N", title="specimen"),
                y=alt.Y("rms_combined:Q", title="combined RMS (m/s²)",
                        scale=alt.Scale(type="log")),
                color=_color_material(),
            ).properties(height=300)
        if "direction" in df.columns:
            variants["By direction"] = alt.Chart(df).mark_boxplot().encode(
                x=alt.X("direction:N", title="stroke direction"),
                y=alt.Y("rms_combined:Q", title="combined RMS (m/s²)",
                        scale=alt.Scale(type="log")),
                color=_color_material(),
                column=alt.Column("rpm_cat:O", title="speed class (RPM)"),
            ).properties(height=250)
        if variants:
            v.chart_variants.append(("Split the comparison", variants))
    return v


def _view_order(out_root: str) -> StageView:
    v = StageView("order", "Order Analysis",
                  purpose="Vibration mapped to shaft position instead of "
                          "time: a peak at order N means 'N times per "
                          "revolution' — the fingerprint that separates "
                          "ball-pass defects from ordinary shaft rotation.")
    meta = _read_json(out_root, "output_order", "run_metadata.json")
    spec = _spectrum_df(out_root, "output_order", "order_spectra_mean")
    if spec.empty:
        return v
    v.available = True

    harmonics = list(meta.get("bpf_harmonics") or [])
    v.takeaways = [
        {"label": "Actuations", "value": str(meta.get("n_actuations", "—")),
         "help": "Strokes averaged into the mean spectrum."},
        {"label": "Ball-pass order",
         "value": f"{meta.get('ball_pass_order', '—')}",
         "help": "Where a recirculation defect must appear: this many "
                 "impacts per shaft revolution."},
        {"label": "Mean revolutions",
         "value": f"{meta.get('mean_revolutions', '—')}",
         "help": "Average stroke length. Order resolution ≈ 1 / revolutions, "
                 "so short strokes mean coarse order lines."},
    ]
    v.charts.append(
        ("Mean order spectrum (X+Y+Z combined) — dashed lines mark ball-pass "
         "harmonics", _spectrum_chart(spec, harmonics, "order (events per "
                                      "revolution)", "amplitude (m/s²)")))

    bg = _read_csv(out_root, "output_order", "order_spectra_by_group.csv")
    variants = _split_variants(bg, harmonics,
                               "order (events per revolution)",
                               "amplitude (m/s²)")
    if variants:
        v.chart_variants.append(("Split the spectrum", variants))

    camp_csv = os.path.join(out_root, "output_order", "order_campbell.csv")
    _campbell_heatmap(v, os.path.join(out_root, "output_order"), camp_csv,
                      "Campbell map — order content across the measured "
                      "speed sweep", x_title="order")
    if os.path.isfile(camp_csv):
        try:
            cm = pd.read_csv(camp_csv)
            rpm_bin = float(meta.get("config", {}).get("campbell_rpm_bin",
                                                       50.0) or 50.0)
            _campbell_resolution_note(v, cm, rpm_bin)
        except Exception:
            pass

    diag = _read_csv(out_root, "output_order", "order_diagnostics_summary.csv")
    if diag is not None and "native_order_res_median" in diag.columns:
        worst = float(diag["native_order_res_median"].max())
        if worst >= 0.15:
            v.notes.append(
                f"⚠ Native order resolution is limited to ≈ {worst:.2f} "
                "order by short strokes — peaks closer together than that "
                "(e.g. ball-pass 5.35 vs shaft order 5) cannot be cleanly "
                "separated. Longer constant-speed dwell at acquisition is "
                "the only fix.")
    return v


def _view_envelope(out_root: str) -> StageView:
    v = StageView("envelope", "Envelope Analysis",
                  purpose="Listens for repetitive impacts: filters to the "
                          "structural resonance the impacts ring, then reads "
                          "the impact repetition rate in orders. A line at "
                          "the ball-pass order = a real recirculation "
                          "defect signature.")
    spec = _spectrum_df(out_root, "output_envelope", "envelope_spectra_mean")
    meta = _read_json(out_root, "output_envelope", "run_metadata.json")
    if spec.empty:
        return v
    v.available = True

    harmonics = list(meta.get("bpf_harmonics") or [])
    n_parts = len(meta.get("part_bands") or {})
    v.takeaways = [
        {"label": "Segments demodulated",
         "value": str(meta.get("n_processed", "—"))},
        {"label": "Resonance bands", "value": str(n_parts) or "—",
         "help": "One fixed band per specimen (a structural resonance does "
                 "not move with speed), so by-speed spectra stay comparable."},
        {"label": "Band scope", "value": str(meta.get("band_scope", "—"))},
    ]
    v.charts.append(
        ("Mean envelope order spectrum — dashed lines mark ball-pass "
         "harmonics", _spectrum_chart(spec, harmonics,
                                      "envelope order (impacts per "
                                      "revolution)", "envelope amplitude")))

    bg = _read_csv(out_root, "output_envelope",
                   "envelope_spectra_by_group.csv")
    variants = _split_variants(bg, harmonics,
                               "envelope order (impacts per revolution)",
                               "envelope amplitude")
    if variants:
        v.chart_variants.append(("Split the spectrum", variants))

    bands = _read_csv(out_root, "output_envelope", "part_bands.csv")
    if bands is not None and {"part", "f_low_hz", "f_high_hz"} <= set(bands.columns):
        bc = alt.Chart(bands).mark_bar(height=14, cornerRadius=3).encode(
            x=alt.X("f_low_hz:Q", title="frequency (Hz)"),
            x2="f_high_hz:Q",
            y=alt.Y("part:N", title="specimen"),
            color=_color_material(),
            tooltip=["part:N", "material:N",
                     alt.Tooltip("f_low_hz:Q", format=".0f"),
                     alt.Tooltip("f_high_hz:Q", format=".0f"),
                     alt.Tooltip("kurtosis:Q", format=".1f"),
                     "rpm_used:Q"],
        ).properties(height=30 + 24 * max(1, bands["part"].nunique()))
        v.charts.append(
            ("Per-specimen resonance band — the frequency window each "
             "specimen's impacts ring", bc))

    grid = _read_csv(out_root, "output_envelope", "kurtogram_grid.csv")
    if grid is not None and {"f_center_hz", "level", "kurtosis"} <= set(grid.columns):
        heat = alt.Chart(grid).mark_rect().encode(
            x=alt.X("f_center_hz:Q", bin=alt.Bin(maxbins=60),
                    title="band center (Hz)"),
            y=alt.Y("level:O", title="kurtogram level (finer ↓)"),
            color=alt.Color("kurtosis:Q", aggregate="max",
                            scale=alt.Scale(scheme="inferno"),
                            legend=alt.Legend(title="kurtosis")),
            tooltip=[alt.Tooltip("f_center_hz:Q", bin=alt.Bin(maxbins=60)),
                     "level:O",
                     alt.Tooltip("kurtosis:Q", aggregate="max",
                                 format=".2f")],
        ).properties(height=200)
        if "part" in grid.columns and grid["part"].nunique() > 1:
            heat = heat.facet(facet=alt.Facet("part:N", title=None),
                              columns=2)
        v.charts.append(
            ("Kurtogram — a sharp hot cell means a robust impact band; a "
             "flat field means the band is latching onto noise", heat))

    camp_csv = os.path.join(out_root, "output_envelope",
                            "envelope_campbell.csv")
    _campbell_heatmap(v, os.path.join(out_root, "output_envelope"), camp_csv,
                      "Envelope Campbell map — impact content across the "
                      "speed sweep", x_title="envelope order")

    v.notes.append(
        "⚠ Envelope analysis is only meaningful on METAL specimens: "
        "plastic's damping suppresses the impact ringing, so its kurtogram "
        "latches onto noise (handover doc §6). Read plastic envelope lines "
        "as noise, not defects.")
    return v


def _view_torque(out_root: str) -> StageView:
    v = StageView("torque", "Torque & Efficiency",
                  purpose="Mechanical cost: average drive torque per speed "
                          "class and an efficiency proxy, metal vs plastic. "
                          "Steady-state only — ramps are gated out.")
    tq = _read_csv(out_root, "output_torque", "torque_agg.csv")
    ef = _read_csv(out_root, "output_efficiency", "efficiency_agg.csv")
    if tq is None and ef is None:
        return v
    v.available = True

    def _bar(df: pd.DataFrame, ycol: str, ecol: str, y_title: str):
        df = df.rename(columns={"material_type": "material",
                                "rpm_category": "rpm"})
        bars = alt.Chart(df).mark_bar().encode(
            x=alt.X("rpm:O", title="speed class (RPM)"),
            xOffset="material:N",
            y=alt.Y(f"{ycol}:Q", title=y_title),
            color=_color_material(),
            tooltip=["material:N", "rpm:O",
                     alt.Tooltip(f"{ycol}:Q", format=".3g"),
                     alt.Tooltip(f"{ecol}:Q", format=".3g",
                                 title="±1 SD") if ecol in df else
                     alt.Tooltip(f"{ycol}:Q", format=".3g"),
                     "n_segments:Q" if "n_segments" in df else "rpm:O"],
        ).properties(height=280)
        if ecol in df.columns:
            df = df.assign(_lo=df[ycol] - df[ecol].fillna(0),
                           _hi=df[ycol] + df[ecol].fillna(0))
            err = alt.Chart(df).mark_rule(color="#333").encode(
                x="rpm:O", xOffset="material:N", y="_lo:Q", y2="_hi:Q")
            return bars + err
        return bars

    if tq is not None and "mean_torque_nm" in tq.columns:
        v.charts.append(
            ("Mean |torque| by speed class — error bars are ±1 SD across "
             "strokes", _bar(tq, "mean_torque_nm", "std_torque_nm",
                             "mean |torque| (Nm)")))
        piv = tq.pivot_table(index="rpm_category", columns="material_type",
                             values="mean_torque_nm", aggfunc="mean")
        if {"metal", "plastic"} <= set(piv.columns):
            r = (piv["plastic"] / piv["metal"]).mean()
            v.takeaways.append(
                {"label": "Plastic / metal torque", "value": f"{r:.2f}×",
                 "help": "Mean drag-torque ratio across speed classes."})
    if ef is not None and "eff_proxy_mean" in ef.columns:
        v.charts.append(
            ("Efficiency proxy by speed class",
             _bar(ef, "eff_proxy_mean", "eff_proxy_std",
                  "efficiency proxy")))
    return v


def _view_summary(out_root: str) -> StageView:
    v = StageView("summary", "Summary",
                  purpose="The cross-stage one-pager: vibration level, "
                          "dominant orders, resonance bands and data-quality "
                          "flags per material — ready to paste into a "
                          "report.")
    ch = _read_csv(out_root, "output_handover", "characteristics.csv")
    if ch is None:
        return v
    v.available = True
    if "plastic_metal_rms_ratio" in ch.columns:
        r = ch["plastic_metal_rms_ratio"].dropna()
        if not r.empty:
            v.takeaways.append(
                {"label": "Plastic / metal RMS", "value": f"{r.iloc[0]:.2f}×",
                 "help": "Overall vibration-level ratio. Above 1× means "
                         "plastic shakes more."})
    if "rms_mean_m_s2" in ch.columns and "material" in ch.columns:
        bar = alt.Chart(ch).mark_bar().encode(
            x=alt.X("material:N", title=None),
            y=alt.Y("rms_mean_m_s2:Q", title="mean RMS (m/s²)"),
            color=_color_material(),
            tooltip=list(ch.columns),
        ).properties(height=220)
        v.charts.append(("Vibration level by material", bar))
    return v


# ─────────────────────────────────────────────────────────────────────────────
#  Assembly
# ─────────────────────────────────────────────────────────────────────────────

# Which manifest stage keys feed each view (for file routing).
_VIEW_STAGES = {
    "data_quality": (),
    "segment": ("segment",),
    "loudness": ("vibration", "loudness", "loudness_plot",
                 "loudness_by_sample"),
    "order": ("order",),
    "envelope": ("envelope", "bandpass"),
    "torque": ("efficiency", "torque"),
    "summary": ("report", "sample_summary", "characteristics",
                "actuation_plot"),
}

_FIG_EXTS = (".png", ".pdf")
_TABLE_EXTS = (".csv", ".html")


def build_views(manifest: dict, out_root: str, seg_dir: str,
                requested_parts=(), requested_rpms=()) -> list:
    """Build every StageView for one finished run (read-only)."""
    views = [
        _view_data_quality(out_root, seg_dir, requested_parts,
                           requested_rpms),
        _view_segment(out_root, seg_dir),
        _view_loudness(out_root),
        _view_order(out_root),
        _view_envelope(out_root),
        _view_torque(out_root),
        _view_summary(out_root),
    ]

    files_by_stage = {rec.get("stage"): rec.get("files") or []
                      for rec in (manifest or {}).get("stages", [])}
    for view in views:
        for stage_key in _VIEW_STAGES.get(view.key, ()):
            for f in files_by_stage.get(stage_key, []):
                ext = os.path.splitext(f)[1].lower()
                if ext in _FIG_EXTS and f not in view.figure_files:
                    view.figure_files.append(f)
                elif ext in _TABLE_EXTS and f not in view.table_files:
                    view.table_files.append(f)
    return views
