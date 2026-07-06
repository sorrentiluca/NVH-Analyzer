"""
PipelineConfig — the single source of truth for every stage.

One dataclass holds all user-editable settings.  It round-trips through
``nvh_config.json`` (``from_json`` / ``to_json``), can be discovered from the
``NVH_CONFIG`` environment variable or the working directory, and is injected
into each stage's local ``Config`` dataclass via :func:`apply_to_stage` — the
stage scripts keep their own defaults so they still run bare, but a pipeline
run overrides them consistently from one place.

Path helpers (:func:`seg_dir`, :func:`seg_data_dir`, :func:`stage_out`) root
every stage's output under ``output_root`` so nothing depends on the process
working directory.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field, fields
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  The config dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # ── IO ───────────────────────────────────────────────────────────────────
    # Folder of raw DEWESOFT CSVs named <part>-<rpm>rpm-<trial>.csv.
    data_dir: str = "data"
    # Every output_* folder is created under this root.
    output_root: str = "."

    # ── CAMPAIGN ────────────────────────────────────────────────────────────
    # Part identifiers; material is auto-detected from the prefix
    # ('m*' → metal, 'T*' → plastic).
    parts: tuple = ("m627", "m625", "m635", "T8", "T11", "T13")
    rpms: tuple = (100, 200, 400, 800, 1600, 2300)

    # ── ANALYSIS MODE ────────────────────────────────────────────────────────
    # 'reciprocating' (ballscrew rig: a folder of files, stroke segmentation) or
    # 'continuous' (one steady rotating-machine recording analysed whole).
    analysis_mode: str = "reciprocating"
    single_file: Optional[str] = None     # continuous: the one CSV to analyse
    nominal_rpm: Optional[float] = None   # continuous: steady speed (None = auto)
    signal_label: str = "signal"          # continuous: label instead of part id

    # ── COLUMNS ─────────────────────────────────────────────────────────────
    time_col: str = "Time (s)"
    rpm_col: str = "CNT 1/Frequency (RPM)"
    angle_col: str = "CNT 1/Angle (Degrees)"
    torque_col: str = "Torque (Nm)"

    # Accelerometer columns.  The Y/Z swap is intentional: physical Y is logged
    # under the "Z Accel" column and vice versa (handover doc §7).
    accel_cols: dict = field(default_factory=lambda: {
        "X": "X Accel (m/s2)",
        "Y": "Z Accel (m/s2)",
        "Z": "Y Accel (m/s2)",
    })
    material_colors: dict = field(default_factory=lambda: {
        "metal":   "#1f77b4",
        "plastic": "#ff7f0e",
    })

    # ── SIGNAL CONDITIONING ─────────────────────────────────────────────────
    # Remove the per-segment DC offset before RMS (AC RMS) so sensor bias can't
    # inflate the level comparison.
    remove_dc: bool = True

    # ── ORDER / ENVELOPE ────────────────────────────────────────────────────
    samples_per_rev: int = 128
    ball_pass_order: float = 5.35
    n_bpf_harmonics: int = 4
    bpfo_band_halfwidth: float = 0.25
    # Minimum shaft revolutions per actuation to include it in order/envelope.
    min_revolutions: float = 3.0

    # ── KURTOGRAM / BANDPASS ────────────────────────────────────────────────
    bp_min_hz: float = 1000.0        # search floor at max RPM (scaled per stroke)
    bp_min_hz_abs: float = 50.0      # absolute floor — never search below this
    bp_min_bw_hz: float = 200.0      # candidate-band minimum bandwidth
    bp_max_hz: float = 12000.0       # accelerometer calibrated-range ceiling
    kurtogram_levels: int = 6
    bp_filter_order: int = 4
    min_kurtosis: float = 3.0        # impulsiveness gate (Gaussian baseline = 3)
    normalize_segments: bool = True
    # 'per_part' = one resonance band per specimen (comparable by-speed spectra);
    # 'per_segment' = legacy per-stroke re-detection.
    band_scope: str = "per_part"

    # ── STEADY-STATE GATING / AGGREGATION ───────────────────────────────────
    plateau_gating: bool = True
    plateau_frac: float = 0.90
    order_average: str = "power"     # 'power' = RMS spectrum, 'magnitude' = mean
    cumulative_mode: str = "rss"     # 'rss' = sqrt(x²+y²+z²), 'sum' = linear

    # ── CAMPBELL (order × measured-RPM waterfall over the ramp sweep) ───────
    campbell: bool = True
    campbell_window_rev: float = 1.5
    campbell_hop_rev: float = 0.75
    campbell_rpm_bin: float = 50.0
    campbell_order_step: float = 0.1

    # ── FILTERS ─────────────────────────────────────────────────────────────
    only_parts: Optional[tuple] = None
    only_rpms: Optional[tuple] = None

    # ── OUTPUT BEHAVIOUR ────────────────────────────────────────────────────
    show_plots: bool = False
    render_plots: bool = True

    # Stage groups to run (canonical runner order).
    stages: tuple = ("segment", "vibration", "efficiency",
                     "order", "envelope", "report")

    # ── Serialization ────────────────────────────────────────────────────────

    _TUPLE_FIELDS = ("parts", "rpms", "only_parts", "only_rpms", "stages")

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Build a config from a plain dict; unknown keys are ignored and
        list-valued tuple fields are normalised to tuples (None preserved)."""
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in (d or {}).items() if k in known}
        for name in cls._TUPLE_FIELDS:
            if name in kwargs and kwargs[name] is not None:
                kwargs[name] = tuple(kwargs[name])
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_json(self, path: str) -> None:
        d = self.to_dict()
        for name in self._TUPLE_FIELDS:      # tuples → lists for clean JSON
            if d.get(name) is not None:
                d[name] = list(d[name])
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
#  Active-config discovery
# ─────────────────────────────────────────────────────────────────────────────
# Precedence: explicit set_active() > $NVH_CONFIG file > ./nvh_config.json >
# built-in defaults.  Stage scripts call active() (or apply_to_stage with
# p=None) so a bare `python EP_loudness.py` run picks up the same file the
# pipeline used.

_ACTIVE: Optional[PipelineConfig] = None


def set_active(cfg: Optional[PipelineConfig]) -> None:
    global _ACTIVE
    _ACTIVE = cfg


def _discover_config_file() -> Optional[str]:
    env = os.environ.get("NVH_CONFIG")
    if env and os.path.isfile(env):
        return env
    local = os.path.join(os.getcwd(), "nvh_config.json")
    if os.path.isfile(local):
        return local
    return None


def active() -> PipelineConfig:
    """The pipeline config in effect for this process (see precedence above)."""
    if _ACTIVE is not None:
        return _ACTIVE
    path = _discover_config_file()
    if path:
        cfg = PipelineConfig.from_json(path)
        set_active(cfg)
        return cfg
    return PipelineConfig()


# ─────────────────────────────────────────────────────────────────────────────
#  Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def seg_dir(p: PipelineConfig) -> str:
    """EP_segment's output directory (parquet + plots live below it)."""
    return os.path.join(p.output_root, "output_seg")


def seg_data_dir(p: PipelineConfig) -> str:
    """The segmented-parquet folder every analysis stage reads."""
    return os.path.join(seg_dir(p), "segmented_data")


def stage_out(p: PipelineConfig, dirname: str) -> str:
    """A stage's output folder under output_root (e.g. 'output_order')."""
    return os.path.join(p.output_root, dirname)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-stage injection
# ─────────────────────────────────────────────────────────────────────────────

def _set(obj, **values) -> None:
    """Assign each value onto obj only when obj carries that attribute — a
    stage that doesn't know a setting simply doesn't receive it."""
    for name, value in values.items():
        if hasattr(obj, name):
            setattr(obj, name, value)


def _common_display(obj, p: PipelineConfig) -> None:
    _set(obj,
         show_plots=bool(p.show_plots),
         show_plot=bool(p.show_plots),       # EP_bandpass names it show_plot
         render_plots=bool(p.render_plots))


def _axis_overrides(p: PipelineConfig) -> dict:
    """Explicit axis columns for order/envelope — ONLY in continuous mode.

    Reciprocating rig data auto-detects the X/Y/Z accel columns by literal name
    (the Y/Z logger swap therefore does not change the summed totals — §7).
    Continuous single-channel data can carry arbitrarily named channels, so the
    user's explicit mapping is threaded through instead.
    """
    if p.analysis_mode != "continuous":
        return {}
    return {
        "axis_x": (p.accel_cols or {}).get("X") or None,
        "axis_y": (p.accel_cols or {}).get("Y") or None,
        "axis_z": (p.accel_cols or {}).get("Z") or None,
    }


def apply_to_stage(stage: str, obj, p: Optional[PipelineConfig] = None) -> None:
    """Overlay the pipeline config onto one stage's local Config object.

    ``stage`` is the fine-grained stage name each EP_*.py passes (not the
    runner's group name).  Unknown stages are a no-op.  ``p=None`` uses the
    active / discovered config.
    """
    if p is None:
        p = active()

    if stage == "segment":
        _set(obj,
             base_path=p.data_dir,
             parts=tuple(p.parts),
             rpms=tuple(p.rpms),
             output_dir=seg_dir(p),
             time_col=p.time_col,
             rpm_col=p.rpm_col,
             angle_col=p.angle_col,
             analysis_mode=p.analysis_mode,
             single_file=p.single_file,
             nominal_rpm=p.nominal_rpm,
             signal_label=p.signal_label)
        _common_display(obj, p)

    elif stage in ("loudness", "loudness_by_sample"):
        out = ("output_loudness" if stage == "loudness"
               else "output_loudness_samples")
        _set(obj,
             segmented_data_dir=seg_data_dir(p),
             accel_cols=dict(p.accel_cols),
             material_colors=dict(p.material_colors),
             output_dir=stage_out(p, out),
             time_col=p.time_col,
             remove_dc=bool(p.remove_dc))
        _common_display(obj, p)

    elif stage == "order":
        _set(obj,
             seg_output_dir=seg_dir(p),
             output_dir=stage_out(p, "output_order"),
             time_col=p.time_col,
             rpm_col=p.rpm_col,
             angle_col=p.angle_col,
             samples_per_rev=int(p.samples_per_rev),
             ball_pass_order=float(p.ball_pass_order),
             n_bpf_harmonics=int(p.n_bpf_harmonics),
             bpfo_band_halfwidth=float(p.bpfo_band_halfwidth),
             min_revolutions=float(p.min_revolutions),
             only_parts=tuple(p.only_parts) if p.only_parts else None,
             only_rpms=tuple(p.only_rpms) if p.only_rpms else None,
             plateau_gating=bool(p.plateau_gating),
             plateau_frac=float(p.plateau_frac),
             order_average=p.order_average,
             cumulative_mode=p.cumulative_mode,
             campbell=bool(p.campbell),
             campbell_window_rev=float(p.campbell_window_rev),
             campbell_hop_rev=float(p.campbell_hop_rev),
             campbell_rpm_bin=float(p.campbell_rpm_bin),
             campbell_order_step=float(p.campbell_order_step),
             analysis_mode=p.analysis_mode,
             **_axis_overrides(p))
        _common_display(obj, p)

    elif stage == "envelope":
        _set(obj,
             seg_output_dir=seg_dir(p),
             output_dir=stage_out(p, "output_envelope"),
             time_col=p.time_col,
             rpm_col=p.rpm_col,
             angle_col=p.angle_col,
             samples_per_rev=int(p.samples_per_rev),
             ball_pass_order=float(p.ball_pass_order),
             n_bpf_harmonics=int(p.n_bpf_harmonics),
             min_revolutions=float(p.min_revolutions),
             bp_min_hz=float(p.bp_min_hz),
             bp_min_hz_abs=float(p.bp_min_hz_abs),
             bp_min_bw_hz=float(p.bp_min_bw_hz),
             bp_max_hz=float(p.bp_max_hz),
             kurtogram_levels=int(p.kurtogram_levels),
             bp_filter_order=int(p.bp_filter_order),
             min_kurtosis=float(p.min_kurtosis),
             normalize_segments=bool(p.normalize_segments),
             band_scope=p.band_scope,
             only_rpms=tuple(p.only_rpms) if p.only_rpms else None,
             plateau_gating=bool(p.plateau_gating),
             plateau_frac=float(p.plateau_frac),
             order_average=p.order_average,
             cumulative_mode=p.cumulative_mode,
             campbell=bool(p.campbell),
             campbell_window_rev=float(p.campbell_window_rev),
             campbell_hop_rev=float(p.campbell_hop_rev),
             campbell_rpm_bin=float(p.campbell_rpm_bin),
             campbell_order_step=float(p.campbell_order_step),
             analysis_mode=p.analysis_mode,
             **_axis_overrides(p))
        _common_display(obj, p)

    elif stage == "bandpass":
        rpm_pool = p.only_rpms or p.rpms
        _set(obj,
             seg_output_dir=seg_dir(p),
             output_dir=stage_out(p, "output_envelope"),
             time_col=p.time_col,
             rpm_col=p.rpm_col,
             angle_col=p.angle_col,
             bp_min_hz=float(p.bp_min_hz),
             bp_min_hz_abs=float(p.bp_min_hz_abs),
             bp_min_bw_hz=float(p.bp_min_bw_hz),
             kurtogram_levels=int(p.kurtogram_levels),
             bp_filter_order=int(p.bp_filter_order),
             ball_pass_order=float(p.ball_pass_order),
             target_rpm=int(max(rpm_pool)) if rpm_pool else None,
             analysis_mode=p.analysis_mode,
             **_axis_overrides(p))
        _common_display(obj, p)

    elif stage == "sample_summary":
        _set(obj,
             segmented_data_dir=seg_data_dir(p),
             output_dir=stage_out(p, "output_handover"))
        _common_display(obj, p)

    elif stage == "actuation_plot":
        rpm_pool = p.only_rpms or p.rpms
        _set(obj,
             segmented_data_dir=seg_data_dir(p),
             output_dir=stage_out(p, "output_actuation_plot"),
             rpm_col=p.rpm_col,
             angle_col=p.angle_col,
             torque_col=p.torque_col,
             accel_cols=dict(p.accel_cols),
             rpm_category=int(max(rpm_pool)) if rpm_pool else None)
        _common_display(obj, p)

    elif stage == "characteristics":
        _set(obj, output_root=p.output_root)

    # Unknown stage: deliberately a no-op (forward compatibility).


# ─────────────────────────────────────────────────────────────────────────────
#  Example config
# ─────────────────────────────────────────────────────────────────────────────

def write_example(path: str) -> None:
    """Write a starter nvh_config.json with the built-in defaults."""
    PipelineConfig().to_json(path)
