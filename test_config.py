"""Tests for nvh_pipeline/config.py — PipelineConfig serialization and apply_to_stage."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import pytest

from nvh_pipeline.config import PipelineConfig, apply_to_stage, seg_dir, seg_data_dir


# ─────────────────────────────────────────────────────────────────────────────
#  PipelineConfig serialization round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineConfigSerialization:
    def test_from_dict_roundtrip(self):
        original = PipelineConfig()
        restored = PipelineConfig.from_dict(asdict(original))
        assert asdict(original) == asdict(restored)

    def test_from_dict_unknown_keys_ignored(self):
        d = asdict(PipelineConfig())
        d["completely_unknown_field"] = "ignored"
        cfg = PipelineConfig.from_dict(d)
        assert not hasattr(cfg, "completely_unknown_field")

    def test_from_dict_list_to_tuple_parts(self):
        d = asdict(PipelineConfig())
        d["parts"] = ["m627", "T8"]
        cfg = PipelineConfig.from_dict(d)
        assert isinstance(cfg.parts, tuple)
        assert cfg.parts == ("m627", "T8")

    def test_from_dict_list_to_tuple_rpms(self):
        d = asdict(PipelineConfig())
        d["rpms"] = [100, 800, 2300]
        cfg = PipelineConfig.from_dict(d)
        assert isinstance(cfg.rpms, tuple)
        assert cfg.rpms == (100, 800, 2300)

    def test_from_dict_only_parts_none_preserved(self):
        d = asdict(PipelineConfig())
        d["only_parts"] = None
        cfg = PipelineConfig.from_dict(d)
        assert cfg.only_parts is None

    def test_to_json_roundtrip(self, tmp_path):
        original = PipelineConfig(data_dir="/custom/data", bp_min_hz_abs=75.0)
        path = str(tmp_path / "nvh_config.json")
        original.to_json(path)
        restored = PipelineConfig.from_json(path)
        assert restored.data_dir == "/custom/data"
        assert restored.bp_min_hz_abs == pytest.approx(75.0)

    def test_bp_min_hz_abs_default(self):
        cfg = PipelineConfig()
        assert cfg.bp_min_hz_abs == pytest.approx(50.0)


# ─────────────────────────────────────────────────────────────────────────────
#  apply_to_stage
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _SegConfig:
    base_path: str = ""
    parts: tuple = ()
    rpms: tuple = ()
    output_dir: str = ""
    time_col: str = ""
    rpm_col: str = ""
    angle_col: str = ""
    show_plots: bool = True


@dataclass
class _LoudnessConfig:
    segmented_data_dir: str = ""
    accel_cols: dict = None
    material_colors: dict = None
    output_dir: str = ""
    time_col: str = ""
    rpm_col: str = ""
    show_plots: bool = True


@dataclass
class _EnvConfig:
    seg_output_dir: str = ""
    data_subdir: str = ""
    samples_per_rev: int = 0
    ball_pass_order: float = 0.0
    n_bpf_harmonics: int = 0
    bp_min_hz: float = 0.0
    bp_min_hz_abs: float = 0.0
    bp_min_bw_hz: float = 0.0
    kurtogram_levels: int = 0
    bp_filter_order: int = 0
    output_dir: str = ""
    show_plots: bool = True


class TestApplyToStage:
    def test_segment_stage(self, tmp_path):
        p = PipelineConfig(data_dir=str(tmp_path), output_root=str(tmp_path))
        obj = _SegConfig()
        apply_to_stage("segment", obj, p)
        assert obj.base_path == str(tmp_path)
        assert obj.parts == p.parts
        assert obj.rpms == p.rpms
        assert obj.output_dir == seg_dir(p)

    def test_loudness_stage(self, tmp_path):
        p = PipelineConfig(output_root=str(tmp_path))
        obj = _LoudnessConfig()
        apply_to_stage("loudness", obj, p)
        assert obj.segmented_data_dir == seg_data_dir(p)
        assert obj.accel_cols == dict(p.accel_cols)

    def test_envelope_stage_threads_bp_min_hz_abs(self, tmp_path):
        p = PipelineConfig(output_root=str(tmp_path), bp_min_hz_abs=123.0)
        obj = _EnvConfig()
        apply_to_stage("envelope", obj, p)
        assert obj.bp_min_hz_abs == pytest.approx(123.0)

    def test_unknown_stage_no_crash(self):
        p = PipelineConfig()

        @dataclass
        class _Dummy:
            some_attr: str = "original"

        obj = _Dummy()
        apply_to_stage("nonexistent_stage", obj, p)
        assert obj.some_attr == "original"

    def test_show_plots_forced_false_via_config(self):
        p = PipelineConfig(show_plots=False)
        obj = _SegConfig(show_plots=True)
        apply_to_stage("segment", obj, p)
        assert obj.show_plots is False
