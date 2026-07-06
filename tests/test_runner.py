"""Tests for nvh_pipeline/runner.py — error handling and output collection."""

from __future__ import annotations

import os
import time

import pytest

from nvh_pipeline.runner import STAGES, _collect_outputs, run_stage


class TestRunStage:
    def test_unknown_stage_returns_error_dict(self):
        from nvh_pipeline.config import PipelineConfig
        result = run_stage("no_such_stage_xyz", PipelineConfig())
        assert result["status"] == "error"
        assert result["stage"] == "no_such_stage_xyz"
        assert "unknown stage" in (result.get("error") or "").lower()
        assert result["files"] == []

    def test_known_stages_are_registered(self):
        # Stages were consolidated: 12 individual modules → 6 grouped stages
        expected = {"segment", "vibration", "efficiency", "order", "envelope", "report"}
        assert expected == set(STAGES.keys())


class TestCollectOutputs:
    def test_returns_only_result_extensions(self, tmp_path):
        (tmp_path / "result.png").write_bytes(b"fake png")
        (tmp_path / "data.csv").write_bytes(b"a,b\n1,2\n")
        (tmp_path / "ignored.txt").write_bytes(b"not a result")
        since = time.time() - 1   # everything written before "now"
        found = _collect_outputs(str(tmp_path), ["."], since)
        names = {os.path.basename(f) for f in found}
        assert "result.png" in names
        assert "data.csv" in names
        assert "ignored.txt" not in names

    def test_respects_since_timestamp(self, tmp_path):
        old = tmp_path / "old.png"
        old.write_bytes(b"old")
        # Set mtime to past
        old_ts = time.time() - 10
        os.utime(str(old), (old_ts, old_ts))
        since = time.time() - 5   # only files newer than this
        found = _collect_outputs(str(tmp_path), ["."], since)
        assert str(old.resolve()) not in found

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        found = _collect_outputs(str(tmp_path / "missing"), ["."], 0.0)
        assert found == []

    def test_nested_subdirectories_are_walked(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.png").write_bytes(b"png")
        since = time.time() - 1
        found = _collect_outputs(str(tmp_path), ["."], since)
        assert any("nested.png" in f for f in found)
