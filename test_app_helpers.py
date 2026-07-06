"""Tests for nvh_pipeline/app_helpers.py — pure UI helper functions."""

from __future__ import annotations

import json
import os
import re

import pytest

from nvh_pipeline.app_helpers import (
    as_str_list,
    csv_list,
    default_for,
    generate_run_name,
    scan_past_runs,
)


# ─────────────────────────────────────────────────────────────────────────────
#  csv_list
# ─────────────────────────────────────────────────────────────────────────────

class TestCsvList:
    def test_basic(self):
        assert csv_list("a, b , c") == ["a", "b", "c"]

    def test_empty_string(self):
        assert csv_list("") == []

    def test_trailing_comma(self):
        assert csv_list("x,y,") == ["x", "y"]

    def test_single_item(self):
        assert csv_list("only") == ["only"]


# ─────────────────────────────────────────────────────────────────────────────
#  as_str_list
# ─────────────────────────────────────────────────────────────────────────────

class TestAsStrList:
    def test_list_input(self):
        assert as_str_list(["a", "b", "c"]) == ["a", "b", "c"]

    def test_tuple_input(self):
        assert as_str_list(("x", "y")) == ["x", "y"]

    def test_string_input(self):
        assert as_str_list("a, b , c") == ["a", "b", "c"]

    def test_filters_empty_items(self):
        assert as_str_list(["a", "", "  ", "b"]) == ["a", "b"]


# ─────────────────────────────────────────────────────────────────────────────
#  generate_run_name
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateRunName:
    _PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}_.+_.+$")

    def test_pattern_matches(self):
        name = generate_run_name(["m627"], [100])
        assert self._PATTERN.match(name), f"Unexpected name: {name}"

    def test_parts_truncated_at_three(self):
        name = generate_run_name(["a", "b", "c", "d"], [100])
        assert "+" in name

    def test_no_parts(self):
        name = generate_run_name([], [100])
        assert "noparts" in name

    def test_no_rpms(self):
        name = generate_run_name(["m627"], [])
        assert "norpm" in name

    def test_rpms_sorted(self):
        name1 = generate_run_name(["m627"], [2300, 100])
        name2 = generate_run_name(["m627"], [100, 2300])
        assert name1 == name2


# ─────────────────────────────────────────────────────────────────────────────
#  scan_past_runs
# ─────────────────────────────────────────────────────────────────────────────

class TestScanPastRuns:
    def test_empty_string_root(self):
        assert scan_past_runs("") == []

    def test_nonexistent_root(self, tmp_path):
        assert scan_past_runs(str(tmp_path / "missing")) == []

    def test_empty_root(self, tmp_path):
        assert scan_past_runs(str(tmp_path)) == []

    def test_finds_manifest(self, results_root_with_run):
        runs = scan_past_runs(results_root_with_run)
        assert len(runs) == 1
        run = runs[0]
        assert run["name"] == "2024-01-01_1200_m627_100rpm"
        assert run["stages_ok"] == 2
        assert run["stages_total"] == 2
        assert run["parts"] == ["m627"]
        assert run["rpms"] == [100]
        assert os.path.isfile(run["manifest"])

    def test_skips_dirs_without_manifest(self, tmp_path):
        (tmp_path / "no_manifest").mkdir()
        (tmp_path / "no_manifest" / "output.png").write_bytes(b"fake")
        assert scan_past_runs(str(tmp_path)) == []

    def test_sorted_newest_first(self, tmp_path):
        for name, ts in [
            ("run_A", "2024-01-01 10:00:00"),
            ("run_B", "2024-03-01 10:00:00"),
        ]:
            d = tmp_path / name
            d.mkdir()
            m = {"generated": ts, "stages": []}
            (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        runs = scan_past_runs(str(tmp_path))
        assert runs[0]["name"] == "run_B"
        assert runs[1]["name"] == "run_A"


# ─────────────────────────────────────────────────────────────────────────────
#  default_for
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaultFor:
    def test_exact_match_returned(self):
        opts = ["Time (s)", "RPM", "Accel X"]
        assert default_for("RPM", opts) == "RPM"

    def test_hint_fallback_when_default_absent(self):
        opts = ["Time (s)", "CNT 1/Frequency (RPM)", "X Accel (m/s2)"]
        result = default_for("no_match", opts, "rpm", "frequency")
        assert result == "CNT 1/Frequency (RPM)"

    def test_first_option_fallback(self):
        opts = ["ColA", "ColB", "ColC"]
        result = default_for("missing", opts, "nohint")
        assert result == "ColA"

    def test_empty_options_returns_default(self):
        assert default_for("saved_col", [], "hint") == "saved_col"

    def test_hint_case_insensitive(self):
        opts = ["X ACCEL"]
        assert default_for("", opts, "accel") == "X ACCEL"
