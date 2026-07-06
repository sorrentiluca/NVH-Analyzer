"""Tests for nvh_pipeline/common.py — inspect_headers, discover_dataset, helpers."""

from __future__ import annotations

import os

import numpy as np
import pytest

from nvh_pipeline.common import (
    combined_axes_rms,
    discover_dataset,
    inspect_headers,
    material_type,
    per_axis_rms,
)


# ─────────────────────────────────────────────────────────────────────────────
#  inspect_headers
# ─────────────────────────────────────────────────────────────────────────────

class TestInspectHeaders:
    def test_consistent_folder(self, csv_dir_consistent):
        result = inspect_headers(csv_dir_consistent)
        assert result["n_files"] == 3
        assert result["n_mismatched"] == 0
        assert result["reference_columns"] == [
            "Time (s)", "CNT 1/Frequency (RPM)", "X Accel (m/s2)"
        ]
        assert result["all_columns"] == sorted([
            "Time (s)", "CNT 1/Frequency (RPM)", "X Accel (m/s2)"
        ])
        assert result["unreadable"] == []
        for fr in result["files"]:
            assert fr["matches_reference"] is True
            assert fr["missing"] == []
            assert fr["extra"] == []

    def test_mismatched_folder(self, csv_dir_mismatch):
        result = inspect_headers(csv_dir_mismatch)
        assert result["n_files"] == 3
        assert result["n_mismatched"] == 1
        # The file with fewer columns should be flagged
        flagged = [f for f in result["files"] if not f["matches_reference"]]
        assert len(flagged) == 1
        assert flagged[0]["name"] == "T8-2300rpm-1.csv"
        assert "X Accel (m/s2)" in flagged[0]["missing"]
        assert flagged[0]["extra"] == []

    def test_nonexistent_dir(self, tmp_path):
        result = inspect_headers(str(tmp_path / "does_not_exist"))
        assert result["n_files"] == 0
        assert result["n_mismatched"] == 0
        assert result["all_columns"] == []
        assert result["files"] == []

    def test_empty_csv(self, tmp_path):
        (tmp_path / "empty.csv").write_bytes(b"")
        result = inspect_headers(str(tmp_path))
        # Sole empty CSV becomes the (empty) reference — identical to itself,
        # so n_mismatched stays 0.  The important thing is it doesn't raise.
        assert result["n_files"] == 1
        assert result["n_mismatched"] == 0

    def test_empty_csv_among_valid_files(self, tmp_path):
        (tmp_path / "a.csv").write_text("Time (s),X Accel (m/s2)\n1,2\n")
        (tmp_path / "empty.csv").write_bytes(b"")
        result = inspect_headers(str(tmp_path))
        # The empty file should be classified as 'empty' against the real reference
        empty_rec = next(f for f in result["files"] if f["name"] == "empty.csv")
        assert empty_rec["status"] == "empty"
        assert not empty_rec["matches_reference"]
        assert result["n_mismatched"] == 1

    def test_all_columns_is_union(self, csv_dir_mismatch):
        result = inspect_headers(csv_dir_mismatch)
        expected_union = sorted(["Time (s)", "CNT 1/Frequency (RPM)", "X Accel (m/s2)"])
        assert result["all_columns"] == expected_union


# ─────────────────────────────────────────────────────────────────────────────
#  discover_dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestDiscoverDataset:
    def test_matched_files(self, csv_dir_grammar):
        result = discover_dataset(csv_dir_grammar)
        assert result["matched"] == 3
        assert set(result["parts"]) == {"m627", "T8"}
        assert set(result["rpms"]) == {100, 2300}
        assert result["n_files"] == 3  # README.txt is not a CSV

    def test_unmatched_files(self, tmp_path):
        (tmp_path / "badname.csv").write_text("h\n1\n", encoding="utf-8")
        result = discover_dataset(str(tmp_path))
        assert result["matched"] == 0
        assert "badname.csv" in result["unmatched"]

    def test_empty_dir(self, tmp_path):
        result = discover_dataset(str(tmp_path))
        assert result["n_files"] == 0
        assert result["parts"] == []
        assert result["rpms"] == []

    def test_nonexistent_dir(self, tmp_path):
        result = discover_dataset(str(tmp_path / "missing"))
        assert result["matched"] == 0

    def test_counts_per_part_rpm(self, csv_dir_grammar):
        result = discover_dataset(csv_dir_grammar)
        counts = {(c["part"], c["rpm"]): c["n_files"] for c in result["counts"]}
        assert counts[("m627", 100)] == 1
        assert counts[("m627", 2300)] == 1
        assert counts[("T8", 100)] == 1


# ─────────────────────────────────────────────────────────────────────────────
#  material_type
# ─────────────────────────────────────────────────────────────────────────────

class TestMaterialType:
    def test_metal_prefix(self):
        assert material_type("m627") == "metal"
        assert material_type("M100") == "metal"

    def test_plastic_prefix(self):
        assert material_type("T8") == "plastic"
        assert material_type("t13") == "plastic"

    def test_unknown_prefix(self):
        assert material_type("X99") == "unknown"
        assert material_type("") == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
#  RMS helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestRmsHelpers:
    def test_per_axis_rms_constant(self):
        sig = np.ones(100) * 3.0
        assert per_axis_rms(sig) == pytest.approx(3.0)

    def test_per_axis_rms_zero(self):
        assert per_axis_rms(np.zeros(50)) == pytest.approx(0.0)

    def test_combined_axes_rms_single(self):
        sig = np.ones(100) * 4.0
        rms = combined_axes_rms([sig])
        assert rms == pytest.approx(4.0)

    def test_combined_axes_rms_three_equal(self):
        sig = np.ones(100) * 2.0
        rms = combined_axes_rms([sig, sig, sig])
        assert rms == pytest.approx(2.0 * (3 ** 0.5))

    def test_combined_axes_rms_empty(self):
        import math
        assert math.isnan(combined_axes_rms([]))
