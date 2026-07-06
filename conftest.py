"""
Shared pytest fixtures for the NVH pipeline test suite.

Run from the repo root:  pytest tests/ -v
"""

from __future__ import annotations

import json
import os
import sys

import pytest

# Make the repo root importable so tests can find nvh_pipeline/ and EP_*.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
#  CSV data folder fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def csv_dir_consistent(tmp_path):
    """Three CSVs all with the same three columns — no mismatch."""
    cols = "Time (s),CNT 1/Frequency (RPM),X Accel (m/s2)"
    for name in ("m627-100rpm-1.csv", "m627-100rpm-2.csv", "T8-2300rpm-1.csv"):
        (tmp_path / name).write_text(f"{cols}\n0.0,100,0.01\n", encoding="utf-8")
    return str(tmp_path)


@pytest.fixture()
def csv_dir_mismatch(tmp_path):
    """Two CSVs with three columns and one CSV missing the third column."""
    full = "Time (s),CNT 1/Frequency (RPM),X Accel (m/s2)"
    short = "Time (s),CNT 1/Frequency (RPM)"
    (tmp_path / "m627-100rpm-1.csv").write_text(f"{full}\n0.0,100,0.01\n",
                                                  encoding="utf-8")
    (tmp_path / "m627-100rpm-2.csv").write_text(f"{full}\n0.1,100,0.02\n",
                                                  encoding="utf-8")
    (tmp_path / "T8-2300rpm-1.csv").write_text(f"{short}\n0.0,2300\n",
                                                encoding="utf-8")
    return str(tmp_path)


@pytest.fixture()
def csv_dir_grammar(tmp_path):
    """Files that follow the <part>-<rpm>rpm-<trial>.csv naming grammar."""
    cols = "Time (s),RPM"
    for name in ("m627-100rpm-1.csv", "m627-2300rpm-1.csv", "T8-100rpm-1.csv"):
        (tmp_path / name).write_text(f"{cols}\n0.0,100\n", encoding="utf-8")
    (tmp_path / "README.txt").write_text("not a csv", encoding="utf-8")
    return str(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Past-run fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def results_root_with_run(tmp_path):
    """A results_root containing one completed run (manifest + config)."""
    run_dir = tmp_path / "2024-01-01_1200_m627_100rpm"
    run_dir.mkdir()
    manifest = {
        "generated": "2024-01-01 12:00:00",
        "data_dir": "/data",
        "output_root": str(run_dir),
        "stages": [
            {"stage": "segment", "label": "Segmentation", "status": "ok",
             "files": [], "error": None},
            {"stage": "loudness", "label": "Vibration level", "status": "ok",
             "files": [], "error": None},
        ],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    config = {"parts": ["m627"], "rpms": [100], "only_parts": None, "only_rpms": None}
    (run_dir / "nvh_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    return str(tmp_path)
