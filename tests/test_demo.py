"""Tests for nvh_pipeline/demo.py — the starter/teaching workspace.

The full-pipeline outcomes (defect order recovery, calibration amplitude,
loudness ratio) are covered by the golden e2e suite on equivalent signals;
here we verify the generated campaign is structurally sound and that its
deliberate teaching traps are present.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from nvh_pipeline.common import discover_dataset, inspect_headers
from nvh_pipeline.demo import (BPF_ORDER, CAL_AMP, CAL_ORDER,
                               create_demo_workspace)


def test_workspace_layout_and_discovery(tmp_path):
    ws = create_demo_workspace(str(tmp_path / "ws"))
    assert os.path.isdir(ws["data_dir"])
    assert os.path.isfile(ws["guide"])
    assert os.path.isfile(ws["continuous_file"])

    d = discover_dataset(ws["data_dir"])
    assert set(d["parts"]) == {"m101", "m202", "m303", "T404"}
    assert set(d["rpms"]) == {800, 2300}
    # The deliberate coverage gap: T404 exists at 2300 only.
    combos = {(c["part"], c["rpm"]) for c in d["counts"]}
    assert ("T404", 2300) in combos and ("T404", 800) not in combos
    # Every file follows the naming grammar.
    assert d["unmatched"] == []


def test_headers_consistent(tmp_path):
    ws = create_demo_workspace(str(tmp_path / "ws"))
    h = inspect_headers(ws["data_dir"])
    assert h["n_files"] == len([f for f in ws["files"]
                                if ws["data_dir"] in f])
    assert h["n_mismatched"] == 0


def test_defect_and_calibration_content(tmp_path):
    """The injected signals carry what the guide promises, checked directly
    on the raw data (no pipeline)."""
    ws = create_demo_workspace(str(tmp_path / "ws"))

    def _load(name):
        return pd.read_csv(os.path.join(ws["data_dir"], name))

    # m303: tone amplitude on X while moving ≈ CAL_AMP / sqrt(2) RMS.
    df = _load("m303-2300rpm-1.csv")
    moving = df["CNT 1/Frequency (RPM)"].abs() > 100
    x = df.loc[moving, "X Accel (m/s2)"].to_numpy()
    rms = float(np.sqrt(np.mean((x - x.mean()) ** 2)))
    assert abs(rms - CAL_AMP / np.sqrt(2)) / (CAL_AMP / np.sqrt(2)) < 0.10

    # m202: impacts present (kurtosis well above the Gaussian 3.0).
    df = _load("m202-2300rpm-1.csv")
    moving = df["CNT 1/Frequency (RPM)"].abs() > 100
    x = df.loc[moving, "X Accel (m/s2)"].to_numpy()
    dev = x - x.mean()
    kurt = float(np.mean(dev**4) / np.mean(dev**2) ** 2)
    assert kurt > 6.0, f"defect file kurtosis {kurt:.1f} not impulsive"

    # T404 clearly louder than the healthy metal baseline.
    t = _load("T404-2300rpm-1.csv")
    m = _load("m101-2300rpm-1.csv")

    def _rms(frame):
        mov = frame["CNT 1/Frequency (RPM)"].abs() > 100
        v = frame.loc[mov, "X Accel (m/s2)"].to_numpy()
        return float(np.sqrt(np.mean((v - v.mean()) ** 2)))

    assert _rms(t) > 2.0 * _rms(m)


def test_continuous_file_shape(tmp_path):
    ws = create_demo_workspace(str(tmp_path / "ws"))
    df = pd.read_csv(ws["continuous_file"])
    assert list(df.columns) == ["Time (s)", "Speed (RPM)", "Vibration (g)"]
    assert (df["Speed (RPM)"] > 1000).all()


def test_guide_mentions_every_specimen(tmp_path):
    ws = create_demo_workspace(str(tmp_path / "ws"))
    guide = open(ws["guide"], encoding="utf-8").read()
    for part in ("m101", "m202", "m303", "T404"):
        assert part in guide
    assert str(BPF_ORDER) in guide and str(CAL_ORDER)[0] in guide
