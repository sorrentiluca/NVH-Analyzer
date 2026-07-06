"""
Tests for the dynamic kurtogram floor formula introduced to fix bandpass
analysis at RPM below 2300.

These tests are pure arithmetic — no stage import, no numpy heavy lifting.
The formula being tested is the one embedded in EP_Envelope.py and EP_bandpass.py:

    eff_min_hz = max(bp_min_hz_abs, 25 * rpm / 60 * 1.2)

where:
  25        = max shaft order (matches comment in both stage files)
  1.2       = 20% safety margin above the top shaft harmonic
  bp_min_hz_abs = absolute floor so near-zero RPM can't send the floor to 0
"""

from __future__ import annotations

import pytest


def _effective_min_hz(rpm: float, abs_floor: float = 50.0) -> float:
    """Mirror of the formula used in EP_Envelope.py and EP_bandpass.py."""
    return max(abs_floor, 25 * rpm / 60 * 1.2)


class TestDynamicMinHz:
    def test_high_rpm_uses_formula(self):
        # At 2300 RPM: 25 * 2300/60 * 1.2 = 1150 Hz > 50 Hz floor
        result = _effective_min_hz(2300)
        assert result == pytest.approx(25 * 2300 / 60 * 1.2)
        assert result > 1000  # well above the old hardcoded floor

    def test_low_rpm_hits_abs_floor(self):
        # At 100 RPM: 25 * 100/60 * 1.2 ≈ 50 Hz — hits the 50 Hz floor
        result = _effective_min_hz(100, abs_floor=50.0)
        formula = 25 * 100 / 60 * 1.2
        assert result == pytest.approx(max(50.0, formula))
        # With default floor of 50, the result should be at least 50
        assert result >= 50.0

    def test_mid_rpm_scales_correctly(self):
        # At 800 RPM: 25 * 800/60 * 1.2 = 400 Hz
        result = _effective_min_hz(800)
        assert result == pytest.approx(25 * 800 / 60 * 1.2)
        assert 350 < result < 450  # sanity range

    def test_abs_floor_dominates_very_low_rpm(self):
        # At 1 RPM the formula gives ~0.5 Hz; floor (50) should dominate
        result = _effective_min_hz(1, abs_floor=50.0)
        assert result == pytest.approx(50.0)

    def test_custom_abs_floor(self):
        result = _effective_min_hz(100, abs_floor=200.0)
        assert result == pytest.approx(200.0)

    def test_monotonically_increasing_with_rpm(self):
        rpms = [100, 200, 400, 800, 1600, 2300]
        floors = [_effective_min_hz(r) for r in rpms]
        assert floors == sorted(floors)

    def test_zero_rpm_uses_abs_floor(self):
        result = _effective_min_hz(0, abs_floor=50.0)
        assert result == pytest.approx(50.0)


class TestDynamicMinHzIntegration:
    """Verify the formula works against the actual PipelineConfig field."""

    def test_bp_min_hz_abs_exists_in_config(self):
        from nvh_pipeline.config import PipelineConfig
        cfg = PipelineConfig()
        assert hasattr(cfg, "bp_min_hz_abs")
        assert cfg.bp_min_hz_abs == pytest.approx(50.0)

    def test_bp_min_hz_abs_in_envelope_local_config(self):
        import importlib.util, sys, types
        # Load EP_Envelope without running any heavy imports we might not have.
        # We only need to inspect the dataclass, not call any functions.
        import EP_Envelope
        cfg = EP_Envelope.Config()
        assert hasattr(cfg, "bp_min_hz_abs")
        assert cfg.bp_min_hz_abs == pytest.approx(50.0)

    def test_bp_min_hz_abs_in_bandpass_local_config(self):
        import EP_bandpass
        cfg = EP_bandpass.Config()
        assert hasattr(cfg, "bp_min_hz_abs")
        assert cfg.bp_min_hz_abs == pytest.approx(50.0)

    def test_detect_band_kurtogram_accepts_min_hz(self):
        """Smoke-test: function accepts the min_hz keyword without crashing."""
        import numpy as np
        import EP_Envelope
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(4096)
        fs = 10_000.0
        cfg = EP_Envelope.Config()
        # Should not raise; we only check the return shape, not the band choice.
        result = EP_Envelope.detect_band_kurtogram(sig, fs, cfg, min_hz=400.0)
        assert len(result) == 4
        f_low, f_ctr, f_high, _ = result
        assert f_low >= 400.0
        assert f_low < f_ctr < f_high
