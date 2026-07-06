"""
Validity-fix coverage: steady-state (plateau) gating, the per-specimen envelope
band cap, and the order-spectrum aggregation modes (power-averaging + RSS
cumulative).  These are the enterprise-grade correctness changes layered on top
of the older calibration suite.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

for _m in ('matplotlib', 'matplotlib.pyplot', 'matplotlib.ticker',
           'matplotlib.patches', 'matplotlib.figure', 'matplotlib.colors',
           'matplotlib.gridspec'):
    sys.modules.setdefault(_m, MagicMock())

import EP_order_analysis as oa
import EP_Envelope as ep
from nvh_pipeline.common import plateau_mask, apply_plateau


# ═══════════════════════════════════════════════════════════════════════════════
#  PLATEAU MASK
# ═══════════════════════════════════════════════════════════════════════════════

def _ramp_plateau_ramp(category_rpm=2300, n=300):
    """One stroke: linear ramp-up, constant plateau, linear ramp-down."""
    up = np.linspace(0, category_rpm, n // 4)
    pl = np.full(n // 2, float(category_rpm))
    dn = np.linspace(category_rpm, 0, n - up.size - pl.size)
    return np.concatenate([up, pl, dn])


class TestPlateauMask:
    def test_selects_only_plateau(self):
        rpm = _ramp_plateau_ramp(2300, 400)
        mask = plateau_mask(rpm, 2300, frac=0.90)
        # Every kept sample is at/above 90 % of nominal …
        assert np.all(np.abs(rpm[mask]) >= 0.90 * 2300)
        # … and the plateau (half the stroke) is essentially all kept.
        assert mask.sum() >= 0.45 * len(rpm)
        # The ramp ends (near zero) are excluded.
        assert not mask[0] and not mask[-1]

    def test_negative_direction_handled(self):
        rpm = -_ramp_plateau_ramp(800, 200)      # retract stroke
        mask = plateau_mask(rpm, 800, frac=0.90)
        assert mask.sum() > 0
        assert np.all(np.abs(rpm[mask]) >= 0.90 * 800)

    def test_fallback_when_no_plateau(self):
        # Stroke never reaches 90 % of a too-high nominal → full-stroke fallback.
        rpm = _ramp_plateau_ramp(500, 200)
        mask, used = apply_plateau(rpm, 5000, frac=0.90, min_samples=2)
        assert used is False
        assert mask.all()

    def test_missing_category_uses_peak(self):
        rpm = _ramp_plateau_ramp(1600, 200)
        mask = plateau_mask(rpm, None, frac=0.90)      # ref = peak |rpm|
        assert np.all(np.abs(rpm[mask]) >= 0.90 * rpm.max())


# ═══════════════════════════════════════════════════════════════════════════════
#  ENVELOPE BAND CEILING  +  band_kurtosis
# ═══════════════════════════════════════════════════════════════════════════════

class TestBandCeiling:
    def test_bp_max_hz_caps_search(self):
        """No detected band may exceed the absolute bp_max_hz ceiling."""
        rng = np.random.default_rng(0)
        fs = 50_000.0
        # Strong narrowband energy at 22 kHz (the noise-wall region we must reject).
        n = 8192
        t = np.arange(n) / fs
        sig = (rng.standard_normal(n)
               + 5.0 * np.sin(2 * np.pi * 22_000 * t))
        cfg = ep.Config(bp_min_hz=1000.0, bp_min_hz_abs=50.0, bp_min_bw_hz=200.0,
                        kurtogram_levels=4, bp_max_hz_frac=1.0, bp_max_hz=12_000.0)
        f_low, f_ctr, f_high, _ = ep.detect_band_kurtogram(sig, fs, cfg)
        assert f_high <= 12_000.0 + 1e-6, f"band ran to {f_high} Hz past the cap"

    def test_return_grid_max_matches_selected_band(self):
        """return_grid must expose the full search grid, and its max-kurtosis
        cell must equal the band the function selects (so the kurtogram heatmap's
        outlined cell is the chosen band)."""
        rng = np.random.default_rng(3)
        fs = 50_000.0
        n = 8192
        t = np.arange(n) / fs
        carrier = np.sin(2 * np.pi * 6_000 * t)
        gate = (np.sin(2 * np.pi * 40 * t) > 0.97).astype(float)
        sig = carrier * gate + 0.1 * rng.standard_normal(n)
        cfg = ep.Config(bp_min_hz=1000.0, bp_min_hz_abs=50.0, bp_min_bw_hz=200.0,
                        kurtogram_levels=4, bp_max_hz_frac=1.0, bp_max_hz=12_000.0)
        f_low, f_ctr, f_high, kurt, grid = ep.detect_band_kurtogram(
            sig, fs, cfg, return_grid=True)
        assert grid, "grid should be non-empty"
        cols = {'level', 'f_low_hz', 'f_center_hz', 'f_high_hz',
                'bandwidth_hz', 'kurtosis'}
        assert cols <= set(grid[0].keys())
        top = max(grid, key=lambda r: r['kurtosis'])
        assert abs(top['f_low_hz'] - f_low) < 0.5
        assert abs(top['f_high_hz'] - f_high) < 0.5
        assert abs(top['kurtosis'] - kurt) < 0.5

    def test_band_kurtosis_impulsive_beats_gaussian(self):
        fs = 50_000.0
        n = 8192
        t = np.arange(n) / fs
        carrier = np.sin(2 * np.pi * 5_000 * t)
        # Impulsive amplitude modulation → high kurtosis in the carrier band.
        gate = (np.sin(2 * np.pi * 40 * t) > 0.97).astype(float)
        impulsive = carrier * gate
        gaussian  = np.random.default_rng(1).standard_normal(n)
        k_imp = ep.band_kurtosis(impulsive, fs, 4_000, 6_000)
        k_gauss = ep.band_kurtosis(gaussian, fs, 4_000, 6_000)
        assert k_imp > 3.0 * k_gauss


# ═══════════════════════════════════════════════════════════════════════════════
#  ORDER ACCUMULATOR  —  power vs magnitude, RSS vs sum
# ═══════════════════════════════════════════════════════════════════════════════

class TestAccumulatorModes:
    def test_power_is_rms_above_magnitude_mean(self):
        """Power averaging returns the RMS (energy-mean) spectrum; by Jensen's
        inequality each bin is >= the magnitude (arithmetic-mean) spectrum, and
        both preserve a coherent tone.  (Incoherent averaging does NOT null the
        floor — that is the honest behaviour we assert here.)"""
        rng = np.random.default_rng(2)
        n_orders, n_seg = 50, 200
        tone_bin = 10
        pwr = oa.Accumulator(n_orders, average='power', cumulative_mode='rss')
        mag = oa.Accumulator(n_orders, average='magnitude', cumulative_mode='sum')
        for _ in range(n_seg):
            a = np.abs(rng.standard_normal(n_orders))     # incoherent floor
            a[tone_bin] += 5.0                            # coherent tone
            z = np.zeros(n_orders)
            pwr.add(a, z, z, 1.0)
            mag.add(a, z, z, 1.0)
        mp, mm = pwr.means()['x'], mag.means()['x']
        floor = [i for i in range(n_orders) if i != tone_bin]
        # RMS floor >= arithmetic-mean floor (Jensen), and the tone is preserved.
        assert np.median(mp[floor]) >= np.median(mm[floor])
        assert mp[tone_bin] > 4.0 and mm[tone_bin] > 4.0

    def test_rss_vs_sum_cumulative(self):
        n = 5
        acc_rss = oa.Accumulator(n, average='magnitude', cumulative_mode='rss')
        acc_sum = oa.Accumulator(n, average='magnitude', cumulative_mode='sum')
        x = np.full(n, 3.0); y = np.full(n, 4.0); z = np.zeros(n)
        acc_rss.add(x, y, z, 1.0)
        acc_sum.add(x, y, z, 1.0)
        # RSS of (3,4,0) = 5; linear sum = 7.
        assert np.allclose(acc_rss.means()['cumulative'], 5.0)
        assert np.allclose(acc_sum.means()['cumulative'], 7.0)

    def test_power_mean_matches_magnitude_for_constant_tone(self):
        """For a tone of identical amplitude in every segment, sqrt(mean|A|²) ==
        mean(|A|): the calibration is unchanged, only the floor differs."""
        n = 4
        acc = oa.Accumulator(n, average='power', cumulative_mode='sum')
        for _ in range(10):
            acc.add(np.full(n, 2.0), np.zeros(n), np.zeros(n), 1.0)
        assert np.allclose(acc.means()['x'], 2.0)
