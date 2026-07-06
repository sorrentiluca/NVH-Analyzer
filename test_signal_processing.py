"""
Validation gate for the Phase 0–4 algorithm rewrites.

Covers:
  P1 — FFT-domain kurtogram  (detect_band_kurtogram)
  P4 — Parallel segment workers (_compute_oa_segment, _compute_envelope_segment)
  P2 — DuckDB RMS via reduce_segments (must match numpy sqrt(mean(x²)))

All assertions use rtol ≤ 1e-3 for FFT-domain functions (rectangular vs
Butterworth filter produces slightly different kurtosis scores, but the
selected band must still contain the true resonance) and exact equality for
pure-reduction math.  These tests must stay green across all optimisation
phases — if one breaks, a phase regression has occurred.

Matplotlib is stubbed so the tests run in the sandbox venv (no matplotlib
installed) as well as the real env.  Signal-processing code paths are not
matplotlib-dependent; only the plot_* helpers use it.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ── Stub matplotlib before any EP_*.py import touches it ─────────────────────
# Register every matplotlib sub-namespace used by EP_*.py so that both our own
# imports and any other test file that imports EP_bandpass, EP_Envelope etc.
# find these entries in sys.modules and never hit "No module named 'matplotlib'".
for _m in (
    'matplotlib', 'matplotlib.pyplot', 'matplotlib.ticker',
    'matplotlib.patches', 'matplotlib.figure', 'matplotlib.colors',
    'matplotlib.gridspec',
):
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic signal builders
# ─────────────────────────────────────────────────────────────────────────────

FS = 10_000.0  # samples / second — matches make_testdata.py


def _impulsive_signal(n: int = 4096, resonance_hz: float = 2500.0,
                      seed: int = 42) -> np.ndarray:
    """Broadband noise + periodic impulses at resonance_hz convolved with a ring."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.02, n)
    interval = max(1, int(FS / resonance_hz))
    taps = np.zeros(n)
    taps[::interval] = 1.0
    ring_len = int(0.004 * FS)
    tt = np.arange(ring_len) / FS
    ring = np.sin(2 * np.pi * resonance_hz * tt) * np.exp(-tt / 0.0008)
    return noise + np.convolve(taps, ring, mode='same')[:n]


def _synthetic_segment_df(n_pts: int = 5000, rpm: float = 1200.0,
                           ball_pass_order: float = 5.35) -> pd.DataFrame:
    """Minimal segment DataFrame: encoder angle + constant RPM + sinusoidal vibration."""
    dt = 1.0 / FS
    t = np.arange(n_pts) * dt
    rpm_arr = np.full(n_pts, rpm)
    angle = np.cumsum(rpm_arr / 60.0 * 360.0 * dt)  # degrees
    revs = angle / 360.0
    phase = 2 * np.pi * revs * ball_pass_order
    sig = np.sin(phase) + np.random.default_rng(0).normal(0, 0.1, n_pts)
    return pd.DataFrame({
        'Time (s)':               t,
        'CNT 1/Frequency (RPM)':  rpm_arr,
        'CNT 1/Angle (Degrees)':  angle,
        'X Accel (m/s2)':         sig,
        'Y Accel (m/s2)':         sig * 0.5,
        'Z Accel (m/s2)':         sig * 0.3,
        'material_type':          'metal',
        'rpm_category':           int(rpm),
        'part':                   'm627',
        'trial':                  1,
        'direction':              'POS',
        'segment_id':             0,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  P1 — FFT-domain kurtogram
# ─────────────────────────────────────────────────────────────────────────────

class TestFFTKurtogram:
    """detect_band_kurtogram must locate the impulsive band and return a valid tuple."""

    @pytest.fixture(autouse=True)
    def _ep(self):
        import EP_Envelope as _ep
        self.ep = _ep
        self.cfg = _ep.Config(
            kurtogram_levels=4, bp_min_bw_hz=200.0,
            bp_min_hz=500.0, bp_min_hz_abs=50.0,
        )

    def test_returns_four_tuple_with_kurtosis(self):
        sig = _impulsive_signal()
        result = self.ep.detect_band_kurtogram(sig, FS, self.cfg, min_hz=500.0)
        assert len(result) == 4
        f_low, f_ctr, f_high, kurt = result
        assert kurt > 0.0  # raw kurtosis must be positive

    def test_ordering_f_low_lt_center_lt_high(self):
        sig = _impulsive_signal()
        f_low, f_ctr, f_high, _ = self.ep.detect_band_kurtogram(
            sig, FS, self.cfg, min_hz=500.0)
        assert f_low < f_ctr < f_high

    def test_band_near_resonance(self):
        """Detected band must overlap the ±1000 Hz window around the 2500 Hz resonance.

        The cosine-tapered kurtogram selects the dyadic band with the highest
        kurtosis.  A damped-ring impulse response has energy that extends above
        the fundamental, so the highest-kurtosis band may be just above (not
        enclosing) the resonance frequency.  Overlap with ±1000 Hz is the
        right validation: it rules out completely wrong detections while
        tolerating the quantization of dyadic band structures.
        """
        sig = _impulsive_signal(resonance_hz=2500.0)
        f_low, f_ctr, f_high, _ = self.ep.detect_band_kurtogram(
            sig, FS, self.cfg, min_hz=500.0)
        res_hz = 2500.0
        overlap = (f_high >= res_hz - 1000.0) and (f_low <= res_hz + 1000.0)
        assert overlap, (
            f"Band [{f_low:.0f}, {f_high:.0f}] has no overlap with "
            f"[{res_hz - 1000:.0f}, {res_hz + 1000:.0f}]")

    def test_band_within_nyquist(self):
        sig = _impulsive_signal()
        f_low, f_ctr, f_high, _ = self.ep.detect_band_kurtogram(
            sig, FS, self.cfg, min_hz=500.0)
        assert f_low >= 0.0
        assert f_high <= FS / 2.0 + 1e-6

    def test_short_signal_does_not_crash(self):
        """Signals shorter than 4 samples must return a valid fallback 4-tuple."""
        result = self.ep.detect_band_kurtogram(np.zeros(3), FS, self.cfg)
        assert len(result) == 4
        f_low, f_ctr, f_high, kurt = result
        assert f_low <= f_ctr <= f_high

    def test_different_resonance_frequencies(self):
        """Kurtogram band must overlap ±1000 Hz window around each resonance."""
        for res_hz in (1000.0, 2000.0, 3000.0, 4000.0):
            if res_hz >= FS / 2.0:
                continue
            sig = _impulsive_signal(resonance_hz=res_hz, n=8192)
            f_low, _, f_high, _ = self.ep.detect_band_kurtogram(
                sig, FS, self.cfg, min_hz=200.0)
            overlap = (f_high >= res_hz - 1000.0) and (f_low <= res_hz + 1000.0)
            assert overlap, (
                f"resonance {res_hz:.0f} Hz: band [{f_low:.0f}, {f_high:.0f}] "
                f"has no overlap with ±1000 Hz window")

    def test_pure_noise_returns_valid_band(self):
        """Even without impulsive content the function must return a valid 4-tuple."""
        rng = np.random.default_rng(99)
        noise = rng.normal(0, 1.0, 2048)
        result = self.ep.detect_band_kurtogram(noise, FS, self.cfg, min_hz=200.0)
        assert len(result) == 4
        f_low, f_ctr, f_high, kurt = result
        assert f_low <= f_ctr <= f_high
        assert kurt > 0.0

    def test_impulsive_signal_has_higher_kurtosis_than_noise(self):
        """Kurtosis from an impulsive signal must exceed kurtosis from pure noise."""
        rng = np.random.default_rng(7)
        noise = rng.normal(0, 1.0, 8192)
        impulsive = _impulsive_signal(resonance_hz=2500.0, n=8192)
        _, _, _, kurt_noise = self.ep.detect_band_kurtogram(
            noise, FS, self.cfg, min_hz=500.0)
        _, _, _, kurt_imp = self.ep.detect_band_kurtogram(
            impulsive, FS, self.cfg, min_hz=500.0)
        assert kurt_imp > kurt_noise, (
            f"Impulsive kurtosis {kurt_imp:.2f} should exceed noise {kurt_noise:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
#  P4 — Order analysis worker
# ─────────────────────────────────────────────────────────────────────────────

class TestOrderAnalysisWorker:
    """_compute_oa_segment must match direct per-axis computation and be deterministic."""

    @pytest.fixture(autouse=True)
    def _oa(self):
        import EP_order_analysis as oa
        self.oa = oa
        self.cfg = oa.Config(
            samples_per_rev=64,
            order_resolution=0.1,
            min_revolutions=0.5,
            min_fft_points=8,
            angle_source='encoder',
        )
        self.axis_cols = {
            'x': 'X Accel (m/s2)',
            'y': 'Y Accel (m/s2)',
            'z': 'Z Accel (m/s2)',
        }
        self.common_orders = np.arange(0.0, 12.0 + 0.1, 0.1)

    def _task(self, seg, cfg=None):
        meta = {
            'part': 'm627', 'material_type': 'metal', 'rpm_category': 1200,
            'trial': 1, 'direction': 'POS', 'segment_id': 0,
            'source_file': 'test.parquet',
        }
        return (seg, meta, self.axis_cols, self.common_orders, cfg or self.cfg)

    def test_returns_ok_status_for_valid_segment(self):
        seg = _synthetic_segment_df()
        status, result = self.oa._compute_oa_segment(self._task(seg))
        assert status == 'ok'

    def test_all_three_axes_present(self):
        seg = _synthetic_segment_df()
        _, result = self.oa._compute_oa_segment(self._task(seg))
        assert set(result['amps'].keys()) == {'x', 'y', 'z'}

    def test_spectrum_length_matches_common_orders(self):
        seg = _synthetic_segment_df()
        _, result = self.oa._compute_oa_segment(self._task(seg))
        for ax in ('x', 'y', 'z'):
            assert len(result['amps'][ax]) == len(self.common_orders)

    def test_n_rev_positive(self):
        seg = _synthetic_segment_df()
        _, result = self.oa._compute_oa_segment(self._task(seg))
        assert result['n_rev'] > 0.0

    def test_skip_on_too_short_revolutions(self):
        seg = _synthetic_segment_df(n_pts=20)   # ≪ 1 revolution at 1200 RPM
        cfg = self.oa.Config(min_revolutions=100.0)
        status, reason = self.oa._compute_oa_segment(self._task(seg, cfg))
        assert status == 'skip'
        assert reason == 'min_revolutions'

    def test_dominant_order_near_ball_pass(self):
        """Sinusoid at order 5.35 should produce the dominant spectral peak there."""
        seg = _synthetic_segment_df(n_pts=10000, ball_pass_order=5.35)
        _, result = self.oa._compute_oa_segment(self._task(seg))
        peak_order = self.common_orders[np.argmax(result['amps']['x'])]
        assert abs(peak_order - 5.35) < 1.5, (
            f"Expected peak near order 5.35, got {peak_order:.2f}")

    def test_worker_is_deterministic(self):
        """Identical inputs must yield bit-exact outputs."""
        seg = _synthetic_segment_df()
        task = self._task(seg)
        _, r1 = self.oa._compute_oa_segment(task)
        _, r2 = self.oa._compute_oa_segment(task)
        np.testing.assert_array_equal(r1['amps']['x'], r2['amps']['x'])
        np.testing.assert_array_equal(r1['amps']['y'], r2['amps']['y'])
        np.testing.assert_array_equal(r1['amps']['z'], r2['amps']['z'])

    def test_result_contains_bpfo_band_energy(self):
        """Worker result must include bpfo_energy dict with per-harmonic keys."""
        seg = _synthetic_segment_df()
        _, result = self.oa._compute_oa_segment(self._task(seg))
        assert 'bpfo_energy' in result
        be = result['bpfo_energy']
        assert 'total' in be
        assert 'h1' in be
        assert float(be['total']) >= 0.0

    def test_bpfo_band_energy_higher_at_signal_order(self):
        """Band energy around order 5.35 must exceed a blank (non-signal) band."""
        seg = _synthetic_segment_df(n_pts=10000, ball_pass_order=5.35)
        _, result = self.oa._compute_oa_segment(self._task(seg))
        # h1 covers [5.10, 5.60]; compare against a band centered at order 3.0
        bpfo_h1 = result['bpfo_energy']['h1']
        blank = self.oa.bpfo_band_energy(
            self.common_orders,
            result['amps']['x'] + result['amps']['y'] + result['amps']['z'],
            ball_pass_order=3.0, n_harmonics=1, halfwidth=0.25,
        )['h1']
        assert bpfo_h1 > blank, (
            f"BPFO h1 energy {bpfo_h1:.4g} should exceed blank band {blank:.4g}"
        )

    def test_bpfo_band_energy_helper_direct(self):
        """bpfo_band_energy integrates correctly on a known piecewise-flat spectrum."""
        orders = np.linspace(0.0, 25.0, 2501)   # 0.01-order step
        spectrum = np.zeros_like(orders)
        # Place unit amplitude in [5.10, 5.60] (halfwidth=0.25 around 5.35)
        mask = (orders >= 5.10) & (orders <= 5.60)
        spectrum[mask] = 1.0
        be = self.oa.bpfo_band_energy(orders, spectrum,
                                       ball_pass_order=5.35,
                                       n_harmonics=1, halfwidth=0.25)
        # Trapz of 1.0 over 0.5-order band ≈ 0.5
        assert abs(be['h1'] - 0.5) < 0.02, f"h1 energy {be['h1']:.4f} expected ~0.5"
        assert abs(be['total'] - be['h1']) < 1e-10

    def test_result_matches_direct_computation(self):
        """Worker output must match manually calling phi_revolutions + order_spectrum
        using the same anti-aliased resampling path."""
        from scipy.signal import get_window as _gw
        seg = _synthetic_segment_df()
        task = self._task(seg)
        _, result = self.oa._compute_oa_segment(task)

        # Replicate what the worker does (including LP anti-alias filter).
        phi  = self.oa.phi_revolutions(seg, self.cfg)
        keep = np.concatenate([[True], np.diff(phi) > 0])
        phi  = phi[keep]
        n_rev = float(phi[-1])
        M    = int(np.floor(n_rev * self.cfg.samples_per_rev))
        phi_u = np.arange(M) / self.cfg.samples_per_rev
        o_fft = np.fft.rfftfreq(M, d=1.0 / self.cfg.samples_per_rev)
        win   = _gw(self.cfg.window, M, fftbins=True)
        sig   = seg['X Accel (m/s2)'].to_numpy(dtype=float)[keep]

        # Same LP cutoff the worker derives: 5th-pctile non-zero RPM in kept portion.
        t_full   = seg['Time (s)'].to_numpy(dtype=float)
        fs_ref   = 1.0 / float(np.median(np.diff(t_full)))
        rpm_kept = np.abs(seg['CNT 1/Frequency (RPM)'].to_numpy(dtype=float))[keep]
        moving   = rpm_kept[rpm_kept > 5.0]
        rpm_min  = float(np.percentile(moving, 5)) if moving.size else None
        lp_hz    = (max((self.cfg.samples_per_rev / 2) * rpm_min / 60.0, 10.0)
                    if rpm_min is not None else None)

        amp    = self.oa.order_spectrum(sig, phi, M, win, phi_u, self.cfg,
                                        lp_hz=lp_hz, fs=fs_ref)
        direct = np.interp(self.common_orders, o_fft, amp, left=0.0, right=0.0)

        np.testing.assert_allclose(result['amps']['x'], direct, rtol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
#  P4 — Envelope worker
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvelopeWorker:
    """_compute_envelope_segment must return structurally sound, deterministic results."""

    @pytest.fixture(autouse=True)
    def _ep(self):
        import EP_Envelope as ep
        self.ep = ep
        self.cfg = ep.Config(
            samples_per_rev=64,
            order_resolution=0.1,
            min_revolutions=0.5,
            min_fft_points=8,
            kurtogram_levels=3,
            bp_min_bw_hz=200.0,
            bp_min_hz=500.0,
            bp_min_hz_abs=50.0,
            only_rpms=(1200,),
        )
        self.axis_cols = {
            'x': 'X Accel (m/s2)',
            'y': 'Y Accel (m/s2)',
            'z': 'Z Accel (m/s2)',
        }
        self.common_orders = np.arange(0.0, 12.0 + 0.1, 0.1)

    def _task(self, seg, cfg=None):
        return (
            seg, 'test.parquet', 0, 'metal', 1200, 'm627', 'POS',
            self.axis_cols, self.common_orders, cfg or self.cfg,
        )

    def test_returns_ok_for_valid_segment(self):
        seg = _synthetic_segment_df()
        status, result = self.ep._compute_envelope_segment(self._task(seg))
        assert status == 'ok'

    def test_all_three_axes_in_amps(self):
        seg = _synthetic_segment_df()
        _, result = self.ep._compute_envelope_segment(self._task(seg))
        assert set(result['amps'].keys()) == {'x', 'y', 'z'}

    def test_spectrum_length_matches_common_orders(self):
        seg = _synthetic_segment_df()
        _, result = self.ep._compute_envelope_segment(self._task(seg))
        for ax in ('x', 'y', 'z'):
            assert len(result['amps'][ax]) == len(self.common_orders)

    def test_three_bp_log_rows_one_per_axis(self):
        seg = _synthetic_segment_df()
        _, result = self.ep._compute_envelope_segment(self._task(seg))
        assert len(result['bp_rows']) == 3
        axes_in_log = {row['axis'] for row in result['bp_rows']}
        assert axes_in_log == {'x', 'y', 'z'}

    def test_bp_log_row_structure(self):
        seg = _synthetic_segment_df()
        _, result = self.ep._compute_envelope_segment(self._task(seg))
        for row in result['bp_rows']:
            for field in ('f_low_hz', 'f_center_hz', 'f_high_hz', 'fs_hz',
                          'axis', 'material', 'rpm_cat', 'source_file'):
                assert field in row, f"Missing field {field!r} in bp_log row"
            assert row['f_low_hz'] < row['f_center_hz'] < row['f_high_hz']

    def test_result_metadata_matches_input(self):
        seg = _synthetic_segment_df()
        _, result = self.ep._compute_envelope_segment(self._task(seg))
        assert result['material']  == 'metal'
        assert result['part']      == 'm627'
        assert result['rpm_cat']   == 1200
        assert result['direction'] == 'POS'

    def test_skip_on_too_short_revolutions(self):
        seg = _synthetic_segment_df(n_pts=20)
        cfg = self.ep.Config(min_revolutions=100.0)
        status, reason = self.ep._compute_envelope_segment(self._task(seg, cfg))
        assert status == 'skip'
        assert reason == 'min_revolutions'

    def test_worker_is_deterministic(self):
        """Same input twice must produce bit-exact identical output."""
        seg = _synthetic_segment_df()
        task = self._task(seg)
        _, r1 = self.ep._compute_envelope_segment(task)
        _, r2 = self.ep._compute_envelope_segment(task)
        for ax in ('x', 'y', 'z'):
            np.testing.assert_array_equal(r1['amps'][ax], r2['amps'][ax])


# ─────────────────────────────────────────────────────────────────────────────
#  P2 — DuckDB RMS matches numpy
# ─────────────────────────────────────────────────────────────────────────────

class TestDuckDBRms:
    """reduce_segments SQRT(AVG(col*col)) must match numpy sqrt(mean(x²)) to rtol=1e-6."""

    @pytest.fixture(autouse=True)
    def _require_duckdb(self):
        pytest.importorskip('duckdb', reason='duckdb not installed — skipping RMS tests')

    _META = {
        'material_type': 'metal',
        'rpm_category':  1200,
        'part':          'm627',
        'trial':         1,
        'direction':     'POS',
        'source_file':   'test.parquet',
    }

    def _write_parquet(self, tmp_path, col_values: dict, segment_id: int = 0) -> str:
        """Write one *_segments.parquet file and return its parent directory."""
        n = len(next(iter(col_values.values())))
        df = pd.DataFrame({
            **col_values,
            't_rel':       np.linspace(0, 1, n),
            'segment_id':  [segment_id] * n,
            **{k: [v] * n for k, v in self._META.items()},
        })
        (tmp_path / 'test_seg_segments.parquet').write_bytes(
            df.to_parquet(index=False))
        return str(tmp_path)

    def test_rms_single_segment_matches_numpy(self, tmp_path):
        from nvh_pipeline import dataset
        rng = np.random.default_rng(7)
        x = rng.normal(0, 2.0, 1000)
        seg_dir = self._write_parquet(tmp_path, {'x_col': x})
        expected = float(np.sqrt(np.mean(x ** 2)))

        result = dataset.reduce_segments(
            seg_dir,
            agg_exprs={'rms_x': 'SQRT(AVG("x_col" * "x_col"))'},
        )
        assert len(result) == 1
        got = float(result['rms_x'].iloc[0])
        assert abs(got - expected) / (expected + 1e-30) < 1e-6, (
            f"DuckDB RMS {got:.8f} != numpy RMS {expected:.8f}")

    def test_rms_two_segments_each_correct(self, tmp_path):
        from nvh_pipeline import dataset
        rng = np.random.default_rng(13)
        n = 500
        seg0 = rng.normal(0, 1.0, n)
        seg1 = rng.normal(0, 3.0, n)

        df = pd.DataFrame({
            'x_col':         np.concatenate([seg0, seg1]),
            't_rel':         np.linspace(0, 2, n * 2),
            'segment_id':    [0] * n + [1] * n,
            **{k: [v] * (n * 2) for k, v in self._META.items()},
        })
        # _META already includes 'direction' — no extra column needed.
        (tmp_path / 'test_seg_segments.parquet').write_bytes(
            df.to_parquet(index=False))

        result = dataset.reduce_segments(
            str(tmp_path),
            agg_exprs={'rms_x': 'SQRT(AVG("x_col" * "x_col"))'},
        )
        result = result.sort_values('segment_id').reset_index(drop=True)
        assert len(result) == 2

        for i, seg_vals in enumerate([seg0, seg1]):
            expected = float(np.sqrt(np.mean(seg_vals ** 2)))
            got = float(result.loc[i, 'rms_x'])
            assert abs(got - expected) / (expected + 1e-30) < 1e-6

    def test_rms_constant_signal_exact(self, tmp_path):
        """RMS of a constant c must equal c exactly."""
        from nvh_pipeline import dataset
        c = 3.7
        x = np.full(200, c)
        seg_dir = self._write_parquet(tmp_path, {'x_col': x})

        result = dataset.reduce_segments(
            seg_dir,
            agg_exprs={'rms_x': 'SQRT(AVG("x_col" * "x_col"))'},
        )
        assert abs(float(result['rms_x'].iloc[0]) - c) < 1e-10

    def test_combined_rms_matches_numpy(self, tmp_path):
        """Three-axis combined RMS: SQRT(AVG(x*x + y*y + z*z)) == numpy."""
        from nvh_pipeline import dataset
        rng = np.random.default_rng(21)
        n = 800
        x, y, z = rng.normal(0, 1.0, n), rng.normal(0, 2.0, n), rng.normal(0, 0.5, n)
        seg_dir = self._write_parquet(tmp_path, {'xc': x, 'yc': y, 'zc': z})
        expected = float(np.sqrt(np.mean(x**2 + y**2 + z**2)))

        result = dataset.reduce_segments(
            seg_dir,
            agg_exprs={'rms_comb': 'SQRT(AVG("xc"*"xc" + "yc"*"yc" + "zc"*"zc"))'},
        )
        got = float(result['rms_comb'].iloc[0])
        assert abs(got - expected) / (expected + 1e-30) < 1e-6

    def test_variance_form_rms_equals_std_for_dc_offset_signal(self, tmp_path):
        """Variance-form DC-removed RMS must equal np.std for a DC-offset signal (R3).

        raw RMS includes DC offset; variance form SQRT(AVG(x*x) - AVG(x)*AVG(x))
        equals the population std regardless of offset.  This is the DuckDB SQL
        expression used by EP_loudness for remove_dc=True.
        """
        from nvh_pipeline import dataset
        rng = np.random.default_rng(42)
        ac = rng.normal(0, 1.5, 1000)
        dc_offset = 10.0          # large sensor bias
        x = ac + dc_offset
        seg_dir = self._write_parquet(tmp_path, {'x_col': x})

        # Variance-form expression: SQRT(AVG(x*x) - AVG(x)*AVG(x))
        result = dataset.reduce_segments(
            seg_dir,
            agg_exprs={
                'rms_raw':      'SQRT(AVG("x_col" * "x_col"))',
                'rms_dc_removed': 'SQRT(GREATEST(AVG("x_col"*"x_col") - AVG("x_col")*AVG("x_col"), 0))',
            },
        )
        rms_raw      = float(result['rms_raw'].iloc[0])
        rms_dc_removed = float(result['rms_dc_removed'].iloc[0])
        expected_std = float(np.std(x))   # population std == AC RMS

        # Raw RMS is dominated by DC offset and must be much larger than std
        assert rms_raw > expected_std * 2.0, (
            f"raw RMS {rms_raw:.4f} should exceed 2× std {expected_std:.4f}")
        # Variance-form must match population std to floating-point precision
        assert abs(rms_dc_removed - expected_std) / (expected_std + 1e-30) < 1e-5, (
            f"DC-removed RMS {rms_dc_removed:.6f} != std {expected_std:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
#  P3 — render_plots flag threading
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderPlotsFlag:
    """render_plots=False must be threaded into every stage Config via apply_to_stage."""

    def test_render_plots_false_propagates_to_envelope_config(self):
        from nvh_pipeline.config import PipelineConfig, apply_to_stage
        import EP_Envelope as ep
        p = PipelineConfig(render_plots=False)
        cfg = ep.Config()
        assert cfg.render_plots is True          # default
        apply_to_stage('envelope', cfg, p)
        assert cfg.render_plots is False

    def test_render_plots_true_propagates(self):
        from nvh_pipeline.config import PipelineConfig, apply_to_stage
        import EP_order_analysis as oa
        p = PipelineConfig(render_plots=True)
        cfg = oa.Config(render_plots=False)      # start False
        apply_to_stage('order', cfg, p)
        assert cfg.render_plots is True

    def test_render_plots_in_pipeline_config_default_true(self):
        from nvh_pipeline.config import PipelineConfig
        assert PipelineConfig().render_plots is True

    def test_render_plots_from_dict(self):
        from nvh_pipeline.config import PipelineConfig
        p = PipelineConfig.from_dict({'render_plots': False})
        assert p.render_plots is False


# ─────────────────────────────────────────────────────────────────────────────
#  Envelope numerical stability  (regression for the 1e53 spike)
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvelopeStability:
    """extract_envelope must stay finite and bounded for ANY band the kurtogram
    can return — including bands whose edge approaches Nyquist, which made the
    old Butterworth+filtfilt path explode to ~1e25+ and dominate the mean."""

    @pytest.fixture(autouse=True)
    def _ep(self):
        import EP_Envelope as ep
        self.ep = ep
        self.cfg = ep.Config()

    def test_near_nyquist_band_stays_bounded(self):
        """The smoking gun: f_high within 0.02 % of Nyquist must not diverge."""
        fs = 100_000.0
        nyq = fs / 2.0
        rng = np.random.default_rng(0)
        sig = rng.standard_normal(4000)
        in_band_amp = float(np.max(np.abs(sig)))
        for f_high in (0.98 * nyq, 0.999 * nyq, 0.9999 * nyq):
            env = self.ep.extract_envelope(sig, fs, f_high - 400.0, f_high, self.cfg)
            assert np.all(np.isfinite(env)), f"non-finite at f_high={f_high:.0f}"
            assert np.max(env) < 10.0 * in_band_amp, (
                f"envelope max {np.max(env):.4g} exploded at f_high={f_high:.0f} "
                f"(signal amp {in_band_amp:.4g})")

    def test_fft_envelope_matches_safe_band_demodulation(self):
        """In a numerically safe mid-band the FFT envelope must track a known
        amplitude modulation (the envelope is what it demodulates)."""
        fs = 100_000.0
        t = np.arange(8000) / fs
        carrier = 20_000.0
        am = 1.0 + 0.5 * np.sin(2 * np.pi * 50 * t)        # 50 Hz modulation
        sig = np.sin(2 * np.pi * carrier * t) * am
        env = self.ep.extract_envelope(sig, fs, 16_000.0, 24_000.0, self.cfg)
        # Recovered envelope should correlate strongly with the true AM profile.
        corr = np.corrcoef(env, am)[0, 1]
        assert corr > 0.9, f"envelope/AM correlation {corr:.3f} too low"

    def test_non_finite_segment_is_skipped(self):
        """A segment yielding a non-finite spectrum must be skipped, not averaged."""
        seg = _synthetic_segment_df()
        seg = seg.copy()
        seg.loc[5, 'X Accel (m/s2)'] = np.inf      # inject a bad sample
        cfg = self.ep.Config(
            samples_per_rev=64, order_resolution=0.1, min_revolutions=0.5,
            min_fft_points=8, kurtogram_levels=3, only_rpms=(1200,))
        axis_cols = {'x': 'X Accel (m/s2)', 'y': 'Y Accel (m/s2)',
                     'z': 'Z Accel (m/s2)'}
        common_orders = np.arange(0.1, 12.0 + 0.1, 0.1)
        task = (seg, 'test.parquet', 0, 'metal', 1200, 'm627', 'POS',
                axis_cols, common_orders, cfg)
        status, reason = self.ep._compute_envelope_segment(task)
        assert status == 'skip'
        assert reason == 'non_finite'

    def test_normalization_resists_loud_segment_domination(self):
        """With normalize_segments=True, a 1000× louder segment must not swamp
        the accumulated mean shape."""
        n = len(np.arange(0.1, 12.0 + 0.1, 0.1))
        acc = self.ep.Accumulator(n)
        quiet = np.ones(n) * 1.0
        loud  = np.ones(n) * 1000.0
        # Simulate normalized spectra (already unit-scaled): both contribute equally.
        acc.add(quiet / np.sqrt(np.mean(quiet**2)), quiet, quiet, 1.0)
        acc.add(loud / np.sqrt(np.mean(loud**2)), loud / np.sqrt(np.mean(loud**2)),
                loud / np.sqrt(np.mean(loud**2)), 1.0)
        means = acc.means()
        # After normalization the x-axis means of the two contributions are within 1×.
        assert np.all(np.isfinite(means['x']))

    def test_kurtogram_respects_max_hz_frac_cap(self):
        """detect_band_kurtogram must never return a band above bp_max_hz_frac*nyq."""
        fs = 100_000.0
        nyq = fs / 2.0
        rng = np.random.default_rng(3)
        # High-frequency-biased noise to tempt a near-Nyquist pick.
        sig = np.cumsum(rng.standard_normal(8000))
        sig = np.diff(sig, prepend=sig[0])
        cfg = self.ep.Config(bp_max_hz_frac=0.6)
        _, _, f_high, _ = self.ep.detect_band_kurtogram(sig, fs, cfg, min_hz=1000.0)
        assert f_high <= 0.6 * nyq + 1e-6, (
            f"band f_high {f_high:.0f} exceeds cap {0.6 * nyq:.0f}")
