"""
High-fidelity *accuracy* validation for the NVH analysis pipeline.

Unlike ``test_signal_processing.py`` (a regression guard that checks peak
*location* and *relative* energy), this suite asserts the pipeline reproduces
the **closed-form** answer of synthetic signals whose spectral content, RMS and
band are known by construction — proving the numbers are calibrated, not merely
self-consistent.  Tolerances are tied to empirically measured floor behaviour
(see the plan file for the derivations and the measured residuals):

  * order-spectrum amplitude calibration: rtol ≤ 1e-3 on the native FFT grid for
    an on-bin tone (measured residual ~3e-4);
  * RMS of an A·sine: A/√2 to rtol ≤ 1e-6;
  * anti-aliasing suppresses an above-Nyquist tone by ≥ 20×;
  * kurtogram: impulsive kurtosis ≫ Gaussian floor; band overlaps the resonance;
  * envelope demodulation recovers the ball-pass order AND the modulation depth.

All signals come from ``tests/synth.py`` so each test reads like the math it
checks.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Stub matplotlib so importing EP_*.py never needs a display (mirrors the
# existing suite; these tests touch no plotting code).
for _m in ('matplotlib', 'matplotlib.pyplot', 'matplotlib.ticker',
           'matplotlib.patches', 'matplotlib.figure', 'matplotlib.colors',
           'matplotlib.gridspec'):
    sys.modules.setdefault(_m, MagicMock())

from scipy.signal import get_window

import synth  # tests/ is on sys.path via conftest
import EP_order_analysis as oa
import EP_Envelope as ep
from nvh_pipeline.common import per_axis_rms, combined_axes_rms


FS = synth.FS
# Production-resolution order grid (matches Config.order_resolution = 0.02).
ORDERS = np.arange(0.0, 12.0 + 0.02, 0.02)
AXIS_COLS = {'x': synth.AXIS_COLS['x'], 'y': synth.AXIS_COLS['y'],
             'z': synth.AXIS_COLS['z']}


def _oa_cfg(**kw):
    base = dict(samples_per_rev=64, min_revolutions=0.5, min_fft_points=8,
                detrend='mean', window='hann', angle_source='encoder',
                ball_pass_order=5.35, n_bpf_harmonics=4,
                bpfo_band_halfwidth=0.25)
    base.update(kw)
    return oa.Config(**base)


def _ep_cfg(**kw):
    base = dict(samples_per_rev=64, min_revolutions=0.5, min_fft_points=8,
                detrend='mean', window='hann', angle_source='encoder',
                kurtogram_levels=4, bp_min_hz=500.0, bp_min_hz_abs=50.0,
                bp_min_bw_hz=200.0, bp_max_hz_frac=0.8, ball_pass_order=5.35,
                only_rpms=(1200,))
    base.update(kw)
    return ep.Config(**base)


def _oa_meta():
    return {'part': 'm627', 'material_type': 'metal', 'rpm_category': 1200,
            'trial': 1, 'direction': 'POS', 'segment_id': 0,
            'source_file': 'synthetic.parquet'}


def _native_order_spectrum(df, cfg, axis='x', lp=None, fs=None):
    """Run order_spectrum on the *native* angular FFT grid (no 0.1/0.02 interp),
    returning (orders_fft, amplitude).  This is the calibration-accurate path."""
    phi = oa.phi_revolutions(df, cfg)
    keep = np.concatenate([[True], np.diff(phi) > 0])
    phi = phi[keep]
    M = int(np.floor(phi[-1] * cfg.samples_per_rev))
    phi_u = np.arange(M) / cfg.samples_per_rev
    o_fft = np.fft.rfftfreq(M, d=1.0 / cfg.samples_per_rev)
    win = get_window(cfg.window, M, fftbins=True)
    sig = df[synth.AXIS_COLS[axis]].to_numpy(dtype=float)[keep]
    amp = oa.order_spectrum(sig, phi, M, win, phi_u, cfg, lp_hz=lp, fs=fs)
    return o_fft, amp


# ═══════════════════════════════════════════════════════════════════════════════
#  ORDER-SPECTRUM AMPLITUDE CALIBRATION  (the gap the existing suite misses)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrderAmplitudeCalibration:
    """A pure tone of amplitude A at an on-bin order must read back A."""

    @pytest.mark.parametrize('order,amp', [
        (3.0, 1.0), (5.0, 0.7), (5.0, 2.5), (8.0, 0.4), (10.0, 1.3)])
    def test_native_grid_amplitude_equals_A(self, order, amp):
        df, exp = synth.pure_order_tone(order, amp, rpm=1200, n_rev=20)
        cfg = _oa_cfg()
        o_fft, a = _native_order_spectrum(df, cfg, axis='x')
        peak = a.max()
        peak_order = o_fft[np.argmax(a)]
        assert abs(peak_order - order) < 1e-9, f"peak at {peak_order}, want {order}"
        # Residual ~ linear-resample attenuation, grows mildly with order
        # (measured ~3e-4 at order 5, ~1.3e-3 at order 10).  2e-3 is a tight,
        # physically-justified calibration bound (≤ 0.2 %).
        assert abs(peak - amp) / amp < 2e-3, (
            f"recovered amplitude {peak:.6f} != A={amp} (rtol "
            f"{abs(peak-amp)/amp:.2e})")

    def test_per_axis_scaling_preserved(self):
        """Axis amplitudes scale exactly as the injected per-axis gains."""
        df, exp = synth.pure_order_tone(5.0, 1.0, rpm=1200, n_rev=20,
                                        axis_scale=(1.0, 0.5, 0.3))
        cfg = _oa_cfg()
        peaks = {ax: _native_order_spectrum(df, cfg, axis=ax)[1].max()
                 for ax in 'xyz'}
        assert abs(peaks['x'] - 1.0) / 1.0 < 1e-3
        assert abs(peaks['y'] - 0.5) / 0.5 < 1e-3
        assert abs(peaks['z'] - 0.3) / 0.3 < 1e-3

    def test_worker_amplitude_on_production_grid(self):
        """Through the real worker on a 0.02-order grid, amplitude is within 1 %."""
        df, exp = synth.pure_order_tone(5.0, 0.7, rpm=1200, n_rev=20)
        cfg = _oa_cfg()
        status, res = oa._compute_oa_segment((df, _oa_meta(), AXIS_COLS, ORDERS, cfg))
        assert status == 'ok'
        peak = res['amps']['x'].max()
        assert abs(peak - 0.7) / 0.7 < 0.01, f"worker amplitude {peak:.5f} != 0.7"

    def test_amplitude_scales_linearly(self):
        """Doubling A doubles the recovered peak (system is linear)."""
        cfg = _oa_cfg()
        p1 = _native_order_spectrum(synth.pure_order_tone(5.0, 1.0)[0], cfg)[1].max()
        p2 = _native_order_spectrum(synth.pure_order_tone(5.0, 2.0)[0], cfg)[1].max()
        assert abs(p2 / p1 - 2.0) < 1e-3


class TestOrderLocalizationAndResolution:
    """Peaks land at the right orders; nearby tones are separately resolved."""

    @pytest.mark.parametrize('order', [2.0, 3.5, 5.0, 7.0, 9.0])
    def test_peak_at_injected_order(self, order):
        df, _ = synth.pure_order_tone(order, 1.0, rpm=1200, n_rev=20)
        cfg = _oa_cfg()
        status, res = oa._compute_oa_segment((df, _oa_meta(), AXIS_COLS, ORDERS, cfg))
        peak_order = ORDERS[np.argmax(res['amps']['x'])]
        assert abs(peak_order - order) <= 0.02, f"{peak_order} != {order}"

    def test_two_tones_resolved(self):
        df, exp = synth.multi_order([(3.0, 1.0), (8.0, 0.6)], rpm=1200, n_rev=20)
        cfg = _oa_cfg()
        status, res = oa._compute_oa_segment((df, _oa_meta(), AXIS_COLS, ORDERS, cfg))
        peaks = oa.top_peaks(ORDERS, res['amps']['x'], n=2, min_order=0.5)
        found = sorted(o for o, _ in peaks)
        assert len(found) == 2
        assert abs(found[0] - 3.0) <= 0.05 and abs(found[1] - 8.0) <= 0.05
        # relative amplitude is preserved (0.6 vs 1.0)
        amp_at = {round(o): a for o, a in peaks}
        assert abs(amp_at[8] / amp_at[3] - 0.6) < 0.03

    def test_sub_min_order_content_ignored(self):
        """top_peaks must not report a peak below its min_order floor."""
        df, _ = synth.pure_order_tone(0.2, 1.0, rpm=1200, n_rev=20)
        cfg = _oa_cfg()
        _, res = oa._compute_oa_segment((df, _oa_meta(), AXIS_COLS, ORDERS, cfg))
        peaks = oa.top_peaks(ORDERS, res['amps']['x'], n=3, min_order=0.3)
        assert all(o >= 0.3 for o, _ in peaks)


class TestBpfoBandEnergy:
    """Band-integrated BPFO energy is resolution-independent and correct."""

    def test_multi_harmonic_flat_spectrum(self):
        orders = np.linspace(0.0, 25.0, 2501)      # 0.01-order grid
        spectrum = np.zeros_like(orders)
        bpf, hw = 5.35, 0.25
        # Put unit amplitude in the ±0.25 band around the first 3 harmonics.
        for h in (1, 2, 3):
            c = bpf * h
            spectrum[(orders >= c - hw) & (orders <= c + hw)] = 1.0
        be = oa.bpfo_band_energy(orders, spectrum, bpf, n_harmonics=4, halfwidth=hw)
        # Each populated harmonic integrates 1.0 over a 0.5-order band ≈ 0.5.
        for h in (1, 2, 3):
            assert abs(be[f'h{h}'] - 0.5) < 0.02, f"h{h}={be[f'h{h}']}"
        assert be['h4'] == 0.0                      # 4th harmonic band is empty
        assert abs(be['total'] - 1.5) < 0.06

    def test_energy_higher_at_signal_order(self):
        df, _ = synth.pure_order_tone(5.35, 1.0, rpm=1200, n_rev=20)
        cfg = _oa_cfg()
        _, res = oa._compute_oa_segment((df, _oa_meta(), AXIS_COLS, ORDERS, cfg))
        cum = res['amps']['x'] + res['amps']['y'] + res['amps']['z']
        signal = oa.bpfo_band_energy(ORDERS, cum, 5.35, 1, 0.25)['h1']
        blank = oa.bpfo_band_energy(ORDERS, cum, 3.0, 1, 0.25)['h1']
        assert signal > 10 * blank


# ═══════════════════════════════════════════════════════════════════════════════
#  ANTI-ALIASING  (R1)  —  prove the LP removes fold-back, not just "runs"
# ═══════════════════════════════════════════════════════════════════════════════

class TestAntiAliasing:
    def test_above_nyquist_tone_is_suppressed(self):
        """A tone above the angular Nyquist folds to a spurious low order with
        plain interpolation; the anti-alias LP must suppress it ≥ 20×."""
        cfg = _oa_cfg(samples_per_rev=64)
        df, exp = synth.aliasing_probe(50.0, amp=1.0, rpm=2300, n_rev=20,
                                       samples_per_rev=64)
        rpm_min = 2300.0
        lp = max((cfg.samples_per_rev / 2) * rpm_min / 60.0, 10.0)
        _, amp_plain = _native_order_spectrum(df, cfg, lp=None, fs=None)
        o_fft, amp_aa = _native_order_spectrum(df, cfg, lp=lp, fs=FS)
        assert amp_plain.max() > 0.5, "alias should be strong without the LP"
        suppression = amp_plain.max() / max(amp_aa.max(), 1e-12)
        assert suppression >= 20.0, (
            f"anti-alias only suppressed the fold-back {suppression:.1f}× "
            f"(plain {amp_plain.max():.3f} -> AA {amp_aa.max():.3f})")

    def test_in_band_tone_passes_unattenuated(self):
        """The LP must not touch a tone well below the angular Nyquist."""
        cfg = _oa_cfg(samples_per_rev=64)
        df, _ = synth.pure_order_tone(5.0, 1.0, rpm=1200, n_rev=20)
        lp = (cfg.samples_per_rev / 2) * 1200.0 / 60.0
        _, amp = _native_order_spectrum(df, cfg, lp=lp, fs=FS)
        assert abs(amp.max() - 1.0) < 0.02


# ═══════════════════════════════════════════════════════════════════════════════
#  LOUDNESS / RMS  —  analytic A/√2
# ═══════════════════════════════════════════════════════════════════════════════

class TestRmsCalibration:
    @pytest.mark.parametrize('amp', [0.5, 1.0, 3.7])
    def test_ac_rms_of_sine_is_amp_over_sqrt2(self, amp):
        x, exp = synth.dc_offset_tone(amp, dc=0.0)
        got = per_axis_rms(x, remove_dc=True)
        assert abs(got - exp['ac_rms']) / exp['ac_rms'] < 1e-3

    def test_dc_offset_removed(self):
        x, exp = synth.dc_offset_tone(1.0, dc=10.0)
        ac = per_axis_rms(x, remove_dc=True)
        raw = per_axis_rms(x, remove_dc=False)
        assert abs(ac - exp['ac_rms']) / exp['ac_rms'] < 1e-3
        assert abs(raw - exp['raw_rms']) / exp['raw_rms'] < 1e-3
        assert raw > ac * 2.0                       # DC dominates the raw level

    def test_combined_axes_rms_is_root_sum_square(self):
        rng = np.random.default_rng(5)
        x, y, z = (rng.normal(0, s, 4000) for s in (1.0, 2.0, 0.5))
        expected = float(np.sqrt(np.mean(x**2 + y**2 + z**2)))
        got = combined_axes_rms([x, y, z], remove_dc=False)
        assert abs(got - expected) / expected < 1e-6

    def test_duckdb_rms_matches_analytic(self, tmp_path):
        """The DuckDB variance-form RMS of an A·sine equals A/√2 analytically."""
        import pandas as pd
        pytest.importorskip('duckdb')
        from nvh_pipeline import dataset
        amp = 2.0
        x, exp = synth.dc_offset_tone(amp, dc=4.0, n=6000)
        df = pd.DataFrame({
            'x_col': x, 't_rel': np.linspace(0, 1, len(x)),
            'segment_id': 0, 'material_type': 'metal', 'rpm_category': 1200,
            'part': 'm627', 'trial': 1, 'direction': 'POS',
            'source_file': 'syn.parquet',
        })
        (tmp_path / 'syn_segments.parquet').write_bytes(df.to_parquet(index=False))
        res = dataset.reduce_segments(str(tmp_path), agg_exprs={
            'rms_ac': 'SQRT(GREATEST(AVG("x_col"*"x_col") - AVG("x_col")*AVG("x_col"), 0))'})
        got = float(res['rms_ac'].iloc[0])
        assert abs(got - exp['ac_rms']) / exp['ac_rms'] < 1e-3


# ═══════════════════════════════════════════════════════════════════════════════
#  KURTOGRAM + ENVELOPE DEMODULATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestKurtogramFidelity:
    def test_impulsive_kurtosis_far_above_gaussian_floor(self):
        cfg = _ep_cfg()
        df_imp, _ = synth.impulse_train(2500.0, 5.35, rpm=1200, n_rev=20, seed=0)
        df_noise, _ = synth.pure_noise(8192, 1.0, seed=99, rpm=1200)
        _, _, _, k_imp = ep.detect_band_kurtogram(
            df_imp[synth.AXIS_COLS['x']].to_numpy(), FS, cfg, min_hz=500.0)
        _, _, _, k_noise = ep.detect_band_kurtogram(
            df_noise[synth.AXIS_COLS['x']].to_numpy(), FS, cfg, min_hz=500.0)
        assert k_imp > 6.0, f"impulsive kurtosis {k_imp:.2f} too low"
        assert k_noise < 5.0, f"Gaussian-noise kurtosis {k_noise:.2f} too high"
        assert k_imp > 2.0 * k_noise

    def test_band_overlaps_resonance(self):
        """Dyadic band must overlap the resonance within one band width."""
        cfg = _ep_cfg()
        df, _ = synth.impulse_train(2500.0, 5.35, rpm=1200, n_rev=20, seed=0)
        f_low, _, f_high, _ = ep.detect_band_kurtogram(
            df[synth.AXIS_COLS['x']].to_numpy(), FS, cfg, min_hz=500.0)
        res = 2500.0
        assert (f_high >= res - 1000.0) and (f_low <= res + 1000.0), (
            f"band [{f_low:.0f},{f_high:.0f}] does not overlap {res:.0f}±1000")

    def test_max_hz_frac_cap_respected(self):
        cfg = _ep_cfg(bp_max_hz_frac=0.6)
        rng = np.random.default_rng(3)
        hf = np.diff(np.cumsum(rng.standard_normal(8000)), prepend=0.0)
        _, _, f_high, _ = ep.detect_band_kurtogram(hf, 100_000.0, cfg, min_hz=1000.0)
        assert f_high <= 0.6 * 50_000.0 + 1e-6


class TestEnvelopeDemodulation:
    def test_direct_envelope_recovers_bpf_order_and_depth(self):
        """extract_envelope on the known carrier band recovers the ball-pass
        ORDER and the injected modulation DEPTH."""
        cfg = _ep_cfg()
        depth = 0.5
        df, exp = synth.am_carrier(2500.0, 5.35, depth, rpm=1200, n_rev=20)
        sig = df[synth.AXIS_COLS['x']].to_numpy()
        env = ep.extract_envelope(sig, FS, 2000.0, 3000.0, cfg)
        # correlation with the true AM profile must be ~1
        corr = np.corrcoef(env, exp['am_profile'])[0, 1]
        assert corr > 0.99, f"envelope/AM correlation {corr:.4f}"
        # envelope-order spectrum peaks at the ball-pass order with depth amplitude
        phi = ep.phi_revolutions(df, cfg)
        keep = np.concatenate([[True], np.diff(phi) > 0]); phi = phi[keep]
        M = int(np.floor(phi[-1] * cfg.samples_per_rev))
        phi_u = np.arange(M) / cfg.samples_per_rev
        o_fft = np.fft.rfftfreq(M, d=1.0 / cfg.samples_per_rev)
        win = get_window(cfg.window, M, fftbins=True)
        amp = ep.envelope_order_spectrum(env[keep], phi, M, win, phi_u, cfg)
        m = o_fft >= 1.0
        peak_order = o_fft[m][np.argmax(amp[m])]
        assert abs(peak_order - 5.35) < 0.1, f"env-order peak {peak_order}"
        assert abs(amp[m].max() - depth) / depth < 0.05, (
            f"recovered depth {amp[m].max():.4f} != {depth}")

    def test_worker_envelope_recovers_bpf_order(self):
        """Full envelope worker on an impulsive signal peaks at the ball-pass
        order (the real defect-detection deliverable)."""
        cfg = _ep_cfg()
        df, _ = synth.impulse_train(2500.0, 5.35, rpm=1200, n_rev=20, seed=0)
        task = (df, 'syn.parquet', 0, 'metal', 1200, 'm627', 'POS',
                AXIS_COLS, ORDERS, cfg)
        status, res = ep._compute_envelope_segment(task)
        assert status == 'ok'
        cum = res['amps']['x'] + res['amps']['y'] + res['amps']['z']
        m = ORDERS >= 1.0
        peak_order = ORDERS[m][np.argmax(cum[m])]
        assert abs(peak_order - 5.35) < 0.1, f"env-order peak {peak_order}"

    def test_impulsive_envelope_dwarfs_noise_envelope(self):
        """Discrimination: a real periodic defect produces a SHARP envelope-order
        line (high peak-to-median prominence) locked at the ball-pass order; a
        broadband-noise segment produces a flat envelope spectrum with no line.

        This documents the §6 plastic trap directly — plastic's damping yields
        noise, not impulses, so a confident-looking BPFO peak there would be
        spurious.  Prominence (peak/median) is the right metric because the
        per-segment RMS normalization removes absolute level.
        """
        cfg = _ep_cfg()
        df_imp, _ = synth.impulse_train(2500.0, 5.35, amp=0.6, noise=0.02,
                                        rpm=1200, n_rev=20, seed=0)
        df_noise, _ = synth.pure_noise(
            int(20 / (1200 / 60) * FS), sigma=0.1, seed=7, rpm=1200)

        def prominence(df, material):
            task = (df, 'syn.parquet', 0, material, 1200, 'm627', 'POS',
                    AXIS_COLS, ORDERS, cfg)
            st, res = ep._compute_envelope_segment(task)
            assert st == 'ok'
            cum = res['amps']['x'] + res['amps']['y'] + res['amps']['z']
            m = ORDERS >= 1.0
            sp = cum[m]
            dom_order = ORDERS[m][np.argmax(sp)]
            return float(sp.max() / np.median(sp)), float(dom_order)

        prom_imp, dom_imp = prominence(df_imp, 'metal')
        prom_noise, dom_noise = prominence(df_noise, 'plastic')
        # Impulsive defect: a tall, sharp line locked at the ball-pass order.
        assert prom_imp > 20.0, f"impulsive prominence {prom_imp:.1f} too low"
        assert abs(dom_imp - 5.35) < 0.1, f"impulsive line at order {dom_imp}"
        # Noise: flat spectrum, no dominant line.
        assert prom_noise < 6.0, f"noise prominence {prom_noise:.1f} too high"
        assert prom_imp > 5.0 * prom_noise, (
            f"impulsive prominence {prom_imp:.1f} not >> noise {prom_noise:.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ANGULAR DOMAIN — phi_revolutions
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhiRevolutions:
    def test_constant_rpm_slope(self):
        """dφ/dt for a constant-RPM run equals rpm/60 rev/s."""
        cfg = _oa_cfg()
        df, _ = synth.pure_order_tone(5.0, 1.0, rpm=1200, n_rev=20)
        phi = oa.phi_revolutions(df, cfg)
        t = df[synth.TIME_COL].to_numpy()
        slope = phi[-1] / t[-1]
        rpm_eff = df[synth.RPM_COL].iloc[0]
        assert abs(slope - rpm_eff / 60.0) / (rpm_eff / 60.0) < 1e-3

    def test_encoder_and_rpm_integration_agree(self):
        """Encoder-angle φ and RPM-time-integration φ must agree."""
        df, _ = synth.pure_order_tone(5.0, 1.0, rpm=1200, n_rev=20)
        phi_enc = oa.phi_revolutions(df, _oa_cfg(angle_source='encoder'))
        phi_rpm = oa.phi_revolutions(df, _oa_cfg(angle_source='rpm'))
        assert abs(phi_enc[-1] - phi_rpm[-1]) / phi_enc[-1] < 2e-3

    def test_monotonic_clamp_and_warning_on_glitch(self):
        """Encoder wrapping/glitches raise the documented warning and φ stays
        monotonic (clamped via maximum.accumulate)."""
        df, exp = synth.encoder_glitch(5.0, 1.0, rpm=1200, n_rev=20,
                                       frac_backward=0.10)
        with pytest.warns(UserWarning, match='non-monotonic'):
            phi = oa.phi_revolutions(df, _oa_cfg())
        assert np.all(np.diff(phi) >= 0), "φ must be non-decreasing after clamp"


# ═══════════════════════════════════════════════════════════════════════════════
#  EDGE CASES  —  "all possible situations"
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_at_rest_segment_skips(self):
        df, exp = synth.at_rest()
        cfg = _oa_cfg(min_revolutions=2.0)
        status, reason = oa._compute_oa_segment(
            (df, _oa_meta(), AXIS_COLS, ORDERS, cfg))
        assert status == 'skip' and reason == exp['expect_skip']

    def test_too_short_segment_skips(self):
        df, exp = synth.too_short(n_rev=0.2, rpm=1200)
        cfg = _oa_cfg(min_revolutions=2.0)
        status, reason = oa._compute_oa_segment(
            (df, _oa_meta(), AXIS_COLS, ORDERS, cfg))
        assert status == 'skip' and reason == 'min_revolutions'

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_nonfinite_sample_skips_envelope(self):
        df, exp = synth.with_nonfinite(5.0, 1.0, rpm=1200, n_rev=20,
                                       where=5, kind=np.inf)
        cfg = _ep_cfg()
        task = (df, 'syn.parquet', 0, 'metal', 1200, 'm627', 'POS',
                AXIS_COLS, ORDERS, cfg)
        status, reason = ep._compute_envelope_segment(task)
        assert status == 'skip' and reason == 'non_finite'

    def test_reversing_direction_recovers_order(self):
        """A NEG actuation (negative RPM / decreasing encoder) must yield the
        same order content — phi uses |angle - angle0|."""
        df, _ = synth.pure_order_tone(5.0, 1.0, rpm=1200, n_rev=20)
        df = df.copy()
        # Reverse: negate RPM and mirror the encoder so it decreases monotonically.
        df[synth.RPM_COL] = -df[synth.RPM_COL]
        df[synth.ANGLE_COL] = -df[synth.ANGLE_COL]
        cfg = _oa_cfg()
        status, res = oa._compute_oa_segment(
            (df, _oa_meta(), AXIS_COLS, ORDERS, cfg))
        assert status == 'ok'
        peak_order = ORDERS[np.argmax(res['amps']['x'])]
        assert abs(peak_order - 5.0) <= 0.05

    def test_ramp_speed_order_tracking(self):
        """Order tracking must recover an order-locked tone under a speed ramp."""
        df, exp = synth.ramp_speed(order=5.0, amp=1.0, rpm_lo=600, rpm_hi=1800,
                                   n_rev=20)
        cfg = _oa_cfg(min_revolutions=2.0)
        status, res = oa._compute_oa_segment(
            (df, _oa_meta(), AXIS_COLS, ORDERS, cfg))
        assert status == 'ok'
        peak_order = ORDERS[np.argmax(res['amps']['x'])]
        assert abs(peak_order - 5.0) <= 0.1, f"ramp peak {peak_order}"

    def test_pure_noise_returns_valid_band(self):
        df, exp = synth.pure_noise(8192, 1.0, seed=99, rpm=1200)
        cfg = _ep_cfg()
        f_low, f_ctr, f_high, kurt = ep.detect_band_kurtogram(
            df[synth.AXIS_COLS['x']].to_numpy(), FS, cfg, min_hz=500.0)
        assert f_low <= f_ctr <= f_high
        assert kurt > 0.0
        assert abs(kurt - exp['gaussian_kurtosis']) < 2.0   # near the Gaussian 3
