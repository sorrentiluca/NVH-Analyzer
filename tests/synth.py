"""
Analytically-grounded synthetic signal library for the NVH validation suite.

Every builder returns ``(df, expected)`` where ``df`` is a segment DataFrame in
the exact shape the EP_* workers consume (encoder angle + RPM + 3 accel axes +
the metadata columns the order/envelope ``run`` paths read), and ``expected`` is
a dict of the *closed-form* answers that builder's construction guarantees.

The point of this module is that the validation tests assert against MATH, not
against a prior run.  Each signal's spectral content / RMS / band is known by
construction, so a passing test means the pipeline reproduces the known truth —
not merely that it reproduces yesterday's output.

Ground-truth derivations (see the plan file for code references):
  * ``order_spectrum`` / ``envelope_order_spectrum`` return
    ``|rfft(x·w)|·2/Σw`` with the DC bin halved.  For a pure tone of amplitude A
    landing on an FFT bin, the single-sided amplitude reads back A exactly.
    An order lands on a bin when ``order·n_rev`` is an integer (bin spacing in
    the order domain is ``1/n_rev``).
  * Per-axis RMS of ``A·sin`` is ``A/√2``; combined axis RMS is
    ``sqrt(Σ axis²)``; a DC offset cancels when ``remove_dc=True``.
  * Angular Nyquist (in orders) is ``samples_per_rev/2``; content above it folds
    to ``samples_per_rev − order`` unless anti-alias filtered first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Time-domain sample rate (Hz).  Matches make_testdata.py and the existing tests.
FS = 10_000.0

TIME_COL = 'Time (s)'
RPM_COL = 'CNT 1/Frequency (RPM)'
ANGLE_COL = 'CNT 1/Angle (Degrees)'
AXIS_COLS = {
    'x': 'X Accel (m/s2)',
    'y': 'Y Accel (m/s2)',
    'z': 'Z Accel (m/s2)',
}

# Standard order grid used by the workers (0.1-order step, like the test suite).
COMMON_ORDERS = np.arange(0.0, 12.0 + 0.1, 0.1)


def _meta_columns(df: pd.DataFrame, *, part='m627', material='metal',
                  rpm_cat=1200, trial=1, direction='POS', segment_id=0):
    """Attach the metadata columns the order/envelope ``run`` paths expect."""
    df['material_type'] = material
    df['rpm_category'] = int(rpm_cat)
    df['part'] = part
    df['trial'] = int(trial)
    df['direction'] = direction
    df['segment_id'] = int(segment_id)
    return df


def _constant_rpm_frame(rpm: float, n_rev: float, fs: float = FS) -> dict:
    """Time base + linear encoder angle for a constant-RPM segment.

    The encoder angle is built with ``linspace(0, n_rev·360, n)`` so the segment
    ends at *exactly* ``n_rev`` revolutions.  This matters for amplitude
    calibration: when ``n_rev`` is an integer the worker's angular FFT length is
    ``n_rev·samples_per_rev`` exactly, so an integer order lands precisely on an
    FFT bin and the recovered amplitude equals A to ~1e-3 (verified empirically).
    A plain ``rpm/60·t`` ramp would stop a fraction of a rev short and add ~4 %
    scalloping loss.  The RPM column carries the matching constant mean speed.

    Returns ``t``, ``rpm`` (array), ``angle`` (deg), ``revs`` and ``n``.
    """
    duration = n_rev / (rpm / 60.0)
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    angle = np.linspace(0.0, n_rev * 360.0, n)   # exact: revs[-1] == n_rev
    revs = angle / 360.0
    # Constant RPM consistent with this exact angle span (used only by the
    # anti-alias LP; encoder angle is the order-analysis source of truth).
    rpm_const = (n_rev / t[-1]) * 60.0 if n > 1 and t[-1] > 0 else rpm
    return {'t': t, 'rpm': np.full(n, rpm_const), 'angle': angle,
            'revs': revs, 'n': n}


# ─────────────────────────────────────────────────────────────────────────────
#  ORDER-DOMAIN BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def pure_order_tone(order: float, amp: float, rpm: float = 1200.0,
                    n_rev: int = 20, axis_scale=(1.0, 0.5, 0.3),
                    fs: float = FS):
    """A·sin(2π·order·rev) on each axis (scaled).  On-grid when order·n_rev ∈ ℤ.

    Expected order-spectrum peak per axis = ``amp * axis_scale[axis]`` at
    ``order``; with an integer ``n_rev`` and integer-ish ``order`` the tone lands
    exactly on an FFT bin so the recovered amplitude equals A with no leakage.
    """
    fr = _constant_rpm_frame(rpm, n_rev, fs)
    sig = amp * np.sin(2 * np.pi * order * fr['revs'])
    df = pd.DataFrame({
        TIME_COL: fr['t'],
        RPM_COL: fr['rpm'],
        ANGLE_COL: fr['angle'],
        AXIS_COLS['x']: sig * axis_scale[0],
        AXIS_COLS['y']: sig * axis_scale[1],
        AXIS_COLS['z']: sig * axis_scale[2],
    })
    _meta_columns(df, rpm_cat=int(rpm))
    on_grid = abs((order * n_rev) - round(order * n_rev)) < 1e-9
    expected = {
        'order': order, 'amp': amp, 'rpm': rpm, 'n_rev': n_rev,
        'on_grid': on_grid,
        'peak_amp': {ax: amp * s for ax, s in zip('xyz', axis_scale)},
        'order_bin_spacing': 1.0 / n_rev,
    }
    return df, expected


def multi_order(components, rpm: float = 1200.0, n_rev: int = 20,
                fs: float = FS):
    """Superposition of (order, amp) tones on the X axis (y, z = 0.5×, 0.3×).

    Expected: an isolated spectral peak of height ``amp`` at each component's
    ``order`` (components must be separated by > 1 order to be independently
    resolvable at 0.1-order grid).
    """
    fr = _constant_rpm_frame(rpm, n_rev, fs)
    sig = np.zeros(fr['n'])
    for order, amp in components:
        sig = sig + amp * np.sin(2 * np.pi * order * fr['revs'])
    df = pd.DataFrame({
        TIME_COL: fr['t'],
        RPM_COL: fr['rpm'],
        ANGLE_COL: fr['angle'],
        AXIS_COLS['x']: sig,
        AXIS_COLS['y']: sig * 0.5,
        AXIS_COLS['z']: sig * 0.3,
    })
    _meta_columns(df, rpm_cat=int(rpm))
    expected = {'components': list(components), 'rpm': rpm, 'n_rev': n_rev}
    return df, expected


def aliasing_probe(order_above_nyquist: float, amp: float = 1.0,
                   rpm: float = 2300.0, n_rev: int = 20,
                   samples_per_rev: int = 64, fs: float = FS):
    """A tone ABOVE the angular Nyquist (samples_per_rev/2 orders).

    Without anti-aliasing it folds to ``samples_per_rev − order`` and appears as
    a spurious low-order peak.  With the R1 anti-alias LP it should be removed.
    The tone's *time* frequency (order·rpm/60) is kept below fs/2 so it is
    physically present in the time series.
    """
    fr = _constant_rpm_frame(rpm, n_rev, fs)
    sig = amp * np.sin(2 * np.pi * order_above_nyquist * fr['revs'])
    df = pd.DataFrame({
        TIME_COL: fr['t'],
        RPM_COL: fr['rpm'],
        ANGLE_COL: fr['angle'],
        AXIS_COLS['x']: sig,
        AXIS_COLS['y']: sig,
        AXIS_COLS['z']: sig,
    })
    _meta_columns(df, rpm_cat=int(rpm))
    alias_order = samples_per_rev - order_above_nyquist
    expected = {
        'true_order': order_above_nyquist,
        'alias_order': alias_order,
        'time_freq_hz': order_above_nyquist * rpm / 60.0,
        'angular_nyquist_order': samples_per_rev / 2.0,
        'samples_per_rev': samples_per_rev,
    }
    return df, expected


# ─────────────────────────────────────────────────────────────────────────────
#  RMS / LOUDNESS BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def dc_offset_tone(amp: float, dc: float, n: int = 4000, freq_hz: float = 137.0,
                   fs: float = FS):
    """A·sin + DC offset on a plain time axis.  Expected AC-RMS = amp/√2,
    independent of ``dc``; raw RMS = sqrt((amp/√2)² + dc²)."""
    t = np.arange(n) / fs
    x = amp * np.sin(2 * np.pi * freq_hz * t) + dc
    expected = {
        'ac_rms': amp / np.sqrt(2.0),
        'raw_rms': float(np.sqrt((amp ** 2) / 2.0 + dc ** 2)),
        'amp': amp, 'dc': dc,
    }
    return x, expected


# ─────────────────────────────────────────────────────────────────────────────
#  ENVELOPE / KURTOGRAM BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def am_carrier(resonance_hz: float, bpf_order: float, depth: float,
               rpm: float = 1200.0, n_rev: int = 20, fs: float = FS):
    """Resonance carrier amplitude-modulated at the ball-pass ORDER.

    sig(t) = sin(2π·f_res·t) · (1 + depth·sin(2π·bpf_order·rev))

    The demodulated envelope is ``1 + depth·sin(2π·bpf_order·rev)``, so the
    envelope-order spectrum must peak at ``bpf_order``.  ``f_res`` must sit below
    Nyquist and inside the kurtogram search band.
    """
    fr = _constant_rpm_frame(rpm, n_rev, fs)
    am = 1.0 + depth * np.sin(2 * np.pi * bpf_order * fr['revs'])
    carrier = np.sin(2 * np.pi * resonance_hz * fr['t'])
    sig = carrier * am
    df = pd.DataFrame({
        TIME_COL: fr['t'],
        RPM_COL: fr['rpm'],
        ANGLE_COL: fr['angle'],
        AXIS_COLS['x']: sig,
        AXIS_COLS['y']: sig,
        AXIS_COLS['z']: sig,
    })
    _meta_columns(df, rpm_cat=int(rpm))
    expected = {
        'resonance_hz': resonance_hz, 'bpf_order': bpf_order, 'depth': depth,
        'rpm': rpm, 'n_rev': n_rev, 'am_profile': am,
    }
    return df, expected


def impulse_train(resonance_hz: float, bpf_order: float, amp: float = 0.6,
                  noise: float = 0.02, rpm: float = 1200.0, n_rev: int = 20,
                  seed: int = 0, fs: float = FS):
    """Periodic ball-pass taps (at ``bpf_order``) ringing a damped resonance,
    plus a broadband floor.  Mirrors make_testdata.py's metal-part construction.

    Expected: kurtogram band CONTAINS ``resonance_hz``; raw kurtosis ≫ 3 (the
    Gaussian floor); envelope-order spectrum peaks at ``bpf_order``.
    """
    fr = _constant_rpm_frame(rpm, n_rev, fs)
    rng = np.random.default_rng(seed)
    n = fr['n']
    taps = np.zeros(n)
    bpf_phase = fr['revs'] * bpf_order
    crossings = np.where(np.diff(np.floor(bpf_phase)) > 0)[0]
    taps[crossings] = 1.0
    tt = np.arange(int(0.004 * fs)) / fs
    ring = np.sin(2 * np.pi * resonance_hz * tt) * np.exp(-tt / 0.0008)
    imp = np.convolve(taps, ring, mode='same')[:n]
    sig = amp * imp + rng.normal(0, noise, n)
    df = pd.DataFrame({
        TIME_COL: fr['t'],
        RPM_COL: fr['rpm'],
        ANGLE_COL: fr['angle'],
        AXIS_COLS['x']: sig,
        AXIS_COLS['y']: sig,
        AXIS_COLS['z']: sig,
    })
    _meta_columns(df, rpm_cat=int(rpm))
    expected = {
        'resonance_hz': resonance_hz, 'bpf_order': bpf_order,
        'n_taps': int(taps.sum()), 'rpm': rpm, 'n_rev': n_rev,
    }
    return df, expected


# ─────────────────────────────────────────────────────────────────────────────
#  EDGE-CASE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def at_rest(n: int = 2000, fs: float = FS):
    """Shaft at rest: RPM≈0, encoder angle flat.  Order analysis must skip
    (zero revolutions)."""
    t = np.arange(n) / fs
    df = pd.DataFrame({
        TIME_COL: t,
        RPM_COL: np.zeros(n),
        ANGLE_COL: np.zeros(n),
        AXIS_COLS['x']: np.random.default_rng(0).normal(0, 0.02, n),
        AXIS_COLS['y']: np.random.default_rng(1).normal(0, 0.02, n),
        AXIS_COLS['z']: np.random.default_rng(2).normal(0, 0.02, n),
    })
    _meta_columns(df, rpm_cat=0)
    return df, {'expect_skip': 'min_revolutions'}


def too_short(n_rev: float = 0.2, rpm: float = 1200.0, fs: float = FS):
    """Fewer revolutions than any sane ``min_revolutions`` → skip."""
    df, _ = pure_order_tone(5.0, 1.0, rpm=rpm,
                            n_rev=max(1, int(round(n_rev * fs / fs))) or 1)
    # rebuild explicitly with a fractional rev count
    fr = _constant_rpm_frame(rpm, n_rev, fs)
    sig = np.sin(2 * np.pi * 5.0 * fr['revs'])
    df = pd.DataFrame({
        TIME_COL: fr['t'], RPM_COL: fr['rpm'], ANGLE_COL: fr['angle'],
        AXIS_COLS['x']: sig, AXIS_COLS['y']: sig, AXIS_COLS['z']: sig,
    })
    _meta_columns(df, rpm_cat=int(rpm))
    return df, {'expect_skip': 'min_revolutions', 'n_rev': n_rev}


def with_nonfinite(order: float = 5.0, amp: float = 1.0, rpm: float = 1200.0,
                   n_rev: int = 20, where: int = 5, kind: float = np.inf):
    """A clean tone with a single non-finite sample injected (inf or nan)."""
    df, expected = pure_order_tone(order, amp, rpm=rpm, n_rev=n_rev)
    df = df.copy()
    df.loc[where, AXIS_COLS['x']] = kind
    expected['expect_skip'] = 'non_finite'
    return df, expected


def encoder_glitch(order: float = 5.0, amp: float = 1.0, rpm: float = 1200.0,
                   n_rev: int = 20, frac_backward: float = 0.10):
    """A clean tone whose encoder angle has many backward steps (wrapping /
    glitches) — must trigger the phi_revolutions non-monotonic warning and be
    clamped via maximum.accumulate."""
    df, expected = pure_order_tone(order, amp, rpm=rpm, n_rev=n_rev)
    df = df.copy()
    # .copy(): with pandas copy-on-write, to_numpy() can be a read-only view.
    ang = df[ANGLE_COL].to_numpy().copy()
    rng = np.random.default_rng(3)
    n_back = int(frac_backward * len(ang))
    idx = rng.choice(np.arange(1, len(ang)), size=n_back, replace=False)
    ang[idx] -= 50.0                     # push selected steps backward
    df[ANGLE_COL] = ang
    expected['expect_warning'] = True
    return df, expected


def ramp_speed(order: float = 5.0, amp: float = 1.0, rpm_lo: float = 600.0,
               rpm_hi: float = 1800.0, n_rev: int = 20, fs: float = FS):
    """Variable-speed run: RPM ramps linearly, but the vibration stays locked to
    shaft ORDER, so order tracking must still recover the tone at ``order``.

    Built in the angle domain so the tone is exactly periodic in revolutions
    regardless of the speed profile."""
    # Integrate a linear-in-time RPM ramp to get angle, then place the tone in
    # revolutions so it is order-locked.
    # Choose duration so that mean speed yields ~n_rev revolutions.
    mean_rpm = 0.5 * (rpm_lo + rpm_hi)
    duration = n_rev / (mean_rpm / 60.0)
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    rpm = rpm_lo + (rpm_hi - rpm_lo) * (t / t[-1])
    dt = 1.0 / fs
    angle = np.cumsum(rpm / 60.0 * 360.0 * dt)       # deg
    revs = angle / 360.0
    sig = amp * np.sin(2 * np.pi * order * revs)
    df = pd.DataFrame({
        TIME_COL: t, RPM_COL: rpm, ANGLE_COL: angle,
        AXIS_COLS['x']: sig, AXIS_COLS['y']: sig * 0.5, AXIS_COLS['z']: sig * 0.3,
    })
    _meta_columns(df, rpm_cat=int(round(mean_rpm)))
    return df, {'order': order, 'amp': amp, 'n_rev': float(revs[-1])}


def pure_noise(n: int = 8192, sigma: float = 1.0, seed: int = 99,
               rpm: float = 1200.0, fs: float = FS):
    """Gaussian noise on a constant-RPM frame.  Kurtogram must return a valid
    band; raw kurtosis ≈ 3 (Gaussian)."""
    n_rev = n / fs * (rpm / 60.0)
    fr = _constant_rpm_frame(rpm, max(1.0, n_rev), fs)
    m = fr['n']
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, m)
    df = pd.DataFrame({
        TIME_COL: fr['t'], RPM_COL: fr['rpm'], ANGLE_COL: fr['angle'],
        AXIS_COLS['x']: noise, AXIS_COLS['y']: noise, AXIS_COLS['z']: noise,
    })
    _meta_columns(df, rpm_cat=int(rpm), material='plastic', part='T8')
    return df, {'sigma': sigma, 'gaussian_kurtosis': 3.0}
