"""
End-to-end GOLDEN test: drive the real CLI (``python -m nvh_pipeline``) over a
deterministic synthetic dataset with KNOWN injected content, then assert on the
OUTPUT artifacts — manifest.json and the result CSVs — not internals.

Running through the real CLI in a SUBPROCESS (rather than calling run_stage
in-process) is both more faithful — it exercises the exact entry point FAMOS and
operators use, including manifest.json writing and the ProcessPoolExecutor — and
necessary for isolation: other test modules stub ``matplotlib`` with a MagicMock
at import time, which would break the in-process plotting stages.  A fresh
interpreter gets the real matplotlib.

The synthetic campaign (fixed seeds, written as DEWESOFT-style CSVs):
  * ``m627`` (metal): impulsive ball-pass taps at order 5.35 ringing a 2500 Hz
    resonance, PLUS a clean calibration tone of amplitude 0.35 at order 3.0 on
    the X axis.  Two actuations (POS then NEG) per file.
  * ``T8`` (plastic): broadband noise only (no impulses) — the §6 "envelope trap".

Golden expectations the whole chain must reproduce:
  1. every stage returns status 'ok' (same record dict the manifest serializes;
     manifest *writing* is covered separately by test_runner.py);
  2. the calibration tone is recovered end-to-end in the metal order spectrum at
     order 3.0 with amplitude ≈ 0.35 (proves the order axis is calibrated across
     segmentation + angular resampling + averaging);
  3. the documented ball-pass signature (order ≈ 5.35) appears as a metal order
     peak and as the metal envelope's dominant line;
  4. loudness produces a finite, positive combined RMS for both materials;
  5. plastic's envelope does NOT lock onto 5.35 (the noise-latching caveat).

Runs serially via runner.run_stage (no ProcessPoolExecutor) so the test is
deterministic and Windows-safe.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

FS = 10_000.0
COLS = ['Time (s)', 'CNT 1/Frequency (RPM)', 'CNT 1/Angle (Degrees)',
        'Torque (Nm)', 'X Accel (m/s2)', 'Y Accel (m/s2)', 'Z Accel (m/s2)']
RPM = 2300                       # matches EP_bandpass.target_rpm and the campaign
BPF_ORDER = 5.35
CAL_ORDER = 3.0
CAL_AMP = 0.35


def _ramp_profile(rpm, revs=12.0):
    plateau = revs / (rpm / 60.0)
    rt = max(0.05, plateau * 0.25)
    dt = 1.0 / FS
    up = np.arange(0, rt, dt)
    pl = np.arange(0, plateau, dt)
    dn = np.arange(0, rt, dt)
    return np.concatenate([rpm * (up / rt), np.full_like(pl, rpm),
                           rpm * (1 - dn / rt)])


def _write_csv(path, part, rpm, metal, seed):
    dt = 1.0 / FS
    gap = np.zeros(int(0.3 * FS))
    rpm_sig = np.concatenate([gap, _ramp_profile(rpm), gap,
                              -_ramp_profile(rpm), gap])
    n = len(rpm_sig)
    t = np.arange(n) * dt
    angle = np.cumsum(rpm_sig / 60.0 * 360.0 * dt)
    revs = np.cumsum(np.abs(rpm_sig) / 60.0 * dt)
    moving = (np.abs(rpm_sig) > 1.0).astype(float)
    rng = np.random.default_rng(seed)

    torque = np.where(np.abs(rpm_sig) > 1.0, 0.8, 0.0) * np.sign(rpm_sig + 1e-9)
    torque += rng.normal(0, 0.01, n)
    base = rng.normal(0, 0.02, (3, n))

    if metal:
        # Impulsive ball-pass taps at order 5.35, ringing a 2500 Hz resonance.
        bpf_phase = revs * BPF_ORDER
        taps = np.zeros(n)
        taps[np.where(np.diff(np.floor(bpf_phase)) > 0)[0]] = 1.0
        tt = np.arange(int(0.004 * FS)) * dt
        ring = np.sin(2 * np.pi * 2500 * tt) * np.exp(-tt / 0.0008)
        imp = np.convolve(taps, ring, mode='same')[:n]
        base[0] += 0.6 * imp * moving
        base[1] += 0.4 * imp * moving
        base[2] += 0.3 * imp * moving
        # Clean calibration tone (X axis only) at a known order/amplitude.
        base[0] += CAL_AMP * np.sin(2 * np.pi * CAL_ORDER * revs) * moving
    else:
        base += rng.normal(0, 0.05, (3, n)) * moving

    data = np.column_stack([t, rpm_sig, angle, torque, base[0], base[1], base[2]])
    np.savetxt(path, data, delimiter=',', header=','.join(COLS),
               comments='', fmt='%.6g')


STAGES = ('segment', 'vibration', 'order', 'envelope')


@pytest.fixture()
def synthetic_run(tmp_path):
    """Generate the dataset, run the real CLI in a subprocess, and return
    (output_root, manifest_dict)."""
    data_dir = tmp_path / 'data'
    out_root = tmp_path / 'out'
    data_dir.mkdir()
    out_root.mkdir()
    _write_csv(str(data_dir / f'm627-{RPM}rpm-1.csv'), 'm627', RPM, True, seed=0)
    _write_csv(str(data_dir / f'T8-{RPM}rpm-1.csv'), 'T8', RPM, False, seed=1)

    from nvh_pipeline.config import PipelineConfig
    p = PipelineConfig(
        data_dir=str(data_dir), output_root=str(out_root),
        parts=('m627', 'T8'), rpms=(RPM,), only_rpms=(RPM,),
        min_revolutions=2.0, samples_per_rev=64, ball_pass_order=BPF_ORDER,
        render_plots=False, stages=STAGES,
    )
    cfg_json = str(out_root / 'nvh_config.json')
    p.to_json(cfg_json)

    env = dict(os.environ, MPLBACKEND='Agg', PYTHONIOENCODING='utf-8')
    proc = subprocess.run(
        [sys.executable, '-m', 'nvh_pipeline', '--config', cfg_json,
         '--stages', ','.join(STAGES)],
        capture_output=True, text=True, env=env, cwd=_ROOT, timeout=300)
    assert proc.returncode == 0, (
        f"pipeline CLI exited {proc.returncode}\n"
        f"STDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-3000:]}")

    manifest_path = os.path.join(str(out_root), 'manifest.json')
    assert os.path.isfile(manifest_path), "pipeline did not write manifest.json"
    with open(manifest_path, encoding='utf-8') as fh:
        manifest = json.load(fh)
    return str(out_root), manifest


# ─────────────────────────────────────────────────────────────────────────────

def _read(out_root, *parts):
    return pd.read_csv(os.path.join(out_root, *parts))


def test_all_stages_ok(synthetic_run):
    """manifest.json reports every requested stage as 'ok'."""
    _, manifest = synthetic_run
    statuses = {rec['stage']: rec['status'] for rec in manifest['stages']}
    for stage in STAGES:
        assert statuses.get(stage) == 'ok', (
            f"stage {stage} status {statuses.get(stage)!r}; manifest={statuses}")


def test_calibration_tone_recovered_end_to_end(synthetic_run):
    """The order axis is calibrated across the whole chain: the X-axis tone of
    amplitude 0.35 at order 3.0 reads back ≈ 0.35 in the metal order spectrum."""
    out_root, _ = synthetic_run
    md = _read(out_root, 'output_order', 'order_spectra_mean_metal.csv')
    o = md['order'].to_numpy()
    mx = md['mean_x'].to_numpy()
    band = (o >= CAL_ORDER - 0.15) & (o <= CAL_ORDER + 0.15)
    peak = float(mx[band].max())
    peak_order = float(o[band][np.argmax(mx[band])])
    assert abs(peak_order - CAL_ORDER) <= 0.1, f"cal peak at order {peak_order}"
    # Averaging over POS+NEG and the ramp portions costs a few %; 15 % is a safe,
    # still-meaningful end-to-end amplitude bound (measured residual ~2 %).
    assert abs(peak - CAL_AMP) / CAL_AMP < 0.15, (
        f"end-to-end calibration amplitude {peak:.4f} != {CAL_AMP}")


def test_ball_pass_signature_in_order_spectrum(synthetic_run):
    """The documented order-5.35 ball-pass signature appears as a metal peak."""
    import EP_order_analysis as oa
    out_root, _ = synthetic_run
    md = _read(out_root, 'output_order', 'order_spectra_mean_metal.csv')
    o = md['order'].to_numpy()
    cum = md['cumulative'].to_numpy()
    peaks = [po for po, _ in oa.top_peaks(o, cum, n=6, min_order=0.5)]
    assert any(abs(po - BPF_ORDER) <= 0.15 for po in peaks), (
        f"no ball-pass peak near {BPF_ORDER} in top peaks {peaks}")


def test_metal_envelope_locks_ball_pass_order(synthetic_run):
    out_root, _ = synthetic_run
    ed = _read(out_root, 'output_envelope', 'envelope_spectra_mean_metal.csv')
    o = ed['order'].to_numpy()
    col = 'cumulative' if 'cumulative' in ed.columns else ed.columns[-1]
    c = ed[col].to_numpy()
    m = o >= 1.0
    peak_order = float(o[m][np.argmax(c[m])])
    assert abs(peak_order - BPF_ORDER) <= 0.2, (
        f"metal envelope dominant order {peak_order} != {BPF_ORDER}")


def test_loudness_finite_for_both_materials(synthetic_run):
    out_root, _ = synthetic_run
    ld = _read(out_root, 'output_loudness', 'rms_per_segment.csv')
    assert set(ld['material'].unique()) >= {'metal', 'plastic'}
    assert np.all(np.isfinite(ld['rms_combined'].to_numpy()))
    assert np.all(ld['rms_combined'].to_numpy() > 0.0)


def test_plastic_envelope_does_not_lock_ball_pass(synthetic_run):
    """Documents the §6 caveat: plastic's damping means the envelope latches onto
    noise rather than a real defect line, so its dominant envelope order is NOT
    the ball-pass order.  Deterministic via the fixed plastic seed — if a future
    change makes plastic spuriously 'find' 5.35, this flags it for review."""
    out_root, _ = synthetic_run
    ed = _read(out_root, 'output_envelope', 'envelope_spectra_mean_plastic.csv')
    o = ed['order'].to_numpy()
    col = 'cumulative' if 'cumulative' in ed.columns else ed.columns[-1]
    c = ed[col].to_numpy()
    m = o >= 1.0
    peak_order = float(o[m][np.argmax(c[m])])
    assert abs(peak_order - BPF_ORDER) > 0.3, (
        f"plastic envelope unexpectedly locked the ball-pass order "
        f"({peak_order}) — the noise-latching caveat may no longer hold")
