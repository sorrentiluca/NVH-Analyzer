"""
Signable VISUAL validation report for the NVH analysis pipeline.

Runs the real pipeline functions on the analytically-grounded synthetic signals
from ``tests/synth.py`` and renders, per case, an EXPECTED-vs-MEASURED overlay
plus a PASS/FAIL summary table — an artifact an engineer can review and sign,
backing the automated pytest gate in ``test_validation_fidelity.py``.

Run it standalone (fresh interpreter → real matplotlib):

    python tests/validation_report.py
    # writes output_validation/validation_report.pdf  and  validation_summary.csv

Exit code is the number of FAILED cases (0 = all pass), so it doubles as a CI
check.  The ``report``-marked test in test_validation_fidelity-style suites runs
this in a subprocess and asserts a clean report.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Make repo root + tests/ importable whether run as a script or `-m`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import get_window

import synth
import EP_order_analysis as oa
import EP_Envelope as ep
from nvh_pipeline.common import per_axis_rms

FS = synth.FS
ORDERS = np.arange(0.0, 12.0 + 0.02, 0.02)
AXIS_COLS = {k: synth.AXIS_COLS[k] for k in 'xyz'}
OUTDIR = os.path.join(_ROOT, 'output_validation')


def _long(path: str) -> str:
    """Windows extended-length path so writes survive a >260-char repo root
    (this project lives under a deep OneDrive path).  No-op elsewhere."""
    ap = os.path.abspath(path)
    if os.name == 'nt' and not ap.startswith('\\\\?\\'):
        return '\\\\?\\' + ap
    return ap


def _oa_cfg(**kw):
    base = dict(samples_per_rev=64, min_revolutions=0.5, min_fft_points=8,
                detrend='mean', window='hann', angle_source='encoder',
                ball_pass_order=5.35, n_bpf_harmonics=4)
    base.update(kw)
    return oa.Config(**base)


def _ep_cfg(**kw):
    base = dict(samples_per_rev=64, min_revolutions=0.5, min_fft_points=8,
                detrend='mean', window='hann', kurtogram_levels=4,
                bp_min_hz=500.0, bp_min_hz_abs=50.0, bp_min_bw_hz=200.0,
                bp_max_hz_frac=0.8, ball_pass_order=5.35, only_rpms=(1200,))
    base.update(kw)
    return ep.Config(**base)


def _native_order_spectrum(df, cfg, axis='x', lp=None, fs=None):
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


# ─────────────────────────────────────────────────────────────────────────────
#  Cases  —  each returns (title, passed, detail, draw(ax))
# ─────────────────────────────────────────────────────────────────────────────

def case_order_amplitude():
    order, amp = 5.0, 0.7
    df, _ = synth.pure_order_tone(order, amp, rpm=1200, n_rev=20)
    o, a = _native_order_spectrum(df, _oa_cfg())
    measured = float(a.max())
    err = abs(measured - amp) / amp
    passed = err < 2e-3

    def draw(ax):
        ax.plot(o, a, lw=0.8, color='#1f77b4', label='measured order spectrum')
        ax.axhline(amp, ls='--', color='k', lw=0.8, label=f'expected A={amp}')
        ax.plot([order], [measured], 'rv', label=f'measured peak={measured:.4f}')
        ax.set_xlim(0, 8); ax.set_xlabel('order'); ax.set_ylabel('amplitude')
        ax.legend(fontsize=6, loc='upper right')
    return ('Order amplitude calibration (A=0.7 @ order 5)', passed,
            f'measured {measured:.4f}, rel err {err:.2e} (tol 2e-3)', draw)


def case_two_tone_resolution():
    df, _ = synth.multi_order([(3.0, 1.0), (8.0, 0.6)], rpm=1200, n_rev=20)
    _, res = oa._compute_oa_segment((df, _oa_meta(), AXIS_COLS, ORDERS, _oa_cfg()))
    sp = res['amps']['x']
    peaks = dict((round(po), pa) for po, pa in oa.top_peaks(ORDERS, sp, 2, 0.5))
    ratio = peaks.get(8, 0) / peaks.get(3, 1e-9)
    passed = (set(peaks) == {3, 8}) and abs(ratio - 0.6) < 0.03

    def draw(ax):
        ax.plot(ORDERS, sp, color='#1f77b4', lw=0.8, label='measured')
        for ordr, exp_a in ((3.0, 1.0), (8.0, 0.6)):
            ax.axvline(ordr, ls=':', color='r', lw=0.8)
            ax.plot([ordr], [exp_a], 'k_', ms=14)
        ax.set_xlim(0, 11); ax.set_xlabel('order'); ax.set_ylabel('amplitude')
        ax.set_title('expected lines: order 3 (1.0) & 8 (0.6)', fontsize=7)
        ax.legend(fontsize=6)
    return ('Two-tone order resolution', passed,
            f'peaks {sorted(peaks)}, amp ratio {ratio:.3f} (expect 0.60)', draw)


def case_anti_aliasing():
    cfg = _oa_cfg(samples_per_rev=64)
    df, exp = synth.aliasing_probe(50.0, 1.0, rpm=2300, n_rev=20,
                                   samples_per_rev=64)
    lp = (cfg.samples_per_rev / 2) * 2300.0 / 60.0
    o, plain = _native_order_spectrum(df, cfg)
    _, aa = _native_order_spectrum(df, cfg, lp=lp, fs=FS)
    supp = plain.max() / max(aa.max(), 1e-12)
    passed = supp >= 20.0

    def draw(ax):
        ax.plot(o, plain, color='#d62728', lw=0.8, label='plain interp (aliased)')
        ax.plot(o, aa, color='#2ca02c', lw=0.8, label='anti-aliased (R1)')
        ax.axvline(exp['alias_order'], ls=':', color='k', lw=0.8,
                   label=f"fold-back order {exp['alias_order']:.0f}")
        ax.set_xlim(0, 20); ax.set_xlabel('order'); ax.set_ylabel('amplitude')
        ax.legend(fontsize=6)
    return ('Anti-aliasing suppresses above-Nyquist fold-back', passed,
            f'fold-back suppression {supp:.1f}x (tol >= 20x)', draw)


def case_rms_calibration():
    amps = [0.5, 1.0, 3.7]
    measured = [per_axis_rms(synth.dc_offset_tone(a, dc=10.0)[0], remove_dc=True)
                for a in amps]
    expected = [a / np.sqrt(2.0) for a in amps]
    errs = [abs(m - e) / e for m, e in zip(measured, expected)]
    passed = all(e < 1e-3 for e in errs)

    def draw(ax):
        x = np.arange(len(amps))
        ax.bar(x - 0.2, expected, 0.4, label='expected A/√2', color='k', alpha=0.5)
        ax.bar(x + 0.2, measured, 0.4, label='measured (DC removed)',
               color='#1f77b4')
        ax.set_xticks(x); ax.set_xticklabels([f'A={a}' for a in amps])
        ax.set_ylabel('RMS'); ax.legend(fontsize=6)
        ax.set_title('DC offset = 10 must cancel', fontsize=7)
    return ('Loudness RMS calibration (A/√2, DC removed)', passed,
            f'max rel err {max(errs):.2e} (tol 1e-3)', draw)


def case_kurtogram():
    cfg = _ep_cfg()
    df, _ = synth.impulse_train(2500.0, 5.35, rpm=1200, n_rev=20, seed=0)
    sig = df[synth.AXIS_COLS['x']].to_numpy()
    f_low, f_ctr, f_high, kurt = ep.detect_band_kurtogram(sig, FS, cfg, min_hz=500.0)
    res = 2500.0
    overlap = (f_high >= res - 1000.0) and (f_low <= res + 1000.0)
    passed = overlap and kurt > 6.0
    fr = np.fft.rfftfreq(len(sig), d=1.0 / FS)
    mag = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))

    def draw(ax):
        ax.plot(fr, mag, color='#1f77b4', lw=0.6, label='|spectrum|')
        ax.axvspan(f_low, f_high, color='#2ca02c', alpha=0.25,
                   label=f'kurtogram band [{f_low:.0f},{f_high:.0f}]')
        ax.axvline(res, ls='--', color='r', lw=1.0, label='resonance 2500 Hz')
        ax.set_xlim(0, 5000); ax.set_xlabel('Hz'); ax.set_ylabel('mag')
        ax.legend(fontsize=6)
    return ('Kurtogram band vs resonance (impulsive)', passed,
            f'band [{f_low:.0f},{f_high:.0f}] overlap={overlap}, kurt={kurt:.1f}',
            draw)


def case_envelope_demod():
    cfg = _ep_cfg()
    depth = 0.5
    df, exp = synth.am_carrier(2500.0, 5.35, depth, rpm=1200, n_rev=20)
    sig = df[synth.AXIS_COLS['x']].to_numpy()
    env = ep.extract_envelope(sig, FS, 2000.0, 3000.0, cfg)
    phi = ep.phi_revolutions(df, cfg)
    keep = np.concatenate([[True], np.diff(phi) > 0]); phi = phi[keep]
    M = int(np.floor(phi[-1] * cfg.samples_per_rev))
    phi_u = np.arange(M) / cfg.samples_per_rev
    o = np.fft.rfftfreq(M, d=1.0 / cfg.samples_per_rev)
    win = get_window(cfg.window, M, fftbins=True)
    amp = ep.envelope_order_spectrum(env[keep], phi, M, win, phi_u, cfg)
    m = o >= 1.0
    peak_order = float(o[m][np.argmax(amp[m])])
    rec_depth = float(amp[m].max())
    passed = abs(peak_order - 5.35) < 0.1 and abs(rec_depth - depth) / depth < 0.05

    def draw(ax):
        ax.plot(o, amp, color='#1f77b4', lw=0.8, label='envelope-order spectrum')
        ax.axvline(5.35, ls='--', color='r', lw=0.8, label='ball-pass order 5.35')
        ax.axhline(depth, ls=':', color='k', lw=0.8, label=f'expected depth {depth}')
        ax.plot([peak_order], [rec_depth], 'gv',
                label=f'measured {rec_depth:.3f} @ {peak_order:.2f}')
        ax.set_xlim(0, 11); ax.set_xlabel('order'); ax.set_ylabel('amplitude')
        ax.legend(fontsize=6)
    return ('Envelope demodulation recovers ball-pass order + depth', passed,
            f'peak order {peak_order:.3f}, depth {rec_depth:.4f} (expect 5.35/0.5)',
            draw)


def _oa_meta():
    return {'part': 'm627', 'material_type': 'metal', 'rpm_category': 1200,
            'trial': 1, 'direction': 'POS', 'segment_id': 0,
            'source_file': 'synthetic.parquet'}


CASES = [case_order_amplitude, case_two_tone_resolution, case_anti_aliasing,
         case_rms_calibration, case_kurtogram, case_envelope_demod]


def build_report():
    os.makedirs(OUTDIR, exist_ok=True)
    results = [fn() for fn in CASES]
    pdf_path = os.path.join(OUTDIR, 'validation_report.pdf')
    csv_path = os.path.join(OUTDIR, 'validation_summary.csv')

    with PdfPages(_long(pdf_path)) as pdf:
        # Summary page
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.axis('off')
        ax.set_title('NVH pipeline — accuracy validation summary',
                     fontsize=13, weight='bold')
        rows = [['#', 'Case', 'Result', 'Detail']]
        for i, (title, passed, detail, _) in enumerate(results, 1):
            rows.append([str(i), title, 'PASS' if passed else 'FAIL', detail])
        tbl = ax.table(cellText=rows, loc='center', cellLoc='left',
                       colWidths=[0.04, 0.42, 0.10, 0.44])
        tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1, 1.6)
        for c in range(4):
            tbl[(0, c)].set_facecolor('#dddddd')
        for r, (_, passed, _, _) in enumerate(results, 1):
            tbl[(r, 2)].set_facecolor('#c8e6c9' if passed else '#ffcdd2')
        n_fail = sum(1 for _, p, _, _ in results if not p)
        ax.text(0.5, 0.02,
                f"{len(results) - n_fail}/{len(results)} PASS"
                + ('' if n_fail == 0 else f'  —  {n_fail} FAILED'),
                ha='center', fontsize=11,
                color='green' if n_fail == 0 else 'red', transform=ax.transAxes)
        pdf.savefig(fig); plt.close(fig)

        # One detail page per case
        for i, (title, passed, detail, draw) in enumerate(results, 1):
            fig, ax = plt.subplots(figsize=(8.5, 5))
            draw(ax)
            ax.set_title(f"[{i}] {title}   —   "
                         f"{'PASS' if passed else 'FAIL'}\n{detail}",
                         fontsize=9, color='green' if passed else 'red')
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

    with open(_long(csv_path), 'w', encoding='utf-8') as fh:
        fh.write('case,result,detail\n')
        for title, passed, detail, _ in results:
            fh.write(f'"{title}",{"PASS" if passed else "FAIL"},"{detail}"\n')

    n_fail = sum(1 for _, p, _, _ in results if not p)
    return pdf_path, csv_path, results, n_fail


def _ascii(s: str) -> str:
    """Console-safe text (this project's default console is cp1252)."""
    return s.encode('ascii', 'replace').decode('ascii')


def main():
    pdf_path, csv_path, results, n_fail = build_report()
    for i, (title, passed, detail, _) in enumerate(results, 1):
        print(_ascii(f"[{i}] {'PASS' if passed else 'FAIL'}  {title}  ::  {detail}"))
    print(f"\nReport: {pdf_path}")
    print(f"Summary CSV: {csv_path}")
    print(f"{len(results) - n_fail}/{len(results)} cases PASS")
    return n_fail


if __name__ == '__main__':
    sys.exit(main())
