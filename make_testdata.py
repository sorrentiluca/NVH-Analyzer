"""Generate small synthetic DEWESOFT-style CSVs to smoke-test the pipeline.

Files: <part>-<rpm>rpm-<trial>.csv with the columns the pipeline relies on.
Each file has two actuations (POS then NEG), each spanning a few shaft revs.
Metal parts get impulsive ball-pass taps (order 5.35); plastic gets broadband
noise only — so envelope/order have realistic-ish content and nothing crashes.
"""
import os
import numpy as np

FS = 10000.0
OUT = 'testdata'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

COLS = ['Time (s)', 'CNT 1/Frequency (RPM)', 'CNT 1/Angle (Degrees)',
        'Torque (Nm)', 'X Accel (m/s2)', 'Y Accel (m/s2)', 'Z Accel (m/s2)']


def ramp_profile(rpm, revs=3.0):
    """RPM vs time for one actuation: ramp up, plateau (revs), ramp down."""
    plateau_t = revs / (rpm / 60.0)
    ramp_t = max(0.05, plateau_t * 0.25)
    dt = 1.0 / FS
    t_up = np.arange(0, ramp_t, dt)
    t_pl = np.arange(0, plateau_t, dt)
    t_dn = np.arange(0, ramp_t, dt)
    r_up = rpm * (t_up / ramp_t)
    r_pl = np.full_like(t_pl, rpm)
    r_dn = rpm * (1 - t_dn / ramp_t)
    return np.concatenate([r_up, r_pl, r_dn])


def build_file(part, rpm, metal):
    dt = 1.0 / FS
    gap = np.zeros(int(0.3 * FS))               # zero-RPM rest between actuations
    seg_pos = ramp_profile(rpm)
    seg_neg = -ramp_profile(rpm)
    rpm_sig = np.concatenate([gap, seg_pos, gap, seg_neg, gap])
    n = len(rpm_sig)
    t = np.arange(n) * dt

    # encoder angle = integral of rpm (deg).  rpm/60 rev/s * 360 deg/rev
    angle = np.cumsum(rpm_sig / 60.0 * 360.0 * dt)

    # revolutions completed, for phase of ball-pass taps
    revs = np.cumsum(np.abs(rpm_sig) / 60.0 * dt)

    torque = np.where(np.abs(rpm_sig) > 1.0, 0.8, 0.0) * np.sign(rpm_sig + 1e-9)
    torque += rng.normal(0, 0.01, n)

    base = rng.normal(0, 0.02, (3, n))          # broadband floor on 3 axes
    if metal:
        # impulsive taps at order 5.35: a tap each time revs crosses k/5.35
        bpf_phase = revs * 5.35
        taps = np.zeros(n)
        crossings = np.where(np.diff(np.floor(bpf_phase)) > 0)[0]
        for c in crossings:
            taps[c] = 1.0
        # ring each tap with a damped high-freq resonance (~2500 Hz)
        tt = np.arange(int(0.004 * FS)) * dt
        ring = np.sin(2 * np.pi * 2500 * tt) * np.exp(-tt / 0.0008)
        imp = np.convolve(taps, ring, mode='same')
        moving = (np.abs(rpm_sig) > 1.0).astype(float)
        base[0] += 0.6 * imp * moving
        base[1] += 0.4 * imp * moving
        base[2] += 0.3 * imp * moving
    else:
        # plastic: just a bit more broadband energy while moving, no impulses
        moving = (np.abs(rpm_sig) > 1.0).astype(float)
        base += rng.normal(0, 0.05, (3, n)) * moving

    data = np.column_stack([t, rpm_sig, angle, torque, base[0], base[1], base[2]])
    header = ','.join(COLS)
    path = os.path.join(OUT, f'{part}-{rpm}rpm-1.csv')
    np.savetxt(path, data, delimiter=',', header=header, comments='', fmt='%.6g')
    print(f'wrote {path}  ({n} rows)')


for part, metal in [('m627', True), ('T8', False)]:
    for rpm in (100, 2300):
        build_file(part, rpm, metal)
print('done')
