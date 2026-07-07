"""
demo — the starter workspace: a synthetic dataset suite with KNOWN answers.

``create_demo_workspace(root)`` writes a small measurement campaign whose
content is constructed mathematically, so every analysis the tool offers has
a known, teachable outcome:

    m101  metal, healthy      broadband noise only — the quiet baseline.
    m202  metal, worn         impacts at 5.35 per revolution ringing a
                              2 500 Hz resonance — the defect signature the
                              order and envelope analyses must find.
    m303  metal, reference    a pure tone at order 3.0 with amplitude 0.35 —
                              the order spectrum must read back 0.35 there,
                              proving the amplitude axis is calibrated.
    T404  plastic             louder broadband noise, higher drag torque,
                              recorded at ONE speed only — teaches the
                              loudness ratio, the torque comparison, the
                              plastic envelope caveat AND triggers the
                              data-quality "missing speed" flag.

Plus ``continuous/fan_rig.csv``: one steady single-channel recording with a
tone at order 4.5, for the continuous (non-reciprocating) mode.

A WORKSPACE_GUIDE.md is written alongside the data explaining, per dataset,
what to look at and what the correct answer is — the same file doubles as the
onboarding walkthrough inside the app.
"""

from __future__ import annotations

import os

import numpy as np

FS = 10_000.0          # sample rate of the synthetic recordings (Hz)
BPF_ORDER = 5.35       # ball-pass order of the simulated defect
CAL_ORDER = 3.0        # calibration-tone order (m303)
CAL_AMP = 0.35         # calibration-tone amplitude (m/s²)
RESONANCE_HZ = 2500.0  # structural resonance the defect impacts ring

COLS = ["Time (s)", "CNT 1/Frequency (RPM)", "CNT 1/Angle (Degrees)",
        "Torque (Nm)", "X Accel (m/s2)", "Y Accel (m/s2)", "Z Accel (m/s2)"]

SPEEDS = (800, 2300)
PARTS = ("m101", "m202", "m303", "T404")


def _ramp_profile(rpm: float, revs: float, fs: float) -> np.ndarray:
    """One stroke's speed profile: ramp up, hold ``revs`` revolutions, ramp
    down.  The ramps are what the Campbell map sweeps across."""
    plateau_t = revs / (rpm / 60.0)
    ramp_t = max(0.05, plateau_t * 0.25)
    dt = 1.0 / fs
    up = np.arange(0, ramp_t, dt)
    pl = np.arange(0, plateau_t, dt)
    return np.concatenate([rpm * (up / ramp_t),
                           np.full_like(pl, rpm),
                           rpm * (1 - up / ramp_t)])


def _rig_file(path: str, rpm: float, *, kind: str, seed: int,
              revs: float = 8.0, fs: float = FS) -> None:
    """One rig recording: two actuations (extend then retract) with gaps.

    ``kind``: 'healthy' | 'defect' | 'calibration' | 'plastic'.
    """
    dt = 1.0 / fs
    gap = np.zeros(int(0.3 * fs))
    stroke = _ramp_profile(rpm, revs, fs)
    rpm_sig = np.concatenate([gap, stroke, gap, -stroke, gap])
    n = len(rpm_sig)
    t = np.arange(n) * dt
    angle = np.cumsum(rpm_sig / 60.0 * 360.0 * dt)
    revs_cum = np.cumsum(np.abs(rpm_sig) / 60.0 * dt)
    moving = (np.abs(rpm_sig) > 1.0).astype(float)
    rng = np.random.default_rng(seed)

    # Drag torque: plastic runs ~40 % higher drag than metal.
    drag = 1.15 if kind == "plastic" else 0.8
    torque = (np.where(np.abs(rpm_sig) > 1.0, drag, 0.0)
              * np.sign(rpm_sig + 1e-9) + rng.normal(0, 0.01, n))

    base = rng.normal(0, 0.02, (3, n))
    if kind == "defect":
        # A tap every 1/5.35 of a revolution, each ringing the resonance.
        taps = np.zeros(n)
        taps[np.where(np.diff(np.floor(revs_cum * BPF_ORDER)) > 0)[0]] = 1.0
        tt = np.arange(int(0.004 * fs)) * dt
        ring = np.sin(2 * np.pi * RESONANCE_HZ * tt) * np.exp(-tt / 0.0008)
        imp = np.convolve(taps, ring, mode="same")[:n]
        base[0] += 0.6 * imp * moving
        base[1] += 0.4 * imp * moving
        base[2] += 0.3 * imp * moving
    elif kind == "calibration":
        # Known-amplitude tone locked to shaft angle: the order spectrum must
        # read back exactly CAL_AMP at CAL_ORDER.
        base[0] += CAL_AMP * np.sin(2 * np.pi * CAL_ORDER * revs_cum) * moving
    elif kind == "plastic":
        # Clearly louder, structureless shaking (≈2× the metal group even
        # with m202's impacts included) — and damping means no clean impacts.
        base += rng.normal(0, 0.13, (3, n)) * moving

    data = np.column_stack([t, rpm_sig, angle, torque,
                            base[0], base[1], base[2]])
    np.savetxt(path, data, delimiter=",", header=",".join(COLS),
               comments="", fmt="%.6g")


def _continuous_file(path: str, *, rpm: float = 1500.0, order: float = 4.5,
                     duration_s: float = 6.0, seed: int = 9,
                     fs: float = FS) -> None:
    """A steady rotating-machine recording: one vibration channel, slight
    speed wander, and a tone at ``order`` — for the continuous mode."""
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    wander = 1.0 + 0.01 * np.sin(2 * np.pi * 0.4 * t)
    rpm_sig = rpm * wander
    revs = np.cumsum(rpm_sig / 60.0) / fs
    vib = (0.5 * np.sin(2 * np.pi * order * revs)
           + rng.normal(0, 0.05, n))
    data = np.column_stack([t, rpm_sig, vib])
    np.savetxt(path, data, delimiter=",",
               header="Time (s),Speed (RPM),Vibration (g)",
               comments="", fmt="%.6g")


_GUIDE = f"""# Demo workspace guide

This folder contains a small synthetic test campaign. Every signal in it was
built mathematically, so the correct answer for each analysis is known in
advance. Use it to learn the tool — and to check that every analysis still
gives the right answer.

## The specimens

| Specimen | Material | What was put into the signal | What you should see |
|----------|----------|------------------------------|---------------------|
| m101 | metal | Background noise only | The quiet baseline. No peaks in Order or Envelope. |
| m202 | metal | {BPF_ORDER} impacts per shaft revolution, each ringing a {RESONANCE_HZ:.0f} Hz resonance | **Order Analysis**: peaks at order {BPF_ORDER} and its multiples (10.7, 16.05, 21.4) — the strongest is usually a higher multiple because the impacts are sharp. **Envelope Analysis**: its dominant line at {BPF_ORDER}, and the detected band overlapping {RESONANCE_HZ:.0f} Hz. |
| m303 | metal | A clean tone at order {CAL_ORDER:.0f} with amplitude {CAL_AMP} m/s² | **Order Analysis**: the peak at order {CAL_ORDER:.0f} reads ≈ {CAL_AMP} (within ~15 %; window and averaging cost a few %) — proof the amplitude axis is calibrated. The exact-math gate lives in the automated validation report. |
| T404 | plastic | Louder background noise and ~40 % higher drag torque; recorded at 2300 RPM only | **Vibration Level**: plastic sits above metal. **Torque**: higher drag. **Envelope**: no stable line — plastic damping suppresses impacts (do not read plastic envelope lines as defects). **Data Quality**: flags that T404 has no 800 RPM data. |

Each rig file is named `<specimen>-<speed>rpm-<trial>.csv` and holds two
actuations: one extend stroke, one retract stroke.

## The continuous recording

`continuous/fan_rig.csv` is one steady single-channel recording (≈1500 RPM
with slight speed wander) carrying a tone at 4.5 events per revolution.
Switch the app to *Single continuous signal* mode, point it at this file, and
the order spectrum should show one clean peak at order 4.5.

## Suggested walkthrough

1. **Run everything.** All six analyses on all specimens at both speeds.
2. **Data Quality** — note the flag: T404 was requested at 800 RPM but has
   no data. That is how a coverage gap in a real campaign looks.
3. **Vibration Level** — plastic (T404) above the metals; m101/m202/m303
   near each other. The ratio annotation uses trial means, not stroke counts.
4. **Order Analysis** — m202 shows the {BPF_ORDER}-order family; m303 shows
   the {CAL_ORDER:.0f}-order tone at amplitude ≈ {CAL_AMP}. Use "Split the
   spectrum → By specimen" to isolate each one.
5. **Envelope Analysis** — the m202 band sits near {RESONANCE_HZ:.0f} Hz
   with a sharp kurtogram cell; T404's kurtogram is a flat field (noise).
6. **Torque & Efficiency** — T404's drag is visibly higher at 2300 RPM.
7. **Handover Report** — the one-page summary table collects all of the
   above per material.

If any of these answers ever changes after a software update, treat it as a
regression and investigate before trusting new results.
"""


def create_demo_workspace(root: str) -> dict:
    """Write the demo campaign under ``root`` and return its layout.

    Returns ``{root, data_dir, results_dir, continuous_file, guide, files}``.
    Existing files are overwritten so the demo can always be regenerated to a
    known-good state.
    """
    root = os.path.abspath(root)
    data_dir = os.path.join(root, "data")
    results_dir = os.path.join(root, "results")
    cont_dir = os.path.join(root, "continuous")
    for d in (data_dir, results_dir, cont_dir):
        os.makedirs(d, exist_ok=True)

    files = []
    seed = 0
    for part, kind, speeds, trials in (
            ("m101", "healthy", SPEEDS, (1, 2)),
            ("m202", "defect", SPEEDS, (1, 2)),
            ("m303", "calibration", SPEEDS, (1,)),
            # One speed only — deliberately, to teach the coverage flag.
            ("T404", "plastic", (2300,), (1, 2))):
        for rpm in speeds:
            for trial in trials:
                seed += 1
                revs = 12.0 if kind == "calibration" else 8.0
                path = os.path.join(data_dir, f"{part}-{rpm}rpm-{trial}.csv")
                _rig_file(path, rpm, kind=kind, seed=seed, revs=revs)
                files.append(path)

    cont = os.path.join(cont_dir, "fan_rig.csv")
    _continuous_file(cont)
    files.append(cont)

    guide = os.path.join(root, "WORKSPACE_GUIDE.md")
    with open(guide, "w", encoding="utf-8") as fh:
        fh.write(_GUIDE)

    return {"root": root, "data_dir": data_dir, "results_dir": results_dir,
            "continuous_file": cont, "guide": guide, "files": files}
