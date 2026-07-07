# NVH Analyzer

Point-and-click NVH analysis for rotating and reciprocating machinery.
Raw time-series CSV exports (e.g. DEWESOFT) are split into individual
actuations and run through vibration-level, torque, efficiency, order and
envelope analyses — from one dashboard, with no scripting.

Built for the plastic-vs-metal ballscrew recirculation study, but the
analysis modes (test-rig campaign or one continuous recording) apply to any
rotating hardware.

---

## Quick start

```bash
pip install -r requirements-app.txt
streamlit run streamlit_app.py
```

First time here? Click **Create demo workspace** on the overview. It writes a
small synthetic test campaign whose correct answers are known in advance,
runs a full analysis on it immediately, and drops a `WORKSPACE_GUIDE.md`
walkthrough next to the data. Every screen of the tool can then be explored
with data where you know what the right result looks like — the same suite
doubles as a functional check of the whole tool.

## The dashboard

One overview page with three panels; deep dives sit one click behind each:

| Panel | What it does | Deep dive |
|-------|--------------|-----------|
| **Data source** | Folder picker or drag & drop, recognized-file explorer, summary (files / channels / sampling rate), workspace guide | **Setup**: column mapping with auto-detected defaults, specimens & speeds, header-consistency check |
| **Analyze** | One tile per analysis with a plain-English description, run naming, advanced parameters (every field has a hover explanation) | Advanced options expander |
| **Investigate & status** | Live progress with per-stage checklist and abort, run history cards, compare-runs table | **Report**: full interactive results for any run |

A step bar across the top (Import data → Map signals → Run analysis →
Review results) tracks real readiness — each node becomes a green checkmark
only when that step is genuinely complete.

Workspaces persist: the app reopens on the data folder and results folder you
last used, and the sidebar switches between recent projects in one click.

## The analyses

| Tile | Answers |
|------|---------|
| Segmentation | Splits each recording into individual strokes (runs first). |
| Vibration Level | How much does each specimen shake? RMS per stroke, metal vs plastic, trial-level statistics. |
| Torque & Efficiency | What does motion cost? Steady-state drag torque and an efficiency proxy per speed class. |
| Order Analysis | Is the vibration locked to shaft rotation? Amplitude vs order (events per revolution), ball-pass markers, Campbell speed map. |
| Envelope Analysis | Are there repetitive impacts? Kurtogram band selection + envelope demodulation; per-specimen fixed bands. |
| Handover Report | The cross-stage one-pager: per-material characteristics table ready for a report. |

## Headless / CI use

The same pipeline runs from the command line:

```bash
python -m nvh_pipeline --write-example nvh_config.json   # starter config
python -m nvh_pipeline --config nvh_config.json          # run everything
python -m nvh_pipeline --config nvh_config.json --stages segment,order
```

Exit code = number of failed stages. Each run writes `manifest.json` listing
every file produced.

## Engineering-validity controls

The defaults keep steady-state comparisons defensible:

- **Plateau gating** (`plateau_frac = 0.90`): order, torque, efficiency and
  envelope use only the constant-speed portion of each stroke, so ramps
  neither bias torque nor collapse the order ceiling. Strokes without a
  usable plateau fall back to the full stroke (counted in diagnostics).
- **Per-specimen envelope band** (`band_scope = per_part`): a structural
  resonance does not move with speed, so one band per specimen (picked from
  its highest-speed plateau strokes, capped at the accelerometer's calibrated
  range) demodulates all of that specimen's strokes — by-speed spectra stay
  comparable.
- **Energy-correct aggregation** (`order_average = power`,
  `cumulative_mode = rss`): cross-stroke averages are RMS spectra and the
  X+Y+Z total is the vector magnitude, not a linear sum.
- **Diagnostics, not assertions**: per-speed-class CSVs report stroke
  lengths, native order resolution and the effective anti-alias ceiling, so
  resolution limits are auditable. Short strokes (3–5 revolutions) give
  ~0.2–0.33 order resolution — that cannot separate ball-pass 5.35 from
  shaft order 5; only longer constant-speed dwell at acquisition can.
- **Data-quality audit**: requested-but-missing specimen×speed combinations
  and unbalanced extend/retract counts are flagged on every run.

## Testing

```bash
pip install -e . && pip install pytest duckdb altair streamlit
pytest tests/            # unit + DSP validation + golden end-to-end
pytest tests/ -m report  # signable expected-vs-measured validation PDF
```

The validation suite asserts against closed-form math (a tone of amplitude A
must read back A; anti-aliasing must suppress fold-back; envelope
demodulation must recover a known modulation depth), and the golden e2e test
drives the real CLI over a synthetic campaign. CI runs all of it on every
push (`.github/workflows/ci.yml`).

## Windows bundle (no Python install)

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_windows_app.ps1 -Zip
```

produces a self-contained folder with an embedded Python runtime — recipients
double-click `Run NVH Analyzer.bat`. See `packaging/README_PACKAGING.md`.

## Layout

```
EP_*.py                     analysis stages (the proven numerics)
nvh_pipeline/
  config.py                 PipelineConfig + nvh_config.json + per-stage injection
  common.py                 shared numerics (RMS, plateau gating, resampling,
                            dataset discovery, header QA)
  dataset.py                DuckDB-over-parquet layer (column pushdown,
                            single-query reductions, data-quality audit)
  campbell.py               Campbell-diagram building blocks
  runner.py                 stage orchestration (parallel after segmentation),
                            manifest.json, progress events
  demo.py                   the synthetic teaching/validation workspace
  app_helpers.py            pure helpers for the front end
  viz.py                    interactive Altair result views
  __main__.py               CLI  (python -m nvh_pipeline)
streamlit_app.py            the dashboard front end
tests/                      pytest suite incl. exact-math validation cases,
                            golden end-to-end, and the demo-workspace checks
packaging/                  Windows no-install bundle builder + launcher
.streamlit/config.toml      pinned app theme
nvh_config.example.json     starter config for headless use
```

## Data conventions

- Rig recordings are named `<specimen>-<speed>rpm-<trial>.csv`; specimen
  prefix `m…` = metal, `T…` = plastic.
- **Y/Z swap**: on this rig, physical Y is logged under the "Z Accel" column
  and vice versa. The vibration-level stages apply the swap via the column
  mapping; order/envelope sum all axes, so their totals are unaffected.
- Envelope analysis is only meaningful on metal specimens — plastic's damping
  makes the kurtogram latch onto noise. The app repeats this caveat wherever
  plastic envelope data is shown.
