# Ballscrew Recirculation NVH Pipeline

Analysis pipeline for the plastic-vs-metal ballscrew recirculation NVH study.
Raw DEWESOFT CSVs are segmented into per-actuation parquet files and then run
through loudness, torque, efficiency, order, and envelope analyses. The whole
thing is driven by **one config file** and can be launched from the command line
or from **imc FAMOS** (see [`famos/SETUP_FAMOS.md`](famos/SETUP_FAMOS.md)).

> New here? Read the handover document `Ballscrew_NVH_Testing.docx` for the
> engineering background (why the study exists, the plastic "envelope" trap in
> §6, and the gotchas in §7). This README is the *how to run it* map.

---

## What changed in the hand-off

The eleven `EP_*.py` analysis scripts are unchanged in their numerics, but they
no longer each carry their own paths and settings. Everything that used to be
edited in code now lives in **`nvh_config.json`**:

- the raw-data folder (no more hardcoded OneDrive path),
- which parts and RPM classes to process,
- column names and the Y/Z accelerometer swap,
- the order/envelope and kurtogram parameters.

A small `nvh_pipeline/` package reads that config, runs the stages in order, and
writes a `manifest.json` listing every PNG/CSV produced — which is what FAMOS
reads back to display the results.

```
nvh_config.json  ──►  python -m nvh_pipeline  ──►  output_*/ + manifest.json
       ▲                                                    │
   FAMOS panel  ◄────────── loads the listed PNG/CSV ───────┘
```

---

## Install

```bash
pip install -r requirements.txt          # pandas numpy scipy matplotlib pyarrow
pip install -e .                          # makes `nvh_pipeline` importable
```

(For FAMOS, install these into the Python interpreter the FAMOS Python Kit uses —
see `famos/SETUP_FAMOS.md`.)

## Configure

```bash
python -m nvh_pipeline --write-example nvh_config.json
```

Then edit `nvh_config.json` — at minimum set `data_dir` (the folder of raw
`<part>-<rpm>rpm-<trial>.csv` files) and `output_root` (where results go). Adding
a new specimen or speed is now just an edit to the `parts` / `rpms` lists.

## Run

```bash
# everything
python -m nvh_pipeline --config nvh_config.json

# only some stages
python -m nvh_pipeline --config nvh_config.json --stages segment,loudness,order

# override the data folder without touching the config
python -m nvh_pipeline --config nvh_config.json --data-dir "D:/tests/Plastic"
```

Stages (run in this dependency order; `segment` must run first):

| stage | script | output folder |
|-------|--------|---------------|
| `segment` | EP_segment.py | `output_seg/` (parquet every stage reads) |
| `loudness` | EP_loudness.py | `output_loudness/` |
| `loudness_plot` | EP_loudness_plot.py | `output_loudness/` (executive figures) |
| `loudness_by_sample` | EP_loudness_by_sample.py | `output_loudness_samples/` |
| `torque` | EP_torque.py | `output_torque/` |
| `efficiency` | EP_efficiency.py | `output_efficiency/` |
| `order` | EP_order_analysis.py | `output_order/` |
| `envelope` | EP_Envelope.py | `output_envelope/` |
| `bandpass` | EP_bandpass.py | `output_envelope/` |
| `sample_summary` | EP_Sample_summary.py | `output_handover/` |
| `actuation_plot` | EP_actuation_plot.py | `output_actuation_plot/` |
| `characteristics` | EP_characteristics.py | `output_handover/` (cross-stage summary) |

All output folders are created under `output_root`.

### Running an individual script the old way

The documented `python EP_segment.py … python EP_Envelope.py` order still works.
Each script picks up `nvh_config.json` automatically — from the `NVH_CONFIG`
environment variable, or `./nvh_config.json` in the current folder, or built-in
defaults. So either of these is fine:

```bash
NVH_CONFIG=/path/to/nvh_config.json python EP_loudness.py
# …or just run from a folder that contains nvh_config.json
python EP_loudness.py
```

---

## Standalone app (no Python install needed)

For colleagues who can't install Python (e.g. locked-down corporate laptops),
there's a point-and-click **Streamlit** front end that wraps the exact same
pipeline, plus a builder that bundles it with an embedded Python runtime into a
self-contained, shareable Windows folder.

You point it at a folder of raw `<part>-<rpm>rpm-<trial>.csv` files using the
**in-app folder browser** (no copy-pasting of paths), and it **auto-detects**
which parts and RPM classes are present. Runs the chosen stages with a per-stage
progress bar, shows the handover summary / loudness tables and the new
characteristics cross-stage summary inline. Data stays **local** — raw DEWESOFT
CSVs are never uploaded through the browser.

The tech stack (pandas + numpy + scipy + DuckDB + Streamlit) is fully pinned in
`requirements-app.txt`. DuckDB is a single self-contained wheel — no server, no
admin — and acts as the shared SQL engine for all nine analysis stages, replacing
redundant per-stage `glob+read_parquet` calls with column-pushdown queries.

```bash
# run the UI in dev
pip install -r requirements-app.txt
streamlit run streamlit_app.py
```

To produce the no-install Windows bundle, run on a Windows machine with internet:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_windows_app.ps1 -Zip
```

The recipient unzips `dist\NVH-Analyzer.zip` and double-clicks
`Run NVH Analyzer.bat` — no admin, no installer, offline. See
[`packaging/README_PACKAGING.md`](packaging/README_PACKAGING.md).

---

## Smoke test (no real data needed)

```bash
python make_testdata.py                                   # writes testdata/*.csv
python -m nvh_pipeline --write-example nvh_config.json
# edit nvh_config.json:  "data_dir": "testdata", "output_root": "results"
python -m nvh_pipeline --config nvh_config.json
```

You should see `PIPELINE DONE — 12/12 stage(s) ok` and a populated `results/`
folder with a `manifest.json`. The synthetic metal part carries impulsive
ball-pass taps so order analysis recovers a peak near order 5.35.

---

## Steady-state validity controls

To keep the steady-state comparisons defensible, the analysis stages gate on the
constant-speed **plateau** of each stroke and constrain envelope band selection:

- **Plateau gating** (`plateau_gating`, `plateau_frac` = 0.90): order, torque,
  efficiency and envelope analyse only samples where `|RPM| ≥ plateau_frac ×
  nominal`, excluding the ramp-up/ramp-down. This removes ramp bias from
  torque/efficiency and stops the order anti-alias cutoff (set from the slowest
  speed in the window) from collapsing the high-RPM order ceiling. Segments with
  no usable plateau fall back to the full stroke (counted in the diagnostics).
- **Per-specimen envelope band** (`band_scope = per_part`): a structural
  resonance is speed-independent, so the kurtogram picks **one** band per part
  from its highest-RPM plateau segments and demodulates every segment of that
  part with it — making the by-speed envelope spectra comparable. Capped at
  `bp_max_hz` (default 12000 Hz, the accelerometer's ±10 % calibrated edge; use
  9000 for ±5 %) and at 0.8 × Nyquist, so bands cannot latch onto the sensor's
  uncalibrated noise region.
- **Order aggregation** (`order_average = power`, `cumulative_mode = rss`): the
  cross-actuation average is the RMS (energy-mean) spectrum and the X+Y+Z
  cumulative is the vector magnitude `√(x²+y²+z²)` rather than a linear sum.
- **Diagnostics**: `output_order/order_diagnostics{,_summary}.csv` report, per
  RPM class / material, the `n_rev` distribution, native order resolution
  (`1/n_rev`), and the effective anti-alias order — so the resolution limits of
  short (3–5 rev) strokes are auditable, not hidden. Per-part bands are recorded
  in `output_envelope/run_metadata.json` and `part_bands.csv`; the representative
  kurtogram grid in `kurtogram_grid.csv`.
- **Diagnostic charts** (Streamlit): the Envelope tab shows each specimen's band
  as an interval plot and a per-specimen **kurtogram heatmap** (a sharp hot cell
  = a robust band; a flat field = noise-latching).
- **Campbell waterfalls** (`order_campbell.csv`, `envelope_campbell.csv`): a
  short-time order analysis in sliding angular windows across each stroke's
  **ramp sweep** (0 → nominal → 0) tags every window by its *measured* RPM,
  giving a continuous order × RPM map with many bins — not just the ~6 nominal
  speed classes. A vertical stripe at a fixed order across speeds is a true
  speed-synchronous order; a ridge that drifts with RPM is a fixed-Hz resonance.
  Tunables: `campbell_window_rev` (order resolution ≈ 1/window), `campbell_rpm_bin`.
  Caveat: 3–5 rev strokes make this an overview, not a high-Q run-up.

> Kurtogram resolution: `kurtogram_levels` is 6 (finer search on high-fs data),
> but `bp_min_bw_hz` is held at 200 Hz — narrower candidate bands drop the
> ball-pass modulation sidebands envelope demod needs and collapse defect
> recovery (golden-test verified). The bandwidth floor also caps the effective
> search depth on low-sample-rate data.

> Resolution caveat: with 3–5 rev strokes the native order resolution is
> ~0.2–0.33 order, which **cannot** cleanly separate BPFO 5.35 from shaft order 5
> regardless of the display grid. Resolving them requires longer constant-speed
> dwell at acquisition — see the diagnostics summary.

## Keep in mind (from the handover doc)

- **Y/Z swap** (`accel_cols`): physical Y is logged under the *Z Accel* column
  and vice versa. The loudness scripts apply the swap; the order/envelope scripts
  auto-detect axis columns by literal name and sum X+Y+Z, so the swap doesn't
  change their totals — but per-axis interpretation differs between the two. §7.
- **1000 Hz kurtogram floor** (`bp_min_hz`) keeps the envelope band above the top
  shaft order (order 25 @ 2300 RPM ≈ 958 Hz). Re-check it if you raise max RPM.
- **Plastic envelope caveat**: envelope analysis is only meaningful on metal;
  plastic's damping means the kurtogram latches onto noise. §6.

## Layout

```
EP_*.py                     analysis stages (the proven numerics)
nvh_pipeline/
  config.py                 PipelineConfig + nvh_config.json load/save + injection
  common.py                 shared helpers (material_type, RMS)
  runner.py                 runs stages, writes manifest.json
  __main__.py               CLI  (python -m nvh_pipeline)
  famos_entry.py            the function FAMOS calls
famos/                      FAMOS setup guide + panel/sequence templates
nvh_config.example.json     starter config
make_testdata.py            synthetic data generator for smoke testing
```
