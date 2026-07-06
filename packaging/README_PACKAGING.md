# Packaging the NVH Analyzer as a standalone Windows app

This produces a **self-contained Streamlit app with a bundled Python runtime** —
no install, no admin rights, no internet on the target machine. Built for
sharing on locked-down corporate laptops.

```
NVH-Analyzer/                  ← the folder you share / zip
├─ Run NVH Analyzer.bat        ← end user double-clicks this
├─ runtime/                    ← embedded CPython + all wheels (numpy, scipy,
│                                pandas, matplotlib, pyarrow, streamlit, duckdb)
├─ app/                        ← streamlit_app.py, nvh_pipeline/, EP_*.py
└─ .streamlit/config.toml      ← telemetry off, localhost only
```

## Build it (once, on a Windows box with internet)

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_windows_app.ps1 -Zip
```

- Downloads the official Python *embeddable* runtime and pins every dependency
  into it (from `requirements-app.txt`), then copies the app in.
- `-Zip` also produces `dist\NVH-Analyzer.zip` — the single artifact to hand out.
- Override the Python version with `-PyVersion 3.11.9` or the location with
  `-OutDir`.

> Must be built **on Windows** — the bundle contains Windows binary wheels. The
> *build* needs internet; the *distributed* app does not.

## What the recipient does

1. Unzip `NVH-Analyzer.zip` anywhere (Desktop, USB, network share).
2. Double-click **`Run NVH Analyzer.bat`**.
3. A browser tab opens at `http://localhost:8501`. Use the **📁 Browse** expander
   beside each path field to navigate to your raw CSV folder and output folder
   without copy-pasting paths. The app **auto-detects** the parts and RPM classes
   present, shows a per-part file count, then pick stages and press **Run**. The
   CSVs are read **locally** from the path you give; nothing is uploaded.
4. Closing the black console window stops the app.

Nothing is installed, nothing touches the registry, and the app never leaves
`localhost` — it does not open any network port to the outside.

> **"Bundled runtime not found at …\packaging\runtime\python.exe"** means you ran
> `Run NVH Analyzer.bat` from the **source checkout**, not from a built bundle.
> The `runtime\` folder only exists inside `dist\NVH-Analyzer\` after the build
> script runs. Either build the bundle (above), or for dev just run the app
> directly: `pip install -r requirements.txt streamlit && streamlit run streamlit_app.py`.

## Notes for corporate environments

- **SmartScreen / antivirus**: a `.bat` that launches a bundled `python.exe` can
  trip Application Control (WDAC/AppLocker) policies that block unsigned EXEs from
  user-writable paths. If `python.exe` is blocked, IT must whitelist the folder,
  or you ship it under a managed/allowed path. Test on one managed laptop first.
- **Size**: the bundle is ~300–500 MB (scipy + matplotlib dominate). That is the
  cost of "everything included."
- **Updating the app**: re-run the build script and re-share, or for code-only
  changes just replace the files under `app\` — the `runtime\` folder is stable.
- **Streamlit telemetry** is disabled in `.streamlit/config.toml`
  (`gatherUsageStats = false`) and the server binds to `127.0.0.1` only.

## Alternatives considered

| Approach | Verdict |
|----------|---------|
| **Embeddable Python + launcher (this)** | Most robust for Streamlit; everything included; no admin. ✅ |
| PyInstaller one-file `.exe` | Streamlit's dynamic imports / metadata make frozen builds fragile and slow to start. |
| Hosted Streamlit (intranet server) | Best UX if you can host it, but then it's not "standalone / offline". |
