"""
nvh_pipeline — configuration, orchestration and shared numerics for the
ballscrew recirculation NVH analysis pipeline.

The eleven EP_*.py stage scripts at the repo root carry the proven analysis
numerics; this package supplies everything they share:

    config       PipelineConfig + nvh_config.json load/save + per-stage injection
    common       shared helpers (material_type, RMS, plateau gating, resampling,
                 dataset discovery / header inspection)
    dataset      DuckDB-over-parquet access layer (column pushdown, one engine)
    campbell     Campbell-diagram building blocks (angular windows, RPM grid)
    runner       stage orchestration, manifest.json, progress events
    app_helpers  pure helpers for the Streamlit front end
    viz          interactive (Altair) result views for the app

CLI:  python -m nvh_pipeline --config nvh_config.json [--stages a,b,c]
"""

__version__ = "1.0.0"
