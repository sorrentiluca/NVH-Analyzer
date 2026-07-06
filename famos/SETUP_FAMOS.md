# Running the NVH pipeline from imc FAMOS

The pipeline is driven from FAMOS through one stable entry point:
`nvh_pipeline.famos_entry.run_pipeline`.

## 1. Install into the FAMOS Python interpreter

FAMOS's Python Kit uses its own interpreter. Into **that** interpreter:

```bash
pip install -r requirements.txt        # pandas numpy scipy matplotlib pyarrow
pip install duckdb                     # shared SQL engine for the stages
pip install -e .                       # makes `nvh_pipeline` importable
```

(Confirm which interpreter FAMOS uses under *Extras → Options → Python*.)

## 2. Configure

Create/edit `nvh_config.json` (start from `nvh_config.example.json`, or run
`python -m nvh_pipeline --write-example nvh_config.json`). At minimum set
`data_dir` and `output_root`.

## 3. Call from a FAMOS sequence

```python
# Inside a FAMOS Python code block:
from nvh_pipeline.famos_entry import run_pipeline

manifest_path = run_pipeline(r"C:/path/to/nvh_config.json")
# Optional: subset of stages and a data-dir override
# manifest_path = run_pipeline(r"C:/.../nvh_config.json",
#                              stages=["segment", "order"],
#                              data_dir=r"D:/tests/Plastic")
```

`run_pipeline` blocks until the run finishes and returns the absolute path of
`manifest.json`. The manifest lists, per stage, every PNG/CSV produced — the
FAMOS panel walks that list to load and display the results:

```json
{
  "stages": [
    {"stage": "order", "label": "Order Analysis", "status": "ok",
     "files": ["…/output_order/plots/order_cumulative.png", "…"]}
  ]
}
```

Plotting is forced headless (`MPLBACKEND=Agg`, `show_plots=False`) so no
window ever opens inside FAMOS.

## Notes

- Exit/status checking: each stage record carries `status` (`ok` / `error`)
  and, on error, the full traceback in `error`.
- The equivalent CLI (`python -m nvh_pipeline --config … --stages …`) exits
  with the number of failed stages, which suits `ExecuteAndWait`-style calls.
- Long OneDrive paths on Windows can exceed MAX_PATH; keep `output_root`
  short (e.g. `C:\nvh\results`) or enable Windows long-path support.
