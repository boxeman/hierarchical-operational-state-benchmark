# Running Named Experiments

Use the bundled Python environment so the required scientific packages are available:

```powershell
cd "<repository-root>"
```

## Smoke Test

Fast end-to-end pipeline check. This writes to `outputs/smoke_test/`.

```powershell
python run_benchmark.py --config configs/smoke_test_config.json
```

Expected runtime: usually under a few minutes. Use this before changing code or configuration.

## Default Manuscript Run

Current manuscript-facing run. This writes to `outputs/prototype_package/`.

```powershell
python run_benchmark.py --config configs/default_manuscript_config.json
```

Expected runtime: roughly 4-6 minutes on the current machine.

## Large Replication Run

Larger stress/replication run with more held-out scenarios and wider bootstrap checks. This writes to `outputs/large_replication/`.

```powershell
python run_benchmark.py --config configs/large_replication_config.json
```

Expected runtime: potentially 15-45+ minutes. Run this only when you are ready to let the machine work for a while.

## Compare Default vs Large Replication

After the default manuscript run and large replication run both finish, generate
the replication comparison table:

```powershell
python compare_replication.py
```

This writes:

- `outputs/replication_comparison_table.csv`
- `outputs/replication_comparison_table.md`

Expected runtime: usually under one minute.

## Output Provenance

Every run writes:

- `run_config.json`: resolved configuration used by the run
- `input_config.json`: copied input config
- `benchmark_summary.md`: full textual output summary
- `summary_results_table.csv` and `.md`: compact manuscript table
- `figures/`: manuscript SVG figures

