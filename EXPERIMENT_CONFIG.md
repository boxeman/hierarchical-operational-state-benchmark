# Experiment Configuration

The benchmark can be run from an explicit JSON configuration file. The named
configs live in `configs/`.

```powershell
python run_benchmark.py --config configs/default_manuscript_config.json
```

Running without `--config` uses the same built-in defaults:

```powershell
python run_benchmark.py
```

The resolved configuration for each run is copied to:

```text
outputs/prototype_package/run_config.json
```

If a config file is supplied, the original input file is also copied to:

```text
outputs/prototype_package/input_config.json
```

## Named Configs

`configs/smoke_test_config.json` is a small end-to-end pipeline check. It uses
few scenarios and very small bootstrap counts, writes to `outputs/smoke_test/`,
and should be used before longer runs.

`configs/default_manuscript_config.json` reproduces the current manuscript run.
It writes to `outputs/prototype_package/`.

`configs/large_replication_config.json` increases train/validation/test scenario
counts and bootstrap counts. It writes to `outputs/large_replication/` and is
intended to address the manuscript limitation that the current held-out test set
is small.

See `RUN_EXPERIMENTS.md` for copy-paste commands.

## Configuration Blocks

`experiment_name` names the run for manuscript and provenance tracking.

`output_dir` controls where all tables, figures, copied configs, and summaries are written.

`splits` controls train, validation, and held-out test scenario counts and seed offsets.

`scenario_sweep` controls the synthetic phase-space settings:

- systemic exploit account counts
- systemic message-score means
- benign shared-topic score means
- cluster-size normalization values
- sub-threshold evidence thresholds

`evaluation` controls review budgets, the manuscript-facing review budget, and the validation precision constraint used for threshold tuning.

`routing` controls the adversarial false-alert penalty sweep used to generate the product-investigation operating frontier.

`bootstrap` controls bootstrap replicate counts and random seeds for:

- account-review intervals
- system-alert intervals
- targeted ablation intervals
- routing intervals
- product-investigation frontier intervals
- workload/time-to-alert intervals
- deduplicated backlog intervals

`workload` controls the practical simulation:

- monitoring horizon
- daily analyst capacity
- route cost assumptions for `auto_handle`, `human_review`, `policy_review`, and `product_investigation`

`backlog` controls finite-capacity queue simulation:

- analyst capacity values
- FIFO/product-first/product-plus-policy-first/severity-weighted queue policies

`methods` controls the method subsets used for practical workload, backlog bootstrap, and system-alert bootstrap summaries.

## Reviewer-Reproducibility Rationale

This layer separates manuscript settings from benchmark code. A reviewer can inspect `experiment_config.json`, rerun the benchmark, and verify that the output package includes the exact resolved run settings in `run_config.json`.

Future variants can be created without editing code, for example:

- `configs/baseline_config.json`
- `configs/stress_test_config.json`
- `configs/sensitivity_sweep_config.json`

The recommended next replication step is a larger held-out scenario run that increases `test_scenarios` and bootstrap replicate counts.

