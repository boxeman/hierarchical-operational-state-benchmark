# Hierarchical Operational-State Monitoring Benchmark

Synthetic benchmark for testing when hierarchical operational state improves AI misuse monitoring, escalation routing, and analyst workload relative to flat scoring and simpler aggregate baselines.

## Research Question

Many monitoring systems reduce activity to a flat score at the message or account level. This benchmark asks when that collapse loses decision-critical information:

> For monitoring problems involving intrinsically hierarchical signals, does preserving operational state across message, account, signature/campaign, and system levels improve detection, calibration, escalation quality, or analyst workload?

The benchmark tests that question with synthetic mechanism scenarios where weak local signals can become meaningful only when evidence, uncertainty, provenance, signature/campaign context, legitimacy context, and route authority are preserved across levels.

## What This Repository Contains

- Synthetic scenario generator with harmful archetypes, benign hard negatives, and adversarial false-systemic lookalikes.
- Flat baselines, aggregate baselines, learned flat baselines, learned hierarchical baselines, and feature ablations.
- Validation-tuned thresholding and escalation routing.
- Account-review metrics, system-alert metrics, calibration curves, routing metrics, workload simulation, alert deduplication, backlog simulation, and priority-queue analysis.
- Named configs for quick smoke tests, manuscript-scale runs, and larger replication runs.
- Manuscript-facing tables and figures.

## Quick Start

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the fast smoke test:

```powershell
python run_benchmark.py --config configs/smoke_test_config.json
```

Run the minimal CI-style test suite:

```powershell
python -m unittest discover -s tests
```

## Expected Runtime

Approximate local runtimes from the release-candidate QA pass:

| Config | Purpose | Expected runtime |
| --- | --- | --- |
| `configs/ci_smoke_config.json` | Minimal automated test config | Seconds to under a minute |
| `configs/smoke_test_config.json` | Public smoke test | About 1 minute |
| `configs/default_manuscript_config.json` | Manuscript-scale run | Several minutes |
| `configs/large_replication_config.json` | Larger held-out replication | Longer optional run |

Runtime depends on hardware and Python environment.

## Main Commands

```powershell
python run_benchmark.py --config configs/smoke_test_config.json
python run_benchmark.py --config configs/default_manuscript_config.json
python run_benchmark.py --config configs/large_replication_config.json
python compare_replication.py
```

Outputs are written under `outputs/`.

## Generated Outputs

Benchmark runs generate tables such as:

- `benchmark_summary.md`
- `summary_results_table.csv`
- `summary_results_table.md`
- `account_metrics_by_split.csv`
- `system_alert_metrics_by_split.csv`
- `calibration_summary.csv`
- `routing_metrics_by_split.csv`
- `time_to_alert_workload_summary.csv`
- `dedup_backlog_summary.csv`

The figure writer generates nine SVG figures:

1. Account-review performance.
2. Targeted component ablations.
3. Calibration and system-alert behavior.
4. Routing and escalation quality.
5. Adversarial false-systemic negatives.
6. Product-investigation frontier.
7. Time-to-alert and analyst workload.
8. Alert deduplication and backlog.
9. Priority-queue behavior.

The manuscript package consolidates a smaller main-figure set plus supplemental assets.

## What This Benchmark Does Not Prove

This repository does not provide a production misuse detector. It does not use real user data, deployment logs, or operational abuse traces. It does not prove that any particular monitoring system is safe.

The benchmark is intended to test mechanisms:

- when hierarchical state helps,
- when flat scoring is enough,
- when system-level alerting becomes fragile,
- when hard benign lookalikes cause false escalation,
- and how operational routing interacts with analyst workload.

External validation on real or institutionally governed datasets would be required before making deployment claims.

## Reproducibility Notes

The release candidate was smoke-tested from its own repository root. The smoke test completed successfully with exit code `0` and generated the expected tables and figures. Details are recorded in `RELEASE_CANDIDATE_QA.md`.

For configuration details, see:

- `EXPERIMENT_CONFIG.md`
- `RUN_EXPERIMENTS.md`
- `METHODS_PROTOCOL.md`

## Manuscript Status

This repository is prepared as a public release candidate for a manuscript currently framed as:

> A synthetic mechanism benchmark for hierarchical operational-state monitoring of AI misuse.

The intended article identity is a methods/benchmark/systems prototype rather than a claim of a new AI paradigm.

## Citation

If you use this benchmark, please cite the manuscript and archived release once available. Repository citation metadata is provided in `CITATION.cff`.

## Safety and Scope

See `SECURITY_AND_SAFETY.md` before reusing or extending this benchmark.
