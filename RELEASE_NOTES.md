# Release Notes

## v0.1.0 - Manuscript Release Candidate

This first release candidate packages the synthetic mechanism benchmark used to evaluate when hierarchical operational state improves AI misuse monitoring relative to flat scoring and simpler aggregate baselines.

### Included

- Synthetic train/validation/test scenario generator.
- Flat rule baselines, aggregate baselines, learned flat baselines, learned hierarchical baselines, and component ablations.
- Validation-tuned thresholds for account review and escalation routing.
- System-alert ranking metrics, calibration metrics, routing/escalation metrics, workload simulation, alert deduplication, backlog simulation, and priority-queue analysis.
- Named configurations for:
  - `configs/ci_smoke_config.json`
  - `configs/smoke_test_config.json`
  - `configs/default_manuscript_config.json`
  - `configs/large_replication_config.json`
- Manuscript-facing tables and figure assets.
- Public safety/scope note, citation file, license, and reproducibility documentation.

### Validation

The release candidate was smoke-tested from its own repository root. The public smoke test completed successfully with exit code `0` and generated the expected output tables and SVG figures.

### Scope

This release is a synthetic benchmark and systems prototype. It is not a production misuse detector, deployment validation study, or substitute for external evaluation on real monitoring data.
