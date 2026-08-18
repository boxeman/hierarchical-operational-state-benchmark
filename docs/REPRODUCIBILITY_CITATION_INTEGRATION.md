# Reproducibility and Citation Integration

## Current Public Artifact

Code and synthetic benchmark materials are available at:

https://github.com/boxeman/hierarchical-operational-state-benchmark

The manuscript release candidate is archived as GitHub release `v0.1.0`:

https://github.com/boxeman/hierarchical-operational-state-benchmark/releases/tag/v0.1.0

## Manuscript Integration

The manuscript now treats the GitHub repository and tagged release as the
current reproducibility artifact. The Data and Code Availability section states
that all data are synthetic, identifies the public repository, points to the
`v0.1.0` release, and names the benchmark components included in the code.

The Methods section includes a `Software and Reproducibility` subsection that
identifies the rerunnable configurations:

- `configs/smoke_test_config.json`
- `configs/default_manuscript_config.json`
- `configs/large_replication_config.json`

It also identifies `run_benchmark.py` as the executable entry point,
`RUN_EXPERIMENTS.md` as the run-command guide, and `EXPERIMENT_CONFIG.md` as the
configuration-format guide.

## DOI Status

A DOI-backed archive has not yet been assigned. If the repository is deposited
through Zenodo or another archival service, update:

- `MANUSCRIPT_DRAFT.md`
- `DATA_CODE_AVAILABILITY_STATEMENT.md`
- `COVER_LETTER_DRAFT.md`
- `CITATION.cff`
- lean submission bundle copies
- reviewer package copies

## Current Citation Language

Use the following text until a DOI exists:

> Code and synthetic benchmark materials are available at
> https://github.com/boxeman/hierarchical-operational-state-benchmark. The
> manuscript release candidate is archived as GitHub release `v0.1.0` at
> https://github.com/boxeman/hierarchical-operational-state-benchmark/releases/tag/v0.1.0.

