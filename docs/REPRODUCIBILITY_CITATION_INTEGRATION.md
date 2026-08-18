# Reproducibility and Citation Integration

## Current Public Artifact

Code and synthetic benchmark materials are available at:

https://github.com/boxeman/hierarchical-operational-state-benchmark

The DOI-archived release is available at:

https://doi.org/10.5281/zenodo.21988606

This DOI corresponds to GitHub release `v0.1.1`:

https://github.com/boxeman/hierarchical-operational-state-benchmark/releases/tag/v0.1.1

## Manuscript Integration

The manuscript now treats the GitHub repository and Zenodo DOI as the current
reproducibility artifact. The Data and Code Availability section states that all
data are synthetic, identifies the public repository, points to the DOI-archived
`v0.1.1` release, and names the benchmark components included in the code.

The Methods section includes a `Software and Reproducibility` subsection that
identifies the rerunnable configurations:

- `configs/smoke_test_config.json`
- `configs/default_manuscript_config.json`
- `configs/large_replication_config.json`

It also identifies `run_benchmark.py` as the executable entry point,
`RUN_EXPERIMENTS.md` as the run-command guide, and `EXPERIMENT_CONFIG.md` as the
configuration-format guide.

## DOI Status

Zenodo has assigned the version DOI `10.5281/zenodo.21988606` for the
DOI-archived release. The concept DOI for all versions is
`10.5281/zenodo.21988605`.

The following materials have been updated or should be kept synchronized with
the version DOI:

- `MANUSCRIPT_DRAFT.md`
- `DATA_CODE_AVAILABILITY_STATEMENT.md`
- `COVER_LETTER_DRAFT.md`
- `CITATION.cff`
- lean submission bundle copies
- reviewer package copies

## Current Citation Language

Use the following text for manuscript and submission materials:

> Code and synthetic benchmark materials are available at
> https://github.com/boxeman/hierarchical-operational-state-benchmark. The
> DOI-archived release is available at
> https://doi.org/10.5281/zenodo.21988606, corresponding to GitHub release
> `v0.1.1` at
> https://github.com/boxeman/hierarchical-operational-state-benchmark/releases/tag/v0.1.1.
