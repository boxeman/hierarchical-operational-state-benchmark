# GitHub Publishing Checklist

This repository is ready to publish as:

```text
hierarchical-operational-state-benchmark
```

## Recommended GitHub Settings

- Visibility: public when ready, private while doing final review.
- Default branch: `main`.
- Repository description: `Synthetic benchmark for hierarchical operational-state monitoring of AI misuse.`
- Topics:
  - `ai-safety`
  - `misuse-monitoring`
  - `synthetic-benchmark`
  - `hierarchical-modeling`
  - `calibration`
  - `human-review`
  - `workflow-simulation`

## Current Published State

The repository is now public and DOI-archived:

- Repository: https://github.com/boxeman/hierarchical-operational-state-benchmark
- GitHub release `v0.1.1`: https://github.com/boxeman/hierarchical-operational-state-benchmark/releases/tag/v0.1.1
- Zenodo version DOI: https://doi.org/10.5281/zenodo.21988606
- Zenodo concept DOI: `10.5281/zenodo.21988605`

## Publish From Local Repository

After creating an empty GitHub repository, run:

```powershell
git remote add origin https://github.com/<your-github-username>/hierarchical-operational-state-benchmark.git
git push -u origin main
git push origin v0.1.1
```

If the remote already exists:

```powershell
git remote set-url origin https://github.com/<your-github-username>/hierarchical-operational-state-benchmark.git
git push -u origin main
git push origin v0.1.1
```

## Verify After Push

On GitHub:

- Confirm `README.md` renders clearly.
- Confirm the `smoke-test` GitHub Action starts.
- Confirm generated `outputs/` are not committed.
- Confirm manuscript figures and tables are present.
- Confirm `RELEASE_NOTES.md`, `CITATION.cff`, `LICENSE`, and `SECURITY_AND_SAFETY.md` are visible.

## DOI Archive Release

The DOI archive release is:

```text
v0.1.1
```

Release title:

```text
v0.1.1 - DOI archive release
```

Release description:

```text
DOI archive release for the manuscript-facing benchmark package.

This release preserves the synthetic benchmark for hierarchical operational-state monitoring of AI misuse, including named configs, smoke-test workflow, manuscript-facing documentation, citation metadata, and data/code availability materials.

All data are synthetic. This is not a production misuse detector or deployment validation study.
```
