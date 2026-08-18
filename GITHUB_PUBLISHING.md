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

## Publish From Local Repository

After creating an empty GitHub repository, run:

```powershell
git remote add origin https://github.com/<your-github-username>/hierarchical-operational-state-benchmark.git
git push -u origin main
git push origin v0.1.0
```

If the remote already exists:

```powershell
git remote set-url origin https://github.com/<your-github-username>/hierarchical-operational-state-benchmark.git
git push -u origin main
git push origin v0.1.0
```

## Verify After Push

On GitHub:

- Confirm `README.md` renders clearly.
- Confirm the `smoke-test` GitHub Action starts.
- Confirm generated `outputs/` are not committed.
- Confirm manuscript figures and tables are present.
- Confirm `RELEASE_NOTES.md`, `CITATION.cff`, `LICENSE`, and `SECURITY_AND_SAFETY.md` are visible.

## Suggested First Release

Create a GitHub release from tag:

```text
v0.1.0
```

Release title:

```text
v0.1.0 - Manuscript release candidate
```

Release description:

```text
Synthetic benchmark release supporting smoke, manuscript-scale, and large-replication configurations for evaluating flat versus hierarchical operational-state monitoring, escalation routing, and analyst workload.
```

