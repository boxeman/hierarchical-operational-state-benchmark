# Release Candidate QA

Date: 2026-08-17

## Purpose

This note records the first public-release-candidate smoke test for the AI-MLA Studio hierarchical operational-state monitoring benchmark. The goal was to verify that a clean release folder can run the benchmark from its own root using the public-facing smoke-test configuration.

## Release Candidate

Release candidate folder:

```text
release_candidate/
```

Builder:

```text
tools/create_release_candidate.ps1
```

The builder creates a public-facing folder with source code, named configs, methods/release documentation, selected manuscript tables, selected figures, a README, license, citation file, requirements file, safety note, and gitignore.

## Smoke-Test Command

The smoke test was run from inside the release-candidate folder:

```powershell
cd release_candidate
python run_benchmark.py --config configs/smoke_test_config.json
```

Local execution used the bundled Codex Python runtime, but the release-facing command above is the intended public command.
The final QA run set `PYTHONDONTWRITEBYTECODE=1` so the smoke test would not leave Python cache files in the release candidate.

## Result

Status: passed.

Observed exit code: `0`.

Observed runtime: approximately `50` seconds.

The run completed all benchmark stages:

- Train/validation/test scenario generation.
- Model training.
- Account-review and system-alert evaluation.
- Routing-policy tuning.
- Product-investigation frontier evaluation.
- Scenario-bootstrap confidence intervals.
- Time-to-alert and analyst workload simulation.
- Alert deduplication and finite-capacity backlog simulation.
- Calibration and targeted ablation reporting.
- Output and figure writing.

## Generated Smoke-Test Artifacts

The smoke test wrote outputs under:

```text
release_candidate/outputs/smoke_test/
```

Observed artifact shape:

- `49` smoke-test output files.
- `9` generated SVG manuscript figures.
- Core outputs included `run_config.json`, `input_config.json`, `benchmark_summary.md`, `summary_results_table.csv`, `summary_results_table.md`, routing metrics, calibration tables, bootstrap interval tables, workload/backlog tables, and figure SVGs.

## Portability Fix Found During QA

The release-candidate smoke test exposed one portability issue: JSON config loading failed on a UTF-8 BOM produced by Windows PowerShell file writing. The benchmark runner was updated to load config files with `utf-8-sig`, which accepts both ordinary UTF-8 and UTF-8-with-BOM JSON files.

Patched file:

```text
run_benchmark.py
```

Relevant change:

```python
path.open("r", encoding="utf-8-sig")
```

## Hygiene Checks

Pre-smoke and post-smoke release build checks found no intentionally bundled Python bytecode, cache directories, local Codex metadata folders, ZIP files, or local absolute paths in source and documentation files.

The smoke test writes local outputs under `outputs/smoke_test/`. These are ignored by `.gitignore` and should not be committed as source artifacts unless a deliberate example-output folder is prepared.

Public-release hygiene targets:

- No `__pycache__/` directories in committed source.
- No `.pyc` or `.pyo` files.
- No `.agents/` or `.codex/` directories.
- No development ZIP archives.
- No local absolute paths in public-facing documentation.

## Conclusion

The release candidate is functionally runnable from its own root using the smoke-test configuration. The most important public-release readiness result is that the benchmark can now be regenerated from a clean public-facing folder, and the portability issue found during QA has been fixed in source.
