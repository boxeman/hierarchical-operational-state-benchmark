# Public Code and Reproducibility Release Plan

Date: 2026-08-17

Project identity:

**A synthetic mechanism benchmark for hierarchical operational-state monitoring of AI misuse**

## Purpose

This plan translates the manuscript and reviewer/audit package into a public, citable, reproducible benchmark release. The goal is not to publish every internal artifact, but to provide a clean repository that lets readers regenerate the synthetic benchmark, run named configurations, inspect expected outputs, and understand the safety limits of the prototype.

Recommended release route:

1. Clean GitHub repository.
2. Tagged release.
3. Zenodo archive/DOI.
4. Updated `DATA_CODE_AVAILABILITY_STATEMENT.md` with repository and DOI links.

## Public Repository Identity

Recommended repository name:

`hierarchical-operational-state-benchmark`

Acceptable alternatives:

- `ai-mla-studio-benchmark`
- `hierarchical-ai-misuse-monitoring-benchmark`
- `operational-state-monitoring-benchmark`

Recommended short description:

> Synthetic benchmark for testing when hierarchical operational state improves AI misuse monitoring, escalation routing, and analyst workload.

Recommended topic tags:

- `ai-safety`
- `misuse-monitoring`
- `synthetic-benchmark`
- `hierarchical-modeling`
- `calibration`
- `provenance`
- `human-review`
- `workflow-simulation`

## Proposed Repository Structure

```text
hierarchical-operational-state-benchmark/
  README.md
  LICENSE
  CITATION.cff
  pyproject.toml
  requirements.txt
  .gitignore
  src/
    ai_mla_monitor/
      __init__.py
      scenarios.py
      features.py
      baselines.py
      routing.py
      metrics.py
      bootstrap.py
      workload.py
      figures.py
  scripts/
    run_benchmark.py
    compare_replication.py
    convert_submission_figures.py
    qa_converted_figures.py
  configs/
    smoke_test_config.json
    default_manuscript_config.json
    large_replication_config.json
  docs/
    METHODS_PROTOCOL.md
    EXPERIMENT_CONFIG.md
    RUN_EXPERIMENTS.md
    PATTERNS_METHODS_PACKAGE.md
    OVERCLAIM_CONTROL_CHECKLIST.md
    PUBLIC_RELEASE_PLAN.md
  manuscript/
    MANUSCRIPT_DRAFT.md
    DATA_CODE_AVAILABILITY_STATEMENT.md
    figures/
      main_svg/
      submission_pdf/
    tables/
      summary_results_table.md
      replication_comparison_table.md
      methods_components_table.md
  outputs/
    example_smoke_test/
      README.md
      run_config.json
      summary_results_table.md
      selected_metrics.csv
  tests/
    test_smoke_config.py
    test_scenario_generation.py
    test_metric_shapes.py
  SECURITY_AND_SAFETY.md
```

Current local files can map into this structure without changing the manuscript:

- `ai_mla_monitor/` -> `src/ai_mla_monitor/`
- `run_benchmark.py` -> `scripts/run_benchmark.py`
- `compare_replication.py` -> `scripts/compare_replication.py`
- `configs/` -> `configs/`
- `METHODS_PROTOCOL.md`, `EXPERIMENT_CONFIG.md`, `RUN_EXPERIMENTS.md` -> `docs/`
- lean manuscript figures/tables -> `manuscript/`

## Include in Public Release

Include:

- Synthetic scenario generator.
- Flat, aggregate, learned flat, learned hierarchical, and ablation baselines.
- Validation-tuned threshold and routing-policy code.
- System-alert, calibration, routing, workload, backlog, and priority-queue metrics.
- Scenario-bootstrap procedures.
- Named configs:
  - `smoke_test_config.json`
  - `default_manuscript_config.json`
  - `large_replication_config.json`
- Run instructions for smoke, default, and large replication runs.
- Figure-generation and figure-format conversion scripts.
- Compact manuscript-facing tables.
- Example smoke-test outputs.
- Methods/protocol documentation.
- Synthetic-data statement.
- Safety/use-limits note.
- License.
- Citation file.

## Exclude from Public Release

Exclude:

- Raw development notes.
- Chat logs or model-conversation transcripts.
- Full intermediate output dumps unless intentionally archived as supplemental data.
- Local absolute paths.
- Temporary files, caches, `__pycache__`, `.agents`, `.codex`, and local work folders.
- Oversized TIFF derivatives unless the release specifically includes submission assets.
- Duplicate generated artifacts that can be regenerated.
- Private reviewer comments or collaborator-only notes.
- Any material that could be misread as operational misuse instructions rather than abstract synthetic feature generation.

## Release Phases

### Phase 1: Clean Private Release Candidate

Goal: create a clean local/public-ready repository without immediately publishing it.

Actions:

1. Create a clean release folder outside the development workspace.
2. Copy source code, configs, docs, manuscript-facing tables, and selected figures.
3. Remove local paths and internal-only notes.
4. Add `README.md`, `LICENSE`, `CITATION.cff`, `.gitignore`, and `SECURITY_AND_SAFETY.md`.
5. Run the smoke test from a clean terminal.
6. Confirm the smoke output matches expected shape/count checks.

Exit criteria:

- Fresh clone/folder can run the smoke test in minutes.
- No local absolute paths appear in public-facing docs except examples explicitly marked as local development paths.
- The README explains the benchmark, limitations, and synthetic-only scope.

Status note:

- A local `release_candidate/` folder was created with `tools/create_release_candidate.ps1`.
- The smoke test was run from inside the release-candidate folder with `configs/smoke_test_config.json`.
- The smoke test completed successfully with exit code `0` in approximately `50` seconds.
- The run generated the expected smoke-test tables and figures under `release_candidate/outputs/smoke_test/`.
- QA details are recorded in `RELEASE_CANDIDATE_QA.md`.

### Phase 2: Public GitHub Repository

Goal: publish a usable repository.

Actions:

1. Create GitHub repository.
2. Push the clean release candidate.
3. Add repository metadata and topics.
4. Add a release branch or tag candidate.
5. Confirm README renders correctly.
6. Confirm configs and commands work from the repository root.

Exit criteria:

- Public URL exists.
- Smoke test instructions are clear.
- The repository does not include raw development logs or unnecessary intermediate outputs.

### Phase 3: Citable Archived Release

Goal: provide a stable DOI-backed artifact.

Actions:

1. Create a tagged GitHub release, e.g. `v0.1.0-manuscript`.
2. Archive the release on Zenodo.
3. Obtain DOI.
4. Update `CITATION.cff`.
5. Update `DATA_CODE_AVAILABILITY_STATEMENT.md`.
6. Update manuscript and cover letter if needed.

Exit criteria:

- Repository URL and DOI are available.
- Manuscript availability statement cites the DOI.
- Release notes describe what was used for the manuscript.

## Minimum Reproducibility Targets

The public release should support three levels of reproducibility:

| Level | Target user | Command target | Expected time | Required output |
| --- | --- | --- | --- | --- |
| Smoke test | Editor, quick reviewer, collaborator | `configs/smoke_test_config.json` | Minutes | Pipeline completes and writes core output tables. |
| Default manuscript run | Technical reviewer | `configs/default_manuscript_config.json` | Moderate | Recreates manuscript-scale summary outputs and figures. |
| Large replication | Methods reviewer | `configs/large_replication_config.json` | Longer optional run | Recreates larger held-out replication outputs. |

Minimum acceptance checks:

- All commands run from repository root.
- Each run writes resolved `run_config.json` and copied `input_config.json`.
- Smoke test produces expected output folders and summary tables.
- Default run produces the six manuscript-facing figure files.
- Large replication is documented as optional/slower.

## Recommended README Structure

```markdown
# Hierarchical Operational-State Monitoring Benchmark

## What this is
Synthetic benchmark for testing when hierarchical operational state improves AI misuse monitoring.

## What this is not
Not a production misuse detector, not deployment validation, and not a replacement for classifiers or red teaming.

## Quick start
Install dependencies and run the smoke test.

## Reproduce manuscript-scale outputs
Run the default manuscript config.

## Optional large replication
Run the large replication config.

## Benchmark design
Scenario generator, archetypes, hard negatives, typed state, baselines, routing, workload simulations.

## Outputs
Where figures and tables are written.

## Safety and limitations
Synthetic-only, abstract features, no operational misuse instructions.

## Citation
CITATION.cff and DOI.
```

## Safety and Use-Limits Note

The public repository should include `SECURITY_AND_SAFETY.md` with language like:

> This benchmark uses synthetic, abstract feature representations to study monitoring and routing failure modes. It does not contain operational misuse instructions, deployment logs, or real user data. The code is intended for research on monitoring representations, calibration, escalation routing, and analyst workload. It should not be used as a production misuse detector without external validation, domain-specific review, and appropriate governance.

## Licensing Recommendation

Recommended default:

- MIT License for code.
- CC BY 4.0 for manuscript text, figures, and documentation if desired.

Alternative:

- Apache-2.0 if a more explicit patent grant is preferred for code.

Decision needed before public release:

- Confirm whether all manuscript/figure assets should share the code license or receive a separate documentation/media license.

## Citation File Plan

Create `CITATION.cff` with:

- title
- author list
- repository URL
- version
- DOI once available
- release date
- preferred citation text

Placeholder citation before DOI:

```yaml
cff-version: 1.2.0
title: "A synthetic mechanism benchmark for hierarchical operational-state monitoring of AI misuse"
message: "If you use this benchmark, please cite the manuscript and archived release."
authors:
  - family-names: "Blaszczak-Boxe"
    given-names: "Christopher"
version: "0.1.0"
date-released: "2026-08-17"
```

## Data/Code Availability Update

The public GitHub repository and release page are now available:

- Repository: https://github.com/boxeman/hierarchical-operational-state-benchmark
- GitHub release `v0.1.0`: https://github.com/boxeman/hierarchical-operational-state-benchmark/releases/tag/v0.1.0

Current manuscript-facing wording:

> Code and synthetic benchmark materials are available at:
> https://github.com/boxeman/hierarchical-operational-state-benchmark
>
> The manuscript release candidate is archived as GitHub release `v0.1.0`:
> https://github.com/boxeman/hierarchical-operational-state-benchmark/releases/tag/v0.1.0

A DOI-backed archive should be added if the repository is later deposited through Zenodo or another archival service.

## Public Release Checklist

- [x] Choose repository name.
- [x] Choose license.
- [x] Create clean release folder.
- [ ] Move source into `src/ai_mla_monitor/` or decide to keep flat scripts for v0.1.
- [x] Remove local paths from public docs.
- [x] Add README.
- [x] Add `SECURITY_AND_SAFETY.md`.
- [x] Add `CITATION.cff`.
- [x] Add `.gitignore`.
- [x] Add minimal tests or smoke assertions.
- [x] Run smoke test from clean folder.
- [x] Run default manuscript config or document expected runtime.
- [x] Create GitHub repository.
- [x] Create release tag.
- [ ] Archive on Zenodo.
- [x] Update `DATA_CODE_AVAILABILITY_STATEMENT.md`.
- [x] Update manuscript and submission materials with repository/release link.

## Immediate Next Action

The next implementation step after this plan should be:

**Create a DOI-backed archive, preferably through Zenodo, and then update the citation metadata and manuscript availability statement with the DOI.**
