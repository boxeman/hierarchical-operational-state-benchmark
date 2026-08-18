# Methods Protocol

## Objective

This protocol defines the reproducible benchmark used to test whether hierarchical operational state improves monitoring when harmful signals are weak locally but coherent across account, signature, campaign, or product/system levels.

The benchmark is synthetic by design. Its purpose is not to claim production misuse-detection performance. Its purpose is to isolate mechanisms, compare flat and hierarchical representations, and identify the conditions under which preserving operational state improves review, calibration, escalation, workload, or product-alert outcomes.

## Research Question

Under what conditions does preserving hierarchical operational information produce measurably better monitoring outcomes than collapsing information into flat account- or message-level scores?

## Benchmark Units

The independent experimental unit is the scenario. Each scenario generates accounts, signatures, campaign identifiers, message-level risk signals, uncertainty fields, provenance fields, and expected operational routes.

Scenario-level resampling is used for bootstrap intervals so the scenario remains the independent unit.

## Scenario Splits

The benchmark uses independent train, validation, and held-out test scenarios.

- Training scenarios fit learned flat and hierarchical models.
- Validation scenarios tune account-review thresholds and routing policies.
- Held-out test scenarios report manuscript-facing metrics and figures.

Validation information is not used to fit model weights, and held-out test scenarios are not used for threshold or routing-policy selection.

## Scenario Variation

Scenarios vary:

- systemic exploit account count
- systemic message-score mean
- benign shared-topic score mean
- cluster-size normalization
- sub-threshold evidence threshold
- adversarial benign coordinated activity
- near-overlap systemic/benign frontier cases

This makes the benchmark a small phase-space test rather than a single fixed demonstration.

## Archetype Families

Harmful archetypes include:

- obvious violations
- repeated borderline behavior
- escalating behavior
- distributed jailbreak-like behavior
- systemic distributed exploits
- uncertainty-required harmful cases
- signature-required campaigns
- provenance-required harm

Benign or hard-negative archetypes include:

- ordinary benign accounts
- policy-ambiguous accounts
- benign shared-topic clusters
- legitimate ambiguous cases
- benign signature lookalikes
- low-provenance artifacts
- authorized security testing cohorts
- classroom or lab exercise lookalikes
- public-news-driven bursts
- support/debugging workflows
- legitimate dual-use research cohorts
- copied public example text spreading across accounts

## Preserved Operational State

The hierarchical representation can include:

- local risk score
- repeated low-amplitude signal count
- evidence count
- intent uncertainty
- policy uncertainty
- provenance quality
- signature confidence
- shared signature
- campaign identifier
- signature/campaign account count
- signature/campaign mean risk
- system-level score
- legitimacy context

These fields are intended to represent state that may be lost when monitoring collapses evidence into a single local score too early.

## Compared Methods

Flat baselines:

- `flat_max`
- `flat_mean`
- `flat_top3_mean`
- `flat_count_above`
- `learned_logistic_flat`
- `boosted_stumps_flat`

Account/rule baselines:

- `account_rule`
- `account_logistic_rule`

Hierarchical learned methods:

- `learned_logistic_hier`
- `boosted_stumps_hier`

Hierarchical ablations:

- `ablation_no_uncertainty`
- `ablation_no_signature`
- `ablation_no_provenance`

System-alert baselines:

- `signature_count_only`
- `signature_mean_risk`
- `system_hierarchy_alert`
- `typed_system_hierarchy_alert`

## Threshold Tuning

Account-review thresholds are selected on validation scenarios and applied to held-out test scenarios.

Threshold evaluation reports:

- selected accounts
- precision
- recall
- harm-weighted recall
- harm caught per 100 reviews
- false-positive reviews
- policy-ambiguous accounts flagged
- false-systemic accounts flagged
- systemic account recall

## Calibration

Calibration is evaluated on held-out test scenarios using:

- expected calibration error
- Brier score
- calibration bins
- calibration curves

Calibration matters because operational monitoring systems use scores not only for ranking but also for routing, escalation, and review-budget allocation.

## Routing Policy

Routing evaluates whether each account is assigned to the expected operational action:

- `auto_handle`
- `human_review`
- `policy_review`
- `product_investigation`

Routing policies are selected by grid search on validation scenarios. Tuned parameters include:

- human-review score threshold
- policy uncertainty threshold
- product-investigation signature-confidence threshold
- product-investigation evidence threshold
- product-investigation score threshold
- provenance suppression threshold
- legitimacy suppression threshold
- adversarial benign false-alert penalty

The routing utility rewards correct escalation and penalizes:

- missed product investigations
- unnecessary product investigations
- adversarial benign coordinated product-investigation alerts
- over-escalation
- under-escalation

## System Alerts

System-alert evaluation asks whether a shared harmful signature/campaign is surfaced as a product-level pattern.

Metrics include:

- product-alert hit rate
- missed product alerts
- false systemic alerts
- adversarial false systemic alerts
- rank of the true exploit signature

The current benchmark treats top-three product alerting as difficult and fragile. Mean exploit rank is therefore reported alongside hit rate.

## Targeted Ablations

Targeted ablations are designed so one state component is required to solve a matched failure mode:

- uncertainty distinguishes legitimate ambiguity from harmful ambiguity
- signature/campaign state distinguishes benign lookalikes from coordinated harm
- provenance distinguishes low-quality artifacts from actionable evidence

The ablation metric is selection rate at validation-tuned thresholds within each matched family.

## Product-Investigation Frontier

The product-investigation frontier varies the penalty for adversarial benign coordinated product-investigation alerts. For each penalty value, routing is retuned on validation scenarios and evaluated on held-out test scenarios.

This produces an operating frontier between:

- missed true product investigations
- adversarial false systemic product-investigation alerts
- macro route F1
- routing utility
- product-investigation precision and recall

## Practical Workload Simulation

Accounts are assigned deterministic synthetic arrival times over a 14-day monitoring window.

Route costs are:

- `auto_handle`: 0 hours
- `human_review`: 0.25 hours
- `policy_review`: 0.75 hours
- `product_investigation`: 2 hours

Metrics include:

- product-campaign alert rate
- missed product campaigns
- median and p90 time-to-alert
- total analyst hours
- analyst days
- peak daily analyst hours
- days over capacity
- false product investigations
- adversarial false systemic product investigations

## Deduplicated Backlog Simulation

Product-investigation alerts are deduplicated by:

- `scenario_id`
- `signature`
- `campaign_id`

Human-review and policy-review tasks remain account-level. Queue simulation is evaluated under finite analyst capacity at 8, 16, and 32 analyst-hours/day.

Queue policies include:

- FIFO
- product-first
- product-plus-policy-first
- severity-weighted

Metrics include:

- unique product alerts
- true unique product alerts
- false unique product alerts
- duplicate product alerts avoided
- peak backlog hours
- mean backlog hours
- days with unresolved backlog
- p90 product-investigation delay
- true product alerts resolved within 24 and 48 hours
- spillover non-product backlog

## Bootstrap Intervals

Bootstrap confidence intervals are computed by resampling held-out scenarios with replacement.

Scenario-block bootstrap intervals are reported for:

- account-review metrics
- system-alert metrics
- targeted ablation diagnostics
- routing metrics
- product-investigation frontier metrics
- workload/time-to-alert metrics
- deduplicated backlog and priority-queue metrics

## Reproducibility Notes

The benchmark uses deterministic seeds in the scenario generator and reproducible validation/test splits. The current learned models are dependency-free implementations used to keep the prototype runnable without external ML libraries.

The manuscript-facing output package is generated by:

```powershell
python run_benchmark.py --config configs/default_manuscript_config.json
```

Outputs are written to:

```text
outputs/prototype_package/
```

Each run writes the resolved configuration to:

```text
outputs/prototype_package/run_config.json
```

If a config file is supplied, it is copied to:

```text
outputs/prototype_package/input_config.json
```

The configuration format is documented in `EXPERIMENT_CONFIG.md`; named run
commands are documented in `RUN_EXPERIMENTS.md`.

After both the default manuscript run and the large replication run have been
generated, replication stability is summarized with:

```powershell
python compare_replication.py
```

This reads:

```text
outputs/prototype_package/
outputs/large_replication/
```

and writes:

```text
outputs/replication_comparison_table.csv
outputs/replication_comparison_table.md
```

The replication comparison is a robustness check, not a replacement for the
scenario-bootstrap intervals. It asks whether the manuscript-level directional
claims persist when the held-out scenario count is increased.

## Known Limitations

This is a synthetic mechanism test. It cannot establish real-world deployment performance without external traces, public benchmark data, or operationally realistic event logs.

The default manuscript held-out scenario count is modest, so bootstrap intervals
should be interpreted as stability checks rather than definitive population
estimates. The large replication run partially addresses this limitation, but
additional independent seeds and external datasets are still needed before
making deployment-level claims.

The route policy class is threshold-based. It demonstrates that operational routing can be evaluated, but it is not yet a mature scheduling or incident-response policy.

The system-alert layer remains fragile. Current results show rank improvement in a harder setup, but reliable top-three product alerting has not yet been established.

The benchmark does not yet include analyst specialization, real policy-review workflows, post-alert action quality, or realistic organizational handoff constraints.

