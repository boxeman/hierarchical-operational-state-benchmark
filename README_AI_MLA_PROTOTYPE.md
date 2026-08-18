# AI-MLA Studio Prototype: Hierarchical Operational-State Monitoring

## Purpose

This prototype tests whether preserving hierarchical operational state across
message, account, signature/campaign, and product/system levels improves
monitoring when harmful signals are weak locally but coherent globally.

It is intentionally synthetic and transparent. The goal is not to claim a
production misuse detector; the goal is to create a falsifiable, reproducible
research artifact for comparing flat, aggregate, and hierarchical monitoring
representations.

## What Is Implemented

- Clean Python package: `ai_mla_monitor/`
- Configurable synthetic scenario generator
- Held-out train/validation/test scenario splits
- Hard negatives that resemble false systemic clusters
- Adversarial false-systemic lookalikes:
  - authorized security testing cohorts
  - classroom/lab exercise lookalikes
  - public-news-driven bursts
  - support/debugging workflows with repeated technical terms
  - legitimate dual-use research cohorts
  - copied public example text spreading across accounts
- Flat, account-level, and system-level features
- Rule baselines
- Dependency-free learned baselines:
  - logistic regression via NumPy gradient descent
  - gradient-boosted decision stumps
- Validation-tuned operating thresholds
- Calibration bins, ECE, Brier scores, and SVG calibration curves
- Scenario-bootstrap confidence intervals
- Feature ablations:
  - no uncertainty fields
  - no signature/campaign fields
  - no provenance field
- Targeted matched failure-mode families:
  - uncertainty-required harmful vs legitimate ambiguous cases
  - signature-required campaign vs locally similar benign accounts
  - provenance-required harm vs low-provenance artifacts
- Account-review metrics
- System-alert metrics
- Routing/escalation metrics for:
  - auto-handle
  - human review
  - policy review
  - product investigation
- Validation-tuned routing policy search
- Product-investigation frontier bootstrap intervals
- Time-to-alert and analyst-workload simulation
- Product-alert deduplication and finite-capacity backlog simulation
- Priority queue policies for product-first, product-plus-policy-first, and
  severity-weighted analyst processing
- Backlog bootstrap intervals for finite-capacity queue metrics
- Reviewer-facing experiment configuration via `experiment_config.json`
- Named smoke, manuscript, and large-replication run configurations
- Default-vs-large replication comparison table
- Journal-facing claim-to-evidence manuscript map
- Journal submission package outline
- Related-work scaffold with citation slots
- Manuscript-facing output tables
- Manuscript-facing figures and captions
- Manuscript scaffold, methods protocol, and methods component table

## Run

```powershell
python run_benchmark.py --config configs/default_manuscript_config.json
```

Outputs are written to:

```text
outputs/prototype_package/
```

## Key Outputs

- `MANUSCRIPT_DRAFT.md`
- `MANUSCRIPT_SCAFFOLD.md`
- `MANUSCRIPT_CLAIM_EVIDENCE_MAP.md`
- `SUBMISSION_PACKAGE_OUTLINE.md`
- `RELATED_WORK_SCAFFOLD.md`
- `METHODS_PROTOCOL.md`
- `RESULTS_ROADMAP.md`
- `EXPERIMENT_CONFIG.md`
- `RUN_EXPERIMENTS.md`
- `experiment_config.json`
- `configs/smoke_test_config.json`
- `configs/default_manuscript_config.json`
- `configs/large_replication_config.json`
- `compare_replication.py`
- `REPLICATION_COMPARISON.md`
- `outputs/replication_comparison_table.csv`
- `outputs/replication_comparison_table.md`
- `run_config.json`
- `input_config.json`
- `benchmark_summary.md`
- `methods_components_table.csv`
- `methods_components_table.md`
- `summary_results_table.csv`
- `summary_results_table.md`
- `account_metrics_by_split.csv`
- `type_recall_by_split.csv`
- `system_alert_metrics_by_split.csv`
- `validation_tuned_thresholds.csv`
- `test_metrics_at_validation_thresholds.csv`
- `routing_metrics_by_split.csv`
- `routing_confusion_by_split.csv`
- `routing_policy_tuning.csv`
- `routing_account_predictions_test.csv`
- `adversarial_routing_stress.csv`
- `routing_penalty_sweep.csv`
- `routing_frontier_account_predictions_test.csv`
- `bootstrap_routing_frontier_ci.csv`
- `time_to_alert_workload_summary.csv`
- `analyst_workload_daily.csv`
- `bootstrap_time_to_alert_workload_ci.csv`
- `dedup_backlog_summary.csv`
- `dedup_backlog_daily.csv`
- `dedup_backlog_tasks.csv`
- `bootstrap_dedup_backlog_ci.csv`
- `calibration_summary.csv`
- `calibration_bins.csv`
- `calibration_curves_test.svg`
- `targeted_ablation_diagnostics.csv`
- `targeted_ablation_diagnostics_by_scenario.csv`
- `bootstrap_account_ci.csv`
- `bootstrap_system_alert_ci.csv`
- `bootstrap_targeted_ablation_ci.csv`
- `bootstrap_routing_ci.csv`
- `test_account_scores_for_bootstrap.csv`
- `figures/figure1_account_review.svg`
- `figures/figure2_targeted_ablations.svg`
- `figures/figure3_calibration_system_alerts.svg`
- `figures/figure4_routing_escalation.svg`
- `figures/figure5_adversarial_false_systemic.svg`
- `figures/figure6_product_investigation_frontier.svg`
- `figures/figure7_time_to_alert_workload.svg`
- `figures/figure8_dedup_backlog.svg`
- `figures/figure9_priority_queues.svg`
- `figures/figure_captions.md`
- `train_accounts.csv`
- `validation_accounts.csv`
- `test_accounts.csv`
- `signature_state_all_splits.csv`

## Replication Comparison

After running both the default manuscript configuration and the larger
replication configuration, compare the claims with:

```powershell
python compare_replication.py
```

This writes:

```text
outputs/replication_comparison_table.csv
outputs/replication_comparison_table.md
```

The comparison asks which manuscript claims survive when the held-out scenario
count increases. In the current run, account-review, component-ablation,
calibration, routing, false-systemic stress, workload, backlog, and
priority-queue claims remain directionally stable. The clearest caveats are
that system-alert top-k hit rate remains fragile and finite-capacity product
delays increase substantially in the larger replication run.

## Current Test-Set Finding

At 15% account-review budget under the adversarial false-systemic stress test,
the learned hierarchical model outperforms the learned flat baseline on
harm-weighted recall while maintaining very high precision. Aggregate ablations
remain close to the full model, but the targeted diagnostic families now show
distinct failure modes:

- Removing uncertainty causes legitimate ambiguous cases to be selected at high
  rates, while the full hierarchy rejects them.
- Removing signature/campaign state causes locally similar benign accounts to
  be selected at high rates, while the full hierarchy preserves the campaign
  distinction.
- Removing provenance causes low-provenance artifact accounts to be selected at
  high rates, while the full hierarchy mostly rejects them.

This is the key manuscript-strengthening result: the value of each state
component is clearest in matched hard cases, not necessarily in aggregate
review-budget curves.

For system alerts, the current harder run is more cautious: the typed
hierarchical alert improves true-exploit ranking relative to
signature-count-only and signature-mean-risk baselines, but no method reliably
surfaces the exploit in the top three alerts. False systemic alerts remain a
central failure mode.

Calibration currently favors learned logistic models over rule scores and
boosted stumps, with the full hierarchical logistic model among the best test
set calibration performers.

Routing metrics now make the operational-state claim more explicit. The learned
hierarchical model currently has the strongest held-out routing utility among
the evaluated methods, because it can route cases beyond the binary
auto-handle/human-review distinction and use uncertainty, signature/campaign
state, and provenance to choose policy review or product investigation.

The adversarial false-systemic stress test made the result harder and more
credible. The routing utility now explicitly penalizes benign coordinated
lookalikes routed to product investigation. Under this validation-tuned routing
objective, the learned hierarchical model routed only 2 of 1,104 adversarial
benign lookalikes to product investigation (0.18%). By contrast, hand-built
account/system hierarchy rules routed roughly 46-47% of those adversarial
benign lookalikes to product investigation. The remaining false product
investigations for the learned hierarchy came from the public-news-burst
lookalike family.

The same stress test also shows that the problem is not solved. Scenario-
bootstrap intervals now estimate validation-tuned learned-hierarchy macro route
F1 at 0.795, 95% CI 0.782-0.806, compared with 0.690, 95% CI 0.670-0.711 for
validation-tuned learned flat routing. Missed product investigations remain much
lower for validation-tuned hierarchy than learned flat routing: mean 115.6 versus
589.9. However, unnecessary product investigations remain nonzero, and simple
rule hierarchies over-alert badly on coordinated benign activity. Figure 5 now
visualizes this stress test.

The product-investigation penalty sweep now retunes a broadened learned
hierarchical routing policy under adversarial false-alert penalties from 0 to
0.05. The benchmark includes near-overlap frontier cases and a
`legitimacy_context` state variable, so Figure 6 now exposes a real operating
frontier. At low penalties, the selected policy misses 128 true product
investigations while allowing 4 adversarial benign coordinated product alerts.
At penalties of 0.005 and above, the selected policy suppresses adversarial
false systemic alerts to 0, but misses 153 true product investigations. In
other words, the system can trade 25 additional missed product investigations
for complete suppression of the observed adversarial false systemic alerts in
this stress test.

Stress-specific bootstrap intervals now quantify that frontier. At low
penalties, missed product investigations average about 127-128 with 95%
intervals roughly spanning 104-151, while adversarial false systemic product
alerts average about 4 with intervals spanning approximately 0-9. At penalties
of 0.005 and above, adversarial false systemic product alerts remain 0 across
bootstrap resamples, while missed product investigations rise to about 152-154
with intervals spanning roughly 134-178.

The practical monitoring simulation adds a simple deployment-facing layer.
Accounts are assigned deterministic synthetic arrival times over a 14-day
window, and route costs are set to 0 hours for auto-handle, 0.25 hours for
human review, 0.75 hours for policy review, and 2 hours for product
investigation. Under validation-tuned routing, the learned hierarchy alerted
87.5% of product campaigns, missed 3 product campaigns, had median
time-to-alert of 0 hours among alerted campaigns, and required 1,891.5 analyst
hours. The typed rule hierarchy achieved the same campaign alert rate but
required 4,324.75 analyst hours and produced far more false product
investigations. Figure 7 visualizes this practical tradeoff.

Figure 7 now includes scenario-bootstrap intervals. The learned hierarchy's
product-campaign alert rate averaged 0.875 with a 95% interval of 0.750-0.958,
while total analyst hours averaged 1,879.2 with a 95% interval of
1,594.4-2,132.7. The typed rule hierarchy had a similar campaign alert rate
but much higher workload: 4,289.7 analyst hours on average, with a 95% interval
of 3,552.5-4,901.4. It also produced far more false product investigations:
994.5 on average versus 121.5 for the learned hierarchy.

The deduplicated backlog simulation adds finite analyst capacity. Product
investigation alerts are collapsed by `scenario_id + signature + campaign_id`
before queue simulation, and the queue is evaluated at 8, 16, and 32
analyst-hours/day. At 16 analyst-hours/day, the learned hierarchy produces 32
unique product alerts, including 21 true and 11 false unique product alerts,
while avoiding 470 duplicate product alerts. The typed rule hierarchy produces
51 unique product alerts, including 21 true and 30 false unique product alerts,
while avoiding 1,115 duplicates. The learned hierarchy has lower peak backlog
than the typed rule hierarchy: 545.5 versus 1,118.75 hours. However, backlog
remains unresolved on all 14 simulated days, which makes the conclusion more
practical: hierarchy reduces queue pressure, but finite staffing still matters.
Figure 8 visualizes this deduplicated finite-capacity result.

The priority-queue simulation then asks whether scarce analyst capacity can be
directed toward product-level action quickly enough. At 16 analyst-hours/day,
the learned hierarchy's FIFO queue had p90 product-investigation delay of
214.9 hours, with 28.6% of true product alerts resolved within 24 hours and
47.6% within 48 hours. Product-first priority reduced p90 product-investigation
delay to 40.1 hours and resolved 100% of true product alerts within 48 hours,
while severity-weighted priority had p90 delay of 41.9 hours and resolved
61.9% within 24 hours. The tradeoff is spillover: peak non-product backlog rose
from 531.75 hours under FIFO to 545.75 hours under product-first priority.
Scenario-bootstrap intervals support the same qualitative conclusion. Figure 9
visualizes this priority-queue operating layer.

## Manuscript Claim

The defensible claim is conditional:

Hierarchical operational-state representations can improve monitoring when
harmful signals are weak locally but coherent globally, but the benefit must be
tested against strong flat and aggregate baselines and evaluated under review
budget, escalation quality, and false systemic alert constraints.

## Next Technical Steps

1. Tune priority policies against explicit service-level objectives.
2. Add analyst specialization and route-specific service rates.
3. Expand system-alert evaluation beyond top-3 alert hit rate.
4. Increase held-out scenario diversity for narrower workload intervals.
5. Add real or public benchmark traces where possible.

