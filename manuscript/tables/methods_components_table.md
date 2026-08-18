# Methods Components Table

| component | purpose | inputs | output | evaluated_by | related_figure |
| --- | --- | --- | --- | --- | --- |
| Scenario generator | Create transparent train/validation/test monitoring cases | ScenarioConfig and split seeds | Synthetic account/signature/campaign records | All downstream metrics | Figures 1-6 |
| Archetype families | Represent harmful benign and hard-negative behavior | Account type definitions and scenario parameters | Labeled harmful/benign archetypes | Per-type recall and targeted diagnostics | Figures 1-2 and 5 |
| Hierarchical operational state | Preserve evidence uncertainty provenance and campaign context | Local scores uncertainty provenance signatures campaign IDs and legitimacy context | Flat and hierarchical feature matrices | Ablations calibration routing and workload metrics | Figures 1-6 |
| Flat baselines | Test whether local score collapse is sufficient | Flat local risk features | Flat account scores | Account-review calibration and routing metrics | Figures 1 3 and 4 |
| Learned hierarchical model | Test whether preserved state improves monitoring | Flat plus hierarchical feature columns | Hierarchical account scores and routes | Account-review calibration routing stress and workload metrics | Figures 1-6 |
| Feature ablations | Identify state-component-specific failure modes | Hierarchical features with uncertainty signature or provenance removed | Ablated model scores and routes | Targeted false-selection rates and routing metrics | Figures 2 and 4 |
| System-alert baselines | Test whether shared signatures surface product-level patterns | Signature counts mean risk and typed system state | System alert rankings | Product-alert hit rate exploit rank and false systemic alerts | Figure 3 |
| Validation threshold tuning | Select operating thresholds without using test labels | Validation scores and labels | Chosen account-review thresholds | Held-out threshold metrics | Figure 1 |
| Validation routing policy search | Select escalation thresholds and penalties | Validation scores operational state and expected routes | Tuned routing policy | Held-out route accuracy macro F1 utility and escalation errors | Figures 4-5 |
| Scenario bootstrap | Quantify stability across held-out scenarios | Held-out scenario IDs and metric tables | 95 percent scenario-bootstrap intervals | Confidence intervals across manuscript figures | Figures 1-6 |
| Adversarial false-systemic stress | Test benign coordinated lookalikes | Authorized testing classroom news support research and copied-example cohorts | False product-investigation burden | False systemic product routes and human-review burden | Figure 5 |
| Product-investigation frontier | Expose recall versus false systemic escalation tradeoff | Penalty sweep and validation-retuned routing policies | Operating frontier | Missed product investigations false systemic alerts precision recall | Figure 5B |
| Workload simulation | Convert routing into analyst burden and alert time | Predicted routes arrival times and route costs | Analyst-hours and time-to-alert tables | Campaign alert rate analyst hours and false product investigations | Figure 6A |
| Deduplicated backlog simulation | Model repeated product alerts and finite analyst capacity | Predicted routes signatures campaigns and capacity settings | Deduplicated queue traces | Peak backlog unique false alerts and product-alert delay | Figure 6B |
| Priority queue simulation | Test whether scarce capacity can protect product response | Queue tasks route priorities capacity and severity scores | Priority-queue metrics | p90 product delay 24h/48h resolution and spillover backlog | Figure 6C |
| Manuscript summary table | Map results to claims and figures | Current benchmark output tables | Compact CSV and Markdown result summary | Reviewer-facing interpretation | Figures 1-6 |

