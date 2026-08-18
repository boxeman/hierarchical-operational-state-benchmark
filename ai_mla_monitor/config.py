from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioConfig:
    seed: int = 42
    benign_n: int = 500
    obvious_n: int = 30
    borderline_n: int = 50
    distributed_n: int = 50
    systemic_n: int = 35
    policy_ambiguous_n: int = 40
    escalating_n: int = 50
    uncertainty_required_n: int = 36
    signature_required_n: int = 36
    provenance_required_n: int = 36
    adversarial_false_systemic_n: int = 18
    frontier_overlap_n: int = 18
    systemic_mean: float = 0.205
    benign_cluster_mean: float = 0.145
    sub_thresh: float = 0.18
    evidence_norm: int = 8
    cluster_norm: int = 25


@dataclass(frozen=True)
class SplitConfig:
    train_scenarios: int = 24
    val_scenarios: int = 8
    test_scenarios: int = 8
    train_seed0: int = 1000
    val_seed0: int = 2000
    test_seed0: int = 3000


FEATURE_COLUMNS = [
    "flat_max",
    "flat_mean",
    "flat_top3_mean",
    "flat_count_above",
    "evidence_count",
    "evidence_volume",
    "trend",
    "account_rule",
    "intent_uncertainty",
    "policy_uncertainty",
    "provenance_quality",
    "legitimacy_context",
    "signature_count",
    "signature_mean_risk",
    "signature_mean_evidence",
    "signature_confidence",
    "system_score",
    "typed_system_score",
]

FLAT_FEATURE_COLUMNS = [
    "flat_max",
    "flat_mean",
    "flat_top3_mean",
    "flat_count_above",
]

HIER_NO_UNCERTAINTY_COLUMNS = [
    "flat_max",
    "flat_mean",
    "flat_top3_mean",
    "flat_count_above",
    "evidence_count",
    "evidence_volume",
    "trend",
    "account_rule",
    "provenance_quality",
    "legitimacy_context",
    "signature_count",
    "signature_mean_risk",
    "signature_mean_evidence",
    "signature_confidence",
    "system_score",
]

HIER_NO_SIGNATURE_COLUMNS = [
    "flat_max",
    "flat_mean",
    "flat_top3_mean",
    "flat_count_above",
    "evidence_count",
    "evidence_volume",
    "trend",
    "account_rule",
    "intent_uncertainty",
    "policy_uncertainty",
    "provenance_quality",
    "legitimacy_context",
]

HIER_NO_PROVENANCE_COLUMNS = [
    "flat_max",
    "flat_mean",
    "flat_top3_mean",
    "flat_count_above",
    "evidence_count",
    "evidence_volume",
    "trend",
    "account_rule",
    "intent_uncertainty",
    "policy_uncertainty",
    "legitimacy_context",
    "signature_count",
    "signature_mean_risk",
    "signature_mean_evidence",
    "signature_confidence",
    "system_score",
]
