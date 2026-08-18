import numpy as np
import pandas as pd

from .config import ScenarioConfig


ADVERSARIAL_FALSE_SYSTEMIC_TYPES = {
    "authorized_security_testing_cohort",
    "classroom_lab_exercise_lookalike",
    "public_news_burst_lookalike",
    "support_debugging_repeated_terms",
    "legitimate_dual_use_research_cohort",
    "copied_public_example_spread",
    "frontier_legitimate_coordination",
}


FALSE_SYSTEMIC_TYPES = {
    "benign_research_cohort",
    "classroom_assignment",
    "popular_news_event",
    "coordinated_legitimate_activity",
    "support_debugging_workflow",
    "uncertainty_legit_ambiguous",
    "signature_matched_benign",
    "low_provenance_artifact",
} | ADVERSARIAL_FALSE_SYSTEMIC_TYPES


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def make_account(
    rng,
    account_id,
    archetype,
    harmful,
    harm_weight,
    scores,
    signature=None,
    campaign_id=None,
    expected_route="auto_handle",
    intent_uncertainty=None,
    policy_uncertainty=None,
    provenance_quality=None,
    legitimacy_context=None,
):
    scores = clip01(np.asarray(scores, dtype=float))
    if intent_uncertainty is None:
        intent_uncertainty = 1.0 - float(np.mean(scores)) + rng.normal(0, 0.04)
    if policy_uncertainty is None:
        policy_uncertainty = 0.20 + rng.normal(0, 0.05)
    if provenance_quality is None:
        provenance_quality = 0.82 + rng.normal(0, 0.07)
    if legitimacy_context is None:
        legitimacy_context = 0.10 + rng.normal(0, 0.04)
    return {
        "id": account_id,
        "type": archetype,
        "harmful": bool(harmful),
        "harm_weight": harm_weight,
        "scores": scores,
        "signature": signature,
        "campaign_id": campaign_id,
        "expected_route": expected_route,
        "intent_uncertainty": float(clip01(intent_uncertainty)),
        "policy_uncertainty": float(clip01(policy_uncertainty)),
        "provenance_quality": float(clip01(provenance_quality)),
        "legitimacy_context": float(clip01(legitimacy_context)),
    }


def gen_benign(rng, cfg):
    accounts = []
    for i in range(cfg.benign_n):
        n = rng.integers(3, 9)
        accounts.append(
            make_account(
                rng,
                f"benign_{i}",
                "benign",
                False,
                0,
                rng.normal(0.10, 0.05, n),
                intent_uncertainty=0.18 + rng.normal(0, 0.05),
                policy_uncertainty=0.12 + rng.normal(0, 0.04),
            )
        )
    return accounts


def gen_false_systemic_clusters(rng, cfg):
    specs = [
        ("benign_research_cohort", "research_dual_use_methods", 18, cfg.benign_cluster_mean + 0.020, 0.38),
        ("classroom_assignment", "classroom_security_lab", 22, cfg.benign_cluster_mean + 0.015, 0.45),
        ("popular_news_event", "news_event_discussion", 28, cfg.benign_cluster_mean + 0.005, 0.32),
        ("coordinated_legitimate_activity", "legitimate_team_debug", 20, cfg.benign_cluster_mean + 0.025, 0.30),
        ("support_debugging_workflow", "support_debug_topic", 16, cfg.benign_cluster_mean + 0.030, 0.50),
    ]
    accounts = []
    for archetype, signature, n_accounts, mean, policy_uncertainty in specs:
        for i in range(n_accounts):
            n = rng.integers(5, 12)
            accounts.append(
                make_account(
                    rng,
                    f"{archetype}_{i}",
                    archetype,
                    False,
                    0,
                    rng.normal(mean, 0.04, n),
                    signature=signature,
                    expected_route="auto_handle",
                    intent_uncertainty=0.52 + rng.normal(0, 0.08),
                    policy_uncertainty=policy_uncertainty + rng.normal(0, 0.06),
                    provenance_quality=0.72 + rng.normal(0, 0.10),
                    legitimacy_context=0.62 + rng.normal(0, 0.08),
                )
            )
    return accounts


def gen_adversarial_false_systemic_lookalikes(rng, cfg):
    """Coordinated legitimate activity designed to resemble a systemic exploit."""
    specs = [
        (
            "authorized_security_testing_cohort",
            "authorized_security_eval_shared_method",
            cfg.adversarial_false_systemic_n,
            cfg.systemic_mean + 0.010,
            "policy_review",
            0.44,
            0.78,
            0.92,
        ),
        (
            "classroom_lab_exercise_lookalike",
            "classroom_lab_shared_prompt",
            cfg.adversarial_false_systemic_n + 4,
            cfg.systemic_mean + 0.000,
            "policy_review",
            0.36,
            0.82,
            0.88,
        ),
        (
            "public_news_burst_lookalike",
            "public_news_shared_discussion",
            cfg.adversarial_false_systemic_n + 10,
            cfg.systemic_mean - 0.010,
            "auto_handle",
            0.24,
            0.48,
            0.86,
        ),
        (
            "support_debugging_repeated_terms",
            "support_debugging_shared_error",
            cfg.adversarial_false_systemic_n + 2,
            cfg.systemic_mean + 0.015,
            "auto_handle",
            0.28,
            0.52,
            0.74,
        ),
        (
            "legitimate_dual_use_research_cohort",
            "legitimate_research_shared_dual_use",
            cfg.adversarial_false_systemic_n + 6,
            cfg.systemic_mean + 0.020,
            "policy_review",
            0.48,
            0.84,
            0.90,
        ),
        (
            "copied_public_example_spread",
            "copied_public_example_shared_text",
            cfg.adversarial_false_systemic_n + 8,
            cfg.systemic_mean + 0.005,
            "auto_handle",
            0.18,
            0.44,
            0.50,
        ),
    ]
    accounts = []
    for archetype, signature, n_accounts, mean, route, intent_u, policy_u, provenance_q in specs:
        for i in range(n_accounts):
            n = rng.integers(6, 12)
            accounts.append(
                make_account(
                    rng,
                    f"{archetype}_{i}",
                    archetype,
                    False,
                    0,
                    rng.normal(mean, 0.035, n),
                    signature=signature,
                    campaign_id=f"benign_{signature}",
                    expected_route=route,
                    intent_uncertainty=intent_u + rng.normal(0, 0.05),
                    policy_uncertainty=policy_u + rng.normal(0, 0.05),
                    provenance_quality=provenance_q + rng.normal(0, 0.06),
                    legitimacy_context=0.72 + rng.normal(0, 0.08),
                )
            )
    return accounts


def gen_frontier_overlap_cases(rng, cfg):
    """Near-overlap product cases where legitimacy suppression creates a real tradeoff."""
    accounts = []
    for i in range(cfg.frontier_overlap_n):
        n = rng.integers(5, 9)
        accounts.append(
            make_account(
                rng,
                f"frontier_systemic_{i}",
                "frontier_systemic_overlap_exploit",
                True,
                4,
                rng.normal(cfg.systemic_mean + 0.012, 0.034, n),
                signature="frontier_overlap_shared_pattern",
                campaign_id="campaign_frontier_overlap",
                expected_route="product_investigation",
                intent_uncertainty=0.62 + rng.normal(0, 0.06),
                policy_uncertainty=0.46 + rng.normal(0, 0.06),
                provenance_quality=0.82 + rng.normal(0, 0.06),
                legitimacy_context=0.52 + rng.normal(0, 0.08),
            )
        )
    for i in range(cfg.frontier_overlap_n):
        n = rng.integers(5, 9)
        accounts.append(
            make_account(
                rng,
                f"frontier_legitimate_{i}",
                "frontier_legitimate_coordination",
                False,
                0,
                rng.normal(cfg.systemic_mean + 0.008, 0.034, n),
                signature="frontier_legitimate_shared_pattern",
                campaign_id="benign_frontier_overlap",
                expected_route="auto_handle",
                intent_uncertainty=0.58 + rng.normal(0, 0.06),
                policy_uncertainty=0.44 + rng.normal(0, 0.06),
                provenance_quality=0.82 + rng.normal(0, 0.06),
                legitimacy_context=0.64 + rng.normal(0, 0.08),
            )
        )
    return accounts


def gen_obvious(rng, cfg):
    accounts = []
    for i in range(cfg.obvious_n):
        n = rng.integers(3, 9)
        scores = clip01(rng.normal(0.10, 0.05, n))
        scores[rng.integers(0, n)] = rng.uniform(0.85, 0.98)
        accounts.append(
            make_account(
                rng,
                f"obvious_{i}",
                "obvious_violation",
                True,
                3,
                scores,
                expected_route="human_review",
                intent_uncertainty=0.15 + rng.normal(0, 0.04),
                policy_uncertainty=0.12 + rng.normal(0, 0.04),
            )
        )
    return accounts


def gen_borderline(rng, cfg):
    accounts = []
    for i in range(cfg.borderline_n):
        n = rng.integers(5, 13)
        accounts.append(
            make_account(
                rng,
                f"repeated_{i}",
                "borderline_repeated",
                True,
                2,
                rng.normal(0.42, 0.05, n),
                expected_route="human_review",
                intent_uncertainty=0.42 + rng.normal(0, 0.07),
                policy_uncertainty=0.25 + rng.normal(0, 0.05),
            )
        )
    return accounts


def gen_distributed(rng, cfg):
    accounts = []
    for i in range(cfg.distributed_n):
        n = rng.integers(15, 31)
        accounts.append(
            make_account(
                rng,
                f"distributed_{i}",
                "distributed_jailbreak",
                True,
                3,
                rng.normal(0.22, 0.05, n),
                expected_route="human_review",
                intent_uncertainty=0.62 + rng.normal(0, 0.08),
                policy_uncertainty=0.22 + rng.normal(0, 0.05),
            )
        )
    return accounts


def gen_systemic(rng, cfg):
    accounts = []
    for i in range(cfg.systemic_n):
        n = rng.integers(4, 8)
        accounts.append(
            make_account(
                rng,
                f"systemic_{i}",
                "systemic_distributed_exploit",
                True,
                4,
                rng.normal(cfg.systemic_mean, 0.035, n),
                signature="exploit_A",
                campaign_id="campaign_exploit_A",
                expected_route="product_investigation",
                intent_uncertainty=0.72 + rng.normal(0, 0.06),
                policy_uncertainty=0.30 + rng.normal(0, 0.05),
                provenance_quality=0.78 + rng.normal(0, 0.08),
            )
        )
    return accounts


def gen_policy_ambiguous(rng, cfg):
    accounts = []
    for i in range(cfg.policy_ambiguous_n):
        n = rng.integers(1, 3)
        accounts.append(
            make_account(
                rng,
                f"ambiguous_{i}",
                "policy_ambiguous",
                False,
                0,
                rng.normal(0.55, 0.05, n),
                expected_route="policy_review",
                intent_uncertainty=0.55 + rng.normal(0, 0.08),
                policy_uncertainty=0.82 + rng.normal(0, 0.06),
            )
        )
    return accounts


def gen_escalating(rng, cfg):
    accounts = []
    for i in range(cfg.escalating_n):
        n = rng.integers(6, 11)
        scores = np.linspace(0.15, 0.85, n) + rng.normal(0, 0.05, n)
        accounts.append(
            make_account(
                rng,
                f"escalating_{i}",
                "escalating",
                True,
                2,
                scores,
                expected_route="human_review",
                intent_uncertainty=0.35 + rng.normal(0, 0.05),
                policy_uncertainty=0.18 + rng.normal(0, 0.04),
            )
        )
    return accounts


def gen_uncertainty_required(rng, cfg):
    """Matched local risk; uncertainty fields disambiguate harmful intent."""
    accounts = []
    half = cfg.uncertainty_required_n // 2
    for i in range(half):
        n = rng.integers(5, 9)
        accounts.append(
            make_account(
                rng,
                f"uncertainty_harm_{i}",
                "uncertainty_required_harm",
                True,
                3,
                rng.normal(0.34, 0.045, n),
                expected_route="human_review",
                intent_uncertainty=0.82 + rng.normal(0, 0.04),
                policy_uncertainty=0.18 + rng.normal(0, 0.04),
                provenance_quality=0.86 + rng.normal(0, 0.05),
            )
        )
    for i in range(cfg.uncertainty_required_n - half):
        n = rng.integers(5, 9)
        accounts.append(
            make_account(
                rng,
                f"uncertainty_benign_{i}",
                "uncertainty_legit_ambiguous",
                False,
                0,
                rng.normal(0.34, 0.045, n),
                expected_route="policy_review",
                intent_uncertainty=0.30 + rng.normal(0, 0.05),
                policy_uncertainty=0.86 + rng.normal(0, 0.04),
                provenance_quality=0.84 + rng.normal(0, 0.05),
            )
        )
    return accounts


def gen_signature_required(rng, cfg):
    """Matched low-amplitude local risk; shared campaign state is decisive."""
    accounts = []
    for i in range(cfg.signature_required_n):
        n = rng.integers(4, 8)
        accounts.append(
            make_account(
                rng,
                f"signature_harm_{i}",
                "signature_required_campaign",
                True,
                4,
                rng.normal(0.195, 0.035, n),
                signature="signature_required_exploit",
                campaign_id="campaign_signature_required",
                expected_route="product_investigation",
                intent_uncertainty=0.66 + rng.normal(0, 0.05),
                policy_uncertainty=0.25 + rng.normal(0, 0.04),
                provenance_quality=0.82 + rng.normal(0, 0.06),
            )
        )
    # Locally similar but scattered across separate benign signatures, so only
    # signature/campaign aggregation should separate these from the campaign.
    for i in range(cfg.signature_required_n):
        n = rng.integers(4, 8)
        accounts.append(
            make_account(
                rng,
                f"signature_benign_{i}",
                "signature_matched_benign",
                False,
                0,
                rng.normal(0.195, 0.035, n),
                signature=f"benign_signature_singleton_{i}",
                expected_route="auto_handle",
                intent_uncertainty=0.64 + rng.normal(0, 0.05),
                policy_uncertainty=0.27 + rng.normal(0, 0.04),
                provenance_quality=0.82 + rng.normal(0, 0.06),
            )
        )
    return accounts


def gen_provenance_required(rng, cfg):
    """Matched risk and uncertainty; provenance quality distinguishes artifact from real harm."""
    accounts = []
    half = cfg.provenance_required_n // 2
    for i in range(half):
        n = rng.integers(4, 8)
        accounts.append(
            make_account(
                rng,
                f"provenance_harm_{i}",
                "provenance_required_harm",
                True,
                3,
                rng.normal(0.46, 0.04, n),
                expected_route="human_review",
                intent_uncertainty=0.45 + rng.normal(0, 0.05),
                policy_uncertainty=0.25 + rng.normal(0, 0.04),
                provenance_quality=0.92 + rng.normal(0, 0.03),
            )
        )
    for i in range(cfg.provenance_required_n - half):
        n = rng.integers(4, 8)
        accounts.append(
            make_account(
                rng,
                f"provenance_artifact_{i}",
                "low_provenance_artifact",
                False,
                0,
                rng.normal(0.46, 0.04, n),
                expected_route="auto_handle",
                intent_uncertainty=0.45 + rng.normal(0, 0.05),
                policy_uncertainty=0.25 + rng.normal(0, 0.04),
                provenance_quality=0.18 + rng.normal(0, 0.04),
            )
        )
    return accounts


def generate_accounts(cfg: ScenarioConfig):
    rng = np.random.default_rng(cfg.seed)
    accounts = (
        gen_benign(rng, cfg)
        + gen_false_systemic_clusters(rng, cfg)
        + gen_adversarial_false_systemic_lookalikes(rng, cfg)
        + gen_frontier_overlap_cases(rng, cfg)
        + gen_obvious(rng, cfg)
        + gen_borderline(rng, cfg)
        + gen_distributed(rng, cfg)
        + gen_systemic(rng, cfg)
        + gen_policy_ambiguous(rng, cfg)
        + gen_escalating(rng, cfg)
        + gen_uncertainty_required(rng, cfg)
        + gen_signature_required(rng, cfg)
        + gen_provenance_required(rng, cfg)
    )
    rng.shuffle(accounts)
    return accounts


def build_dataset(cfg: ScenarioConfig):
    from .features import add_features

    return add_features(pd.DataFrame(generate_accounts(cfg)), cfg)
