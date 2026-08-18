import numpy as np
import pandas as pd

from .data import ADVERSARIAL_FALSE_SYSTEMIC_TYPES, FALSE_SYSTEMIC_TYPES


ROUTES = ["auto_handle", "human_review", "policy_review", "product_investigation"]
ROUTE_LEVEL = {
    "auto_handle": 0,
    "policy_review": 1,
    "human_review": 2,
    "product_investigation": 3,
}
DEFAULT_ROUTING_POLICY = {
    "human_threshold": None,
    "policy_uncertainty_threshold": 0.70,
    "product_signature_confidence_threshold": 0.45,
    "product_evidence_threshold": 0.12,
    "provenance_suppression_threshold": 0.35,
    "product_score_threshold": 0.30,
    "legitimacy_suppression_threshold": 1.10,
}


def account_metrics(df, score, budget=0.15):
    k = max(1, int(len(df) * budget))
    tmp = df.copy()
    tmp["_score"] = score
    reviewed = tmp.sort_values("_score", ascending=False).head(k)
    harm_total = tmp.loc[tmp["harmful"], "harm_weight"].sum()
    harm_caught = reviewed.loc[reviewed["harmful"], "harm_weight"].sum()
    systemic_total = max(1, int((tmp["type"] == "systemic_distributed_exploit").sum()))
    return {
        "budget": budget,
        "k": k,
        "precision": float(reviewed["harmful"].mean()),
        "harm_weighted_recall": float(harm_caught / harm_total),
        "harm_per_100_reviews": float(100 * harm_caught / k),
        "systemic_account_recall": float(
            (reviewed["type"] == "systemic_distributed_exploit").sum() / systemic_total
        ),
        "false_positive_reviews": int((~reviewed["harmful"]).sum()),
        "policy_ambiguous_flagged": int((reviewed["type"] == "policy_ambiguous").sum()),
        "false_systemic_accounts_flagged": int(reviewed["type"].isin(FALSE_SYSTEMIC_TYPES).sum()),
    }


def route_capabilities(method):
    flat_methods = {"flat_max", "flat_mean", "flat_top3_mean", "flat_count_above", "learned_logistic_flat", "boosted_stumps_flat"}
    return {
        "uncertainty": method not in flat_methods and method != "ablation_no_uncertainty",
        "signature": method not in flat_methods and method != "ablation_no_signature",
        "provenance": method not in flat_methods and method != "ablation_no_provenance",
    }


def predict_routes(df, score, threshold, method, policy=None):
    """Map model scores and available operational state to escalation routes."""
    tmp = df.copy()
    tmp["_score"] = np.asarray(score, dtype=float)
    routing_policy = dict(DEFAULT_ROUTING_POLICY)
    if policy:
        routing_policy.update({k: v for k, v in policy.items() if v is not None})
    human_threshold = routing_policy["human_threshold"]
    if human_threshold is None:
        human_threshold = threshold
    caps = route_capabilities(method)
    selected = tmp["_score"] >= human_threshold
    routes = pd.Series("auto_handle", index=tmp.index, dtype=object)

    if caps["uncertainty"]:
        policy_mask = (
            (tmp["policy_uncertainty"] >= routing_policy["policy_uncertainty_threshold"])
            & ((tmp["_score"] >= min(human_threshold, 0.25)) | (tmp["flat_max"] >= 0.45))
        )
        routes.loc[policy_mask] = "policy_review"

    low_provenance = pd.Series(False, index=tmp.index)
    if caps["provenance"]:
        low_provenance = tmp["provenance_quality"] < routing_policy["provenance_suppression_threshold"]

    human_mask = selected & ~low_provenance & (routes == "auto_handle")
    routes.loc[human_mask] = "human_review"

    if caps["signature"]:
        legitimacy_suppressed = (
            tmp["legitimacy_context"] >= routing_policy["legitimacy_suppression_threshold"]
        )
        product_mask = (
            selected
            & ~low_provenance
            & ~legitimacy_suppressed
            & (tmp["signature_confidence"] >= routing_policy["product_signature_confidence_threshold"])
            & (tmp["signature_mean_evidence"] >= routing_policy["product_evidence_threshold"])
            & (tmp["typed_system_score"] >= routing_policy["product_score_threshold"])
        )
        if caps["uncertainty"]:
            product_mask = product_mask & (
                tmp["policy_uncertainty"] < routing_policy["policy_uncertainty_threshold"]
            )
        routes.loc[product_mask] = "product_investigation"
    return routes.to_numpy()


def fixed_routing_policy(threshold):
    policy = dict(DEFAULT_ROUTING_POLICY)
    policy["human_threshold"] = float(threshold)
    return policy


def routing_utility(metrics, adversarial_false_alert_penalty=0.0025):
    # Use rate-like terms where possible and count penalties normalized by a
    # stable proxy. Counts remain in metrics for interpretability.
    return float(
        1.40 * metrics["macro_route_f1"]
        + 0.50 * metrics["route_accuracy"]
        + 0.35 * _safe_metric(metrics["product_investigation_recall"])
        + 0.20 * _safe_metric(metrics["policy_review_recall"])
        - 0.75 * metrics["under_escalation_rate"]
        - 0.35 * metrics["over_escalation_rate"]
        - 0.0008 * metrics["missed_product_investigations"]
        - 0.0005 * metrics["unnecessary_product_investigations"]
        - adversarial_false_alert_penalty * metrics.get(
            "adversarial_false_systemic_product_investigations", 0
        )
        - 0.0002 * metrics["unnecessary_human_reviews"]
        - 0.0002 * metrics["missed_policy_reviews"]
    )


def tune_routing_policy(
    df,
    score,
    method,
    base_threshold,
    adversarial_false_alert_penalty=0.0025,
    wide_grid=False,
):
    best = None
    for candidate in routing_policy_candidates(
        df,
        score,
        method,
        base_threshold,
        wide_grid=wide_grid,
    ):
        utility = routing_utility(
            candidate,
            adversarial_false_alert_penalty=adversarial_false_alert_penalty,
        )
        scored = dict(candidate)
        scored["adversarial_false_alert_penalty"] = float(adversarial_false_alert_penalty)
        scored["routing_utility"] = utility
        if best is None or utility > best["routing_utility"]:
            best = scored
    return best


def routing_policy_candidates(df, score, method, base_threshold, wide_grid=False):
    caps = route_capabilities(method)
    score = np.asarray(score, dtype=float)
    human_thresholds = sorted(
        set(
            float(x)
            for x in np.r_[
                np.quantile(score, [0.25, 0.38, 0.50]),
                base_threshold,
            ]
        )
    )
    if wide_grid:
        policy_uncertainty_thresholds = [0.60, 0.75] if caps["uncertainty"] else [0.70]
        product_signature_thresholds = [0.35] if caps["signature"] else [0.45]
        product_evidence_thresholds = [0.16] if caps["signature"] else [0.12]
        product_score_thresholds = [0.30, 0.45, 0.60] if caps["signature"] else [0.30]
        legitimacy_thresholds = [0.45, 0.55, 0.65, 0.75, 1.10] if caps["signature"] else [1.10]
    else:
        policy_uncertainty_thresholds = [0.65, 0.75] if caps["uncertainty"] else [0.70]
        product_signature_thresholds = [0.35, 0.55] if caps["signature"] else [0.45]
        product_evidence_thresholds = [0.10, 0.16] if caps["signature"] else [0.12]
        product_score_thresholds = [0.30, 0.45, 0.60] if caps["signature"] else [0.30]
        legitimacy_thresholds = [1.10] if caps["signature"] else [1.10]
    provenance_thresholds = [0.30, 0.40] if caps["provenance"] else [0.35]

    candidates = []
    for human_threshold in human_thresholds:
        for policy_threshold in policy_uncertainty_thresholds:
            for signature_threshold in product_signature_thresholds:
                for evidence_threshold in product_evidence_thresholds:
                    for product_score_threshold in product_score_thresholds:
                        for provenance_threshold in provenance_thresholds:
                            for legitimacy_threshold in legitimacy_thresholds:
                                policy = {
                                    "human_threshold": float(human_threshold),
                                    "policy_uncertainty_threshold": float(policy_threshold),
                                    "product_signature_confidence_threshold": float(signature_threshold),
                                    "product_evidence_threshold": float(evidence_threshold),
                                    "provenance_suppression_threshold": float(provenance_threshold),
                                    "product_score_threshold": float(product_score_threshold),
                                    "legitimacy_suppression_threshold": float(legitimacy_threshold),
                                }
                                routes = predict_routes(df, score, human_threshold, method, policy=policy)
                                metrics = routing_metrics(df, routes)
                                candidate = dict(policy)
                                candidate.update(metrics)
                                candidates.append(candidate)
    return candidates


def _safe_metric(value):
    return 0.0 if pd.isna(value) else float(value)


def routing_metrics(df, predicted_routes):
    tmp = df.copy()
    tmp["_predicted_route"] = np.asarray(predicted_routes, dtype=object)
    expected = tmp["expected_route"].astype(str)
    predicted = tmp["_predicted_route"].astype(str)
    total = max(1, len(tmp))
    correct = expected == predicted

    rows = {
        "n_accounts": int(total),
        "route_accuracy": float(correct.mean()),
        "wrong_escalation_level": int((~correct).sum()),
        "unnecessary_human_reviews": int(((predicted == "human_review") & (expected == "auto_handle")).sum()),
        "unnecessary_policy_reviews": int(((predicted == "policy_review") & (expected == "auto_handle")).sum()),
        "unnecessary_product_investigations": int(
            ((predicted == "product_investigation") & (expected != "product_investigation")).sum()
        ),
        "adversarial_false_systemic_product_investigations": int(
            (
                (predicted == "product_investigation")
                & tmp["type"].isin(ADVERSARIAL_FALSE_SYSTEMIC_TYPES)
            ).sum()
        ),
        "missed_human_reviews": int(((expected == "human_review") & (predicted != "human_review")).sum()),
        "missed_policy_reviews": int(((expected == "policy_review") & (predicted != "policy_review")).sum()),
        "missed_product_investigations": int(
            ((expected == "product_investigation") & (predicted != "product_investigation")).sum()
        ),
        "auto_handle_precision": _route_precision(expected, predicted, "auto_handle"),
        "human_review_precision": _route_precision(expected, predicted, "human_review"),
        "policy_review_precision": _route_precision(expected, predicted, "policy_review"),
        "product_investigation_precision": _route_precision(expected, predicted, "product_investigation"),
        "auto_handle_recall": _route_recall(expected, predicted, "auto_handle"),
        "human_review_recall": _route_recall(expected, predicted, "human_review"),
        "policy_review_recall": _route_recall(expected, predicted, "policy_review"),
        "product_investigation_recall": _route_recall(expected, predicted, "product_investigation"),
        "macro_route_f1": _macro_route_f1(expected, predicted),
    }

    expected_level = expected.map(ROUTE_LEVEL)
    predicted_level = predicted.map(ROUTE_LEVEL)
    rows["over_escalation_rate"] = float((predicted_level > expected_level).sum() / total)
    rows["under_escalation_rate"] = float((predicted_level < expected_level).sum() / total)
    return rows


def routing_confusion(df, predicted_routes):
    expected = df["expected_route"].astype(str)
    predicted = pd.Series(np.asarray(predicted_routes, dtype=object), index=df.index)
    rows = []
    for exp in ROUTES:
        for pred in ROUTES:
            rows.append(
                {
                    "expected_route": exp,
                    "predicted_route": pred,
                    "count": int(((expected == exp) & (predicted == pred)).sum()),
                }
            )
    return pd.DataFrame(rows)


def _route_precision(expected, predicted, route):
    denom = int((predicted == route).sum())
    if denom == 0:
        return np.nan
    return float(((expected == route) & (predicted == route)).sum() / denom)


def _route_recall(expected, predicted, route):
    denom = int((expected == route).sum())
    if denom == 0:
        return np.nan
    return float(((expected == route) & (predicted == route)).sum() / denom)


def _macro_route_f1(expected, predicted):
    f1s = []
    for route in ROUTES:
        p = _route_precision(expected, predicted, route)
        r = _route_recall(expected, predicted, route)
        if np.isnan(p) or np.isnan(r) or (p + r) == 0:
            continue
        f1s.append(2 * p * r / (p + r))
    return float(np.mean(f1s)) if f1s else 0.0


def threshold_metrics(df, score, threshold):
    tmp = df.copy()
    tmp["_score"] = score
    selected = tmp[tmp["_score"] >= threshold]
    harm_total = tmp.loc[tmp["harmful"], "harm_weight"].sum()
    harm_caught = selected.loc[selected["harmful"], "harm_weight"].sum()
    systemic_total = max(1, int((tmp["type"] == "systemic_distributed_exploit").sum()))
    fp = int((~selected["harmful"]).sum())
    negatives = max(1, int((~tmp["harmful"]).sum()))
    precision = float(selected["harmful"].mean()) if len(selected) else 0.0
    recall = float(selected["harmful"].sum() / max(1, tmp["harmful"].sum()))
    return {
        "threshold": float(threshold),
        "selected": int(len(selected)),
        "selection_rate": float(len(selected) / len(tmp)),
        "precision": precision,
        "recall": recall,
        "harm_weighted_recall": float(harm_caught / harm_total),
        "harm_per_100_reviews": float(100 * harm_caught / max(1, len(selected))),
        "false_positive_reviews": fp,
        "false_positive_rate": float(fp / negatives),
        "policy_ambiguous_flagged": int((selected["type"] == "policy_ambiguous").sum()),
        "false_systemic_accounts_flagged": int(selected["type"].isin(FALSE_SYSTEMIC_TYPES).sum()),
        "systemic_account_recall": float(
            (selected["type"] == "systemic_distributed_exploit").sum() / systemic_total
        ),
    }


def tune_threshold(df, score, min_precision=0.90):
    thresholds = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 97)))
    best = None
    for threshold in thresholds:
        m = threshold_metrics(df, score, threshold)
        if m["selected"] == 0:
            continue
        precision_penalty = max(0.0, min_precision - m["precision"])
        utility = (
            m["harm_weighted_recall"]
            + 0.05 * m["systemic_account_recall"]
            - 0.20 * m["false_positive_rate"]
            - 0.30 * precision_penalty
        )
        m["validation_utility"] = float(utility)
        if best is None or utility > best["validation_utility"]:
            best = m
    return best if best is not None else threshold_metrics(df, score, 1.0)


def calibration_curve(df, score, n_bins=10):
    tmp = pd.DataFrame({"score": np.asarray(score, dtype=float), "label": df["harmful"].astype(float)})
    tmp["score"] = tmp["score"].clip(0, 1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            sub = tmp[(tmp["score"] >= lo) & (tmp["score"] <= hi)]
        else:
            sub = tmp[(tmp["score"] >= lo) & (tmp["score"] < hi)]
        if len(sub) == 0:
            rows.append(
                {
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "n": 0,
                    "mean_score": np.nan,
                    "empirical_rate": np.nan,
                    "abs_gap": np.nan,
                }
            )
            continue
        mean_score = float(sub["score"].mean())
        empirical = float(sub["label"].mean())
        gap = abs(mean_score - empirical)
        ece += len(sub) / len(tmp) * gap
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": int(len(sub)),
                "mean_score": mean_score,
                "empirical_rate": empirical,
                "abs_gap": float(gap),
            }
        )
    curve = pd.DataFrame(rows)
    brier = float(np.mean((tmp["score"] - tmp["label"]) ** 2))
    return curve, {"ece": float(ece), "brier": brier}


def recall_by_type(df, score, budget=0.15):
    k = max(1, int(len(df) * budget))
    tmp = df.copy()
    tmp["_score"] = score
    reviewed = tmp.sort_values("_score", ascending=False).head(k)
    rows = []
    for archetype in sorted(tmp["type"].unique()):
        sub = tmp[tmp["type"] == archetype]
        if not bool(sub["harmful"].iloc[0]):
            continue
        rows.append(
            {
                "type": archetype,
                "reviewed": int((reviewed["type"] == archetype).sum()),
                "total": int(len(sub)),
                "recall": float((reviewed["type"] == archetype).sum() / len(sub)),
            }
        )
    return pd.DataFrame(rows)


def system_alert_metrics(sig, score_col, alert_budget=3):
    if sig.empty:
        return {
            "product_alert_hit": 0,
            "missed_product_alert": 1,
            "false_systemic_alerts": 0,
            "exploit_A_rank": np.nan,
        }
    ranked = sig.sort_values(score_col, ascending=False).reset_index(drop=True)
    alerts = ranked.head(alert_budget)
    hit = int((alerts["signature"] == "exploit_A").any())
    false_alerts = int(
        alerts["archetypes"].apply(lambda s: all(t in FALSE_SYSTEMIC_TYPES for t in s.split(","))).sum()
    )
    adversarial_false_alerts = int(
        alerts["archetypes"].apply(
            lambda s: all(t in ADVERSARIAL_FALSE_SYSTEMIC_TYPES for t in s.split(","))
        ).sum()
    )
    exploit_rank = np.nan
    if "exploit_A" in set(ranked["signature"]):
        exploit_rank = int(ranked.index[ranked["signature"] == "exploit_A"][0] + 1)
    return {
        "product_alert_hit": hit,
        "missed_product_alert": 1 - hit,
        "false_systemic_alerts": false_alerts,
        "adversarial_false_systemic_alerts": adversarial_false_alerts,
        "exploit_A_rank": exploit_rank,
    }
