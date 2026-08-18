import numpy as np


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def topk_mean(scores, k=3):
    scores = np.sort(np.asarray(scores, dtype=float))
    return float(np.mean(scores[-min(k, len(scores)) :]))


def add_features(df, cfg):
    rows = []
    for _, row in df.iterrows():
        scores = np.asarray(row["scores"], dtype=float)
        max_s = float(np.max(scores))
        mean_s = float(np.mean(scores))
        count_sub = int(np.sum(scores > cfg.sub_thresh))
        evidence_volume = min(count_sub / cfg.evidence_norm, 1.0)
        trend = 0.0
        if len(scores) >= 2:
            trend = float(clip01(np.polyfit(np.arange(len(scores)), scores, 1)[0] * len(scores) * 2))
        amplitude = float(clip01((mean_s - 0.14) / 0.10))
        evidence_strength = evidence_volume * amplitude
        account_rule = max(max_s, 0.35 * max_s + 0.50 * evidence_volume + 0.15 * trend)
        logistic_rule = clip01(
            0.45 * max_s
            + 0.20 * topk_mean(scores)
            + 0.25 * evidence_volume
            + 0.15 * trend
            + 0.08 * row["intent_uncertainty"]
            - 0.05 * row["policy_uncertainty"]
        )
        out = row.to_dict()
        out.update(
            flat_max=max_s,
            flat_mean=mean_s,
            flat_top3_mean=topk_mean(scores),
            flat_count_above=float(evidence_volume),
            evidence_count=float(count_sub),
            evidence_volume=float(evidence_volume),
            trend=float(trend),
            account_rule=float(account_rule),
            account_logistic_rule=float(logistic_rule),
            account_evidence_strength=float(evidence_strength),
        )
        rows.append(out)
    out_df = df.__class__(rows)

    sig = (
        out_df.dropna(subset=["signature"])
        .groupby("signature")
        .agg(
            signature_count=("id", "count"),
            signature_mean_risk=("flat_mean", "mean"),
            signature_max_risk=("flat_max", "max"),
            signature_mean_evidence=("account_evidence_strength", "mean"),
            signature_mean_intent_uncertainty=("intent_uncertainty", "mean"),
            signature_mean_policy_uncertainty=("policy_uncertainty", "mean"),
            signature_mean_provenance=("provenance_quality", "mean"),
            signature_mean_legitimacy=("legitimacy_context", "mean"),
            signature_harmful_count=("harmful", "sum"),
            archetypes=("type", lambda s: ",".join(sorted(set(s)))),
        )
        .reset_index()
    )

    if sig.empty:
        for col in [
            "signature_count",
            "signature_mean_risk",
            "signature_mean_evidence",
            "signature_confidence",
            "signature_mean_legitimacy",
            "system_score",
            "typed_system_score",
        ]:
            out_df[col] = 0.0
        return out_df, sig

    sig["signature_confidence"] = clip01(sig["signature_count"] / cfg.cluster_norm)
    sig["signature_count_only"] = sig["signature_confidence"]
    sig["signature_mean_risk_detector"] = (
        0.55 * sig["signature_mean_risk"] + 0.45 * sig["signature_confidence"]
    )
    sig["system_alert_score"] = (
        sig["signature_confidence"]
        * sig["signature_mean_evidence"]
        * (0.55 + 0.45 * sig["signature_max_risk"])
    )
    sig["typed_system_alert_score"] = clip01(
        sig["system_alert_score"]
        + 0.10 * sig["signature_mean_intent_uncertainty"]
        - 0.08 * sig["signature_mean_policy_uncertainty"]
        + 0.06 * sig["signature_mean_provenance"]
        - 0.06 * sig["signature_mean_legitimacy"]
    )

    merge_cols = [
        "signature",
        "signature_count",
        "signature_mean_risk",
        "signature_mean_evidence",
        "signature_confidence",
        "system_alert_score",
        "typed_system_alert_score",
        "signature_mean_legitimacy",
    ]
    out_df = out_df.merge(sig[merge_cols], how="left", on="signature")
    for col in merge_cols[1:]:
        out_df[col] = out_df[col].fillna(0.0)

    cluster_boost = 0.38 * out_df["signature_confidence"] * out_df["signature_mean_evidence"]
    out_df["system_score"] = clip01(out_df["account_rule"] + cluster_boost)
    out_df["typed_system_score"] = clip01(
        out_df["system_score"]
        + 0.04 * out_df["intent_uncertainty"]
        - 0.04 * out_df["policy_uncertainty"]
        + 0.03 * out_df["provenance_quality"]
        - 0.03 * out_df["legitimacy_context"]
    )
    return out_df, sig
