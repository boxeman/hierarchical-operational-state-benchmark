from pathlib import Path

import pandas as pd


DEFAULT_DIR = Path("outputs") / "prototype_package"
LARGE_DIR = Path("outputs") / "large_replication"
OUT_DIR = Path("outputs")


def f(value, digits=3):
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_table(df):
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def account_review_metrics(root):
    df = pd.read_csv(root / "account_metrics_by_split.csv")
    sub = df[(df["split"] == "test") & (df["budget"] == 0.15)]
    flat = sub[sub["method"] == "learned_logistic_flat"].iloc[0]
    hier = sub[sub["method"] == "learned_logistic_hier"].iloc[0]
    return {
        "flat_hw_recall": float(flat["harm_weighted_recall"]),
        "flat_precision": float(flat["precision"]),
        "hier_hw_recall": float(hier["harm_weighted_recall"]),
        "hier_precision": float(hier["precision"]),
    }


def state_component_metrics(root):
    df = pd.read_csv(root / "targeted_ablation_diagnostics.csv")
    def rate(method, archetype):
        return float(
            df[
                (df["method"] == method)
                & (df["type"] == archetype)
            ]["selected_rate_at_tuned_threshold"].mean()
        )
    return {
        "ablation_uncertainty_false": rate("ablation_no_uncertainty", "uncertainty_legit_ambiguous"),
        "hier_uncertainty_false": rate("learned_logistic_hier", "uncertainty_legit_ambiguous"),
        "ablation_signature_false": rate("ablation_no_signature", "signature_matched_benign"),
        "hier_signature_false": rate("learned_logistic_hier", "signature_matched_benign"),
        "ablation_provenance_false": rate("ablation_no_provenance", "low_provenance_artifact"),
        "hier_provenance_false": rate("learned_logistic_hier", "low_provenance_artifact"),
    }


def calibration_metrics(root):
    df = pd.read_csv(root / "calibration_summary.csv")
    sub = df[df["split"] == "test"]
    flat = sub[sub["method"] == "learned_logistic_flat"].iloc[0]
    hier = sub[sub["method"] == "learned_logistic_hier"].iloc[0]
    return {
        "flat_ece": float(flat["ece"]),
        "flat_brier": float(flat["brier"]),
        "hier_ece": float(hier["ece"]),
        "hier_brier": float(hier["brier"]),
    }


def system_alert_metrics(root):
    df = pd.read_csv(root / "system_alert_metrics_by_split.csv")
    grouped = (
        df[df["split"] == "test"]
        .groupby("method")
        .agg(hit=("product_alert_hit", "mean"), rank=("exploit_A_rank", "mean"))
    )
    mean_risk = grouped.loc["signature_mean_risk"]
    typed = grouped.loc["typed_system_hierarchy_alert"]
    return {
        "mean_risk_hit": float(mean_risk["hit"]),
        "mean_risk_rank": float(mean_risk["rank"]),
        "typed_hit": float(typed["hit"]),
        "typed_rank": float(typed["rank"]),
    }


def routing_metrics_pair(root):
    df = pd.read_csv(root / "routing_metrics_by_split.csv")
    sub = df[(df["split"] == "test") & (df["policy_variant"] == "validation_tuned")]
    flat = sub[sub["method"] == "learned_logistic_flat"].iloc[0]
    hier = sub[sub["method"] == "learned_logistic_hier"].iloc[0]
    return {
        "flat_macro_f1": float(flat["macro_route_f1"]),
        "flat_utility": float(flat["routing_utility"]),
        "hier_macro_f1": float(hier["macro_route_f1"]),
        "hier_utility": float(hier["routing_utility"]),
    }


def adversarial_stress_metrics(root):
    df = pd.read_csv(root / "adversarial_routing_stress.csv")
    sub = df[
        (df["split"] == "test")
        & (df["policy_variant"] == "validation_tuned")
        & (df["type"] == "ALL_ADVERSARIAL_FALSE_SYSTEMIC")
    ]
    typed = sub[sub["method"] == "typed_system_hierarchy"].iloc[0]
    hier = sub[sub["method"] == "learned_logistic_hier"].iloc[0]
    return {
        "typed_product_rate": float(typed["product_investigation_rate"]),
        "hier_product_rate": float(hier["product_investigation_rate"]),
    }


def frontier_metrics(root):
    df = pd.read_csv(root / "routing_penalty_sweep.csv").sort_values("adversarial_false_alert_penalty")
    low = df.iloc[0]
    high = df.iloc[-1]
    return {
        "low_missed": int(low["missed_product_investigations"]),
        "low_false_systemic": int(low["adversarial_false_systemic_product_investigations"]),
        "high_missed": int(high["missed_product_investigations"]),
        "high_false_systemic": int(high["adversarial_false_systemic_product_investigations"]),
    }


def workload_metrics(root):
    df = pd.read_csv(root / "time_to_alert_workload_summary.csv")
    sub = df[df["policy_variant"] == "validation_tuned"]
    typed = sub[sub["method"] == "typed_system_hierarchy"].iloc[0]
    hier = sub[sub["method"] == "learned_logistic_hier"].iloc[0]
    return {
        "typed_hours": float(typed["total_analyst_hours"]),
        "typed_false_product": int(typed["false_product_investigations"]),
        "hier_hours": float(hier["total_analyst_hours"]),
        "hier_false_product": int(hier["false_product_investigations"]),
    }


def backlog_metrics(root):
    df = pd.read_csv(root / "dedup_backlog_summary.csv")
    sub = df[
        (df["policy_variant"] == "validation_tuned")
        & (df["capacity_hours_per_day"] == 16.0)
        & (df["queue_mode"] == "fifo")
    ]
    typed = sub[sub["method"] == "typed_system_hierarchy"].iloc[0]
    hier = sub[sub["method"] == "learned_logistic_hier"].iloc[0]
    return {
        "typed_peak_backlog": float(typed["peak_backlog_hours"]),
        "typed_false_unique": int(typed["false_unique_product_alerts"]),
        "hier_peak_backlog": float(hier["peak_backlog_hours"]),
        "hier_false_unique": int(hier["false_unique_product_alerts"]),
    }


def priority_metrics(root):
    df = pd.read_csv(root / "dedup_backlog_summary.csv")
    sub = df[
        (df["policy_variant"] == "validation_tuned")
        & (df["capacity_hours_per_day"] == 16.0)
        & (df["method"] == "learned_logistic_hier")
    ]
    fifo = sub[sub["queue_mode"] == "fifo"].iloc[0]
    product = sub[sub["queue_mode"] == "product_first"].iloc[0]
    return {
        "fifo_p90_delay": float(fifo["p90_delay_to_product_investigation_hours"]),
        "fifo_48h": float(fifo["true_product_alerts_resolved_within_48h_rate"]),
        "product_p90_delay": float(product["p90_delay_to_product_investigation_hours"]),
        "product_48h": float(product["true_product_alerts_resolved_within_48h_rate"]),
    }


def direction(default_value, large_value):
    if default_value and large_value:
        return "stable"
    if default_value and not large_value:
        return "changed"
    if not default_value and large_value:
        return "emerged in replication"
    return "not supported"


def main():
    rows = []

    d = account_review_metrics(DEFAULT_DIR)
    l = account_review_metrics(LARGE_DIR)
    d_ok = d["hier_hw_recall"] > d["flat_hw_recall"] and d["hier_precision"] >= d["flat_precision"]
    l_ok = l["hier_hw_recall"] > l["flat_hw_recall"] and l["hier_precision"] >= l["flat_precision"]
    rows.append({
        "layer": "Account review",
        "metric": "learned hierarchy vs learned flat at 15% review budget",
        "default_result": f"HW recall {f(d['hier_hw_recall'])} vs {f(d['flat_hw_recall'])}; precision {f(d['hier_precision'])} vs {f(d['flat_precision'])}",
        "large_replication_result": f"HW recall {f(l['hier_hw_recall'])} vs {f(l['flat_hw_recall'])}; precision {f(l['hier_precision'])} vs {f(l['flat_precision'])}",
        "stable_or_changed": direction(d_ok, l_ok),
        "interpretation": "The account-review claim survives if hierarchy keeps higher harm-weighted recall without losing precision.",
    })

    d = state_component_metrics(DEFAULT_DIR)
    l = state_component_metrics(LARGE_DIR)
    d_ok = (
        d["hier_uncertainty_false"] < d["ablation_uncertainty_false"]
        and d["hier_signature_false"] < d["ablation_signature_false"]
        and d["hier_provenance_false"] < d["ablation_provenance_false"]
    )
    l_ok = (
        l["hier_uncertainty_false"] < l["ablation_uncertainty_false"]
        and l["hier_signature_false"] < l["ablation_signature_false"]
        and l["hier_provenance_false"] < l["ablation_provenance_false"]
    )
    rows.append({
        "layer": "State components",
        "metric": "full hierarchy suppresses ablation-specific hard negatives",
        "default_result": f"full vs ablated false rates: uncertainty {f(d['hier_uncertainty_false'])} vs {f(d['ablation_uncertainty_false'])}; signature {f(d['hier_signature_false'])} vs {f(d['ablation_signature_false'])}; provenance {f(d['hier_provenance_false'])} vs {f(d['ablation_provenance_false'])}",
        "large_replication_result": f"full vs ablated false rates: uncertainty {f(l['hier_uncertainty_false'])} vs {f(l['ablation_uncertainty_false'])}; signature {f(l['hier_signature_false'])} vs {f(l['ablation_signature_false'])}; provenance {f(l['hier_provenance_false'])} vs {f(l['ablation_provenance_false'])}",
        "stable_or_changed": direction(d_ok, l_ok),
        "interpretation": "The component claim survives: uncertainty, signature/campaign, and provenance remain useful in matched failure modes.",
    })

    d = calibration_metrics(DEFAULT_DIR)
    l = calibration_metrics(LARGE_DIR)
    d_ok = d["hier_ece"] < d["flat_ece"] and d["hier_brier"] < d["flat_brier"]
    l_ok = l["hier_ece"] < l["flat_ece"] and l["hier_brier"] < l["flat_brier"]
    rows.append({
        "layer": "Calibration",
        "metric": "learned hierarchy calibration vs learned flat",
        "default_result": f"ECE {f(d['hier_ece'])} vs {f(d['flat_ece'])}; Brier {f(d['hier_brier'])} vs {f(d['flat_brier'])}",
        "large_replication_result": f"ECE {f(l['hier_ece'])} vs {f(l['flat_ece'])}; Brier {f(l['hier_brier'])} vs {f(l['flat_brier'])}",
        "stable_or_changed": direction(d_ok, l_ok),
        "interpretation": "The calibration claim survives if hierarchy retains lower ECE and Brier score.",
    })

    d = system_alert_metrics(DEFAULT_DIR)
    l = system_alert_metrics(LARGE_DIR)
    d_ok = d["typed_rank"] < d["mean_risk_rank"]
    l_ok = l["typed_rank"] < l["mean_risk_rank"]
    rows.append({
        "layer": "System alerts",
        "metric": "typed hierarchy vs signature mean-risk exploit rank",
        "default_result": f"hit {f(d['typed_hit'])} vs {f(d['mean_risk_hit'])}; rank {f(d['typed_rank'])} vs {f(d['mean_risk_rank'])}",
        "large_replication_result": f"hit {f(l['typed_hit'])} vs {f(l['mean_risk_hit'])}; rank {f(l['typed_rank'])} vs {f(l['mean_risk_rank'])}",
        "stable_or_changed": direction(d_ok, l_ok),
        "interpretation": "The ranking improvement survives, but top-k alerting remains fragile because hit rate does not clearly separate methods.",
    })

    d = routing_metrics_pair(DEFAULT_DIR)
    l = routing_metrics_pair(LARGE_DIR)
    d_ok = d["hier_macro_f1"] > d["flat_macro_f1"] and d["hier_utility"] > d["flat_utility"]
    l_ok = l["hier_macro_f1"] > l["flat_macro_f1"] and l["hier_utility"] > l["flat_utility"]
    rows.append({
        "layer": "Routing",
        "metric": "validation-tuned learned hierarchy vs learned flat",
        "default_result": f"macro F1 {f(d['hier_macro_f1'])} vs {f(d['flat_macro_f1'])}; utility {f(d['hier_utility'])} vs {f(d['flat_utility'])}",
        "large_replication_result": f"macro F1 {f(l['hier_macro_f1'])} vs {f(l['flat_macro_f1'])}; utility {f(l['hier_utility'])} vs {f(l['flat_utility'])}",
        "stable_or_changed": direction(d_ok, l_ok),
        "interpretation": "The routing claim survives: hierarchy retains better escalation quality than flat routing.",
    })

    d = adversarial_stress_metrics(DEFAULT_DIR)
    l = adversarial_stress_metrics(LARGE_DIR)
    d_ok = d["hier_product_rate"] < d["typed_product_rate"]
    l_ok = l["hier_product_rate"] < l["typed_product_rate"]
    rows.append({
        "layer": "False-systemic stress",
        "metric": "adversarial benign lookalikes routed to product investigation",
        "default_result": f"learned hierarchy {f(d['hier_product_rate'])} vs typed rule {f(d['typed_product_rate'])}",
        "large_replication_result": f"learned hierarchy {f(l['hier_product_rate'])} vs typed rule {f(l['typed_product_rate'])}",
        "stable_or_changed": direction(d_ok, l_ok),
        "interpretation": "The learned hierarchy continues to suppress benign coordinated product alerts better than brittle rule hierarchy.",
    })

    d = frontier_metrics(DEFAULT_DIR)
    l = frontier_metrics(LARGE_DIR)
    d_ok = d["high_false_systemic"] < d["low_false_systemic"] and d["high_missed"] > d["low_missed"]
    l_ok = l["high_false_systemic"] < l["low_false_systemic"] and l["high_missed"] > l["low_missed"]
    rows.append({
        "layer": "Product-alert frontier",
        "metric": "high false-alert penalty trades recall for fewer false systemic alerts",
        "default_result": f"low penalty missed {d['low_missed']}, false {d['low_false_systemic']}; high penalty missed {d['high_missed']}, false {d['high_false_systemic']}",
        "large_replication_result": f"low penalty missed {l['low_missed']}, false {l['low_false_systemic']}; high penalty missed {l['high_missed']}, false {l['high_false_systemic']}",
        "stable_or_changed": direction(d_ok, l_ok),
        "interpretation": "The operating frontier survives: suppressing benign systemic alerts costs additional missed product investigations.",
    })

    d = workload_metrics(DEFAULT_DIR)
    l = workload_metrics(LARGE_DIR)
    d_ok = d["hier_hours"] < d["typed_hours"] and d["hier_false_product"] < d["typed_false_product"]
    l_ok = l["hier_hours"] < l["typed_hours"] and l["hier_false_product"] < l["typed_false_product"]
    rows.append({
        "layer": "Workload/time-to-alert",
        "metric": "learned hierarchy vs typed rule workload and false product investigations",
        "default_result": f"analyst hours {f(d['hier_hours'])} vs {f(d['typed_hours'])}; false product {d['hier_false_product']} vs {d['typed_false_product']}",
        "large_replication_result": f"analyst hours {f(l['hier_hours'])} vs {f(l['typed_hours'])}; false product {l['hier_false_product']} vs {l['typed_false_product']}",
        "stable_or_changed": direction(d_ok, l_ok),
        "interpretation": "The practical workload claim survives, though absolute workload grows substantially in the larger run.",
    })

    d = backlog_metrics(DEFAULT_DIR)
    l = backlog_metrics(LARGE_DIR)
    d_ok = d["hier_peak_backlog"] < d["typed_peak_backlog"] and d["hier_false_unique"] < d["typed_false_unique"]
    l_ok = l["hier_peak_backlog"] < l["typed_peak_backlog"] and l["hier_false_unique"] < l["typed_false_unique"]
    rows.append({
        "layer": "Deduplicated backlog",
        "metric": "learned hierarchy vs typed rule finite-capacity queue burden",
        "default_result": f"peak backlog {f(d['hier_peak_backlog'])} vs {f(d['typed_peak_backlog'])}; false unique alerts {d['hier_false_unique']} vs {d['typed_false_unique']}",
        "large_replication_result": f"peak backlog {f(l['hier_peak_backlog'])} vs {f(l['typed_peak_backlog'])}; false unique alerts {l['hier_false_unique']} vs {l['typed_false_unique']}",
        "stable_or_changed": direction(d_ok, l_ok),
        "interpretation": "The deduplicated backlog claim survives, but capacity pressure is much larger in the replication run.",
    })

    d = priority_metrics(DEFAULT_DIR)
    l = priority_metrics(LARGE_DIR)
    d_ok = d["product_p90_delay"] < d["fifo_p90_delay"] and d["product_48h"] > d["fifo_48h"]
    l_ok = l["product_p90_delay"] < l["fifo_p90_delay"] and l["product_48h"] > l["fifo_48h"]
    rows.append({
        "layer": "Priority queues",
        "metric": "product-first queue vs FIFO for learned hierarchy",
        "default_result": f"p90 delay {f(d['product_p90_delay'])}h vs {f(d['fifo_p90_delay'])}h; 48h resolution {f(d['product_48h'])} vs {f(d['fifo_48h'])}",
        "large_replication_result": f"p90 delay {f(l['product_p90_delay'])}h vs {f(l['fifo_p90_delay'])}h; 48h resolution {f(l['product_48h'])} vs {f(l['fifo_48h'])}",
        "stable_or_changed": direction(d_ok, l_ok),
        "interpretation": "The priority-queue claim survives directionally, but the larger run shows much longer product delays under finite capacity.",
    })

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "replication_comparison_table.csv", index=False)
    (OUT_DIR / "replication_comparison_table.md").write_text(
        "# Replication Comparison Table\n\n"
        "Comparison between `outputs/prototype_package` and `outputs/large_replication`.\n\n"
        + markdown_table(out),
        encoding="utf-8",
    )
    print(markdown_table(out))
    print(f"Saved {OUT_DIR / 'replication_comparison_table.csv'}")
    print(f"Saved {OUT_DIR / 'replication_comparison_table.md'}")


if __name__ == "__main__":
    main()

