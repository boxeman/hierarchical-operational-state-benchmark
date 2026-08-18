from pathlib import Path

import numpy as np
import pandas as pd

from .data import ADVERSARIAL_FALSE_SYSTEMIC_TYPES
from .evaluation import ROUTES, ROUTE_LEVEL, account_metrics, routing_metrics, routing_utility


MANUSCRIPT_METHODS = [
    "flat_max",
    "learned_logistic_flat",
    "learned_logistic_hier",
    "ablation_no_uncertainty",
    "ablation_no_signature",
    "ablation_no_provenance",
]

METHOD_LABELS = {
    "flat_max": "Flat max",
    "learned_logistic_flat": "Learned flat",
    "learned_logistic_hier": "Learned hierarchy",
    "ablation_no_uncertainty": "No uncertainty",
    "ablation_no_signature": "No signature",
    "ablation_no_provenance": "No provenance",
    "typed_system_hierarchy_alert": "Typed system hierarchy",
    "system_hierarchy_alert": "System hierarchy",
    "signature_mean_risk": "Signature mean risk",
    "signature_count_only": "Signature count only",
}


def ci_bounds(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    return (
        float(np.mean(values)),
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    )


def bootstrap_account_cis(score_df, methods, budget=0.15, n_boot=300, seed=910):
    rng = np.random.default_rng(seed)
    scenarios = np.array(sorted(score_df["scenario_id"].unique()))
    rows = []
    by_scenario = {sid: score_df[score_df["scenario_id"] == sid] for sid in scenarios}
    for method in methods:
        boot = {
            "precision": [],
            "harm_weighted_recall": [],
            "harm_per_100_reviews": [],
            "systemic_account_recall": [],
            "false_positive_reviews": [],
            "false_systemic_accounts_flagged": [],
        }
        for _ in range(n_boot):
            sampled = rng.choice(scenarios, size=len(scenarios), replace=True)
            sample_df = pd.concat([by_scenario[sid] for sid in sampled], ignore_index=True)
            m = account_metrics(sample_df, sample_df[method].to_numpy(dtype=float), budget=budget)
            for key in boot:
                boot[key].append(m[key])
        for metric, values in boot.items():
            mean, lo, hi = ci_bounds(values)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "budget": budget,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_boot": n_boot,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_system_alert_cis(alert_df, methods, n_boot=300, seed=911):
    rng = np.random.default_rng(seed)
    test = alert_df[alert_df["split"] == "test"].copy()
    scenarios = np.array(sorted(test["scenario_id"].unique()))
    rows = []
    metrics = ["product_alert_hit", "false_systemic_alerts", "exploit_A_rank"]
    for method in methods:
        method_df = test[test["method"] == method]
        boot = {metric: [] for metric in metrics}
        for _ in range(n_boot):
            sampled = rng.choice(scenarios, size=len(scenarios), replace=True)
            sample_df = pd.concat(
                [method_df[method_df["scenario_id"] == sid] for sid in sampled],
                ignore_index=True,
            )
            for metric in metrics:
                boot[metric].append(float(sample_df[metric].mean()))
        for metric, values in boot.items():
            mean, lo, hi = ci_bounds(values)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_boot": n_boot,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_targeted_cis(targeted_scenario_df, n_boot=300, seed=912):
    rng = np.random.default_rng(seed)
    scenarios = np.array(sorted(targeted_scenario_df["scenario_id"].unique()))
    rows = []
    grouped = targeted_scenario_df.groupby(["method", "type"], sort=True)
    for (method, archetype), group in grouped:
        boot = []
        for _ in range(n_boot):
            sampled = rng.choice(scenarios, size=len(scenarios), replace=True)
            sample = pd.concat(
                [group[group["scenario_id"] == sid] for sid in sampled],
                ignore_index=True,
            )
            if len(sample):
                boot.append(float(sample["selected_rate_at_tuned_threshold"].mean()))
        mean, lo, hi = ci_bounds(boot)
        rows.append(
            {
                "method": method,
                "type": archetype,
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "n_boot": n_boot,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_routing_cis(routing_accounts_df, n_boot=300, seed=913):
    rng = np.random.default_rng(seed)
    scenarios = np.array(sorted(routing_accounts_df["scenario_id"].unique()))
    rows = []
    metrics = [
        "macro_route_f1",
        "route_accuracy",
        "routing_utility",
        "missed_product_investigations",
        "unnecessary_product_investigations",
        "adversarial_false_systemic_product_investigations",
        "over_escalation_rate",
        "under_escalation_rate",
    ]
    grouped = routing_accounts_df.groupby(["method", "policy_variant"], sort=True)
    for (method, policy_variant), group in grouped:
        by_scenario = {}
        for sid in scenarios:
            sub = group[group["scenario_id"] == sid]
            counts = (
                sub.groupby(["expected_route", "predicted_route"])
                .size()
                .reindex(pd.MultiIndex.from_product([ROUTES, ROUTES]), fill_value=0)
                .to_numpy(dtype=int)
                .reshape(len(ROUTES), len(ROUTES))
            )
            adversarial_false_product = int(
                (
                    (sub["predicted_route"] == "product_investigation")
                    & sub["type"].isin(ADVERSARIAL_FALSE_SYSTEMIC_TYPES)
                ).sum()
            )
            by_scenario[sid] = (counts, adversarial_false_product)
        boot = {metric: [] for metric in metrics}
        for _ in range(n_boot):
            sampled = rng.choice(scenarios, size=len(scenarios), replace=True)
            counts = np.sum([by_scenario[sid][0] for sid in sampled], axis=0)
            adversarial_false_product = int(np.sum([by_scenario[sid][1] for sid in sampled]))
            m = _routing_metrics_from_counts(
                counts,
                adversarial_false_systemic_product_investigations=adversarial_false_product,
            )
            m["routing_utility"] = routing_utility(m)
            for metric in metrics:
                boot[metric].append(m[metric])
        for metric, values in boot.items():
            mean, lo, hi = ci_bounds(values)
            rows.append(
                {
                    "method": method,
                    "policy_variant": policy_variant,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_boot": n_boot,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_routing_frontier_cis(frontier_accounts_df, n_boot=300, seed=914):
    rng = np.random.default_rng(seed)
    scenarios = np.array(sorted(frontier_accounts_df["scenario_id"].unique()))
    rows = []
    metrics = [
        "missed_product_investigations",
        "adversarial_false_systemic_product_investigations",
        "product_investigation_precision",
        "product_investigation_recall",
        "macro_route_f1",
        "routing_utility",
    ]
    grouped = frontier_accounts_df.groupby(
        ["method", "adversarial_false_alert_penalty"], sort=True
    )
    for (method, penalty), group in grouped:
        by_scenario = {sid: group[group["scenario_id"] == sid] for sid in scenarios}
        boot = {metric: [] for metric in metrics}
        for _ in range(n_boot):
            sampled = rng.choice(scenarios, size=len(scenarios), replace=True)
            sample = pd.concat([by_scenario[sid] for sid in sampled], ignore_index=True)
            m = routing_metrics(sample, sample["predicted_route"].to_numpy(dtype=object))
            m["routing_utility"] = routing_utility(
                m,
                adversarial_false_alert_penalty=float(penalty),
            )
            for metric in metrics:
                boot[metric].append(m[metric])
        for metric, values in boot.items():
            mean, lo, hi = ci_bounds(values)
            rows.append(
                {
                    "method": method,
                    "adversarial_false_alert_penalty": float(penalty),
                    "metric": metric,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_boot": n_boot,
                }
            )
    return pd.DataFrame(rows)


def _routing_metrics_from_counts(counts, adversarial_false_systemic_product_investigations=0):
    total = max(1, int(counts.sum()))
    correct = int(np.trace(counts))
    route_index = {route: i for i, route in enumerate(ROUTES)}

    def cell(expected, predicted):
        return int(counts[route_index[expected], route_index[predicted]])

    expected_totals = counts.sum(axis=1)
    predicted_totals = counts.sum(axis=0)

    rows = {
        "n_accounts": total,
        "route_accuracy": float(correct / total),
        "wrong_escalation_level": int(total - correct),
        "unnecessary_human_reviews": cell("auto_handle", "human_review"),
        "unnecessary_policy_reviews": cell("auto_handle", "policy_review"),
        "unnecessary_product_investigations": int(
            predicted_totals[route_index["product_investigation"]]
            - cell("product_investigation", "product_investigation")
        ),
        "adversarial_false_systemic_product_investigations": int(
            adversarial_false_systemic_product_investigations
        ),
        "missed_human_reviews": int(
            expected_totals[route_index["human_review"]]
            - cell("human_review", "human_review")
        ),
        "missed_policy_reviews": int(
            expected_totals[route_index["policy_review"]]
            - cell("policy_review", "policy_review")
        ),
        "missed_product_investigations": int(
            expected_totals[route_index["product_investigation"]]
            - cell("product_investigation", "product_investigation")
        ),
    }

    f1s = []
    for route in ROUTES:
        idx = route_index[route]
        precision = np.nan
        recall = np.nan
        if predicted_totals[idx] > 0:
            precision = float(counts[idx, idx] / predicted_totals[idx])
        if expected_totals[idx] > 0:
            recall = float(counts[idx, idx] / expected_totals[idx])
        rows[f"{route}_precision"] = precision
        rows[f"{route}_recall"] = recall
        if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0:
            f1s.append(2 * precision * recall / (precision + recall))
    rows["macro_route_f1"] = float(np.mean(f1s)) if f1s else 0.0

    over = 0
    under = 0
    for exp in ROUTES:
        for pred in ROUTES:
            count = cell(exp, pred)
            if ROUTE_LEVEL[pred] > ROUTE_LEVEL[exp]:
                over += count
            elif ROUTE_LEVEL[pred] < ROUTE_LEVEL[exp]:
                under += count
    rows["over_escalation_rate"] = float(over / total)
    rows["under_escalation_rate"] = float(under / total)
    return rows


def _fmt_float(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def _svg_text(x, y, text, size=13, weight=400, anchor="start"):
    safe = (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="#222">{safe}</text>'
    )


def _bar(parts, x, y, width, height, color, label):
    parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{color}"/>')
    parts.append(f'<title>{label}</title>')


def _errorbar(parts, x, y_low, y_high):
    parts.append(f'<line x1="{x:.1f}" y1="{y_low:.1f}" x2="{x:.1f}" y2="{y_high:.1f}" stroke="#222" stroke-width="1.2"/>')
    parts.append(f'<line x1="{x-4:.1f}" y1="{y_low:.1f}" x2="{x+4:.1f}" y2="{y_low:.1f}" stroke="#222" stroke-width="1.2"/>')
    parts.append(f'<line x1="{x-4:.1f}" y1="{y_high:.1f}" x2="{x+4:.1f}" y2="{y_high:.1f}" stroke="#222" stroke-width="1.2"/>')


def write_figure1(account_metrics_df, account_ci_df, out_path):
    selected_methods = ["flat_max", "learned_logistic_flat", "learned_logistic_hier"]
    metric_specs = [
        ("precision", "Precision", 0, 1),
        ("harm_weighted_recall", "Harm-weighted recall", 0, 1),
        ("systemic_account_recall", "Systemic account recall", 0, 1),
        ("false_positive_reviews", "False positives", 0, None),
    ]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    width, height = 980, 620
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(42, 36, "Figure 1. Held-out account review performance at 15% review budget", 20, 600),
    ]
    panel_w, panel_h = 420, 205
    origins = [(62, 76), (545, 76), (62, 350), (545, 350)]
    metric_df = account_metrics_df[
        (account_metrics_df["split"] == "test")
        & (account_metrics_df["budget"].round(4) == 0.15)
        & (account_metrics_df["method"].isin(selected_methods))
    ]
    ci = account_ci_df[account_ci_df["method"].isin(selected_methods)]
    for idx, (metric, title, ymin, ymax) in enumerate(metric_specs):
        ox, oy = origins[idx]
        sub = metric_df.set_index("method")
        ci_sub = ci[ci["metric"] == metric].set_index("method")
        vals = [float(sub.loc[m, metric]) for m in selected_methods]
        upper = [float(ci_sub.loc[m, "ci_high"]) for m in selected_methods]
        if ymax is None:
            ymax = max(max(upper), max(vals)) * 1.15
        parts.append(_svg_text(ox, oy - 16, title, 15, 600))
        parts.append(f'<rect x="{ox}" y="{oy}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#333" stroke-width="1"/>')
        for t in np.linspace(ymin, ymax, 5):
            y = oy + panel_h - (t - ymin) / (ymax - ymin) * panel_h
            parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+panel_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
            parts.append(_svg_text(ox - 9, y + 4, f"{t:.2f}" if ymax <= 1.2 else f"{t:.0f}", 11, 400, "end"))
        group_w = panel_w / len(selected_methods)
        bar_w = 66
        for i, method in enumerate(selected_methods):
            value = float(sub.loc[method, metric])
            lo = float(ci_sub.loc[method, "ci_low"])
            hi = float(ci_sub.loc[method, "ci_high"])
            bar_h = (value - ymin) / (ymax - ymin) * panel_h
            x = ox + i * group_w + (group_w - bar_w) / 2
            y = oy + panel_h - bar_h
            _bar(parts, x, y, bar_w, bar_h, colors[i], f"{METHOD_LABELS[method]} {metric}: {_fmt_float(value)}")
            ey_low = oy + panel_h - (lo - ymin) / (ymax - ymin) * panel_h
            ey_high = oy + panel_h - (hi - ymin) / (ymax - ymin) * panel_h
            _errorbar(parts, x + bar_w / 2, ey_high, ey_low)
            parts.append(_svg_text(x + bar_w / 2, oy + panel_h + 20, METHOD_LABELS[method], 11, 400, "middle"))
            parts.append(_svg_text(x + bar_w / 2, y - 8, _fmt_float(value), 11, 500, "middle"))
    parts.append(_svg_text(62, 592, "Whiskers show 95% bootstrap intervals from scenario-block resampling.", 12))
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_figure2(targeted_ci_df, out_path):
    panels = [
        ("uncertainty_legit_ambiguous", ["learned_logistic_hier", "ablation_no_uncertainty", "learned_logistic_flat"], "Benign ambiguity false selection"),
        ("signature_matched_benign", ["learned_logistic_hier", "ablation_no_signature", "learned_logistic_flat"], "Benign signature lookalike false selection"),
        ("low_provenance_artifact", ["learned_logistic_hier", "ablation_no_provenance", "learned_logistic_flat"], "Low-provenance artifact false selection"),
    ]
    colors = ["#54a24b", "#e45756", "#f58518"]
    width, height = 980, 455
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(42, 36, "Figure 2. Targeted ablations expose distinct hierarchical-state failure modes", 20, 600),
    ]
    panel_w, panel_h = 270, 250
    for idx, (archetype, methods, title) in enumerate(panels):
        ox = 68 + idx * 310
        oy = 84
        parts.append(_svg_text(ox, oy - 17, title, 13, 600))
        parts.append(f'<rect x="{ox}" y="{oy}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#333" stroke-width="1"/>')
        for t in np.linspace(0, 1, 6):
            y = oy + panel_h - t * panel_h
            parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+panel_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
            parts.append(_svg_text(ox - 8, y + 4, f"{t:.1f}", 11, 400, "end"))
        for i, method in enumerate(methods):
            row = targeted_ci_df[(targeted_ci_df["type"] == archetype) & (targeted_ci_df["method"] == method)].iloc[0]
            value = float(row["mean"])
            lo = float(row["ci_low"])
            hi = float(row["ci_high"])
            x = ox + 35 + i * 78
            bar_w = 48
            y = oy + panel_h - value * panel_h
            _bar(parts, x, y, bar_w, value * panel_h, colors[i], f"{METHOD_LABELS[method]} false selection: {_fmt_float(value)}")
            _errorbar(parts, x + bar_w / 2, oy + panel_h - hi * panel_h, oy + panel_h - lo * panel_h)
            label = METHOD_LABELS[method].replace("Learned ", "")
            parts.append(_svg_text(x + bar_w / 2, oy + panel_h + 18, label, 11, 400, "middle"))
            parts.append(_svg_text(x + bar_w / 2, y - 8, _fmt_float(value), 11, 500, "middle"))
    parts.append(_svg_text(68, 404, "Each panel is a matched hard-negative case. A high bar means the method falsely routes benign accounts for review.", 12))
    parts.append(_svg_text(68, 426, "The full learned hierarchy stays near zero; removing the required state component recreates the intended failure mode.", 12))
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_figure3(calibration_summary_df, system_ci_df, out_path):
    width, height = 980, 520
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(42, 36, "Figure 3. Calibration and system-alert performance", 20, 600),
    ]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    cal_methods = ["learned_logistic_flat", "learned_logistic_hier", "ablation_no_signature", "ablation_no_provenance"]
    cal = calibration_summary_df[
        (calibration_summary_df["split"] == "test")
        & (calibration_summary_df["method"].isin(cal_methods))
    ].set_index("method")
    ox, oy, panel_w, panel_h = 65, 86, 395, 270
    parts.append(_svg_text(ox, oy - 18, "Calibration error (lower is better)", 15, 600))
    parts.append(f'<rect x="{ox}" y="{oy}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#333" stroke-width="1"/>')
    ymax = 0.10
    for t in np.linspace(0, ymax, 6):
        y = oy + panel_h - t / ymax * panel_h
        parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+panel_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        parts.append(_svg_text(ox - 8, y + 4, f"{t:.2f}", 11, 400, "end"))
    group_w = panel_w / len(cal_methods)
    for i, method in enumerate(cal_methods):
        ece = float(cal.loc[method, "ece"])
        brier = float(cal.loc[method, "brier"])
        x0 = ox + i * group_w + 20
        for j, (metric, value, shade) in enumerate([("ECE", ece, colors[i]), ("Brier", brier, "#999")]):
            x = x0 + j * 28
            bar_h = value / ymax * panel_h
            _bar(parts, x, oy + panel_h - bar_h, 22, bar_h, shade, f"{METHOD_LABELS[method]} {metric}: {_fmt_float(value)}")
        parts.append(_svg_text(x0 + 15, oy + panel_h + 18, METHOD_LABELS[method].replace("Learned ", ""), 10, 400, "middle"))
    parts.append(_svg_text(ox + 225, oy + panel_h + 54, "Colored = ECE; gray = Brier score.", 11))

    alert_methods = [
        "signature_count_only",
        "signature_mean_risk",
        "system_hierarchy_alert",
        "typed_system_hierarchy_alert",
    ]
    alert_axis_labels = {
        "signature_count_only": "Count",
        "signature_mean_risk": "Mean risk",
        "system_hierarchy_alert": "System",
        "typed_system_hierarchy_alert": "Typed system",
    }
    ox2, oy2, panel_w2, panel_h2 = 545, 86, 360, 270
    parts.append(_svg_text(ox2, oy2 - 18, "System-alert hit rate (higher is better)", 15, 600))
    parts.append(f'<rect x="{ox2}" y="{oy2}" width="{panel_w2}" height="{panel_h2}" fill="none" stroke="#333" stroke-width="1"/>')
    for t in np.linspace(0, 1, 6):
        y = oy2 + panel_h2 - t * panel_h2
        parts.append(f'<line x1="{ox2}" y1="{y:.1f}" x2="{ox2+panel_w2}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        parts.append(_svg_text(ox2 - 8, y + 4, f"{t:.1f}", 11, 400, "end"))
    metric_df = system_ci_df[system_ci_df["metric"] == "product_alert_hit"].set_index("method")
    group_w = panel_w2 / len(alert_methods)
    for i, method in enumerate(alert_methods):
        row = metric_df.loc[method]
        value = float(row["mean"])
        lo = float(row["ci_low"])
        hi = float(row["ci_high"])
        bar_w = 48
        x = ox2 + i * group_w + (group_w - bar_w) / 2
        y = oy2 + panel_h2 - value * panel_h2
        _bar(parts, x, y, bar_w, value * panel_h2, colors[i % len(colors)], f"{METHOD_LABELS[method]} hit rate: {_fmt_float(value)}")
        _errorbar(parts, x + bar_w / 2, oy2 + panel_h2 - hi * panel_h2, oy2 + panel_h2 - lo * panel_h2)
        parts.append(_svg_text(x + bar_w / 2, oy2 + panel_h2 + 18, alert_axis_labels[method], 10, 400, "middle"))
        parts.append(_svg_text(x + bar_w / 2, y - 8, _fmt_float(value), 11, 500, "middle"))
    parts.append(_svg_text(65, 432, "This figure separates two deployment questions: calibrated account routing and product-level alerting.", 12))
    parts.append(_svg_text(65, 454, "System-alert whiskers show 95% bootstrap intervals from held-out scenario resampling.", 12))
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def _ci_lookup(ci_df, method, policy_variant, metric):
    row = ci_df[
        (ci_df["method"] == method)
        & (ci_df["policy_variant"] == policy_variant)
        & (ci_df["metric"] == metric)
    ].iloc[0]
    return float(row["mean"]), float(row["ci_low"]), float(row["ci_high"])


def write_figure4(routing_ci_df, out_path):
    width, height = 980, 620
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(42, 36, "Figure 4. Validation-tuned routing and escalation quality", 20, 600),
    ]
    colors = {
        "flat": "#f58518",
        "hier": "#54a24b",
        "ablation": "#e45756",
        "fixed": "#4c78a8",
        "tuned": "#54a24b",
    }

    # Panel A: fixed vs tuned for learned flat and learned hierarchy.
    ox, oy, panel_w, panel_h = 68, 88, 390, 220
    parts.append(_svg_text(ox, oy - 18, "A. Macro route F1: fixed vs tuned", 15, 600))
    parts.append(f'<rect x="{ox}" y="{oy}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#333" stroke-width="1"/>')
    for t in np.linspace(0, 1, 6):
        y = oy + panel_h - t * panel_h
        parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+panel_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        parts.append(_svg_text(ox - 8, y + 4, f"{t:.1f}", 11, 400, "end"))
    bars = [
        ("learned_logistic_flat", "fixed", "Flat fixed", colors["flat"]),
        ("learned_logistic_flat", "validation_tuned", "Flat tuned", colors["flat"]),
        ("learned_logistic_hier", "fixed", "Hier fixed", colors["fixed"]),
        ("learned_logistic_hier", "validation_tuned", "Hier tuned", colors["hier"]),
    ]
    bar_w = 44
    for i, (method, variant, label, color) in enumerate(bars):
        mean, lo, hi = _ci_lookup(routing_ci_df, method, variant, "macro_route_f1")
        x = ox + 38 + i * 82
        y = oy + panel_h - mean * panel_h
        _bar(parts, x, y, bar_w, mean * panel_h, color, f"{label} macro route F1: {_fmt_float(mean)}")
        if "tuned" not in label.lower():
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{mean*panel_h:.1f}" fill="none" stroke="#222" stroke-dasharray="4,3"/>')
        _errorbar(parts, x + bar_w / 2, oy + panel_h - hi * panel_h, oy + panel_h - lo * panel_h)
        parts.append(_svg_text(x + bar_w / 2, oy + panel_h + 18, label, 10, 400, "middle"))
        parts.append(_svg_text(x + bar_w / 2, y - 7, _fmt_float(mean), 11, 500, "middle"))

    # Panel B: tuned macro F1 across hierarchy and ablations.
    ox2, oy2, panel_w2, panel_h2 = 540, 88, 365, 220
    parts.append(_svg_text(ox2, oy2 - 18, "B. Tuned hierarchy vs ablations", 15, 600))
    parts.append(f'<rect x="{ox2}" y="{oy2}" width="{panel_w2}" height="{panel_h2}" fill="none" stroke="#333" stroke-width="1"/>')
    for t in np.linspace(0, 1, 6):
        y = oy2 + panel_h2 - t * panel_h2
        parts.append(f'<line x1="{ox2}" y1="{y:.1f}" x2="{ox2+panel_w2}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        parts.append(_svg_text(ox2 - 8, y + 4, f"{t:.1f}", 11, 400, "end"))
    methods = [
        ("learned_logistic_hier", "Full", colors["hier"]),
        ("ablation_no_uncertainty", "No uncert.", colors["ablation"]),
        ("ablation_no_signature", "No sig.", colors["ablation"]),
        ("ablation_no_provenance", "No prov.", colors["ablation"]),
    ]
    for i, (method, label, color) in enumerate(methods):
        mean, lo, hi = _ci_lookup(routing_ci_df, method, "validation_tuned", "macro_route_f1")
        x = ox2 + 26 + i * 82
        y = oy2 + panel_h2 - mean * panel_h2
        _bar(parts, x, y, bar_w, mean * panel_h2, color, f"{label} tuned macro route F1: {_fmt_float(mean)}")
        _errorbar(parts, x + bar_w / 2, oy2 + panel_h2 - hi * panel_h2, oy2 + panel_h2 - lo * panel_h2)
        parts.append(_svg_text(x + bar_w / 2, oy2 + panel_h2 + 18, label, 10, 400, "middle"))
        parts.append(_svg_text(x + bar_w / 2, y - 7, _fmt_float(mean), 11, 500, "middle"))

    # Panel C: missed product investigations, lower is better.
    ox3, oy3, panel_w3, panel_h3 = 68, 385, 837, 150
    parts.append(_svg_text(ox3, oy3 - 18, "C. Missed product investigations under tuned routing (lower is better)", 15, 600))
    parts.append(f'<rect x="{ox3}" y="{oy3}" width="{panel_w3}" height="{panel_h3}" fill="none" stroke="#333" stroke-width="1"/>')
    miss_methods = [
        ("learned_logistic_flat", "Learned flat", colors["flat"]),
        ("learned_logistic_hier", "Learned hierarchy", colors["hier"]),
        ("ablation_no_uncertainty", "No uncertainty", colors["ablation"]),
        ("ablation_no_signature", "No signature", colors["ablation"]),
        ("ablation_no_provenance", "No provenance", colors["ablation"]),
    ]
    miss_vals = [
        _ci_lookup(routing_ci_df, m, "validation_tuned", "missed_product_investigations")
        for m, _, _ in miss_methods
    ]
    ymax = max(v[2] for v in miss_vals) * 1.12
    for t in np.linspace(0, ymax, 4):
        y = oy3 + panel_h3 - t / ymax * panel_h3
        parts.append(f'<line x1="{ox3}" y1="{y:.1f}" x2="{ox3+panel_w3}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        parts.append(_svg_text(ox3 - 8, y + 4, f"{t:.0f}", 11, 400, "end"))
    for i, ((method, label, color), (mean, lo, hi)) in enumerate(zip(miss_methods, miss_vals)):
        group_w = panel_w3 / len(miss_methods)
        x = ox3 + i * group_w + (group_w - 62) / 2
        bar_h = mean / ymax * panel_h3
        y = oy3 + panel_h3 - bar_h
        _bar(parts, x, y, 62, bar_h, color, f"{label} missed product investigations: {_fmt_float(mean)}")
        _errorbar(parts, x + 31, oy3 + panel_h3 - hi / ymax * panel_h3, oy3 + panel_h3 - lo / ymax * panel_h3)
        parts.append(_svg_text(x + 31, oy3 + panel_h3 + 18, label, 10, 400, "middle"))
        parts.append(_svg_text(x + 31, y - 7, f"{mean:.0f}", 11, 500, "middle"))

    parts.append(_svg_text(68, 594, "Whiskers show 95% scenario-bootstrap intervals. Dashed bar outlines indicate fixed policies.", 12))
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_figure5(adversarial_stress_df, out_path):
    width, height = 980, 620
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(42, 36, "Figure 5. Adversarial false-systemic routing stress test", 20, 600),
    ]
    colors = {
        "learned": "#54a24b",
        "flat": "#f58518",
        "rule": "#e45756",
        "ablation": "#b279a2",
        "neutral": "#4c78a8",
    }
    method_rows = [
        ("learned_logistic_hier", "Learned hierarchy", "Learned hier.", colors["learned"]),
        ("learned_logistic_flat", "Learned flat", "Learned flat", colors["flat"]),
        ("typed_system_hierarchy", "Typed system rule", "Typed rule", colors["rule"]),
        ("system_hierarchy", "System rule", "System rule", colors["rule"]),
        ("account_rule", "Account rule", "Acct. rule", colors["rule"]),
        ("ablation_no_uncertainty", "No uncertainty", "No uncert.", colors["ablation"]),
    ]
    test = adversarial_stress_df[
        (adversarial_stress_df["split"] == "test")
        & (adversarial_stress_df["policy_variant"] == "validation_tuned")
    ].copy()
    all_rows = test[test["type"] == "ALL_ADVERSARIAL_FALSE_SYSTEMIC"].set_index("method")

    # Panel A: product-investigation false alert rate on all adversarial benign lookalikes.
    ox, oy, panel_w, panel_h = 68, 88, 390, 240
    parts.append(_svg_text(ox, oy - 18, "A. Benign lookalikes routed to product investigation", 15, 600))
    parts.append(f'<rect x="{ox}" y="{oy}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#333" stroke-width="1"/>')
    ymax = 0.55
    for t in np.linspace(0, ymax, 6):
        y = oy + panel_h - t / ymax * panel_h
        parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+panel_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        parts.append(_svg_text(ox - 8, y + 4, f"{t:.1f}", 11, 400, "end"))
    group_w = panel_w / len(method_rows)
    for i, (method, label, short_label, color) in enumerate(method_rows):
        if method not in all_rows.index:
            continue
        value = float(all_rows.loc[method, "product_investigation_rate"])
        x = ox + i * group_w + (group_w - 38) / 2
        bar_h = value / ymax * panel_h
        y = oy + panel_h - bar_h
        _bar(parts, x, y, 38, bar_h, color, f"{label}: {_fmt_float(value)} product-investigation rate")
        parts.append(_svg_text(x + 19, oy + panel_h + 17, short_label, 9, 400, "middle"))
        parts.append(_svg_text(x + 19, y - 7, _fmt_float(value), 11, 500, "middle"))

    # Panel B: human-review burden on adversarial benign lookalikes.
    ox2, oy2, panel_w2, panel_h2 = 540, 88, 365, 240
    parts.append(_svg_text(ox2, oy2 - 18, "B. Benign lookalikes routed to human review", 15, 600))
    parts.append(f'<rect x="{ox2}" y="{oy2}" width="{panel_w2}" height="{panel_h2}" fill="none" stroke="#333" stroke-width="1"/>')
    for t in np.linspace(0, 1, 6):
        y = oy2 + panel_h2 - t * panel_h2
        parts.append(f'<line x1="{ox2}" y1="{y:.1f}" x2="{ox2+panel_w2}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        parts.append(_svg_text(ox2 - 8, y + 4, f"{t:.1f}", 11, 400, "end"))
    group_w2 = panel_w2 / len(method_rows)
    for i, (method, label, short_label, color) in enumerate(method_rows):
        if method not in all_rows.index:
            continue
        value = float(all_rows.loc[method, "human_review_rate"])
        x = ox2 + i * group_w2 + (group_w2 - 34) / 2
        bar_h = value * panel_h2
        y = oy2 + panel_h2 - bar_h
        _bar(parts, x, y, 34, bar_h, color, f"{label}: {_fmt_float(value)} human-review rate")
        parts.append(_svg_text(x + 17, oy2 + panel_h2 + 17, short_label, 9, 400, "middle"))
        parts.append(_svg_text(x + 17, y - 7, _fmt_float(value), 11, 500, "middle"))

    # Panel C: family-level product-investigation rate for the learned hierarchy.
    family = test[
        (test["method"] == "learned_logistic_hier")
        & (test["type"] != "ALL_ADVERSARIAL_FALSE_SYSTEMIC")
    ].copy()
    family_labels = {
        "authorized_security_testing_cohort": "Authorized testing",
        "classroom_lab_exercise_lookalike": "Classroom/lab",
        "copied_public_example_spread": "Copied public example",
        "legitimate_dual_use_research_cohort": "Dual-use research",
        "public_news_burst_lookalike": "Public-news burst",
        "support_debugging_repeated_terms": "Support/debugging",
    }
    family["label"] = family["type"].map(family_labels).fillna(family["type"])
    family = family.sort_values("label")
    ox3, oy3, panel_w3, panel_h3 = 68, 410, 837, 135
    parts.append(_svg_text(ox3, oy3 - 18, "C. Learned hierarchy: product-investigation rate by benign lookalike family", 15, 600))
    parts.append(f'<rect x="{ox3}" y="{oy3}" width="{panel_w3}" height="{panel_h3}" fill="none" stroke="#333" stroke-width="1"/>')
    ymax3 = max(0.02, float(family["product_investigation_rate"].max()) * 1.4)
    for t in np.linspace(0, ymax3, 3):
        y = oy3 + panel_h3 - t / ymax3 * panel_h3
        parts.append(f'<line x1="{ox3}" y1="{y:.1f}" x2="{ox3+panel_w3}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        parts.append(_svg_text(ox3 - 8, y + 4, f"{t:.3f}", 11, 400, "end"))
    group_w3 = panel_w3 / max(1, len(family))
    for i, (_, row) in enumerate(family.iterrows()):
        value = float(row["product_investigation_rate"])
        x = ox3 + i * group_w3 + (group_w3 - 54) / 2
        bar_h = value / ymax3 * panel_h3
        y = oy3 + panel_h3 - bar_h
        _bar(parts, x, y, 54, bar_h, colors["learned"], f'{row["label"]}: {_fmt_float(value)} product-investigation rate')
        parts.append(_svg_text(x + 27, oy3 + panel_h3 + 17, row["label"], 9, 400, "middle"))
        parts.append(_svg_text(x + 27, max(oy3 + 14, y - 7), _fmt_float(value), 11, 500, "middle"))

    parts.append(_svg_text(68, 594, "Adversarial false-systemic accounts are benign coordinated activity that should not trigger product investigation.", 12))
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_figure6(routing_penalty_sweep_df, out_path, frontier_ci_df=None):
    width, height = 980, 620
    df = routing_penalty_sweep_df[
        routing_penalty_sweep_df["method"] == "learned_logistic_hier"
    ].copy()
    df = df.sort_values("adversarial_false_alert_penalty")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(42, 36, "Figure 6. Product-investigation operating frontier", 20, 600),
    ]
    ox, oy, panel_w, panel_h = 92, 82, 590, 390
    parts.append(
        _svg_text(
            ox,
            oy - 18,
            "Validation-tuned learned hierarchy under adversarial false-alert penalties",
            15,
            600,
        )
    )
    parts.append(f'<rect x="{ox}" y="{oy}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#333" stroke-width="1"/>')
    ci = pd.DataFrame() if frontier_ci_df is None else frontier_ci_df.copy()
    ci = ci[ci["method"] == "learned_logistic_hier"] if not ci.empty else ci
    x_high = float(df["adversarial_false_systemic_product_investigations"].max())
    y_high = float(df["missed_product_investigations"].max())
    if not ci.empty:
        x_ci = ci[ci["metric"] == "adversarial_false_systemic_product_investigations"]
        y_ci = ci[ci["metric"] == "missed_product_investigations"]
        if not x_ci.empty:
            x_high = max(x_high, float(x_ci["ci_high"].max()))
        if not y_ci.empty:
            y_high = max(y_high, float(y_ci["ci_high"].max()))
    x_max = max(1.0, x_high * 1.25)
    y_max = max(1.0, y_high * 1.25)
    for t in np.linspace(0, x_max, 5):
        x = ox + t / x_max * panel_w
        parts.append(f'<line x1="{x:.1f}" y1="{oy}" x2="{x:.1f}" y2="{oy+panel_h}" stroke="#e5e5e5"/>')
        parts.append(_svg_text(x, oy + panel_h + 22, f"{t:.0f}", 11, 400, "middle"))
    for t in np.linspace(0, y_max, 5):
        y = oy + panel_h - t / y_max * panel_h
        parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+panel_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        parts.append(_svg_text(ox - 10, y + 4, f"{t:.0f}", 11, 400, "end"))
    parts.append(_svg_text(ox + panel_w / 2, oy + panel_h + 52, "Adversarial benign product-investigation alerts", 13, 500, "middle"))
    parts.append(
        f'<text x="30" y="{oy + panel_h / 2:.1f}" font-family="Arial" font-size="13" font-weight="500" '
        f'text-anchor="middle" fill="#222" transform="rotate(-90 30 {oy + panel_h / 2:.1f})">Missed true product investigations</text>'
    )

    def sx(value):
        return ox + float(value) / x_max * panel_w

    def sy(value):
        return oy + panel_h - float(value) / y_max * panel_h

    grouped_points = (
        df.groupby(
            [
                "adversarial_false_systemic_product_investigations",
                "missed_product_investigations",
            ],
            as_index=False,
        )
        .agg(
            penalty_min=("adversarial_false_alert_penalty", "min"),
            penalty_max=("adversarial_false_alert_penalty", "max"),
            n_penalties=("adversarial_false_alert_penalty", "size"),
            macro_route_f1=("macro_route_f1", "mean"),
            product_investigation_precision=("product_investigation_precision", "mean"),
            product_investigation_recall=("product_investigation_recall", "mean"),
        )
        .sort_values(["adversarial_false_systemic_product_investigations", "missed_product_investigations"])
    )
    points = [
        (
            sx(row["adversarial_false_systemic_product_investigations"]),
            sy(row["missed_product_investigations"]),
            row,
        )
        for _, row in grouped_points.iterrows()
    ]
    def ci_for(row, metric):
        if ci.empty:
            return None
        sub = ci[
            (ci["metric"] == metric)
            & (ci["adversarial_false_alert_penalty"] >= float(row["penalty_min"]))
            & (ci["adversarial_false_alert_penalty"] <= float(row["penalty_max"]))
        ]
        if sub.empty:
            return None
        return {
            "mean": float(sub["mean"].mean()),
            "ci_low": float(sub["ci_low"].mean()),
            "ci_high": float(sub["ci_high"].mean()),
        }

    if len(points) > 1:
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
        parts.append(f'<polyline points="{path}" fill="none" stroke="#54a24b" stroke-width="2.2"/>')
    for i, (x, y, row) in enumerate(points):
        x_ci = ci_for(row, "adversarial_false_systemic_product_investigations")
        y_ci = ci_for(row, "missed_product_investigations")
        if x_ci is not None:
            x_lo = sx(x_ci["ci_low"])
            x_hi = sx(x_ci["ci_high"])
            parts.append(
                f'<line x1="{x_lo:.1f}" y1="{y:.1f}" x2="{x_hi:.1f}" y2="{y:.1f}" '
                f'stroke="#222" stroke-width="1.1" opacity="0.8"/>'
            )
            parts.append(
                f'<line x1="{x_lo:.1f}" y1="{y-4:.1f}" x2="{x_lo:.1f}" y2="{y+4:.1f}" '
                f'stroke="#222" stroke-width="1.1" opacity="0.8"/>'
            )
            parts.append(
                f'<line x1="{x_hi:.1f}" y1="{y-4:.1f}" x2="{x_hi:.1f}" y2="{y+4:.1f}" '
                f'stroke="#222" stroke-width="1.1" opacity="0.8"/>'
            )
        if y_ci is not None:
            y_lo = sy(y_ci["ci_low"])
            y_hi = sy(y_ci["ci_high"])
            _errorbar(parts, x, y_hi, y_lo)
        penalty_min = float(row["penalty_min"])
        penalty_max = float(row["penalty_max"])
        false_alerts = int(row["adversarial_false_systemic_product_investigations"])
        missed = int(row["missed_product_investigations"])
        label = f"{penalty_min:g}" if penalty_min == penalty_max else f"{penalty_min:g}-{penalty_max:g}"
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#54a24b">'
            f'<title>penalty={label}; false alerts={false_alerts}; missed={missed}</title></circle>'
        )
        if int(row["n_penalties"]) > 1:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="none" stroke="#54a24b" stroke-width="1.4"/>'
            )
        label_y = y - 11 if i % 2 == 0 else y + 20
        parts.append(_svg_text(x + 8, label_y, label, 11, 500))

    # Right-side metric strip keeps the tradeoff interpretable without crowding the frontier.
    box_x, box_y = 725, 92
    parts.append(_svg_text(box_x, box_y, "Penalty sweep", 15, 600))
    parts.append(_svg_text(box_x, box_y + 28, "Each point retunes routing", 12))
    parts.append(_svg_text(box_x, box_y + 48, "thresholds on validation data.", 12))
    parts.append(_svg_text(box_x, box_y + 86, "Best test utility", 13, 600))
    best = df.sort_values("routing_utility", ascending=False).iloc[0]
    parts.append(_svg_text(box_x, box_y + 112, f"penalty = {float(best['adversarial_false_alert_penalty']):g}", 12))
    parts.append(_svg_text(box_x, box_y + 134, f"utility = {float(best['routing_utility']):.3f}", 12))
    parts.append(_svg_text(box_x, box_y + 156, f"macro F1 = {float(best['macro_route_f1']):.3f}", 12))
    parts.append(_svg_text(box_x, box_y + 178, f"product recall = {float(best['product_investigation_recall']):.3f}", 12))
    parts.append(_svg_text(box_x, box_y + 222, "Lower-left is preferred:", 12, 600))
    parts.append(_svg_text(box_x, box_y + 244, "fewer false systemic alerts", 12))
    parts.append(_svg_text(box_x, box_y + 264, "and fewer missed product cases.", 12))
    if len(grouped_points) == 1 and len(df) > 1:
        parts.append(_svg_text(box_x, box_y + 306, "Sweep result", 13, 600))
        parts.append(_svg_text(box_x, box_y + 328, "All penalties selected the", 12))
        parts.append(_svg_text(box_x, box_y + 348, "same held-out operating point.", 12))
    if not ci.empty:
        parts.append(_svg_text(88, 590, "Whiskers show 95% scenario-bootstrap intervals; point labels are validation penalty values.", 12))
    else:
        parts.append(_svg_text(88, 590, "Point labels are adversarial false-alert penalty values used in validation tuning.", 12))
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_figure7(time_to_alert_df, out_path, workload_ci_df=None):
    df = time_to_alert_df[
        (time_to_alert_df["policy_variant"] == "validation_tuned")
        & time_to_alert_df["method"].isin(
            [
                "learned_logistic_flat",
                "learned_logistic_hier",
                "ablation_no_uncertainty",
                "ablation_no_signature",
                "ablation_no_provenance",
                "typed_system_hierarchy",
            ]
        )
    ].copy()
    ci = pd.DataFrame() if workload_ci_df is None else workload_ci_df.copy()
    ci = ci[
        (ci["policy_variant"] == "validation_tuned")
        & ci["method"].isin(
            [
                "learned_logistic_flat",
                "learned_logistic_hier",
                "ablation_no_uncertainty",
                "ablation_no_signature",
                "ablation_no_provenance",
                "typed_system_hierarchy",
            ]
        )
    ] if not ci.empty else ci
    methods = [
        ("learned_logistic_flat", "Flat", "#f58518"),
        ("learned_logistic_hier", "Hierarchy", "#54a24b"),
        ("ablation_no_uncertainty", "No uncert.", "#b279a2"),
        ("ablation_no_signature", "No sig.", "#e45756"),
        ("ablation_no_provenance", "No prov.", "#b279a2"),
        ("typed_system_hierarchy", "Typed rule", "#4c78a8"),
    ]
    df = df.set_index("method")
    width, height = 980, 620
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(42, 36, "Figure 7. Practical monitoring simulation: alert time and analyst workload", 20, 600),
    ]

    panels = [
        ("product_campaign_alert_rate", "A. Product campaign alert rate", 0, 1, "rate"),
        ("total_analyst_hours", "B. Total analyst hours", 0, None, "hours"),
        ("false_product_investigations", "C. False product investigations", 0, None, "count"),
    ]
    origins = [(68, 88), (540, 88), (68, 385)]
    sizes = [(390, 220), (365, 220), (837, 145)]
    for idx, (metric, title, ymin, ymax, kind) in enumerate(panels):
        ox, oy = origins[idx]
        panel_w, panel_h = sizes[idx]
        values = [float(df.loc[m, metric]) for m, _, _ in methods if m in df.index]
        if ymax is None:
            ymax = max(values) * 1.15 if values else 1.0
            if not ci.empty:
                ci_sub = ci[ci["metric"] == metric]
                if not ci_sub.empty:
                    ymax = max(ymax, float(ci_sub["ci_high"].max()) * 1.15)
        parts.append(_svg_text(ox, oy - 18, title, 15, 600))
        parts.append(f'<rect x="{ox}" y="{oy}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#333" stroke-width="1"/>')
        for t in np.linspace(ymin, ymax, 5):
            y = oy + panel_h - (t - ymin) / (ymax - ymin) * panel_h
            label = f"{t:.1f}" if kind == "rate" else f"{t:.0f}"
            parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+panel_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
            parts.append(_svg_text(ox - 8, y + 4, label, 11, 400, "end"))
        group_w = panel_w / len(methods)
        bar_w = 42 if idx < 2 else 64
        for i, (method, label, color) in enumerate(methods):
            if method not in df.index:
                continue
            value = float(df.loc[method, metric])
            x = ox + i * group_w + (group_w - bar_w) / 2
            bar_h = (value - ymin) / (ymax - ymin) * panel_h
            y = oy + panel_h - bar_h
            _bar(parts, x, y, bar_w, bar_h, color, f"{label} {metric}: {_fmt_float(value)}")
            if not ci.empty:
                ci_row = ci[(ci["method"] == method) & (ci["metric"] == metric)]
                if not ci_row.empty:
                    lo = float(ci_row.iloc[0]["ci_low"])
                    hi = float(ci_row.iloc[0]["ci_high"])
                    if not np.isnan(lo) and not np.isnan(hi):
                        ey_low = oy + panel_h - (lo - ymin) / (ymax - ymin) * panel_h
                        ey_high = oy + panel_h - (hi - ymin) / (ymax - ymin) * panel_h
                        _errorbar(parts, x + bar_w / 2, ey_high, ey_low)
            value_label = _fmt_float(value) if kind == "rate" else f"{value:.0f}"
            parts.append(_svg_text(x + bar_w / 2, max(oy + 13, y - 7), value_label, 10, 500, "middle"))
            parts.append(_svg_text(x + bar_w / 2, oy + panel_h + 17, label, 9, 400, "middle"))

    hier = df.loc["learned_logistic_hier"]
    typed = df.loc["typed_system_hierarchy"]
    parts.append(_svg_text(540, 392, "Time-to-alert readout", 15, 600))
    parts.append(_svg_text(540, 420, f"Learned hierarchy alerted {hier['product_campaign_alert_rate']:.3f} of product campaigns.", 12))
    parts.append(_svg_text(540, 442, f"Median time-to-alert: {hier['median_time_to_product_alert_hours']:.1f} h.", 12))
    parts.append(_svg_text(540, 464, f"Analyst hours: {hier['total_analyst_hours']:.0f}, vs typed rule {typed['total_analyst_hours']:.0f}.", 12))
    parts.append(_svg_text(540, 486, f"False product investigations: {hier['false_product_investigations']:.0f}, vs typed rule {typed['false_product_investigations']:.0f}.", 12))
    if not ci.empty:
        parts.append(_svg_text(68, 594, "Whiskers show 95% scenario-bootstrap intervals; arrivals span 14 days and route costs are 0, 0.25, 0.75, and 2 hours.", 12))
    else:
        parts.append(_svg_text(68, 594, "Arrival times are deterministic synthetic arrivals over a 14-day window; route costs are 0, 0.25, 0.75, and 2 hours.", 12))
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_figure8(backlog_summary_df, out_path, backlog_ci_df=None):
    df = backlog_summary_df[
        (backlog_summary_df["policy_variant"] == "validation_tuned")
        & (backlog_summary_df["capacity_hours_per_day"] == 16.0)
        & (backlog_summary_df["queue_mode"] == "fifo")
        & backlog_summary_df["method"].isin(
            [
                "learned_logistic_flat",
                "learned_logistic_hier",
                "ablation_no_uncertainty",
                "ablation_no_signature",
                "ablation_no_provenance",
                "typed_system_hierarchy",
            ]
        )
    ].copy()
    ci = pd.DataFrame() if backlog_ci_df is None else backlog_ci_df.copy()
    ci = ci[
        (ci["policy_variant"] == "validation_tuned")
        & (ci["capacity_hours_per_day"] == 16.0)
        & (ci["queue_mode"] == "fifo")
    ] if not ci.empty else ci
    methods = [
        ("learned_logistic_flat", "Flat", "#f58518"),
        ("learned_logistic_hier", "Hierarchy", "#54a24b"),
        ("ablation_no_uncertainty", "No uncert.", "#b279a2"),
        ("ablation_no_signature", "No sig.", "#e45756"),
        ("ablation_no_provenance", "No prov.", "#b279a2"),
        ("typed_system_hierarchy", "Typed rule", "#4c78a8"),
    ]
    df = df.set_index("method")
    width, height = 980, 620
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(42, 36, "Figure 8. Deduplicated alert queue under finite analyst capacity", 20, 600),
    ]
    panels = [
        ("peak_backlog_hours", "A. Peak backlog hours", 0, None, "hours"),
        ("true_product_alerts_resolved_within_24h_rate", "B. True product alerts resolved within 24h", 0, 1, "rate"),
        ("false_unique_product_alerts", "C. Unique false product alerts", 0, None, "count"),
    ]
    origins = [(68, 88), (540, 88), (68, 385)]
    sizes = [(390, 220), (365, 220), (837, 145)]
    for idx, (metric, title, ymin, ymax, kind) in enumerate(panels):
        ox, oy = origins[idx]
        panel_w, panel_h = sizes[idx]
        values = [float(df.loc[m, metric]) for m, _, _ in methods if m in df.index]
        if ymax is None:
            ymax = max(values) * 1.15 if values else 1.0
            if not ci.empty:
                ci_sub = ci[ci["metric"] == metric]
                if not ci_sub.empty:
                    ymax = max(ymax, float(ci_sub["ci_high"].max()) * 1.15)
        parts.append(_svg_text(ox, oy - 18, title, 15, 600))
        parts.append(f'<rect x="{ox}" y="{oy}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#333" stroke-width="1"/>')
        for t in np.linspace(ymin, ymax, 5):
            y = oy + panel_h - (t - ymin) / (ymax - ymin) * panel_h
            label = f"{t:.1f}" if kind == "rate" else f"{t:.0f}"
            parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+panel_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
            parts.append(_svg_text(ox - 8, y + 4, label, 11, 400, "end"))
        group_w = panel_w / len(methods)
        bar_w = 42 if idx < 2 else 64
        for i, (method, label, color) in enumerate(methods):
            if method not in df.index:
                continue
            value = float(df.loc[method, metric])
            x = ox + i * group_w + (group_w - bar_w) / 2
            bar_h = (value - ymin) / (ymax - ymin) * panel_h
            y = oy + panel_h - bar_h
            _bar(parts, x, y, bar_w, bar_h, color, f"{label} {metric}: {_fmt_float(value)}")
            if not ci.empty:
                ci_row = ci[(ci["method"] == method) & (ci["metric"] == metric)]
                if not ci_row.empty:
                    lo = float(ci_row.iloc[0]["ci_low"])
                    hi = float(ci_row.iloc[0]["ci_high"])
                    if not np.isnan(lo) and not np.isnan(hi):
                        ey_low = oy + panel_h - (lo - ymin) / (ymax - ymin) * panel_h
                        ey_high = oy + panel_h - (hi - ymin) / (ymax - ymin) * panel_h
                        _errorbar(parts, x + bar_w / 2, ey_high, ey_low)
            value_label = _fmt_float(value) if kind == "rate" else f"{value:.0f}"
            parts.append(_svg_text(x + bar_w / 2, max(oy + 13, y - 7), value_label, 10, 500, "middle"))
            parts.append(_svg_text(x + bar_w / 2, oy + panel_h + 17, label, 9, 400, "middle"))

    if "learned_logistic_hier" in df.index and "typed_system_hierarchy" in df.index:
        hier = df.loc["learned_logistic_hier"]
        typed = df.loc["typed_system_hierarchy"]
        parts.append(_svg_text(540, 392, "Deduplication readout", 15, 600))
        parts.append(_svg_text(540, 420, "Capacity shown: 16 analyst-hours/day.", 12))
        parts.append(
            _svg_text(
                540,
                442,
                f"Hierarchy avoids {hier['duplicate_product_alerts_avoided']:.0f} duplicate product alerts.",
                12,
            )
        )
        parts.append(
            _svg_text(
                540,
                464,
                f"Hierarchy unique false alerts: {hier['false_unique_product_alerts']:.0f}; typed rule: {typed['false_unique_product_alerts']:.0f}.",
                12,
            )
        )
        parts.append(
            _svg_text(
                540,
                486,
                f"Hierarchy p90 product delay: {hier['p90_delay_to_product_investigation_hours']:.1f} h.",
                12,
            )
        )
    if not ci.empty:
        parts.append(_svg_text(68, 594, "FIFO queue. Whiskers show 95% scenario-bootstrap intervals after product-alert deduplication.", 12))
    else:
        parts.append(_svg_text(68, 594, "Product-investigation alerts are deduplicated by scenario, signature, and campaign before queue simulation.", 12))
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_figure9(backlog_summary_df, backlog_ci_df, out_path):
    df = backlog_summary_df[
        (backlog_summary_df["policy_variant"] == "validation_tuned")
        & (backlog_summary_df["capacity_hours_per_day"] == 16.0)
        & (backlog_summary_df["method"] == "learned_logistic_hier")
    ].copy()
    ci = pd.DataFrame() if backlog_ci_df is None else backlog_ci_df.copy()
    ci = ci[
        (ci["policy_variant"] == "validation_tuned")
        & (ci["capacity_hours_per_day"] == 16.0)
        & (ci["method"] == "learned_logistic_hier")
    ] if not ci.empty else ci
    modes = [
        ("fifo", "FIFO", "#4c78a8"),
        ("product_first", "Product first", "#54a24b"),
        ("product_policy_first", "Product+policy", "#b279a2"),
        ("severity_weighted", "Severity", "#f58518"),
    ]
    df = df.set_index("queue_mode")
    width, height = 980, 620
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(42, 36, "Figure 9. Priority queue policies for learned hierarchical routing", 20, 600),
    ]
    panels = [
        ("p90_delay_to_product_investigation_hours", "A. p90 product-investigation delay", 0, None, "hours"),
        ("true_product_alerts_resolved_within_24h_rate", "B. True product alerts resolved within 24h", 0, 1, "rate"),
        ("peak_non_product_backlog_hours", "C. Peak non-product backlog", 0, None, "hours"),
    ]
    origins = [(68, 88), (540, 88), (68, 385)]
    sizes = [(390, 220), (365, 220), (837, 145)]
    for idx, (metric, title, ymin, ymax, kind) in enumerate(panels):
        ox, oy = origins[idx]
        panel_w, panel_h = sizes[idx]
        values = [float(df.loc[m, metric]) for m, _, _ in modes if m in df.index]
        if ymax is None:
            ymax = max(values) * 1.15 if values else 1.0
            if not ci.empty:
                ci_sub = ci[ci["metric"] == metric]
                if not ci_sub.empty:
                    ymax = max(ymax, float(ci_sub["ci_high"].max()) * 1.15)
        parts.append(_svg_text(ox, oy - 18, title, 15, 600))
        parts.append(f'<rect x="{ox}" y="{oy}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#333" stroke-width="1"/>')
        for t in np.linspace(ymin, ymax, 5):
            y = oy + panel_h - (t - ymin) / (ymax - ymin) * panel_h
            label = f"{t:.1f}" if kind == "rate" else f"{t:.0f}"
            parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+panel_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
            parts.append(_svg_text(ox - 8, y + 4, label, 11, 400, "end"))
        group_w = panel_w / len(modes)
        bar_w = 58 if idx == 2 else 42
        for i, (mode, label, color) in enumerate(modes):
            if mode not in df.index:
                continue
            value = float(df.loc[mode, metric])
            x = ox + i * group_w + (group_w - bar_w) / 2
            bar_h = (value - ymin) / (ymax - ymin) * panel_h
            y = oy + panel_h - bar_h
            _bar(parts, x, y, bar_w, bar_h, color, f"{label} {metric}: {_fmt_float(value)}")
            if not ci.empty:
                ci_row = ci[(ci["queue_mode"] == mode) & (ci["metric"] == metric)]
                if not ci_row.empty:
                    lo = float(ci_row.iloc[0]["ci_low"])
                    hi = float(ci_row.iloc[0]["ci_high"])
                    if not np.isnan(lo) and not np.isnan(hi):
                        ey_low = oy + panel_h - (lo - ymin) / (ymax - ymin) * panel_h
                        ey_high = oy + panel_h - (hi - ymin) / (ymax - ymin) * panel_h
                        _errorbar(parts, x + bar_w / 2, ey_high, ey_low)
            value_label = _fmt_float(value) if kind == "rate" else f"{value:.0f}"
            parts.append(_svg_text(x + bar_w / 2, max(oy + 13, y - 7), value_label, 10, 500, "middle"))
            parts.append(_svg_text(x + bar_w / 2, oy + panel_h + 17, label, 9, 400, "middle"))
    if "fifo" in df.index and "product_first" in df.index:
        fifo = df.loc["fifo"]
        product = df.loc["product_first"]
        parts.append(_svg_text(540, 392, "Priority readout", 15, 600))
        parts.append(_svg_text(540, 420, "Capacity shown: 16 analyst-hours/day.", 12))
        parts.append(_svg_text(540, 442, f"FIFO p90 product delay: {fifo['p90_delay_to_product_investigation_hours']:.1f} h.", 12))
        parts.append(_svg_text(540, 464, f"Product-first p90 delay: {product['p90_delay_to_product_investigation_hours']:.1f} h.", 12))
        parts.append(_svg_text(540, 486, f"Product-first 24h resolution: {product['true_product_alerts_resolved_within_24h_rate']:.3f}.", 12))
    parts.append(_svg_text(68, 594, "Whiskers show 95% scenario-bootstrap intervals. Panel C shows spillover backlog outside product investigations.", 12))
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_captions(out_path):
    text = """# Manuscript Figure Captions

**Figure 1. Held-out account review performance at 15% review budget.** Bars compare a single-message flat baseline, a learned flat baseline, and a learned hierarchical-state model. The hierarchy is evaluated on precision, harm-weighted recall, systemic account recall, and false-positive reviews. Whiskers are 95% bootstrap intervals obtained by resampling held-out scenarios, preserving each scenario as the independent unit.

**Figure 2. Targeted ablations expose distinct hierarchical-state failure modes.** Each panel uses a matched hard-negative family designed so that one state component is necessary: uncertainty distinguishes legitimate ambiguity, signature/campaign state distinguishes benign lookalikes from coordinated campaigns, and provenance quality distinguishes low-quality artifacts from actionable evidence. A high bar indicates false selection of benign accounts at validation-tuned thresholds.

**Figure 3. Calibration and system-alert performance.** The left panel summarizes test-set calibration for learned flat, learned hierarchical, and ablated hierarchical models using expected calibration error and Brier score. The right panel evaluates system-level alerting as the probability that the true exploit signature appears in the top three product alerts, with scenario-bootstrap intervals.

**Figure 4. Validation-tuned routing and escalation quality.** Panel A compares fixed and validation-tuned routing policies for learned flat and learned hierarchical models. Panel B compares the validation-tuned full hierarchy against ablations that remove uncertainty, signature/campaign, or provenance state. Panel C reports missed product investigations under validation-tuned routing. Whiskers are 95% bootstrap intervals obtained by resampling held-out scenarios.

**Figure 5. Adversarial false-systemic routing stress test.** Panels A and B evaluate benign coordinated lookalikes that should not trigger product investigation, including authorized security testing, classroom/lab exercises, public-news bursts, support/debugging workflows, legitimate dual-use research cohorts, and copied public examples. Panel A reports false product-investigation rates, Panel B reports human-review burden, and Panel C breaks down the learned hierarchy's residual product-investigation alerts by lookalike family.

**Figure 6. Product-investigation operating frontier.** Each point retunes the learned hierarchical routing policy on validation scenarios under a different penalty for adversarial benign coordinated accounts routed to product investigation. The x-axis reports adversarial false-systemic product-investigation alerts on held-out test scenarios; the y-axis reports missed true product investigations. This figure asks whether the routing policy can trade off product-level recall against false systemic escalation.

**Figure 7. Practical monitoring simulation: alert time and analyst workload.** Accounts are assigned deterministic synthetic arrival times over a 14-day monitoring window. Each predicted route carries an analyst-time cost: auto-handle = 0 hours, human review = 0.25 hours, policy review = 0.75 hours, and product investigation = 2 hours. The figure reports product-campaign alert rate, total analyst hours, false product investigations, and the learned hierarchy's time-to-alert summary. Whiskers show 95% scenario-bootstrap intervals.

**Figure 8. Deduplicated alert queue under finite analyst capacity.** Product-investigation alerts are collapsed by scenario, signature, and campaign before queue simulation. The figure shows peak backlog hours, true product alerts resolved within 24 hours, and unique false product alerts at 16 analyst-hours per day. This evaluates whether routing remains operationally manageable once repeated product alerts and finite analyst capacity are modeled.

**Figure 9. Priority queue policies for learned hierarchical routing.** The learned hierarchical router is evaluated under FIFO, product-first, product-plus-policy-first, and severity-weighted queue policies at 16 analyst-hours per day. Panels compare p90 product-investigation delay, true product alerts resolved within 24 hours, and spillover backlog for non-product work. Whiskers show 95% scenario-bootstrap intervals.
"""
    out_path.write_text(text, encoding="utf-8")


def write_manuscript_figures(out_dir):
    out_dir = Path(out_dir)
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    account_metrics_df = pd.read_csv(out_dir / "account_metrics_by_split.csv")
    account_ci_df = pd.read_csv(out_dir / "bootstrap_account_ci.csv")
    targeted_ci_df = pd.read_csv(out_dir / "bootstrap_targeted_ablation_ci.csv")
    calibration_summary_df = pd.read_csv(out_dir / "calibration_summary.csv")
    system_ci_df = pd.read_csv(out_dir / "bootstrap_system_alert_ci.csv")
    routing_ci_df = pd.read_csv(out_dir / "bootstrap_routing_ci.csv")
    adversarial_stress_df = pd.read_csv(out_dir / "adversarial_routing_stress.csv")
    routing_penalty_sweep_df = pd.read_csv(out_dir / "routing_penalty_sweep.csv")
    frontier_ci_path = out_dir / "bootstrap_routing_frontier_ci.csv"
    frontier_ci_df = pd.read_csv(frontier_ci_path) if frontier_ci_path.exists() else None
    time_to_alert_path = out_dir / "time_to_alert_workload_summary.csv"
    time_to_alert_df = pd.read_csv(time_to_alert_path) if time_to_alert_path.exists() else None
    workload_ci_path = out_dir / "bootstrap_time_to_alert_workload_ci.csv"
    workload_ci_df = pd.read_csv(workload_ci_path) if workload_ci_path.exists() else None
    backlog_summary_path = out_dir / "dedup_backlog_summary.csv"
    backlog_summary_df = pd.read_csv(backlog_summary_path) if backlog_summary_path.exists() else None
    backlog_ci_path = out_dir / "bootstrap_dedup_backlog_ci.csv"
    backlog_ci_df = pd.read_csv(backlog_ci_path) if backlog_ci_path.exists() else None

    write_figure1(account_metrics_df, account_ci_df, figure_dir / "figure1_account_review.svg")
    write_figure2(targeted_ci_df, figure_dir / "figure2_targeted_ablations.svg")
    write_figure3(calibration_summary_df, system_ci_df, figure_dir / "figure3_calibration_system_alerts.svg")
    write_figure4(routing_ci_df, figure_dir / "figure4_routing_escalation.svg")
    write_figure5(adversarial_stress_df, figure_dir / "figure5_adversarial_false_systemic.svg")
    write_figure6(
        routing_penalty_sweep_df,
        figure_dir / "figure6_product_investigation_frontier.svg",
        frontier_ci_df=frontier_ci_df,
    )
    if time_to_alert_df is not None:
        write_figure7(
            time_to_alert_df,
            figure_dir / "figure7_time_to_alert_workload.svg",
            workload_ci_df=workload_ci_df,
        )
    if backlog_summary_df is not None:
        write_figure8(
            backlog_summary_df,
            figure_dir / "figure8_dedup_backlog.svg",
            backlog_ci_df=backlog_ci_df,
        )
        write_figure9(
            backlog_summary_df,
            backlog_ci_df,
            figure_dir / "figure9_priority_queues.svg",
        )
    write_captions(figure_dir / "figure_captions.md")
    return figure_dir
