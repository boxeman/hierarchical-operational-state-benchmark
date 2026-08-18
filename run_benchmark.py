import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from ai_mla_monitor.config import (
    FEATURE_COLUMNS,
    FLAT_FEATURE_COLUMNS,
    HIER_NO_PROVENANCE_COLUMNS,
    HIER_NO_SIGNATURE_COLUMNS,
    HIER_NO_UNCERTAINTY_COLUMNS,
    ScenarioConfig,
    SplitConfig,
)
from ai_mla_monitor.data import ADVERSARIAL_FALSE_SYSTEMIC_TYPES, build_dataset
from ai_mla_monitor.evaluation import (
    account_metrics,
    calibration_curve,
    fixed_routing_policy,
    predict_routes,
    recall_by_type,
    routing_confusion,
    routing_metrics,
    routing_policy_candidates,
    routing_utility,
    system_alert_metrics,
    threshold_metrics,
    tune_routing_policy,
    tune_threshold,
)
from ai_mla_monitor.models import GradientBoostedStumps, LogisticRegressionGD, ScoreColumnModel
from ai_mla_monitor.reporting import (
    MANUSCRIPT_METHODS,
    bootstrap_account_cis,
    bootstrap_routing_frontier_cis,
    bootstrap_routing_cis,
    bootstrap_system_alert_cis,
    bootstrap_targeted_cis,
    write_manuscript_figures,
)


DEFAULT_EXPERIMENT_CONFIG = {
    "experiment_name": "default_manuscript_run",
    "output_dir": "outputs/prototype_package",
    "splits": {
        "train_scenarios": 24,
        "validation_scenarios": 8,
        "test_scenarios": 8,
        "train_seed0": 1000,
        "validation_seed0": 2000,
        "test_seed0": 3000,
    },
    "scenario_sweep": {
        "systemic_account_counts": [5, 10, 20, 35, 50],
        "systemic_message_score_means": [0.16, 0.18, 0.20, 0.22],
        "benign_shared_topic_score_means": [0.12, 0.15, 0.18],
        "cluster_size_normalizations": [15, 25, 40],
        "sub_thresholds": [0.15, 0.18, 0.22],
    },
    "evaluation": {
        "review_budgets": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "manuscript_review_budget": 0.15,
        "threshold_min_precision": 0.90,
    },
    "routing": {
        "adversarial_false_alert_penalties": [0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05],
    },
    "bootstrap": {
        "account_n": 300,
        "account_seed": 910,
        "system_alert_n": 300,
        "system_alert_seed": 911,
        "targeted_n": 300,
        "targeted_seed": 912,
        "routing_n": 300,
        "routing_seed": 913,
        "frontier_n": 150,
        "frontier_seed": 914,
        "workload_n": 120,
        "workload_seed": 915,
        "backlog_n": 20,
        "backlog_seed": 916,
    },
    "workload": {
        "horizon_days": 14,
        "daily_capacity_hours": 8.0,
        "route_cost_hours": {
            "auto_handle": 0.0,
            "human_review": 0.25,
            "policy_review": 0.75,
            "product_investigation": 2.0,
        },
    },
    "backlog": {
        "analyst_capacity_hours_per_day": [8.0, 16.0, 32.0],
        "queue_policies": ["fifo", "product_first", "product_policy_first", "severity_weighted"],
    },
    "methods": {
        "practical_methods": [
            "learned_logistic_flat",
            "learned_logistic_hier",
            "ablation_no_uncertainty",
            "ablation_no_signature",
            "ablation_no_provenance",
            "typed_system_hierarchy",
        ],
        "backlog_bootstrap_methods": [
            "learned_logistic_flat",
            "learned_logistic_hier",
            "typed_system_hierarchy",
        ],
        "system_alert_methods": [
            "signature_count_only",
            "signature_mean_risk",
            "system_hierarchy_alert",
            "typed_system_hierarchy_alert",
        ],
    },
}


OUT_DIR = Path(DEFAULT_EXPERIMENT_CONFIG["output_dir"])


def log_step(message):
    print(f"[benchmark] {message}", flush=True)


def deep_update(base, override):
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_experiment_config(path=None):
    cfg = deepcopy(DEFAULT_EXPERIMENT_CONFIG)
    if path is None:
        cfg["config_source"] = "built-in defaults"
        return cfg
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as f:
        override = json.load(f)
    cfg = deep_update(cfg, override)
    cfg["config_source"] = str(path)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the AI-MLA Studio hierarchical monitoring benchmark."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON experiment configuration file. Defaults preserve the manuscript run.",
    )
    return parser.parse_args()


def configured_split_config(cfg):
    split = cfg["splits"]
    return SplitConfig(
        train_scenarios=int(split["train_scenarios"]),
        val_scenarios=int(split["validation_scenarios"]),
        test_scenarios=int(split["test_scenarios"]),
        train_seed0=int(split["train_seed0"]),
        val_seed0=int(split["validation_seed0"]),
        test_seed0=int(split["test_seed0"]),
    )


def scenario_variants(n, seed0, sweep=None):
    variants = []
    sweep = sweep or DEFAULT_EXPERIMENT_CONFIG["scenario_sweep"]
    systemic_ns = sweep["systemic_account_counts"]
    systemic_means = sweep["systemic_message_score_means"]
    benign_means = sweep["benign_shared_topic_score_means"]
    cluster_norms = sweep["cluster_size_normalizations"]
    sub_thresholds = sweep["sub_thresholds"]
    idx = 0
    for i in range(n):
        variants.append(
            ScenarioConfig(
                seed=seed0 + i,
                systemic_n=systemic_ns[idx % len(systemic_ns)],
                systemic_mean=systemic_means[(idx // 2) % len(systemic_means)],
                benign_cluster_mean=benign_means[(idx // 3) % len(benign_means)],
                cluster_norm=cluster_norms[(idx // 5) % len(cluster_norms)],
                sub_thresh=sub_thresholds[(idx // 7) % len(sub_thresholds)],
            )
        )
        idx += 1
    return variants


def build_split(configs, split_name):
    frames = []
    sig_frames = []
    for i, cfg in enumerate(configs):
        df, sig = build_dataset(cfg)
        df = df.copy()
        df["scenario_id"] = f"{split_name}_{i:03d}"
        df["split"] = split_name
        for col in ["scores"]:
            if col in df:
                df = df.drop(columns=[col])
        sig = sig.copy()
        sig["scenario_id"] = f"{split_name}_{i:03d}"
        sig["split"] = split_name
        frames.append(df)
        sig_frames.append(sig)
    return pd.concat(frames, ignore_index=True), pd.concat(sig_frames, ignore_index=True)


def fit_learned_models(train_df):
    x = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = train_df["harmful"].astype(int).to_numpy()
    return {
        "learned_logistic_flat": (
            LogisticRegressionGD(lr=0.08, epochs=900, l2=0.001),
            FLAT_FEATURE_COLUMNS,
        ),
        "learned_logistic_hier": (
            LogisticRegressionGD(lr=0.08, epochs=900, l2=0.001),
            FEATURE_COLUMNS,
        ),
        "ablation_no_uncertainty": (
            LogisticRegressionGD(lr=0.08, epochs=900, l2=0.001),
            HIER_NO_UNCERTAINTY_COLUMNS,
        ),
        "ablation_no_signature": (
            LogisticRegressionGD(lr=0.08, epochs=900, l2=0.001),
            HIER_NO_SIGNATURE_COLUMNS,
        ),
        "ablation_no_provenance": (
            LogisticRegressionGD(lr=0.08, epochs=900, l2=0.001),
            HIER_NO_PROVENANCE_COLUMNS,
        ),
        "boosted_stumps_flat": (
            GradientBoostedStumps(n_estimators=70, lr=0.10),
            FLAT_FEATURE_COLUMNS,
        ),
        "boosted_stumps_hier": (
            GradientBoostedStumps(n_estimators=70, lr=0.10),
            FEATURE_COLUMNS,
        ),
    }, x, y


def train_models(train_df):
    models = {
        "flat_max": ScoreColumnModel("flat_max"),
        "flat_mean": ScoreColumnModel("flat_mean"),
        "flat_top3_mean": ScoreColumnModel("flat_top3_mean"),
        "flat_count_above": ScoreColumnModel("flat_count_above"),
        "account_rule": ScoreColumnModel("account_rule"),
        "account_logistic_rule": ScoreColumnModel("account_logistic_rule"),
        "system_hierarchy": ScoreColumnModel("system_score"),
        "typed_system_hierarchy": ScoreColumnModel("typed_system_score"),
    }
    learned, _, _ = fit_learned_models(train_df)
    for name, (model, cols) in learned.items():
        model.fit(train_df[cols].to_numpy(dtype=float), train_df["harmful"].astype(int).to_numpy())
        models[name] = (model, cols)
    return models


def score_model(model_entry, df):
    if isinstance(model_entry, ScoreColumnModel):
        return model_entry.predict_score(df)
    model, cols = model_entry
    return model.predict_proba(df[cols].to_numpy(dtype=float))


def evaluate_accounts(models, df, split_name, budgets=None):
    rows = []
    type_rows = []
    budgets = budgets or DEFAULT_EXPERIMENT_CONFIG["evaluation"]["review_budgets"]
    for budget in budgets:
        for name, model in models.items():
            score = score_model(model, df)
            row = account_metrics(df, score, budget=budget)
            row["method"] = name
            row["split"] = split_name
            rows.append(row)
            tr = recall_by_type(df, score, budget=budget)
            tr["method"] = name
            tr["split"] = split_name
            tr["budget"] = budget
            type_rows.append(tr)
    return pd.DataFrame(rows), pd.concat(type_rows, ignore_index=True)


def tune_and_apply_thresholds(models, val_df, test_df, min_precision=0.90):
    threshold_rows = []
    test_rows = []
    for name, model in models.items():
        val_score = score_model(model, val_df)
        test_score = score_model(model, test_df)
        tuned = tune_threshold(val_df, val_score, min_precision=min_precision)
        tuned["method"] = name
        tuned["split"] = "validation"
        threshold_rows.append(tuned)
        test_metric = threshold_metrics(test_df, test_score, tuned["threshold"])
        test_metric["method"] = name
        test_metric["split"] = "test"
        test_metric["validation_utility"] = tuned["validation_utility"]
        test_rows.append(test_metric)
    return pd.DataFrame(threshold_rows), pd.DataFrame(test_rows)


def tune_routing_policies(models, val_df, min_precision=0.90):
    rows = []
    policies = {}
    for name, model in models.items():
        val_score = score_model(model, val_df)
        threshold = float(tune_threshold(val_df, val_score, min_precision=min_precision)["threshold"])
        fixed_policy = fixed_routing_policy(threshold)
        tuned_policy = tune_routing_policy(val_df, val_score, name, threshold)
        policies[name] = {
            "threshold": threshold,
            "fixed": fixed_policy,
            "validation_tuned": tuned_policy,
        }
        row = {
            "method": name,
            "base_threshold": threshold,
        }
        for key, value in tuned_policy.items():
            row[key] = value
        rows.append(row)
    return policies, pd.DataFrame(rows)


def evaluate_routing_penalty_sweep(
    models,
    val_df,
    test_df,
    methods=("learned_logistic_hier",),
    penalties=(0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05),
    min_precision=0.90,
):
    rows = []
    account_rows = []
    for method in methods:
        model = models[method]
        val_score = score_model(model, val_df)
        test_score = score_model(model, test_df)
        threshold = float(tune_threshold(val_df, val_score, min_precision=min_precision)["threshold"])
        validation_candidates = routing_policy_candidates(
            val_df,
            val_score,
            method,
            threshold,
            wide_grid=True,
        )
        for penalty in penalties:
            policy = max(
                validation_candidates,
                key=lambda candidate: routing_utility(
                    candidate,
                    adversarial_false_alert_penalty=penalty,
                ),
            )
            routes = predict_routes(test_df, test_score, threshold, method, policy=policy)
            metrics = routing_metrics(test_df, routes)
            metrics["routing_utility"] = routing_utility(
                metrics,
                adversarial_false_alert_penalty=penalty,
            )
            row = {
                "method": method,
                "split": "test",
                "adversarial_false_alert_penalty": float(penalty),
                "threshold": threshold,
            }
            for key, value in policy.items():
                if key in row:
                    continue
                row[key] = value
            row.update(metrics)
            rows.append(row)

            accounts = test_df[
                [
                    "scenario_id",
                    "id",
                    "type",
                    "harmful",
                    "expected_route",
                    "signature",
                    "campaign_id",
                ]
            ].copy()
            accounts["method"] = method
            accounts["split"] = "test"
            accounts["adversarial_false_alert_penalty"] = float(penalty)
            accounts["score"] = test_score
            accounts["threshold"] = float(threshold)
            accounts["predicted_route"] = routes
            for key, value in policy.items():
                if key in accounts.columns:
                    continue
                accounts[key] = value
            account_rows.append(accounts)
    return pd.DataFrame(rows), pd.concat(account_rows, ignore_index=True)


def _stable_unit_interval(*parts):
    key = "::".join(str(part) for part in parts)
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def add_simulated_arrival_times(df, horizon_days=14):
    out = df.copy()
    horizon_hours = float(horizon_days * 24)
    out["arrival_hour"] = [
        horizon_hours * _stable_unit_interval(row.scenario_id, row.id, row.method, row.policy_variant)
        for row in out.itertuples(index=False)
    ]
    out["arrival_day"] = np.floor(out["arrival_hour"] / 24).astype(int)
    return out


def _route_hour_costs(route_costs=None):
    if route_costs is None:
        route_costs = DEFAULT_EXPERIMENT_CONFIG["workload"]["route_cost_hours"]
    return {key: float(value) for key, value in route_costs.items()}


def _time_to_alert_workload_metrics(sub, horizon_days=14, daily_capacity_hours=8.0):
    true_product = sub[sub["expected_route"] == "product_investigation"].copy()
    campaign_rows = []
    campaign_keys = ["scenario_id", "campaign_id", "signature"]
    if "_bootstrap_unit" in true_product.columns:
        campaign_keys = ["_bootstrap_unit"] + campaign_keys
    for _, group in true_product.groupby(campaign_keys, sort=True):
        predicted = group[group["predicted_route"] == "product_investigation"]
        first_relevant = float(group["arrival_hour"].min())
        if predicted.empty:
            campaign_rows.append(
                {
                    "alerted": False,
                    "time_to_alert_hours": np.nan,
                }
            )
        else:
            campaign_rows.append(
                {
                    "alerted": True,
                    "time_to_alert_hours": float(predicted["arrival_hour"].min() - first_relevant),
                }
            )
    campaign_df = pd.DataFrame(campaign_rows)
    if campaign_df.empty:
        campaign_df = pd.DataFrame(columns=["alerted", "time_to_alert_hours"])
    alerted_times = campaign_df.loc[campaign_df["alerted"], "time_to_alert_hours"]
    daily = (
        sub.groupby("arrival_day")
        .agg(
            analyst_hours=("analyst_hours", "sum"),
            human_reviews=("predicted_route", lambda s: int((s == "human_review").sum())),
            policy_reviews=("predicted_route", lambda s: int((s == "policy_review").sum())),
            product_investigations=("predicted_route", lambda s: int((s == "product_investigation").sum())),
        )
        .reindex(range(horizon_days), fill_value=0)
        .reset_index()
    )
    daily["capacity_hours"] = float(daily_capacity_hours)
    daily["over_capacity_hours"] = np.maximum(0.0, daily["analyst_hours"] - daily_capacity_hours)

    product_routes = sub[sub["predicted_route"] == "product_investigation"]
    false_product = product_routes[product_routes["expected_route"] != "product_investigation"]
    adversarial_false_product = false_product[
        false_product["type"].isin(ADVERSARIAL_FALSE_SYSTEMIC_TYPES)
    ]
    return {
        "horizon_days": int(horizon_days),
        "daily_capacity_hours": float(daily_capacity_hours),
        "n_product_campaigns": int(len(campaign_df)),
        "product_campaigns_alerted": int(campaign_df["alerted"].sum()) if len(campaign_df) else 0,
        "product_campaign_alert_rate": float(campaign_df["alerted"].mean()) if len(campaign_df) else 0.0,
        "missed_product_campaigns": int((~campaign_df["alerted"]).sum()) if len(campaign_df) else 0,
        "median_time_to_product_alert_hours": float(alerted_times.median()) if len(alerted_times) else np.nan,
        "p90_time_to_product_alert_hours": float(alerted_times.quantile(0.90)) if len(alerted_times) else np.nan,
        "total_analyst_hours": float(sub["analyst_hours"].sum()),
        "analyst_days": float(sub["analyst_hours"].sum() / daily_capacity_hours),
        "mean_daily_analyst_hours": float(daily["analyst_hours"].mean()),
        "peak_daily_analyst_hours": float(daily["analyst_hours"].max()),
        "days_over_capacity": int((daily["analyst_hours"] > daily_capacity_hours).sum()),
        "total_human_reviews": int((sub["predicted_route"] == "human_review").sum()),
        "total_policy_reviews": int((sub["predicted_route"] == "policy_review").sum()),
        "total_product_investigations": int((sub["predicted_route"] == "product_investigation").sum()),
        "false_product_investigations": int(len(false_product)),
        "adversarial_false_systemic_product_investigations": int(len(adversarial_false_product)),
    }, daily


def simulate_time_to_alert_and_workload(
    routing_accounts_df,
    horizon_days=14,
    daily_capacity_hours=8.0,
    route_costs=None,
):
    route_hours = _route_hour_costs(route_costs)
    df = add_simulated_arrival_times(routing_accounts_df, horizon_days=horizon_days)
    df["analyst_hours"] = df["predicted_route"].map(route_hours).astype(float)
    rows = []
    daily_rows = []
    for (method, policy_variant), sub in df.groupby(["method", "policy_variant"], sort=True):
        metrics, daily = _time_to_alert_workload_metrics(
            sub,
            horizon_days=horizon_days,
            daily_capacity_hours=daily_capacity_hours,
        )
        daily["method"] = method
        daily["policy_variant"] = policy_variant
        daily_rows.append(daily)
        row = {"method": method, "policy_variant": policy_variant}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(daily_rows, ignore_index=True)


def bootstrap_time_to_alert_workload_cis(
    routing_accounts_df,
    horizon_days=14,
    daily_capacity_hours=8.0,
    n_boot=120,
    seed=915,
    methods=None,
    policy_variants=("validation_tuned",),
    route_costs=None,
):
    rng = np.random.default_rng(seed)
    route_hours = _route_hour_costs(route_costs)
    df = add_simulated_arrival_times(routing_accounts_df, horizon_days=horizon_days)
    if methods is not None:
        df = df[df["method"].isin(methods)].copy()
    if policy_variants is not None:
        df = df[df["policy_variant"].isin(policy_variants)].copy()
    df["analyst_hours"] = df["predicted_route"].map(route_hours).astype(float)
    scenarios = np.array(sorted(df["scenario_id"].unique()))
    metrics = [
        "product_campaign_alert_rate",
        "missed_product_campaigns",
        "median_time_to_product_alert_hours",
        "p90_time_to_product_alert_hours",
        "total_analyst_hours",
        "false_product_investigations",
        "adversarial_false_systemic_product_investigations",
    ]
    rows = []
    for (method, policy_variant), group in df.groupby(["method", "policy_variant"], sort=True):
        by_scenario = {sid: group[group["scenario_id"] == sid] for sid in scenarios}
        boot = {metric: [] for metric in metrics}
        for _ in range(n_boot):
            sampled = rng.choice(scenarios, size=len(scenarios), replace=True)
            samples = []
            for sample_idx, sid in enumerate(sampled):
                sample = by_scenario[sid].copy()
                sample["_bootstrap_unit"] = f"{sample_idx:03d}_{sid}"
                samples.append(sample)
            sample_df = pd.concat(samples, ignore_index=True)
            m, _ = _time_to_alert_workload_metrics(
                sample_df,
                horizon_days=horizon_days,
                daily_capacity_hours=daily_capacity_hours,
            )
            for metric in metrics:
                boot[metric].append(m[metric])
        for metric, values in boot.items():
            values = np.asarray(values, dtype=float)
            values = values[~np.isnan(values)]
            if len(values) == 0:
                mean, lo, hi = np.nan, np.nan, np.nan
            else:
                mean = float(np.mean(values))
                lo = float(np.quantile(values, 0.025))
                hi = float(np.quantile(values, 0.975))
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


def simulate_deduplicated_backlog(
    routing_accounts_df,
    horizon_days=14,
    capacities=(8.0, 16.0, 32.0),
    queue_modes=("fifo", "product_first", "product_policy_first", "severity_weighted"),
    methods=None,
    policy_variants=("validation_tuned",),
    route_costs=None,
):
    route_hours = _route_hour_costs(route_costs)
    df = add_simulated_arrival_times(routing_accounts_df, horizon_days=horizon_days)
    if methods is not None:
        df = df[df["method"].isin(methods)].copy()
    if policy_variants is not None:
        df = df[df["policy_variant"].isin(policy_variants)].copy()
    df["analyst_hours"] = df["predicted_route"].map(route_hours).astype(float)
    summary_rows = []
    daily_rows = []
    task_rows = []
    for (method, policy_variant), sub in df.groupby(["method", "policy_variant"], sort=True):
        review_tasks = sub[sub["predicted_route"].isin(["human_review", "policy_review"])].copy()
        review_tasks = review_tasks[
            [
                "scenario_id",
                "id",
                "type",
                "expected_route",
                "predicted_route",
                "signature",
                "campaign_id",
                "arrival_hour",
                "analyst_hours",
                "score",
            ]
        ].copy()
        review_tasks["task_kind"] = review_tasks["predicted_route"]
        review_tasks["dedup_key"] = review_tasks["id"]
        review_tasks["is_true_product_alert"] = False
        review_tasks["is_false_product_alert"] = False
        review_tasks["is_adversarial_false_product_alert"] = False

        product_routes = sub[sub["predicted_route"] == "product_investigation"].copy()
        product_task_rows = []
        for key, group in product_routes.groupby(["scenario_id", "signature", "campaign_id"], sort=True):
            scenario_id, signature, campaign_id = key
            is_true = bool((group["expected_route"] == "product_investigation").any())
            is_adversarial_false = bool(
                (not is_true) and group["type"].isin(ADVERSARIAL_FALSE_SYSTEMIC_TYPES).any()
            )
            first = group.sort_values("arrival_hour").iloc[0]
            product_task_rows.append(
                {
                    "scenario_id": scenario_id,
                    "id": f"dedup::{scenario_id}::{signature}::{campaign_id}",
                    "type": ",".join(sorted(group["type"].astype(str).unique())),
                    "expected_route": "product_investigation" if is_true else "auto_handle",
                    "predicted_route": "product_investigation",
                    "signature": signature,
                    "campaign_id": campaign_id,
                    "arrival_hour": float(first["arrival_hour"]),
                    "analyst_hours": route_hours["product_investigation"],
                    "score": float(group["score"].max()) if "score" in group else 0.0,
                    "task_kind": "product_investigation",
                    "dedup_key": f"{scenario_id}::{signature}::{campaign_id}",
                    "is_true_product_alert": is_true,
                    "is_false_product_alert": not is_true,
                    "is_adversarial_false_product_alert": is_adversarial_false,
                    "duplicate_product_alerts_avoided": int(len(group) - 1),
                }
            )
        product_tasks = pd.DataFrame(product_task_rows)
        if product_tasks.empty:
            product_tasks = pd.DataFrame(
                columns=[
                    "scenario_id",
                    "id",
                    "type",
                    "expected_route",
                    "predicted_route",
                    "signature",
                    "campaign_id",
                    "arrival_hour",
                    "analyst_hours",
                    "score",
                    "task_kind",
                    "dedup_key",
                    "is_true_product_alert",
                    "is_false_product_alert",
                    "is_adversarial_false_product_alert",
                    "duplicate_product_alerts_avoided",
                ]
            )
        if "duplicate_product_alerts_avoided" not in review_tasks:
            review_tasks["duplicate_product_alerts_avoided"] = 0
        tasks = pd.concat([review_tasks, product_tasks], ignore_index=True, sort=False)
        tasks["duplicate_product_alerts_avoided"] = tasks["duplicate_product_alerts_avoided"].fillna(0).astype(int)
        tasks["score"] = tasks["score"].fillna(0.0).astype(float)
        tasks = tasks.sort_values(["arrival_hour", "task_kind", "id"]).reset_index(drop=True)

        for capacity in capacities:
            for queue_mode in queue_modes:
                processed = process_queue(tasks, capacity, queue_mode)
                processed["method"] = method
                processed["policy_variant"] = policy_variant
                processed["capacity_hours_per_day"] = float(capacity)
                processed["queue_mode"] = queue_mode
                task_rows.append(processed)

                daily_df = backlog_daily_summary(tasks, processed, horizon_days, capacity)
                daily_df["method"] = method
                daily_df["policy_variant"] = policy_variant
                daily_df["capacity_hours_per_day"] = float(capacity)
                daily_df["queue_mode"] = queue_mode
                daily_rows.append(daily_df)

                product_alerts = processed[processed["task_kind"] == "product_investigation"].copy()
                true_product = product_alerts[product_alerts["is_true_product_alert"]]
                false_product = product_alerts[product_alerts["is_false_product_alert"]]
                p50_delay = float(true_product["delay_hours"].median()) if len(true_product) else np.nan
                p90_delay = float(true_product["delay_hours"].quantile(0.90)) if len(true_product) else np.nan
                summary_rows.append(
                    {
                        "method": method,
                        "policy_variant": policy_variant,
                        "capacity_hours_per_day": float(capacity),
                        "queue_mode": queue_mode,
                        "n_tasks_after_dedup": int(len(tasks)),
                        "unique_product_alerts": int(len(product_alerts)),
                        "true_unique_product_alerts": int(len(true_product)),
                        "false_unique_product_alerts": int(len(false_product)),
                        "adversarial_false_unique_product_alerts": int(
                            product_alerts["is_adversarial_false_product_alert"].sum()
                        ),
                        "duplicate_product_alerts_avoided": int(
                            product_tasks["duplicate_product_alerts_avoided"].sum()
                        ) if len(product_tasks) else 0,
                        "peak_backlog_hours": float(daily_df["backlog_hours"].max()),
                        "mean_backlog_hours": float(daily_df["backlog_hours"].mean()),
                        "peak_non_product_backlog_hours": float(daily_df["non_product_backlog_hours"].max()),
                        "mean_non_product_backlog_hours": float(daily_df["non_product_backlog_hours"].mean()),
                        "days_with_unresolved_backlog": int((daily_df["backlog_hours"] > 0).sum()),
                        "median_delay_to_product_investigation_hours": p50_delay,
                        "p90_delay_to_product_investigation_hours": p90_delay,
                        "true_product_alerts_resolved_within_24h": int((true_product["delay_hours"] <= 24).sum()),
                        "true_product_alerts_resolved_within_48h": int((true_product["delay_hours"] <= 48).sum()),
                        "true_product_alerts_total": int(len(true_product)),
                        "true_product_alerts_resolved_within_24h_rate": float(
                            (true_product["delay_hours"] <= 24).mean()
                        ) if len(true_product) else np.nan,
                        "true_product_alerts_resolved_within_48h_rate": float(
                            (true_product["delay_hours"] <= 48).mean()
                        ) if len(true_product) else np.nan,
                    }
                )
    return (
        pd.DataFrame(summary_rows),
        pd.concat(daily_rows, ignore_index=True),
        pd.concat(task_rows, ignore_index=True),
    )


def queue_priority_key(row, queue_mode):
    task_kind = str(row.task_kind)
    if queue_mode == "fifo":
        route_priority = 0
        severity = 0.0
    elif queue_mode == "product_first":
        route_priority = 0 if task_kind == "product_investigation" else 1
        severity = 0.0
    elif queue_mode == "product_policy_first":
        route_priority = 0 if task_kind == "product_investigation" else 1 if task_kind == "policy_review" else 2
        severity = 0.0
    elif queue_mode == "severity_weighted":
        route_priority = 0 if task_kind == "product_investigation" else 1 if task_kind == "policy_review" else 2
        severity = -float(getattr(row, "score", 0.0))
    else:
        route_priority = 0
        severity = 0.0
    return (route_priority, severity, float(row.arrival_hour), str(row.id))


def process_queue(tasks, capacity_hours_per_day, queue_mode):
    rate = float(capacity_hours_per_day) / 24.0
    waiting = []
    rows = []
    task_list = list(tasks.sort_values(["arrival_hour", "id"]).itertuples(index=False))
    idx = 0
    now = 0.0
    while idx < len(task_list) or waiting:
        if not waiting and idx < len(task_list) and now < float(task_list[idx].arrival_hour):
            now = float(task_list[idx].arrival_hour)
        while idx < len(task_list) and float(task_list[idx].arrival_hour) <= now:
            waiting.append(task_list[idx])
            idx += 1
        if not waiting:
            continue
        waiting.sort(key=lambda row: queue_priority_key(row, queue_mode))
        task = waiting.pop(0)
        service_hours = float(task.analyst_hours)
        start_hour = max(now, float(task.arrival_hour))
        completion_hour = start_hour + service_hours / rate if service_hours > 0 else start_hour
        now = completion_hour
        rows.append(
            {
                "scenario_id": task.scenario_id,
                "id": task.id,
                "task_kind": task.task_kind,
                "signature": task.signature,
                "campaign_id": task.campaign_id,
                "arrival_hour": float(task.arrival_hour),
                "service_hours": service_hours,
                "score": float(getattr(task, "score", 0.0)),
                "start_hour": float(start_hour),
                "completion_hour": float(completion_hour),
                "delay_hours": float(max(0.0, completion_hour - float(task.arrival_hour))),
                "is_true_product_alert": bool(task.is_true_product_alert),
                "is_false_product_alert": bool(task.is_false_product_alert),
                "is_adversarial_false_product_alert": bool(task.is_adversarial_false_product_alert),
                "duplicate_product_alerts_avoided": int(task.duplicate_product_alerts_avoided),
            }
        )
    return pd.DataFrame(rows)


def backlog_daily_summary(tasks, processed, horizon_days, capacity_hours_per_day):
    daily = []
    for day in range(horizon_days):
        end_hour = float((day + 1) * 24)
        arrived = tasks[tasks["arrival_hour"] <= end_hour]["analyst_hours"].sum()
        processed_hours = min(float(arrived), float(capacity_hours_per_day) * (day + 1))
        backlog = max(0.0, float(arrived - processed_hours))
        pending = processed[
            (processed["arrival_hour"] <= end_hour)
            & (processed["completion_hour"] > end_hour)
        ]
        non_product_backlog = float(
            pending[pending["task_kind"] != "product_investigation"]["service_hours"].sum()
        )
        product_backlog = float(
            pending[pending["task_kind"] == "product_investigation"]["service_hours"].sum()
        )
        day_completed = processed[
            (processed["completion_hour"] > day * 24)
            & (processed["completion_hour"] <= end_hour)
        ]
        daily.append(
            {
                "arrival_day": int(day),
                "arrived_hours_cumulative": float(arrived),
                "processed_capacity_cumulative": float(capacity_hours_per_day) * (day + 1),
                "backlog_hours": float(backlog),
                "non_product_backlog_hours": non_product_backlog,
                "product_backlog_hours": product_backlog,
                "completed_tasks": int(len(day_completed)),
                "completed_product_alerts": int(
                    (day_completed["task_kind"] == "product_investigation").sum()
                ),
            }
        )
    return pd.DataFrame(daily)


def bootstrap_dedup_backlog_cis(
    backlog_tasks_df,
    horizon_days=14,
    capacity=16.0,
    queue_modes=("fifo", "product_first", "product_policy_first", "severity_weighted"),
    n_boot=20,
    seed=916,
    methods=None,
    policy_variants=("validation_tuned",),
):
    rng = np.random.default_rng(seed)
    df = backlog_tasks_df.copy()
    if methods is not None:
        df = df[df["method"].isin(methods)].copy()
    if policy_variants is not None:
        df = df[df["policy_variant"].isin(policy_variants)].copy()
    df = df[df["capacity_hours_per_day"] == float(capacity)].copy()
    df = df[df["queue_mode"].isin(queue_modes)].copy()
    scenarios = np.array(sorted(df["scenario_id"].unique()))
    metrics = [
        "peak_backlog_hours",
        "mean_backlog_hours",
        "days_with_unresolved_backlog",
        "p90_delay_to_product_investigation_hours",
        "true_product_alerts_resolved_within_24h_rate",
        "true_product_alerts_resolved_within_48h_rate",
        "false_unique_product_alerts",
        "peak_non_product_backlog_hours",
        "mean_non_product_backlog_hours",
    ]
    rows = []
    grouped = df.groupby(["method", "policy_variant", "queue_mode"], sort=True)
    for (method, policy_variant, queue_mode), group in grouped:
        by_scenario = {sid: group[group["scenario_id"] == sid] for sid in scenarios}
        boot = {metric: [] for metric in metrics}
        for _ in range(n_boot):
            sampled = rng.choice(scenarios, size=len(scenarios), replace=True)
            sample_df = pd.concat([by_scenario[sid] for sid in sampled], ignore_index=True)
            m = backlog_metrics_from_processed_tasks(sample_df, horizon_days, capacity)
            for metric in metrics:
                boot[metric].append(float(m[metric]))
        for metric in metrics:
            values = np.asarray(boot[metric], dtype=float)
            values = values[~np.isnan(values)]
            if len(values) == 0:
                mean, lo, hi = np.nan, np.nan, np.nan
            else:
                mean = float(np.mean(values))
                lo = float(np.quantile(values, 0.025))
                hi = float(np.quantile(values, 0.975))
            rows.append(
                {
                    "method": method,
                    "policy_variant": policy_variant,
                    "capacity_hours_per_day": float(capacity),
                    "queue_mode": queue_mode,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_boot": n_boot,
                }
            )
    return pd.DataFrame(rows)


def backlog_metrics_from_processed_tasks(processed, horizon_days, capacity_hours_per_day):
    daily_rows = []
    for day in range(horizon_days):
        end_hour = float((day + 1) * 24)
        arrived = processed[processed["arrival_hour"] <= end_hour]["service_hours"].sum()
        processed_hours = min(float(arrived), float(capacity_hours_per_day) * (day + 1))
        pending = processed[
            (processed["arrival_hour"] <= end_hour)
            & (processed["completion_hour"] > end_hour)
        ]
        daily_rows.append(
            {
                "backlog_hours": max(0.0, float(arrived - processed_hours)),
                "non_product_backlog_hours": float(
                    pending[pending["task_kind"] != "product_investigation"]["service_hours"].sum()
                ),
            }
        )
    daily = pd.DataFrame(daily_rows)
    product = processed[processed["task_kind"] == "product_investigation"]
    true_product = product[product["is_true_product_alert"]]
    false_product = product[product["is_false_product_alert"]]
    return {
        "peak_backlog_hours": float(daily["backlog_hours"].max()),
        "mean_backlog_hours": float(daily["backlog_hours"].mean()),
        "days_with_unresolved_backlog": int((daily["backlog_hours"] > 0).sum()),
        "p90_delay_to_product_investigation_hours": float(true_product["delay_hours"].quantile(0.90)) if len(true_product) else np.nan,
        "true_product_alerts_resolved_within_24h_rate": float((true_product["delay_hours"] <= 24).mean()) if len(true_product) else np.nan,
        "true_product_alerts_resolved_within_48h_rate": float((true_product["delay_hours"] <= 48).mean()) if len(true_product) else np.nan,
        "false_unique_product_alerts": int(len(false_product)),
        "peak_non_product_backlog_hours": float(daily["non_product_backlog_hours"].max()),
        "mean_non_product_backlog_hours": float(daily["non_product_backlog_hours"].mean()),
    }


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


def build_summary_results_table(
    account_metrics_df,
    calibration_summary_df,
    alert_metrics_df,
    targeted_diagnostics_df,
    routing_metrics_df,
    adversarial_routing_stress_df,
    routing_penalty_sweep_df,
    time_to_alert_df,
    backlog_summary_df,
    manuscript_budget=0.15,
    backlog_capacity=16.0,
):
    rows = []

    def f(value, digits=3):
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"

    test_15 = account_metrics_df[
        (account_metrics_df["split"] == "test") & (account_metrics_df["budget"] == manuscript_budget)
    ]
    flat = test_15[test_15["method"] == "learned_logistic_flat"].iloc[0]
    hier = test_15[test_15["method"] == "learned_logistic_hier"].iloc[0]
    rows.append(
        {
            "layer": "Account review",
            "figure": "Figure 1",
            "question": "Does hierarchy improve finite-budget account triage?",
            "baseline": f"learned flat HW recall {f(flat['harm_weighted_recall'])}, precision {f(flat['precision'])}",
            "hierarchical_result": f"learned hierarchy HW recall {f(hier['harm_weighted_recall'])}, precision {f(hier['precision'])}",
            "interpretation": "Hierarchy improves harm capture at the account-review layer while retaining high precision.",
        }
    )

    targeted = targeted_diagnostics_df.copy()
    uncertainty_bad = targeted[
        (targeted["method"] == "ablation_no_uncertainty")
        & (targeted["type"] == "uncertainty_legit_ambiguous")
    ]["selected_rate_at_tuned_threshold"].mean()
    signature_bad = targeted[
        (targeted["method"] == "ablation_no_signature")
        & (targeted["type"] == "signature_matched_benign")
    ]["selected_rate_at_tuned_threshold"].mean()
    provenance_bad = targeted[
        (targeted["method"] == "ablation_no_provenance")
        & (targeted["type"] == "low_provenance_artifact")
    ]["selected_rate_at_tuned_threshold"].mean()
    full_unc = targeted[
        (targeted["method"] == "learned_logistic_hier")
        & (targeted["type"] == "uncertainty_legit_ambiguous")
    ]["selected_rate_at_tuned_threshold"].mean()
    full_sig = targeted[
        (targeted["method"] == "learned_logistic_hier")
        & (targeted["type"] == "signature_matched_benign")
    ]["selected_rate_at_tuned_threshold"].mean()
    full_prov = targeted[
        (targeted["method"] == "learned_logistic_hier")
        & (targeted["type"] == "low_provenance_artifact")
    ]["selected_rate_at_tuned_threshold"].mean()
    rows.append(
        {
            "layer": "State components",
            "figure": "Figure 2",
            "question": "Does each state field resolve a distinct failure mode?",
            "baseline": f"ablated false-selection rates: uncertainty {f(uncertainty_bad)}, signature {f(signature_bad)}, provenance {f(provenance_bad)}",
            "hierarchical_result": f"full hierarchy false-selection rates: uncertainty {f(full_unc)}, signature {f(full_sig)}, provenance {f(full_prov)}",
            "interpretation": "Targeted ablations show uncertainty, signature/campaign, and provenance each suppress a different hard-negative failure mode.",
        }
    )

    cal = calibration_summary_df[calibration_summary_df["split"] == "test"]
    flat_cal = cal[cal["method"] == "learned_logistic_flat"].iloc[0]
    hier_cal = cal[cal["method"] == "learned_logistic_hier"].iloc[0]
    rows.append(
        {
            "layer": "Calibration",
            "figure": "Figure 3",
            "question": "Does hierarchical state preserve useful score calibration?",
            "baseline": f"learned flat ECE {f(flat_cal['ece'])}, Brier {f(flat_cal['brier'])}",
            "hierarchical_result": f"learned hierarchy ECE {f(hier_cal['ece'])}, Brier {f(hier_cal['brier'])}",
            "interpretation": "The learned hierarchy is better calibrated than the learned flat model in this held-out run.",
        }
    )

    alerts = (
        alert_metrics_df[alert_metrics_df["split"] == "test"]
        .groupby("method")
        .agg(
            product_alert_hit_rate=("product_alert_hit", "mean"),
            exploit_rank=("exploit_A_rank", "mean"),
            false_systemic_alerts=("false_systemic_alerts", "mean"),
        )
    )
    mean_risk = alerts.loc["signature_mean_risk"]
    typed_alert = alerts.loc["typed_system_hierarchy_alert"]
    rows.append(
        {
            "layer": "System alerts",
            "figure": "Figure 3",
            "question": "Can system-level state surface weak coordinated exploits?",
            "baseline": f"signature mean-risk hit rate {f(mean_risk['product_alert_hit_rate'])}, rank {f(mean_risk['exploit_rank'])}",
            "hierarchical_result": f"typed hierarchy hit rate {f(typed_alert['product_alert_hit_rate'])}, rank {f(typed_alert['exploit_rank'])}",
            "interpretation": "System-level state improves exploit ranking, while false systemic alerts remain a central constraint.",
        }
    )

    routing = routing_metrics_df[
        (routing_metrics_df["split"] == "test")
        & (routing_metrics_df["policy_variant"] == "validation_tuned")
    ]
    flat_route = routing[routing["method"] == "learned_logistic_flat"].iloc[0]
    hier_route = routing[routing["method"] == "learned_logistic_hier"].iloc[0]
    rows.append(
        {
            "layer": "Routing",
            "figure": "Figure 4",
            "question": "Does hierarchy improve operational escalation quality?",
            "baseline": f"learned flat macro F1 {f(flat_route['macro_route_f1'])}, utility {f(flat_route['routing_utility'])}",
            "hierarchical_result": f"learned hierarchy macro F1 {f(hier_route['macro_route_f1'])}, utility {f(hier_route['routing_utility'])}",
            "interpretation": "Hierarchy adds policy/product routing capacity beyond binary account review.",
        }
    )

    adv = adversarial_routing_stress_df[
        (adversarial_routing_stress_df["split"] == "test")
        & (adversarial_routing_stress_df["policy_variant"] == "validation_tuned")
        & (adversarial_routing_stress_df["type"] == "ALL_ADVERSARIAL_FALSE_SYSTEMIC")
    ]
    rule_adv = adv[adv["method"] == "typed_system_hierarchy"].iloc[0]
    hier_adv = adv[adv["method"] == "learned_logistic_hier"].iloc[0]
    rows.append(
        {
            "layer": "False-systemic stress",
            "figure": "Figure 5",
            "question": "Does hierarchy avoid over-alerting on benign coordination?",
            "baseline": f"typed rule hierarchy product-routes {f(rule_adv['product_investigation_rate'])} of adversarial lookalikes",
            "hierarchical_result": f"learned hierarchy product-routes {f(hier_adv['product_investigation_rate'])} of adversarial lookalikes",
            "interpretation": "Learned hierarchical state suppresses most benign coordinated false product investigations; brittle rule hierarchy does not.",
        }
    )

    low_penalty = routing_penalty_sweep_df.sort_values("adversarial_false_alert_penalty").iloc[0]
    high_penalty = routing_penalty_sweep_df.sort_values("adversarial_false_alert_penalty").iloc[-1]
    rows.append(
        {
            "layer": "Product-alert frontier",
            "figure": "Figure 6",
            "question": "What is the tradeoff between product recall and benign systemic false alerts?",
            "baseline": f"low penalty misses {int(low_penalty['missed_product_investigations'])}, false systemic alerts {int(low_penalty['adversarial_false_systemic_product_investigations'])}",
            "hierarchical_result": f"high penalty misses {int(high_penalty['missed_product_investigations'])}, false systemic alerts {int(high_penalty['adversarial_false_systemic_product_investigations'])}",
            "interpretation": "The hierarchy exposes an operating frontier rather than a single best threshold.",
        }
    )

    workload = time_to_alert_df[
        (time_to_alert_df["policy_variant"] == "validation_tuned")
    ]
    typed_work = workload[workload["method"] == "typed_system_hierarchy"].iloc[0]
    hier_work = workload[workload["method"] == "learned_logistic_hier"].iloc[0]
    rows.append(
        {
            "layer": "Workload/time-to-alert",
            "figure": "Figure 7",
            "question": "Does routing quality translate into practical analyst workload?",
            "baseline": f"typed rule hierarchy {f(typed_work['total_analyst_hours'])} analyst hours, false product investigations {int(typed_work['false_product_investigations'])}",
            "hierarchical_result": f"learned hierarchy {f(hier_work['total_analyst_hours'])} analyst hours, false product investigations {int(hier_work['false_product_investigations'])}",
            "interpretation": "Learned hierarchy preserves product-campaign alerting with much lower workload than brittle system rules.",
        }
    )

    backlog = backlog_summary_df[
        (backlog_summary_df["policy_variant"] == "validation_tuned")
        & (backlog_summary_df["capacity_hours_per_day"] == backlog_capacity)
        & (backlog_summary_df["queue_mode"] == "fifo")
    ]
    typed_backlog = backlog[backlog["method"] == "typed_system_hierarchy"].iloc[0]
    hier_backlog = backlog[backlog["method"] == "learned_logistic_hier"].iloc[0]
    rows.append(
        {
            "layer": "Deduplicated backlog",
            "figure": "Figure 8",
            "question": "Does finite analyst capacity preserve the hierarchy advantage?",
            "baseline": f"typed rule hierarchy peak backlog {f(typed_backlog['peak_backlog_hours'])}, false unique alerts {int(typed_backlog['false_unique_product_alerts'])}",
            "hierarchical_result": f"learned hierarchy peak backlog {f(hier_backlog['peak_backlog_hours'])}, false unique alerts {int(hier_backlog['false_unique_product_alerts'])}",
            "interpretation": "Deduplication reduces repeated alerts, but backlog remains a capacity-planning constraint.",
        }
    )

    priority = backlog_summary_df[
        (backlog_summary_df["policy_variant"] == "validation_tuned")
        & (backlog_summary_df["capacity_hours_per_day"] == backlog_capacity)
        & (backlog_summary_df["method"] == "learned_logistic_hier")
    ]
    fifo = priority[priority["queue_mode"] == "fifo"].iloc[0]
    product_first = priority[priority["queue_mode"] == "product_first"].iloc[0]
    rows.append(
        {
            "layer": "Priority queues",
            "figure": "Figure 9",
            "question": "Can scarce analyst capacity protect product-level response?",
            "baseline": f"FIFO p90 product delay {f(fifo['p90_delay_to_product_investigation_hours'])} h, 48h resolution {f(fifo['true_product_alerts_resolved_within_48h_rate'])}",
            "hierarchical_result": f"product-first p90 product delay {f(product_first['p90_delay_to_product_investigation_hours'])} h, 48h resolution {f(product_first['true_product_alerts_resolved_within_48h_rate'])}",
            "interpretation": "Priority queueing sharply improves product response while shifting some backlog to non-product work.",
        }
    )

    return pd.DataFrame(rows)


def evaluate_routing(models, routing_policies, eval_df, split_name):
    routing_rows = []
    confusion_rows = []
    account_rows = []
    for name, model in models.items():
        eval_score = score_model(model, eval_df)
        method_policies = routing_policies[name]
        threshold = float(method_policies["threshold"])
        for variant, policy in [
            ("fixed", method_policies["fixed"]),
            ("validation_tuned", method_policies["validation_tuned"]),
        ]:
            routes = predict_routes(eval_df, eval_score, threshold, name, policy=policy)
            row = routing_metrics(eval_df, routes)
            row["routing_utility"] = routing_utility(row)
            row["method"] = name
            row["split"] = split_name
            row["policy_variant"] = variant
            row["threshold"] = float(threshold)
            for key, value in policy.items():
                if key in row or key in {"route_accuracy", "macro_route_f1"}:
                    continue
                row[key] = value
            routing_rows.append(row)

            conf = routing_confusion(eval_df, routes)
            conf["method"] = name
            conf["split"] = split_name
            conf["policy_variant"] = variant
            confusion_rows.append(conf)

            accounts = eval_df[
                [
                    "scenario_id",
                    "id",
                    "type",
                    "harmful",
                    "expected_route",
                    "signature",
                    "campaign_id",
                ]
            ].copy()
            accounts["method"] = name
            accounts["split"] = split_name
            accounts["policy_variant"] = variant
            accounts["score"] = eval_score
            accounts["threshold"] = float(threshold)
            accounts["predicted_route"] = routes
            account_rows.append(accounts)

    return (
        pd.DataFrame(routing_rows),
        pd.concat(confusion_rows, ignore_index=True),
        pd.concat(account_rows, ignore_index=True),
    )


def evaluate_calibration(models, val_df, test_df):
    summary_rows = []
    curve_rows = []
    for split_name, df in [("validation", val_df), ("test", test_df)]:
        for name, model in models.items():
            score = score_model(model, df)
            curve, summary = calibration_curve(df, score, n_bins=10)
            summary["method"] = name
            summary["split"] = split_name
            summary_rows.append(summary)
            curve["method"] = name
            curve["split"] = split_name
            curve_rows.append(curve)
    return pd.DataFrame(summary_rows), pd.concat(curve_rows, ignore_index=True)


def targeted_ablation_diagnostics(models, val_df, test_df, min_precision=0.90):
    target_types = [
        "uncertainty_required_harm",
        "uncertainty_legit_ambiguous",
        "signature_required_campaign",
        "signature_matched_benign",
        "provenance_required_harm",
        "low_provenance_artifact",
    ]
    methods = [
        "learned_logistic_flat",
        "learned_logistic_hier",
        "ablation_no_uncertainty",
        "ablation_no_signature",
        "ablation_no_provenance",
    ]
    rows = []
    for method in methods:
        val_score = score_model(models[method], val_df)
        score = score_model(models[method], test_df)
        tmp = test_df.copy()
        tmp["_score"] = score
        threshold = tune_threshold(val_df, val_score, min_precision=min_precision)["threshold"]
        tmp["_selected"] = tmp["_score"] >= threshold
        for archetype in target_types:
            sub = tmp[tmp["type"] == archetype]
            if len(sub) == 0:
                continue
            rows.append(
                {
                    "method": method,
                    "type": archetype,
                    "n": int(len(sub)),
                    "harmful": bool(sub["harmful"].iloc[0]),
                    "mean_score": float(sub["_score"].mean()),
                    "selected_rate_at_tuned_threshold": float(sub["_selected"].mean()),
                    "threshold": float(threshold),
                }
            )
    return pd.DataFrame(rows)


def targeted_ablation_diagnostics_by_scenario(models, val_df, test_df, min_precision=0.90):
    target_types = [
        "uncertainty_required_harm",
        "uncertainty_legit_ambiguous",
        "signature_required_campaign",
        "signature_matched_benign",
        "provenance_required_harm",
        "low_provenance_artifact",
    ]
    methods = [
        "learned_logistic_flat",
        "learned_logistic_hier",
        "ablation_no_uncertainty",
        "ablation_no_signature",
        "ablation_no_provenance",
    ]
    rows = []
    thresholds = {}
    for method in methods:
        val_score = score_model(models[method], val_df)
        thresholds[method] = tune_threshold(val_df, val_score, min_precision=min_precision)["threshold"]
    for scenario_id, scenario_df in test_df.groupby("scenario_id"):
        for method in methods:
            score = score_model(models[method], scenario_df)
            tmp = scenario_df.copy()
            tmp["_score"] = score
            tmp["_selected"] = tmp["_score"] >= thresholds[method]
            for archetype in target_types:
                sub = tmp[tmp["type"] == archetype]
                if len(sub) == 0:
                    continue
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "method": method,
                        "type": archetype,
                        "n": int(len(sub)),
                        "harmful": bool(sub["harmful"].iloc[0]),
                        "mean_score": float(sub["_score"].mean()),
                        "selected_rate_at_tuned_threshold": float(sub["_selected"].mean()),
                        "threshold": float(thresholds[method]),
                    }
                )
    return pd.DataFrame(rows)


def account_score_table(models, df, methods):
    score_df = df[
        [
            "scenario_id",
            "id",
            "type",
            "harmful",
            "harm_weight",
            "expected_route",
            "signature",
            "campaign_id",
        ]
    ].copy()
    for method in methods:
        score_df[method] = score_model(models[method], df)
    return score_df


def evaluate_system_alerts(sig, split_name):
    rows = []
    methods = {
        "signature_count_only": "signature_count_only",
        "signature_mean_risk": "signature_mean_risk_detector",
        "system_hierarchy_alert": "system_alert_score",
        "typed_system_hierarchy_alert": "typed_system_alert_score",
    }
    for scenario_id, sub in sig.groupby("scenario_id"):
        for name, col in methods.items():
            row = system_alert_metrics(sub, col, alert_budget=3)
            row["method"] = name
            row["split"] = split_name
            row["scenario_id"] = scenario_id
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_adversarial_routing_stress(routing_accounts_df):
    stress = routing_accounts_df[
        routing_accounts_df["type"].isin(ADVERSARIAL_FALSE_SYSTEMIC_TYPES)
    ].copy()
    if stress.empty:
        return pd.DataFrame()
    rows = []
    for (split, method, policy_variant), sub in stress.groupby(["split", "method", "policy_variant"]):
        rows.append(
            {
                "split": split,
                "method": method,
                "policy_variant": policy_variant,
                "n_adversarial_accounts": int(len(sub)),
                "product_investigation_rate": float((sub["predicted_route"] == "product_investigation").mean()),
                "unnecessary_product_investigations": int(
                    (sub["predicted_route"] == "product_investigation").sum()
                ),
                "policy_review_rate": float((sub["predicted_route"] == "policy_review").mean()),
                "human_review_rate": float((sub["predicted_route"] == "human_review").mean()),
                "auto_handle_rate": float((sub["predicted_route"] == "auto_handle").mean()),
            }
        )
    for (split, method, policy_variant, archetype), sub in stress.groupby(
        ["split", "method", "policy_variant", "type"]
    ):
        rows.append(
            {
                "split": split,
                "method": method,
                "policy_variant": policy_variant,
                "type": archetype,
                "n_adversarial_accounts": int(len(sub)),
                "product_investigation_rate": float((sub["predicted_route"] == "product_investigation").mean()),
                "unnecessary_product_investigations": int(
                    (sub["predicted_route"] == "product_investigation").sum()
                ),
                "policy_review_rate": float((sub["predicted_route"] == "policy_review").mean()),
                "human_review_rate": float((sub["predicted_route"] == "human_review").mean()),
                "auto_handle_rate": float((sub["predicted_route"] == "auto_handle").mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["type"] = out["type"].fillna("ALL_ADVERSARIAL_FALSE_SYSTEMIC")
    return out


def write_calibration_svg(calibration_bins_df, out_path):
    methods = [
        "learned_logistic_flat",
        "learned_logistic_hier",
        "ablation_no_uncertainty",
        "ablation_no_signature",
        "ablation_no_provenance",
        "boosted_stumps_flat",
        "boosted_stumps_hier",
    ]
    colors = {
        "learned_logistic_flat": "#1f77b4",
        "learned_logistic_hier": "#ff7f0e",
        "ablation_no_uncertainty": "#2ca02c",
        "ablation_no_signature": "#d62728",
        "ablation_no_provenance": "#9467bd",
        "boosted_stumps_flat": "#8c564b",
        "boosted_stumps_hier": "#17becf",
    }
    df = calibration_bins_df[
        (calibration_bins_df["split"] == "test")
        & (calibration_bins_df["method"].isin(methods))
        & (calibration_bins_df["n"] > 0)
    ].copy()
    width, height = 920, 620
    left, top, plot = 90, 55, 440
    legend_x = 575

    def sx(x):
        return left + x * plot

    def sy(y):
        return top + (1 - y) * plot

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="90" y="30" font-family="Arial" font-size="20" font-weight="600">Calibration curves on held-out test scenarios</text>',
        f'<rect x="{left}" y="{top}" width="{plot}" height="{plot}" fill="none" stroke="#222" stroke-width="1"/>',
    ]
    for i in range(6):
        v = i / 5
        x = sx(v)
        y = sy(v)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot}" stroke="#ddd"/>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{x-10:.1f}" y="{top+plot+24}" font-family="Arial" font-size="12">{v:.1f}</text>')
        parts.append(f'<text x="{left-38}" y="{y+4:.1f}" font-family="Arial" font-size="12">{v:.1f}</text>')
    parts.append(f'<line x1="{left}" y1="{top+plot}" x2="{left+plot}" y2="{top}" stroke="#111" stroke-dasharray="5,5"/>')
    parts.append(f'<text x="{left+145}" y="{top+plot+52}" font-family="Arial" font-size="14">Mean predicted score</text>')
    parts.append(f'<text x="22" y="{top+260}" font-family="Arial" font-size="14" transform="rotate(-90 22 {top+260})">Empirical harmful rate</text>')

    for method in methods:
        sub = df[df["method"] == method].sort_values("mean_score")
        if sub.empty:
            continue
        points = [(sx(r["mean_score"]), sy(r["empirical_rate"])) for _, r in sub.iterrows()]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        color = colors[method]
        parts.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')

    parts.append(f'<text x="{legend_x}" y="80" font-family="Arial" font-size="15" font-weight="600">Methods</text>')
    for i, method in enumerate(methods):
        y = 110 + i * 28
        color = colors[method]
        label = method.replace("_", " ")
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<circle cx="{legend_x+14}" cy="{y}" r="3" fill="{color}"/>')
        parts.append(f'<text x="{legend_x+38}" y="{y+4}" font-family="Arial" font-size="13">{label}</text>')
    parts.append('<text x="575" y="340" font-family="Arial" font-size="13">Dashed line = perfect calibration.</text>')
    parts.append('<text x="575" y="362" font-family="Arial" font-size="13">Curves below the line overestimate risk;</text>')
    parts.append('<text x="575" y="384" font-family="Arial" font-size="13">curves above the line underestimate risk.</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    global OUT_DIR
    args = parse_args()
    experiment_cfg = load_experiment_config(args.config)
    OUT_DIR = Path(experiment_cfg["output_dir"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(experiment_cfg, f, indent=2)
    if args.config is not None:
        shutil.copyfile(args.config, OUT_DIR / "input_config.json")

    split_cfg = configured_split_config(experiment_cfg)
    scenario_sweep = experiment_cfg["scenario_sweep"]
    review_budgets = [float(x) for x in experiment_cfg["evaluation"]["review_budgets"]]
    manuscript_budget = float(experiment_cfg["evaluation"]["manuscript_review_budget"])
    threshold_min_precision = float(experiment_cfg["evaluation"]["threshold_min_precision"])
    route_costs = experiment_cfg["workload"]["route_cost_hours"]
    horizon_days = int(experiment_cfg["workload"]["horizon_days"])
    daily_capacity_hours = float(experiment_cfg["workload"]["daily_capacity_hours"])
    practical_methods = list(experiment_cfg["methods"]["practical_methods"])
    backlog_bootstrap_methods = list(experiment_cfg["methods"]["backlog_bootstrap_methods"])
    system_alert_methods = list(experiment_cfg["methods"]["system_alert_methods"])
    backlog_capacities = [float(x) for x in experiment_cfg["backlog"]["analyst_capacity_hours_per_day"]]
    queue_policies = list(experiment_cfg["backlog"]["queue_policies"])
    bootstrap_cfg = experiment_cfg["bootstrap"]

    log_step("building train/validation/test scenarios")
    train_df, train_sig = build_split(
        scenario_variants(split_cfg.train_scenarios, split_cfg.train_seed0, scenario_sweep),
        "train",
    )
    val_df, val_sig = build_split(
        scenario_variants(split_cfg.val_scenarios, split_cfg.val_seed0, scenario_sweep),
        "validation",
    )
    test_df, test_sig = build_split(
        scenario_variants(split_cfg.test_scenarios, split_cfg.test_seed0, scenario_sweep),
        "test",
    )

    log_step("training models")
    models = train_models(train_df)

    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        df.to_csv(OUT_DIR / f"{name}_accounts.csv", index=False)
    pd.concat([train_sig, val_sig, test_sig], ignore_index=True).to_csv(
        OUT_DIR / "signature_state_all_splits.csv", index=False
    )

    log_step("evaluating account-review and system-alert metrics")
    account_tables = []
    type_tables = []
    for split_name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        a, t = evaluate_accounts(models, df, split_name, budgets=review_budgets)
        account_tables.append(a)
        type_tables.append(t)
    account_metrics_df = pd.concat(account_tables, ignore_index=True)
    type_recall_df = pd.concat(type_tables, ignore_index=True)

    alert_metrics_df = pd.concat(
        [
            evaluate_system_alerts(train_sig, "train"),
            evaluate_system_alerts(val_sig, "validation"),
            evaluate_system_alerts(test_sig, "test"),
        ],
        ignore_index=True,
    )
    tuned_thresholds_df, tuned_test_df = tune_and_apply_thresholds(
        models,
        val_df,
        test_df,
        min_precision=threshold_min_precision,
    )
    log_step("tuning routing policies")
    routing_policies, routing_policy_df = tune_routing_policies(
        models,
        val_df,
        min_precision=threshold_min_precision,
    )
    routing_tables = []
    routing_conf_tables = []
    routing_account_tables = []
    for split_name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        rm, rc, ra = evaluate_routing(models, routing_policies, df, split_name)
        routing_tables.append(rm)
        routing_conf_tables.append(rc)
        if split_name == "test":
            routing_account_tables.append(ra)
    routing_metrics_df = pd.concat(routing_tables, ignore_index=True)
    routing_confusion_df = pd.concat(routing_conf_tables, ignore_index=True)
    routing_accounts_df = pd.concat(routing_account_tables, ignore_index=True)
    adversarial_routing_stress_df = evaluate_adversarial_routing_stress(routing_accounts_df)
    log_step("evaluating product-investigation frontier")
    routing_penalty_sweep_df, routing_frontier_accounts_df = evaluate_routing_penalty_sweep(
        models,
        val_df,
        test_df,
        penalties=tuple(float(x) for x in experiment_cfg["routing"]["adversarial_false_alert_penalties"]),
        min_precision=threshold_min_precision,
    )
    log_step("bootstrapping product-investigation frontier")
    bootstrap_frontier_df = bootstrap_routing_frontier_cis(
        routing_frontier_accounts_df,
        n_boot=int(bootstrap_cfg["frontier_n"]),
        seed=int(bootstrap_cfg["frontier_seed"]),
    )
    log_step("simulating time-to-alert and analyst workload")
    time_to_alert_df, analyst_workload_daily_df = simulate_time_to_alert_and_workload(
        routing_accounts_df,
        horizon_days=horizon_days,
        daily_capacity_hours=daily_capacity_hours,
        route_costs=route_costs,
    )
    log_step("bootstrapping time-to-alert and analyst workload")
    bootstrap_workload_df = bootstrap_time_to_alert_workload_cis(
        routing_accounts_df,
        horizon_days=horizon_days,
        daily_capacity_hours=daily_capacity_hours,
        n_boot=int(bootstrap_cfg["workload_n"]),
        seed=int(bootstrap_cfg["workload_seed"]),
        methods=practical_methods,
        policy_variants=("validation_tuned",),
        route_costs=route_costs,
    )
    log_step("simulating alert deduplication and analyst backlog")
    backlog_summary_df, backlog_daily_df, backlog_tasks_df = simulate_deduplicated_backlog(
        routing_accounts_df,
        horizon_days=horizon_days,
        capacities=tuple(backlog_capacities),
        queue_modes=tuple(queue_policies),
        methods=practical_methods,
        policy_variants=("validation_tuned",),
        route_costs=route_costs,
    )
    log_step("bootstrapping alert deduplication and backlog")
    bootstrap_backlog_df = bootstrap_dedup_backlog_cis(
        backlog_tasks_df,
        horizon_days=horizon_days,
        capacity=16.0 if 16.0 in backlog_capacities else backlog_capacities[0],
        queue_modes=tuple(queue_policies),
        n_boot=int(bootstrap_cfg["backlog_n"]),
        seed=int(bootstrap_cfg["backlog_seed"]),
        methods=backlog_bootstrap_methods,
        policy_variants=("validation_tuned",),
    )
    log_step("evaluating calibration and targeted ablations")
    calibration_summary_df, calibration_bins_df = evaluate_calibration(models, val_df, test_df)
    targeted_diagnostics_df = targeted_ablation_diagnostics(
        models,
        val_df,
        test_df,
        min_precision=threshold_min_precision,
    )
    targeted_scenario_df = targeted_ablation_diagnostics_by_scenario(
        models,
        val_df,
        test_df,
        min_precision=threshold_min_precision,
    )
    score_df = account_score_table(models, test_df, MANUSCRIPT_METHODS)
    bootstrap_account_df = bootstrap_account_cis(
        score_df,
        MANUSCRIPT_METHODS,
        budget=manuscript_budget,
        n_boot=int(bootstrap_cfg["account_n"]),
        seed=int(bootstrap_cfg["account_seed"]),
    )
    bootstrap_system_df = bootstrap_system_alert_cis(
        alert_metrics_df,
        system_alert_methods,
        n_boot=int(bootstrap_cfg["system_alert_n"]),
        seed=int(bootstrap_cfg["system_alert_seed"]),
    )
    bootstrap_targeted_df = bootstrap_targeted_cis(
        targeted_scenario_df,
        n_boot=int(bootstrap_cfg["targeted_n"]),
        seed=int(bootstrap_cfg["targeted_seed"]),
    )
    bootstrap_routing_df = bootstrap_routing_cis(
        routing_accounts_df,
        n_boot=int(bootstrap_cfg["routing_n"]),
        seed=int(bootstrap_cfg["routing_seed"]),
    )
    summary_results_df = build_summary_results_table(
        account_metrics_df,
        calibration_summary_df,
        alert_metrics_df,
        targeted_diagnostics_df,
        routing_metrics_df,
        adversarial_routing_stress_df,
        routing_penalty_sweep_df,
        time_to_alert_df,
        backlog_summary_df,
        manuscript_budget=manuscript_budget,
        backlog_capacity=16.0 if 16.0 in backlog_capacities else backlog_capacities[0],
    )

    log_step("saving outputs and writing figures")
    account_metrics_df.to_csv(OUT_DIR / "account_metrics_by_split.csv", index=False)
    type_recall_df.to_csv(OUT_DIR / "type_recall_by_split.csv", index=False)
    alert_metrics_df.to_csv(OUT_DIR / "system_alert_metrics_by_split.csv", index=False)
    tuned_thresholds_df.to_csv(OUT_DIR / "validation_tuned_thresholds.csv", index=False)
    tuned_test_df.to_csv(OUT_DIR / "test_metrics_at_validation_thresholds.csv", index=False)
    routing_metrics_df.to_csv(OUT_DIR / "routing_metrics_by_split.csv", index=False)
    routing_confusion_df.to_csv(OUT_DIR / "routing_confusion_by_split.csv", index=False)
    routing_accounts_df.to_csv(OUT_DIR / "routing_account_predictions_test.csv", index=False)
    routing_policy_df.to_csv(OUT_DIR / "routing_policy_tuning.csv", index=False)
    adversarial_routing_stress_df.to_csv(OUT_DIR / "adversarial_routing_stress.csv", index=False)
    routing_penalty_sweep_df.to_csv(OUT_DIR / "routing_penalty_sweep.csv", index=False)
    routing_frontier_accounts_df.to_csv(OUT_DIR / "routing_frontier_account_predictions_test.csv", index=False)
    bootstrap_frontier_df.to_csv(OUT_DIR / "bootstrap_routing_frontier_ci.csv", index=False)
    time_to_alert_df.to_csv(OUT_DIR / "time_to_alert_workload_summary.csv", index=False)
    analyst_workload_daily_df.to_csv(OUT_DIR / "analyst_workload_daily.csv", index=False)
    bootstrap_workload_df.to_csv(OUT_DIR / "bootstrap_time_to_alert_workload_ci.csv", index=False)
    backlog_summary_df.to_csv(OUT_DIR / "dedup_backlog_summary.csv", index=False)
    backlog_daily_df.to_csv(OUT_DIR / "dedup_backlog_daily.csv", index=False)
    backlog_tasks_df.to_csv(OUT_DIR / "dedup_backlog_tasks.csv", index=False)
    bootstrap_backlog_df.to_csv(OUT_DIR / "bootstrap_dedup_backlog_ci.csv", index=False)
    calibration_summary_df.to_csv(OUT_DIR / "calibration_summary.csv", index=False)
    calibration_bins_df.to_csv(OUT_DIR / "calibration_bins.csv", index=False)
    targeted_diagnostics_df.to_csv(OUT_DIR / "targeted_ablation_diagnostics.csv", index=False)
    targeted_scenario_df.to_csv(OUT_DIR / "targeted_ablation_diagnostics_by_scenario.csv", index=False)
    score_df.to_csv(OUT_DIR / "test_account_scores_for_bootstrap.csv", index=False)
    bootstrap_account_df.to_csv(OUT_DIR / "bootstrap_account_ci.csv", index=False)
    bootstrap_system_df.to_csv(OUT_DIR / "bootstrap_system_alert_ci.csv", index=False)
    bootstrap_targeted_df.to_csv(OUT_DIR / "bootstrap_targeted_ablation_ci.csv", index=False)
    bootstrap_routing_df.to_csv(OUT_DIR / "bootstrap_routing_ci.csv", index=False)
    summary_results_df.to_csv(OUT_DIR / "summary_results_table.csv", index=False)
    (OUT_DIR / "summary_results_table.md").write_text(
        "# Summary Results Table\n\n" + markdown_table(summary_results_df),
        encoding="utf-8",
    )
    write_calibration_svg(calibration_bins_df, OUT_DIR / "calibration_curves_test.svg")
    figure_dir = write_manuscript_figures(OUT_DIR)

    test_15 = account_metrics_df[
        (account_metrics_df["split"] == "test") & (account_metrics_df["budget"] == manuscript_budget)
    ].sort_values(["harm_weighted_recall", "precision"], ascending=False)
    alert_test = (
        alert_metrics_df[alert_metrics_df["split"] == "test"]
        .groupby("method")
        .agg(
            product_alert_hit_rate=("product_alert_hit", "mean"),
            mean_false_systemic_alerts=("false_systemic_alerts", "mean"),
            mean_adversarial_false_systemic_alerts=("adversarial_false_systemic_alerts", "mean"),
            mean_exploit_A_rank=("exploit_A_rank", "mean"),
        )
        .reset_index()
        .sort_values("product_alert_hit_rate", ascending=False)
    )
    tuned_test = tuned_test_df.sort_values(["harm_weighted_recall", "precision"], ascending=False)
    calibration_test = (
        calibration_summary_df[calibration_summary_df["split"] == "test"]
        .sort_values(["ece", "brier"])
    )
    routing_test = routing_metrics_df[routing_metrics_df["split"] == "test"].sort_values(
        ["macro_route_f1", "route_accuracy"], ascending=False
    )
    adversarial_routing_test = adversarial_routing_stress_df[
        (adversarial_routing_stress_df["split"] == "test")
        & (adversarial_routing_stress_df["type"] == "ALL_ADVERSARIAL_FALSE_SYSTEMIC")
    ].sort_values(["product_investigation_rate", "method", "policy_variant"])
    routing_comparison = (
        routing_metrics_df[routing_metrics_df["split"] == "test"]
        .pivot_table(
            index="method",
            columns="policy_variant",
            values=["macro_route_f1", "route_accuracy", "routing_utility", "missed_product_investigations"],
            aggfunc="first",
        )
        .reset_index()
    )
    routing_comparison.columns = [
        "_".join([str(c) for c in col if c]).strip("_") if isinstance(col, tuple) else col
        for col in routing_comparison.columns
    ]
    if {
        "macro_route_f1_validation_tuned",
        "macro_route_f1_fixed",
        "routing_utility_validation_tuned",
        "routing_utility_fixed",
    }.issubset(routing_comparison.columns):
        routing_comparison["delta_macro_route_f1"] = (
            routing_comparison["macro_route_f1_validation_tuned"]
            - routing_comparison["macro_route_f1_fixed"]
        )
        routing_comparison["delta_routing_utility"] = (
            routing_comparison["routing_utility_validation_tuned"]
            - routing_comparison["routing_utility_fixed"]
        )
        routing_comparison = routing_comparison.sort_values(
            ["delta_routing_utility", "delta_macro_route_f1"], ascending=False
        )
    ablation_test = test_15[
        test_15["method"].isin(
            [
                "learned_logistic_hier",
                "ablation_no_uncertainty",
                "ablation_no_signature",
                "ablation_no_provenance",
                "learned_logistic_flat",
            ]
        )
    ].copy()
    targeted_summary = targeted_diagnostics_df[
        targeted_diagnostics_df["method"].isin(
            [
                "learned_logistic_hier",
                "ablation_no_uncertainty",
                "ablation_no_signature",
                "ablation_no_provenance",
                "learned_logistic_flat",
            ]
        )
    ].sort_values(["type", "method"])

    summary = []
    summary.append("# AI-MLA Prototype Benchmark Summary\n")
    summary.append(f"## Test-set account review at {manuscript_budget:.0%} budget\n")
    summary.append(test_15.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    summary.append(f"\n\n## Test-set ablation comparison at {manuscript_budget:.0%} budget\n")
    summary.append(ablation_test.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    summary.append("\n\n## Targeted ablation diagnostics on matched failure-mode families\n")
    summary.append(targeted_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    summary.append("\n\n## Test-set metrics using validation-tuned thresholds\n")
    summary.append(tuned_test.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    summary.append("\n\n## Test-set calibration summary\n")
    summary.append(calibration_test.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    summary.append("\n\n## Test-set routing and escalation summary\n")
    summary.append(routing_test.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    summary.append("\n\n## Test-set fixed vs validation-tuned routing comparison\n")
    summary.append(routing_comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    summary.append("\n\n## Test-set adversarial false-systemic routing stress\n")
    summary.append(adversarial_routing_test.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    summary.append("\n\n## Product-investigation tradeoff sweep: learned hierarchy\n")
    summary.append(
        routing_penalty_sweep_df[
            [
                "adversarial_false_alert_penalty",
                "missed_product_investigations",
                "adversarial_false_systemic_product_investigations",
                "macro_route_f1",
                "routing_utility",
                "product_investigation_precision",
                "product_investigation_recall",
                "product_score_threshold",
                "product_signature_confidence_threshold",
                "product_evidence_threshold",
                "legitimacy_suppression_threshold",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    frontier_ci_summary = bootstrap_frontier_df[
        bootstrap_frontier_df["metric"].isin(
            [
                "missed_product_investigations",
                "adversarial_false_systemic_product_investigations",
                "product_investigation_precision",
                "product_investigation_recall",
                "macro_route_f1",
                "routing_utility",
            ]
        )
    ].sort_values(["adversarial_false_alert_penalty", "metric"])
    summary.append("\n\n## Bootstrap confidence intervals: product-investigation frontier\n")
    summary.append(frontier_ci_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    practical_summary = time_to_alert_df[
        time_to_alert_df["policy_variant"].eq("validation_tuned")
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
    ].sort_values(["product_campaign_alert_rate", "total_analyst_hours"], ascending=[False, True])
    summary.append("\n\n## Practical monitoring simulation: time-to-alert and analyst workload\n")
    summary.append(
        practical_summary[
            [
                "method",
                "policy_variant",
                "product_campaign_alert_rate",
                "missed_product_campaigns",
                "median_time_to_product_alert_hours",
                "p90_time_to_product_alert_hours",
                "total_analyst_hours",
                "analyst_days",
                "peak_daily_analyst_hours",
                "days_over_capacity",
                "total_human_reviews",
                "total_policy_reviews",
                "total_product_investigations",
                "false_product_investigations",
                "adversarial_false_systemic_product_investigations",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    workload_ci_summary = bootstrap_workload_df[
        bootstrap_workload_df["policy_variant"].eq("validation_tuned")
        & bootstrap_workload_df["method"].isin(
            [
                "learned_logistic_flat",
                "learned_logistic_hier",
                "ablation_no_uncertainty",
                "ablation_no_signature",
                "ablation_no_provenance",
                "typed_system_hierarchy",
            ]
        )
        & bootstrap_workload_df["metric"].isin(
            [
                "product_campaign_alert_rate",
                "missed_product_campaigns",
                "median_time_to_product_alert_hours",
                "p90_time_to_product_alert_hours",
                "total_analyst_hours",
                "false_product_investigations",
                "adversarial_false_systemic_product_investigations",
            ]
        )
    ].sort_values(["metric", "method"])
    summary.append("\n\n## Bootstrap confidence intervals: practical monitoring simulation\n")
    summary.append(workload_ci_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    backlog_summary = backlog_summary_df[
        backlog_summary_df["policy_variant"].eq("validation_tuned")
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
        & backlog_summary_df["queue_mode"].isin(["fifo", "product_first", "product_policy_first", "severity_weighted"])
    ].sort_values(["capacity_hours_per_day", "peak_backlog_hours", "false_unique_product_alerts"])
    summary.append("\n\n## Deduplicated alert queue and analyst-backlog simulation\n")
    summary.append(
        backlog_summary[
            [
                "method",
                "policy_variant",
                "capacity_hours_per_day",
                "queue_mode",
                "unique_product_alerts",
                "true_unique_product_alerts",
                "false_unique_product_alerts",
                "adversarial_false_unique_product_alerts",
                "duplicate_product_alerts_avoided",
                "peak_backlog_hours",
                "mean_backlog_hours",
                "days_with_unresolved_backlog",
                "median_delay_to_product_investigation_hours",
                "p90_delay_to_product_investigation_hours",
                "true_product_alerts_resolved_within_24h_rate",
                "true_product_alerts_resolved_within_48h_rate",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    backlog_ci_summary = bootstrap_backlog_df[
        bootstrap_backlog_df["metric"].isin(
            [
                "peak_backlog_hours",
                "mean_backlog_hours",
                "p90_delay_to_product_investigation_hours",
                "true_product_alerts_resolved_within_24h_rate",
                "true_product_alerts_resolved_within_48h_rate",
                "false_unique_product_alerts",
                "peak_non_product_backlog_hours",
            ]
        )
    ].sort_values(["metric", "method", "queue_mode"])
    summary.append("\n\n## Bootstrap confidence intervals: deduplicated backlog simulation\n")
    summary.append(backlog_ci_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    summary.append("\n\n## Manuscript-facing summary results table\n")
    summary.append(markdown_table(summary_results_df))
    summary.append("\n\n## Test-set system alert summary\n")
    summary.append(alert_test.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    summary.append(
        f"\n\n## Bootstrap confidence intervals: account metrics at {manuscript_budget:.0%} review budget\n"
    )
    summary.append(bootstrap_account_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    summary.append("\n\n## Bootstrap confidence intervals: system alerts\n")
    summary.append(bootstrap_system_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    summary.append("\n\n## Bootstrap confidence intervals: targeted ablations\n")
    summary.append(bootstrap_targeted_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    routing_ci_summary = bootstrap_routing_df[
        bootstrap_routing_df["metric"].isin(
            [
                "macro_route_f1",
                "route_accuracy",
                "routing_utility",
                "missed_product_investigations",
                "unnecessary_product_investigations",
                "adversarial_false_systemic_product_investigations",
                "over_escalation_rate",
                "under_escalation_rate",
            ]
        )
    ].sort_values(["metric", "method", "policy_variant"])
    summary.append("\n\n## Bootstrap confidence intervals: routing and escalation\n")
    summary.append(routing_ci_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    summary.append(
        "\n\nInterpretation: this benchmark separates rule baselines, learned flat baselines, "
        "learned hierarchical baselines, validation-tuned operating thresholds, calibration, "
        "and ablations on held-out scenarios. The manuscript-relevant question is whether "
        "hierarchical features improve harm-weighted recall, systemic recall, calibration, "
        "or product-alert ranking after strong flat and simple aggregate baselines are included.\n"
    )
    (OUT_DIR / "benchmark_summary.md").write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))
    print(f"\nSaved benchmark outputs to {OUT_DIR.resolve()}")
    print(f"Saved manuscript figures to {figure_dir.resolve()}")


if __name__ == "__main__":
    main()
