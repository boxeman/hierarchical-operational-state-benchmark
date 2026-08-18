import json
import subprocess
import sys
import unittest
from pathlib import Path

from ai_mla_monitor.config import FEATURE_COLUMNS, ScenarioConfig
from ai_mla_monitor.data import build_dataset
from run_benchmark import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


class ReleaseCandidateTests(unittest.TestCase):
    def test_config_loads_with_expected_overrides(self):
        cfg = load_experiment_config(REPO_ROOT / "configs" / "ci_smoke_config.json")

        self.assertEqual(cfg["experiment_name"], "ci_smoke")
        self.assertEqual(cfg["splits"]["train_scenarios"], 1)
        self.assertEqual(cfg["bootstrap"]["account_n"], 1)
        self.assertEqual(cfg["output_dir"], "outputs/ci_smoke")

    def test_scenario_generation_returns_expected_fields(self):
        df, signature_state = build_dataset(
            ScenarioConfig(
                seed=123,
                benign_n=20,
                obvious_n=3,
                borderline_n=3,
                distributed_n=3,
                systemic_n=10,
                policy_ambiguous_n=3,
                escalating_n=3,
                uncertainty_required_n=3,
                signature_required_n=3,
                provenance_required_n=3,
                adversarial_false_systemic_n=3,
                frontier_overlap_n=3,
            )
        )

        required_columns = {
            "id",
            "type",
            "harmful",
            "harm_weight",
            "signature",
            "campaign_id",
            "expected_route",
        } | set(FEATURE_COLUMNS)

        self.assertTrue(required_columns.issubset(set(df.columns)))
        self.assertGreater(len(signature_state), 0)
        self.assertGreater(len(df), 0)
        self.assertGreater(df["harmful"].sum(), 0)
        self.assertTrue({"auto_handle", "human_review", "policy_review", "product_investigation"}.intersection(set(df["expected_route"])))

    def test_ci_smoke_run_produces_key_outputs(self):
        config_path = REPO_ROOT / "configs" / "ci_smoke_config.json"
        result = subprocess.run(
            [sys.executable, "run_benchmark.py", "--config", str(config_path)],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout[-4000:])

        out_dir = REPO_ROOT / "outputs" / "ci_smoke"
        expected_files = [
            "run_config.json",
            "input_config.json",
            "benchmark_summary.md",
            "summary_results_table.csv",
            "summary_results_table.md",
            "routing_metrics_by_split.csv",
            "calibration_summary.csv",
        ]
        for filename in expected_files:
            self.assertTrue((out_dir / filename).exists(), msg=f"Missing {filename}")

        expected_figures = [
            "figure1_account_review.svg",
            "figure2_targeted_ablations.svg",
            "figure3_calibration_system_alerts.svg",
            "figure4_routing_escalation.svg",
            "figure5_adversarial_false_systemic.svg",
            "figure6_product_investigation_frontier.svg",
            "figure7_time_to_alert_workload.svg",
            "figure8_dedup_backlog.svg",
            "figure9_priority_queues.svg",
        ]
        for filename in expected_figures:
            self.assertTrue((out_dir / "figures" / filename).exists(), msg=f"Missing {filename}")

        with (out_dir / "run_config.json").open("r", encoding="utf-8") as f:
            run_cfg = json.load(f)
        self.assertEqual(run_cfg["experiment_name"], "ci_smoke")


if __name__ == "__main__":
    unittest.main()
