from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

RENAME_CHECKS = {
    "model_quality.best_macro_f1": "candidate_test.macro_f1",
    "model_quality.best_pr_auc": "candidate_test.pr_auc",
    "model_quality.best_ece": "candidate_test.ece",
    "model_quality.best_brier": "candidate_test.brier",
    "model_quality.best_log_loss": "candidate_test.log_loss",
    "model_quality.worst_slice_f1": "candidate_test.worst_slice_f1",
    "split_integrity.no_overlap": "data_integrity.record_id_no_overlap",
    "split_integrity.class_coverage": "data_integrity.class_coverage",
}

AMBIGUOUS = {
    "best_macro_f1",
    "best_pr_auc",
    "best_ece",
    "best_brier",
    "best_log_loss",
}


def clean(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def row_dict(row: pd.Series) -> dict[str, Any]:
    return {str(key): clean(value) for key, value in row.to_dict().items()}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def backup_once(path: Path) -> None:
    if not path.exists():
        return
    backup = path.with_name(f"{path.stem}_legacy{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)


def _rank(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    metric = str(config.get("primary_metric", "macro_f1"))
    if metric not in frame.columns:
        raise KeyError(f"Selection leaderboard is missing primary metric: {metric}")

    sort_cols = [metric]
    ascending = [not bool(config.get("greater_is_better", True))]

    for item in config.get("tie_breakers") or []:
        item = str(item)
        if item.endswith("_desc"):
            column, order = item[:-5], False
        elif item.endswith("_asc"):
            column, order = item[:-4], True
        else:
            continue
        if column in frame.columns:
            sort_cols.append(column)
            ascending.append(order)

    return frame.sort_values(
        sort_cols,
        ascending=ascending,
        kind="mergesort",
    ).reset_index(drop=True)


def select_winner(
    selection: pd.DataFrame,
    config: dict[str, Any],
    policy_mode: str,
) -> tuple[pd.Series, pd.Series | None, dict[str, Any]]:
    target = str(config.get("primary_target", "product"))
    target_frame = selection.loc[
        selection["target_col"].astype(str).eq(target)
    ].copy()

    if target_frame.empty:
        raise RuntimeError(
            f"No model exists for primary target_col={target!r}."
        )

    eligible = target_frame.copy()
    constraints = config.get("eligibility_constraints") or {}
    rules = [
        ("evaluation_class_coverage", "evaluation_class_coverage_min", "min"),
        ("unknown_target_rate", "unknown_target_rate_max", "max"),
        ("ece", "ece_max", "max"),
        ("brier", "brier_max", "max"),
        ("log_loss", "log_loss_max", "max"),
    ]

    constraint_results: list[dict[str, Any]] = []
    for column, key, direction in rules:
        if key not in constraints or column not in eligible.columns:
            continue

        threshold = float(constraints[key])
        values = pd.to_numeric(eligible[column], errors="coerce")
        mask = (
            values.ge(threshold)
            if direction == "min"
            else values.le(threshold)
        )
        constraint_results.append(
            {
                "metric": column,
                "operator": ">=" if direction == "min" else "<=",
                "threshold": threshold,
                "passing_models_before_next_constraint": int(mask.sum()),
            }
        )
        eligible = eligible.loc[mask]

    if eligible.empty:
        if policy_mode == "strict":
            raise RuntimeError(
                "No model satisfies model-selection eligibility constraints."
            )
        ranked = _rank(target_frame, config)
        eligibility_status = "NO_ELIGIBLE_CANDIDATE"
        selection_mode = "diagnostic_fallback"
    else:
        ranked = _rank(eligible, config)
        eligibility_status = "ELIGIBLE_CANDIDATE_SELECTED"
        selection_mode = "strict_eligible_selection"

    winner = ranked.iloc[0]
    runner_up = ranked.iloc[1] if len(ranked) > 1 else None
    metadata = {
        "policy_mode": policy_mode,
        "selection_mode": selection_mode,
        "eligibility_status": eligibility_status,
        "primary_target_candidate_count": int(len(target_frame)),
        "eligible_candidate_count": int(len(eligible)),
        "constraint_results": constraint_results,
    }
    return winner, runner_up, metadata


def portfolio_diagnostics(test: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "diagnostic_only": True,
        "used_for_selection": False,
    }
    metrics = [
        ("macro_f1", "max"),
        ("pr_auc", "max"),
        ("worst_slice_f1", "max"),
        ("ece", "min"),
        ("brier", "min"),
        ("log_loss", "min"),
    ]
    for metric, direction in metrics:
        if metric not in test.columns:
            continue
        values = pd.to_numeric(test[metric], errors="coerce").dropna()
        if values.empty:
            continue
        index = values.idxmax() if direction == "max" else values.idxmin()
        result[f"top_{metric}"] = clean(test.loc[index, metric])
        result[f"top_{metric}_model"] = str(test.loc[index, "model"])
    return result


def prepend_semantics(
    path: Path,
    selection: dict[str, Any],
    candidate: dict[str, Any],
    launch: dict[str, Any],
) -> None:
    if not path.exists():
        return

    begin = "<!-- selection-test-launch-v3 -->"
    end = "<!-- end-selection-test-launch-v3 -->"
    original = path.read_text(encoding="utf-8")

    for old_begin, old_end in [
        (
            "<!-- selection-test-launch-v2 -->",
            "<!-- end-selection-test-launch-v2 -->",
        ),
        (begin, end),
    ]:
        if old_begin in original and old_end in original:
            original = original.split(old_end, 1)[1].lstrip()

    original = original.replace(
        "# Model Card: Quality Sign-Off Champion",
        "# Model Card: Promotion Candidate and Launch Readiness",
    )
    original = original.replace(
        "## Champion model",
        "## Legacy Selection Label (deprecated)",
    )

    metrics = candidate["candidate_test_metrics"]
    header = f"""{begin}
# Selection, Untouched Test, and Launch Readiness

A model selected on the dedicated selection window is a **selection winner**.
It becomes a **promotion candidate** only when model-selection eligibility is met.
It becomes the **approved champion** only when the untouched-test launch gate returns PASS.

## Model Selection Decision

- Selection winner: `{selection["selection_winner"]}`
- Selection policy mode: `{selection["policy_mode"]}`
- Selection mode: `{selection["selection_mode"]}`
- Eligibility status: `{selection["eligibility_status"]}`
- Eligible candidate count: `{selection["eligible_candidate_count"]}`
- Runner-up: `{selection.get("runner_up")}`
- Primary metric: `{selection["primary_metric"]}`
- Selection Macro-F1: `{selection["winner_metrics"].get("macro_f1")}`
- Test used for selection: `false`

## Untouched Test Evaluation

- Evaluated selection winner: `{candidate["evaluated_selection_winner"]}`
- Promotion candidate: `{candidate["promotion_candidate"]}`
- Candidate test Macro-F1: `{metrics.get("macro_f1")}`
- Candidate test PR-AUC: `{metrics.get("pr_auc")}`
- Candidate test ECE: `{metrics.get("ece")}`
- Candidate test Brier: `{metrics.get("brier")}`
- Candidate test worst-slice F1: `{metrics.get("worst_slice_f1")}`
- Test used for reselection: `false`

## Launch Readiness Decision

- Decision: `{launch["decision"]}`
- Promotion candidate: `{launch["promotion_candidate"]}`
- Approved champion: `{launch["approved_champion"]}`
- Fallback model: `{launch["fallback_model"]}`

{end}

"""
    path.write_text(header + original, encoding="utf-8")


def finalize(
    outputs: Path,
    reports: Path,
    config_path: Path,
    policy_mode: str,
) -> dict[str, Any]:
    selection_file = outputs / "model_selection_leaderboard.csv"
    test_file = outputs / "model_test_leaderboard.csv"
    if not selection_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            "Selection and test leaderboards are required."
        )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    selection = pd.read_csv(selection_file)
    test = pd.read_csv(test_file)

    winner, runner_up, selection_meta = select_winner(
        selection,
        config,
        policy_mode,
    )
    winner_name = str(winner["model"])
    test_rows = test.loc[test["model"].astype(str).eq(winner_name)]
    if test_rows.empty:
        raise RuntimeError(
            f"Selection winner missing from test leaderboard: {winner_name}"
        )
    winner_test = test_rows.iloc[0]

    eligibility_status = str(selection_meta["eligibility_status"])
    has_eligible_candidate = (
        eligibility_status == "ELIGIBLE_CANDIDATE_SELECTED"
    )
    promotion_candidate = winner_name if has_eligible_candidate else None

    primary_metric = str(config.get("primary_metric", "macro_f1"))
    selection_result = {
        "schema_version": 2,
        "selection_split": "selection",
        "primary_target": str(config.get("primary_target", "product")),
        "primary_metric": primary_metric,
        "selection_winner": winner_name,
        "runner_up": (
            str(runner_up["model"]) if runner_up is not None else None
        ),
        "winner_metrics": row_dict(winner),
        "runner_up_metrics": (
            row_dict(runner_up) if runner_up is not None else None
        ),
        "selection_margin": (
            clean(
                float(winner[primary_metric])
                - float(runner_up[primary_metric])
            )
            if runner_up is not None
            else None
        ),
        "selection_split_only": True,
        "test_used_for_selection": False,
        **selection_meta,
    }

    candidate_metrics = row_dict(winner_test)
    selection_value = clean(winner.get(primary_metric))
    test_value = clean(winner_test.get(primary_metric))
    candidate_result = {
        "schema_version": 2,
        "evaluated_selection_winner": winner_name,
        "promotion_candidate": promotion_candidate,
        "eligibility_status": eligibility_status,
        "candidate_test_metrics": candidate_metrics,
        "selection_test_primary_metric_gap": (
            abs(float(selection_value) - float(test_value))
            if selection_value is not None and test_value is not None
            else None
        ),
        "test_used_for_reselection": False,
    }

    diagnostics = portfolio_diagnostics(test)

    gate_file = outputs / "launch_gate_result.json"
    legacy_gate = read_json(gate_file)
    backup_once(gate_file)
    legacy_decision = str(
        legacy_gate.get("decision")
        or legacy_gate.get("status")
        or "REVIEW"
    ).upper()

    decision = (
        legacy_decision
        if has_eligible_candidate
        else "BLOCK"
    )

    failed_checks = []
    for item in legacy_gate.get("failed_checks") or []:
        if not isinstance(item, dict):
            continue
        converted = {str(k): clean(v) for k, v in item.items()}
        old_name = str(converted.get("name", ""))
        converted["name"] = RENAME_CHECKS.get(old_name, old_name)
        failed_checks.append(converted)

    if not has_eligible_candidate:
        failed_checks.insert(
            0,
            {
                "name": "model_selection.eligible_candidate_available",
                "value": 0.0,
                "threshold": 1.0,
                "violation_ratio": 1.0,
                "hard_failure": True,
                "detail": (
                    "Smoke-mode diagnostic fallback selected the best "
                    "primary-target model for reporting, but no model met "
                    "promotion eligibility constraints."
                ),
            },
        )

    fallback = str(
        (config.get("fallback_policy") or {}).get("model") or ""
    )
    launch_result = {
        "schema_version": 3,
        "selection_winner": winner_name,
        "promotion_candidate": promotion_candidate,
        "fallback_model": fallback,
        "decision": decision,
        "approved_champion": (
            promotion_candidate
            if decision == "PASS" and promotion_candidate is not None
            else None
        ),
        "selection_policy_mode": policy_mode,
        "eligibility_status": eligibility_status,
        "candidate_source": "model_selection_decision.json",
        "metric_source": "candidate_test_evaluation.json",
        "failed_checks": failed_checks,
        "legacy_gate_decision": legacy_decision,
        "legacy_gate_backup": "launch_gate_result_legacy.json",
    }

    write_json(
        outputs / "model_selection_decision.json",
        selection_result,
    )
    write_json(
        outputs / "candidate_test_evaluation.json",
        candidate_result,
    )
    write_json(
        outputs / "portfolio_test_diagnostics.json",
        diagnostics,
    )
    write_json(gate_file, launch_result)

    all_metrics_file = outputs / "all_metrics.json"
    if all_metrics_file.exists():
        backup_once(all_metrics_file)
        all_metrics = read_json(all_metrics_file)
        all_metrics.pop("model_quality", None)
        all_metrics["model_selection"] = selection_result
        all_metrics["candidate_test"] = candidate_result
        all_metrics["portfolio_test_diagnostics"] = diagnostics
        all_metrics["launch_readiness"] = launch_result
        write_json(all_metrics_file, all_metrics)

    manifest_file = outputs / "pipeline_manifest.json"
    if manifest_file.exists():
        backup_once(manifest_file)
        manifest = read_json(manifest_file)
        manifest.pop("champion_model", None)
        manifest["selection_winner"] = winner_name
        manifest["promotion_candidate"] = promotion_candidate
        manifest["fallback_model"] = fallback
        manifest["launch_decision"] = decision
        manifest["approved_champion"] = launch_result[
            "approved_champion"
        ]
        manifest["selection_policy_mode"] = policy_mode
        manifest["selection_eligibility_status"] = eligibility_status
        manifest["selection_protocol"] = (
            "train -> calibration -> selection -> untouched test "
            "-> launch gate"
        )
        manifest["test_used_for_selection"] = False
        manifest["test_used_for_reselection"] = False
        write_json(manifest_file, manifest)

    prepend_semantics(
        reports / "executive_summary.md",
        selection_result,
        candidate_result,
        launch_result,
    )
    prepend_semantics(
        reports / "model_card.md",
        selection_result,
        candidate_result,
        launch_result,
    )

    return {
        "selection_winner": winner_name,
        "promotion_candidate": promotion_candidate,
        "fallback_model": fallback,
        "decision": decision,
        "approved_champion": launch_result["approved_champion"],
        "policy_mode": policy_mode,
        "eligibility_status": eligibility_status,
    }


def audit(outputs: Path) -> dict[str, Any]:
    required = [
        "model_selection_decision.json",
        "candidate_test_evaluation.json",
        "portfolio_test_diagnostics.json",
        "launch_gate_result.json",
    ]
    missing = [
        name for name in required if not (outputs / name).exists()
    ]
    if missing:
        raise AssertionError(
            f"Missing semantics artifacts: {missing}"
        )

    selection = read_json(
        outputs / "model_selection_decision.json"
    )
    candidate = read_json(
        outputs / "candidate_test_evaluation.json"
    )
    launch = read_json(outputs / "launch_gate_result.json")

    serialized_gate = json.dumps(launch)
    for key in AMBIGUOUS:
        if key in serialized_gate:
            raise AssertionError(
                f"Ambiguous key remains in launch gate: {key}"
            )

    if selection.get("test_used_for_selection") is not False:
        raise AssertionError(
            "test_used_for_selection must be false"
        )
    if candidate.get("test_used_for_reselection") is not False:
        raise AssertionError(
            "test_used_for_reselection must be false"
        )

    eligibility_status = launch.get("eligibility_status")
    promotion_candidate = launch.get("promotion_candidate")
    decision = launch.get("decision")
    approved = launch.get("approved_champion")

    if eligibility_status == "NO_ELIGIBLE_CANDIDATE":
        if promotion_candidate is not None:
            raise AssertionError(
                "No eligible candidate must imply promotion_candidate=null"
            )
        if decision != "BLOCK":
            raise AssertionError(
                "No eligible candidate must force BLOCK"
            )
        if approved is not None:
            raise AssertionError(
                "No eligible candidate must not approve a champion"
            )
    else:
        if decision == "PASS" and approved != promotion_candidate:
            raise AssertionError(
                "PASS must approve the promotion candidate"
            )
        if decision in {"REVIEW", "BLOCK"} and approved is not None:
            raise AssertionError(
                "REVIEW/BLOCK must not create an approved champion"
            )

    return {
        "status": "PASS",
        "selection_winner": selection.get("selection_winner"),
        "promotion_candidate": promotion_candidate,
        "fallback_model": launch.get("fallback_model"),
        "decision": decision,
        "approved_champion": approved,
        "policy_mode": launch.get("selection_policy_mode"),
        "eligibility_status": eligibility_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument(
        "--config",
        default="configs/model_selection.yaml",
    )
    parser.add_argument(
        "--policy-mode",
        choices=["strict", "smoke"],
        default="strict",
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    outputs = Path(args.outputs_dir)
    if not args.audit_only:
        result = finalize(
            outputs,
            Path(args.reports_dir),
            Path(args.config),
            args.policy_mode,
        )
        print(json.dumps({"finalization": result}, indent=2))

    print(json.dumps({"audit": audit(outputs)}, indent=2))


if __name__ == "__main__":
    main()
