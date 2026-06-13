param(
    [string]$RepoPath = "${PSScriptRoot}",
    [string]$BaselineAuditRoot = "",
    [int[]]$Seeds = @(7, 17, 27),
    [int]$Epochs = 2,
    [int]$BatchSize = 32,
    [string]$TransformerModel = "distilbert-base-uncased"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Seeds.Count -ne 3) {
    throw "Exactly three seeds are required. Received: $($Seeds -join ', ')"
}
if (($Seeds | Select-Object -Unique).Count -ne 3) {
    throw "Seeds must be unique. Received: $($Seeds -join ', ')"
}

Set-Location $RepoPath

$Python = Join-Path $RepoPath ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Virtual-environment Python not found: $Python"
}

$RequiredFiles = @(
    "src\model_quality\models\train_transformer.py",
    "data\golden\train.csv",
    "data\golden\calibration.csv",
    "data\golden\selection.csv",
    "data\golden\test.csv"
)

foreach ($RelativePath in $RequiredFiles) {
    $FullPath = Join-Path $RepoPath $RelativePath
    if (-not (Test-Path $FullPath)) {
        throw "Required file not found: $FullPath"
    }
}

if ([string]::IsNullOrWhiteSpace($BaselineAuditRoot)) {
    $StableCandidates = @(
        Get-ChildItem `
            (Join-Path $RepoPath "sa") `
            -Directory `
            -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending
    )

    foreach ($Candidate in $StableCandidates) {
        $DecisionPath = Join-Path `
            $Candidate.FullName `
            "transformer_3seed_stability_decision.json"

        if (-not (Test-Path $DecisionPath)) {
            continue
        }

        $Decision = Get-Content $DecisionPath -Raw | ConvertFrom-Json
        if ([string]$Decision.verdict -eq "STABLE") {
            $BaselineAuditRoot = $Candidate.FullName
            break
        }
    }
}

if ([string]::IsNullOrWhiteSpace($BaselineAuditRoot)) {
    throw (
        "No STABLE baseline audit was found under '$RepoPath\sa'. " +
        "Pass -BaselineAuditRoot explicitly."
    )
}

$BaselineAuditRoot = (
    Resolve-Path $BaselineAuditRoot
).Path

$BaselineDecisionPath = Join-Path `
    $BaselineAuditRoot `
    "transformer_3seed_stability_decision.json"
$BaselineManifestPath = Join-Path `
    $BaselineAuditRoot `
    "fixed_input\fixed_input_manifest.json"

foreach ($Path in @(
    $BaselineDecisionPath,
    $BaselineManifestPath,
    (Join-Path $BaselineAuditRoot "transformer_3seed_metrics.csv")
)) {
    if (-not (Test-Path $Path)) {
        throw "Baseline audit artifact not found: $Path"
    }
}

$BaselineDecision = Get-Content `
    $BaselineDecisionPath `
    -Raw |
    ConvertFrom-Json

if ([string]$BaselineDecision.verdict -ne "STABLE") {
    throw (
        "Baseline audit must have verdict STABLE. " +
        "Found: $($BaselineDecision.verdict)"
    )
}

$BaselineSeeds = @(
    $BaselineDecision.seeds | ForEach-Object { [int]$_ }
)

$SeedDifferences = @(
    Compare-Object `
        -ReferenceObject @($Seeds | Sort-Object) `
        -DifferenceObject @($BaselineSeeds | Sort-Object)
)

if ($SeedDifferences.Count -ne 0) {
    throw (
        "Requested seeds do not match the baseline audit seeds. " +
        "Requested=$($Seeds -join ','); " +
        "Baseline=$($BaselineSeeds -join ',')"
    )
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ExperimentRoot = Join-Path `
    (Join-Path $RepoPath "cw") `
    "w_$Stamp"
$LogsDir = Join-Path $ExperimentRoot "logs"
$RunnerPath = Join-Path $ExperimentRoot "_run_weighted_challenger.py"
$LogPath = Join-Path $LogsDir "weighted_challenger.log"

New-Item -ItemType Directory -Force $ExperimentRoot | Out-Null
New-Item -ItemType Directory -Force $LogsDir | Out-Null

Write-Host ""
Write-Host "================================================================================"
Write-Host "CLASS-WEIGHTED TRANSFORMER 3-SEED CHALLENGER"
Write-Host "================================================================================"
Write-Host "Repo:             $RepoPath"
Write-Host "Baseline audit:   $BaselineAuditRoot"
Write-Host "Experiment root:  $ExperimentRoot"
Write-Host "Seeds:            $($Seeds -join ', ')"
Write-Host "Epochs:           $Epochs"
Write-Host "Batch size:       $BatchSize"
Write-Host "Model:            $TransformerModel"
Write-Host "Weighting:        sqrt-balanced, normalized to mean 1.0"
Write-Host "Selection rule:   selection split only"
Write-Host "Test role:        post-selection confirmation only"
Write-Host ""

@'
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if hasattr(value, "item"):
        return json_safe(value.item())
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_fixed_input_hash(
    split_frames: dict[str, pd.DataFrame],
    output_csv: Path,
) -> tuple[str, int]:
    fixed = pd.concat(
        [split_frames[name] for name in (
            "train",
            "calibration",
            "selection",
            "test",
        )],
        ignore_index=True,
    )

    sort_columns = [
        column
        for column in ("date_received", "complaint_id")
        if column in fixed.columns
    ]
    if sort_columns:
        fixed = fixed.sort_values(
            sort_columns,
            kind="mergesort",
        ).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fixed.to_csv(output_csv, index=False)

    return (
        hashlib.sha256(output_csv.read_bytes()).hexdigest(),
        int(len(fixed)),
    )


def metric(
    frame: pd.DataFrame,
    model: str,
    column: str,
) -> float:
    row = frame.loc[frame["model"].astype(str).eq(model)]
    if len(row) != 1:
        raise RuntimeError(
            f"Expected one row for model={model!r}; found {len(row)}"
        )
    return float(pd.to_numeric(row.iloc[0][column]))


repo = Path(sys.argv[1]).resolve()
baseline_root = Path(sys.argv[2]).resolve()
experiment_root = Path(sys.argv[3]).resolve()
seeds = [int(item) for item in sys.argv[4].split(",")]
epochs = int(sys.argv[5])
batch_size = int(sys.argv[6])
model_name = sys.argv[7]

sys.path.insert(0, str(repo / "src"))

from model_quality.models.train_transformer import (  # noqa: E402
    train_transformer_classifier,
)

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this challenger experiment.")

baseline_manifest = json.loads(
    (
        baseline_root
        / "fixed_input"
        / "fixed_input_manifest.json"
    ).read_text(encoding="utf-8")
)

split_paths = {
    "train": repo / "data" / "golden" / "train.csv",
    "calibration": repo / "data" / "golden" / "calibration.csv",
    "selection": repo / "data" / "golden" / "selection.csv",
    "test": repo / "data" / "golden" / "test.csv",
}
splits = {
    name: pd.read_csv(path, low_memory=False)
    for name, path in split_paths.items()
}

fixed_csv = (
    experiment_root
    / "fixed_input"
    / "cfpb_fixed_challenger_input.csv"
)
fixed_sha256, fixed_rows = build_fixed_input_hash(
    splits,
    fixed_csv,
)

if fixed_sha256 != baseline_manifest["sha256"]:
    raise RuntimeError(
        "Current golden splits do not match the STABLE baseline audit. "
        f"Current={fixed_sha256}; "
        f"Baseline={baseline_manifest['sha256']}"
    )
if fixed_rows != int(baseline_manifest["row_count"]):
    raise RuntimeError(
        "Frozen row count differs from the baseline audit. "
        f"Current={fixed_rows}; "
        f"Baseline={baseline_manifest['row_count']}"
    )

train_labels = splits["train"]["product"].astype(str)
label_encoder = LabelEncoder()
encoded_train = label_encoder.fit_transform(train_labels)
classes = np.arange(len(label_encoder.classes_))

balanced_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=encoded_train,
)
sqrt_weights = np.sqrt(balanced_weights)
normalized_weights = sqrt_weights / sqrt_weights.mean()

if (
    len(normalized_weights) != len(label_encoder.classes_)
    or not np.isfinite(normalized_weights).all()
    or not (normalized_weights > 0).all()
):
    raise RuntimeError("Computed class weights are invalid.")

weight_metadata = {
    "schema_version": 1,
    "mode": "sqrt_balanced",
    "source_split": "train",
    "formula": (
        "sqrt(n_samples / (n_classes * class_count)); "
        "normalized to arithmetic mean 1.0"
    ),
    "class_count": int(len(label_encoder.classes_)),
    "classes": label_encoder.classes_.tolist(),
    "weights": {
        str(label): float(weight)
        for label, weight in zip(
            label_encoder.classes_,
            normalized_weights,
            strict=True,
        )
    },
    "minimum_weight": float(normalized_weights.min()),
    "maximum_weight": float(normalized_weights.max()),
    "mean_weight": float(normalized_weights.mean()),
    "fixed_input_sha256": fixed_sha256,
    "fixed_input_rows": fixed_rows,
}
write_json(
    experiment_root / "class_weight_metadata.json",
    weight_metadata,
)

import transformers  # noqa: E402

OriginalTrainer = transformers.Trainer
class_weights_tensor = torch.tensor(
    normalized_weights,
    dtype=torch.float32,
)


class WeightedTrainer(OriginalTrainer):
    """Trainer with train-only sqrt-balanced multiclass weights."""

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        copied_inputs = dict(inputs)
        labels = copied_inputs.pop("labels")
        outputs = model(**copied_inputs)
        logits = outputs.get("logits")
        loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights_tensor.to(logits.device)
        )
        loss = loss_function(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


weighted_rows: list[dict[str, Any]] = []

for seed in seeds:
    print("=" * 88, flush=True)
    print(f"WEIGHTED CHALLENGER SEED {seed}", flush=True)
    print("=" * 88, flush=True)

    seed_everything(seed)

    seed_root = experiment_root / f"s{seed}"
    model_dir = seed_root / "model"
    seed_root.mkdir(parents=True, exist_ok=True)

    transformers.Trainer = WeightedTrainer
    try:
        result = train_transformer_classifier(
            splits["train"],
            splits["calibration"],
            selection_df=splits["selection"],
            test_df=splits["test"],
            model_name=model_name,
            out_dir=str(model_dir),
            epochs=epochs,
            batch_size=batch_size,
            device="cuda",
            random_state=seed,
        )
    finally:
        transformers.Trainer = OriginalTrainer

    selection_metrics_raw = getattr(
        result,
        "selection_metrics",
        None,
    )
    test_metrics_raw = getattr(
        result,
        "metrics",
        None,
    )
    result_note = str(getattr(result, "note", ""))

    if not isinstance(selection_metrics_raw, Mapping):
        available_fields = sorted(
            vars(result).keys()
            if hasattr(result, "__dict__")
            else []
        )
        raise RuntimeError(
            "Weighted Transformer result is missing a mapping-valued "
            f"selection_metrics field for seed {seed}. "
            f"Available fields: {available_fields}"
        )

    if not isinstance(test_metrics_raw, Mapping):
        available_fields = sorted(
            vars(result).keys()
            if hasattr(result, "__dict__")
            else []
        )
        raise RuntimeError(
            "Weighted Transformer result is missing a mapping-valued "
            f"metrics field for seed {seed}. "
            f"Available fields: {available_fields}"
        )

    required_metric = "macro_f1"
    if required_metric not in selection_metrics_raw:
        raise RuntimeError(
            f"Seed {seed} selection metrics do not contain "
            f"{required_metric!r}: {sorted(selection_metrics_raw)}"
        )
    if required_metric not in test_metrics_raw:
        raise RuntimeError(
            f"Seed {seed} test metrics do not contain "
            f"{required_metric!r}: {sorted(test_metrics_raw)}"
        )

    selection_metrics = {
        str(key): json_safe(value)
        for key, value in selection_metrics_raw.items()
    }
    test_metrics = {
        str(key): json_safe(value)
        for key, value in test_metrics_raw.items()
    }

    provenance_path = (
        seed_root / "transformer_training_provenance.json"
    )
    if not provenance_path.exists():
        fallback_path = (
            model_dir / "transformer_training_provenance.json"
        )
        if not fallback_path.exists():
            raise FileNotFoundError(
                f"Seed {seed} provenance artifact was not generated."
            )
        shutil.copy2(fallback_path, provenance_path)

    provenance = json.loads(
        provenance_path.read_text(encoding="utf-8")
    )
    provenance.update(
        {
            "class_weighting_enabled": True,
            "class_weighting_mode": "sqrt_balanced",
            "class_weight_source_split": "train",
            "class_weight_min": weight_metadata["minimum_weight"],
            "class_weight_max": weight_metadata["maximum_weight"],
            "class_weight_mean": weight_metadata["mean_weight"],
            "class_weight_metadata_path": str(
                experiment_root / "class_weight_metadata.json"
            ),
            "test_used_for_challenger_selection": False,
            "test_role": "post_selection_confirmation_only",
        }
    )
    write_json(provenance_path, provenance)
    write_json(
        model_dir / "transformer_training_provenance.json",
        provenance,
    )

    seed_payload = {
        "schema_version": 1,
        "seed": seed,
        "model": "transformer_text_classifier_sqrt_weighted",
        "class_weighting_enabled": True,
        "class_weighting_mode": "sqrt_balanced",
        "selection_metrics": selection_metrics,
        "test_metrics": test_metrics,
        "test_used_for_challenger_selection": False,
        "test_role": "post_selection_confirmation_only",
        "artifact_dir": str(model_dir),
        "note": result_note,
    }
    write_json(
        seed_root / "weighted_seed_metrics.json",
        seed_payload,
    )

    weighted_rows.append(
        {
            "seed": seed,
            "selection_macro_f1": selection_metrics.get("macro_f1"),
            "selection_pr_auc": selection_metrics.get("pr_auc"),
            "selection_ece": selection_metrics.get("ece"),
            "selection_worst_slice_f1": selection_metrics.get(
                "worst_slice_f1"
            ),
            "test_macro_f1": test_metrics.get("macro_f1"),
            "test_pr_auc": test_metrics.get("pr_auc"),
            "test_ece": test_metrics.get("ece"),
            "test_worst_slice_f1": test_metrics.get(
                "worst_slice_f1"
            ),
            "best_checkpoint_macro_f1": provenance.get(
                "best_checkpoint_macro_f1"
            ),
            "best_model_checkpoint": provenance.get(
                "best_model_checkpoint"
            ),
        }
    )

weighted = pd.DataFrame(weighted_rows).sort_values(
    "seed"
).reset_index(drop=True)
weighted.to_csv(
    experiment_root / "weighted_3seed_metrics.csv",
    index=False,
)

paired_rows: list[dict[str, Any]] = []
transformer_name = "transformer_text_classifier"

for row in weighted.to_dict(orient="records"):
    seed = int(row["seed"])
    baseline_outputs = baseline_root / f"seed_{seed}" / "outputs"

    baseline_selection = pd.read_csv(
        baseline_outputs / "model_selection_leaderboard.csv"
    )
    baseline_test = pd.read_csv(
        baseline_outputs / "model_test_leaderboard.csv"
    )

    baseline_values = {
        "selection_macro_f1": metric(
            baseline_selection,
            transformer_name,
            "macro_f1",
        ),
        "selection_pr_auc": metric(
            baseline_selection,
            transformer_name,
            "pr_auc",
        ),
        "selection_ece": metric(
            baseline_selection,
            transformer_name,
            "ece",
        ),
        "selection_worst_slice_f1": metric(
            baseline_selection,
            transformer_name,
            "worst_slice_f1",
        ),
        "test_macro_f1": metric(
            baseline_test,
            transformer_name,
            "macro_f1",
        ),
        "test_pr_auc": metric(
            baseline_test,
            transformer_name,
            "pr_auc",
        ),
        "test_ece": metric(
            baseline_test,
            transformer_name,
            "ece",
        ),
        "test_worst_slice_f1": metric(
            baseline_test,
            transformer_name,
            "worst_slice_f1",
        ),
    }

    paired: dict[str, Any] = {"seed": seed}
    for metric_name, baseline_value in baseline_values.items():
        weighted_value = float(row[metric_name])
        paired[f"baseline_{metric_name}"] = baseline_value
        paired[f"weighted_{metric_name}"] = weighted_value
        paired[f"delta_{metric_name}"] = (
            weighted_value - baseline_value
        )
    paired_rows.append(paired)

paired = pd.DataFrame(paired_rows).sort_values(
    "seed"
).reset_index(drop=True)
paired.to_csv(
    experiment_root / "paired_baseline_vs_weighted.csv",
    index=False,
)

delta_columns = [
    column
    for column in paired.columns
    if column.startswith("delta_")
]
summary_rows: list[dict[str, Any]] = []

for column in delta_columns:
    values = pd.to_numeric(paired[column], errors="coerce")
    summary_rows.append(
        {
            "metric": column.removeprefix("delta_"),
            "mean_delta": float(values.mean()),
            "sample_std_delta": float(values.std(ddof=1)),
            "min_delta": float(values.min()),
            "max_delta": float(values.max()),
            "improved_seed_count": int(values.gt(0).sum()),
            "non_degraded_seed_count": int(values.ge(-0.005).sum()),
        }
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(
    experiment_root / "paired_delta_summary.csv",
    index=False,
)


def mean_delta(metric_name: str) -> float:
    column = f"delta_{metric_name}"
    return float(pd.to_numeric(paired[column]).mean())


def improved_count(metric_name: str) -> int:
    column = f"delta_{metric_name}"
    return int(pd.to_numeric(paired[column]).gt(0).sum())


selection_checks = [
    {
        "name": "selection.mean_macro_f1_delta",
        "value": mean_delta("selection_macro_f1"),
        "operator": ">=",
        "threshold": -0.005,
    },
    {
        "name": "selection.macro_f1_improved_seed_count",
        "value": improved_count("selection_macro_f1"),
        "operator": ">=",
        "threshold": 2,
    },
    {
        "name": "selection.mean_worst_slice_f1_delta",
        "value": mean_delta("selection_worst_slice_f1"),
        "operator": ">=",
        "threshold": 0.040,
    },
    {
        "name": "selection.worst_slice_improved_seed_count",
        "value": improved_count("selection_worst_slice_f1"),
        "operator": ">=",
        "threshold": 2,
    },
    {
        "name": "selection.mean_ece_delta",
        "value": mean_delta("selection_ece"),
        "operator": "<=",
        "threshold": 0.020,
    },
    {
        "name": "selection.mean_pr_auc_delta",
        "value": mean_delta("selection_pr_auc"),
        "operator": ">=",
        "threshold": -0.010,
    },
]

test_confirmation_checks = [
    {
        "name": "test_confirmation.mean_macro_f1_delta",
        "value": mean_delta("test_macro_f1"),
        "operator": ">=",
        "threshold": -0.005,
    },
    {
        "name": "test_confirmation.mean_worst_slice_f1_delta",
        "value": mean_delta("test_worst_slice_f1"),
        "operator": ">=",
        "threshold": 0.020,
    },
    {
        "name": "test_confirmation.worst_slice_improved_seed_count",
        "value": improved_count("test_worst_slice_f1"),
        "operator": ">=",
        "threshold": 2,
    },
    {
        "name": "test_confirmation.mean_ece_delta",
        "value": mean_delta("test_ece"),
        "operator": "<=",
        "threshold": 0.020,
    },
    {
        "name": "test_confirmation.mean_pr_auc_delta",
        "value": mean_delta("test_pr_auc"),
        "operator": ">=",
        "threshold": -0.010,
    },
]


def apply_checks(
    checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for check in checks:
        if check["operator"] == ">=":
            check["passed"] = bool(
                check["value"] >= check["threshold"]
            )
        else:
            check["passed"] = bool(
                check["value"] <= check["threshold"]
            )
    return checks


selection_checks = apply_checks(selection_checks)
test_confirmation_checks = apply_checks(
    test_confirmation_checks
)

selection_passed = all(
    bool(check["passed"]) for check in selection_checks
)
test_confirmation_passed = all(
    bool(check["passed"])
    for check in test_confirmation_checks
)

if not selection_passed:
    selection_decision = "REJECT_CHALLENGER"
    final_decision = "REJECT_WEIGHTED"
    next_action = (
        "Keep the unweighted Transformer. The weighted challenger "
        "did not satisfy the predeclared selection-split criteria."
    )
elif test_confirmation_passed:
    selection_decision = "PROMOTE_TO_TEST_CONFIRMATION"
    final_decision = "ADOPT_WEIGHTED"
    next_action = (
        "Adopt the sqrt-balanced Transformer as the new challenger "
        "configuration, then run the normal full launch-gate pipeline."
    )
else:
    selection_decision = "PROMOTE_TO_TEST_CONFIRMATION"
    final_decision = "REVIEW_WEIGHTED"
    next_action = (
        "Do not replace the baseline yet. Selection evidence supported "
        "the weighted challenger, but post-selection test confirmation "
        "did not meet all launch guardrails."
    )

decision = {
    "schema_version": 1,
    "experiment": "sqrt_balanced_transformer_3seed_challenger",
    "baseline_audit_root": str(baseline_root),
    "experiment_root": str(experiment_root),
    "seeds": seeds,
    "epochs": epochs,
    "batch_size": batch_size,
    "transformer_model": model_name,
    "fixed_input_sha256": fixed_sha256,
    "fixed_input_rows": fixed_rows,
    "class_weighting": weight_metadata,
    "selection_decision": selection_decision,
    "selection_checks": selection_checks,
    "selection_failed_checks": [
        check for check in selection_checks
        if not check["passed"]
    ],
    "test_confirmation_role": (
        "Post-selection launch confirmation only; not used to choose "
        "between baseline and challenger."
    ),
    "test_used_for_challenger_selection": False,
    "test_confirmation_checks": test_confirmation_checks,
    "test_confirmation_failed_checks": [
        check for check in test_confirmation_checks
        if not check["passed"]
    ],
    "final_decision": final_decision,
    "next_action": next_action,
}
write_json(
    experiment_root / "weighted_challenger_decision.json",
    decision,
)

report = [
    "# Class-Weighted Transformer 3-Seed Challenger",
    "",
    "## Experimental control",
    "",
    f"- Baseline audit: `{baseline_root}`",
    f"- Seeds: `{', '.join(map(str, seeds))}`",
    f"- Epochs: `{epochs}`",
    f"- Batch size: `{batch_size}`",
    f"- Frozen input rows: `{fixed_rows}`",
    f"- Frozen input SHA-256: `{fixed_sha256}`",
    "- Weighting: `sqrt-balanced`, train split only",
    "- Checkpoint selection: calibration Macro-F1",
    "- Challenger comparison: dedicated selection split",
    "- Test role: post-selection confirmation only",
    "- Test used for challenger selection: `false`",
    "",
    "## Class weights",
    "",
    pd.DataFrame(
        {
            "class": label_encoder.classes_,
            "weight": normalized_weights,
        }
    ).to_markdown(index=False),
    "",
    "## Paired baseline versus weighted results",
    "",
    paired.to_markdown(index=False),
    "",
    "## Paired delta summary",
    "",
    summary.to_markdown(index=False),
    "",
    "## Selection checks",
    "",
    pd.DataFrame(selection_checks).to_markdown(index=False),
    "",
    f"Selection decision: **{selection_decision}**",
    "",
    "## Post-selection test confirmation",
    "",
    pd.DataFrame(
        test_confirmation_checks
    ).to_markdown(index=False),
    "",
    f"Final decision: **{final_decision}**",
    "",
    next_action,
    "",
    (
        "The challenger decision is based on the dedicated selection "
        "split. Test metrics are retained as post-selection launch "
        "confirmation and are not used to reselect the model."
    ),
]
(
    experiment_root
    / "weighted_challenger_report.md"
).write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

print("\nPAIRED RESULTS", flush=True)
print(paired.to_string(index=False), flush=True)
print("\nDELTA SUMMARY", flush=True)
print(summary.to_string(index=False), flush=True)
print("\nSELECTION CHECKS", flush=True)
print(pd.DataFrame(selection_checks).to_string(index=False), flush=True)
print(
    f"\nSELECTION DECISION: {selection_decision}",
    flush=True,
)
print("\nTEST CONFIRMATION CHECKS", flush=True)
print(
    pd.DataFrame(test_confirmation_checks).to_string(index=False),
    flush=True,
)
print(f"\nFINAL DECISION: {final_decision}", flush=True)
print(f"NEXT ACTION: {next_action}", flush=True)
'@ | Set-Content $RunnerPath -Encoding UTF8

Write-Host "STEP 1 - Static/import preflight"

Write-Host "Auto-fixing generated-runner import order..."
& $Python -m ruff check --select I --fix $RunnerPath
if ($LASTEXITCODE -ne 0) {
    throw "Ruff import auto-fix failed for the challenger runner."
}

Write-Host "Running full Ruff validation..."
& $Python -m ruff check $RunnerPath
if ($LASTEXITCODE -ne 0) {
    throw "Ruff failed for the challenger runner."
}

& $Python -c "import torch, transformers, sklearn, pandas, numpy; print({'cuda': torch.cuda.is_available(), 'torch': torch.__version__, 'transformers': transformers.__version__})"
if ($LASTEXITCODE -ne 0) {
    throw "Python dependency preflight failed."
}

Write-Host ""
Write-Host "STEP 2 - Run the weighted 3-seed challenger"

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"

& $Python `
    $RunnerPath `
    $RepoPath `
    $BaselineAuditRoot `
    $ExperimentRoot `
    ($Seeds -join ",") `
    "$Epochs" `
    "$BatchSize" `
    $TransformerModel 2>&1 |
    Tee-Object -FilePath $LogPath

$ExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference

if ($ExitCode -ne 0) {
    throw (
        "Weighted challenger failed with exit code $ExitCode. " +
        "Log: $LogPath"
    )
}

Write-Host ""
Write-Host "STEP 3 - Verify final artifacts"

$RequiredArtifacts = @(
    "class_weight_metadata.json",
    "weighted_3seed_metrics.csv",
    "paired_baseline_vs_weighted.csv",
    "paired_delta_summary.csv",
    "weighted_challenger_decision.json",
    "weighted_challenger_report.md"
)

foreach ($RelativePath in $RequiredArtifacts) {
    $Artifact = Join-Path $ExperimentRoot $RelativePath
    if (-not (Test-Path $Artifact)) {
        throw "Missing challenger artifact: $Artifact"
    }
    Write-Host "PASS: $RelativePath"
}

foreach ($Seed in $Seeds) {
    $SeedMetrics = Join-Path `
        $ExperimentRoot `
        "s$Seed\weighted_seed_metrics.json"
    $SeedProvenance = Join-Path `
        $ExperimentRoot `
        "s$Seed\transformer_training_provenance.json"

    foreach ($Artifact in @($SeedMetrics, $SeedProvenance)) {
        if (-not (Test-Path $Artifact)) {
            throw "Missing seed artifact: $Artifact"
        }
    }
    Write-Host "PASS: seed $Seed metrics and provenance"
}

$DecisionPath = Join-Path `
    $ExperimentRoot `
    "weighted_challenger_decision.json"
$Decision = Get-Content $DecisionPath -Raw | ConvertFrom-Json

Write-Host ""
Write-Host "================================================================================"
Write-Host "CLASS-WEIGHTED CHALLENGER COMPLETED"
Write-Host "================================================================================"
Write-Host "Selection decision: $($Decision.selection_decision)"
Write-Host "Final decision:     $($Decision.final_decision)"
Write-Host "Experiment root:    $ExperimentRoot"
Write-Host "Decision artifact:  $DecisionPath"
Write-Host "Report:             $(Join-Path $ExperimentRoot 'weighted_challenger_report.md')"
Write-Host "Paired results:     $(Join-Path $ExperimentRoot 'paired_baseline_vs_weighted.csv')"
Write-Host "Delta summary:      $(Join-Path $ExperimentRoot 'paired_delta_summary.csv')"
Write-Host ""
Write-Host "Next action:"
Write-Host $Decision.next_action
