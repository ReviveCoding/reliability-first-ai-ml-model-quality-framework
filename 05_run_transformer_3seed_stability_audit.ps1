param(
    [string]$RepoPath = "${PSScriptRoot}",
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

$RequiredSplits = @(
    "data\golden\train.csv",
    "data\golden\calibration.csv",
    "data\golden\selection.csv",
    "data\golden\test.csv"
)

foreach ($RelativePath in $RequiredSplits) {
    $FullPath = Join-Path $RepoPath $RelativePath
    if (-not (Test-Path $FullPath)) {
        throw "Required frozen split not found: $FullPath"
    }
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$AuditBase = Join-Path $RepoPath "sa"
$AuditRoot = Join-Path $AuditBase "s_$Stamp"
$FixedInputDir = Join-Path $AuditRoot "fixed_input"
$FixedInputCsv = Join-Path $FixedInputDir "cfpb_fixed_audit_input.csv"
$FixedManifest = Join-Path $FixedInputDir "fixed_input_manifest.json"
$LogsDir = Join-Path $AuditRoot "logs"

New-Item -ItemType Directory -Force $FixedInputDir | Out-Null
New-Item -ItemType Directory -Force $LogsDir | Out-Null

Write-Host ""
Write-Host "================================================================================"
Write-Host "TRANSFORMER 3-SEED STABILITY AUDIT"
Write-Host "================================================================================"
Write-Host "Repo:          $RepoPath"
Write-Host "Audit root:    $AuditRoot"
Write-Host "Path policy:   short Windows-safe working paths"
Write-Host "Seeds:         $($Seeds -join ', ')"
Write-Host "Epochs:        $Epochs"
Write-Host "Batch size:    $BatchSize"
Write-Host "Model:         $TransformerModel"
Write-Host ""

Write-Host "STEP 1 - CUDA preflight"

$CudaPreflightScript = Join-Path $AuditRoot "_cuda_preflight.py"

@'
import json

import torch


result = {
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "device_name": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None
    ),
}

print(json.dumps(result, indent=2))

if not result["cuda_available"]:
    raise SystemExit("CUDA is required for this audit.")
'@ | Set-Content $CudaPreflightScript -Encoding UTF8

& $Python $CudaPreflightScript
$CudaExitCode = $LASTEXITCODE

Remove-Item $CudaPreflightScript -Force -ErrorAction SilentlyContinue

if ($CudaExitCode -ne 0) {
    throw "CUDA preflight failed with exit code $CudaExitCode."
}

Write-Host ""
Write-Host "STEP 2 - Build one frozen audit dataset from the verified golden splits"

$BuildFixedInput = Join-Path $AuditRoot "_build_fixed_input.py"

@'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


repo = Path(sys.argv[1])
output_csv = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])

split_paths = {
    "train": repo / "data" / "golden" / "train.csv",
    "calibration": repo / "data" / "golden" / "calibration.csv",
    "selection": repo / "data" / "golden" / "selection.csv",
    "test": repo / "data" / "golden" / "test.csv",
}

required_columns = {
    "product",
    "issue",
    "consumer_complaint_narrative",
    "company_response_to_consumer",
    "timely_response",
    "date_received",
    "state",
}

frames: list[pd.DataFrame] = []
source_counts: dict[str, int] = {}

for split_name, path in split_paths.items():
    frame = pd.read_csv(path, low_memory=False)
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise RuntimeError(f"{split_name} is missing columns: {missing}")

    source_counts[split_name] = int(len(frame))
    frames.append(frame)

fixed = pd.concat(frames, ignore_index=True)

if "complaint_id" in fixed.columns:
    duplicate_ids = int(fixed["complaint_id"].astype(str).duplicated().sum())
    if duplicate_ids:
        raise RuntimeError(
            f"Frozen splits contain {duplicate_ids} duplicate complaint_id values."
        )
else:
    duplicate_ids = None

sort_columns = [
    column
    for column in ("date_received", "complaint_id")
    if column in fixed.columns
]
if sort_columns:
    fixed = fixed.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

output_csv.parent.mkdir(parents=True, exist_ok=True)
fixed.to_csv(output_csv, index=False)

sha256 = hashlib.sha256(output_csv.read_bytes()).hexdigest()
date_values = pd.to_datetime(fixed["date_received"], errors="coerce")

manifest = {
    "schema_version": 1,
    "purpose": "Frozen input for Transformer model-seed stability audit",
    "row_count": int(len(fixed)),
    "source_split_counts": source_counts,
    "source_split_total": int(sum(source_counts.values())),
    "sha256": sha256,
    "duplicate_complaint_id_count": duplicate_ids,
    "date_min": (
        str(date_values.min())
        if date_values.notna().any()
        else None
    ),
    "date_max": (
        str(date_values.max())
        if date_values.notna().any()
        else None
    ),
    "product_class_count": int(fixed["product"].nunique(dropna=True)),
    "required_columns_present": sorted(required_columns),
    "sampling_policy": (
        "The fixed CSV contains every row from the current verified "
        "train/calibration/selection/test splits. Each seed run reads all rows."
    ),
}

manifest_path.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print(json.dumps(manifest, indent=2))
'@ | Set-Content $BuildFixedInput -Encoding UTF8

& $Python `
    $BuildFixedInput `
    $RepoPath `
    $FixedInputCsv `
    $FixedManifest

if ($LASTEXITCODE -ne 0) {
    throw "Failed to construct the frozen audit dataset."
}

Remove-Item $BuildFixedInput -Force

$FixedInfo = Get-Content $FixedManifest -Raw | ConvertFrom-Json
$FixedRows = [int]$FixedInfo.row_count
$FixedSha256 = [string]$FixedInfo.sha256

if ($FixedRows -lt 1000) {
    throw "Frozen audit dataset is unexpectedly small: $FixedRows rows."
}

Write-Host ""
Write-Host "Frozen rows:   $FixedRows"
Write-Host "Frozen SHA256: $FixedSha256"

Write-Host ""
Write-Host "STEP 3 - Run the identical frozen dataset with three model seeds"
Write-Host "Class weighting is NOT enabled in this audit."
Write-Host "CrossEncoder and LightGBM are omitted because this audit isolates Transformer stability."

foreach ($Seed in $Seeds) {
    $SeedRoot = Join-Path $AuditRoot "seed_$Seed"
    $DataDir = Join-Path $SeedRoot "data"
    $OutputsDir = Join-Path $SeedRoot "outputs"
    $ReportsDir = Join-Path $SeedRoot "reports"
    $TrackingDir = Join-Path $SeedRoot "m"
    $LogPath = Join-Path $LogsDir "seed_$Seed.log"

    # Keep a safety margin below the traditional Windows MAX_PATH boundary.
    $EstimatedFallbackArtifact = Join-Path `
        $TrackingDir `
        "fallback_runs\tfidf_logistic_temperature_scaled\20260101T000000000000Z_00000000\artifacts\tfidf_logistic_temperature_scaled.joblib"

    if ($EstimatedFallbackArtifact.Length -ge 240) {
        throw (
            "Estimated fallback artifact path is too long for a portable " +
            "Windows run: $($EstimatedFallbackArtifact.Length) characters. " +
            "Path: $EstimatedFallbackArtifact"
        )
    }

    New-Item -ItemType Directory -Force $SeedRoot | Out-Null

    Write-Host ""
    Write-Host "--------------------------------------------------------------------------------"
    Write-Host "SEED $Seed"
    Write-Host "Estimated fallback artifact path length: $($EstimatedFallbackArtifact.Length)"
    Write-Host "--------------------------------------------------------------------------------"

    $PipelineArguments = @(
        "scripts\run_full_pipeline_with_semantics.py",
        "--cfpb-path", $FixedInputCsv,
        "--sample-size", "$FixedRows",
        "--sampling-strategy", "auto",
        "--archive-cache-dir", (Join-Path $RepoPath "data\archive_cache"),
        "--min-target-count", "30",
        "--enable-transformer",
        "--transformer-model", $TransformerModel,
        "--transformer-epochs", "$Epochs",
        "--transformer-batch-size", "$BatchSize",
        "--device", "cuda",
        "--random-state", "$Seed",
        "--data-dir", $DataDir,
        "--outputs-dir", $OutputsDir,
        "--reports-dir", $ReportsDir,
        "--tracking-uri", $TrackingDir,
        "--disable-mlflow",
        "--semantics-mode", "smoke"
    )

    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    & $Python @PipelineArguments 2>&1 |
        Tee-Object -FilePath $LogPath

    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference

    if ($ExitCode -ne 0) {
        throw "Seed $Seed pipeline failed with exit code $ExitCode. Log: $LogPath"
    }

    $Provenance = Join-Path `
        $OutputsDir `
        "transformer_training_provenance.json"
    $SelectionDecision = Join-Path `
        $OutputsDir `
        "model_selection_decision.json"
    $TestLeaderboard = Join-Path `
        $OutputsDir `
        "model_test_leaderboard.csv"

    foreach ($RequiredArtifact in @(
        $Provenance,
        $SelectionDecision,
        $TestLeaderboard
    )) {
        if (-not (Test-Path $RequiredArtifact)) {
            throw "Seed $Seed missing artifact: $RequiredArtifact"
        }
    }

    $CurrentSha256 = (
        Get-FileHash $FixedInputCsv -Algorithm SHA256
    ).Hash.ToLowerInvariant()

    if ($CurrentSha256 -ne $FixedSha256.ToLowerInvariant()) {
        throw "Frozen input changed during seed $Seed."
    }

    Write-Host "Seed $Seed completed and frozen-input hash verified."
}

Write-Host ""
Write-Host "STEP 4 - Aggregate seed metrics and issue a stability decision"

$AggregateScript = Join-Path $AuditRoot "_aggregate_seed_audit.py"

@'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


audit_root = Path(sys.argv[1])
seeds = [int(value) for value in sys.argv[2].split(",")]
epochs = int(sys.argv[3])
fixed_manifest = json.loads(
    (audit_root / "fixed_input" / "fixed_input_manifest.json").read_text(
        encoding="utf-8"
    )
)

rows: list[dict[str, object]] = []

for seed in seeds:
    outputs = audit_root / f"seed_{seed}" / "outputs"

    provenance = json.loads(
        (outputs / "transformer_training_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    selection = json.loads(
        (outputs / "model_selection_decision.json").read_text(
            encoding="utf-8"
        )
    )
    launch = json.loads(
        (outputs / "launch_gate_result.json").read_text(
            encoding="utf-8"
        )
    )
    leaderboard = pd.read_csv(outputs / "model_test_leaderboard.csv")
    transformer_row = leaderboard.loc[
        leaderboard["model"].eq("transformer_text_classifier")
    ]
    fallback_row = leaderboard.loc[
        leaderboard["model"].eq(
            "tfidf_logistic_temperature_scaled"
        )
    ]

    if len(transformer_row) != 1:
        raise RuntimeError(
            f"Seed {seed}: expected one Transformer test row."
        )

    transformer = transformer_row.iloc[0]
    fallback = fallback_row.iloc[0] if len(fallback_row) == 1 else None

    if provenance.get("test_used_for_checkpoint_selection") is not False:
        raise RuntimeError(
            f"Seed {seed}: test was used for checkpoint selection."
        )
    if provenance.get("test_used_for_framework_selection") is not False:
        raise RuntimeError(
            f"Seed {seed}: test was used for framework selection."
        )
    if provenance.get("test_used_for_reselection") is not False:
        raise RuntimeError(
            f"Seed {seed}: test was used for reselection."
        )

    row = {
        "seed": seed,
        "selection_winner": selection.get("selection_winner"),
        "eligibility_status": selection.get("eligibility_status"),
        "launch_decision": launch.get("decision"),
        "best_checkpoint_macro_f1": provenance.get(
            "best_checkpoint_macro_f1"
        ),
        "selection_macro_f1": provenance.get(
            "framework_selection_macro_f1"
        ),
        "test_macro_f1": provenance.get("candidate_test_macro_f1"),
        "test_pr_auc": provenance.get("candidate_test_pr_auc"),
        "test_ece": provenance.get("candidate_test_ece"),
        "test_worst_slice_f1": provenance.get(
            "candidate_test_worst_slice_f1"
        ),
        "best_model_checkpoint": provenance.get(
            "best_model_checkpoint"
        ),
        "selection_rows": provenance.get("selection_rows_evaluated"),
        "test_rows": provenance.get("test_rows_evaluated"),
        "fallback_test_macro_f1": (
            float(fallback["macro_f1"])
            if fallback is not None
            else None
        ),
        "transformer_minus_fallback_test_macro_f1": (
            float(transformer["macro_f1"])
            - float(fallback["macro_f1"])
            if fallback is not None
            else None
        ),
    }
    rows.append(row)

per_seed = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
per_seed.to_csv(audit_root / "transformer_3seed_metrics.csv", index=False)

metric_columns = [
    "best_checkpoint_macro_f1",
    "selection_macro_f1",
    "test_macro_f1",
    "test_pr_auc",
    "test_ece",
    "test_worst_slice_f1",
    "transformer_minus_fallback_test_macro_f1",
]

summary_rows: list[dict[str, object]] = []

for metric in metric_columns:
    values = pd.to_numeric(per_seed[metric], errors="coerce").dropna()
    if values.empty:
        continue

    mean = float(values.mean())
    sample_std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    minimum = float(values.min())
    maximum = float(values.max())
    metric_range = maximum - minimum
    coefficient_of_variation = (
        abs(sample_std / mean)
        if not math.isclose(mean, 0.0)
        else None
    )

    summary_rows.append(
        {
            "metric": metric,
            "n": int(len(values)),
            "mean": mean,
            "sample_std": sample_std,
            "min": minimum,
            "max": maximum,
            "range": metric_range,
            "coefficient_of_variation": coefficient_of_variation,
        }
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(audit_root / "transformer_3seed_summary.csv", index=False)

# These are project-level stability guardrails, not universal statistical laws.
thresholds = {
    "selection_macro_f1_sample_std_max": 0.020,
    "test_macro_f1_sample_std_max": 0.020,
    "test_macro_f1_range_max": 0.050,
    "test_pr_auc_sample_std_max": 0.010,
    "test_ece_sample_std_max": 0.020,
    "test_worst_slice_f1_sample_std_max": 0.050,
    "test_worst_slice_f1_range_max": 0.120,
    "transformer_selection_rate_min": 2.0 / 3.0,
}

def stat(metric: str, field: str) -> float:
    result = summary.loc[summary["metric"].eq(metric), field]
    if len(result) != 1:
        raise RuntimeError(f"Missing summary metric: {metric}.{field}")
    return float(result.iloc[0])


transformer_selection_rate = float(
    per_seed["selection_winner"]
    .eq("transformer_text_classifier")
    .mean()
)

checks = [
    {
        "name": "selection_macro_f1.sample_std",
        "value": stat("selection_macro_f1", "sample_std"),
        "operator": "<=",
        "threshold": thresholds[
            "selection_macro_f1_sample_std_max"
        ],
    },
    {
        "name": "test_macro_f1.sample_std",
        "value": stat("test_macro_f1", "sample_std"),
        "operator": "<=",
        "threshold": thresholds[
            "test_macro_f1_sample_std_max"
        ],
    },
    {
        "name": "test_macro_f1.range",
        "value": stat("test_macro_f1", "range"),
        "operator": "<=",
        "threshold": thresholds["test_macro_f1_range_max"],
    },
    {
        "name": "test_pr_auc.sample_std",
        "value": stat("test_pr_auc", "sample_std"),
        "operator": "<=",
        "threshold": thresholds[
            "test_pr_auc_sample_std_max"
        ],
    },
    {
        "name": "test_ece.sample_std",
        "value": stat("test_ece", "sample_std"),
        "operator": "<=",
        "threshold": thresholds["test_ece_sample_std_max"],
    },
    {
        "name": "test_worst_slice_f1.sample_std",
        "value": stat("test_worst_slice_f1", "sample_std"),
        "operator": "<=",
        "threshold": thresholds[
            "test_worst_slice_f1_sample_std_max"
        ],
    },
    {
        "name": "test_worst_slice_f1.range",
        "value": stat("test_worst_slice_f1", "range"),
        "operator": "<=",
        "threshold": thresholds[
            "test_worst_slice_f1_range_max"
        ],
    },
    {
        "name": "transformer_selection_rate",
        "value": transformer_selection_rate,
        "operator": ">=",
        "threshold": thresholds[
            "transformer_selection_rate_min"
        ],
    },
]

for check in checks:
    if check["operator"] == "<=":
        check["passed"] = bool(
            check["value"] <= check["threshold"]
        )
    else:
        check["passed"] = bool(
            check["value"] >= check["threshold"]
        )

failed = [check for check in checks if not check["passed"]]
core_names = {
    "selection_macro_f1.sample_std",
    "test_macro_f1.sample_std",
    "test_macro_f1.range",
    "transformer_selection_rate",
}
failed_core = [
    check for check in failed if check["name"] in core_names
]

if not failed:
    verdict = "STABLE"
elif failed_core:
    verdict = "UNSTABLE"
else:
    verdict = "BORDERLINE"

decision = {
    "schema_version": 1,
    "audit": "transformer_3seed_stability",
    "seed_count": len(seeds),
    "seeds": seeds,
    "fixed_input_sha256": fixed_manifest["sha256"],
    "fixed_input_row_count": fixed_manifest["row_count"],
    "class_weighting_enabled": False,
    "epochs": epochs,
    "transformer_selection_rate": transformer_selection_rate,
    "threshold_note": (
        "Project-specific engineering guardrails for a 3-seed audit; "
        "not universal statistical thresholds."
    ),
    "thresholds": thresholds,
    "checks": checks,
    "failed_checks": failed,
    "verdict": verdict,
    "next_action": (
        "Proceed to a class-weighted Transformer challenger experiment."
        if verdict == "STABLE"
        else (
            "Proceed cautiously: report slice instability and compare "
            "class weighting without claiming a stable improvement."
            if verdict == "BORDERLINE"
            else (
                "Do not attribute changes to class weighting yet. "
                "First stabilize initialization/data-order behavior "
                "and repeat the seed audit."
            )
        )
    ),
}

(audit_root / "transformer_3seed_stability_decision.json").write_text(
    json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

report_lines = [
    "# Transformer 3-Seed Stability Audit",
    "",
    "## Experimental control",
    "",
    f"- Seeds: `{', '.join(map(str, seeds))}`",
    f"- Frozen input rows: `{fixed_manifest['row_count']}`",
    f"- Frozen input SHA-256: `{fixed_manifest['sha256']}`",
    "- Class weighting: `disabled`",
    "- Checkpoint selection: calibration Macro-F1",
    "- Framework model selection: dedicated selection split",
    "- Final evaluation: untouched test split",
    "- Test used for tuning or reselection: `false`",
    "",
    "## Per-seed results",
    "",
    per_seed.to_markdown(index=False),
    "",
    "## Stability summary",
    "",
    summary.to_markdown(index=False),
    "",
    "## Guardrail checks",
    "",
    pd.DataFrame(checks).to_markdown(index=False),
    "",
    "## Decision",
    "",
    f"**{verdict}**",
    "",
    decision["next_action"],
    "",
    (
        "This is a three-seed engineering stability audit. "
        "It characterizes run-to-run sensitivity but is not a "
        "high-powered statistical study."
    ),
]
(audit_root / "transformer_3seed_stability_report.md").write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

print("\nPER-SEED METRICS")
print(per_seed.to_string(index=False))
print("\nSTABILITY SUMMARY")
print(summary.to_string(index=False))
print("\nGUARDRAIL CHECKS")
print(pd.DataFrame(checks).to_string(index=False))
print(f"\nFINAL VERDICT: {verdict}")
print(f"NEXT ACTION: {decision['next_action']}")
'@ | Set-Content $AggregateScript -Encoding UTF8

& $Python `
    $AggregateScript `
    $AuditRoot `
    ($Seeds -join ",") `
    "$Epochs"

if ($LASTEXITCODE -ne 0) {
    throw "Failed to aggregate the 3-seed audit."
}

Remove-Item $AggregateScript -Force

Write-Host ""
Write-Host "STEP 5 - Final artifact verification"

$RequiredAuditArtifacts = @(
    "fixed_input\fixed_input_manifest.json",
    "transformer_3seed_metrics.csv",
    "transformer_3seed_summary.csv",
    "transformer_3seed_stability_decision.json",
    "transformer_3seed_stability_report.md"
)

foreach ($RelativePath in $RequiredAuditArtifacts) {
    $ArtifactPath = Join-Path $AuditRoot $RelativePath
    if (-not (Test-Path $ArtifactPath)) {
        throw "Missing audit artifact: $ArtifactPath"
    }
    Write-Host "PASS: $RelativePath"
}

$DecisionPath = Join-Path `
    $AuditRoot `
    "transformer_3seed_stability_decision.json"
$Decision = Get-Content $DecisionPath -Raw | ConvertFrom-Json

Write-Host ""
Write-Host "================================================================================"
Write-Host "3-SEED STABILITY AUDIT COMPLETED"
Write-Host "================================================================================"
Write-Host "Verdict:      $($Decision.verdict)"
Write-Host "Audit root:   $AuditRoot"
Write-Host "Decision:     $DecisionPath"
Write-Host "Report:       $(Join-Path $AuditRoot 'transformer_3seed_stability_report.md')"
Write-Host "Per-seed CSV: $(Join-Path $AuditRoot 'transformer_3seed_metrics.csv')"
Write-Host "Summary CSV:  $(Join-Path $AuditRoot 'transformer_3seed_summary.csv')"
Write-Host ""
Write-Host "Next action:"
Write-Host $Decision.next_action
