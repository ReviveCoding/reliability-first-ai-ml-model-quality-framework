[CmdletBinding()]
param(
    [string]$RepoPath = "${PSScriptRoot}",
    [string]$DatasetPath = "${env:USERPROFILE}\Downloads\Project_Data\CFPB Complaint Data\CFPB complaints 20200518-20260518 complaints only with narratives.csv",
    [ValidateSet("Smoke", "Full", "All")]
    [string]$Mode = "All",
    [string]$PythonVersion = "3.11",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu121",
    [string]$TorchVersion = "2.5.1",
    [int]$CpuThreads = 8,
    [int]$CpuSmokeSampleSize = 1000,
    [int]$GpuSmokeSampleSize = 5000,
    [int]$FullSampleSize = 20000,
    [int]$SmokeTransformerEpochs = 1,
    [int]$FullTransformerEpochs = 2,
    [int]$TransformerBatchSize = 32,
    [int]$CrossEncoderBatchSize = 64,
    [int]$MonteCarloRunsPerScenario = 2,
    [int]$MonteCarloJobs = 3,
    [switch]$RecreateVenv,
    [switch]$ReinstallTorch,
    [switch]$SkipTests,
    [switch]$SkipMonteCarlo,
    [switch]$LaunchUIs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# BEGIN LAUNCH DECISION SCHEMA COMPATIBILITY
function Get-LaunchDecision {
    param(
        [Parameter(Mandatory = $true)]
        [object]$GateObject
    )

    if ($null -eq $GateObject) {
        throw "Launch-gate object is null."
    }

    $DecisionProperty = $GateObject.PSObject.Properties["decision"]
    if (
        $null -ne $DecisionProperty -and
        -not [string]::IsNullOrWhiteSpace([string]$DecisionProperty.Value)
    ) {
        return ([string]$DecisionProperty.Value).ToUpperInvariant()
    }

    $StatusProperty = $GateObject.PSObject.Properties["status"]
    if (
        $null -ne $StatusProperty -and
        -not [string]::IsNullOrWhiteSpace([string]$StatusProperty.Value)
    ) {
        return ([string]$StatusProperty.Value).ToUpperInvariant()
    }

    $AvailableProperties = @(
        $GateObject.PSObject.Properties.Name
    ) -join ", "

    throw (
        "Launch-gate object contains neither 'decision' nor legacy " +
        "'status'. Available properties: $AvailableProperties"
    )
}
# END LAUNCH DECISION SCHEMA COMPATIBILITY

# BEGIN MODEL ROLE SCHEMA COMPATIBILITY
function Get-SelectionWinner {
    param(
        [Parameter(Mandatory = $true)]
        [object]$InputObject
    )

    if ($null -eq $InputObject) {
        throw "Model-role object is null."
    }

    # Current semantics schema: selection winner is always the model selected
    # on the dedicated selection split, even when it is not promotion-eligible.
    foreach ($PropertyName in @(
        "selection_winner",
        "promotion_candidate",
        "approved_champion",
        "champion_model"
    )) {
        $Property = $InputObject.PSObject.Properties[$PropertyName]

        if (
            $null -ne $Property -and
            -not [string]::IsNullOrWhiteSpace([string]$Property.Value)
        ) {
            return [string]$Property.Value
        }
    }

    $AvailableProperties = @(
        $InputObject.PSObject.Properties.Name
    ) -join ", "

    throw (
        "Model-role object contains none of: selection_winner, " +
        "promotion_candidate, approved_champion, legacy champion_model. " +
        "Available properties: $AvailableProperties"
    )
}
# END MODEL ROLE SCHEMA COMPATIBILITY

function Write-Stage {
    param([string]$Message)
    Write-Host ""
    Write-Host ("=" * 96) -ForegroundColor DarkCyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host ("=" * 96) -ForegroundColor DarkCyan
}

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogPath
    )

    Write-Host "[$Label]" -ForegroundColor Yellow
    Write-Host ("  " + $Executable + " " + ($Arguments -join " ")) -ForegroundColor DarkGray

    $logDir = Split-Path -Parent $LogPath
    if ($logDir) {
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    }

    # Native programs often write warnings/progress to stderr. Under Windows PowerShell 5.1,
    # $ErrorActionPreference = "Stop" can turn that harmless stderr into a terminating NativeCommandError.
    # Temporarily allow native stderr, then decide success strictly from the process exit code.
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Executable @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode. See: $LogPath"
    }
}

function Backup-PathIfPresent {
    param(
        [string]$PathToBackup,
        [string]$BackupRoot
    )
    if (Test-Path $PathToBackup) {
        New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
        $leaf = Split-Path -Leaf $PathToBackup
        Move-Item -Force $PathToBackup (Join-Path $BackupRoot $leaf)
    }
}

function Assert-GpuRunArtifacts {
    param([string]$OutputsDir)

    $manifestPath = Join-Path $OutputsDir "pipeline_manifest.json"
    $leaderboardPath = Join-Path $OutputsDir "model_test_leaderboard.csv"
    $evidencePath = Join-Path $OutputsDir "evidence_quality_summary.json"
    $gatePath = Join-Path $OutputsDir "launch_gate_result.json"
    $transformerDir = Join-Path $OutputsDir "transformer_model"

    foreach ($required in @($manifestPath, $leaderboardPath, $evidencePath, $gatePath)) {
        if (-not (Test-Path $required)) {
            throw "Required GPU-run artifact is missing: $required"
        }
    }

    $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
    $evidence = Get-Content $evidencePath -Raw | ConvertFrom-Json
    $gate = Get-Content $gatePath -Raw | ConvertFrom-Json
    $leaderboard = Import-Csv $leaderboardPath
    $transformerRow = $leaderboard | Where-Object { $_.model -eq "transformer_text_classifier" }

    if (-not $transformerRow) {
        throw "Transformer row was not found in model_test_leaderboard.csv. Check the pipeline log."
    }
    if (-not (Test-Path $transformerDir)) {
        throw "Transformer artifact directory was not created: $transformerDir"
    }
    if (-not [bool]$evidence.used_cross_encoder) {
        throw "CrossEncoder was requested but was not effectively used. Check model download/device logs."
    }
    if ($manifest.optional_features.device -notmatch "cuda") {
        throw "Pipeline manifest does not report a CUDA device: $($manifest.optional_features.device)"
    }

    Write-Host "GPU artifact verification: PASS" -ForegroundColor Green
    Write-Host "  Launch gate: $((Get-LaunchDecision -GateObject $gate))" -ForegroundColor Green
    Write-Host "  Selection winner: $((Get-SelectionWinner -InputObject $manifest))" -ForegroundColor Green
    Write-Host "  CrossEncoder effective: $($manifest.optional_features.cross_encoder_effective)" -ForegroundColor Green
    Write-Host "  Transformer test macro-F1: $($transformerRow.macro_f1)" -ForegroundColor Green
}

# ----------------------------------------------------------------------------------------------------------------------
# Paths and environment
# ----------------------------------------------------------------------------------------------------------------------
$RepoPath = [System.IO.Path]::GetFullPath($RepoPath)
$DatasetPath = [System.IO.Path]::GetFullPath($DatasetPath)

if (-not (Test-Path $RepoPath -PathType Container)) {
    throw "Repo folder does not exist: $RepoPath"
}
if (-not (Test-Path $DatasetPath -PathType Leaf)) {
    throw "CFPB dataset does not exist: $DatasetPath"
}

$RequiredRepoFiles = @(
    "scripts\run_full_pipeline_with_semantics.py",
    "scripts\run_monte_carlo.py",
    "scripts\gpu_preflight.py",
    "scripts\dataset_preflight.py",
    "requirements.txt",
    "requirements-gpu.txt",
    "pyproject.toml"
)
foreach ($relativePath in $RequiredRepoFiles) {
    $fullPath = Join-Path $RepoPath $relativePath
    if (-not (Test-Path $fullPath)) {
        throw "Required repo file is missing: $fullPath. Use the latest v7 project contents."
    }
}

Set-Location $RepoPath

$TimeStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $RepoPath "runs\$TimeStamp"
$LogsRoot = Join-Path $RunRoot "logs"
$EnvironmentRoot = Join-Path $RunRoot "environment"
$CpuSmokeRoot = Join-Path $RunRoot "01_cpu_smoke"
$GpuSmokeRoot = Join-Path $RunRoot "02_gpu_smoke"
$FullRunArchiveRoot = Join-Path $RunRoot "previous_root_artifacts"
$ArchiveCache = Join-Path $RepoPath "data\archive_cache"
$VenvPath = Join-Path $RepoPath ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$VenvMlflow = Join-Path $VenvPath "Scripts\mlflow.exe"
$VenvStreamlit = Join-Path $VenvPath "Scripts\streamlit.exe"

New-Item -ItemType Directory -Force -Path $RunRoot, $LogsRoot, $EnvironmentRoot, $ArchiveCache | Out-Null

$env:MODEL_QUALITY_CPU_THREADS = "$CpuThreads"
$env:OPENBLAS_NUM_THREADS = "$CpuThreads"
$env:OMP_NUM_THREADS = "$CpuThreads"
$env:MKL_NUM_THREADS = "$CpuThreads"
$env:NUMEXPR_NUM_THREADS = "$CpuThreads"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:HF_HOME = Join-Path $RepoPath ".cache\huggingface"
$env:HF_DATASETS_CACHE = Join-Path $RepoPath ".cache\huggingface\datasets"
$env:TOKENIZERS_PARALLELISM = "false"

Start-Transcript -Path (Join-Path $LogsRoot "master_transcript.log") -Append | Out-Null

try {
    Write-Stage "STEP 0 - Validate local paths and NVIDIA visibility"
    Write-Host "Repo:    $RepoPath"
    Write-Host "Dataset: $DatasetPath"
    Write-Host "Run root: $RunRoot"

    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & nvidia-smi | Tee-Object -FilePath (Join-Path $EnvironmentRoot "nvidia-smi.txt")
    }
    else {
        throw "nvidia-smi was not found. Install/update the NVIDIA driver before GPU training."
    }

    # ------------------------------------------------------------------------------------------------------------------
    # Virtual environment and dependencies
    # ------------------------------------------------------------------------------------------------------------------
    Write-Stage "STEP 1 - Create the repo-local Python virtual environment"
    if ($RecreateVenv -and (Test-Path $VenvPath)) {
        Remove-Item -Recurse -Force $VenvPath
    }
    if (-not (Test-Path $VenvPython)) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            Invoke-LoggedCommand "Create .venv with Python $PythonVersion" "py" @("-$PythonVersion", "-m", "venv", $VenvPath) (Join-Path $LogsRoot "01_create_venv.log")
        }
        elseif (Get-Command python -ErrorAction SilentlyContinue) {
            Invoke-LoggedCommand "Create .venv with default Python" "python" @("-m", "venv", $VenvPath) (Join-Path $LogsRoot "01_create_venv.log")
        }
        else {
            throw "Neither 'py' nor 'python' was found on PATH. Install Python 3.11 first."
        }
    }

    Invoke-LoggedCommand "Upgrade pip/setuptools/wheel" $VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel") (Join-Path $LogsRoot "02_upgrade_packaging.log")

    Write-Stage "STEP 2 - Install core/full project dependencies"
    Invoke-LoggedCommand "Install requirements.txt" $VenvPython @("-m", "pip", "install", "-r", (Join-Path $RepoPath "requirements.txt")) (Join-Path $LogsRoot "03_install_requirements.log")

    Write-Stage "STEP 3 - Install or verify CUDA-enabled PyTorch"
    $cudaReadyBefore = $false
    try {
        $cudaStatus = & $VenvPython -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2>$null
        $cudaReadyBefore = ($LASTEXITCODE -eq 0 -and $cudaStatus -contains "1")
    }
    catch {
        $cudaReadyBefore = $false
    }

    if ($ReinstallTorch -or -not $cudaReadyBefore) {
        # Only uninstall packages that are actually present. `pip uninstall` writes a harmless warning to
        # stderr when a package is absent, which older Windows PowerShell can misclassify as a fatal error.
        $installedTorchPackages = @()
        foreach ($packageName in @("torch", "torchvision", "torchaudio")) {
            # Do not use `pip show` here. When a package is absent, pip writes a harmless
            # warning to stderr, which Windows PowerShell can promote to NativeCommandError
            # under ErrorActionPreference=Stop. `find_spec` is silent and returns only an
            # exit code, so missing optional packages are handled safely.
            & $VenvPython -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$packageName') is not None else 1)"
            if ($LASTEXITCODE -eq 0) {
                $installedTorchPackages += $packageName
            }
        }
        if ($installedTorchPackages.Count -gt 0) {
            Invoke-LoggedCommand "Uninstall existing PyTorch packages" $VenvPython (@("-m", "pip", "uninstall", "-y") + $installedTorchPackages) (Join-Path $LogsRoot "04_uninstall_existing_torch.log")
        }
        else {
            Write-Host "No existing PyTorch packages found; uninstall skipped." -ForegroundColor DarkGray
        }

        Invoke-LoggedCommand "Install CUDA PyTorch" $VenvPython @("-m", "pip", "install", "torch==$TorchVersion", "--index-url", $TorchIndexUrl) (Join-Path $LogsRoot "05_install_cuda_torch.log")
    }
    else {
        Write-Host "Existing CUDA-enabled PyTorch detected; reinstall skipped." -ForegroundColor Green
    }

    Write-Stage "STEP 4 - Install Transformer and CrossEncoder dependencies"
    Invoke-LoggedCommand "Install requirements-gpu.txt" $VenvPython @("-m", "pip", "install", "-r", (Join-Path $RepoPath "requirements-gpu.txt")) (Join-Path $LogsRoot "06_install_gpu_requirements.log")
    Invoke-LoggedCommand "Editable install" $VenvPython @("-m", "pip", "install", "-e", ".", "--no-deps") (Join-Path $LogsRoot "07_editable_install.log")
    Invoke-LoggedCommand "Dependency consistency check" $VenvPython @("-m", "pip", "check") (Join-Path $LogsRoot "08_pip_check.log")
    Invoke-LoggedCommand "Save pip freeze" $VenvPython @("-m", "pip", "freeze") (Join-Path $EnvironmentRoot "pip_freeze.txt")

    # ------------------------------------------------------------------------------------------------------------------
    # Preflight and code validation
    # ------------------------------------------------------------------------------------------------------------------
    Write-Stage "STEP 5 - Strict CUDA/GPU preflight"
    Invoke-LoggedCommand "GPU preflight" $VenvPython @(
        "scripts\gpu_preflight.py",
        "--require-cuda",
        "--output", (Join-Path $EnvironmentRoot "gpu_preflight.json")
    ) (Join-Path $LogsRoot "09_gpu_preflight.log")

    if (-not $SkipTests) {
        Write-Stage "STEP 6 - Static checks and unit/integration tests"
        Invoke-LoggedCommand "Ruff" $VenvPython @("-m", "ruff", "check", ".") (Join-Path $LogsRoot "10_ruff.log")
        Invoke-LoggedCommand "Compile Python files" $VenvPython @("-m", "compileall", "-q", "src", "scripts", "dashboard") (Join-Path $LogsRoot "11_compileall.log")
        Invoke-LoggedCommand "Pytest" $VenvPython @("-m", "pytest", "-q") (Join-Path $LogsRoot "12_pytest.log")
    }

    Write-Stage "STEP 7 - CFPB dataset preflight and schema/sample validation"
    Invoke-LoggedCommand "Dataset preflight" $VenvPython @(
        "scripts\dataset_preflight.py",
        "--cfpb-path", $DatasetPath,
        "--data-dir", (Join-Path $RepoPath "data"),
        "--archive-cache-dir", $ArchiveCache,
        "--sample-size", "$CpuSmokeSampleSize",
        "--sampling-strategy", "auto",
        "--output", (Join-Path $EnvironmentRoot "dataset_preflight.json")
    ) (Join-Path $LogsRoot "13_dataset_preflight.log")

    # ------------------------------------------------------------------------------------------------------------------
    # CPU baseline smoke
    # ------------------------------------------------------------------------------------------------------------------
    Write-Stage "STEP 8 - Real-data CPU smoke: processing, baseline training, evaluation, drift, telemetry, gate"
    New-Item -ItemType Directory -Force -Path $CpuSmokeRoot | Out-Null
    Invoke-LoggedCommand "CPU smoke pipeline" $VenvPython @(
        "scripts\run_full_pipeline_with_semantics.py",
        "--cfpb-path", $DatasetPath,
        "--sample-size", "$CpuSmokeSampleSize",
        "--sampling-strategy", "auto",
        "--archive-cache-dir", $ArchiveCache,
        "--min-target-count", "10",
        "--enable-lightgbm",
        "--device", "cpu",
        "--data-dir", (Join-Path $CpuSmokeRoot "data"),
        "--outputs-dir", (Join-Path $CpuSmokeRoot "outputs"),
        "--reports-dir", (Join-Path $CpuSmokeRoot "reports"),
        "--tracking-uri", (Join-Path $CpuSmokeRoot "mlruns")
    ) (Join-Path $LogsRoot "14_cpu_smoke.log")

    # ------------------------------------------------------------------------------------------------------------------
    # GPU smoke
    # ------------------------------------------------------------------------------------------------------------------
    if ($Mode -in @("Smoke", "All", "Full")) {
        Write-Stage "STEP 9 - Real-data GPU smoke: Transformer fine-tuning and CrossEncoder inference"
        New-Item -ItemType Directory -Force -Path $GpuSmokeRoot | Out-Null
        Invoke-LoggedCommand "GPU smoke pipeline" $VenvPython @(
            "scripts\run_full_pipeline_with_semantics.py",
            "--cfpb-path", $DatasetPath,
            "--sample-size", "$GpuSmokeSampleSize",
            "--sampling-strategy", "auto",
            "--archive-cache-dir", $ArchiveCache,
            "--min-target-count", "20",
            "--enable-lightgbm",
            "--enable-transformer",
            "--transformer-model", "distilbert-base-uncased",
            "--transformer-epochs", "$SmokeTransformerEpochs",
            "--transformer-batch-size", "$TransformerBatchSize",
            "--enable-cross-encoder",
            "--cross-encoder-model", "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "--cross-encoder-batch-size", "$CrossEncoderBatchSize",
            "--device", "cuda",
            "--data-dir", (Join-Path $GpuSmokeRoot "data"),
            "--outputs-dir", (Join-Path $GpuSmokeRoot "outputs"),
            "--reports-dir", (Join-Path $GpuSmokeRoot "reports"),
            "--tracking-uri", (Join-Path $GpuSmokeRoot "mlruns")
        ) (Join-Path $LogsRoot "15_gpu_smoke.log")

        Assert-GpuRunArtifacts (Join-Path $GpuSmokeRoot "outputs")
    }

    # ------------------------------------------------------------------------------------------------------------------
    # Full root run
    # ------------------------------------------------------------------------------------------------------------------
    if ($Mode -in @("Full", "All")) {
        Write-Stage "STEP 10 - Archive previous root artifacts before the full run"
        foreach ($name in @("outputs", "reports", "mlruns")) {
            Backup-PathIfPresent (Join-Path $RepoPath $name) $FullRunArchiveRoot
        }
        foreach ($name in @("processed", "golden")) {
            Backup-PathIfPresent (Join-Path $RepoPath "data\$name") (Join-Path $FullRunArchiveRoot "data")
        }

        Write-Stage "STEP 11 - Full real-data GPU end-to-end run"
        Invoke-LoggedCommand "Full GPU pipeline" $VenvPython @(
            "scripts\run_full_pipeline_with_semantics.py",
            "--cfpb-path", $DatasetPath,
            "--sample-size", "$FullSampleSize",
            "--sampling-strategy", "auto",
            "--archive-cache-dir", $ArchiveCache,
            "--min-target-count", "30",
            "--enable-lightgbm",
            "--enable-transformer",
            "--transformer-model", "distilbert-base-uncased",
            "--transformer-epochs", "$FullTransformerEpochs",
            "--transformer-batch-size", "$TransformerBatchSize",
            "--enable-cross-encoder",
            "--cross-encoder-model", "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "--cross-encoder-batch-size", "$CrossEncoderBatchSize",
            "--device", "cuda",
            "--data-dir", (Join-Path $RepoPath "data"),
            "--outputs-dir", (Join-Path $RepoPath "outputs"),
            "--reports-dir", (Join-Path $RepoPath "reports"),
            "--tracking-uri", (Join-Path $RepoPath "mlruns")
        ) (Join-Path $LogsRoot "16_full_gpu_pipeline.log")

        Assert-GpuRunArtifacts (Join-Path $RepoPath "outputs")
    }

    # ------------------------------------------------------------------------------------------------------------------
    # Monte Carlo sensitivity validation
    # ------------------------------------------------------------------------------------------------------------------
    if (-not $SkipMonteCarlo) {
        Write-Stage "STEP 12 - Monte Carlo nominal/moderate/severe sensitivity validation"
        Invoke-LoggedCommand "Monte Carlo validation" $VenvPython @(
            "scripts\run_monte_carlo.py",
            "--runs-per-scenario", "$MonteCarloRunsPerScenario",
            "--sample-size", "800",
            "--base-seed", "20260612",
            "--scenarios", "nominal", "moderate", "severe",
            "--output-root", (Join-Path $RepoPath "monte_carlo"),
            "--enable-lightgbm",
            "--jobs", "$MonteCarloJobs"
        ) (Join-Path $LogsRoot "17_monte_carlo.log")
    }

    # ------------------------------------------------------------------------------------------------------------------
    # Final result summary
    # ------------------------------------------------------------------------------------------------------------------
    Write-Stage "STEP 13 - Final artifact and result summary"
    $SummaryOutputs = if ($Mode -in @("Full", "All")) {
        Join-Path $RepoPath "outputs"
    }
    else {
        Join-Path $GpuSmokeRoot "outputs"
    }

    $SummaryReports = if ($Mode -in @("Full", "All")) {
        Join-Path $RepoPath "reports"
    }
    else {
        Join-Path $GpuSmokeRoot "reports"
    }

    $gate = Get-Content (Join-Path $SummaryOutputs "launch_gate_result.json") -Raw | ConvertFrom-Json
    $manifest = Get-Content (Join-Path $SummaryOutputs "pipeline_manifest.json") -Raw | ConvertFrom-Json
    $evidence = Get-Content (Join-Path $SummaryOutputs "evidence_quality_summary.json") -Raw | ConvertFrom-Json
    $leaderboard = Import-Csv (Join-Path $SummaryOutputs "model_test_leaderboard.csv")

    Write-Host "Launch gate: $((Get-LaunchDecision -GateObject $gate))" -ForegroundColor Magenta
    Write-Host "Selection winner: $((Get-SelectionWinner -InputObject $manifest))" -ForegroundColor Magenta
    Write-Host "Runtime seconds: $($manifest.runtime_seconds)" -ForegroundColor Magenta
    Write-Host "CrossEncoder effective: $($evidence.used_cross_encoder)" -ForegroundColor Magenta
    Write-Host "Outputs: $SummaryOutputs" -ForegroundColor Magenta
    Write-Host "Reports: $SummaryReports" -ForegroundColor Magenta
    Write-Host "Master transcript: $(Join-Path $LogsRoot 'master_transcript.log')" -ForegroundColor Magenta

    Write-Host ""
    Write-Host "Model test leaderboard:" -ForegroundColor Cyan
    $leaderboard | Format-Table model, target_col, macro_f1, pr_auc, ece, brier, log_loss, worst_slice_f1, device -AutoSize

    if ($LaunchUIs) {
        Write-Stage "STEP 14 - Start MLflow and Streamlit UIs"
        if (Test-Path $VenvMlflow) {
            Start-Process -FilePath $VenvMlflow -ArgumentList @("ui", "--backend-store-uri", (Join-Path $RepoPath "mlruns"), "--host", "127.0.0.1", "--port", "5000") -WorkingDirectory $RepoPath
            Start-Sleep -Seconds 2
            Start-Process "http://127.0.0.1:5000"
        }
        if (Test-Path $VenvStreamlit) {
            Start-Process -FilePath $VenvStreamlit -ArgumentList @("run", "dashboard\app.py", "--server.address", "127.0.0.1", "--server.port", "8501") -WorkingDirectory $RepoPath
            Start-Sleep -Seconds 2
            Start-Process "http://127.0.0.1:8501"
        }
    }

    Write-Host ""
    Write-Host "ALL REQUESTED STEPS COMPLETED." -ForegroundColor Green
    Write-Host "A PASS/REVIEW/BLOCK launch decision is a model-quality result, not a script success/failure code." -ForegroundColor Green
}
finally {
    try {
        Stop-Transcript | Out-Null
    }
    catch {
        # Transcript may not have started if a very early error occurred.
    }
}
