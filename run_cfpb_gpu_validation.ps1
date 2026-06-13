param(
    [ValidateSet('Smoke','Full')]
    [string]$Mode = 'Smoke',
    [switch]$Install,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$RepoRoot = '${PSScriptRoot}'
$DatasetPath = '${env:USERPROFILE}\Downloads\Project_Data\CFPB Complaint Data\CFPB complaints 20200518-20260518 complaints only with narratives.csv'
$VenvPath = Join-Path $RepoRoot '.venv'
$PythonExe = Join-Path $VenvPath 'Scripts\python.exe'
$ActivateScript = Join-Path $VenvPath 'Scripts\Activate.ps1'
$RunRoot = Join-Path $RepoRoot ('gpu_runs\' + $Mode.ToLower())
$OutputsDir = Join-Path $RunRoot 'outputs'
$ReportsDir = Join-Path $RunRoot 'reports'
$LogsDir = Join-Path $RunRoot 'logs'
$GpuInfoPath = Join-Path $LogsDir 'gpu_preflight.json'
$PipelineLog = Join-Path $LogsDir 'pipeline.log'

if (-not (Test-Path $RepoRoot)) {
    throw "Repository folder not found: $RepoRoot"
}
if (-not (Test-Path $DatasetPath)) {
    throw "CFPB dataset not found: $DatasetPath"
}

New-Item -ItemType Directory -Force -Path $RunRoot, $OutputsDir, $ReportsDir, $LogsDir | Out-Null
Set-Location $RepoRoot

if (-not (Test-Path $PythonExe)) {
    Write-Host '[setup] Creating Python 3.11 virtual environment...'
    py -3.11 -m venv $VenvPath
}

if (-not (Test-Path $ActivateScript)) {
    throw "Virtual environment activation script not found: $ActivateScript"
}
. $ActivateScript

if ($Install) {
    Write-Host '[setup] Updating pip and installing project dependencies...'
    & $PythonExe -m pip install --upgrade pip
    & $PythonExe -m pip install -r requirements.txt

    Write-Host '[setup] Installing the official CUDA 12.8 PyTorch wheels...'
    & $PythonExe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

    Write-Host '[setup] Installing Transformer and CrossEncoder dependencies...'
    & $PythonExe -m pip install -r requirements-gpu.txt
    & $PythonExe -m pip install -e . --no-deps
}

Write-Host '[check] Python:' (& $PythonExe --version)
Write-Host '[check] Dataset:' $DatasetPath
Write-Host '[check] Run output:' $RunRoot

try {
    nvidia-smi | Tee-Object -FilePath (Join-Path $LogsDir 'nvidia-smi.txt')
}
catch {
    throw 'nvidia-smi failed. Confirm that the NVIDIA driver is installed and the GPU is visible.'
}

Write-Host '[check] Running strict CUDA preflight...'
& $PythonExe scripts\gpu_preflight.py --require-cuda --output $GpuInfoPath
if ($LASTEXITCODE -ne 0) {
    throw "CUDA preflight failed. Review: $GpuInfoPath"
}

if (-not $SkipTests) {
    Write-Host '[check] Running unit tests...'
    & $PythonExe -m pytest -q
    if ($LASTEXITCODE -ne 0) {
        throw 'Unit tests failed.'
    }
}

if ($Mode -eq 'Smoke') {
    $SampleSize = 5000
    $Epochs = 1
    $TransformerBatch = 32
    $CrossEncoderBatch = 64
}
else {
    $SampleSize = 20000
    $Epochs = 2
    $TransformerBatch = 32
    $CrossEncoderBatch = 128
}

# Keep CPU-only preprocessing from oversubscribing laptop CPU cores.
$env:MODEL_QUALITY_CPU_THREADS = '8'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'

Write-Host "[run] Starting $Mode GPU validation: sample=$SampleSize, epochs=$Epochs"

$PipelineArgs = @(
    'scripts\run_full_pipeline.py',
    '--cfpb-path', $DatasetPath,
    '--sample-size', $SampleSize,
    '--sampling-strategy', 'auto',
    '--random-state', '20260612',
    '--enable-lightgbm',
    '--enable-transformer',
    '--transformer-model', 'distilbert-base-uncased',
    '--transformer-epochs', $Epochs,
    '--transformer-batch-size', $TransformerBatch,
    '--enable-cross-encoder',
    '--cross-encoder-model', 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    '--cross-encoder-batch-size', $CrossEncoderBatch,
    '--device', 'cuda',
    '--outputs-dir', $OutputsDir,
    '--reports-dir', $ReportsDir
)

& $PythonExe @PipelineArgs 2>&1 | Tee-Object -FilePath $PipelineLog
$PipelineExit = $LASTEXITCODE
if ($PipelineExit -ne 0) {
    throw "GPU pipeline failed with exit code $PipelineExit. Review: $PipelineLog"
}

$GatePath = Join-Path $OutputsDir 'launch_gate_result.json'
$ManifestPath = Join-Path $OutputsDir 'pipeline_manifest.json'
$EvidencePath = Join-Path $OutputsDir 'evidence_quality_summary.json'

foreach ($RequiredPath in @($GatePath, $ManifestPath, $EvidencePath)) {
    if (-not (Test-Path $RequiredPath)) {
        throw "Required output missing: $RequiredPath"
    }
}

$Gate = Get-Content $GatePath -Raw | ConvertFrom-Json
$Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
$Evidence = Get-Content $EvidencePath -Raw | ConvertFrom-Json

Write-Host ''
Write-Host '========== GPU VALIDATION SUMMARY =========='
Write-Host 'Mode:' $Mode
Write-Host 'Dataset rows requested:' $SampleSize
Write-Host 'Launch decision:' $Gate.status
Write-Host 'Champion:' $Manifest.champion_model
Write-Host 'Pipeline runtime seconds:' $Manifest.runtime_seconds
Write-Host 'CrossEncoder requested:' $Evidence.cross_encoder_requested
Write-Host 'CrossEncoder effective:' $Evidence.used_cross_encoder
Write-Host 'BM25 Recall@5:' $Evidence.bm25_recall_at_5
Write-Host 'Final Recall@5:' $Evidence.recall_at_5
Write-Host 'Rerank Recall uplift:' $Evidence.rerank_recall_uplift
Write-Host 'GPU preflight:' $GpuInfoPath
Write-Host 'Pipeline log:' $PipelineLog
Write-Host 'Outputs:' $OutputsDir
Write-Host 'Reports:' $ReportsDir
Write-Host '============================================'
