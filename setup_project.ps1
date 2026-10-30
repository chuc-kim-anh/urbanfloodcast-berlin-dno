$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$upstream = Join-Path $projectRoot 'vendor\UrbanFloodCast'

if (-not (Test-Path -LiteralPath $python)) {
    throw 'Create the .venv and install requirements.txt plus PyTorch first. See README.md.'
}
& $python -c "import torch, streamlit, remotezip, rasterio, tifffile"
if ($LASTEXITCODE -ne 0) {
    throw 'Python dependencies are missing. Install requirements.txt and PyTorch, then rerun this script.'
}

if (-not (Test-Path -LiteralPath (Join-Path $upstream 'DNO\models\DNO.py'))) {
    if (Test-Path -LiteralPath $upstream) {
        throw 'vendor\UrbanFloodCast exists but does not contain the upstream DNO code. Remove that folder and rerun setup.'
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $upstream) | Out-Null
    git clone --depth 1 https://github.com/HydroPML/UrbanFloodCast.git $upstream
    if ($LASTEXITCODE -ne 0) { throw 'Could not clone the upstream UrbanFloodCast repository.' }
}

& $python fetch_subset.py
if ($LASTEXITCODE -ne 0) { throw 'Benchmark download failed. Check the internet connection and rerun setup_project.ps1.' }
& $python prepare_urbanfloodcast.py
if ($LASTEXITCODE -ne 0) { throw 'Benchmark tensor conversion failed.' }
Write-Host 'Setup complete. Run run_dashboard.ps1 to start the app.'
