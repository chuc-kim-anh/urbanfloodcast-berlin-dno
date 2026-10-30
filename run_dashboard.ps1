$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw 'Python environment not found. Follow the README setup steps first.'
}
& $python -c "import streamlit, torch, rasterio, tifffile, folium, plotly"
if ($LASTEXITCODE -ne 0) {
    throw 'Required Python packages are missing. Follow the README setup steps first.'
}
if (-not (Test-Path -LiteralPath 'vendor\UrbanFloodCast\DNO\models\DNO.py')) {
    throw 'Upstream DNO code not found. Run setup_project.ps1 first.'
}

$env:URBANFLOODCAST_CODE = Join-Path $projectRoot 'vendor\UrbanFloodCast\DNO'
& $python -m streamlit run app_p2m.py --server.address 127.0.0.1 --server.port 8504
