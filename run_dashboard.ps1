$ErrorActionPreference = 'Stop'
$appDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $appDir
$appPython = Join-Path $appDir '.venv\Scripts\python.exe'
$runtimePython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$runtimePackages = Join-Path $appDir '.runtime_packages'
$selectedPython = $null

if (Test-Path -LiteralPath $appPython) {
    try {
        & $appPython -c "import streamlit, torch, matplotlib, scipy, tifffile" *> $null
        if ($LASTEXITCODE -eq 0) { $selectedPython = $appPython }
    } catch {
        $selectedPython = $null
    }
}

if (-not $selectedPython -and (Test-Path -LiteralPath $runtimePython) -and (Test-Path -LiteralPath $runtimePackages)) {
    $selectedPython = $runtimePython
    $env:PYTHONPATH = $runtimePackages
    $env:STREAMLIT_GLOBAL_DEVELOPMENT_MODE = 'false'
}

if (-not $selectedPython) {
    throw 'Không tìm thấy Python có đủ thư viện. Hãy cài Python, cài requirements.txt và PyTorch rồi chạy lại.'
}

& $selectedPython -m streamlit run app_p2m.py --server.address 127.0.0.1 --server.port 8504
