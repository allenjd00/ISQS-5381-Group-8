$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonExe = $venvPython
} else {
    $pythonExe = "c:/python313/python.exe"
}

Write-Host "Using Python: $pythonExe"
Write-Host "Building recommendation scores..."
& $pythonExe "scripts/07_build_recommendation_scores.py"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Score build failed."
    exit $LASTEXITCODE
}

Write-Host "Starting Streamlit app..."
& $pythonExe -m streamlit run "app/streamlit_app.py"
