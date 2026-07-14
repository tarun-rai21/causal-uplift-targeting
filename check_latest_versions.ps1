$packages = @(
    "python-dotenv", "pandas", "numpy", "scikit-learn", "scipy",
    "SQLAlchemy", "pydantic", "PyYAML", "streamlit", "plotly",
    "fastapi", "uvicorn", "pytest", "ruff", "causalml"
)

Write-Host ("{0,-15} {1,-12} {2,-20} {3}" -f "package", "latest", "requires_python", "has_wheel") -ForegroundColor Cyan
Write-Host ("-" * 65)

foreach ($pkg in $packages) {
    try {
        $url = "https://pypi.org/pypi/$pkg/json"
        $data = Invoke-RestMethod -Uri $url -TimeoutSec 15

        $latest = $data.info.version
        $requiresPython = $data.info.requires_python
        if (-not $requiresPython) { $requiresPython = "none" }

        $releaseFiles = $data.releases.$latest
        $hasCp312Wheel = $releaseFiles | Where-Object {
            $_.packagetype -eq "bdist_wheel" -and $_.filename -match "cp312|py3-none|abi3"
        }
        $hasWheelStr = if ($hasCp312Wheel) { "True" } else { "False (check manually)" }

        Write-Host ("{0,-15} {1,-12} {2,-20} {3}" -f $pkg, $latest, $requiresPython, $hasWheelStr)
    }
    catch {
        Write-Host ("{0,-15} ERROR: {1}" -f $pkg, $_.Exception.Message) -ForegroundColor Red
    }
}