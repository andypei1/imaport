Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-Python {
    $candidates = @(
        @{ cmd = "py"; args = @("-3.12") },
        @{ cmd = "py"; args = @() },
        @{ cmd = "python"; args = @() }
    )

    foreach ($c in $candidates) {
        try {
            $null = & $c.cmd @($c.args + @("-c", "import sys; print(sys.version)")) 2>$null
            if ($LASTEXITCODE -eq 0) {
                return $c
            }
        } catch {
        }
    }
    throw "No working Python interpreter found. Install Python and ensure 'py' or 'python' is available."
}

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$requirements = Join-Path $root "requirements.txt"

if (-not (Test-Path $requirements)) {
    throw "requirements.txt not found at $requirements"
}

$py = Resolve-Python
Write-Host "Using Python: $($py.cmd) $($py.args -join ' ')" -ForegroundColor Cyan

& $py.cmd @($py.args + @("-m", "pip", "install", "--user", "-r", $requirements))
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

Write-Host "Setup complete. Run health check with: .\scripts\doctor.ps1" -ForegroundColor Green
