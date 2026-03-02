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

function Test-Import {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ModuleName,
        [Parameter(Mandatory = $true)]
        $PythonSpec
    )

    & $PythonSpec.cmd @($PythonSpec.args + @("-c", "import $ModuleName")) 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Find-BlpapiWheel {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot
    )

    $searchRoots = @(
        (Join-Path $env:USERPROFILE "Downloads"),
        (Join-Path $RepoRoot "vendor"),
        (Join-Path $RepoRoot "inputs")
    )

    $wheels = @()
    foreach ($dir in $searchRoots) {
        if (Test-Path $dir) {
            $wheels += Get-ChildItem -Path $dir -Filter "blpapi*.whl" -File -ErrorAction SilentlyContinue
        }
    }

    if ($wheels.Count -eq 0) {
        return $null
    }

    return ($wheels | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
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

if (-not (Test-Import -ModuleName "blpapi" -PythonSpec $py)) {
    Write-Host "blpapi not installed. Attempting automatic install..." -ForegroundColor Yellow

    & $py.cmd @(
        $py.args + @(
            "-m", "pip", "install", "--user",
            "--index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/",
            "blpapi"
        )
    ) 2>$null
    if ($LASTEXITCODE -ne 0) {
        & $py.cmd @($py.args + @("-m", "pip", "install", "--user", "blpapi")) 2>$null
    }

    if ($LASTEXITCODE -ne 0) {
        $wheel = Find-BlpapiWheel -RepoRoot $root
        if ($null -ne $wheel) {
            Write-Host "Trying local wheel: $($wheel.FullName)" -ForegroundColor Yellow
            & $py.cmd @($py.args + @("-m", "pip", "install", "--user", $wheel.FullName))
        }
    }

    if (-not (Test-Import -ModuleName "blpapi" -PythonSpec $py)) {
        Write-Host "[WARN] blpapi install did not succeed automatically." -ForegroundColor Yellow
        Write-Host "[WARN] Manual steps: .\docs\BLPAPI_SETUP.md" -ForegroundColor Yellow
    } else {
        Write-Host "[PASS] blpapi import succeeded." -ForegroundColor Green
    }
} else {
    Write-Host "[PASS] blpapi already installed." -ForegroundColor Green
}

Write-Host "Setup complete. Run health check with: .\scripts\doctor.ps1" -ForegroundColor Green
