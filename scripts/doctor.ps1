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
    throw "No working Python interpreter found."
}

Write-Host "== Environment Doctor ==" -ForegroundColor Cyan
$py = Resolve-Python
Write-Host "Python command: $($py.cmd) $($py.args -join ' ')" -ForegroundColor Gray

$pyCheck = @"
import importlib
import sys

print(f"python_executable={sys.executable}")
print(f"python_version={sys.version.split()[0]}")

mods = ["pandas", "matplotlib", "streamlit"]
optional = ["blpapi"]
failed = False

for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "unknown")
        print(f"[PASS] import {m} ({ver})")
    except Exception as e:
        print(f"[FAIL] import {m}: {e}")
        failed = True

for m in optional:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "unknown")
        print(f"[PASS] import {m} ({ver})")
    except Exception as e:
        print(f"[WARN] import {m}: {e}")
        print("[WARN] Bloomberg-specific features will be unavailable in this interpreter.")

raise SystemExit(1 if failed else 0)
"@

$tmpPy = New-TemporaryFile
Set-Content -Path $tmpPy.FullName -Value $pyCheck -Encoding UTF8
& $py.cmd @($py.args + @($tmpPy.FullName))
$code = $LASTEXITCODE
Remove-Item -LiteralPath $tmpPy.FullName -ErrorAction SilentlyContinue

if ($code -ne 0) {
    Write-Host "[FAIL] Core Python packages are missing." -ForegroundColor Red
    Write-Host "Run: .\scripts\setup.ps1" -ForegroundColor Yellow
    exit 1
}

try {
    $socketOpen = Test-NetConnection -ComputerName "localhost" -Port 8194 -WarningAction SilentlyContinue
    if ($socketOpen.TcpTestSucceeded) {
        Write-Host "[PASS] Bloomberg API port 8194 is reachable on localhost." -ForegroundColor Green
    } else {
        Write-Host "[WARN] Bloomberg API port 8194 is not reachable." -ForegroundColor Yellow
        Write-Host "[WARN] If Terminal is open, check API permissions/session." -ForegroundColor Yellow
    }
} catch {
    Write-Host ("[WARN] Could not test localhost:8194 ({0})." -f $_.Exception.Message) -ForegroundColor Yellow
}

Write-Host "[PASS] Doctor check complete." -ForegroundColor Green
exit 0
