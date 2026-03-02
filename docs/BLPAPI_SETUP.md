# Bloomberg `blpapi` Setup (Windows)

Use this when `scripts/doctor.ps1` shows:

- `[WARN] import blpapi: No module named 'blpapi'`

## Recommended install command

Use Bloomberg's Python package index first:

```powershell
py -3.12 -m pip install --user --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
```

## Why automated download can still fail

The Bloomberg API Library site uses anti-bot/captcha checks, so command-line download tools (`pip`, `Invoke-WebRequest`, `curl`) may be blocked even when the URL is correct.

## Manual install (reliable)

1. Open this page in a normal browser session:
   - https://www.bloomberg.com/professional/support/api-library/
2. Download the Python `blpapi` package for Windows matching your interpreter:
   - For this repo default: Python 3.12, 64-bit (`cp312`, `win_amd64`)
3. Save the wheel file (for example in `Downloads`), then install:

```powershell
py -3.12 -m pip install --user "C:\Users\<you>\Downloads\blpapi-<version>-cp312-...-win_amd64.whl"
```

4. Verify:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\doctor.ps1
```

Expected result:

- `[PASS] import blpapi (...)`

## Repo automation behavior

`scripts/setup.ps1` always tries to install `blpapi` if missing:

1. `pip install --user --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi`
2. fallback: `pip install --user blpapi`
3. If that fails, tries a local wheel from:
   - `%USERPROFILE%\Downloads`
   - `.\vendor`
   - `.\inputs`

If all attempts fail, it prints this doc path for manual completion.
