<#
═══════════════════════════════════════════════════════════════════════════════
 build_windows_app.ps1  —  assemble a no-install, shareable NVH Analyzer for Windows
═══════════════════════════════════════════════════════════════════════════════

 WHAT THIS PRODUCES
   A self-contained folder (and optional .zip) that bundles:
     • the official Python "embeddable" runtime (no system Python needed),
     • every dependency (pandas/numpy/scipy/matplotlib/pyarrow/streamlit) baked in,
     • the NVH pipeline code + the Streamlit UI, and
     • a double-click "Run NVH Analyzer.bat".
   The recipient unzips it anywhere (Desktop, USB stick, network share) and
   double-clicks the .bat. No admin rights, no installer, no internet.

 WHERE TO RUN THIS
   On ANY Windows machine WITH internet, once (it downloads the runtime + wheels).
   The resulting dist\ folder is what you share. PowerShell 5+ is enough:

       powershell -ExecutionPolicy Bypass -File packaging\build_windows_app.ps1

   Options:
       -PyVersion 3.11.9     which CPython embeddable to bundle (default 3.11.9)
       -OutDir    dist       where to assemble (default: <repo>\dist)
       -Zip                  also produce NVH-Analyzer.zip next to the folder
#>

[CmdletBinding()]
param(
    [string]$PyVersion = "3.11.9",
    [string]$OutDir    = "",
    [switch]$Zip
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# ── Locations ────────────────────────────────────────────────────────────────
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $RepoRoot "dist" }
$AppName    = "NVH-Analyzer"
$BundleRoot = Join-Path $OutDir $AppName
$RuntimeDir = Join-Path $BundleRoot "runtime"
$AppDir     = Join-Path $BundleRoot "app"

Write-Host "─────────────────────────────────────────────────────────────"
Write-Host " Building $AppName  (Python $PyVersion embeddable)"
Write-Host " Repo : $RepoRoot"
Write-Host " Out  : $BundleRoot"
Write-Host "─────────────────────────────────────────────────────────────"

# Fresh bundle
if (Test-Path $BundleRoot) { Remove-Item $BundleRoot -Recurse -Force }
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
New-Item -ItemType Directory -Force -Path $AppDir     | Out-Null

# ── 1 · Download + extract the embeddable Python runtime ────────────────────--
$embedUrl = "https://www.python.org/ftp/python/$PyVersion/python-$PyVersion-embed-amd64.zip"
$embedZip = Join-Path $env:TEMP "python-$PyVersion-embed-amd64.zip"
Write-Host "[1/6] Downloading embeddable runtime..."
Write-Host "      $embedUrl"
Invoke-WebRequest -Uri $embedUrl -OutFile $embedZip
Expand-Archive -Path $embedZip -DestinationPath $RuntimeDir -Force

# ── 2 · Enable site-packages in the embeddable ._pth ─────────────────────────
Write-Host "[2/6] Enabling site-packages..."
$pth = Get-ChildItem -Path $RuntimeDir -Filter "python*._pth" | Select-Object -First 1
if (-not $pth) { throw "Could not find python*._pth in the embeddable runtime." }
@"
$($pth.Name.Replace('._pth','.zip'))
.
Lib\site-packages
import site
"@ | Set-Content -Encoding ASCII -Path $pth.FullName

# ── 3 · Bootstrap pip into the runtime ───────────────────────────────────────
Write-Host "[3/6] Bootstrapping pip..."
$getpip = Join-Path $env:TEMP "get-pip.py"
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getpip
$pyExe = Join-Path $RuntimeDir "python.exe"
& $pyExe $getpip --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw "get-pip.py failed (exit $LASTEXITCODE)." }

# ── 4 · Install all dependencies into the runtime ────────────────────────────
Write-Host "[4/6] Installing dependencies (this is the slow part)..."
$reqs = Join-Path $RepoRoot "requirements-app.txt"
# --only-binary=:all: forces wheels only. The embeddable runtime has no compiler,
# so any source build (sdist) would fail; failing fast here with a clear pip error
# beats a cryptic mid-build C-compiler error. Every pinned dep ships a
# cp311 win_amd64 wheel for the default -PyVersion 3.11.9.
& $pyExe -m pip install --no-warn-script-location --only-binary=:all: -r $reqs
if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)." }

# ── 5 · Copy the application code + config ────────────────────────────────────
Write-Host "[5/6] Copying application files..."
Copy-Item (Join-Path $RepoRoot "streamlit_app.py")        -Destination $AppDir
Copy-Item (Join-Path $RepoRoot "nvh_pipeline")            -Destination $AppDir -Recurse
Copy-Item (Join-Path $RepoRoot "nvh_config.example.json") -Destination $AppDir
Get-ChildItem -Path $RepoRoot -Filter "EP_*.py" | ForEach-Object {
    Copy-Item $_.FullName -Destination $AppDir
}
# Streamlit config (telemetry off, localhost only) goes at the bundle root,
# because the launcher runs Streamlit with the bundle root as the working dir.
Copy-Item (Join-Path $RepoRoot ".streamlit") -Destination $BundleRoot -Recurse

# The double-click launcher.
Copy-Item (Join-Path $PSScriptRoot "Run NVH Analyzer.bat") -Destination $BundleRoot

# ── 6 · (optional) zip it for sharing ────────────────────────────────────────
if ($Zip) {
    Write-Host "[6/6] Zipping..."
    $zipPath = Join-Path $OutDir "$AppName.zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Compress-Archive -Path $BundleRoot -DestinationPath $zipPath
    Write-Host "      $zipPath"
} else {
    Write-Host "[6/6] Skipping zip (pass -Zip to produce $AppName.zip)."
}

Write-Host ""
Write-Host "✓ Done."
Write-Host "  Share the folder:  $BundleRoot"
Write-Host "  Recipient double-clicks:  Run NVH Analyzer.bat"
