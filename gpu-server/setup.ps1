#Requires -Version 5.1
<#
.SYNOPSIS
    GPU Server setup script — installs ML dependencies and Ollama.

.DESCRIPTION
    Run once on the GPU VM before starting server.py for the first time.
    Re-running is safe — already-installed components are skipped.
    PyTorch is NOT touched — install the correct CUDA build separately.

.NOTES
    Run from the gpu-server directory:
        cd gpu-server
        .\setup.ps1

    To skip Ollama install:   .\setup.ps1 -SkipOllama
    To skip model pull:       .\setup.ps1 -SkipModelPull
#>

param(
    [switch]$SkipOllama,
    [switch]$SkipModelPull,
    [string]$PythonExe = ""          # Override Python executable path
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Helpers ───────────────────────────────────────────────────────────────────

function Write-Header([string]$Text) {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
}

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host "[>] $Text" -ForegroundColor Yellow
}

function Write-Ok([string]$Text) {
    Write-Host "[OK] $Text" -ForegroundColor Green
}

function Write-Warn([string]$Text) {
    Write-Host "[!!] $Text" -ForegroundColor Magenta
}

function Write-Fail([string]$Text) {
    Write-Host "[FAIL] $Text" -ForegroundColor Red
}

# ── Banner ────────────────────────────────────────────────────────────────────

Write-Header "AI Trader — GPU Server Setup"
Write-Host "  This script installs:"
Write-Host "    - ML dependencies (FastAPI, sentence-transformers, etc.)"
Write-Host "    - Ollama (local LLM runtime)"
Write-Host "    - Qwen 2.5 LLM model (14B on GPU / 7B on CPU)"
Write-Host ""
Write-Host "  NOTE: PyTorch is NOT installed by this script."
Write-Host "        The existing installation will be used as-is."
Write-Host ""
Write-Host "  Estimated time: 2–5 min (plus ~8 GB model download)"
Write-Host ""

# ── Step 1 — Locate Python ────────────────────────────────────────────────────

Write-Header "Step 1 — Python"

if ($PythonExe -eq "") {
    foreach ($candidate in @("python", "python3", "py")) {
        try {
            $ver = & $candidate --version 2>&1
            if ($ver -match "Python (\d+)\.(\d+)") {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]
                if ($major -ge 3 -and $minor -ge 10) {
                    $PythonExe = $candidate
                    break
                } else {
                    Write-Warn "$candidate found but version $major.$minor < 3.10 — skipping"
                }
            }
        } catch { }
    }
}

if ($PythonExe -eq "") {
    Write-Fail "Python 3.10+ not found."
    Write-Host ""
    Write-Host "  Install Python 3.11 via winget:"
    Write-Host "    winget install Python.Python.3.11"
    Write-Host "  Then re-run this script."
    exit 1
}

$pyVersion = & $PythonExe --version 2>&1
Write-Ok "Using: $PythonExe  ($pyVersion)"

# ── Step 2 — Virtual Environment ─────────────────────────────────────────────

Write-Header "Step 2 — Virtual Environment"

$VenvDir = Join-Path $PSScriptRoot "venv"
$VenvPy  = Join-Path $VenvDir "Scripts\python.exe"

if (Test-Path $VenvPy) {
    Write-Ok "venv already exists at $VenvDir"
} else {
    Write-Step "Creating venv at $VenvDir"
    & $PythonExe -m venv $VenvDir --system-site-packages
    Write-Ok "venv created (--system-site-packages so existing torch is visible)"
}

Write-Step "Upgrading pip"
& $VenvPy -m pip install --upgrade pip --quiet
Write-Ok "pip up to date"

# ── Step 3 — ML Dependencies ─────────────────────────────────────────────────

Write-Header "Step 3 — ML Dependencies"

$RequirementsFile = Join-Path $PSScriptRoot "requirements.txt"
if (-not (Test-Path $RequirementsFile)) {
    Write-Fail "requirements.txt not found at $RequirementsFile"
    exit 1
}

Write-Step "Installing requirements.txt (skipping torch — not touched)"
$reqs = Get-Content $RequirementsFile |
    Where-Object { $_ -match '\S' -and $_ -notmatch '^\s*#' -and $_ -notmatch '^torch' }
foreach ($req in $reqs) {
    Write-Step "  pip install $req"
    & $VenvPy -m pip install $req --quiet
}
Write-Ok "All ML dependencies installed"

# ── Step 4 — Verify Imports ───────────────────────────────────────────────────

Write-Header "Step 4 — Verify Imports"

$imports = @(
    @{ pkg="torch";                   label="PyTorch"               },
    @{ pkg="fastapi";                 label="FastAPI"               },
    @{ pkg="uvicorn";                 label="Uvicorn"               },
    @{ pkg="numpy";                   label="NumPy"                 },
    @{ pkg="pydantic";                label="Pydantic"              },
    @{ pkg="sentence_transformers";   label="Sentence-Transformers" }
)

$allOk  = $true
$HasGpu = $false
foreach ($imp in $imports) {
    $result = & $VenvPy -c "import $($imp.pkg); print('ok')" 2>&1
    if ($result -eq "ok") {
        Write-Ok "$($imp.label)"
    } else {
        Write-Fail "$($imp.label) — $result"
        $allOk = $false
    }
}

if (-not $allOk) {
    Write-Fail "Some imports failed — check the errors above."
    exit 1
}

# Detect GPU from the existing torch installation
$torchInfo = & $VenvPy -c @"
import torch
cuda = torch.cuda.is_available()
if cuda:
    p = torch.cuda.get_device_properties(0)
    vram = round((getattr(p,'total_memory',None) or getattr(p,'total_mem',0))/1e9, 1)
    print(f'cuda=True gpu={torch.cuda.get_device_name(0)} vram={vram}GB ver={torch.__version__}')
else:
    print(f'cuda=False ver={torch.__version__}')
"@ 2>&1

if ($torchInfo -match "cuda=True") {
    $HasGpu = $true
    Write-Ok "PyTorch: $torchInfo"
} else {
    Write-Warn "PyTorch: $torchInfo (CPU-only — Ollama will use smaller model)"
}

# ── Step 5 — Ollama ───────────────────────────────────────────────────────────

if (-not $SkipOllama) {
    Write-Header "Step 5 — Ollama (Local LLM Runtime)"

    $ollamaPath = Get-Command ollama -ErrorAction SilentlyContinue
    if ($ollamaPath) {
        Write-Ok "Ollama already installed: $($ollamaPath.Source)"
    } else {
        Write-Step "Installing Ollama via winget"
        try {
            winget install Ollama.Ollama --accept-package-agreements --accept-source-agreements
            Write-Ok "Ollama installed"
            # Refresh PATH so 'ollama' is found in this session
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                        [System.Environment]::GetEnvironmentVariable("Path", "User")
        } catch {
            Write-Warn "winget install failed: $_"
            Write-Warn "Install Ollama manually, then re-run this script."
            Write-Warn "Download from: https://ollama.com/download"
            $SkipModelPull = $true
        }
    }

    # ── Step 6 — Model Pull ───────────────────────────────────────────────────

    if (-not $SkipModelPull) {
        Write-Header "Step 6 — LLM Model Pull"

        $OllamaModel = if ($HasGpu) { "qwen2.5:14b" } else { "qwen2.5:7b" }
        if ($env:LOCAL_LLM_MODEL) { $OllamaModel = $env:LOCAL_LLM_MODEL }

        Write-Step "Checking whether '$OllamaModel' is already pulled"

        $serveProc    = $null
        $startedServe = $false
        try {
            $null = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3
            Write-Ok "Ollama serve already running"
        } catch {
            Write-Step "Starting temporary 'ollama serve' to check model list"
            $serveProc    = Start-Process ollama -ArgumentList "serve" -PassThru -WindowStyle Hidden
            $startedServe = $true
            $ready        = $false
            for ($i = 0; $i -lt 15; $i++) {
                Start-Sleep 1
                try {
                    $null = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2
                    $ready = $true
                    break
                } catch { }
            }
            if (-not $ready) {
                Write-Warn "Ollama serve did not start in time — skipping model pull"
                if ($serveProc) { Stop-Process -Id $serveProc.Id -Force -ErrorAction SilentlyContinue }
                $serveProc = $null
            }
        }

        if ($serveProc -ne $null -or -not $startedServe) {
            try {
                $tags       = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags"
                $modelBase  = $OllamaModel.Split(":")[0]
                $alreadyHave = $tags.models | Where-Object { $_.name -like "$modelBase*" }

                if ($alreadyHave) {
                    Write-Ok "Model '$OllamaModel' already present — skipping download"
                } else {
                    $sizeHint = if ($OllamaModel -like "*14b*") { "~8.7 GB" } else { "~4.7 GB" }
                    Write-Step "Pulling '$OllamaModel' ($sizeHint) — this may take a while…"
                    & ollama pull $OllamaModel
                    Write-Ok "Model '$OllamaModel' ready"
                }
            } finally {
                if ($startedServe -and $serveProc) {
                    Stop-Process -Id $serveProc.Id -Force -ErrorAction SilentlyContinue
                }
            }
        }
    } else {
        Write-Warn "Skipping model pull (-SkipModelPull). Run manually: ollama pull qwen2.5:14b"
    }
} else {
    Write-Warn "Skipping Ollama install (-SkipOllama)"
}

# ── Step 7 — Generate start.ps1 ───────────────────────────────────────────────

Write-Header "Step 7 — Generate start.ps1"

$StartScript  = Join-Path $PSScriptRoot "start.ps1"
$StartContent = @"
# Auto-generated by setup.ps1 — run this to start the GPU server
Set-Location `"`$PSScriptRoot`"
Write-Host "Starting AI Trader GPU Server..." -ForegroundColor Cyan
& "`$PSScriptRoot\venv\Scripts\python.exe" server.py
"@
Set-Content -Path $StartScript -Value $StartContent -Encoding UTF8
Write-Ok "Created start.ps1"

# ── Summary ───────────────────────────────────────────────────────────────────

Write-Header "Setup Complete"
Write-Host ""
Write-Host "  To start the GPU server:" -ForegroundColor White
Write-Host "    .\start.ps1" -ForegroundColor Green
Write-Host ""
Write-Host "  Or directly:" -ForegroundColor White
Write-Host "    .\venv\Scripts\python.exe server.py" -ForegroundColor Green
Write-Host ""

if ($HasGpu) {
    Write-Host "  GPU detected — Ollama will use qwen2.5:14b on GPU." -ForegroundColor Green
} else {
    Write-Host "  No GPU — Ollama will use qwen2.5:7b on CPU (slower)." -ForegroundColor Magenta
    Write-Host "  Consider setting LOCAL_LLM_TIMEOUT=180 in the bot's .env." -ForegroundColor Magenta
}

Write-Host ""
Write-Host "  Bot .env settings:" -ForegroundColor White
Write-Host "    GPU_SERVER_URL=http://<this-machine-ip>:9090" -ForegroundColor DarkCyan
Write-Host "    USE_LOCAL_LLM=true" -ForegroundColor DarkCyan
Write-Host "    # LOCAL_LLM_URL is auto-derived from GPU_SERVER_URL (no change needed)" -ForegroundColor DarkGray
Write-Host ""
