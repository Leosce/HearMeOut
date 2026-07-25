function Initialize-AvHubertEnvironment {
    param([Parameter(Mandatory = $true)][string]$Root)

    $toolPaths = @(
        (Join-Path $Root "tools"),
        (Join-Path $Root "tools\ffmpeg-8.1-essentials_build\bin")
    ) | Where-Object { Test-Path -LiteralPath $_ }

    if ($toolPaths.Count -gt 0) {
        $env:Path = (($toolPaths -join ";") + ";$env:Path")
    }
    if (-not $env:AVHUBERT_ROOT) {
        $env:AVHUBERT_ROOT = Join-Path $Root "av_hubert"
    }
}

function Resolve-PythonCandidate {
    param([string]$Candidate)

    if (-not $Candidate) {
        return $null
    }
    if (Test-Path -LiteralPath $Candidate) {
        return (Resolve-Path -LiteralPath $Candidate).Path
    }
    $cmd = Get-Command $Candidate -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    return $null
}

function Get-AvHubertPythonCandidates {
    param([Parameter(Mandatory = $true)][string]$Root)

    $condaPython = if ($env:CONDA_PREFIX) { Join-Path $env:CONDA_PREFIX "python.exe" } else { $null }
    $userProfile = [Environment]::GetFolderPath("UserProfile")

    $candidates = @(
        $condaPython,
        (Join-Path $Root ".venv\Scripts\python.exe"),
        (Join-Path $Root "venv\Scripts\python.exe"),
        (Join-Path $userProfile "anaconda3\envs\avhubert\python.exe"),
        (Join-Path $userProfile "miniconda3\envs\avhubert\python.exe"),
        "C:\ProgramData\anaconda3\envs\avhubert\python.exe",
        "C:\ProgramData\miniconda3\envs\avhubert\python.exe",
        "D:\Anaconda\envs\avhubert\python.exe",
        "E:\Anaconda\envs\avhubert\python.exe",
        "D:\Miniconda3\envs\avhubert\python.exe",
        "E:\Miniconda3\envs\avhubert\python.exe",
        "python.exe",
        "python"
    )

    return $candidates | Where-Object { $_ } | Select-Object -Unique
}

function Test-PythonCandidate {
    param(
        [string]$Candidate,
        [string[]]$RequiredModules = @("torch", "cv2", "numpy")
    )

    $resolved = Resolve-PythonCandidate $Candidate
    if (-not $resolved) {
        return $null
    }

    $moduleLiteral = ($RequiredModules | ForEach-Object { "'" + ($_ -replace "'", "\\'") + "'" }) -join ","
    $code = @"
import importlib
import sys
missing = []
for name in [$moduleLiteral]:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(name + ': ' + str(exc))
if missing:
    print('\n'.join(missing))
    sys.exit(1)
"@
    & $resolved -c $code *> $null
    if ($LASTEXITCODE -eq 0) {
        return $resolved
    }
    return $null
}

function Resolve-AvHubertPython {
    param(
        [string]$Python,
        [Parameter(Mandatory = $true)][string]$Root,
        [string[]]$RequiredModules = @("torch", "cv2", "numpy")
    )

    if ($Python) {
        $py = Test-PythonCandidate -Candidate $Python -RequiredModules $RequiredModules
        if (-not $py) {
            throw "The Python passed with -Python was not usable for AV-HuBERT: $Python"
        }
        return $py
    }

    foreach ($candidate in (Get-AvHubertPythonCandidates -Root $Root)) {
        $py = Test-PythonCandidate -Candidate $candidate -RequiredModules $RequiredModules
        if ($py) {
            return $py
        }
    }

    throw "No usable Python was found. Activate the avhubert conda environment or pass -Python C:\path\to\env\python.exe"
}

function Resolve-InputVideoPath {
    param([Parameter(Mandatory = $true)][string]$VideoPath)

    if (-not (Test-Path -LiteralPath $VideoPath -PathType Leaf)) {
        throw "Video file was not found: $VideoPath"
    }
    return (Resolve-Path -LiteralPath $VideoPath).Path
}
