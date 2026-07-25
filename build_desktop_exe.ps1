param(
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Source = Join-Path $Root "desktop_launcher\HearMeOutDesktop.cs"
$Icon = Join-Path $Root "desktop_launcher\HearMeOut.ico"
if (-not $OutputPath) {
    $OutputPath = Join-Path $Root "HearMeOutDesktop.exe"
}

$CompilerCandidates = @(
    "C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe",
    "C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe"
)

$Compiler = $CompilerCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $Compiler) {
    throw "Could not find the .NET Framework C# compiler."
}

Write-Host "[build] compiler: $Compiler"
Write-Host "[build] source:   $Source"
Write-Host "[build] output:   $OutputPath"
if (Test-Path -LiteralPath $Icon) {
    Write-Host "[build] icon:     $Icon"
}

$CompilerArgs = @(
    "/nologo",
    "/target:winexe",
    "/platform:anycpu",
    "/optimize+",
    "/reference:System.Windows.Forms.dll",
    "/out:$OutputPath"
)
if (Test-Path -LiteralPath $Icon) {
    $CompilerArgs += "/win32icon:$Icon"
}
$CompilerArgs += $Source

& $Compiler @CompilerArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "[build] created:  $OutputPath"
