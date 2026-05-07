# anima-tagger installer for Windows (x64).
#
# Usage:
#   irm https://raw.githubusercontent.com/fwaunstp/anima-tagger/main/install.ps1 | iex
#
# With arguments:
#   $script = irm https://raw.githubusercontent.com/fwaunstp/anima-tagger/main/install.ps1
#   & ([scriptblock]::Create($script)) -Version v0.2.1 -CliOnly
#
# Parameters:
#   -Version <tag>   release tag to install (default: latest)
#   -Prefix <dir>    install root (default: $env:USERPROFILE\bin)
#   -CliOnly         skip GUI install
#   -GuiOnly         skip CLI install
#   -NoVerify        skip SHA256 verification

[CmdletBinding()]
param(
    [string]$Version = "latest",
    [string]$Prefix  = (Join-Path $env:USERPROFILE "bin"),
    [switch]$CliOnly,
    [switch]$GuiOnly,
    [switch]$NoVerify
)

$ErrorActionPreference = "Stop"
$Repo = "fwaunstp/anima-tagger"

function Info($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Error $msg; exit 1 }

$arch = $env:PROCESSOR_ARCHITECTURE
if ($arch -ne "AMD64") {
    Fail "Windows on $arch is not supported by prebuilt binaries. Build from source: cargo install --git https://github.com/$Repo anima-tagger-cli"
}
$Target = "windows-x64"
Info "platform: $Target"

if ($Version -eq "latest") {
    Info "resolving latest release"
    try {
        $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" -UseBasicParsing
        $Tag = $rel.tag_name
    } catch {
        Fail "could not query GitHub API: $_"
    }
} else {
    $Tag = $Version
}
$Ver = $Tag.TrimStart('v')
Info "version: $Tag"

$BaseUrl = "https://github.com/$Repo/releases/download/$Tag"
$TmpDir  = Join-Path $env:TEMP "anima-tagger-install-$([guid]::NewGuid())"
New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null
try {

$Verify = -not $NoVerify
$SumsFile = Join-Path $TmpDir "SHA256SUMS"
if ($Verify) {
    Info "downloading SHA256SUMS"
    try {
        Invoke-WebRequest -Uri "$BaseUrl/SHA256SUMS" -OutFile $SumsFile -UseBasicParsing
    } catch {
        Info "SHA256SUMS not found on this release; skipping verification"
        $Verify = $false
    }
}

function Get-ExpectedHash($name) {
    if (-not $Verify) { return $null }
    $line = Select-String -Path $SumsFile -Pattern "  $([regex]::Escape($name))$" | Select-Object -First 1
    if (-not $line) { Fail "no SHA256 entry for $name" }
    return ($line.Line -split '\s+')[0]
}

function Download-And-Verify($name) {
    $dest = Join-Path $TmpDir $name
    Info "downloading $name"
    Invoke-WebRequest -Uri "$BaseUrl/$name" -OutFile $dest -UseBasicParsing
    $expected = Get-ExpectedHash $name
    if ($expected) {
        $actual = (Get-FileHash -Algorithm SHA256 $dest).Hash.ToLower()
        if ($actual -ne $expected.ToLower()) {
            Fail "checksum mismatch for $name (expected $expected, got $actual)"
        }
    }
    return $dest
}

New-Item -ItemType Directory -Force -Path $Prefix | Out-Null

# CLI install — single-binary zip.
if (-not $GuiOnly) {
    $cliName = "anima-tagger-cli-$Ver-$Target.zip"
    $cliZip  = Download-And-Verify $cliName
    Expand-Archive -Force -Path $cliZip -DestinationPath $Prefix
    Info "installed CLI: $Prefix\anima-tagger.exe"
}

# GUI install — zip with both binaries inside a folder.
if (-not $CliOnly) {
    $guiName = "anima-tagger-$Ver-$Target.zip"
    $guiZip  = Download-And-Verify $guiName
    $extract = Join-Path $TmpDir "extract"
    New-Item -ItemType Directory -Force -Path $extract | Out-Null
    Expand-Archive -Force -Path $guiZip -DestinationPath $extract
    $inner = Get-ChildItem -Directory $extract | Select-Object -First 1
    if (-not $inner) { Fail "GUI archive layout unexpected" }
    Copy-Item -Force -Path (Join-Path $inner.FullName "anima-tagger-gui.exe") -Destination $Prefix
    Info "installed GUI: $Prefix\anima-tagger-gui.exe"
}

# Suggest PATH update
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$Prefix*") {
    Write-Host ""
    Write-Host "note: $Prefix is not on your user PATH. Run this in PowerShell to add it:"
    Write-Host "  [Environment]::SetEnvironmentVariable('Path', `$env:Path + ';$Prefix', 'User')"
}

Info "done."

} finally {
    Remove-Item -Recurse -Force $TmpDir -ErrorAction SilentlyContinue
}
