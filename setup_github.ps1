param(
    [string]$RepoName = "mplads-ai-monitor",
    [ValidateSet("public","private")]
    [string]$Visibility = "public",
    [string]$GitName = "Abdul Hammad",
    [string]$GitEmail = "hammad_jamian12@outlook.com"
)

$ErrorActionPreference = "Stop"

Write-Host "== MPLADS AI MONITOR :: GitHub setup ==" -ForegroundColor Cyan

git --version
gh --version

git config user.name $GitName
git config user.email $GitEmail

if (-not (Test-Path ".git")) {
    git init
}

git branch -M main

git add .
git status --short

git commit -m "feat: initial MPLADS AI Monitor release" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "No new commit created (possibly nothing to commit)." -ForegroundColor Yellow
}

gh auth status

gh repo create "abdulhammad13/$RepoName" `
    "--$Visibility" `
    --description "Explainable anomaly and risk monitoring for MPLADS implementation — SIH 2026, Team Fresh Minds." `
    --source . `
    --remote origin `
    --push

Write-Host ""
Write-Host "Repository created and pushed." -ForegroundColor Green
Write-Host "https://github.com/abdulhammad13/$RepoName" -ForegroundColor Cyan
