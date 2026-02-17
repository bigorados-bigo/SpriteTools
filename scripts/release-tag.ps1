param(
    [Parameter(Mandatory = $true)]
    [string]$Version,

    [switch]$SkipPushMain
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-CommandExists {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

function Normalize-Tag {
    param([string]$InputVersion)
    if ([string]::IsNullOrWhiteSpace($InputVersion)) {
        throw "Version cannot be empty."
    }

    $trimmed = $InputVersion.Trim()
    if ($trimmed -match '^v\d+\.\d+\.\d+$') {
        return $trimmed
    }

    if ($trimmed -match '^\d+\.\d+\.\d+$') {
        return "v$trimmed"
    }

    throw "Invalid version format: '$InputVersion'. Use X.Y.Z or vX.Y.Z"
}

Assert-CommandExists git

$tag = Normalize-Tag -InputVersion $Version

$repoRoot = git rev-parse --show-toplevel
if (-not $repoRoot) {
    throw "Not inside a git repository."
}

Push-Location $repoRoot
try {
    $status = git status --porcelain
    if ($status) {
        throw "Working tree is not clean. Commit or stash changes before releasing."
    }

    $currentBranch = (git rev-parse --abbrev-ref HEAD).Trim()
    if ($currentBranch -ne 'main') {
        throw "Release must be run from 'main'. Current branch: $currentBranch"
    }

    $existingLocalTag = git tag --list $tag
    if ($existingLocalTag) {
        throw "Tag already exists locally: $tag"
    }

    git fetch --tags --quiet

    $remoteTagCheck = git ls-remote --tags origin $tag
    if ($remoteTagCheck) {
        throw "Tag already exists on origin: $tag"
    }

    if (-not $SkipPushMain) {
        Write-Host "Pushing main to origin..."
        git push origin main
    }

    Write-Host "Creating tag $tag..."
    git tag -a $tag -m "SpriteTools $tag"

    Write-Host "Pushing tag $tag to origin..."
    git push origin $tag

    Write-Host "Release tag pushed successfully: $tag"
    Write-Host "GitHub Actions release workflow will publish assets automatically."
}
finally {
    Pop-Location
}