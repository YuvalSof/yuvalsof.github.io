# Auto-replace Medium-style long dashes with regular hyphens.
# Used by afterFileEdit, afterTabFileEdit, sessionStart, and git pre-commit.

param(
    [string]$RepairPath,
    [switch]$ScanStaged
)

$ErrorActionPreference = "Stop"

function Test-IsSiteTextFile {
    param([string]$Path)

    $normalized = $Path.Replace("\", "/").ToLowerInvariant()
    if ($normalized -match "/(\.git|_site|\.jekyll-cache|\.cursor/hooks)/") {
        return $false
    }

    $ext = [System.IO.Path]::GetExtension($Path).ToLowerInvariant()
    return $ext -in @(".md", ".markdown", ".yml", ".yaml", ".html", ".htm", ".txt", ".scss", ".css")
}

function Convert-LongDashes {
    param([string]$Text)

    $em = [char]0x2014
    $en = [char]0x2013
    $bar = [char]0x2015
    $fig = [char]0x2012
    $longClass = "[$em$en$bar$fig]"

    # Numeric or word-adjacent ranges: 2019-2020, make-model
    $Text = [regex]::Replace($Text, "(?<=[\w])$longClass(?=[\w])", "-")
    # Remaining punctuation dashes, keep a short spaced hyphen
    $Text = [regex]::Replace($Text, "\s*$longClass\s*", " - ")
    return $Text
}

function Repair-File {
    param([string]$Path)

    if (-not $Path -or -not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    if (-not (Test-IsSiteTextFile -Path $Path)) {
        return $false
    }

    $utf8 = New-Object System.Text.UTF8Encoding $false
    $original = [System.IO.File]::ReadAllText($Path)
    $updated = Convert-LongDashes -Text $original
    if ($updated -eq $original) {
        return $false
    }

    [System.IO.File]::WriteAllText($Path, $updated, $utf8)
    return $true
}

if ($RepairPath) {
    Repair-File -Path $RepairPath | Out-Null
    exit 0
}

if ($ScanStaged) {
    $repoRoot = (git rev-parse --show-toplevel 2>$null)
    if (-not $repoRoot) {
        exit 0
    }
    $staged = git diff --cached --name-only --diff-filter=ACMR
    foreach ($relative in $staged) {
        $full = Join-Path $repoRoot $relative
        if (Repair-File -Path $full) {
            git add -- "$relative"
        }
    }
    exit 0
}

$raw = [Console]::In.ReadToEnd()
if (-not $raw) {
    exit 0
}

try {
    $payload = $raw | ConvertFrom-Json
} catch {
    exit 0
}

if ($payload.file_path) {
    Repair-File -Path ([string]$payload.file_path) | Out-Null
    exit 0
}

if ($payload.session_id -or $payload.composer_mode) {
    $context = @"
Never use long dashes (em dash U+2014 or en dash U+2013) in this site. Write regular hyphens only: Case 1 - Title, word - word, 2019-2020. When converting Medium posts, replace every long dash in headlines and body text. Leave math minus signs and ASCII -- in code.
"@
    @{ additional_context = $context } | ConvertTo-Json -Compress
    exit 0
}

exit 0
