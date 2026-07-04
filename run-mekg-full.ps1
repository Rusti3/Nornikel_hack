param(
    [ValidateSet('Start', 'Resume', 'Status', 'Stop')]
    [string]$Action = 'Start',
    [string]$RunId = '',
    [double]$DeadlineHours = 12,
    [int]$BenchmarkLimit = 0
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if ($Action -eq 'Stop') {
    docker compose stop backend
    exit $LASTEXITCODE
}

if ($Action -eq 'Status') {
    $uri = 'http://localhost:8000/api/mekg/v1/pipeline/status'
    if ($RunId) { $uri += '?run_id=' + [Uri]::EscapeDataString($RunId) }
    Invoke-RestMethod $uri | ConvertTo-Json -Depth 6
    exit 0
}

if ($Action -eq 'Resume' -and -not $RunId) {
    throw 'Resume requires -RunId.'
}

docker compose up -d --build search-db neo4j backend frontend
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$arguments = @(
    'compose', 'exec', '-T', 'backend',
    'python', '-m', 'src.mekg.cli', 'ingest-corpus', '/corpus',
    '--mode', 'fast-full', '--deadline-hours', $DeadlineHours
)
if ($RunId) { $arguments += @('--resume', $RunId) }
if ($BenchmarkLimit -gt 0) { $arguments += @('--benchmark-limit', $BenchmarkLimit) }

& docker @arguments
exit $LASTEXITCODE
