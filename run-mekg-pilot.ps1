param(
    [switch]$Reset,
    [switch]$ParseOnly,
    [switch]$NoVision,
    [switch]$SkipBuild,
    [switch]$SkipRemotePreflight,
    [switch]$SkipEmbeddings,
    [ValidateRange(0, 1000)]
    [int]$MaxExtractElementsPerDoc = 0,
    [ValidateRange(0, 32)]
    [int]$MaxConcurrency = 0,
    [ValidateRange(1, 8)]
    [int]$EmbeddingConcurrency = 2
)

$ErrorActionPreference = 'Stop'
$project = 'nornikel-knowledge-graph'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root

docker compose -p $project config --quiet
if (-not $SkipBuild) {
    docker compose -p $project up -d --build
} else {
    docker compose -p $project up -d
}

$deadline = (Get-Date).AddMinutes(8)
do {
    $services = docker compose -p $project ps --format json | ConvertFrom-Json
    $unhealthy = @($services | Where-Object { $_.Health -ne 'healthy' })
    if ($unhealthy.Count -eq 0 -and @($services).Count -eq 3) { break }
    if ((Get-Date) -gt $deadline) { throw 'Docker services did not become healthy in time.' }
    Start-Sleep -Seconds 5
} while ($true)

$preflightArgs = @('compose', '-p', $project, 'exec', '-T', 'backend', 'python', '-m', 'src.mekg.cli', 'preflight')
if ($SkipRemotePreflight) { $preflightArgs += '--local-only' }
& docker @preflightArgs
if ($LASTEXITCODE -ne 0) { throw 'MEKG preflight failed. Corpus processing was not started.' }

$ingestArgs = @('compose', '-p', $project, 'exec', '-T')
if ($MaxExtractElementsPerDoc -gt 0) {
    $ingestArgs += @('-e', "MEKG_MAX_EXTRACT_ELEMENTS_PER_DOC=$MaxExtractElementsPerDoc")
}
if ($MaxConcurrency -gt 0) {
    $ingestArgs += @('-e', "MEKG_MAX_CONCURRENCY=$MaxConcurrency")
}
$ingestArgs += @('backend', 'python', '-m', 'src.mekg.cli', 'ingest-manifest', '/code/pilot/manifest.json', '--corpus-root', '/corpus')
if ($Reset) { $ingestArgs += '--reset' }
if ($ParseOnly) { $ingestArgs += '--parse-only' }
if ($NoVision) { $ingestArgs += '--no-vision' }
& docker @ingestArgs
if ($LASTEXITCODE -ne 0) { throw 'MEKG pilot ingestion failed.' }

if (-not $ParseOnly -and -not $SkipEmbeddings) {
    docker compose -p $project exec -T backend python -m src.mekg.cli embed-evidence --concurrency $EmbeddingConcurrency
    if ($LASTEXITCODE -ne 0) { throw 'MEKG evidence embedding failed.' }
}

$artifactTarget = Join-Path $root 'artifacts\mekg-pilot'
New-Item -ItemType Directory -Force -Path $artifactTarget | Out-Null
docker compose -p $project cp backend:/code/artifacts/pilot-report/. $artifactTarget
Write-Host "MEKG pilot completed. QA report: $artifactTarget\qa-report.md"
