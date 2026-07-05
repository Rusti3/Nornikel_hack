param(
    [switch]$Build
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
    Write-Host 'Created .env with demo defaults. Add YANDEX_API_KEY later for full AI/Web mode.'
}

foreach ($seed in @('demo-seed/search-db.pgdump', 'demo-seed/neo4j.dump')) {
    if (-not (Test-Path -LiteralPath $seed)) {
        throw "Missing $seed. Run 'git lfs pull' before starting the demo."
    }
    $stream = [System.IO.File]::OpenRead((Resolve-Path -LiteralPath $seed))
    try {
        $buffer = New-Object byte[] 43
        $read = $stream.Read($buffer, 0, $buffer.Length)
        $prefix = [System.Text.Encoding]::ASCII.GetString($buffer, 0, $read)
    }
    finally {
        $stream.Dispose()
    }
    if ($prefix.StartsWith('version https://git-lfs.github.com/spec/v1')) {
        throw "$seed is still a Git LFS pointer. Run 'git lfs pull' before starting the demo."
    }
}

docker info *> $null
docker compose config --quiet

if ($Build) {
    docker compose -f docker-compose.yml -f docker-compose.build.yml build
} else {
    docker compose pull
}

docker compose up -d
docker compose ps

Write-Host ''
Write-Host 'Agentic RAG chat: http://localhost:8080'
Write-Host 'Neo4j Browser: http://localhost:7474'
