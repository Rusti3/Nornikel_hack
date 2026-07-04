$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

docker compose stop
docker compose ps
