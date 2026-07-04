param(
    [ValidateSet("Run", "Status", "Render")]
    [string]$Action = "Run",
    [string]$RunId = "demo-openrouter-01",
    [int]$Limit = 0,
    [switch]$Preflight
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunDir = Join-Path $Root "artifacts\demo-eval\$RunId"
$Checkpoint = Join-Path $RunDir "checkpoint.json"

switch ($Action) {
    "Run" {
        $arguments = @(
            "backend\scripts\evaluate_demo_questions.py",
            "--run-id", $RunId,
            "--output-dir", "artifacts\demo-eval"
        )
        if ($Limit -gt 0) { $arguments += @("--limit", $Limit) }
        if ($Preflight) { $arguments += "--preflight" }
        & python @arguments
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    "Status" {
        if (-not (Test-Path -LiteralPath $Checkpoint)) {
            throw "Checkpoint not found: $Checkpoint"
        }
        $value = Get-Content -LiteralPath $Checkpoint -Raw -Encoding UTF8 | ConvertFrom-Json
        [pscustomobject]@{
            RunId = $value.run_id
            Provider = $value.provider
            Model = $value.model
            Finished = @($value.results.PSObject.Properties).Count
            UpdatedAt = $value.updated_at
            BlockedReason = $value.blocked_reason
        } | Format-List
    }
    "Render" {
        if (-not (Test-Path -LiteralPath $Checkpoint)) {
            throw "Checkpoint not found: $Checkpoint"
        }
        & python "backend\scripts\build_demo_questions.py" --evaluation $Checkpoint
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
