param(
    [Parameter(Position = 0)]
    [string]$Command = "help",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir
New-Item -ItemType Directory -Force -Path "work" | Out-Null

switch ($Command) {
    "start" {
        & docker compose up -d @RemainingArgs
        $Port = if ($env:LIGQ_WEB_PORT) { $env:LIGQ_WEB_PORT } else { "8080" }
        Write-Host "LigQ 2 is available at http://localhost:$Port"
    }
    "build" { & docker compose build @RemainingArgs }
    "stop" { & docker compose down @RemainingArgs }
    "status" { & docker compose ps @RemainingArgs }
    "logs" { & docker compose logs -f @RemainingArgs }
    "pull" { & docker compose pull @RemainingArgs }
    "init-data" { & docker compose --profile tools run --rm init-data @RemainingArgs }
    "cli" { & docker compose --profile tools run --rm cli @RemainingArgs }
    { $_ -in "help", "--help", "-h" } {
        Write-Host @"
Usage: .\docker\ligq.ps1 COMMAND [ARGS]

Commands:
  start       Start the web application in the background
  build       Build the CPU images locally
  stop        Stop the application (persistent volumes are preserved)
  status      Show container and health status
  logs        Follow application logs
  pull        Pull published images
  init-data   Download/prepare required data from the terminal
  cli         Run run_ligq_2.py with arguments after this command

Example:
  .\docker\ligq.ps1 cli --input-fasta /work/queries.fasta --output-dir /work/results
"@
    }
    default {
        Write-Error "Unknown command: $Command"
        exit 2
    }
}

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
