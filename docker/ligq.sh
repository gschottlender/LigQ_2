#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p work

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$command_name" in
    start)
        docker compose up -d "$@"
        echo "LigQ 2 is available at http://localhost:${LIGQ_WEB_PORT:-8080}"
        ;;
    build)
        docker compose build "$@"
        ;;
    stop)
        docker compose down "$@"
        ;;
    status)
        docker compose ps "$@"
        ;;
    logs)
        docker compose logs -f "$@"
        ;;
    pull)
        docker compose pull "$@"
        ;;
    init-data)
        docker compose --profile tools run --rm init-data "$@"
        ;;
    cli)
        docker compose --profile tools run --rm cli "$@"
        ;;
    help|--help|-h)
        cat <<'EOF'
Usage: ./docker/ligq.sh COMMAND [ARGS]

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
  ./docker/ligq.sh cli --input-fasta /work/queries.fasta --output-dir /work/results
EOF
        ;;
    *)
        echo "Unknown command: $command_name" >&2
        exit 2
        ;;
esac
