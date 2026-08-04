#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

compose=(docker compose -f compose.web.yml)
compose_local=(docker compose -f compose.web.yml -f compose.web.local-data.yml)
command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

require_local_data_dir() {
    if [[ ! -d "$ROOT_DIR/databases" ]]; then
        echo "[ERROR] ./databases does not exist." >&2
        echo "start-local-data only reuses an existing native data directory." >&2
        echo "Use './docker/ligq-web.sh prepare' and 'start' for isolated Docker data." >&2
        exit 1
    fi
    if [[ ! -r "$ROOT_DIR/databases" || ! -x "$ROOT_DIR/databases" ]]; then
        echo "[ERROR] ./databases is not readable by the current user." >&2
        exit 1
    fi
}

case "$command_name" in
    prepare)
        "${compose[@]}" --profile admin run --rm prepare-data "$@"
        "${compose[@]}" --profile admin run --rm validate-data
        ;;
    validate)
        "${compose[@]}" --profile admin run --rm validate-data "$@"
        ;;
    start)
        "${compose[@]}" up -d api web "$@"
        echo "LigQ 2 web test is available at http://${LIGQ_WEB_BIND:-127.0.0.1}:${LIGQ_WEB_PORT:-18081}"
        ;;
    start-local-data)
        require_local_data_dir
        "${compose_local[@]}" --profile admin run --rm validate-data
        "${compose_local[@]}" up -d api web "$@"
        echo "LigQ 2 web test is using ./databases read-only at http://${LIGQ_WEB_BIND:-127.0.0.1}:${LIGQ_WEB_PORT:-18081}"
        ;;
    build)
        "${compose[@]}" build "$@"
        ;;
    pull)
        "${compose[@]}" pull "$@"
        ;;
    stop)
        "${compose[@]}" down "$@"
        ;;
    status)
        "${compose[@]}" ps "$@"
        ;;
    logs)
        "${compose[@]}" logs -f "$@"
        ;;
    help|--help|-h)
        cat <<'EOF'
Usage: ./docker/ligq-web.sh COMMAND [ARGS]

Commands:
  prepare           Download and validate data in the isolated Docker volume
  validate          Validate data in the isolated Docker volume
  start             Start the web stack with the isolated Docker volume
  start-local-data  Validate and reuse an existing ./databases directory read-only
  build             Build the web images
  pull              Pull the published web images
  stop              Stop the web stack
  status            Show container and health status
  logs              Follow web stack logs
EOF
        ;;
    *)
        echo "Unknown command: $command_name" >&2
        exit 2
        ;;
esac
