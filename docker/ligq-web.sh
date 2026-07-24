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
        echo "Usage: ./docker/ligq-web.sh {prepare|validate|start|start-local-data|build|pull|stop|status|logs}"
        ;;
    *)
        echo "Unknown command: $command_name" >&2
        exit 2
        ;;
esac
