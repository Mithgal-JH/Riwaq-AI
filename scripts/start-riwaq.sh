#!/usr/bin/env bash
set -Eeuo pipefail

CONTENT_PORT=8001
POST_PORT=8002
GATEWAY_PORT="${PORT:-8080}"
PIDS=()

terminate_children() {
    local pid
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}

wait_for_service() {
    local name="$1"
    local port="$2"
    local attempts=0

    while (( attempts < 180 )); do
        if ! kill -0 "${PIDS[0]}" 2>/dev/null && [[ "$name" == "content-analysis" ]]; then
            echo "$name exited before becoming healthy" >&2
            return 1
        fi
        if ! kill -0 "${PIDS[1]}" 2>/dev/null && [[ "$name" == "post-recommendation" ]]; then
            echo "$name exited before becoming healthy" >&2
            return 1
        fi
        if curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null; then
            echo "$name is healthy on port $port"
            return 0
        fi
        attempts=$((attempts + 1))
        sleep 2
    done

    echo "$name did not become healthy on port $port" >&2
    return 1
}

cleanup() {
    terminate_children
    local pid
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}

trap cleanup EXIT
trap 'exit 143' TERM INT

/opt/venvs/content-analysis/bin/uvicorn app.main:app \
    --app-dir /opt/riwaq/content-analysis \
    --host 127.0.0.1 \
    --port "$CONTENT_PORT" &
PIDS+=("$!")

/opt/venvs/post-recommendation/bin/uvicorn app.main:app \
    --app-dir /opt/riwaq/post-recommendation \
    --host 127.0.0.1 \
    --port "$POST_PORT" \
    --workers 1 &
PIDS+=("$!")

wait_for_service "content-analysis" "$CONTENT_PORT"
wait_for_service "post-recommendation" "$POST_PORT"

/opt/venvs/ai-gateway/bin/uvicorn app.main:app \
    --app-dir /opt/riwaq/ai-gateway \
    --host 0.0.0.0 \
    --port "$GATEWAY_PORT" &
PIDS+=("$!")

set +e
wait -n "${PIDS[@]}"
status=$?
set -e

echo "A Riwaq service process exited with status $status; stopping the remaining processes" >&2
exit "$status"