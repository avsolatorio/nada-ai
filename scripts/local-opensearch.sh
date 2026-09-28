#!/usr/bin/env bash
# Local OpenSearch setup for developing and testing NADA's OpenSearch search (development only, no security).
#
#   scripts/local-opensearch.sh start    start OpenSearch (Docker) and nada-ai (this checkout, on the host)
#   scripts/local-opensearch.sh stop     stop nada-ai (OpenSearch keeps running; `docker stop` it if you want)
#   scripts/local-opensearch.sh status   what is running and what the index holds
#   scripts/local-opensearch.sh logs     follow the nada-ai log
#
# nada-ai runs on the host, not in Docker, because the embedding model needs the host's CPU/GPU: on Docker for Mac it is
# many times slower. The code is served straight from this checkout, so restart it after changing code.
#
# Ports: OpenSearch 9201, nada-ai 8021. (8020 is the Qdrant stack from docker-compose.qdrant.yml; both can run.)
# Re-indexing is an admin API call on the running nada-ai: POST /admin/catalog/index per metadata type (recreate_index=true
# on the first call only), or POST /admin/ingest/from-catalog/all. Recreating the indexes discards what they hold.
# Point NADA at it: Site configurations > Search > Semantic search settings: API URL http://localhost:8021, engine OpenSearch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/data/opensearch-local"          # gitignored: pid, log, discovery cache, filter facets
CONTAINER="${OPENSEARCH_CONTAINER:-nada-os36-scratch}"
VOLUME="${OPENSEARCH_VOLUME:-nada-os36-scratch-data}"
OS_PORT="${OPENSEARCH_PORT:-9201}"
AI_PORT="${NADA_AI_PORT:-8021}"
NADA_URL="${NADA_URL:-http://localhost/nada-semantic/index.php}"   # where nada-ai reads the catalog (host address)
INDEX="${NADA_INDEX_NAME:-nada-os-baseline}"
PIDFILE="$STATE/nada-ai.pid"
LOGFILE="$STATE/nada-ai.log"

mkdir -p "$STATE/discovery"
[ -f "$STATE/dynamic_filter_facets.json" ] || echo '{}' > "$STATE/dynamic_filter_facets.json"

opensearch_up() {
  if docker inspect "$CONTAINER" >/dev/null 2>&1; then
    local published_host
    published_host="$(docker inspect --format '{{range (index .HostConfig.PortBindings "9200/tcp")}}{{.HostIp}}{{end}}' "$CONTAINER")"
    if [[ "$published_host" != "127.0.0.1" ]]; then
      echo "Existing container $CONTAINER is not bound to 127.0.0.1 (HostIp: ${published_host:-unset}); recreate it to apply the loopback-only binding." >&2
      return 1
    fi
    docker start "$CONTAINER" >/dev/null 2>&1 || true
    docker update --restart unless-stopped "$CONTAINER" >/dev/null
  else
    docker run -d --name "$CONTAINER" --restart unless-stopped -p "127.0.0.1:$OS_PORT:9200" \
      -e discovery.type=single-node -e plugins.security.disabled=true -e DISABLE_INSTALL_DEMO_CONFIG=true \
      -e OPENSEARCH_JAVA_OPTS="-Xms1g -Xmx1g" -v "$VOLUME:/usr/share/opensearch/data" \
      opensearchproject/opensearch:3.6.0 >/dev/null
  fi
  for _ in $(seq 1 60); do
    curl -fs "http://localhost:$OS_PORT/_cluster/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "OpenSearch did not become healthy on port $OS_PORT" >&2; return 1
}

# the catalog key comes from .env (never printed); everything else is set here so a stray .env value cannot win
catalog_key() { grep -E '^AI4DATA_METADATA_CATALOG_X_API_KEY=' "$ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2- || true; }

nada_ai_env() {
  export NADA_SEARCH_BACKEND=opensearch
  export NADA_OPENSEARCH_URL="http://localhost:$OS_PORT"
  export NADA_INDEX_NAME="$INDEX"
  # State reporting mutates NADA's search_index_state; keep it opt-in for local runs.
  export NADA_REPORT_SEARCH_INDEX_STATE_ENABLED="${NADA_REPORT_SEARCH_INDEX_STATE_ENABLED:-false}"
  export NADA_RECONCILE_SEARCH_INDEX_ENABLED=false
  # The API is bound to loopback below, so admin auth can be disabled for this local development process.
  unset NADA_ADMIN_API_KEY
  export NADA_ADMIN_AUTH_DISABLED=true
  # Requests per minute per caller; 0 disables rate limiting (the default, see .env.example).
  export NADA_RATE_LIMIT_SEARCH_PER_MINUTE="${NADA_RATE_LIMIT_SEARCH_PER_MINUTE:-0}"
  export NADA_DYNAMIC_FILTER_FACETS_PATH="$STATE/dynamic_filter_facets.json"
  export AI4DATA_METADATA_CATALOG_URL="$NADA_URL"
  export AI4DATA_DISCOVERY_DATA_PATH="$STATE/discovery"
  local key; key="$(catalog_key)"
  [ -z "$key" ] || export AI4DATA_METADATA_CATALOG_X_API_KEY="$key"
}

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

stop_ai() {
  if running; then kill "$(cat "$PIDFILE")"; sleep 1; fi
  rm -f "$PIDFILE"
  # a process started by hand on the same port (before this script existed)
  pkill -f "uvicorn nada_ai.app.main:app --host 127.0.0.1 --port $AI_PORT" 2>/dev/null || true
}

start_ai() {
  stop_ai
  nada_ai_env
  cd "$ROOT"
  nohup .venv/bin/python -m uvicorn nada_ai.app.main:app --host 127.0.0.1 --port "$AI_PORT" >>"$LOGFILE" 2>&1 &
  echo $! > "$PIDFILE"
  for _ in $(seq 1 40); do
    curl -fs "http://127.0.0.1:$AI_PORT/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "nada-ai did not come up on port $AI_PORT; see $LOGFILE" >&2; return 1
}

status() {
  echo "OpenSearch  : $(curl -fs "http://localhost:$OS_PORT/_cluster/health" 2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); print("up,", d["status"])' 2>/dev/null || echo down) (port $OS_PORT, container $CONTAINER)"
  echo "nada-ai     : $(curl -fs "http://127.0.0.1:$AI_PORT/health" >/dev/null 2>&1 && echo "up (port $AI_PORT, pid $(cat "$PIDFILE" 2>/dev/null || echo '?'))" || echo down)"
  curl -fs "http://127.0.0.1:$AI_PORT/info" 2>/dev/null | python3 -c '
import sys, json
d = json.load(sys.stdin); i = d.get("index") or {}
print("engine      :", d["engine"], d.get("engine_version"))
print("can do      :", ", ".join(k for k, v in d["capabilities"].items() if v))
print("index       : %s, %s studies, model %s" % (i.get("name"), i.get("studies"), i.get("embedding_model")))' 2>/dev/null || true
}

case "${1:-status}" in
  start)  opensearch_up; start_ai; status ;;
  stop)   stop_ai; echo "nada-ai stopped (OpenSearch left running)" ;;
  status) status ;;
  logs)   tail -f "$LOGFILE" ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
