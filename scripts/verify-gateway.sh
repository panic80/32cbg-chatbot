#!/usr/bin/env bash

set -euo pipefail

BASE_URL="${1:-http://localhost:3000}"
VERIFY_SSE="false"

for arg in "${@:2}"; do
  case "$arg" in
    --verify-sse)
      VERIFY_SSE="true"
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

info() {
  printf '\033[1;34m[info]\033[0m %s\n' "$*"
}

warn() {
  printf '\033[1;33m[warn]\033[0m %s\n' "$*"
}

fail() {
  printf '\033[1;31m[fail]\033[0m %s\n' "$*"
  exit 1
}

check_endpoint() {
  local method="$1"
  local url="$2"
  local expected="$3"
  local description="$4"
  local extra_flags=("${@:5}")

  info "Checking ${description} (${method} ${url})"
  http_code=$(curl -sS -o /tmp/cbthis-health.$$ --write-out '%{http_code}' \
    --max-time 10 \
    -X "${method}" \
    "${extra_flags[@]}" \
    "${url}") || true

  if [[ "$http_code" != "$expected" ]]; then
    warn "Expected HTTP ${expected} but received ${http_code}"
    if [[ -s /tmp/cbthis-health.$$ ]]; then
      cat /tmp/cbthis-health.$$
    fi
    rm -f /tmp/cbthis-health.$$
    fail "${description} failed"
  fi

  rm -f /tmp/cbthis-health.$$
}

info "Verifying Express gateway at ${BASE_URL}"

check_endpoint "GET" "${BASE_URL}/health" "200" "Gateway health endpoint"
check_endpoint "GET" "${BASE_URL}/api/config" "200" "Configuration endpoint" "-H" "Accept: application/json"
check_endpoint "GET" "${BASE_URL}/api/travel-instructions" "200" "Travel instructions (cache warm-up)"

if [[ "${VERIFY_SSE}" == "true" ]]; then
  if [[ -z "${ADMIN_USER:-}" || -z "${ADMIN_PASS:-}" ]]; then
    warn "Skipping SSE proxy check (ADMIN_USER/ADMIN_PASS not set)"
  else
    auth_header=$(printf '%s:%s' "$ADMIN_USER" "$ADMIN_PASS" | base64)
    check_endpoint "GET" "${BASE_URL}/api/rag/ingest/progress" "401" \
      "SSE proxy auth guard"
    check_endpoint "GET" "${BASE_URL}/api/rag/ingest/progress" "400" \
      "SSE proxy parameter validation (authenticated)" \
      -H "Authorization: Basic ${auth_header}"
  fi
fi

info "All gateway checks passed"
