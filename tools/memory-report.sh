#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="${AFFON_BINARY:-$ROOT/zig-out/bin/affon}"
CONFIG="${1:-}"
OUTPUT="${2:-}"
HISTORY="${3:-${AFFON_MEMORY_HISTORY:-}}"

if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
  printf 'usage: %s <workflow-config.json> [report.json]\n' "$0" >&2
  exit 1
fi
if [[ ! -x "$BINARY" ]]; then
  printf 'missing executable: %s\n' "$BINARY" >&2
  exit 1
fi

MONITOR_PATH="$(jq -r '.report.monitor.path // empty' "$CONFIG")"
if [[ -z "$MONITOR_PATH" ]]; then
  printf 'workflow config must set report.monitor.path\n' >&2
  exit 1
fi
if [[ "$MONITOR_PATH" != /* ]]; then
  MONITOR_PATH="$ROOT/$MONITOR_PATH"
fi

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="${MONITOR_PATH%.json}-memory-report.json"
elif [[ "$OUTPUT" != /* ]]; then
  OUTPUT="$ROOT/$OUTPUT"
fi
if [[ -n "$HISTORY" && "$HISTORY" != /* ]]; then
  HISTORY="$ROOT/$HISTORY"
fi

mkdir -p "$(dirname "$MONITOR_PATH")" "$(dirname "$OUTPUT")"
LOG_PATH="$(mktemp "${TMPDIR:-/tmp}/affon-memory-report.XXXXXX.log")"
TIME_PATH="$(mktemp "${TMPDIR:-/tmp}/affon-memory-report.XXXXXX.time")"
trap 'rm -f "$LOG_PATH" "$TIME_PATH"' EXIT

set +e
if [[ "$(uname -s)" == "Darwin" ]]; then
  /usr/bin/time -l env AFFON_TRAIN_CONFIG="$CONFIG" "$BINARY" run "$ROOT/apps/decoder-lm/train.ts" >"$LOG_PATH" 2>"$TIME_PATH"
else
  /usr/bin/time -v env AFFON_TRAIN_CONFIG="$CONFIG" "$BINARY" run "$ROOT/apps/decoder-lm/train.ts" >"$LOG_PATH" 2>"$TIME_PATH"
fi
STATUS=$?
set -e

if [[ ! -f "$MONITOR_PATH" ]]; then
  cat "$LOG_PATH" >&2
  cat "$TIME_PATH" >&2
  exit "$STATUS"
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  PEAK_RSS="$(awk '/maximum resident set size/ { print $1; found=1 } END { if (!found) print 0 }' "$TIME_PATH")"
  PEAK_FOOTPRINT="$(awk '/peak memory footprint/ { print $1; found=1 } END { if (!found) print 0 }' "$TIME_PATH")"
else
  PEAK_RSS="$(awk '/Maximum resident set size/ { print $NF; found=1 } END { if (!found) print 0 }' "$TIME_PATH")"
  PEAK_FOOTPRINT=0
fi

jq -n \
  --arg config "$CONFIG" \
  --arg binary "$BINARY" \
  --argjson status "$STATUS" \
  --arg peakResident "$PEAK_RSS" \
  --arg peakFootprint "$PEAK_FOOTPRINT" \
  --slurpfile monitor "$MONITOR_PATH" \
  '{config: $config, binary: $binary, status: $status, peakResidentBytes: (try ($peakResident | tonumber) catch 0), peakFootprintBytes: (try ($peakFootprint | tonumber) catch 0), monitor: $monitor[0]}' >"$OUTPUT"

if [[ -n "$HISTORY" ]]; then
  CONFIG_HASH="$(shasum -a 256 "$CONFIG" | awk '{print $1}')"
  TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  HISTORY_ENTRY="$(jq --arg timestamp "$TIMESTAMP" --arg config "$CONFIG" --arg binary "$BINARY" --arg configHash "$CONFIG_HASH" '
    {
      timestamp: $timestamp,
      config: $config,
      configSha256: $configHash,
      binary: $binary,
      status,
      peakResidentBytes,
      peakFootprintBytes,
      snapshots: (.monitor.snapshots | length),
      maxMetrics: (reduce .monitor.snapshots[].runtimeMetrics[] as $metric ({};
        ($metric.scope + "/" + $metric.name) as $key |
        .[$key] = ([.[$key] // 0, $metric.value] | max)))
    }' "$OUTPUT")"
  mkdir -p "$(dirname "$HISTORY")"
  if [[ -f "$HISTORY" ]]; then
    jq empty "$HISTORY" 2>/dev/null || printf '[]\n' >"$HISTORY"
  else
    printf '[]\n' >"$HISTORY"
  fi
  PREVIOUS="$(jq --arg configHash "$CONFIG_HASH" --arg binary "$BINARY" '[.[] | select(.configSha256 == $configHash and .binary == $binary)] | .[-1] // null' "$HISTORY")"
  jq --argjson entry "$HISTORY_ENTRY" '. + [$entry]' "$HISTORY" >"$HISTORY.tmp"
  mv "$HISTORY.tmp" "$HISTORY"
  jq -n --argjson current "$HISTORY_ENTRY" --argjson previous "$PREVIOUS" '{current: $current, previous: $previous, delta: (if $previous == null then null else {peakResidentBytes: ($current.peakResidentBytes - $previous.peakResidentBytes), peakFootprintBytes: ($current.peakFootprintBytes - $previous.peakFootprintBytes)} end)}' >"${OUTPUT%.json}-comparison.json"
fi

printf '%s\n' "$OUTPUT"
exit "$STATUS"
