#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/metal-footprint-watch.sh [-i seconds] [-s samples] [-o report.jsonl] -- <command> [args...]

Darwin-only helper for Metal memory regression checks. It launches a workload,
samples `vmmap -summary`, writes one JSON object per sample, and fails when
growth exceeds configured limits.

Environment:
  AFFON_METAL_MAX_PHYS_GROWTH_MB       default: 512
  AFFON_METAL_MAX_IOACCEL_REGION_GROWTH default: 5000

Example:
  AFFON_METAL_MAX_PHYS_GROWTH_MB=256 \
  AFFON_METAL_MAX_IOACCEL_REGION_GROWTH=2000 \
    tools/metal-footprint-watch.sh -i 15 -s 20 -o /tmp/metal.jsonl -- \
    bash -lc 'RUNTIME_PACKAGE_PATH=packages apps/decoder-lm/run.sh wikitext-debug'
EOF
}

INTERVAL=10
SAMPLES=12
OUTPUT=""

while getopts ":i:s:o:h" opt; do
  case "$opt" in
    i) INTERVAL="$OPTARG" ;;
    s) SAMPLES="$OPTARG" ;;
    o) OUTPUT="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) printf 'missing value for -%s\n' "$OPTARG" >&2; usage >&2; exit 2 ;;
    \?) printf 'unknown option: -%s\n' "$OPTARG" >&2; usage >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

if [[ "${1:-}" == "--" ]]; then shift; fi
if [[ "$#" -eq 0 ]]; then usage >&2; exit 2; fi
if [[ "$(uname -s)" != "Darwin" ]]; then
  printf 'metal-footprint-watch requires macOS vmmap\n' >&2
  exit 2
fi
if ! [[ "$INTERVAL" =~ ^[0-9]+$ && "$INTERVAL" -gt 0 ]]; then
  printf 'interval must be a positive integer\n' >&2
  exit 2
fi
if ! [[ "$SAMPLES" =~ ^[0-9]+$ && "$SAMPLES" -gt 0 ]]; then
  printf 'samples must be a positive integer\n' >&2
  exit 2
fi

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="$(mktemp "${TMPDIR:-/tmp}/affon-metal-footprint.XXXXXX.jsonl")"
elif [[ "$OUTPUT" != /* ]]; then
  OUTPUT="$(pwd)/$OUTPUT"
fi
mkdir -p "$(dirname "$OUTPUT")"
: >"$OUTPUT"

MAX_PHYS_GROWTH_BYTES=$(( (${AFFON_METAL_MAX_PHYS_GROWTH_MB:-512}) * 1024 * 1024 ))
MAX_IOACCEL_REGION_GROWTH="${AFFON_METAL_MAX_IOACCEL_REGION_GROWTH:-5000}"

"$@" &
PID=$!
START_EPOCH="$(date +%s)"
STATUS=0
BASE_PHYS=""
BASE_IOACCEL_GRAPHICS_REGIONS=""
LAST_PHYS=0
LAST_IOACCEL_GRAPHICS_REGIONS=0
FAILED=0

cleanup() {
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

parse_vmmap() {
  awk '
    function to_bytes(value, n, unit) {
      n = value + 0
      unit = substr(value, length(value), 1)
      if (unit == "K") return int(n * 1024)
      if (unit == "M") return int(n * 1024 * 1024)
      if (unit == "G") return int(n * 1024 * 1024 * 1024)
      if (unit == "T") return int(n * 1024 * 1024 * 1024 * 1024)
      return int(n)
    }
    /^Physical footprint:/ {
      phys = to_bytes($3)
      next
    }
    /^Physical footprint \(peak\):/ {
      phys_peak = to_bytes($4)
      next
    }
    $1 == "IOAccelerator" && $2 == "(graphics)" {
      io_graphics_virtual = to_bytes($3)
      io_graphics_resident = to_bytes($4)
      io_graphics_swapped = to_bytes($6)
      io_graphics_regions = $NF + 0
      next
    }
    $1 == "IOAccelerator" && $2 != "(graphics)" {
      io_virtual = to_bytes($2)
      io_resident = to_bytes($3)
      io_regions = $NF + 0
      next
    }
    END {
      printf "%d %d %d %d %d %d %d %d %d\n",
        phys + 0,
        phys_peak + 0,
        io_virtual + 0,
        io_resident + 0,
        io_regions + 0,
        io_graphics_virtual + 0,
        io_graphics_resident + 0,
        io_graphics_swapped + 0,
        io_graphics_regions + 0
    }
  '
}

for ((sample = 1; sample <= SAMPLES; sample++)); do
  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" || STATUS=$?
    break
  fi

  if SUMMARY="$(vmmap -summary "$PID" 2>/dev/null)"; then
    read -r PHYS PHYS_PEAK IO_VIRT IO_RES IO_REG IO_GFX_VIRT IO_GFX_RES IO_GFX_SWAP IO_GFX_REG < <(printf '%s\n' "$SUMMARY" | parse_vmmap)
    NOW="$(date +%s)"
    ELAPSED=$((NOW - START_EPOCH))
    if [[ -z "$BASE_PHYS" ]]; then
      BASE_PHYS="$PHYS"
      BASE_IOACCEL_GRAPHICS_REGIONS="$IO_GFX_REG"
    fi
    LAST_PHYS="$PHYS"
    LAST_IOACCEL_GRAPHICS_REGIONS="$IO_GFX_REG"
    PHYS_GROWTH=$((PHYS - BASE_PHYS))
    IO_GFX_REGION_GROWTH=$((IO_GFX_REG - BASE_IOACCEL_GRAPHICS_REGIONS))
    printf '{"timestamp":"%s","sample":%d,"elapsedSeconds":%d,"pid":%d,"physicalFootprintBytes":%d,"physicalFootprintPeakBytes":%d,"physicalFootprintGrowthBytes":%d,"ioAcceleratorVirtualBytes":%d,"ioAcceleratorResidentBytes":%d,"ioAcceleratorRegions":%d,"ioAcceleratorGraphicsVirtualBytes":%d,"ioAcceleratorGraphicsResidentBytes":%d,"ioAcceleratorGraphicsSwappedBytes":%d,"ioAcceleratorGraphicsRegions":%d,"ioAcceleratorGraphicsRegionGrowth":%d}\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$sample" "$ELAPSED" "$PID" \
      "$PHYS" "$PHYS_PEAK" "$PHYS_GROWTH" "$IO_VIRT" "$IO_RES" "$IO_REG" \
      "$IO_GFX_VIRT" "$IO_GFX_RES" "$IO_GFX_SWAP" "$IO_GFX_REG" "$IO_GFX_REGION_GROWTH" \
      | tee -a "$OUTPUT"
    if [[ "$PHYS_GROWTH" -gt "$MAX_PHYS_GROWTH_BYTES" || "$IO_GFX_REGION_GROWTH" -gt "$MAX_IOACCEL_REGION_GROWTH" ]]; then
      FAILED=1
      break
    fi
  fi

  sleep "$INTERVAL"
done

if kill -0 "$PID" 2>/dev/null; then
  kill "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
else
  wait "$PID" || STATUS=$?
fi

printf 'report: %s\n' "$OUTPUT" >&2
if [[ "$FAILED" -ne 0 ]]; then
  printf 'metal footprint regression: physical growth=%d bytes, IOAccelerator graphics region growth=%d\n' \
    "$((LAST_PHYS - BASE_PHYS))" "$((LAST_IOACCEL_GRAPHICS_REGIONS - BASE_IOACCEL_GRAPHICS_REGIONS))" >&2
  exit 1
fi
exit "$STATUS"
