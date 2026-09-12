#!/usr/bin/env bash
#
# run_fixtures.sh -- regenerate every regression fixture from scratch and either
# compare it against the committed baseline or refresh that baseline.
#
#   tests/regression/run_fixtures.sh                  # regenerate + compare (gate a refactor)
#   tests/regression/run_fixtures.sh --update-baseline # regenerate + overwrite baseline/
#   tests/regression/run_fixtures.sh --only ea3x3      # one fixture
#   tests/regression/run_fixtures.sh --list            # names and what each covers
#   tests/regression/run_fixtures.sh --shipped-check   # also verify the published
#                                                      # EAST-ASIA-3x3 instances
#                                                      # regenerate from source
#
# Every fixture is generated into a FRESH directory.  This is not tidiness: stage 04
# rewrites flights.csv in place and is not idempotent, so a re-run over an existing
# tree silently produces different output (README, convention ④).  Any non-zero exit
# from any stage aborts the script.
#
# PYTHONHASHSEED is pinned.  Stage 04 iterates a Python set
# (04_simplified_filed_flight_plan_generator.py:613), which makes the row order of
# aircrafts.csv depend on string hash randomisation.  Pinning the seed makes the
# whole pipeline byte-reproducible; see tests/regression/README.md.

set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BASELINE_DIR="$HERE/baseline"

: "${SCRATCH:=/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-28-OpenSky-Symposium-02-JOAS-paper/0e375282-c006-4cf5-9440-2e5311f5ca9e/scratchpad/regression}"
: "${CSV_PATH:=/home/thinklex/Documents/2026_not_for_dropbox/05_ASPaeroFlow_Data/ASPaeroFlow-DataGenerator/flightlist_20190601_20190630.csv}"
: "${NAVDIR:=/home/thinklex/Documents/2026_not_for_dropbox/05_ASPaeroFlow_Data/ASPaeroFlow-DataGenerator/test_navpoints}"
: "${PYTHON:=python}"

export PYTHONHASHSEED=0

# name | config | extra run_pipeline args | what it covers
FIXTURES=(
  "ea3x3|default_configs_small_scaling/30_0_east_asia_3x3.json|--flight-flights 10,50,100 --flight-seeds 42,13|grid navpoints, TG=1, 17-vertex graph, stages 00-05"
  "ce5x5|default_configs_small_scaling/30_1_central_europe_5x5.json|--flight-flights 10,50,100 --flight-seeds 42,13|grid navpoints, TG=1, denser airport set, stages 00-05"
  "india4x10|default_configs_small_scaling/30_2_india_4x10.json|--flight-flights 10,50,100 --flight-seeds 42,13|grid navpoints, TG=1, non-square grid (4x10), stages 00-05"
  "usa7x7|default_configs_small_scaling/30_3_usa_7x7.json|--flight-flights 10,50,100 --flight-seeds 42,13|grid navpoints, TG=1, western-hemisphere longitudes, stages 00-05"
  "majeur10x10|default_configs_small_scaling/30_4_major_europe_10x10.json|--flight-flights 10,50,100 --flight-seeds 42,13|grid navpoints, TG=1, largest small-scaling grid, stages 00-05"
  "ea3x3_tg4|default_configs_small_scaling/30_0_east_asia_3x3.json|--time-granularity 4 --flight-flights 50,100 --flight-seeds 42|same region at TG=4: exercises stage 04 timestep arithmetic that TG=1 does not"
  "dach_gabriel_tg15|default_configs_large_scaling_tg/04_0_dach_TG15.json|--navdir @NAVDIR@ --flight-flights 300 --flight-seeds 42|REAL X-Plane waypoints: BallTree neighbourhoods + Gabriel edge criterion, 1508 vertices, min-dist filter, TG=15, stages 00-05"
)

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

list_fixtures() {
  printf '%-20s %s\n' "FIXTURE" "COVERS"
  local row name cfg extra doc
  for row in "${FIXTURES[@]}"; do
    IFS='|' read -r name cfg extra doc <<<"$row"
    printf '%-20s %s\n' "$name" "$doc"
    printf '%-20s   config: %s %s\n' "" "$cfg" "$extra"
  done
}

ONLY=""
UPDATE=0
SHIPPED=0
COMPARE_FLAGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="${2:?--only needs a fixture name}"; shift 2 ;;
    --update-baseline) UPDATE=1; shift ;;
    --shipped-check) SHIPPED=1; shift ;;
    --allow-row-reorder) COMPARE_FLAGS+=("--allow-row-reorder"); shift ;;
    --list) list_fixtures; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- preflight
command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "FATAL: '$PYTHON' is not on PATH.  run_pipeline.py spawns its stages as" >&2
  echo "       'python <stage>.py', so 'python' specifically must resolve to an" >&2
  echo "       interpreter with the dependencies." >&2
  exit 2
}
"$PYTHON" - <<'PY' || { echo "FATAL: missing dependencies (see requirements.txt)" >&2; exit 2; }
import sys
import sklearn, networkx, pandas, numpy, tqdm  # noqa: F401
assert sys.version_info >= (3, 10), sys.version
PY

[[ -f "$CSV_PATH" ]] || { echo "FATAL: flight list not found: $CSV_PATH" >&2
                          echo "       set CSV_PATH=... to point at it" >&2; exit 2; }

mkdir -p "$SCRATCH" "$BASELINE_DIR"
MANIFEST_DIR="$SCRATCH/manifests"
mkdir -p "$MANIFEST_DIR"

echo "repo     : $REPO"
echo "scratch  : $SCRATCH"
echo "flights  : $CSV_PATH"
echo "navdata  : $NAVDIR"
echo "python   : $($PYTHON -V 2>&1)  PYTHONHASHSEED=$PYTHONHASHSEED"
echo

# ---------------------------------------------------------------- one fixture
declare -a TIMINGS=()
FAILURES=0

run_fixture() {
  local name="$1" cfg="$2" extra="$3"
  local work="$SCRATCH/$name"
  local unparsed="$work/unparsed" parsed="$work/parsed"

  echo "################################################################"
  echo "# fixture: $name"
  echo "################################################################"

  # Clean slate -- stage 04 is not idempotent, a partial tree would be reused.
  rm -rf "$work"
  mkdir -p "$unparsed" "$parsed"

  extra="${extra//@NAVDIR@/$NAVDIR}"
  if [[ "$extra" == *"$NAVDIR"* && ! -d "$NAVDIR" ]]; then
    echo "FATAL: fixture $name needs X-Plane navdata but $NAVDIR is missing." >&2
    echo "       Expected files fix.dat and nav.dat.  Set NAVDIR=... ." >&2
    return 1
  fi

  local t0 t1
  t0=$(date +%s)

  # ---- stages 00-04 ------------------------------------------------------
  # shellcheck disable=SC2086
  ( cd "$REPO" && "$PYTHON" run_pipeline.py \
      --config "$cfg" \
      --csv-path "$CSV_PATH" \
      --out-root "$unparsed" \
      $extra ) || {
    echo "FATAL: run_pipeline.py exited non-zero for fixture '$name'." >&2
    echo "       Judge by the exit code -- run_pipeline captures stage output and" >&2
    echo "       prints it only on failure, so a successful stage 04 prints no [OK]." >&2
    return 1
  }

  # ---- stage 05 ----------------------------------------------------------
  local exp_dir
  exp_dir="$(find "$unparsed" -mindepth 1 -maxdepth 1 -type d | head -1)"
  [[ -n "$exp_dir" ]] || { echo "FATAL: no experiment directory under $unparsed" >&2; return 1; }

  ( cd "$REPO" && "$PYTHON" 05_transform_for_optimizer.py \
      --in-exp-dir "$exp_dir" \
      --out-root "$parsed" ) || {
    echo "FATAL: 05_transform_for_optimizer.py exited non-zero for fixture '$name'." >&2
    return 1
  }

  t1=$(date +%s)
  TIMINGS+=("$name $((t1 - t0))s")

  # ---- fingerprint -------------------------------------------------------
  local manifest="$MANIFEST_DIR/$name.json"
  "$PYTHON" "$HERE/fingerprint.py" --root "$work" --out "$manifest" --label "$name" || return 1

  if [[ "$UPDATE" == "1" ]]; then
    cp "$manifest" "$BASELINE_DIR/$name.json"
    echo "[baseline] updated $BASELINE_DIR/$name.json"
    return 0
  fi

  if [[ ! -f "$BASELINE_DIR/$name.json" ]]; then
    echo "FATAL: no baseline for fixture '$name'.  Generate one on unmodified code:" >&2
    echo "       tests/regression/run_fixtures.sh --update-baseline --only $name" >&2
    return 1
  fi

  "$PYTHON" "$HERE/compare.py" \
      "$BASELINE_DIR/$name.json" "$manifest" \
      --candidate-root "$work" \
      "${COMPARE_FLAGS[@]+"${COMPARE_FLAGS[@]}"}" || return 1
}

RAN=0
for row in "${FIXTURES[@]}"; do
  IFS='|' read -r name cfg extra _doc <<<"$row"
  [[ -n "$ONLY" && "$ONLY" != "$name" ]] && continue
  RAN=$((RAN + 1))
  if ! run_fixture "$name" "$cfg" "$extra"; then
    echo "*** FIXTURE FAILED: $name ***" >&2
    FAILURES=$((FAILURES + 1))
  fi
  echo
done

if [[ -n "$ONLY" && "$RAN" == "0" ]]; then
  echo "FATAL: --only '$ONLY' matched no fixture.  See --list." >&2
  exit 2
fi

# ------------------------------------------------- published-data provenance
if [[ "$SHIPPED" == "1" ]]; then
  echo "################################################################"
  echo "# shipped-instance check: does the published EAST-ASIA-3x3 data"
  echo "# regenerate from this source tree?"
  echo "################################################################"
  ship_work="$SCRATCH/_shipped_check"
  rm -rf "$ship_work"
  mkdir -p "$ship_work/unparsed" "$ship_work/parsed"

  ( cd "$REPO" && "$PYTHON" run_pipeline.py \
      --config default_configs_small_scaling/30_0_east_asia_3x3.json \
      --csv-path "$CSV_PATH" --out-root "$ship_work/unparsed" ) \
    || { echo "FATAL: shipped-check generation failed" >&2; exit 1; }

  ( cd "$REPO" && "$PYTHON" 05_transform_for_optimizer.py \
      --in-exp-dir "$ship_work/unparsed/30-0-EAST-ASIA-3x3-V2" \
      --out-root "$ship_work/parsed" ) \
    || { echo "FATAL: shipped-check transform failed" >&2; exit 1; }

  zip="$REPO/../release_upload/small_scaling/instances/experiment_data_V2_small_scaling.zip"
  if [[ -f "$zip" ]]; then
    rm -rf "$ship_work/shipped"; mkdir -p "$ship_work/shipped"
    unzip -oq "$zip" -d "$ship_work/shipped"
    "$PYTHON" "$HERE/fingerprint.py" \
        --root "$ship_work/shipped/experiment_data_V2_small_scaling/30-0-EAST-ASIA-3x3-V2" \
        --out "$MANIFEST_DIR/_shipped_published.json" --label "published-east-asia-3x3" \
        --skip-name instance_info.json
    "$PYTHON" "$HERE/fingerprint.py" \
        --root "$ship_work/parsed/30-0-EAST-ASIA-3x3-V2" \
        --out "$MANIFEST_DIR/_shipped_regenerated.json" --label "published-east-asia-3x3"
    echo
    echo "--- published instances (zip) vs regenerated ---"
    # instance_info.json is written by build_release_zips.py, not by the pipeline,
    # and records the generator commit; it is excluded above, not normalised away.
    "$PYTHON" "$HERE/compare.py" \
        "$MANIFEST_DIR/_shipped_published.json" "$MANIFEST_DIR/_shipped_regenerated.json" \
        --baseline-root "$ship_work/shipped/experiment_data_V2_small_scaling/30-0-EAST-ASIA-3x3-V2" \
        --candidate-root "$ship_work/parsed/30-0-EAST-ASIA-3x3-V2" \
        "${COMPARE_FLAGS[@]+"${COMPARE_FLAGS[@]}"}" \
      || { echo "*** SHIPPED-INSTANCE CHECK FAILED ***" >&2; FAILURES=$((FAILURES + 1)); }
  else
    echo "[skip] published zip not found at $zip" >&2
  fi
  echo
fi

# ---------------------------------------------------------------- summary
echo "================================================================"
echo "fixture runtimes"
for t in "${TIMINGS[@]+"${TIMINGS[@]}"}"; do echo "  $t"; done
echo "================================================================"
if [[ "$FAILURES" -gt 0 ]]; then
  echo "RESULT: $FAILURES fixture(s) FAILED"
  exit 1
fi
if [[ "$UPDATE" == "1" ]]; then
  echo "RESULT: baseline refreshed"
else
  echo "RESULT: all fixtures match the baseline"
fi
