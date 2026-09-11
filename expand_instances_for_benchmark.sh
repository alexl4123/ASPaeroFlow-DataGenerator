#!/usr/bin/env bash
# Expand the generated instances into the layout the optimizer's benchmark caller expects,
# materialising every capacity level of the PCAP sweep.
#
# WHY THIS IS NEEDED
# The generator ships the sweep as OVERLAYS: capacity_overlays_.../<REGION>/PCAPxxx/<dataset>/
# contains only sectors.csv, because that is the only file that differs between levels. The
# optimizer, by contrast, wants a directory whose immediate subdirectories are complete instances
# (start_benchmark_caller.py iterates instance_dir and treats each subdirectory as one instance).
# So each (region, granularity, capacity level) becomes its own problem directory here.
#
# DISK
# Levels are HARD-LINKED against the base instance and only sectors.csv is a real copy, so the
# ten levels cost about as much as one. A plain copy would be roughly 24 GB for the large family;
# this is roughly 2.5 GB. Nothing writes into these trees, so sharing inodes is safe -- but note
# that if you ever edit a file in place inside one level you will edit it in all of them.
#
# USAGE
#   ./expand_instances_for_benchmark.sh <GENERATOR_DIR> <DEST_05_INSTANCES> [--clean]
# e.g.
#   ./expand_instances_for_benchmark.sh . ../ASPaeroFlow-Optimizer/05_instances --clean
#
# --clean wipes the destination first. Use it: a stale problem directory left over from an
# earlier expansion is not overwritten by this script, it simply survives, and then it appears
# in nobody's manifest while still occupying the array index space -- or worse, a previous
# manifest with a different row order is still lying next to it. Cleaning makes the destination
# a pure function of the inputs.
#
# It writes DEST/problems.tsv, which run_all_benchmarks.slurm consumes as its job-array index.

set -euo pipefail

GEN="${1:?usage: $0 <GENERATOR_DIR> <DEST_05_INSTANCES> [--clean]}"
DEST="${2:?usage: $0 <GENERATOR_DIR> <DEST_05_INSTANCES> [--clean]}"
CLEAN="${3:-}"

GEN="$(cd "$GEN" && pwd)"
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
MANIFEST="$DEST/problems.tsv"

if [ "$CLEAN" = "--clean" ]; then
  # Guards, because this is an rm -rf on a path taken from the command line.
  case "$DEST" in
    "/"|"$HOME"|"$HOME/") echo "[ABORT] refusing to clean $DEST" >&2; exit 1 ;;
  esac
  [ "$(dirname "$DEST")" != "/" ] || { echo "[ABORT] refusing to clean a top-level directory" >&2; exit 1; }
  case "$DEST" in
    *05_instances*) : ;;
    *) echo "[ABORT] --clean expects a path containing '05_instances', got: $DEST" >&2; exit 1 ;;
  esac
  n_existing=$(find "$DEST" -maxdepth 1 -mindepth 1 | wc -l)
  echo "[CLEAN] removing $n_existing entries from $DEST"
  # The trees are hard links to the generator's data; removing them frees only the link, never
  # the underlying instance files.
  rm -rf "${DEST:?}"/*
elif [ -n "$CLEAN" ]; then
  echo "[ABORT] unknown third argument: $CLEAN (did you mean --clean?)" >&2; exit 1
fi

printf 'problem_dir\ttime_granularity\tregion\tcapacity_level\tn_instances\n' > "$MANIFEST"

link_level () {          # <base instance dir> <overlay level dir> <target problem dir>
  local base="$1" overlay="$2" target="$3" n=0 ds name
  mkdir -p "$target"
  for ds in "$overlay"/*/; do
    [ -f "${ds}sectors.csv" ] || continue
    name="$(basename "$ds")"
    [ -d "$base/$name" ] || { echo "  [WARN] no base instance for $name" >&2; continue; }
    rm -rf "${target:?}/$name"
    # -a preserves attributes, -l hard-links the payload instead of copying it
    cp -al "$base/$name" "$target/$name"
    # --remove-destination unlinks first, so the shared inode of the base is NOT modified.
    # Plain `cp` would write through the hard link and silently corrupt every other level.
    cp --remove-destination "${ds}sectors.csv" "$target/$name/sectors.csv"
    n=$((n + 1))
  done
  echo "$n"
}

echo "=== large-scaling families (with capacity sweep) ==="
for tg in 1 4 15 60; do
  parsed="$GEN/experiment_data_V2_large_scaling_TG${tg}"
  overlays="$GEN/capacity_overlays_V2_large_scaling_TG${tg}"
  [ -d "$parsed" ] || { echo "  [SKIP] $parsed not present"; continue; }
  for region_path in "$parsed"/*/; do
    region="$(basename "$region_path")"
    ov_region="$overlays/$region"
    if [ ! -d "$ov_region" ]; then
      echo "  [WARN] no overlays for $region -- shipping base capacities only" >&2
      target="$DEST/${region}-PCAPBASE"
      rm -rf "$target"; cp -al "$region_path" "$target"
      n=$(find "$target" -maxdepth 1 -mindepth 1 -type d | wc -l)
      printf '%s\t%s\t%s\t%s\t%s\n' "${region}-PCAPBASE" "$tg" "$region" "BASE" "$n" >> "$MANIFEST"
      continue
    fi
    for lvl_path in "$ov_region"/PCAP*/; do
      lvl="$(basename "$lvl_path")"
      target="$DEST/${region}-${lvl}"
      n=$(link_level "$region_path" "$lvl_path" "$target")
      printf '%s\t%s\t%s\t%s\t%s\n' "${region}-${lvl}" "$tg" "$region" "$lvl" "$n" >> "$MANIFEST"
      echo "  TG${tg}  ${region}  ${lvl}  ${n} instances"
    done
  done
done

echo "=== small-scaling family (no sweep: cap-enroute is already 1) ==="
small="$GEN/experiment_data_V2_small_scaling"
if [ -d "$small" ]; then
  for region_path in "$small"/*/; do
    region="$(basename "$region_path")"
    target="$DEST/$region"
    rm -rf "$target"; cp -al "$region_path" "$target"
    n=$(find "$target" -maxdepth 1 -mindepth 1 -type d | wc -l)
    printf '%s\t%s\t%s\t%s\t%s\n' "$region" "1" "$region" "NONE" "$n" >> "$MANIFEST"
    echo "  TG1   ${region}  ${n} instances"
  done
else
  echo "  [SKIP] $small not present"
fi

total_problems=$(( $(wc -l < "$MANIFEST") - 1 ))
total_instances=$(awk -F'\t' 'NR>1 {s+=$5} END {print s+0}' "$MANIFEST")
echo
echo "wrote $MANIFEST"
echo "  $total_problems problem directories, $total_instances instances in total"
echo "  array range for sbatch:  --array=1-${total_problems}"
echo
echo "Next:  cd <optimizer>/06_benchmark_start_script && sbatch run_all_benchmarks.slurm"
