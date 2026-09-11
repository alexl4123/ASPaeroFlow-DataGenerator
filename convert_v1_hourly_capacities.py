#!/usr/bin/env python3
"""
Convert V1 instances (per-HOUR sector capacities) to the per-TIMESTEP convention V2 uses.

WHY
The V1 instances carry sectors.csv::Capacity as a per-HOUR budget, and the optimizer of that era
divided it by the number of timesteps per hour before using it. The current optimizer reads the
column as a per-TIMESTEP capacity directly, with no switch to restore the old behaviour. Running
a V1 instance through the current optimizer unchanged therefore hands it TG times too much
capacity -- at TG=4 a PCAP050 instance would look almost feasible.

This applies the division once, on disk, so that both dataset generations are read identically
and a V1-vs-V2 comparison at a matched PCAP level is meaningful.

    capacity_per_timestep = max(1, capacity_per_hour // timesteps_per_hour)

The floor at 1 matches the sweep's own convention (no level is trivially infeasible).

Check with --dry-run first; it prints what would change without writing.

    ./convert_v1_hourly_capacities.py --root <V1 tree> --time-granularity 4 --dry-run
    ./convert_v1_hourly_capacities.py --root <V1 tree> --time-granularity 4
"""
import argparse
import csv
import sys
from pathlib import Path


def convert(path: Path, tg: int, dry: bool) -> tuple[int, int, int]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "Capacity" not in rows[0]:
        return 0, 0, 0
    before = {int(r["Capacity"]) for r in rows}
    for r in rows:
        r["Capacity"] = str(max(1, int(r["Capacity"]) // tg))
    after = {int(r["Capacity"]) for r in rows}
    if not dry:
        tmp = path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        tmp.replace(path)
    return len(rows), min(before), min(after)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="V1 instance tree; every sectors.csv beneath it is converted")
    ap.add_argument("--time-granularity", required=True, type=int,
                    help="timesteps per hour of these instances (1, 4, 15 or 60)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    files = sorted(a.root.rglob("sectors.csv"))
    if not files:
        sys.exit(f"no sectors.csv found under {a.root}")
    print(f"{'DRY RUN: ' if a.dry_run else ''}{len(files)} sectors.csv under {a.root}, "
          f"dividing by TG={a.time_granularity}\n")
    shown = 0
    for f in files:
        n, lo_before, lo_after = convert(f, a.time_granularity, a.dry_run)
        if n and shown < 5:
            print(f"  {f.relative_to(a.root)}   {lo_before} -> {lo_after}  ({n} rows)")
            shown += 1
    print(f"\n{'would convert' if a.dry_run else 'converted'} {len(files)} files")
    if a.dry_run:
        print("re-run without --dry-run to apply")
    else:
        print("NOTE: this rewrites in place. Keep an untouched copy of the V1 tree if you need "
              "to reproduce the originally published numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
