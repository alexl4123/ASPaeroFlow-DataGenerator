#!/usr/bin/env python3
"""Write instance_info.json for every instance in a parsed experiment tree.

Why this exists
---------------
Every published ASPaeroFlow instance carries an ``instance_info.json`` that makes an
extracted directory self-describing: its time granularity, demand level, seed, capacity
units and licence.  Until now **no code produced it** -- it was written by an ad-hoc step
during release preparation.  The consequence was a reproducibility hole: checking out the
tagged generator and re-running reproduced every CSV byte for byte, but did not reproduce
``instance_info.json`` at all.

This script closes that hole.  It derives the whole file from the instance tree, so it can
be re-run on any tree the generator produces, and it is checked against the published
archives (``--verify-zip``) to prove it reproduces them exactly.

It deliberately does **not** live inside a pipeline stage.  The facts it needs -- the time
granularity, whether the region uses grid navpoints, the short region name, the licence --
are spread across stages 02, 04 and the release packaging, and none of them reach stage 05.
Keeping it separate also means it changes no pipeline output, so the output regression
baseline in tests/regression/ is untouched.

Usage
-----
    python write_instance_info.py --root experiment_data_V2_small_scaling
    python write_instance_info.py --root experiment_data_..._TG15 --dry-run
    python write_instance_info.py --verify-zip ../release_upload/.../experiment_data_V2_small_scaling.zip
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path

# Regions whose navgraph vertices come from X-Plane navdata rather than a synthetic grid.
# Grid regions always carry an "<n>x<m>" suffix; these four do not.  The distinction is not
# cosmetic -- it decides the licence, because the X-Plane navdata is GPL-2.0-or-later and
# that obligation travels into any instance built from it.
GRID_SUFFIX = re.compile(r"-\d+x\d+$")

CAPACITY_UNITS = "flights per sector PER TIMESTEP (not per hour)"
SECTORS_KEYED  = "Sector_ID; navaid_sector_assignment.csv maps Navaid_ID -> Sector_ID"

NAVPOINTS_GRID_SMALL = "synthetic rectangular grid (grid-navpoints); no X-Plane navdata used"
NAVPOINT_SRC_GRID    = "synthetic rectangular grid (no third-party navdata)"
NAVPOINT_SRC_XPLANE  = "X-Plane AptNav navdata, data cycle 2013.10 (GPL-2.0-or-later)"

CAPACITY_SWEEP_NONE = ("none -- cap-enroute is 1, the instances are tightly capacitated "
                       "by construction")

LICENCE_SMALL  = "CC-BY-4.0; see LICENSE-DATA.txt at the archive root"
LICENCE_GRID   = ("CC-BY-4.0 -- this region uses synthetic grid navpoints and contains no "
                  "X-Plane navdata. See LICENSING.md.")
LICENCE_XPLANE = ("GPL-2.0-or-later -- this region's navgraph vertices are extracted from the "
                  "X-Plane AptNav navdata (GPL-2.0-or-later). See LICENSING.md.")

PCAP_DIR  = re.compile(r"^PCAP(\d+)$")
INSTANCE  = re.compile(r"^(\d+)_SEED(\d+)$")
TG_SUFFIX = re.compile(r"-TG(\d+)$")
NUM_PREFIX = re.compile(r"^\d+-\d+-")


def region_of(experiment: str) -> str:
    """Short region name.

    Large-scaling experiment names carry a numeric prefix and the fitted date range, e.g.
    ``02-0-MAJOR-EUROPE-40x20-2019-06-01--2019-06-30-CAP-ENROUTE-1200-...-TG15``; the region
    is what sits between them.  Small-scaling names carry no date range, and the published
    files use the experiment name unchanged -- so the presence of the date decides.
    """
    if "-2019-" not in experiment:
        return experiment
    return NUM_PREFIX.sub("", experiment).split("-2019-")[0]


def min_capacity(sectors_csv: Path) -> int:
    """Lowest capacity in sectors.csv -- the en-route value.

    Airport sectors are given a deliberately huge capacity so they never bind, so the
    minimum is the en-route capacity that actually constrains the instance.
    """
    with open(sectors_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{sectors_csv} has no rows")
    col = "Capacity" if "Capacity" in rows[0] else list(rows[0])[-1]
    return min(int(float(r[col])) for r in rows)


def build(instance_dir: Path, experiment: str, pcap: int | None,
          generator_commit: str) -> dict:
    m = INSTANCE.match(instance_dir.name)
    if not m:
        raise ValueError(f"not an instance directory name: {instance_dir.name}")
    flights, seed = int(m.group(1)), int(m.group(2))

    tg_m = TG_SUFFIX.search(experiment)
    tg = int(tg_m.group(1)) if tg_m else 1

    region = region_of(experiment)
    is_grid = bool(GRID_SUFFIX.search(region))

    info = {
        "experiment": experiment,
        "region": region,
        "time_granularity_bins_per_hour": tg,
        "timesteps_per_day": tg * 24,
        "minutes_per_timestep": 60.0 / tg,
        "flights": flights,
        "seed": seed,
    }

    if pcap is None:
        # Small-scaling family: no sweep, capacity is tight by construction.
        info["enroute_capacity_per_timestep"] = min_capacity(instance_dir / "sectors.csv")
        info["capacity_units"] = CAPACITY_UNITS
        info["capacity_sweep"] = CAPACITY_SWEEP_NONE
        info["navpoints"] = NAVPOINTS_GRID_SMALL
        info["sectors_csv_keyed_by"] = SECTORS_KEYED
        info["generator_commit"] = generator_commit
        info["licence"] = LICENCE_SMALL
        return info

    # Large-scaling family.  nominal_capacity cannot be inverted from the swept value --
    # max(1, ceil(nominal * pct)) is not injective -- so read it from the PCAP100 sibling,
    # where the capacity IS the nominal by construction.
    nominal_dir = instance_dir.parent.parent / "PCAP100" / instance_dir.name
    if not (nominal_dir / "sectors.csv").exists():
        raise FileNotFoundError(
            f"need the PCAP100 sibling to recover nominal_capacity, missing: {nominal_dir}")
    info["capacity_level_percent_of_nominal"] = pcap
    info["nominal_capacity"] = min_capacity(nominal_dir / "sectors.csv")
    info["sector_capacity_per_timestep"] = min_capacity(instance_dir / "sectors.csv")
    info["capacity_units"] = CAPACITY_UNITS
    info["sectors_csv_keyed_by"] = SECTORS_KEYED
    info["generator_commit"] = generator_commit
    info["licence"] = LICENCE_GRID if is_grid else LICENCE_XPLANE
    info["navpoint_source"] = NAVPOINT_SRC_GRID if is_grid else NAVPOINT_SRC_XPLANE
    return info


def serialise(info: dict) -> str:
    return json.dumps(info, indent=2) + "\n"


def walk(root: Path):
    """Yield (instance_dir, experiment_name, pcap_level_or_None)."""
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for child in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            pm = PCAP_DIR.match(child.name)
            if pm:
                for inst in sorted(p for p in child.iterdir() if p.is_dir()):
                    if INSTANCE.match(inst.name):
                        yield inst, exp_dir.name, int(pm.group(1))
            elif INSTANCE.match(child.name):
                yield child, exp_dir.name, None


def verify_zip(zip_path: Path, generator_commit: str) -> int:
    """Rebuild every instance_info.json in a published archive and compare byte for byte."""
    checked = mismatched = 0
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith("instance_info.json")]
        for n in sorted(names):
            parts = Path(n).parts               # <archive>/<experiment>/[PCAPnn/]<instance>/file
            experiment = parts[1]
            pcap = None
            pm = PCAP_DIR.match(parts[2])
            if pm:
                pcap = int(pm.group(1))
            inst_name = parts[-2]

            published = json.loads(z.read(n))
            # sectors.csv lives beside it inside the archive; read capacities from there
            def cap(rel):
                with z.open(rel) as fh:
                    rows = list(csv.DictReader(l.decode() for l in fh))
                col = "Capacity" if "Capacity" in rows[0] else list(rows[0])[-1]
                return min(int(float(r[col])) for r in rows)

            rebuilt = {
                "experiment": experiment,
                "region": region_of(experiment),
            }
            tg_m = TG_SUFFIX.search(experiment)
            tg = int(tg_m.group(1)) if tg_m else 1
            rebuilt.update({
                "time_granularity_bins_per_hour": tg,
                "timesteps_per_day": tg * 24,
                "minutes_per_timestep": 60.0 / tg,
                "flights": int(INSTANCE.match(inst_name).group(1)),
                "seed": int(INSTANCE.match(inst_name).group(2)),
            })
            here = "/".join(parts[:-1])
            is_grid = bool(GRID_SUFFIX.search(rebuilt["region"]))
            if pcap is None:
                rebuilt["enroute_capacity_per_timestep"] = cap(here + "/sectors.csv")
                rebuilt["capacity_units"] = CAPACITY_UNITS
                rebuilt["capacity_sweep"] = CAPACITY_SWEEP_NONE
                rebuilt["navpoints"] = NAVPOINTS_GRID_SMALL
                rebuilt["sectors_csv_keyed_by"] = SECTORS_KEYED
                rebuilt["generator_commit"] = generator_commit
                rebuilt["licence"] = LICENCE_SMALL
            else:
                nominal = "/".join(parts[:2]) + f"/PCAP100/{inst_name}/sectors.csv"
                rebuilt["capacity_level_percent_of_nominal"] = pcap
                rebuilt["nominal_capacity"] = cap(nominal)
                rebuilt["sector_capacity_per_timestep"] = cap(here + "/sectors.csv")
                rebuilt["capacity_units"] = CAPACITY_UNITS
                rebuilt["sectors_csv_keyed_by"] = SECTORS_KEYED
                rebuilt["generator_commit"] = generator_commit
                rebuilt["licence"] = LICENCE_GRID if is_grid else LICENCE_XPLANE
                rebuilt["navpoint_source"] = NAVPOINT_SRC_GRID if is_grid else NAVPOINT_SRC_XPLANE

            checked += 1
            if serialise(rebuilt).encode() != z.read(n):
                mismatched += 1
                if mismatched <= 3:
                    print(f"  MISMATCH {n}", file=sys.stderr)
                    for k in dict.fromkeys(list(published) + list(rebuilt)):
                        a, b = published.get(k, "<absent>"), rebuilt.get(k, "<absent>")
                        if a != b:
                            print(f"    {k}: published={a!r} rebuilt={b!r}", file=sys.stderr)
    print(f"{zip_path.name}: {checked} checked, {mismatched} mismatched")
    return mismatched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, help="parsed experiment root to write into")
    ap.add_argument("--verify-zip", type=Path, nargs="*", default=[],
                    help="published archive(s) to check this script reproduces exactly")
    ap.add_argument("--generator-commit", default="9adfe58",
                    help="commit recorded in the file (default: the V2 release commit)")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    if args.verify_zip:
        bad = sum(verify_zip(z, args.generator_commit) for z in args.verify_zip)
        print("VERIFY: exact reproduction" if bad == 0 else f"VERIFY FAILED: {bad} mismatched")
        return 0 if bad == 0 else 1

    if not args.root:
        ap.error("give --root or --verify-zip")
    n = 0
    for inst, experiment, pcap in walk(args.root):
        text = serialise(build(inst, experiment, pcap, args.generator_commit))
        if not args.dry_run:
            (inst / "instance_info.json").write_text(text)
        n += 1
    print(f"{'would write' if args.dry_run else 'wrote'} {n} instance_info.json under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
