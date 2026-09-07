#!/usr/bin/env python3
"""
Nominal sector capacity and relative-capacity (PCAP) instance sweep.

Two steps, both solver-free:

  1. NOMINAL CAPACITY. The smallest uniform sector capacity at which the *unregulated* filed
     plan produces no overload -- i.e. the maximum number of flights simultaneously present in
     any one sector. Previously this required running the ASP solver with the controller
     disabled and scraping stdout (see ASPaeroFlow-Optimizer/10_ANALYZE_NOMINAL_CAPACITY_
     REQUIREMENTS/gather_capacities.py); it is reproduced here directly from the generated
     artifacts so the generator is self-contained.

  2. PCAP SWEEP. Emit copies of the instance at 10%..100% of nominal, overwriting sectors.csv.

OCCUPANCY SEMANTICS -- must match 02_ASP/encoding.lp or the capacities are meaningless:

    flight(ID,S,T)  :- navpoint_flight(ID,X,T), navpoint_sector(X,S,T).          (line 133)
    flightT(ID,T ,T',0) :- navpoint_seq(ID,T,TT), D=TT-T, T'>T,  T'<=T+D/2.      (line 134)
    flightT(ID,TT,T',1) :- navpoint_seq(ID,T,TT), D=TT-T, T'<TT, T'> T+D/2.      (line 135)
    overload(X,T,LOAD-C) :- sector(X,T,C), #count{ID:flight(ID,X,T)}=LOAD, LOAD>C. (line 72)

So a flight occupies the sector of its *nearest* waypoint: between consecutive waypoints at
times T and TT it is attributed to T's sector for T' <= T+D/2 and to TT's sector afterwards
(integer division, matching clingo). Load is a count of DISTINCT flight IDs.

Usage
-----
    # report nominal capacity only
    python 06_capacity_sweep.py --exp-dir <experiment> --report-only

    # emit PCAP010..PCAP100 next to the experiment
    python 06_capacity_sweep.py --exp-dir <experiment> --out-root <dir> \
        [--percentages 0.1,0.2,...] [--time-granularity 4]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_sector_map(ds_dir: Path) -> dict[str, str]:
    """Navaid -> Sector for ONE parsed dataset (each is self-contained)."""
    p = ds_dir / "navaid_sector_assignment.csv"
    df = pd.read_csv(p)
    if not {"Navaid_ID", "Sector_ID"}.issubset(df.columns):
        raise ValueError(f"{p} must have columns Navaid_ID,Sector_ID")
    return dict(zip(df["Navaid_ID"].astype(str), df["Sector_ID"].astype(str)))


def sector_occupancy(filed_csv: Path, sector_of: dict[str, str]) -> dict[tuple[str, int], set]:
    """
    (sector, time) -> set of flight IDs present, replicating the encoding's nearest-waypoint
    attribution including the interpolated timesteps between waypoints.
    """
    df = pd.read_csv(filed_csv, usecols=["Flight_ID", "Position", "Time"])
    df["Time"] = df["Time"].astype(int)
    occ: dict[tuple[str, int], set] = defaultdict(set)

    for fid, g in df.groupby("Flight_ID", sort=False):
        g = g.sort_values("Time")
        times = g["Time"].tolist()
        secs = [sector_of.get(str(p)) for p in g["Position"]]
        for i, (t, s) in enumerate(zip(times, secs)):
            if s is not None:
                occ[(s, t)].add(fid)                      # at the waypoint itself
            if i + 1 < len(times):
                tt, ss = times[i + 1], secs[i + 1]
                d = tt - t
                if d <= 1:
                    continue
                half = t + d // 2                          # clingo integer division
                for tp in range(t + 1, half + 1):          # first half -> this waypoint
                    if s is not None:
                        occ[(s, tp)].add(fid)
                for tp in range(half + 1, tt):             # second half -> next waypoint
                    if ss is not None:
                        occ[(ss, tp)].add(fid)
    return occ


def nominal_capacity(exp_dir: Path, dataset: Path) -> tuple[int, str, int]:
    """
    Return (nominal, argmax_sector, argmax_time) for one PARSED dataset.

    The sweep operates on parsed instances, not the unparsed experiment tree: a parsed dataset
    carries its own sectors.csv, navaid_sector_assignment.csv and flight plan, so a capacity
    level is self-contained. The unparsed tree shares one navgraph/ across all datasets, which
    cannot hold per-dataset capacities -- nominal capacity differs per dataset. This also
    matches the original generate_exps_nominal.py, which ran over 05_instances/.
    """
    sector_of = load_sector_map(dataset)
    occ = sector_occupancy(dataset / "flights.csv", sector_of)
    if not occ:
        return 0, "", 0
    (sec, t), ids = max(occ.items(), key=lambda kv: len(kv[1]))
    return len(ids), sec, t


def write_sectors(path: Path, capacity: int) -> None:
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        ids = [row[0] for row in r if row]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in ids:
            w.writerow([i, capacity])


def main() -> int:
    ap = argparse.ArgumentParser(description="Nominal capacity and PCAP sweep.")
    ap.add_argument("--exp-dir", required=True, type=Path)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--percentages", type=str,
                    default=",".join(f"{x/10:.1f}" for x in range(1, 11)))
    ap.add_argument("--time-granularity", type=int, default=None,
                    help="defaults to the value recorded in the experiment manifest")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--materialise", type=str, default=None,
                    help="rebuild full instances for this PCAP level, e.g. PCAP030")
    ap.add_argument("--overlay-root", type=Path, default=None)
    ap.add_argument("--dest", type=Path, default=None)
    a = ap.parse_args()

    if a.materialise:
        if not (a.overlay_root and a.dest):
            ap.error("--materialise requires --overlay-root and --dest")
        return materialise(a.exp_dir.expanduser(), a.overlay_root.expanduser(),
                           a.materialise, a.dest.expanduser())

    exp = a.exp_dir.expanduser()
    tg = a.time_granularity
    if tg is None:
        tm = next(exp.glob("*/transform_manifest.json"), None)
        if tm is not None:
            tg = int(json.load(open(tm)).get("time_granularity", 0)) or None
    if tg is None:
        print("[ERROR] could not infer time-granularity; pass --time-granularity")
        return 1

    datasets = sorted(p for p in exp.iterdir()
                      if p.is_dir() and (p / "flights.csv").exists()
                      and (p / "navaid_sector_assignment.csv").exists()
                      and (p / "sectors.csv").exists())
    if not datasets:
        print(f"[ERROR] no parsed datasets under {exp}. Point --exp-dir at a PARSED "
              f"experiment (output of 05_transform_for_optimizer.py).")
        return 1

    print(f"experiment      : {exp.name}")
    print(f"time-granularity: {tg}  (window = {tg*24} slots)")
    print(f"\n{'dataset':28s} {'nominal':>8s} {'at sector':>22s} {'t':>6s}")
    noms = {}
    for d in datasets:
        n, sec, t = nominal_capacity(exp, d)
        noms[d.name] = n
        print(f"{d.name:28s} {n:8d} {sec[:22]:>22s} {t:6d}")

    if a.report_only or a.out_root is None:
        return 0

    pcts = [float(x) for x in a.percentages.split(",") if x.strip()]
    out_root = a.out_root.expanduser() / exp.name
    out_root.mkdir(parents=True, exist_ok=True)

    # nominal capacities, so the sweep is reproducible and auditable without rerunning this
    with open(out_root / "nominal_capacities.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "nominal_capacity", "time_granularity"])
        for d in datasets:
            w.writerow([d.name, noms[d.name], tg])

    # OVERLAY LAYOUT. A PCAP level differs from the base instance in exactly one file --
    # navgraph/sectors.csv -- so shipping ten full copies would duplicate flights.csv,
    # filed_flights.csv, vertices.csv and edges.csv ten times for nothing. Measured on the
    # V2 large-scaling family that is 27 GB against 0.5 GB for the same information.
    print(f"\nwriting capacity overlays to {out_root}")
    for pct in pcts:
        tag = f"PCAP{int(round(pct*100)):03d}"
        caps = []
        for d in datasets:
            # Capacity is per timestep and measured directly in slots, so there is NO
            # multiplication by time_granularity here. Floor of 1: a capacity of 0 would make
            # the instance trivially infeasible rather than merely heavily regulated.
            cap = max(1, math.ceil(noms[d.name] * pct))
            caps.append(cap)
            dest = out_root / tag / d.name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(d / "sectors.csv", dest / "sectors.csv")
            write_sectors(dest / "sectors.csv", cap)
        print(f"  {tag}: capacities {min(caps)}..{max(caps)}")

    (out_root / "README.md").write_text(f"""# Relative-capacity (PCAP) overlays -- {exp.name}

Each `PCAP<pct>/<dataset>/sectors.csv` holds the sector capacities for that dataset at `<pct>`
percent of its nominal capacity. **Only `sectors.csv` differs between levels**; every other file
is identical to the base instance, so the levels are shipped as overlays rather than as ten full
copies (27 GB vs 0.5 GB for the V2 large-scaling family).

`nominal_capacities.csv` records the nominal capacity per dataset. Nominal capacity is the
smallest uniform sector capacity at which the unregulated filed plan produces no overload, i.e.
the maximum number of flights simultaneously present in any one sector, counted with the same
semantics as the solver encoding (a flight between two waypoints is attributed to the nearer
one). It is **granularity-dependent**: the same traffic yields a different nominal capacity at
each `time-granularity`, so percentages are not comparable across granularities.

## Reconstructing a full instance

    python 06_capacity_sweep.py --exp-dir <base-experiment> \\
        --materialise PCAP030 --overlay-root <this directory> --dest <output>

or manually: copy the base instance and replace `navgraph/sectors.csv` with the overlay file.

Capacities are floored at 1.
""")
    print(f"  wrote nominal_capacities.csv and README.md")
    return 0


def materialise(exp: Path, overlay_root: Path, tag: str, dest: Path) -> int:
    """Rebuild full instance trees for one PCAP level from base + overlay."""
    src = overlay_root / tag
    if not src.is_dir():
        print(f"[ERROR] no overlay {src}")
        return 1
    n = 0
    for ds in sorted(src.glob("*")):
        if not (ds / "sectors.csv").exists():
            continue
        out = dest / tag / ds.name
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(exp / ds.name, out)
        shutil.copy2(ds / "sectors.csv", out / "sectors.csv")
        n += 1
    print(f"materialised {n} datasets for {tag} -> {dest / tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
