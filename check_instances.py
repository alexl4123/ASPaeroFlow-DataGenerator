#!/usr/bin/env python3
"""
Validity checker for generated instances.

Run it on anything you generated — one instance directory, one experiment, or a
whole parsed root — and it answers one question: **is this instance well formed
enough for a solver to consume?**

    python check_instances.py experiment_data_V2_small_scaling
    python check_instances.py experiment_data_V2_small_scaling/30-0-EAST-ASIA-3x3-V2
    python check_instances.py <root> --time-granularity 4 --verbose

Exit code 0 = every check passed, 1 = at least one violation, 2 = usage error.

It reads the **parsed** (solver-ready) files written by stage 05 —
``flights.csv``, ``graph_edges.csv``, ``sectors.csv``, … — not the unparsed
tree. Those are what a user downloads, so they are what gets checked.

The checks are a port of the audit scripts in
``dataset_analysis_JOAS/00_generator_integrity/`` (``05_parsed_instance_integrity.py``,
``06_parsed_deep_checks.py``, ``07_feasibility_check.py``, plus ``C4`` from
``03_instance_integrity.py``). **The identifiers are stable**: ``P4`` here is the
``P4`` those scripts and the dev log refer to. Several identifiers name the same
predicate in different scripts — ``D4`` and ``F4`` are one check, so are
``P2``/``F9``, ``D1``/``F8``, ``D3``/``F7`` — and are reported together rather
than run twice.

What is checked (a violation fails the run)
-------------------------------------------
  P1        every required file is present
  P2/F9     distinct flight count == the number in the directory name
  P3/F1     every timestep lies in [0, TG*24] -- the 24 h window is a hard
            contract (README convention ③). Reported as two halves: P3 for the
            upper bound, P3b for departures before t=0, which is what a
            backward-shifted over-long trajectory produces
  F3        no single inter-waypoint gap exceeds the whole window
  P4/F6     every sector capacity >= 1 -- 0 is unsatisfiable by construction
  P5/D6/F6  every graph vertex has a sector, every sector used is declared
  P6/F2     each flight's Time strictly increases
  D2        one position per (flight, timestep)
  P6b/F6    every Position is a navaid the instance declares
  P7        every airport vertex is on the graph
  P8/F5     every flight is assigned to exactly one declared airplane
  D1/F8     consecutive positions are graph-adjacent -- no teleporting
  D3/F7     flights start and end at airport vertices
  C4        no flight starts and ends at the same airport (self-loop)
  D4/F4     two legs of one airframe are separated by >= 1 timestep -- an
            airframe is never airborne twice at once
  D8/F10    rotation continuity: an airframe's next leg departs from the airport
            its previous leg landed at. An airframe cannot teleport between legs
            any more than a flight can teleport between waypoints
  P9        transform_manifest.json agrees with the directory name
  P10       navaid_sector_schedule.csv, the sector allocation over time, is well
            formed -- checked only when the file is present, since instances
            generated before it existed do not carry one and stay valid:
    P10a      every From_Time is a whole timestep in [0, TG*24)
    P10b      no two rows share (Navaid_ID, From_Time)
    P10c      every navpoint has exactly one sector at every timestep of the
              window: the rows are change-points, so with P10b this holds
              exactly when every navpoint has a row at From_Time 0
    P10d      every sector the schedule uses is declared in sectors.csv
    P10e      the schedule at t=0 maps every navpoint to the same sector as
              navaid_sector_assignment.csv -- solvers read one file or the
              other, and must see one instance

What is reported but never fails
--------------------------------
  P3-clamp  share of flights ending exactly on the window edge. At TG=1 the
            per-edge slot cost consumes the window for larger graphs, so a high
            share is expected and accepted (FUTURE_WORK.md §D1); it is a fidelity
            property, not a validity one
  D5        share of sectors ever over capacity. An ATFCM instance is *meant* to
            exceed capacity -- that imbalance is the problem a solver resolves --
            so a zero here is a hint the instance is trivial, not an error. It is
            also exactly what a PCAP100 overlay is supposed to show: nominal
            capacity is by definition the smallest capacity with no overload
  P10-present  how many instances carry navaid_sector_schedule.csv. Where it is
            absent the static allocation holds for the whole window
  D8-legs   mean legs per airframe. It falls as TG coarsens, because one route
            eats most of a 24 h window and the rotation has nowhere to continue;
            that is a property of the instance family, not a defect

What is deliberately NOT checked here
-------------------------------------
Anything needing the unparsed tree or the fitted model (the demand and OD audits,
the airport-set and timezone checks ``C1``–``C3``, ``C5``–``C8``), and the
capacity-sweep overlay checks in ``04_capacity_sweep_check.py``, which validate a
separate artefact produced by ``06_capacity_sweep.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REQUIRED_FILES = (
    "airplane_flight_assignment.csv",
    "airplanes.csv",
    "airports.csv",
    "flights.csv",
    "graph_edges.csv",
    "navaid_sector_assignment.csv",
    "sectors.csv",
)

#: The sector allocation with an explicit time axis. Optional, and so not in
#: REQUIRED_FILES: every instance published before stage 03 wrote it lacks it and is
#: still valid, with its static allocation holding for the whole window.
SCHEDULE_FILE = "navaid_sector_schedule.csv"
SCHEDULE_COLUMNS = ("Navaid_ID", "Sector_ID", "From_Time")

DATASET_RE = re.compile(r"^(\d+)_SEED(\w+)$")


# --------------------------------------------------------------------------
# result collection
# --------------------------------------------------------------------------

class Report:
    """Violations and informational statistics for one run."""

    def __init__(self) -> None:
        self.violations: list[str] = []
        self.notes: list[tuple[str, str, float]] = []   # (kind, dataset, value)
        self.datasets = 0

    def check(self, ctx: str, ident: str, ok: bool, detail: str = "") -> bool:
        if not ok:
            self.violations.append(f"{ctx}: {ident}" + (f" -- {detail}" if detail else ""))
        return ok

    def note(self, kind: str, ctx: str, value: float) -> None:
        self.notes.append((kind, ctx, value))


# --------------------------------------------------------------------------
# time granularity
# --------------------------------------------------------------------------

def resolve_tg(ds: Path, override: int | None) -> tuple[int, str]:
    """Return (time granularity, where it came from).

    Order: an explicit ``--time-granularity``; the generator manifest of the
    experiment this instance was transformed from, if that directory is still on
    this machine; a ``TG<n>`` in the path; otherwise 1.
    """
    if override is not None:
        return override, "--time-granularity"

    tm = ds / "transform_manifest.json"
    if tm.exists():
        try:
            src = json.load(open(tm)).get("source_experiment")
            if src:
                man = Path(src) / "manifest.json"
                if man.exists():
                    tg = json.load(open(man)).get("parameters", {}).get("time_granularity")
                    if tg is not None:
                        return int(tg), f"{man}"
        except (OSError, ValueError, KeyError, TypeError):
            pass

    m = re.search(r"TG(\d+)", str(ds.resolve()))
    if m:
        return int(m.group(1)), "TG<n> in the path"
    return 1, "default (no TG found -- pass --time-granularity if that is wrong)"


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------

def check_schedule(ctx: str, sch: pd.DataFrame, sec: pd.DataFrame, sc: pd.DataFrame,
                   verts: set, window: int, rep: Report) -> None:
    """P10: the sector schedule is a well-formed allocation over [0, window).

    The rows are change-points -- "from ``From_Time`` on, until this navpoint's next
    row". So a navpoint has exactly one sector at EVERY timestep of the window when it
    has a row at 0 (nothing before its first change-point is left uncovered) and no two
    rows at one ``From_Time`` (nothing is covered twice). P10b and P10c are those two
    halves; together they are the per-timestep statement without expanding the dense
    navpoint x timestep matrix, which for USA-MAINLAND at TG=60 is 28M cells.
    """
    missing_cols = [c for c in SCHEDULE_COLUMNS if c not in sch.columns]
    if not rep.check(ctx, "P10 schedule has columns Navaid_ID,Sector_ID,From_Time",
                     not missing_cols, f"missing {','.join(missing_cols)}"):
        return

    t = pd.to_numeric(sch["From_Time"], errors="coerce")
    bad_t = t.isna() | (t != t.round()) | (t < 0) | (t >= window)
    rep.check(ctx, "P10a schedule From_Time is a whole timestep in [0, TG*24)",
              not bool(bad_t.any()),
              f"{int(bad_t.sum())} rows outside [0, {window}), "
              f"e.g. {sch.loc[bad_t, 'From_Time'].head(3).tolist()}")

    dup = sch.duplicated(subset=["Navaid_ID", "From_Time"], keep=False)
    rep.check(ctx, "P10b schedule has no duplicate (Navaid_ID, From_Time)",
              not bool(dup.any()),
              f"{int(dup.sum())} rows share a (Navaid_ID, From_Time), "
              f"e.g. {sch.loc[dup, ['Navaid_ID', 'From_Time']].head(3).values.tolist()}")

    # Every navpoint the instance knows -- the graph's and the static file's -- plus any
    # the schedule names, must be allocated from t=0.
    navpoints = set(verts) | set(sec["Navaid_ID"]) | set(sch["Navaid_ID"])
    at_zero = set(sch.loc[t == 0, "Navaid_ID"])
    uncovered = navpoints - at_zero
    rep.check(ctx, "P10c schedule gives every navpoint exactly one sector at every timestep",
              not uncovered and not bool(dup.any()),
              f"{len(uncovered)} navpoints have no sector from t=0"
              + (f" (e.g. {sorted(uncovered)[:3]})" if uncovered else "")
              + (f"; {int(dup.sum())} rows cover a timestep twice" if dup.any() else ""))

    undeclared = set(sch["Sector_ID"]) - set(sc["Sector_ID"])
    rep.check(ctx, "P10d every sector the schedule uses is declared",
              not undeclared, f"{len(undeclared)} undeclared")

    # At t=0 the schedule IS the static allocation. Compare as mappings, so row order
    # does not matter but every navpoint and every sector does.
    static = dict(zip(sec["Navaid_ID"], sec["Sector_ID"]))
    first = sch[t == 0].drop_duplicates(subset=["Navaid_ID"])
    initial = dict(zip(first["Navaid_ID"], first["Sector_ID"]))
    differ = {n for n in static.keys() | initial.keys() if static.get(n) != initial.get(n)}
    rep.check(ctx, "P10e schedule at t=0 agrees with navaid_sector_assignment.csv",
              not differ,
              f"{len(differ)} navpoints differ, e.g. "
              + ", ".join(f"{n}: {static.get(n)} vs {initial.get(n)}"
                          for n in sorted(differ, key=str)[:3]))


def check_dataset(ds: Path, tg: int, rep: Report, ctx: str | None = None) -> int:
    """Run every check on one instance directory. Returns the violation count."""
    ctx = ctx or f"{ds.parent.name}/{ds.name}"
    before = len(rep.violations)
    rep.datasets += 1

    missing = [n for n in REQUIRED_FILES if not (ds / n).exists()]
    if not rep.check(ctx, "P1 all required files present", not missing, ",".join(missing)):
        return len(rep.violations) - before

    fl = pd.read_csv(ds / "flights.csv")
    ed = pd.read_csv(ds / "graph_edges.csv")
    sec = pd.read_csv(ds / "navaid_sector_assignment.csv")
    sc = pd.read_csv(ds / "sectors.csv")
    apt = set(pd.read_csv(ds / "airports.csv")["Airport_Vertex"])
    af = pd.read_csv(ds / "airplane_flight_assignment.csv")
    apl = pd.read_csv(ds / "airplanes.csv")

    n_flights = fl["Flight_ID"].nunique()
    m = DATASET_RE.match(ds.name)
    declared = int(m.group(1)) if m else None
    if declared is not None:
        rep.check(ctx, "P2/F9 flight count matches the directory name",
                  n_flights == declared, f"{n_flights} flights, directory says {declared}")

    # --- the window ------------------------------------------------------
    window = tg * 24
    tmin, tmax = int(fl["Time"].min()), int(fl["Time"].max())
    rep.check(ctx, "P3/F1 every landing within the window", tmax <= window,
              f"max Time {tmax} > TG*24 = {window}")
    starts = fl.groupby("Flight_ID")["Time"].min()
    neg = int((starts < 0).sum())
    rep.check(ctx, "P3b/F1 no departure before t=0", neg == 0,
              f"{neg} flights start before t=0, earliest t={tmin}")

    g = fl.sort_values(["Flight_ID", "Time"])
    same = g["Flight_ID"] == g["Flight_ID"].shift()
    dt = g["Time"].diff()
    if bool(same.any()):
        gap = float(dt[same].max())
        rep.check(ctx, "F3 no waypoint gap larger than the window", gap <= window,
                  f"largest gap {gap:.0f} > {window}")

    ends = fl.groupby("Flight_ID")["Time"].max()
    rep.note("clamp", ctx, float((ends >= window).mean()))

    # --- sectors and capacities -----------------------------------------
    min_cap = int(sc["Capacity"].min())
    rep.check(ctx, "P4/F6 every capacity >= 1", min_cap >= 1, f"min capacity {min_cap}")

    verts = set(ed["source"]) | set(ed["target"])
    assigned = set(sec["Navaid_ID"])
    rep.check(ctx, "P5/F6 every graph vertex has a sector", verts <= assigned,
              f"{len(verts - assigned)} vertices unassigned")
    undeclared = set(sec["Sector_ID"]) - set(sc["Sector_ID"])
    rep.check(ctx, "P5/D6/F6 every sector used is declared", not undeclared,
              f"{len(undeclared)} undeclared")

    # --- the sector allocation over time (optional file) -----------------
    has_schedule = (ds / SCHEDULE_FILE).exists()
    rep.note("schedule", ctx, 1.0 if has_schedule else 0.0)
    if has_schedule:
        check_schedule(ctx, pd.read_csv(ds / SCHEDULE_FILE), sec, sc, verts, window, rep)

    # --- trajectories ----------------------------------------------------
    bad_t = int(((dt <= 0) & same).sum())
    rep.check(ctx, "P6/F2 Time strictly increasing per flight", bad_t == 0,
              f"{bad_t} non-increasing timesteps")
    dup = int(g.duplicated(subset=["Flight_ID", "Time"]).sum())
    rep.check(ctx, "D2 one position per (flight, timestep)", dup == 0, f"{dup} duplicates")

    unknown = set(fl["Position"]) - assigned
    rep.check(ctx, "P6b/F6 every position is a known navaid", not unknown,
              f"{len(unknown)} unknown positions")
    rep.check(ctx, "P7 every airport vertex is on the graph", apt <= verts,
              f"{len(apt - verts)} airports off the graph")

    adj = set(zip(ed["source"], ed["target"])) | set(zip(ed["target"], ed["source"]))
    prev_p = g["Position"].shift()
    steps = zip(prev_p[same].astype(int), g["Position"][same].astype(int))
    illegal = [(a, b) for a, b in steps if a != b and (a, b) not in adj]
    rep.check(ctx, "D1/F8 consecutive positions are graph-adjacent", not illegal,
              f"{len(illegal)} non-adjacent hops, e.g. {illegal[:3]}")

    endpoints = g.groupby("Flight_ID")["Position"].agg(["first", "last"])
    off = int((~endpoints["first"].isin(apt)).sum() + (~endpoints["last"].isin(apt)).sum())
    rep.check(ctx, "D3/F7 flights start and end at airports", off == 0,
              f"{off} endpoints are not airports")
    loops = int((endpoints["first"] == endpoints["last"]).sum())
    rep.check(ctx, "C4 no self-loop flights", loops == 0,
              f"{loops} flights return to their origin airport")

    # --- airframes -------------------------------------------------------
    counts = af["Flight_ID"].value_counts()
    rep.check(ctx, "P8/F5 every flight has exactly one airplane",
              set(af["Flight_ID"]) == set(fl["Flight_ID"]) and int(counts.max()) == 1,
              f"assigned={af['Flight_ID'].nunique()} flights={n_flights} "
              f"maxdup={int(counts.max())}")
    undeclared_ac = set(af["Airplane_ID"]) - set(apl[apl.columns[0]])
    rep.check(ctx, "P8/F5 every airplane used is declared", not undeclared_ac,
              f"{len(undeclared_ac)} undeclared airplanes")

    # A leg must start at least one whole timestep after the previous one ends
    # (arrive t=5, depart t>=6). Sharing the boundary slot puts the airframe at
    # two navpoints in one timestep and double-counts it in that slot.
    span = fl.groupby("Flight_ID")["Time"].agg(["min", "max"])
    joined = af.set_index("Flight_ID").join(span)
    overlap = boundary = 0
    for _, grp in joined.groupby("Airplane_ID"):
        s = grp.sort_values("min")
        nxt, prv = s["min"].values[1:], s["max"].values[:-1]
        overlap += int((nxt < prv).sum())
        boundary += int((nxt == prv).sum())
    rep.check(ctx, "D4/F4 legs of one airframe are >= 1 timestep apart",
              overlap + boundary == 0,
              f"{overlap} overlapping, {boundary} sharing the boundary timestep")

    # Rotation continuity. The airframe that lands at an airport is the airframe that
    # departs from it: leg k+1 must start where leg k finished. Stage 01 builds the
    # rotation that way (an aircraft is only drawn from the pool held at its current
    # airport); nothing downstream may contradict it. The span above gives the order,
    # `endpoints` the two airports, so this costs one join and one comparison.
    joined_p = joined.join(endpoints)
    broken = pairs = 0
    for _, grp in joined_p.groupby("Airplane_ID"):
        s = grp.sort_values("min")
        if len(s) < 2:
            continue
        arrive, depart = s["last"].values[:-1], s["first"].values[1:]
        pairs += len(depart)
        broken += int((arrive != depart).sum())
    rep.note("legs", ctx, len(joined_p) / max(1, joined_p["Airplane_ID"].nunique()))
    rep.check(ctx, "D8/F10 each leg departs where the airframe last landed",
              broken == 0,
              f"{broken} of {pairs} consecutive leg pairs start at a different airport "
              f"from the one the airframe landed at")

    # --- demand-capacity imbalance (informational) -----------------------
    sector_of = dict(zip(sec["Navaid_ID"], sec["Sector_ID"]))
    occ: dict[tuple[object, int], int] = defaultdict(int)
    for pos, t in zip(fl["Position"], fl["Time"]):
        occ[(sector_of.get(pos), int(t))] += 1
    peak: dict[object, int] = defaultdict(int)
    for (s, _t), v in occ.items():
        if s is not None and v > peak[s]:
            peak[s] = v
    cap = dict(zip(sc["Sector_ID"], sc["Capacity"]))
    over = sum(1 for s, v in peak.items() if cap.get(s) is not None and v > cap[s])
    rep.note("imbalance", ctx, over / max(1, len(peak)))

    # --- manifest --------------------------------------------------------
    tm = ds / "transform_manifest.json"
    if tm.exists() and m is not None:
        man = json.load(open(tm))
        rep.check(ctx, "P9 manifest agrees with the directory name",
                  int(man.get("n_flights", -1)) == declared
                  and str(man.get("seed")) == m.group(2),
                  f"manifest n_flights={man.get('n_flights')} seed={man.get('seed')}")

    return len(rep.violations) - before


# --------------------------------------------------------------------------
# walking a path
# --------------------------------------------------------------------------

def find_datasets(path: Path) -> list[Path]:
    """Every instance directory at or under ``path``, however deeply nested.

    An instance directory is one holding ``flights.csv``. The walk does not
    descend into one, so the depth of the tree does not matter: an instance
    directory, an experiment, a parsed root, and a release tree with a
    ``PCAP<nn>/`` capacity level between the experiment and the instance all
    work the same way.
    """
    if (path / "flights.csv").exists():
        return [path]
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(path):
        if "flights.csv" in filenames:
            out.append(Path(dirpath))
            dirnames[:] = []
            continue
        dirnames.sort()
    return sorted(out)


def label(ds: Path, root: Path) -> str:
    """Name an instance relative to the root the user asked about."""
    try:
        rel = ds.resolve().relative_to(root.resolve())
    except ValueError:
        return f"{ds.parent.name}/{ds.name}"
    return str(rel) if str(rel) != "." else ds.name


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check that generated instances are valid: window, adjacency, "
                    "airport endpoints, aircraft separation, sector cover, capacities.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit 0 = all checks passed, 1 = violations found, 2 = usage error.")
    ap.add_argument("path", nargs="+", type=Path,
                    help="an instance directory, an experiment directory, or a parsed root")
    ap.add_argument("--time-granularity", type=int, default=None,
                    help="bins per hour, i.e. the --time-granularity the instance was "
                         "generated with. The 24 h window is TG*24 timesteps. Auto-detected "
                         "from the generator manifest or from a TG<n> in the path if omitted.")
    ap.add_argument("--verbose", action="store_true",
                    help="list every dataset, not only the ones with violations")
    ap.add_argument("--max-report", type=int, default=30,
                    help="how many violations to print (default 30)")
    a = ap.parse_args()

    rep = Report()
    any_found = False

    for root in a.path:
        root = root.expanduser()
        if not root.is_dir():
            print(f"not a directory: {root}", file=sys.stderr)
            return 2
        datasets = find_datasets(root)
        if not datasets:
            print(f"no instance directories found under {root}", file=sys.stderr)
            print("  (expected directories containing flights.csv, as written by "
                  "05_transform_for_optimizer.py)", file=sys.stderr)
            return 2
        any_found = True

        tg, source = resolve_tg(datasets[0], a.time_granularity)
        print(f"{root}")
        print(f"  time granularity : TG={tg}  (window = {tg * 24} timesteps, from {source})")
        print(f"  instances        : {len(datasets)}")

        by_experiment: dict[str, int] = defaultdict(int)
        for ds in datasets:
            name = label(ds, root)
            n = check_dataset(ds, tg, rep, name)
            # group by the directory holding the instance; when the user pointed
            # straight at one instance there is no group, so use its own name
            group = str(Path(name).parent)
            by_experiment[name if group == "." else group] += n
            if a.verbose:
                print(f"    {'FAIL' if n else 'PASS'}  {name}"
                      + (f"  {n} violations" if n else ""))
        if not a.verbose:
            for exp in sorted(by_experiment):
                n = by_experiment[exp]
                print(f"    {'FAIL' if n else 'PASS'}  {exp}"
                      + (f"  {n} violations" if n else ""))
        print()

    if not any_found:
        return 2

    clamp = [v for k, _c, v in rep.notes if k == "clamp"]
    imbal = [v for k, _c, v in rep.notes if k == "imbalance"]
    legs = [v for k, _c, v in rep.notes if k == "legs"]
    sched = [v for k, _c, v in rep.notes if k == "schedule"]
    print("--- reported, not failed -------------------------------------------")
    if clamp:
        print(f"  P3-clamp  flights ending on the window edge: mean {sum(clamp)/len(clamp):.1%} "
              f"[{min(clamp):.1%}-{max(clamp):.1%}]")
    if imbal:
        print(f"  D5        sectors ever over capacity:        mean {sum(imbal)/len(imbal):.1%} "
              f"[{min(imbal):.1%}-{max(imbal):.1%}]")
        if max(imbal) == 0:
            print("            no instance is ever over capacity: trivially feasible, unless "
                  "these are")
            print("            PCAP100 overlays, where zero overload is the definition of "
                  "nominal capacity")
    if sched:
        print(f"  P10-present  {SCHEDULE_FILE}: in {int(sum(sched))} of {len(sched)} "
              f"instance(s)" + ("" if all(sched) else
                               "; where absent, the static allocation holds all day"))
    if legs:
        print(f"  D8-legs   legs per airframe:                   mean {sum(legs)/len(legs):.3f} "
              f"[{min(legs):.3f}-{max(legs):.3f}]")
    print()

    if rep.violations:
        print(f"FAILED: {len(rep.violations)} violation(s) in {rep.datasets} instance(s)")
        for v in rep.violations[:a.max_report]:
            print(f"  - {v}")
        if len(rep.violations) > a.max_report:
            print(f"  ... and {len(rep.violations) - a.max_report} more")
        return 1

    print(f"ALL CHECKS PASSED: {rep.datasets} instance(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
