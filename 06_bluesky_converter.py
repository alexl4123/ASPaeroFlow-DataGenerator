#!/usr/bin/env python3
"""
Convert one DATA_* sample into a BlueSky scenario (flight-only, no sectors).

Inputs (required)
-----------------
--data-dir        : path to DATA_S*_* folder containing:
                    - filed_flights.csv  (Flight_ID,Position,Time)
                    - flights.csv        (flight_id, aircraft_id, origin, destination, departure_time)
                    - aircrafts.csv|aircraft.csv (aircraft_id, speed_kts)
--navgraph-dir    : path to navgraph folder containing:
                    - vertices.csv       (IDENTIFIER, LAT, LON, ...)

Time conversion
---------------
BlueSky timestamps need seconds. filed_flights 'Time' is in slots.
We derive slot_seconds with this priority:
  1) --slot-seconds (explicit seconds per slot)
  2) --time-granularity  (slot_seconds = 3600 / G)
  3) <exp_root>/manifest.json → parameters.time_granularity
  4) default 1.0

Output
------
Writes a .scn scenario text file. By default to:
  <data-dir>/bluesky/<data_basename>.scn

Navpoint names are prefixed (default 'MY_') to avoid collisions.

Example
-------
python 06_to_bluesky_scn.py \
  --data-dir unparsed_experiment_data/DACH-2019-06-15/DATA_S0p5_42 \
  --navgraph-dir unparsed_experiment_data/DACH-2019-06-15/navgraph \
  --time-granularity 60
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
import sys
from collections import defaultdict

# ----------------------------- helpers -----------------------------

def _find_aircrafts_csv(d: Path) -> Path:
    for name in ("aircrafts.csv", "aircraft.csv"):
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No aircrafts.csv / aircraft.csv in {d}")

def _read_vertices(navgraph_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Tuple[float,float]], Dict[int, str]]:
    vpath = navgraph_dir / "vertices.csv"
    if not vpath.exists():
        raise FileNotFoundError(f"vertices.csv not found: {vpath}")
    vdf = pd.read_csv(vpath)
    if not {"IDENTIFIER","LAT","LON"} <= set(vdf.columns):
        raise ValueError("vertices.csv must contain IDENTIFIER, LAT, LON.")
    id2ll: Dict[str, Tuple[float,float]] = {}
    vid2id: Dict[int, str] = {}
    for i, row in vdf.iterrows():
        ident = str(row["IDENTIFIER"]).strip().upper()
        lat = float(row["LAT"]); lon = float(row["LON"])
        id2ll[ident] = (lat, lon)
        vid2id[i] = ident
    return vdf, id2ll, vid2id

def _load_filed(data_dir: Path) -> pd.DataFrame:
    f = data_dir / "filed_flights.csv"
    if not f.exists():
        raise FileNotFoundError(f"filed_flights.csv not found: {f}")
    df = pd.read_csv(f)
    # normalize column names tolerantly
    cmap = {c.lower(): c for c in df.columns}
    need = {"flight_id","position","time"}
    if not need <= set(cmap.keys()):
        raise ValueError("filed_flights.csv must have Flight_ID/flight_id, Position, Time")
    df = df.rename(columns={
        cmap.get("flight_id","Flight_ID"): "Flight_ID",
        cmap.get("position","Position"):   "Position",
        cmap.get("time","Time"):           "Time",
    })
    return df

def _load_flights_and_speeds(data_dir: Path) -> Tuple[pd.DataFrame, Dict[str, float]]:
    # flights.csv: (flight_id, aircraft_id, origin, destination, departure_time)
    f = data_dir / "flights.csv"
    if not f.exists():
        raise FileNotFoundError(f"flights.csv not found: {f}")
    flights = pd.read_csv(f)
    fmap = {c.lower(): c for c in flights.columns}
    need = {"flight_id","aircraft_id","origin","destination"}
    if not need <= set(fmap.keys()):
        raise ValueError("flights.csv missing required columns.")
    flights = flights.rename(columns={
        fmap["flight_id"]:   "flight_id",
        fmap["aircraft_id"]: "aircraft_id",
        fmap["origin"]:      "origin",
        fmap["destination"]: "destination",
    })
    flights["flight_id"]   = flights["flight_id"].astype(str)
    flights["aircraft_id"] = flights["aircraft_id"].astype(str)
    flights["origin"]      = flights["origin"].astype(str).str.strip().str.upper()
    flights["destination"] = flights["destination"].astype(str).str.strip().str.upper()

    # aircrafts
    acp = _find_aircrafts_csv(data_dir)
    ac = pd.read_csv(acp)
    amap = {c.lower(): c for c in ac.columns}
    if not {"aircraft_id","speed_kts"} <= set(amap.keys()):
        raise ValueError("aircrafts.csv must contain aircraft_id and speed_kts.")
    ac = ac.rename(columns={amap["aircraft_id"]: "aircraft_id", amap["speed_kts"]: "speed_kts"})
    ac["aircraft_id"] = ac["aircraft_id"].astype(str)
    speed = dict(zip(ac["aircraft_id"], ac["speed_kts"].astype(float)))
    return flights, speed

def _safe_int(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    vals = pd.to_numeric(s, errors="coerce")
    mask = vals.notna()
    return vals.fillna(-1).astype(int), mask

def _fmt_ts(seconds: float) -> str:
    if seconds < 0: seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - (h*3600 + m*60)
    # BlueSky shows 2 decimals often; keep fixed 2 decimals
    return f"{h}:{m:02d}:{s:05.2f}"

# ----------------------------- main logic -----------------------------

def build_scenario_lines(
    filed: pd.DataFrame,
    flights: pd.DataFrame,
    speed_kts: Dict[str,float],
    id2ll: Dict[str, Tuple[float,float]],
    vid2id: Dict[int,str],
    slot_seconds: float,
    nav_prefix: str = "MY_",
    ac_type: str = "321",
    default_speed_kts: float = 450.0,
    default_heading_deg: float = 0.0,
    default_flightlevel: str = "FL100",
) -> List[str]:

    # Normalize Position → IDENTIFIER strings (if numeric, map via vid2id)
    pos_ints, pos_isnum = _safe_int(filed["Position"])
    if pos_isnum.all():
        # map each numeric vertex id to IDENTIFIER
        filed["Position"] = pos_ints.map(lambda i: vid2id.get(int(i), str(int(i)))).astype(str)
    else:
        filed["Position"] = filed["Position"].astype(str)

    # Uppercase and strip for lookups
    filed["Position"] = filed["Position"].str.strip().str.upper()
    flights["flight_id"] = flights["flight_id"].astype(str)

    # Build per-flight ordered route (drop consecutive duplicates)
    filed["Time"] = pd.to_numeric(filed["Time"], errors="raise")
    filed = filed.sort_values(["Flight_ID","Time"]).reset_index(drop=True)

    routes: Dict[str, List[Tuple[str, float]]] = defaultdict(list)  # flight_id -> [(IDENT, time_slot), ...]
    for fid, g in filed.groupby("Flight_ID", sort=False):
        prev = None
        for _, row in g.iterrows():
            ident = row["Position"]
            if ident != prev:
                routes[fid].append((ident, float(row["Time"])))
                prev = ident

    # Derive set of required navpoints
    required = set()
    for fid, path in routes.items():
        for ident, _ in path:
            required.add(ident)

    # Lines
    lines: List[str] = []

    # Header comment
    lines.append(f"# Generated BlueSky scenario")
    lines.append(f"# Flights: {len(routes)} | SlotSeconds={slot_seconds:.6g} | Navpoints: {len(required)}")
    lines.append(f"# All navpoints defined at t=0 with prefix '{nav_prefix}'\n")

    # 1) Define needed navpoints at t=0
    t0 = _fmt_ts(0.0)
    missing_coords = []
    for ident in sorted(required):
        ll = id2ll.get(ident)
        if ll is None:
            missing_coords.append(ident)
            continue
        lat, lon = ll
        # 0:00:00.00>MY_IDENT,lat,lon,0.0,1
        lines.append(f"{t0}>DEFWPT {nav_prefix}{ident},{lat:.6f},{lon:.6f},FIX")
    if missing_coords:
        print(f"[WARN] {len(missing_coords)} identifiers have no coordinates in vertices.csv "
              f"(examples: {', '.join(missing_coords[:10])})", file=sys.stderr)

    # Build lookups for flights → aircraft speed, origin/destination
    fl_map = flights.set_index("flight_id")[["aircraft_id","origin","destination"]].to_dict(orient="index")

    # 2) For each flight: create & route
    for fid, path in routes.items():
        meta = fl_map.get(str(fid))
        if meta is None:
            print(f"[WARN] Flight '{fid}' not in flights.csv; skipping.", file=sys.stderr)
            continue

        acid = str(meta["aircraft_id"])
        spd = float(speed_kts.get(acid, default_speed_kts))

        # start waypoint & time
        first_wpt, start_slot = path[0]
        start_ts = start_slot * slot_seconds
        start_t = _fmt_ts(start_ts)

        # coords for start waypoint
        ll = id2ll.get(first_wpt)
        if ll is None:
            print(f"[WARN] Missing coords for first waypoint '{first_wpt}' of {fid}; skipping flight.", file=sys.stderr)
            continue
        lat, lon = ll

        # CRE <callsign> <type> <lat> <lon> <hdg> <alt> <spd>
        lines.append(f"{start_t}>CRE {fid} {ac_type} {lat:.6f} {lon:.6f} {int(default_heading_deg)} {default_flightlevel} {int(round(spd))}")

        # Add remaining waypoints (at creation time)
        for ident, _slot in path[0:]:
            lines.append(f"{start_t}>ADDWPT {fid} {nav_prefix}{ident}")

        # Schedule deletion at destination waypoint arrival
        dest_ident = path[-1][0]
        lines.append(f"{start_t}>{fid} AT {nav_prefix}{dest_ident} DO DEL {fid}")

    return lines

# ----------------------------- CLI -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BlueSky scenario from pipeline DATA sample.")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="DATA_S*_* directory with filed_flights.csv, flights.csv, aircrafts.csv")
    p.add_argument("--navgraph-dir", type=Path, required=True,
                   help="navgraph directory with vertices.csv")
    p.add_argument("--out", type=Path, default=None,
                   help="Output .scn file path (default: <data-dir>/bluesky/<data_basename>.scn)")
    # timing
    p.add_argument("--slot-seconds", type=float, default=None,
                   help="Seconds per timeslot in filed_flights.csv (overrides everything).")
    p.add_argument("--time-granularity", type=int, default=None,
                   help="G where slot_seconds = 3600 / G (e.g., G=60 ⇒ 60s/slot).")
    # aircraft defaults
    p.add_argument("--navpoint-prefix", type=str, default="MY_")
    p.add_argument("--aircraft-type", type=str, default="321")
    p.add_argument("--default-speed-kts", type=float, default=450.0)
    p.add_argument("--default-heading", type=float, default=0.0)
    p.add_argument("--default-flightlevel", type=str, default="FL100")
    return p.parse_args()

def _infer_slot_seconds(args, data_dir: Path) -> float:
    # Priority: --slot-seconds > --time-granularity > manifest.json > 1.0
    if args.slot_seconds is not None:
        return float(args.slot_seconds)
    if args.time_granularity is not None and args.time_granularity > 0:
        return 3600.0 / float(args.time_granularity)
    # Look for <exp_root>/manifest.json (parent of DATA_* and navgraph)
    exp_root = data_dir.parent
    mpath = exp_root / "manifest.json"
    if mpath.exists():
        try:
            with open(mpath, "r") as fh:
                man = json.load(fh) or {}
            tg = man.get("parameters", {}).get("time_granularity", None)
            if tg:
                tg = int(tg)
                if tg > 0:
                    return 3600.0 / float(tg)
        except Exception:
            pass
    return 1.0

def main():
    a = parse_args()

    if not a.data_dir.exists():
        raise FileNotFoundError(f"data-dir not found: {a.data_dir}")
    if not a.navgraph_dir.exists():
        raise FileNotFoundError(f"navgraph-dir not found: {a.navgraph_dir}")

    # Load inputs
    vdf, id2ll, vid2id = _read_vertices(a.navgraph_dir)
    filed = _load_filed(a.data_dir)
    flights, speed = _load_flights_and_speeds(a.data_dir)

    slot_seconds = _infer_slot_seconds(a, a.data_dir)

    lines = build_scenario_lines(
        filed=filed,
        flights=flights,
        speed_kts=speed,
        id2ll=id2ll,
        vid2id=vid2id,
        slot_seconds=slot_seconds,
        nav_prefix=a.navpoint_prefix,
        ac_type=a.aircraft_type,
        default_speed_kts=a.default_speed_kts,
        default_heading_deg=a.default_heading,
        default_flightlevel=a.default_flightlevel,
    )

    # Output path default
    out_path = a.out
    if out_path is None:
        out_dir = a.data_dir / "bluesky"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{a.data_dir.name}.scn"

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[✓] Wrote BlueSky scenario → {out_path.resolve()}")

if __name__ == "__main__":
    main()

