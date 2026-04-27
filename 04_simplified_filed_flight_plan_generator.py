#!/usr/bin/env python3
"""
Filed flight plan generator (shortest routes on navgraph)

Inputs
------
--data-dir       : path to folder containing flights.csv and aircrafts.csv
--navgraph-dir   : path to folder containing vertices.csv and edges.csv
--time-granularity : integer G (default 4 -> 15 min slots since 3600/G seconds)
                     (G=1 => 1 hour slots; G=4 => 15-min; G=6 => 10-min; etc.)

Behavior
--------
- For each flight (origin ICAO → destination ICAO), find the shortest path on
  the navgraph using edge weights equal to the number of slots needed to
  traverse the edge at that flight's aircraft speed.
- Departure time is discretized to slots:
    slot_seconds = 3600 / G
    start_slot   = floor(seconds_since_UTC_midnight(departure_time) / slot_seconds)
- Edge duration in slots:
    speed_ms = speed_kts * 0.51444
    duration_seconds = distance_m / speed_ms
    duration_slots = ceil(duration_seconds / slot_seconds), with min 1.

Output
------
<data-dir>/filed_flights.csv with columns:
  Flight_ID,Position,Time
where Position is the vertex IDENTIFIER from vertices.csv (e.g., ICAO or fix/nav id).
If IDENTIFIER is unavailable, we fall back to the numeric vertex index.
 

Notes
-----
- vertices.csv rows are aligned with numeric vertex IDs (0..N-1), which edges.csv uses.
- We map those numeric IDs back to IDENTIFIER strings for the output.
- edges.csv can be (V0,V1,D) or (source,target,dist_m) or any case-variant.
- aircrafts file may be named 'aircrafts.csv' or 'aircraft.csv'.
- Progress bar uses tqdm if available; falls back to periodic prints.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List
from math import ceil
import sys
import time
import numpy as np
import pandas as pd
import networkx as nx


# -------------------------
# I/O helpers
# -------------------------
def _find_aircrafts_csv(data_dir: Path) -> Path:
    cands = [data_dir / "aircrafts.csv", data_dir / "aircraft.csv"]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"aircrafts.csv / aircraft.csv not found in {data_dir}")

def _normalize_edges_df(edf: pd.DataFrame) -> pd.DataFrame:
    low = {c.lower(): c for c in edf.columns}
    # map columns to (u,v,distance_m)
    if {"v0","v1","d"}.issubset(low.keys()):
        edf = edf.rename(columns={low["v0"]:"u", low["v1"]:"v", low["d"]:"dist_m"})
    elif {"source","target","dist_m"}.issubset(low.keys()):
        edf = edf.rename(columns={low["source"]:"u", low["target"]:"v", low["dist_m"]:"dist_m"})
    elif {"source","target","d"}.issubset(low.keys()):
        edf = edf.rename(columns={low["source"]:"u", low["target"]:"v", low["d"]:"dist_m"})
    else:
        # fallback: assume first two are endpoints, last numeric is distance
        cols = list(edf.columns)
        if len(cols) < 3:
            raise ValueError("edges.csv must have ≥3 columns (u,v,dist_m).")
        edf = edf.rename(columns={cols[0]:"u", cols[1]:"v", cols[2]:"dist_m"})
    return edf[["u","v","dist_m"]]

def _load_navgraph(navgraph_dir: Path) -> Tuple[nx.Graph, Dict[str,int], List[str] | None, bool]:
    vpath = navgraph_dir / "vertices.csv"
    epath = navgraph_dir / "edges.csv"
    if not vpath.exists(): raise FileNotFoundError(f"vertices.csv not found: {vpath}")
    if not epath.exists(): raise FileNotFoundError(f"edges.csv not found: {epath}")

    vdf = pd.read_csv(vpath)
    # Build IDENTIFIER -> vertex_id map from rows (row index == vertex id)
    ident_to_vid: Dict[str,int] = {}
    vid_to_ident: List[str] | None = None
    if "IDENTIFIER" in vdf.columns:
        idents = vdf["IDENTIFIER"].astype(str).str.strip().str.upper().tolist()
        ident_to_vid = {s: i for i, s in enumerate(idents)}
        # Keep original strings (before upper) for nicer output? We’ll output upper to be consistent.
        vid_to_ident = idents
    else:
        # If IDENTIFIER missing, we cannot map ICAOs; still load graph for debug use
        print("[WARN] vertices.csv has no IDENTIFIER column; OD mapping may fail.", file=sys.stderr)
        vid_to_ident = None
 
    edf = pd.read_csv(epath)
    edf = _normalize_edges_df(edf)

    # Determine node type: numeric vertex ids vs IDENTIFIER strings
    # Try numeric conversion; if any NaNs appear, treat as strings.
    u_num = pd.to_numeric(edf["u"], errors="coerce")
    v_num = pd.to_numeric(edf["v"], errors="coerce")
    nodes_are_int = not (u_num.isna().any() or v_num.isna().any())

    G = nx.Graph()
    if nodes_are_int:
        u = u_num.astype("int64").to_numpy()
        v = v_num.astype("int64").to_numpy()
        d = edf["dist_m"].astype(float).to_numpy()
        G.add_weighted_edges_from(zip(u, v, d), weight="dist_m")
    else:
        u = edf["u"].astype(str).str.strip().str.upper().to_numpy()
        v = edf["v"].astype(str).str.strip().str.upper().to_numpy()
        d = edf["dist_m"].astype(float).to_numpy()
        G.add_weighted_edges_from(zip(u, v, d), weight="dist_m")
    return G, ident_to_vid, vid_to_ident, nodes_are_int

def _load_flights(data_dir: Path) -> pd.DataFrame:
    fpath = data_dir / "flights.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"flights.csv not found in {data_dir}")
    fdf = pd.read_csv(fpath)
    # tolerant column mapping
    cmap = {c.lower(): c for c in fdf.columns}
    need = ["flight_id","aircraft_id","origin","destination","departure_time"]
    missing = [n for n in need if n not in cmap]
    if missing:
        raise ValueError(f"flights.csv must contain columns {need}. Missing: {missing}")


    fdf["_lineno"] = np.arange(len(fdf), dtype=int) + 2

    # Clean strings
    for c in ("origin","destination","departure_time"):
        fdf[c] = fdf[c].astype(str).str.strip()

    # Flag obviously non-timestamp junk (helps debugging)
    junk_hint = fdf["departure_time"].str.contains(r"\[RUN\]|Traceback|^raise\b|python ", na=False)

    parsed_ts = pd.to_datetime(fdf["departure_time"], utc=True, errors="coerce", format="%Y-%m-%dT%H:%M:%S%z")
    bad = parsed_ts.isna()

    if bad.any():
        bad_rows = fdf.loc[bad, ["_lineno", "flight_id", "departure_time"]]

        print("\n[flights.csv] Invalid departure_time at file lines:")
        for _, r in bad_rows.iterrows():
            print(f"  line {int(r['_lineno'])}: flight_id={r['flight_id']!s} departure_time={r['departure_time']!r}")

        # Also print the raw lines from the file for certainty (what Vim shows is real)
        bad_line_nums = set(int(x) for x in bad_rows["_lineno"].tolist())
        print("\n[flights.csv] Raw file lines with issues:")
        with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, start=1):
                if i in bad_line_nums:
                    print(f"{i:>7}: {line.rstrip()}")

        # Optional: hint if junk-y patterns were seen
        if junk_hint.any():
            junk_lines = fdf.loc[junk_hint, "_lineno"].tolist()
            print(f"\n[hint] Some lines look like logs mixed into CSV. Example line(s): {junk_lines[:5]}")

        raise ValueError("Invalid timestamps in departure_time; see lines above.")


    fdf = fdf.rename(columns={cmap[n]: n for n in need})
    fdf["origin"] = fdf["origin"].astype(str).str.strip().str.upper()
    fdf["destination"] = fdf["destination"].astype(str).str.strip().str.upper()
    # parse ISO time (UTC)
    fdf["departure_time"] = pd.to_datetime(fdf["departure_time"], utc=True, errors="coerce", format="%Y-%m-%dT%H:%M:%S%z")
    if fdf["departure_time"].isna().any():
        raise ValueError("Invalid timestamps in departure_time.")
    return fdf

def _load_aircrafts(data_dir: Path) -> Dict[str, float]:
    apath = _find_aircrafts_csv(data_dir)
    adf = pd.read_csv(apath)
    cmap = {c.lower(): c for c in adf.columns}
    if "aircraft_id" not in cmap or "speed_kts" not in cmap:
        raise ValueError("aircraft(s).csv must contain columns aircraft_id and speed_kts.")
    adf = adf.rename(columns={cmap["aircraft_id"]: "aircraft_id", cmap["speed_kts"]: "speed_kts"})
    adf["aircraft_id"] = adf["aircraft_id"].astype(str)
    return dict(zip(adf["aircraft_id"], adf["speed_kts"].astype(float)))


# -------------------------
# Time/weight helpers (exact behavior as provided)
# -------------------------
def _slot_seconds(time_granularity: int) -> float:
    # factor_to_unit_standard = 3600 / time_granularity
    return 3600.0 / float(time_granularity)

def _edge_duration_slots(distance_m: float, speed_kts: float, time_granularity: int) -> int:
    # duration_in_seconds = distance / (speed_kts * 0.51444)
    speed_ms = float(speed_kts) * 0.51444
    if speed_ms <= 0:
        return 1  # defensive
    duration_seconds = float(distance_m) / speed_ms
    slot_sec = _slot_seconds(time_granularity)
    slots = int(ceil(duration_seconds / slot_sec))
    return max(slots, 1)

def _start_slot_from_timestamp(ts_utc: pd.Timestamp, time_granularity: int) -> int:
    # floor(seconds since UTC midnight / slot_seconds)
    midnight = ts_utc.normalize()  # keeps tz-aware UTC midnight
    seconds = (ts_utc - midnight).total_seconds()
    return int(np.floor(seconds / _slot_seconds(time_granularity)))


# -------------------------
# Core planner
# -------------------------
def _build_speed_graph_cache(G_base: nx.Graph, speeds_kts: List[float], time_granularity: int) -> Dict[float, nx.Graph]:
    """Create per-speed graphs with edge attribute 'weight' = duration in slots."""
    cache: Dict[float, nx.Graph] = {}
    for spd in sorted(set(speeds_kts)):
        H = nx.Graph()
        H.add_nodes_from(G_base.nodes())
        # compute per-edge slot weights
        attrs = {}
        for u, v, data in G_base.edges(data=True):
            dist_m = float(data.get("dist_m", data.get("weight", 0.0)))
            w = _edge_duration_slots(dist_m, spd, time_granularity)
            attrs[(u, v)] = {"weight": w}
        H.add_edges_from((u, v, {"weight": attrs[(u, v)]["weight"]}) for (u, v) in attrs.keys())
        cache[spd] = H
    return cache


def generate_filed_plans(
    G_base, ident_to_vid, vid_to_ident, nodes_are_int,
    flights, aircraft_speed,
    time_granularity: int = 4,
    default_speed_kts: float = 450.0,
) -> pd.DataFrame:
    # Load inputs

    # Build mapping ICAO -> vertex id (airports)
    missing_airports = []
    if nodes_are_int:
        def _map_srcdst_numeric(code: str) -> int | None:
            vid = ident_to_vid.get(code)
            if vid is None:
                missing_airports.append(code)
            return vid
        flights["src"] = flights["origin"].map(_map_srcdst_numeric)
        flights["dst"] = flights["destination"].map(_map_srcdst_numeric)
    else:
        # Graph uses IDENTIFIER strings directly
        g_nodes = set(G_base.nodes())
        def _map_srcdst_string(code: str) -> str | None:
            cc = str(code).strip().upper()
            if cc not in g_nodes:
                missing_airports.append(cc)
                return None
            return cc
        flights["src"] = flights["origin"].map(_map_srcdst_string)
        flights["dst"] = flights["destination"].map(_map_srcdst_string)

    # Drop flights with unknown endpoints
    bad_endpoints = flights["src"].isna() | flights["dst"].isna()
    if bad_endpoints.any():
        unknowns = sorted(set(missing_airports))
        print(f"[WARN] {bad_endpoints.sum()} flights dropped due to unknown airport vertex "
              f"(examples: {', '.join(unknowns[:10])}{' ...' if len(unknowns)>10 else ''})", file=sys.stderr)
        flights = flights[~bad_endpoints].copy()

    # Slotize departures
    flights["start_slot"] = flights["departure_time"].map(lambda t: _start_slot_from_timestamp(t, time_granularity))

    # Speed per flight
    def _speed_for(acid: str) -> float:
        return float(aircraft_speed.get(str(acid), default_speed_kts))
    flights["speed_kts"] = flights["aircraft_id"].astype(str).map(_speed_for)

    # Build per-speed cached graphs (edge weights in slots)
    speed_values = flights["speed_kts"].unique().tolist()
    speed_graphs = _build_speed_graph_cache(G_base, speed_values, time_granularity)

    # Iterate flights and produce (Flight_ID, Position, Time)
    rows: List[Tuple[str,str,int]] = []
    use_bar = False
    try:
        from tqdm import tqdm as _tq
        it = _tq(flights.itertuples(index=False), total=len(flights), desc="Filed plans")
        use_bar = True
    except Exception:
        it = flights.itertuples(index=False)
        last_print = time.time()
        print("Generating filed trajectories...")

    missing_paths = 0
    for rec in it:
        # namedtuple fields from flights dataframe
        # fields: flight_id, aircraft_id, origin, destination, departure_time, src, dst, start_slot, speed_kts
        fid = rec.flight_id
        src = rec.src
        dst = rec.dst

        # normalize types for lookup
        if nodes_are_int:
            src = int(src); dst = int(dst)

        spd = float(rec.speed_kts)
        start = int(rec.start_slot)

        G_spd = speed_graphs[spd]
        try:
            path = nx.shortest_path(G_spd, src, dst, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            missing_paths += 1
            continue

        # Walk the path and accumulate times (exact same update rule as reference)
        t = start

        for hop, node in enumerate(path):
            # Map node to IDENTIFIER string for output
            if nodes_are_int:
                if (vid_to_ident is not None) and (0 <= int(node) < len(vid_to_ident)):
                    pos = str(vid_to_ident[int(node)]).strip().upper()
                else:
                    pos = str(int(node))
            else:
                pos = str(node).strip().upper()

            if hop == 0:
                rows.append((fid, pos, t))
            else:
                prev = path[hop-1]
                w = G_spd[prev][node]["weight"]
                t = t + int(w)
                rows.append((fid, pos, t))

        if (not use_bar) and (time.time() - last_print > 5):
            print(f"  processed {len(rows)} trajectory points so far...")
            last_print = time.time()

    if missing_paths:
        print(f"[WARN] {missing_paths} flights had no path on the navgraph and were skipped.", file=sys.stderr)

    out = pd.DataFrame(rows, columns=["Flight_ID","Position","Time"])
    out["Position"] = out["Position"].astype("string")

    return out


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate filed flight plan trajectories on the navgraph.")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Folder containing flights.csv and aircrafts.csv|aircraft.csv.")
    p.add_argument("--navgraph-dir", type=Path, required=True,
                   help="Folder containing vertices.csv and edges.csv.")
    p.add_argument("--time-granularity", type=int, default=4,
                   help="G where 1 slot = 3600/G seconds (default 4 => 15-minute slots).")
    p.add_argument("--default-speed-kts", type=float, default=450.0,
                   help="Fallback speed if aircraft speed missing (knots).")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.data_dir.exists():
        raise FileNotFoundError(f"data directory not found: {args.data_dir}")
    if not args.navgraph_dir.exists():
        raise FileNotFoundError(f"navgraph directory not found: {args.navgraph_dir}")

    G_base, ident_to_vid, vid_to_ident, nodes_are_int = _load_navgraph(args.navgraph_dir)

    flights = _load_flights(args.data_dir)
    aircraft_speed = _load_aircrafts(args.data_dir)

    df = generate_filed_plans(
        G_base, ident_to_vid, vid_to_ident, nodes_are_int,
        flights, aircraft_speed,
        time_granularity = args.time_granularity,
        default_speed_kts=args.default_speed_kts,
    )

    # Handle potential overlapping flights:
    for aircraft in list(set(flights["aircraft_id"])):
        aircraft_flights = flights[flights["aircraft_id"] == aircraft]
        if aircraft_flights.shape[0] == 1:
            # If only 1 flight, then there cannot be an issue of overlapping flights
            continue
        
        aircraft_flights = aircraft_flights.sort_values(by=["start_slot"], ascending=True)
        for index in range(1,aircraft_flights.shape[0]):

            prev_flight_id = aircraft_flights.iloc[index-1,0]
            cur_flight_id = aircraft_flights.iloc[index,0]

            prev_flight = df[df["Flight_ID"] == prev_flight_id]
            cur_flight = df[df["Flight_ID"] == cur_flight_id]
            
            if len(prev_flight["Time"]) > 0:
                prev_flight_max = max(prev_flight["Time"])
            else:
                prev_flight_max = 0

            if len(cur_flight["Time"]) > 0:
                cur_flight_min = min(cur_flight["Time"])
            else:
                cur_flight_min = 1

            if prev_flight_max >= cur_flight_min:
                diff_needed = (prev_flight_max - cur_flight_min) + 1
                indices_cur_flight = df.index[df["Flight_ID"] == cur_flight_id].tolist()
                df.loc[indices_cur_flight,"Time"] = df.loc[indices_cur_flight,"Time"] + diff_needed

        

    out_path = args.data_dir / "filed_flights.csv"
    df.to_csv(out_path, index=False)
    print(f"Done. Wrote {len(df):,} trajectory rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()




