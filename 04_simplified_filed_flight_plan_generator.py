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
import csv
from pathlib import Path
from typing import Dict, Sequence, Tuple, List
from math import ceil
import sys
from collections import defaultdict
import time
import numpy as np
import pandas as pd
import networkx as nx
from stage_interfaces import FiledFlightPlanStage
from atomic_io import atomic_to_csv, atomic_open, atomic_group


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

    # Same tolerance as _load_flights below: this stage rewrites flights.csv with to_csv(),
    # which uses a space separator, so the strict "T" format broke re-runs on its own output.
    parsed_ts = pd.to_datetime(fdf["departure_time"], utc=True, errors="coerce", format="ISO8601")
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
    # Parse ISO time PRESERVING the written UTC offset. 01_data_generation writes departure
    # times in the configured local frame (e.g. +08:00); forcing utc=True here re-anchored the
    # whole 24h simulation window to UTC midnight, so the end-of-window clamp below landed in
    # the middle of the local day (17.5% of USA flights, 9.7% of EAST-ASIA, 0.1% of DACH).
    # Accept both spellings of the same instant. Stage 01 writes isoformat() ("...T00:15:11-08:00")
    # but THIS stage rewrites flights.csv with to_csv(), which renders the space separator
    # ("... 00:15:11-08:00"). Pinning the strict "%Y-%m-%dT%H:%M:%S%z" made stage 04
    # non-idempotent: re-running it on its own output failed with "Invalid timestamps".
    fdf["departure_time"] = pd.to_datetime(fdf["departure_time"], utc=False, errors="coerce",
                                           format="ISO8601")
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
    # floor(seconds since LOCAL midnight / slot_seconds). ts carries the configured offset, so
    # normalize() gives midnight in that same frame -- which is what makes slot 0 the start of
    # the simulated local day.
    midnight = ts_utc.normalize()
    seconds = (ts_utc - midnight).total_seconds()
    return_value = int(np.floor(seconds / _slot_seconds(time_granularity)))
    return return_value


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



def _walk(G_spd, path, fid, start, nodes_are_int, vid_to_ident):
    """Walk a path accumulating slot times exactly as the reference rule does."""
    t = start
    out = []
    for hop, node in enumerate(path):
        if nodes_are_int:
            if (vid_to_ident is not None) and (0 <= int(node) < len(vid_to_ident)):
                pos = str(vid_to_ident[int(node)]).strip().upper()
            else:
                pos = str(int(node))
        else:
            pos = str(node).strip().upper()
        if hop > 0:
            t = t + int(G_spd[path[hop-1]][node]["weight"])
        out.append((fid, pos, t))
    return out, t


def _resample_destination(G_spd, src, start, window, airport_vs, dest_weight, rng,
                          fid, nodes_are_int, vid_to_ident, max_tries=12):
    """
    Pick a destination reachable from ``src`` at ``start`` whose route fits the window.

    Used for the REPLACEMENT legs that top the instance back up to the requested flight
    count after truncation has removed some.  ``src`` is always the airport the airframe
    actually stands at, so a leg drawn here continues the rotation instead of breaking it.

    One Dijkstra from the origin bounded by the remaining budget gives every reachable airport
    at once; we then draw among those, weighted by the empirical destination marginal so the
    OD distribution is distorted as little as possible. Candidates are verified by actually
    walking the path, because the walk sums int(w) per edge while Dijkstra sums the float
    weights, and the two can disagree by a slot.
    """
    budget = window - start
    if budget <= 0:
        return None
    try:
        lengths = nx.single_source_dijkstra_path_length(G_spd, src, cutoff=budget, weight="weight")
    except Exception:
        return None
    cand = [v for v in lengths if v != src and v in airport_vs]
    if not cand:
        return None
    w = np.array([float(dest_weight.get(v, 0.0)) + 1e-9 for v in cand], dtype=float)
    w /= w.sum()
    order = rng.choice(len(cand), size=min(max_tries, len(cand)), replace=False, p=w)
    for i in order:
        dst2 = cand[int(i)]
        try:
            path = nx.shortest_path(G_spd, src, dst2, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        rows2, end = _walk(G_spd, path, fid, start, nodes_are_int, vid_to_ident)
        if end <= window:
            return dst2, rows2
    return None


def _nearest_fitting(G_spd, src, start, window, airport_vs, fid, nodes_are_int, vid_to_ident):
    """
    Nearest reachable airport, departing early enough that the leg completes in-window.

    Only ever used for a leg on a FRESH airframe, which has no predecessor to stay behind:
    moving the departure earlier cannot collide with a leg this airframe already flew.

    Keep the departure as close to the sampled one as possible: pinning every fallback flight to
    `window - duration` would make them all land exactly on the window edge and build an
    artificial arrival spike in the last slot.
    """
    try:
        lengths = nx.single_source_dijkstra_path_length(G_spd, src, weight="weight")
    except Exception:
        return None
    cand = sorted(((d, v) for v, d in lengths.items() if v != src and v in airport_vs))
    for _, dst2 in cand[:8]:
        try:
            path = nx.shortest_path(G_spd, src, dst2, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        _, dur = _walk(G_spd, path, fid, 0, nodes_are_int, vid_to_ident)
        if dur > window:
            continue                          # cannot fit even departing at slot 0
        start2 = max(0, min(int(start), window - dur))   # as late as fits, never later than sampled
        rows2, end2 = _walk(G_spd, path, fid, start2, nodes_are_int, vid_to_ident)
        if end2 <= window:
            return dst2, rows2, start2
    return None


def _fresh_leg(G_spd, start, window, airport_vs, origin_weight, dest_weight, rng,
               fid, nodes_are_int, vid_to_ident, max_tries=8):
    """
    Draw a whole new leg -- origin and destination both -- for a brand-new airframe.

    This is the top-up of last resort.  It is safe for continuity precisely because the
    airframe it is drawn for has flown nothing yet: with no previous leg there is no airport
    to depart from, so any origin is admissible.  (It is the descendant of ``_replace_origin``,
    which drew a new ORIGIN for an existing airframe's leg -- the single worst source of the
    broken rotations this change removes.)

    Origins are drawn by the empirical origin marginal and destinations by the destination
    marginal, so the OD distribution moves as little as the constraint allows.  Bounded: at
    most ``max_tries`` origins, each with a bounded destination draw, then one sweep that
    also allows an earlier departure.
    """
    cands = list(airport_vs)
    if not cands:
        return None
    w = np.array([float(origin_weight.get(v, 0.0)) + 1e-9 for v in cands], dtype=float)
    w /= w.sum()
    order = [cands[int(i)] for i in
             rng.choice(len(cands), size=min(max_tries, len(cands)), replace=False, p=w)]
    for src2 in order:
        alt = _resample_destination(G_spd, src2, start, window, airport_vs, dest_weight, rng,
                                    fid, nodes_are_int, vid_to_ident)
        if alt is not None:
            dst2, rows2 = alt
            return src2, dst2, rows2, int(start)
    # Nothing fits from the sampled departure slot: keep the origin draw, move the departure
    # earlier.  A fresh airframe has nothing behind it, so this cannot overlap anything.
    for src2 in order:
        alt2 = _nearest_fitting(G_spd, src2, start, window, airport_vs, fid,
                                nodes_are_int, vid_to_ident)
        if alt2 is not None:
            dst2, rows2, start2 = alt2
            return src2, dst2, rows2, int(start2)
    return None


def generate_filed_plans(
    G_base, ident_to_vid, vid_to_ident, nodes_are_int,
    flights, aircraft_speed,
    time_granularity: int = 4,
    default_speed_kts: float = 450.0,
    considered_timespan: int = 24,
    resample_seed: int = 42,
) -> pd.DataFrame:
    """Route every flight over the navgraph, one AIRFRAME at a time.

    The unit of work is the airframe, not the flight row, and that is the whole point.
    Stage 01 hands over a rotation: leg k+1 of an airframe departs from the airport leg k
    landed at, because an aircraft is only ever drawn from the pool standing at that
    airport.  Routing flights independently -- which is what this stage used to do -- meant
    that any leg which would not fit the 24 h window had its endpoints rewritten in
    isolation: a different destination, or, worst of all, a different ORIGIN.  Either edit
    leaves the airframe departing from an airport it never landed at.  On the published
    TG=1 data that was 211 of 224 consecutive leg pairs in USA-MAINLAND.

    So: walk each airframe's legs in departure order, carrying the airport it stands at.
    Every leg departs from there, full stop.  A leg that will not fit ends the airframe's
    day -- it and every later leg of that airframe are dropped, which is the only way to
    shorten a chain without breaking it.

    That leaves the instance short of the requested flight count, which is not negotiable,
    so the dropped legs are then re-specified as REPLACEMENT legs: first by extending an
    airframe from wherever it is parked, and only failing that by putting a single leg on a
    brand-new airframe.  Both are continuous by construction.  The flight ids are the
    dropped legs' own, so ``flights.csv`` keeps exactly the rows and the count it arrived
    with.
    """
    requested = int(len(flights))

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

    # A flight whose origin or destination is not a graph vertex is NOT dropped from the
    # instance -- dropping it silently delivered fewer flights than the configuration asked
    # for.  It is left unrouted here and re-specified by the top-up below, exactly like a
    # leg that does not fit the window.
    bad_endpoints = flights["src"].isna() | flights["dst"].isna()
    if bad_endpoints.any():
        unknowns = sorted(set(missing_airports))
        print(f"[WARN] {int(bad_endpoints.sum())} flights have an airport that is not a graph "
              f"vertex (examples: {', '.join(unknowns[:10])}{' ...' if len(unknowns)>10 else ''}); "
              f"each is re-drawn below so the requested flight count is still met",
              file=sys.stderr)

    # Slotize departures
    flights["start_slot"] = flights["departure_time"].map(lambda t: _start_slot_from_timestamp(t, time_granularity))

    print(flights["start_slot"])

    # Speed per flight
    def _speed_for(acid: str) -> float:
        return float(aircraft_speed.get(str(acid), default_speed_kts))
    flights["speed_kts"] = flights["aircraft_id"].astype(str).map(_speed_for)

    # Build per-speed cached graphs (edge weights in slots)
    speed_values = flights["speed_kts"].unique().tolist()
    speed_graphs = _build_speed_graph_cache(G_base, speed_values, time_granularity)

    window = time_granularity * considered_timespan
    rng = np.random.default_rng(resample_seed)

    # ---- plain python views, so an airframe's legs can be walked by position ----------
    def _vertex(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return int(x) if nodes_are_int else str(x)

    n = len(flights)
    fid_arr   = list(flights["flight_id"])
    ac_arr    = [str(a) for a in flights["aircraft_id"]]
    src_arr   = [_vertex(v) for v in flights["src"]]
    dst_arr   = [_vertex(v) for v in flights["dst"]]
    start_arr = [int(t) for t in flights["start_slot"]]
    spd_arr   = [float(s) for s in flights["speed_kts"]]

    airport_vs = {v for v in src_arr if v is not None} | {v for v in dst_arr if v is not None}
    dest_weight: Dict[object, int] = defaultdict(int)
    origin_weight: Dict[object, int] = defaultdict(int)
    for v in dst_arr:
        if v is not None:
            dest_weight[v] += 1
    for v in src_arr:
        if v is not None:
            origin_weight[v] += 1

    # Airframes in order of first appearance -- stage 01 numbers them in that order, and it
    # is stable without depending on how strings happen to hash.
    frame_order: List[str] = []
    legs_of: Dict[str, List[int]] = defaultdict(list)
    for i in range(n):
        if ac_arr[i] not in legs_of:
            frame_order.append(ac_arr[i])
        legs_of[ac_arr[i]].append(i)
    for ac in frame_order:
        legs_of[ac].sort(key=lambda i: (start_arr[i], str(fid_arr[i])))

    rows: List[Tuple[str, str, int]] = []
    routed = [False] * n
    out_src, out_dst = list(src_arr), list(dst_arr)
    out_ac, out_start = list(ac_arr), list(start_arr)

    parked: Dict[str, Tuple[object, int | None]] = {}   # airframe -> (airport, last landing)
    truncated_frames = 0
    delayed = 0
    forced_origin = 0

    use_bar = False
    last_print = time.time()
    try:
        from tqdm import tqdm as _tq
        it = _tq(frame_order, total=len(frame_order), desc="Filed plans")
        use_bar = True
    except Exception:
        it = frame_order
        print("Generating filed trajectories...")

    # ---- PASS 1: fly each rotation until a leg will not fit --------------------------
    for ac in it:
        idxs = legs_of[ac]
        G_spd = speed_graphs[spd_arr[idxs[0]]]
        at: object | None = src_arr[idxs[0]]
        busy: int | None = None
        cut: int | None = None

        for k, i in enumerate(idxs):
            if at is None or dst_arr[i] is None or dst_arr[i] == at:
                cut = k                      # unmapped airport, or a self-loop from here
                break
            start = start_arr[i] if busy is None else max(start_arr[i], busy + 1)
            if start > window:
                cut = k
                break
            try:
                path = nx.shortest_path(G_spd, at, dst_arr[i], weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                cut = k
                break
            leg_rows, end = _walk(G_spd, path, fid_arr[i], start, nodes_are_int, vid_to_ident)
            if end > window:
                cut = k                      # the rotation's day is over; truncate here
                break

            rows += leg_rows
            routed[i] = True
            if at != src_arr[i]:
                forced_origin += 1           # input disagreed about where the airframe was
            out_src[i] = at
            if start != start_arr[i]:
                delayed += 1                 # held back to keep the one-timestep gap
            out_start[i] = start
            at, busy = dst_arr[i], end

        if at is not None:
            parked[ac] = (at, busy)
        if cut is not None:
            truncated_frames += 1

        if (not use_bar) and (time.time() - last_print > 5):
            print(f"  processed {len(rows)} trajectory points so far...")
            last_print = time.time()

    # ---- PASS 2: top back up to the requested flight count ---------------------------
    # Every leg pass 1 declined to fly is re-specified here.  It keeps its flight id and its
    # sampled departure slot where that still works, and gets a route that both fits the
    # window and continues a rotation.
    orphans = [i for i in range(n) if not routed[i]]
    extended = fresh = 0
    fresh_count: Dict[str, int] = defaultdict(int)
    # Which airframe a carrier's dropped legs should try to continue.  Once its own day is
    # spent and a fresh airframe has been started for it, that fresh one is the live rotation
    # and the next dropped leg continues THAT, instead of asking the exhausted original again
    # and minting a second single-leg airframe for nothing.
    frame_for: Dict[str, str] = {}

    for i in orphans:
        ac = ac_arr[i]
        fid = fid_arr[i]
        G_spd = speed_graphs[spd_arr[i]]
        placed = False

        # (1) Extend the airframe this leg belongs to, from the airport it is parked at.
        frame = frame_for.get(ac, ac)
        if frame in parked:
            at, busy = parked[frame]
            start = start_arr[i] if busy is None else max(start_arr[i], busy + 1)
            if start <= window:
                alt = _resample_destination(G_spd, at, start, window, airport_vs,
                                            dest_weight, rng, fid, nodes_are_int, vid_to_ident)
                if alt is not None:
                    dst2, leg_rows = alt
                    rows += leg_rows
                    out_ac[i] = frame
                    out_src[i], out_dst[i], out_start[i] = at, dst2, start
                    parked[frame] = (dst2, leg_rows[-1][2])
                    routed[i] = True
                    extended += 1
                    placed = True

        # (2) Nothing left in this airframe's day: put the leg on a brand-new airframe,
        #     which has no previous landing and so cannot break continuity.
        if not placed:
            alt = _fresh_leg(G_spd, start_arr[i], window, airport_vs, origin_weight,
                             dest_weight, rng, fid, nodes_are_int, vid_to_ident)
            if alt is None:
                raise RuntimeError(
                    f"flight {fid}: no origin/destination pair on this navgraph fits a "
                    f"{window}-slot window, so the requested flight count cannot be honoured. "
                    f"Check navgraph connectivity (07_check_parsed_experiments_graph_"
                    f"connectedness.py) or use a finer --time-granularity.")
            src2, dst2, leg_rows, start2 = alt
            fresh_count[ac] += 1
            new_ac = f"{ac}_R{fresh_count[ac]}"
            aircraft_speed[new_ac] = spd_arr[i]
            rows += leg_rows
            out_ac[i], out_src[i], out_dst[i], out_start[i] = new_ac, src2, dst2, start2
            parked[new_ac] = (dst2, leg_rows[-1][2])
            frame_for[ac] = new_ac
            routed[i] = True
            fresh += 1

    # ---- write the plan back onto flights.csv ----------------------------------------
    # Every column describing a leg moves together: vertex id, ICAO string, airframe, and --
    # for a leg that was held back or redrawn -- BOTH start_slot and departure_time.  Leaving
    # start_slot and departure_time disagreeing means any downstream step that re-derives the
    # slot from the timestamp (stage 05, or a re-run of this stage) silently undoes the change.
    def _ident(v):
        if v is None:
            raise RuntimeError("a leg reached the write-back with no endpoint; every flight "
                               "must have been either flown or re-specified above")
        if nodes_are_int and vid_to_ident is not None and 0 <= int(v) < len(vid_to_ident):
            return str(vid_to_ident[int(v)]).strip().upper()
        return str(v)

    flights["aircraft_id"] = out_ac
    flights["src"] = out_src
    flights["dst"] = out_dst
    flights["origin"] = [_ident(v) for v in out_src]
    flights["destination"] = [_ident(v) for v in out_dst]
    # A leg whose slot did not move keeps its sampled timestamp to the second; only a leg that
    # was held back or redrawn is re-stamped, from local midnight plus its new slot.
    slot_s = _slot_seconds(time_granularity)
    kept = np.asarray(out_start) == np.asarray(start_arr)
    restamped = (flights["departure_time"].dt.normalize()
                 + pd.to_timedelta(np.asarray(out_start, dtype=float) * slot_s, unit="s"))
    flights["departure_time"] = flights["departure_time"].where(kept, restamped)
    flights["start_slot"] = out_start

    loops = int((flights["origin"].astype(str) == flights["destination"].astype(str)).sum())
    if loops:
        print(f"[WARN] {loops} flights are self-loops.", file=sys.stderr)

    if truncated_frames:
        print(f"[INFO] {truncated_frames} airframes had their day truncated: a leg would not "
              f"have completed inside the {window}-slot window, so it and every later leg of "
              f"that airframe were dropped rather than re-routed from somewhere the airframe "
              f"was not.", file=sys.stderr)
    if extended or fresh:
        print(f"[INFO] {extended + fresh} dropped legs re-specified to hold the requested "
              f"count at {requested}: {extended} continue a parked airframe, {fresh} start a "
              f"new one.", file=sys.stderr)
    if delayed:
        print(f"[INFO] {delayed} legs departed later than sampled, to leave a whole timestep "
              f"between them and the airframe's previous landing.", file=sys.stderr)
    if forced_origin:
        print(f"[WARN] {forced_origin} legs named an origin the airframe was not standing at; "
              f"the airframe's actual position was used.", file=sys.stderr)

    out = pd.DataFrame(rows, columns=["Flight_ID","Position","Time"])
    out["Position"] = out["Position"].astype("string")

    return out, flights

# -------------------------
# CLI
# -------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate filed flight plan trajectories on the navgraph.")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Folder containing flights.csv and aircrafts.csv|aircraft.csv.")
    p.add_argument("--navgraph-dir", type=Path, required=True,
                   help="Folder containing vertices.csv and edges.csv.")
    p.add_argument("--time-granularity", type=int, default=4,
                   help="G where 1 slot = 3600/G seconds (default 4 => 15-minute slots).")
    p.add_argument("--default-speed-kts", type=float, default=450.0,
                   help="Fallback speed if aircraft speed missing (knots).")
    p.add_argument("--considered-timespan", type=int, default=24)
    p.add_argument("--resample-seed", type=int, default=42,
                   help="seed for re-drawing the destination of a flight that does not fit the "
                        "window; keep it tied to the dataset seed for reproducibility")
    return p.parse_args(argv)


class FiledFlightPlanGenerator(FiledFlightPlanStage):
    r"""The shipped stage 04: route each flight over the graph and time-stamp it.

    This is the reference implementation of
    :class:`stage_interfaces.FiledFlightPlanStage`; that class's docstring is the
    contract — ``filed_flights.csv``, the 24-hour window, adjacency, airport
    endpoints, the whole-timestep separation between two legs of one airframe,
    and the rotation itself: leg *k+1* departs from the airport leg *k* landed
    at. ``generate_filed_plans`` above is where all five are established; what
    follows here only verifies them.

    Note the stage is **not idempotent**: it rewrites ``flights.csv`` and
    ``aircrafts.csv`` in place, so a re-run must start from stage 01.

    ``run_pipeline.py`` calls this class directly: it is stage ``filedplans``'s
    ``default``, with no adapter and no subprocess in between. To name it
    explicitly instead::

        python run_pipeline.py --config <cfg> --stage-impl \
            filedplans=04_simplified_filed_flight_plan_generator.py:FiledFlightPlanGenerator
    """

    def start(self, argv: Sequence[str]) -> None:
        args = parse_args(list(argv))
        if not args.data_dir.exists():
            raise FileNotFoundError(f"data directory not found: {args.data_dir}")
        if not args.navgraph_dir.exists():
            raise FileNotFoundError(f"navgraph directory not found: {args.navgraph_dir}")

        G_base, ident_to_vid, vid_to_ident, nodes_are_int = _load_navgraph(args.navgraph_dir)

        flights = _load_flights(args.data_dir)
        aircraft_speed = _load_aircrafts(args.data_dir)

        df, flights = generate_filed_plans(
            G_base, ident_to_vid, vid_to_ident, nodes_are_int,
            flights, aircraft_speed,
            time_granularity = args.time_granularity,
            default_speed_kts=args.default_speed_kts,
            considered_timespan=args.considered_timespan,
            resample_seed=args.resample_seed,
        )

        max_time = args.time_granularity * args.considered_timespan

        # ---- SEQUENCING CHECK --------------------------------------------------------
        # This was a repair: shift a clashing leg forward, and, when the shift pushed it past
        # the window, move it onto a fresh copy of the aircraft.  generate_filed_plans now
        # sequences an airframe's legs as it flies them, so a clash reaching here means that
        # walk is broken.  Repairing it by copying the aircraft is one of the two places a
        # rotation used to be severed -- the copy carries the tail of the chain away from the
        # head, and the two no longer meet -- so this reports and fails instead.
        for aircraft in list(set(flights["aircraft_id"])):
            aircraft_flights = flights[flights["aircraft_id"] == aircraft]
            if aircraft_flights.shape[0] == 1:
                # If only 1 flight, then there cannot be an issue of overlapping flights
                continue

            print(aircraft)

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
                    raise RuntimeError(
                        f"aircraft {aircraft}: leg {cur_flight_id} departs at t="
                        f"{cur_flight_min} but leg {prev_flight_id} lands at t="
                        f"{prev_flight_max}. The airframe walk in generate_filed_plans is "
                        f"supposed to make this impossible; do not ship the result.")

        # ---- AIRCRAFT-DISJOINTNESS CHECK ---------------------------------------------
        # One airplane cannot fly two legs at once.  The loop above sorts a carrier's flights
        # by `start_slot` but compares TRAJECTORY times, and the two diverge as soon as a leg
        # is re-timed, so this second pass walks each aircraft's legs in true trajectory order.
        # It used to move an overlapping leg onto a fresh copy of the aircraft: the other place
        # a rotation was severed.  It now reports and fails.
        span = df.groupby("Flight_ID")["Time"].agg(["min", "max"])
        fid2ac = dict(zip(flights["flight_id"], flights["aircraft_id"]))
        by_ac = defaultdict(list)
        for fid, r in span.iterrows():
            by_ac[fid2ac.get(fid)].append((int(r["min"]), int(r["max"]), fid))

        clashes = []
        for ac, legs in by_ac.items():
            if ac is None or len(legs) < 2:
                continue
            legs.sort()
            busy_until = legs[0][1]
            for lo, hi, fid in legs[1:]:
                # A leg must start at least one timestep AFTER the previous one ends: arrive at
                # t=5, depart no earlier than t=6.  Sharing the boundary slot would put the
                # aircraft at two navpoints in the same timestep and count it twice in that
                # slot's occupancy, so `lo == busy_until` is a violation, not a rounding
                # artefact.
                if lo <= busy_until:
                    clashes.append((ac, fid, lo, busy_until))
                busy_until = max(busy_until, hi)
        if clashes:
            raise RuntimeError(
                f"{len(clashes)} legs are airborne while the same airframe already is, "
                f"e.g. {clashes[:3]} as (airframe, flight, departs, busy until). "
                f"Do not ship the result.")

        # ---- CONTRACT GUARD ---------------------------------------------------
        # The parsed instances the optimizers consume assume 0 <= t <= time_granularity * 24.
        # Enforce it here, at the single point where the filed plan is written, so no combination
        # of window truncation and aircraft re-sequencing can emit an out-of-range timestep.
        before = len(df)
        lo, hi = int(df["Time"].min()), int(df["Time"].max())
        df = df[(df["Time"] >= 0) & (df["Time"] <= max_time)]
        if len(df) != before:
            print(f"[WARN] contract guard dropped {before - len(df)} of {before} trajectory points "
                  f"outside [0, {max_time}] (observed range [{lo}, {hi}]).", file=sys.stderr)
        empty = set(flights["flight_id"]) - set(df["Flight_ID"])
        if empty:
            raise RuntimeError(
                f"{len(empty)} flights lost every trajectory point to the contract guard, so the "
                f"instance would contain fewer flights than requested. This is a bug in the window "
                f"handling above, not a data property -- do not ship the result.")

        # Requested count must be met EXACTLY (Alexander, 2026-09-08).
        n_traj, n_decl = df["Flight_ID"].nunique(), flights["flight_id"].nunique()
        if n_traj != n_decl:
            raise RuntimeError(f"flight-count mismatch: {n_traj} in filed_flights.csv vs {n_decl} "
                               f"in flights.csv")
        if not df.empty:
            assert df["Time"].min() >= 0 and df["Time"].max() <= max_time, "contract guard failed"

        # ---- ROTATION-CONTINUITY GUARD -----------------------------------------------
        # The property this stage exists to stop breaking: an airframe's next leg departs from
        # the airport its previous leg landed at.  Checked on the trajectories actually about
        # to be written, not on flights.csv, because the trajectories are what a solver reads
        # and check_instances.py's D8/F10 is computed from exactly these two columns.
        walk = df.sort_values(["Flight_ID", "Time"])
        ends = walk.groupby("Flight_ID")["Position"].agg(["first", "last"])
        ends = ends.join(walk.groupby("Flight_ID")["Time"].agg(["min", "max"]))
        ends["ac"] = [fid2ac.get(f) for f in ends.index]
        broken = []
        for ac, grp in ends.groupby("ac"):
            if len(grp) < 2:
                continue
            srt = grp.sort_values("min")
            for arrive, depart in zip(srt["last"].values[:-1], srt["first"].values[1:]):
                if arrive != depart:
                    broken.append((ac, arrive, depart))
        if broken:
            raise RuntimeError(
                f"{len(broken)} consecutive leg pairs depart from an airport the airframe "
                f"never landed at, e.g. {broken[:3]} as (airframe, landed, departed). "
                f"Do not ship the result.")
        legs_per_frame = len(ends) / max(1, ends["ac"].nunique())
        print(f"[OK] {n_traj} flights, timesteps within [0, {max_time}], "
              f"{ends['ac'].nunique()} airframes ({legs_per_frame:.3f} legs each), "
              f"rotation continuous.")

        # All three land together or none do.  run_pipeline's guard for this stage
        # checks only filed_flights.csv, while flights.csv and aircrafts.csv already
        # exist from stage 01 -- so a stage that wrote the first and died would be
        # skipped on a re-run, leaving a filed plan beside an un-rewritten flights.csv.
        with atomic_group() as group:
            out_path = args.data_dir / "filed_flights.csv"
            group.to_csv(df, out_path, index=False)

            group.to_csv(flights, args.data_dir / "flights.csv", index=False)

            with group.open(args.data_dir / "aircrafts.csv", mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["aircraft_id", "speed_kts"])
                writer.writerows(aircraft_speed.items())

        print(f"Done. Wrote {len(df):,} trajectory rows to {out_path.resolve()}")


if __name__ == "__main__":
    FiledFlightPlanGenerator().start(sys.argv[1:])
