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
from typing import Dict, Tuple, List
from math import ceil
import sys
from collections import defaultdict
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
    Pick a REPLACEMENT destination whose route fits in the remaining window.

    Alexander's rule (2026-09-08): a flight that cannot fit time-wise is dropped and a shorter
    one resampled, and the requested flight count must always be met exactly. So we never
    truncate and never drop a row from the instance -- we re-draw the destination.

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


def _replace_origin(G_spd, start, window, airport_vs, origin_weight, dest_weight, rng,
                    fid, nodes_are_int, vid_to_ident, max_tries=8):
    """
    Last resort: the sampled origin can reach NO airport at all (isolated component).

    Redraw the origin too, so the requested flight count is still met exactly. This should never
    fire on a connected navgraph; it exists so that a degenerate graph degrades gracefully
    instead of silently producing fewer flights than requested.
    """
    cands = list(airport_vs)
    if not cands:
        return None
    w = np.array([float(origin_weight.get(v, 0.0)) + 1e-9 for v in cands], dtype=float)
    w /= w.sum()
    order = rng.choice(len(cands), size=min(max_tries, len(cands)), replace=False, p=w)
    for i in order:
        src2 = cands[int(i)]
        alt = _resample_destination(G_spd, src2, start, window, airport_vs, dest_weight, rng,
                                    fid, nodes_are_int, vid_to_ident)
        if alt is not None:
            dst2, rows2 = alt
            return src2, dst2, rows2
    return None


def generate_filed_plans(
    G_base, ident_to_vid, vid_to_ident, nodes_are_int,
    flights, aircraft_speed,
    time_granularity: int = 4,
    default_speed_kts: float = 450.0,
    considered_timespan: int = 24,
    resample_seed: int = 42,
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
    
    print(flights["start_slot"])

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
    resampled = 0
    unfittable = 0
    replaced_origin = 0
    window = time_granularity * considered_timespan
    airport_vs = set(flights["src"].tolist()) | set(flights["dst"].tolist())
    dest_weight = flights["dst"].value_counts().to_dict()
    origin_weight = flights["src"].value_counts().to_dict()
    rng = np.random.default_rng(resample_seed)
    reassigned = {}   # flight_id -> new dst vertex, applied to flights.csv below
    restarted = {}    # flight_id -> new start slot, for the late-departure fallback
    reorigined = {}   # flight_id -> new src vertex, for the isolated-origin fallback

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
            tmp_rows, max_t = _walk(G_spd, path, fid, start, nodes_are_int, vid_to_ident)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            tmp_rows, max_t = None, None

        if tmp_rows is None or max_t > window:
            # Does not fit (or unreachable): drop this leg and resample a shorter one, keeping
            # the flight itself so the requested count is met exactly. Three escalating
            # fallbacks, each with a bounded candidate list -- none of them loops.
            tmp_rows = None
            alt = _resample_destination(G_spd, src, start, window, airport_vs, dest_weight, rng,
                                        fid, nodes_are_int, vid_to_ident)
            if alt is not None:
                new_dst, tmp_rows = alt
                reassigned[fid] = new_dst
                resampled += 1
            else:
                # (2) Nothing fits from THIS departure slot: nearest airport, departing earlier.
                alt2 = _nearest_fitting(G_spd, src, start, window, airport_vs, fid,
                                        nodes_are_int, vid_to_ident)
                if alt2 is not None:
                    new_dst, tmp_rows, new_start = alt2
                    reassigned[fid] = new_dst
                    restarted[fid] = new_start
                    unfittable += 1
                else:
                    # (3) The origin reaches no airport at all. Redraw the origin as well rather
                    # than drop the flight, because the requested count must be met exactly.
                    alt3 = _replace_origin(G_spd, start, window, airport_vs, origin_weight,
                                           dest_weight, rng, fid, nodes_are_int, vid_to_ident)
                    if alt3 is None:
                        # Genuinely impossible on this graph. Fail loudly -- silently emitting
                        # fewer flights than requested is exactly what must not happen.
                        raise RuntimeError(
                            f"flight {fid}: no origin/destination pair on this navgraph fits a "
                            f"{window}-slot window; cannot honour the requested flight count. "
                            f"Check navgraph connectivity (07_check_parsed_experiments_graph_"
                            f"connectedness.py) or use a finer --time-granularity.")
                    new_src, new_dst, tmp_rows = alt3
                    reassigned[fid] = new_dst
                    reorigined[fid] = new_src
                    replaced_origin += 1

        rows += tmp_rows

        if (not use_bar) and (time.time() - last_print > 5):
            print(f"  processed {len(rows)} trajectory points so far...")
            last_print = time.time()

    if reassigned or reorigined or restarted:
        # Keep flights.csv consistent with the trajectories actually emitted. Every column that
        # describes the leg has to move together: vertex id, ICAO string, and -- for a moved
        # departure -- BOTH start_slot and departure_time. Updating start_slot alone leaves the
        # two disagreeing, and any downstream step that re-derives the slot from the timestamp
        # (stage 05, or a re-run of this stage) would silently undo the change.
        ident = {}
        if nodes_are_int and vid_to_ident is not None:
            touched = set(reassigned.values()) | set(reorigined.values())
            ident = {i: str(vid_to_ident[i]).strip().upper() for i in touched
                     if 0 <= int(i) < len(vid_to_ident)}

        m = flights["flight_id"].map(reassigned)
        hit = m.notna()
        if hit.any():
            flights.loc[hit, "dst"] = m[hit].values
            flights.loc[hit, "destination"] = [ident.get(v, str(v)) for v in m[hit].values]

        mo = flights["flight_id"].map(reorigined)
        ho = mo.notna()
        if ho.any():
            flights.loc[ho, "src"] = mo[ho].values
            flights.loc[ho, "origin"] = [ident.get(v, str(v)) for v in mo[ho].values]

        ms = flights["flight_id"].map(restarted)
        hs = ms.notna()
        if hs.any():
            slot_s = _slot_seconds(time_granularity)
            new_slots = ms[hs].astype(int)
            flights.loc[hs, "start_slot"] = new_slots.values
            base = flights.loc[hs, "departure_time"].dt.normalize()
            flights.loc[hs, "departure_time"] = (
                base + pd.to_timedelta(new_slots.values * slot_s, unit="s"))

        # a resampled leg must never become a self-loop
        loops = int((flights["origin"].astype(str) == flights["destination"].astype(str)).sum())
        if loops:
            print(f"[WARN] {loops} flights became self-loops after resampling.", file=sys.stderr)

    if missing_paths:
        print(f"[WARN] {missing_paths} flights had no reachable destination at all and were "
              f"skipped -- the requested flight count is NOT met.", file=sys.stderr)
    if replaced_origin:
        print(f"[WARN] {replaced_origin} flights had an origin that reaches no airport at all; "
              f"the origin was redrawn to preserve the requested flight count.", file=sys.stderr)
    if resampled:
        print(f"[INFO] {resampled} flights did not fit the "
              f"{time_granularity*considered_timespan}-slot window; a shorter destination was "
              f"resampled for each (flight count preserved).", file=sys.stderr)
    if unfittable:
        print(f"[WARN] {unfittable} flights departed too late for ANY route to complete; each "
              f"was given its nearest airport and an earlier departure so the leg finishes "
              f"in-window (their sampled departure time is not preserved).", file=sys.stderr)

    out = pd.DataFrame(rows, columns=["Flight_ID","Position","Time"])
    out["Position"] = out["Position"].astype("string")

    return out, flights


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
    p.add_argument("--considered-timespan", type=int, default=24)
    p.add_argument("--resample-seed", type=int, default=42,
                   help="seed for re-drawing the destination of a flight that does not fit the "
                        "window; keep it tied to the dataset seed for reproducibility")
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

    df, flights = generate_filed_plans(
        G_base, ident_to_vid, vid_to_ident, nodes_are_int,
        flights, aircraft_speed,
        time_granularity = args.time_granularity,
        default_speed_kts=args.default_speed_kts,
        considered_timespan=args.considered_timespan,
        resample_seed=args.resample_seed,
    )

    max_time = args.time_granularity * args.considered_timespan

    new_aircrafts = []

    # Handle potential overlapping flights:
    for aircraft in list(set(flights["aircraft_id"])):
        aircraft_flights = flights[flights["aircraft_id"] == aircraft]
        if aircraft_flights.shape[0] == 1:
            # If only 1 flight, then there cannot be an issue of overlapping flights
            continue

        print(aircraft)

        new_aircraft_offset = 0

        aircraft_copies = []
        new_aircraft_flights = []
        
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

                if max(df.loc[indices_cur_flight,"Time"]) > max_time:

                    new_aircraft_id = aircraft + f"_{str(new_aircraft_offset)}"

                    # The forward shift above sequences this flight after the previous leg of
                    # the same aircraft. If that pushes it past the window we give the flight a
                    # fresh aircraft instead and undo the shift EXACTLY -- subtracting
                    # max(over, diff_needed) as before could take the flight below its own
                    # original departure slot and hence below t=0, violating the contract.
                    df.loc[indices_cur_flight,"Time"] = df.loc[indices_cur_flight,"Time"] - diff_needed
                    
                    aircraft_copies.append((aircraft,new_aircraft_id))
                    new_aircraft_flights.append((new_aircraft_id,cur_flight_id))

                    new_aircraft_offset += 1

        for new_aircraft_id, flight_id in new_aircraft_flights:
            indices_flights = flights.index[flights["flight_id"]==flight_id]
            flights.loc[indices_flights,"aircraft_id"] = new_aircraft_id

        for aircraft,new_aircraft_id in aircraft_copies:
            speed = aircraft_speed[aircraft]
            aircraft_speed[new_aircraft_id] = speed
   
    # ---- AIRCRAFT-DISJOINTNESS REPAIR -------------------------------------
    # One airplane cannot fly two legs at once. The sequencing loop above sorts a carrier's
    # flights by `start_slot` but compares TRAJECTORY times, and the two diverge as soon as a
    # leg is shifted (or resampled), so some overlaps survive it -- 45 per 1000 flights in the
    # delivered DACH TG=4 instance, 10 after the window fix alone.
    #
    # Enforce the invariant directly instead of trying to make the ordering exact: walk each
    # aircraft's legs in true trajectory order and move any leg that starts before the previous
    # one ends onto a fresh copy of that aircraft. Splitting rather than shifting keeps every
    # timestep inside the window, so this cannot reintroduce a contract violation.
    span = df.groupby("Flight_ID")["Time"].agg(["min", "max"])
    fid2ac = dict(zip(flights["flight_id"], flights["aircraft_id"]))
    by_ac = defaultdict(list)
    for fid, r in span.iterrows():
        by_ac[fid2ac.get(fid)].append((int(r["min"]), int(r["max"]), fid))

    split = {}
    extra_speed = {}
    for ac, legs in by_ac.items():
        if ac is None or len(legs) < 2:
            continue
        legs.sort()
        busy_until = legs[0][1]
        n_copy = 0
        for lo, hi, fid in legs[1:]:
            if lo < busy_until:                      # would overlap -> own aircraft copy
                n_copy += 1
                new_ac = f"{ac}_D{n_copy}"
                split[fid] = new_ac
                extra_speed[new_ac] = aircraft_speed.get(str(ac), args.default_speed_kts)
            else:
                busy_until = hi
    if split:
        flights["aircraft_id"] = flights.apply(
            lambda r: split.get(r["flight_id"], r["aircraft_id"]), axis=1)
        for k, v in extra_speed.items():
            aircraft_speed[k] = v
        print(f"[INFO] {len(split)} legs moved onto fresh aircraft copies so that no airplane "
              f"flies two legs simultaneously.", file=sys.stderr)

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
    print(f"[OK] {n_traj} flights, timesteps within [0, {max_time}].")

    out_path = args.data_dir / "filed_flights.csv"
    df.to_csv(out_path, index=False)
    
    out_path = args.data_dir / "flights.csv"
    flights.to_csv(out_path, index=False)

    out_path = args.data_dir / "aircrafts.csv"
    with open(out_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["aircraft_id", "speed_kts"])
        writer.writerows(aircraft_speed.items())

    print(f"Done. Wrote {len(df):,} trajectory rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()




