#!/usr/bin/env python3
"""
Navpoint/Airport Voronoi-style graph builder

Vertices: airports (OurAirports) + navpoints (X-Plane fix.dat + nav.dat)
Edges: undirected (A,B) if there is **no** third vertex C that is "closer along the way"
      —implemented via a Relative Neighborhood Graph (RNG) or Gabriel test:

  RNG (default): keep (A,B) iff there is no C with max(d(A,C), d(B,C)) < d(A,B)
  Gabriel:      keep (A,B) iff there is no C with d(A,C)^2 + d(B,C)^2 < d(A,B)^2

Distances are great-circle (haversine), reported in **meters** for edges.

Region restriction:
  Reads the same JSON config you used in the model builder, and applies any
  `considered_geographic_regions` polygons (lat,lon pairs, ray-cast point-in-poly).

Outputs:
  - vertices.csv with columns: IDENTIFIER,LAT,LON,ALTITUDE,IS_AIRPORT
  - edges.csv    with columns: V0,V1,D    (0-based vertex ids, D in meters)

Progress:
  Shows a progress bar with ETA (via tqdm if available) or periodic percentage + ETA prints.

Usage:
  python build_navgraph.py \
      --config ./config.json \
      --ourairports ./ourairports/airports.csv \
      --navdir ./test_navpoints \
      --max-edge-km 350 \
      --criterion rng \
      --out-dir ./navgraph_out

Optional:
  --od-file ./data_out/.../flights.csv            # to keep only airports observed in O/Ds (by ICAO)
  --aircrafts-file ./data_out/.../aircrafts.csv   # loaded (speeds) but not used for edges
  --icao-only true                                # keep only 4-letter ICAO airports (default true)
  --progress-interval 5                           # seconds between ETA prints when tqdm not present
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
from time import time
from typing import Dict, List, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import networkx as nx

# -------------------------
# Helpers: progress
# -------------------------
def _tqdm(seq, **kwargs):
    try:
        from tqdm import tqdm as _t
        return _t(seq, **kwargs)
    except Exception:
        return seq

# -------------------------
# Geo helpers
# -------------------------
EARTH_R_M = 6371008.8  # mean Earth radius (meters)

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance (meters). All angles in degrees."""
    φ1, λ1, φ2, λ2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = np.sin(dφ/2.0)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(dλ/2.0)**2
    return float(2.0 * EARTH_R_M * np.arcsin(np.sqrt(a)))

def haversine_m_vec(latlon_deg: np.ndarray, lat_deg: float, lon_deg: float) -> np.ndarray:
    """
    Vectorized distance from (lat_deg,lon_deg) to many points.
    latlon_deg: shape (N,2) in degrees
    returns: distances in meters (shape (N,))
    """
    φ1 = np.deg2rad(latlon_deg[:,0])
    λ1 = np.deg2rad(latlon_deg[:,1])
    φ2 = np.deg2rad(lat_deg)
    λ2 = np.deg2rad(lon_deg)
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = np.sin(dφ/2.0)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(dλ/2.0)**2
    return 2.0 * EARTH_R_M * np.arcsin(np.sqrt(a))

# -------------------------
# Region filtering (same approach as your model builder)
# -------------------------
def _pairs_from_flat_polygon(flat: list) -> list[tuple[float,float]]:
    if not isinstance(flat, list) or len(flat) < 6 or len(flat) % 2 != 0:
        raise ValueError("Polygon must be a flat list [lat0,lon0,...,latN,lonN] with N>=2")
    return [(float(flat[i]), float(flat[i+1])) for i in range(0, len(flat), 2)]

def _point_in_poly(lat: float, lon: float, poly: list[tuple[float,float]]) -> bool:
    # ray casting; x=lon, y=lat
    x, y = lon, lat
    inside = False
    n = len(poly)
    for i in range(n):
        y1, x1 = poly[i][0], poly[i][1]
        y2, x2 = poly[(i+1) % n][0], poly[(i+1) % n][1]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1):
            inside = not inside
    return inside

def load_regions_from_config(path: Path) -> List[List[Tuple[float,float]]]:
    if not path or not path.exists():
        return []
    with open(path, "r") as fh:
        cfg = json.load(fh) or {}
    regs = []
    for item in cfg.get("considered_geographic_regions", []):
        regs.append(_pairs_from_flat_polygon(item.get("polygon", [])))
    return regs

def in_any_region(lat: float, lon: float, regions: List[List[Tuple[float,float]]]) -> bool:
    if not regions:
        return True
    for poly in regions:
        if _point_in_poly(lat, lon, poly):
            return True
    return False

# -------------------------
# Data loading
# -------------------------
def load_ourairports_df(path: Path, icao_only: bool=True) -> pd.DataFrame:
    """
    Normalize to columns: ident (ICAO if available), lat, lon
    Prefer 4-letter ICAO codes; fall back to ident/gps_code if looks like ICAO.
    """
    raw = pd.read_csv(path, dtype="string", low_memory=False)
    for c in ["icao_code","ident","gps_code"]:
        if c not in raw.columns:
            raw[c] = pd.Series(dtype="string")
    # Build candidate ICAO-like
    cand = pd.concat([raw["icao_code"], raw["ident"], raw["gps_code"]], ignore_index=True)
    icao = cand.dropna().astype(str).str.strip().str.upper()
    pat = r"^[A-Z]{4}$"
    icao = icao[icao.str.match(pat)].drop_duplicates()
    # Compose frame
    df = pd.DataFrame({"ident": raw["icao_code"]})
    if df["ident"].isna().all(): df["ident"] = raw["ident"]
    if df["ident"].isna().all(): df["ident"] = raw["gps_code"]
    df["ident"] = df["ident"].astype("string").str.strip().str.upper()
    df["lat"] = pd.to_numeric(raw.get("latitude_deg", pd.Series(dtype="float64")), errors="coerce")
    df["lon"] = pd.to_numeric(raw.get("longitude_deg", pd.Series(dtype="float64")), errors="coerce")
    df = df.dropna(subset=["ident","lat","lon"])
    if icao_only:
        df = df[df["ident"].str.match(pat)]
    # Keep only idents that appear in candidate ICAOs (guards weird files)
    df = df[df["ident"].isin(set(icao))]

    return df[["ident","lat","lon"]].drop_duplicates(subset=["ident"])


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Helper to access a column by case-insensitive name; returns empty Series if missing."""
    matches = [c for c in df.columns if c.lower() == name]
    return df[matches[0]] if matches else pd.Series(dtype="string")


def parse_fix(path: Path) -> pd.DataFrame:
    """
    X-Plane FIX1101: each line 'lat lon name [....]' until a '99' line.
    """
    rows = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("99"):
                break
            parts = s.split()
            try:
                lat = float(parts[0]); lon = float(parts[1])
                name = parts[2]
            except Exception:
                continue
            rows.append((name, lat, lon))
    df = pd.DataFrame(rows, columns=["ident","lat","lon"]).drop_duplicates(subset=["ident"])
    return df

def parse_nav(path: Path) -> pd.DataFrame:
    """
    X-Plane NAV1150: we keep navaids commonly used in routes:
      record types {2,3,12,13} (VOR/NDB/DME/TACAN, etc).
    Fields vary a bit across cycles; robustly take lat,lon, and the *last* token as ident.
    """
    keep_types = {2,3,12,13}
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("99"):
                continue
            parts = s.split()
            try:
                code = int(parts[0])
            except Exception:
                continue
            if code not in keep_types:
                continue
            try:
                lat = float(parts[1]); lon = float(parts[2])
                ident = parts[8] if len(parts) > 8 else parts[-1]
            except Exception:
                # very defensive fallback
                try:
                    lat = float(parts[1]); lon = float(parts[2]); ident = parts[-1]
                except Exception:
                    continue
            rows.append((ident, lat, lon))
    df = pd.DataFrame(rows, columns=["ident","lat","lon"]).drop_duplicates(subset=["ident"])
    return df

def load_od_pairs(path: Path | None) -> pd.DataFrame:
    """
    Optional: load OD pairs (any CSV with at least columns origin,destination).
    Used only to optionally *restrict* airports to those in OD list.
    """
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["origin","destination"])
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    o = cols.get("origin"); d = cols.get("destination")
    if not o or not d:
        return pd.DataFrame(columns=["origin","destination"])
    return df[[o,d]].rename(columns={o:"origin", d:"destination"}).astype(str).applymap(str.upper)

def load_aircrafts(path: Path | None) -> pd.DataFrame:
    """
    Optional: load aircrafts with speed_kts (if present). Not used for edges; kept for completeness.
    """
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["aircraft_id","speed_kts"])
    df = pd.read_csv(path)
    # keep these two if present
    keep = [c for c in ["aircraft_id","speed_kts"] if c in df.columns]
    return df[keep].copy()

# -------------------------
# Graph building
# -------------------------
def build_vertices(
    airports_csv: Path,
    nav_dir: Path,
    regions: List[List[Tuple[float,float]]],
    icao_only: bool = True,
    od_filter: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: IDENTIFIER, LAT, LON, ALTITUDE (float, meters)
    """
    # Airports
    ap = load_ourairports_df(airports_csv, icao_only=icao_only)

    ap = ap.rename(columns={"ident":"IDENTIFIER","lat":"LAT","lon":"LON"})
    ap["ALTITUDE"] = 0.0
    ap["IS_AIRPORT"] = 1

    # If OD provided: keep only airports that appear in origin or destination
    #if od_filter is not None and not od_filter.empty:
    #    ap = ap[ap["IDENTIFIER"].isin(set(od_filter["origin"]).union(set(od_filter["destination"])))]

    # Navpoints
    fix_path = nav_dir / "fix.dat"
    nav_path = nav_dir / "nav.dat"
    nav_frames = []
    if fix_path.exists():
        fdf = parse_fix(fix_path).rename(columns={"ident":"IDENTIFIER","lat":"LAT","lon":"LON"})
        fdf["ALTITUDE"] = 0.0
        fdf["IS_AIRPORT"] = 0
        nav_frames.append(fdf)
    if nav_path.exists():
        ndf = parse_nav(nav_path).rename(columns={"ident":"IDENTIFIER","lat":"LAT","lon":"LON"})
        ndf["ALTITUDE"] = 0.0
        ndf["IS_AIRPORT"] = 0
        nav_frames.append(ndf)
    nav_cols = ["IDENTIFIER","LAT","LON","ALTITUDE","IS_AIRPORT"]
    nav = pd.concat(nav_frames, ignore_index=True) if nav_frames else pd.DataFrame(columns=nav_cols)

    # Merge, preferring airport rows on IDENTIFIER collisions
    allv = pd.concat([ap, nav], ignore_index=True)

    allv.drop_duplicates(subset=["IDENTIFIER"], keep="first", inplace=True)  # airports first

    # Region filter
    if regions:
        mask = allv.apply(lambda r: in_any_region(float(r["LAT"]), float(r["LON"]), regions), axis=1)
        allv = allv[mask]

    # Clean NaNs and outliers
    allv = allv.dropna(subset=["IDENTIFIER","LAT","LON"]).copy()
    allv["IDENTIFIER"] = allv["IDENTIFIER"].astype(str).str.strip().str.upper()

    # Reindex to have stable 0..N-1 ids
    allv.reset_index(drop=True, inplace=True)
    return allv[["IDENTIFIER","LAT","LON","ALTITUDE","IS_AIRPORT"]]


def _neighbors_within_maxdist_balltree(latlon: np.ndarray, max_edge_m: float, progress_msg: str = "Neighbor search"):
    """
    Use scikit-learn BallTree with haversine metric (expects radians).
    Returns (nbr_idx, nbr_dst, pair_list) like the brute-force version.
    Distances returned in meters.
    """
    try:
        from sklearn.neighbors import BallTree
    except Exception as e:
        raise RuntimeError("BallTree requested but scikit-learn is not available. Install scikit-learn.") from e

    N = latlon.shape[0]
    rad = np.deg2rad(latlon)  # columns: [lat, lon] in radians
    tree = BallTree(rad, metric="haversine")
    radius_rad = max_edge_m / EARTH_R_M

    nbr_idx = [None] * N
    nbr_dst = [None] * N
    pair_list = []

    use_tqdm = False
    try:
        from tqdm import tqdm as _tq
        it = _tq(range(N), desc=progress_msg)
        use_tqdm = True
    except Exception:
        it = range(N)
        print(f"{progress_msg} with BallTree...")
        last_print = time()

    for i in it:
        inds, dists = tree.query_radius(rad[i:i+1], r=radius_rad, return_distance=True, sort_results=True)
        ii = inds[0]
        dd = (dists[0] * EARTH_R_M).astype(float)  # radians -> meters
        # remove self (distance 0)
        mask = (ii != i)
        ii = ii[mask]; dd = dd[mask]
        nbr_idx[i] = ii
        nbr_dst[i] = dd
        # collect i<j
        js = ii[ii > i]
        pair_list.extend((i, int(j), float(dd[np.where(ii == j)[0][0]])) for j in js)

        if not use_tqdm:
            now = time()
            if now - last_print > 5:
                print(f"  processed {i+1}/{N} nodes")
                last_print = now

    return nbr_idx, nbr_dst, pair_list

def _choose_nside_for_radius(radius_rad: float) -> int:
    """
    Choose HEALPix nside so that the equivalent circle radius of a pixel is <= ~radius_rad/2.
    Pixel area = 4π / (12 nside^2); r_eq = sqrt(area/π).
    """
    if radius_rad <= 0:
        return 64
    target = radius_rad / 2.0
    r_eq = target
    area = math.pi * (r_eq ** 2)
    nside = int(np.sqrt((4.0 * math.pi) / (12.0 * area)))
    nside = max(1, min(32768, nside))
    return nside

def _neighbors_within_maxdist(latlon: np.ndarray, max_edge_m: float, progress_msg: str = "Neighbor search"):
    """
    For each point i, compute neighbors j!=i within max_edge_m, and distances.
    Returns:
      nbr_idx:  list of np.array(int) neighbors for each i
      nbr_dst:  list of np.array(float) distances (meters) aligned with nbr_idx
      pair_list: list of (i,j,dij) for all i<j within max_edge_m (candidate edges)
    """
    N = latlon.shape[0]
    nbr_idx: List[np.ndarray] = [None]*N
    nbr_dst: List[np.ndarray] = [None]*N
    pair_list: List[Tuple[int,int,float]] = []

    use_tqdm = False
    try:
        from tqdm import tqdm as _tq
        it = _tq(range(N), desc=progress_msg)
        use_tqdm = True
    except Exception:
        it = range(N)
        last_print = time()
        print(f"{progress_msg}...")

    for i in it:
        d = haversine_m_vec(latlon, latlon[i,0], latlon[i,1])
        d[i] = np.inf
        mask = (d <= max_edge_m)
        idxs = np.nonzero(mask)[0]
        nbr_idx[i] = idxs
        nbr_dst[i] = d[idxs].astype(float)
        # collect pairs i<j
        js = idxs[idxs > i]
        pair_list.extend((i, int(j), float(d[int(j)])) for j in js)

        if not use_tqdm:
            now = time()
            if now - last_print > 5:
                print(f"  processed {i+1}/{N} nodes")
                last_print = now

    return nbr_idx, nbr_dst, pair_list

def _neighbors_within_maxdist_indexed(latlon: np.ndarray, max_edge_m: float, method: str, progress_msg: str):
    method = (method or "balltree").lower()
    if method == "balltree":
        return _neighbors_within_maxdist_balltree(latlon, max_edge_m, progress_msg=progress_msg)
    elif method == "bruteforce":
        return _neighbors_within_maxdist(latlon, max_edge_m, progress_msg=progress_msg)
    else:
        raise ValueError(f"Unknown neighbor index method: {method}")
 
def build_edges_rng_or_gabriel(
    latlon: np.ndarray,
    nbr_idx: List[np.ndarray],
    nbr_dst: List[np.ndarray],
    pair_list: List[Tuple[int,int,float]],
    criterion: str = "rng",
    progress_interval_s: int = 5,
) -> List[Tuple[int,int,float]]:
    """
    RNG: keep (i,j) if NO c with max(d(i,c),d(j,c)) < d(i,j).
    Gabriel: keep (i,j) if NO c with d(i,c)^2 + d(j,c)^2 < d(i,j)^2.
    """
    keep: List[Tuple[int,int,float]] = []

    Npairs = len(pair_list)
    try:
        from tqdm import tqdm as _tq
        it = _tq(pair_list, desc=f"Edge test ({criterion})", total=Npairs)
        use_bar = True
    except Exception:
        it = pair_list
        use_bar = False
        last_print = time()
        print(f"Edge test ({criterion}) over {Npairs:,} candidate pairs...")

    for k, (i, j, dij) in enumerate(it):
        # neighbors with dist < dij (strict)
        ni = nbr_idx[i]
        di = nbr_dst[i]
        nj = nbr_idx[j]
        dj = nbr_dst[j]
        si = ni[di < dij]
        sj = nj[dj < dij]

        if si.size == 0 or sj.size == 0:
            keep.append((i,j,dij))
            continue

        # intersection candidates
        cand = np.intersect1d(si, sj, assume_unique=False)
        if cand.size == 0:
            keep.append((i,j,dij))
            continue

        if criterion == "rng":
            # by construction, all candidates already satisfy d(i,c) < dij and d(j,c) < dij
            # RNG condition met -> the pair (i,j) should be **removed** if ANY such c exists
            # since cand.size>0, we skip adding this edge
            continue
        else:
            # Gabriel: need to check stricter disk condition
            # Gather distances d(i,c) and d(j,c) for cand
            # Map cand -> indices in ni/nj
            # Efficient approach: build dict from neighbor id -> distance for i & j
            di_map = {int(ni_t): float(di_t) for ni_t, di_t in zip(ni, di)}
            dj_map = {int(nj_t): float(dj_t) for nj_t, dj_t in zip(nj, dj)}
            dij2 = dij * dij
            blocked = False
            for c in cand:
                dic = di_map[int(c)]; djc = dj_map[int(c)]
                if (dic*dic + djc*djc) < dij2:
                    blocked = True
                    break
            if not blocked:
                keep.append((i,j,dij))

        if (not use_bar) and (time() - last_print > progress_interval_s):
            pct = (k+1) * 100.0 / Npairs if Npairs else 100.0
            print(f"  tested {k+1:,}/{Npairs:,} pairs ({pct:.1f}%)")
            last_print = time()

    return keep

# -------------------------
# CSV writers
# -------------------------
def write_vertices_csv(df: pd.DataFrame, out_dir: Path):
    p = out_dir / "vertices.csv"
    # Write with IS_AIRPORT (0/1) so downstream knows which vertex ids are airports
    cols = ["IDENTIFIER","LAT","LON","ALTITUDE","IS_AIRPORT"]
    for c in cols:
        if c not in df.columns:
            # Back-compat: if missing, synthesize non-airport flag
            if c == "IS_AIRPORT":
                df = df.assign(IS_AIRPORT=0)
            else:
                raise ValueError(f"vertices.csv missing required column: {c}")
    df.to_csv(p, index=False, columns=cols)

def write_edges_csv(edges: List[Tuple[int,int,float]], out_dir: Path):
    p = out_dir / "edges.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["V0","V1","D"])
        for i,j,d in edges:
            if j < i:
                i,j = j,i
            w.writerow([i, j, f"{d:.3f}"])

# -------------------------
# Connectivity enforcement
# -------------------------
def _graph_from_edges(n_nodes: int, edges: List[Tuple[int,int,float]]) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_weighted_edges_from(edges, weight="D")
    return G

def _component_labels(G: nx.Graph) -> Tuple[np.ndarray, List[List[int]]]:
    comps = [sorted(list(c)) for c in nx.connected_components(G)]
    label = np.empty(G.number_of_nodes(), dtype=int)
    for cid, nodes in enumerate(comps):
        label[nodes] = cid
    return label, comps

def _centroids(latlon: np.ndarray, comps: List[List[int]]) -> np.ndarray:
    # centroid in (lat,lon) deg; OK for smallish patches; for world we only use them
    # to seed candidate pairs (final distances use haversine at point level).
    C = np.zeros((len(comps), 2), dtype=float)
    for i, idx in enumerate(comps):
        pts = latlon[np.asarray(idx)]
        C[i] = pts.mean(axis=0)
    return C

def _haversine_rad(u_rad: np.ndarray, v_rad: np.ndarray) -> np.ndarray:
    # pairwise haversine distances between rows of u_rad and v_rad (radians), returns meters
    # used only for small centroid kNN; for large sets we use BallTree below
    EARTH_R_M = 6371008.8
    uφ, uλ = u_rad[:,0:1], u_rad[:,1:2]
    vφ, vλ = v_rad[None,:,0], v_rad[None,:,1]
    dφ = vφ - uφ
    dλ = vλ - uλ
    a = np.sin(dφ/2.0)**2 + np.cos(uφ)*np.cos(vφ)*np.sin(dλ/2.0)**2
    return 2.0 * EARTH_R_M * np.arcsin(np.sqrt(a))

def _closest_pair_between_sets(idx_a: np.ndarray, idx_b: np.ndarray, latlon: np.ndarray) -> Tuple[int,int,float]:
    """Return (ia, ib, d_m) for the closest pair across two components using BallTree."""
    from sklearn.neighbors import BallTree
    A = latlon[idx_a]
    B = latlon[idx_b]
    # query from smaller → larger
    if len(A) <= len(B):
        q_idx, q_pts = idx_a, A
        t_idx, t_pts = idx_b, B
        flip = False
    else:
        q_idx, q_pts = idx_b, B
        t_idx, t_pts = idx_a, A
        flip = True
    tree = BallTree(np.deg2rad(t_pts), metric="haversine")
    dist_rad, nn = tree.query(np.deg2rad(q_pts), k=1, return_distance=True)
    k = int(dist_rad.argmin())
    d_m = float(dist_rad[k,0] * 6371008.8)
    q_node = int(q_idx[k])
    t_node = int(t_idx[int(nn[k,0])])
    if flip:
        return t_node, q_node, d_m
    return q_node, t_node, d_m

def ensure_connected(latlon: np.ndarray,
                     edges_kept: List[Tuple[int,int,float]],
                     k_centroid_nn: int = 12,
                     method: str = "mst") -> Tuple[List[Tuple[int,int,float]], int]:
    """
    If graph has >1 CC, add minimal set of bridging edges to connect all CCs.
    Strategy (default 'mst'):
      1) Compute components.
      2) Build k-NN graph on component centroids (in haversine).
      3) Compute MST over component centroids.
      4) For each MST edge (comp u, comp v), add the *closest vertex pair*
         across the two comps (via BallTree).
    Returns: (augmented_edges, num_added)
    """
    N = latlon.shape[0]
    G = _graph_from_edges(N, edges_kept)
    _, comps = _component_labels(G)
    if len(comps) <= 1:
        return edges_kept, 0

    print(f"[connectivity] Components before: {len(comps)}")
    C = _centroids(latlon, comps)
    C_rad = np.deg2rad(C)

    # Build sparse kNN graph among centroids
    try:
        if method == "greedy":
            raise Exception("Fallback to greedy")

        from sklearn.neighbors import BallTree
        tree = BallTree(C_rad, metric="haversine")
        k = min(k_centroid_nn+1, len(comps))  # +1 because self is included
        dist, nbrs = tree.query(C_rad, k=k, return_distance=True)
        # Build edge list (avoid self, make undirected unique i<j)
        comp_edges = set()
        for i in range(len(comps)):
            for kk in range(1, nbrs.shape[1]):
                j = int(nbrs[i,kk])
                if i == j: 
                    continue
                u, v = (i,j) if i<j else (j,i)
                comp_edges.add((u,v))
        # Assign centroid edge weights by great-circle meters
        weights = []
        for (u,v) in comp_edges:
            d_m = _haversine_rad(C_rad[[u]], C_rad[[v]])[0,0]
            weights.append((u,v,float(d_m)))
        H = nx.Graph()
        H.add_weighted_edges_from(weights, weight="w")
        T = nx.minimum_spanning_tree(H, weight="w")
        comp_pairs = list(T.edges())
    except Exception:
        # Fallback: greedy chain by nearest centroid (may be slower/different)
        D = _haversine_rad(C_rad, C_rad)
        np.fill_diagonal(D, np.inf)
        parent = list(range(len(comps)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        comp_pairs = []
        # Greedy: repeatedly link closest pair of current trees
        while len({find(i) for i in range(len(comps))}) > 1:
            u, v = np.unravel_index(np.argmin(D), D.shape)
            if find(u) != find(v):
                parent[find(u)] = find(v)
                comp_pairs.append((u,v))
            D[u,v] = D[v,u] = np.inf

    # For each component pair, add the actual closest vertex pair
    added = []
    for (cu, cv) in comp_pairs:
        ia, ib, dm = _closest_pair_between_sets(np.asarray(comps[cu]), np.asarray(comps[cv]), latlon)
        added.append((int(ia), int(ib), float(dm)))

    print(f"[connectivity] Adding {len(added)} bridging edge(s) to enforce a single connected graph.")
    out_edges = list(edges_kept) + added
    return out_edges, len(added)



# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    def str2bool(v: str) -> bool:
        if isinstance(v, bool): return v
        v = v.strip().lower()
        if v in ("y","yes","true","1","on"): return True
        if v in ("n","no","false","0","off"): return False
        raise argparse.ArgumentTypeError(f"Invalid bool: {v}")

    p = argparse.ArgumentParser(description="Build Voronoi-style navpoint graph (RNG/Gabriel) with region filtering.")
    p.add_argument("--config", type=Path, default=None, help="JSON with considered_geographic_regions polygons (same as model builder).")
    p.add_argument("--ourairports", type=Path, default=Path("./ourairports/airports.csv"), help="OurAirports airports.csv")
    p.add_argument("--navdir", type=Path, default=Path("./test_navpoints"), help="Folder containing fix.dat and nav.dat")
    p.add_argument("--od-file", type=Path, default=None, help="Optional CSV with columns origin,destination to restrict airports.")
    p.add_argument("--aircrafts-file", type=Path, default=None, help="Optional aircrafts.csv (speeds); loaded but not used for edges.")
    p.add_argument("--icao-only", type=str, default="true", help="Keep only 4-letter ICAO airports (default true)")
    p.add_argument("--criterion", type=str, default="rng", choices=["rng","gabriel"], help="Edge test: relative-neighborhood (rng) or gabriel.")
    p.add_argument("--max-edge-km", type=float, default=350.0, help="Only consider edges <= this geodesic distance (km).")
    p.add_argument("--out-dir", type=Path, default=Path("./navgraph_out"), help="Output folder for vertices.csv & edges.csv")
    p.add_argument("--neighbor-index", type=str, default="balltree",
                   choices=["balltree","bruteforce"], help="Spatial index for neighbor search.")
    p.add_argument("--progress-interval", type=int, default=5, help="Seconds between ETA prints when tqdm is unavailable.")
    p.add_argument("--enforce-connected", type=str, default="true",
                   help="Ensure the final graph is a single connected component (default true).")
    p.add_argument("--connectivity-method", type=str, default="mst", choices=["mst","greedy"],
                   help="How to choose bridging component pairs (default mst).")
    p.add_argument("--centroid-knn", type=int, default=12, help="k for centroid kNN (default 12).")
    p.add_argument("--flat-out", action="store_true",
                   help="Write files directly into --out-dir (disable auto-named subfolder).")
    return p.parse_args()

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    args.icao_only = True if str(args.icao_only).strip().lower() in ("true","t","1","yes","on") else False

    # Resolve connectivity flag early (used in tag)
    do_connect = str(args.enforce_connected).strip().lower() in ("true","t","1","yes","on")

    # --- build auto-named experiment directory ---
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    if args.flat_out:
        exp_dir = out_root
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
        tag = (
            f"crit-{args.criterion}"
            f"_max{int(args.max_edge_km)}km"
            f"_idx-{args.neighbor_index}"
            f"_conn{'T' if do_connect else 'F'}knn{int(args.centroid_knn)}"
            f"_icao{'T' if args.icao_only else 'F'}"
            f"_nav-{args.navdir.name}_ap-{args.ourairports.name}"
        )
        exp_dir = out_root / f"{ts}__{tag}"
    exp_dir.mkdir(parents=True, exist_ok=True)
 

    # 1) Load config regions
    regions = load_regions_from_config(args.config) if args.config else []
    if regions:
        print(f"[1/6] Loaded {len(regions)} geographic region(s) from {args.config}")

    # 2) Load OD pairs & aircrafts (optional)
    od_df = load_od_pairs(args.od_file)
    if not od_df.empty:
        print(f"[2/6] Loaded OD pairs: {len(od_df):,} rows (restricting airports to observed ICAOs).")
    ac_df = load_aircrafts(args.aircrafts_file)
    if not ac_df.empty:
        nspd = ac_df["speed_kts"].notna().sum() if "speed_kts" in ac_df.columns else 0
        print(f"[2/6] Loaded aircrafts: {len(ac_df):,} rows ({nspd:,} with speed_kts).")

    # 3) Build vertices (airports + navpoints) with region filtering
    print(f"[3/6] Loading airports and navpoints...")
    vertices = build_vertices(
        airports_csv=args.ourairports,
        nav_dir=args.navdir,
        regions=regions,
        icao_only=args.icao_only,
        od_filter=od_df if not od_df.empty else None,
    )
    N = len(vertices)
    if N == 0:
        raise RuntimeError("No vertices after filtering. Check inputs/regions.")
    print(f"      Kept {N:,} vertices.")

    # 4) Save vertices.csv
    write_vertices_csv(vertices, exp_dir)
    print(f"[4/6] Wrote vertices.csv -> {exp_dir/'vertices.csv'}")
 
    # 5) Neighbor search (pre-candidate edges within max distance) with index
    print(f"[5/6] Neighbor search within {args.max_edge_km:g} km using '{args.neighbor_index}'...")
    latlon = vertices[["LAT","LON"]].to_numpy(dtype=float)
    nbr_idx, nbr_dst, pair_list = _neighbors_within_maxdist_indexed(
        latlon, max_edge_m=args.max_edge_km*1000.0, method=args.neighbor_index, progress_msg="Neighbor search"
    )

    #nbr_idx, nbr_dst, pair_list = _neighbors_within_maxdist(latlon, max_edge_m=args.max_edge_km*1000.0, progress_msg="Neighbor search")
    print(f"      Candidate pairs: {len(pair_list):,}")

    # 6) Edge test (RNG / Gabriel)
    print(f"[6/6] Testing candidate edges with '{args.criterion}' criterion...")
    edges_kept = build_edges_rng_or_gabriel(
        latlon=latlon,
        nbr_idx=nbr_idx,
        nbr_dst=nbr_dst,
        pair_list=pair_list,
        criterion=args.criterion.lower(),
        progress_interval_s=int(args.progress_interval),
    )
    print(f"      Kept edges: {len(edges_kept):,}")

    # 7) Enforce connectivity (optional, default on)
    if do_connect:
        print("[7/7] Enforcing connectivity...")
        latlon = vertices[["LAT","LON"]].to_numpy(dtype=float)
        edges_kept, n_added = ensure_connected(
            latlon=latlon,
            edges_kept=edges_kept,
            k_centroid_nn=int(args.centroid_knn),
            method=args.connectivity_method.lower(),
        )
        # quick report
        G_final = _graph_from_edges(len(vertices), edges_kept)
        k_final = nx.number_connected_components(G_final)
        print(f"      Components after: {k_final} (added {n_added} bridging edge(s))")
    else:
        print("[7/7] Skipped connectivity enforcement by user request.")


    # Write edges.csv
    write_edges_csv(edges_kept, exp_dir)

    # Persist a run_config.json for traceability (paths as strings)
    try:
        import json
        payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        payload["resolved_out_dir"] = str(exp_dir.resolve())
        payload["stats"] = {
            "num_vertices": int(N),
            "num_edges_written": int(len(edges_kept)),
            "enforce_connected": bool(do_connect),
        }
        with open(exp_dir / "run_config.json", "w") as fh:
            json.dump(payload, fh, indent=2)
    except Exception:
        pass

    print(f"Done. Wrote edges.csv and vertices.csv to {exp_dir.resolve()}")
 

if __name__ == "__main__":
    main()
