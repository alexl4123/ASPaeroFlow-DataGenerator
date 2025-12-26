#!/usr/bin/env python3
"""
Navpoint/Airport Voronoi-style graph builder

Vertices: airports (OurAirports) + navpoints (X-Plane fix.dat + nav.dat)
Edges: undirected (A,B) if there is **no** third vertex C that is "closer along the way"
      —implemented via a Relative Neighborhood Graph (RNG) or Gabriel test:

  RNG (default): keep (A,B) iff there is no C with max(d(A,C), d(B,C)) < d(A,B)
  Gabriel:      keep (A,B) iff there is no C with d(A,C)^2 + d(B,C)^2 < d(A,B)^2

Grid override (when --grid-navpoints true):
  To avoid airports "blocking" adjacency between grid points, edges are NOT
  computed via RNG/Gabriel. Instead, grid navpoints are connected by 8-neighborhood:
    connect (x,y) to (x2,y2) iff |x-x2|<=1 and |y-y2|<=1, excluding self.
  Airports are NOT used for grid connectivity, but are attached to the grid by connecting
  each airport to its closest grid navpoint(s) (closest-neighbor rule, as before).
 


Distances are great-circle (haversine), reported in **meters** for edges.

Region restriction:
  Reads the same JSON config you used in the model builder, and applies any
  `considered_geographic_regions` polygons (lat,lon pairs, ray-cast point-in-poly).

Outputs:
  - vertices.csv with columns: IDENTIFIER,LAT,LON,ALTITUDE,IS_AIRPORT
  - edges.csv    with columns: V0,V1,D    (vertex IDENTIFIER pairs, D in meters)

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
import re
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
# Helpers: airport include list
# -------------------------
def parse_airport_include_spec(spec: str | None) -> set[str] | None:
    """
    Accepts either:
      - comma/space/semicolon separated ICAO codes (e.g., "LOWW, EDDM")
      - a path to a text/CSV file containing ICAO codes (anywhere in the file)

    Returns a set of 4-letter ICAO codes (uppercased), or None if not provided.
    """
    if spec is None:
        return None
    s = str(spec).strip()
    if not s:
        return None

    p = Path(s)
    if p.exists() and p.is_file():
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = p.read_text(errors="ignore")
    else:
        text = s

    toks = re.split(r"[,\s;]+", text)
    codes = [t.strip().upper() for t in toks if t and t.strip()]
    # Keep strictly ICAO-like tokens (4 letters)
    codes = [c for c in codes if re.fullmatch(r"[A-Z]{4}", c)]
    return set(codes) if codes else None

# -------------------------
# Geo helpers
# -------------------------
EARTH_R_M = 6371008.8  # mean Earth radius (meters)

# Altitude helpers
FT_PER_FL = 100.0
M_PER_FT = 0.3048
def to_altitude_m(alt_value: float, unit: str) -> float:
    unit = (unit or "m").lower()
    if unit == "fl":
        return float(alt_value * FT_PER_FL * M_PER_FT)  # FL * 100 ft → meters
    if unit == "m":
        return float(alt_value)
    raise ValueError("Invalid --altitude-unit: use 'm' or 'fl'")


def chord_distance_3d_m(lat1, lon1, alt1, lat2, lon2, alt2) -> float:
    """Straight-line distance through 3D space (meters) on a spherical Earth.""" 
    #φ1, λ1, φ2, λ2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    #r1 = EARTH_R_M + float(alt1)
    #r2 = EARTH_R_M + float(alt2)
    #cosγ = np.sin(φ1)*np.sin(φ2) + np.cos(φ1)*np.cos(φ2)*np.cos(λ2 - λ1)
    #d2 = r1*r1 + r2*r2 - 2.0*r1*r2*cosγ
    #return float(np.sqrt(max(0.0, d2)))

    """Great-circle arc at effective radius (avg altitude), combined with vertical Δh (meters)."""
    φ1, λ1, φ2, λ2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = np.sin(dφ/2.0)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(dλ/2.0)**2
    a = float(np.clip(a, 0.0, 1.0))
    γ = 2.0 * np.arcsin(np.sqrt(a))  # central angle (rad)
    r_eff = EARTH_R_M + 0.5*(float(alt1) + float(alt2))
    L_h = γ * r_eff
    dh = float(alt2) - float(alt1)
    return float(np.hypot(L_h, dh))

def chord_distance_3d_m_vec(latlonalt: np.ndarray, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """
    Vectorized 3D chord distance from (lat,lon,alt) to many points (meters).
    latlonalt: shape (N,3) with columns [LAT, LON, ALTITUDE]
    """
    #φ1 = np.deg2rad(latlonalt[:,0]); λ1 = np.deg2rad(latlonalt[:,1])
    #r1 = EARTH_R_M + latlonalt[:,2].astype(float)
    #φ2 = np.deg2rad(lat_deg); λ2 = np.deg2rad(lon_deg)
    #r2 = EARTH_R_M + float(alt_m)
    #cosγ = np.sin(φ1)*np.sin(φ2) + np.cos(φ1)*np.cos(φ2)*np.cos(λ2 - λ1)
    #d2 = r1*r1 + r2*r2 - 2.0*r1*r2*cosγ
    #return np.sqrt(np.maximum(0.0, d2))

    """
    Vectorized GC+alt distance to many points (meters).
    latlonalt: (N,3) columns [LAT, LON, ALTITUDE]
    """
    φ1 = np.deg2rad(latlonalt[:,0]); λ1 = np.deg2rad(latlonalt[:,1])
    φ2 = np.deg2rad(lat_deg);         λ2 = np.deg2rad(lon_deg)
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = np.sin(dφ/2.0)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(dλ/2.0)**2
    a = np.clip(a, 0.0, 1.0)
    γ = 2.0 * np.arcsin(np.sqrt(a))
    r_eff = EARTH_R_M + 0.5*(latlonalt[:,2].astype(float) + float(alt_m))
    L_h = γ * r_eff
    dh = latlonalt[:,2].astype(float) - float(alt_m)
    return np.hypot(L_h, dh)

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


def load_region_items_from_config(path: Path) -> List[Tuple[str, List[Tuple[float,float]]]]:
    """
    Returns (region_name, polygon_pairs) from config.
    Keeps compatibility with older configs by tolerating missing names.
    """
    if not path or not path.exists():
        return []
    with open(path, "r") as fh:
        cfg = json.load(fh) or {}
    out: List[Tuple[str, List[Tuple[float,float]]]] = []
    for k, item in enumerate(cfg.get("considered_geographic_regions", [])):
        name = (item.get("region-name") or item.get("region_name") or f"REGION_{k}").strip()
        poly = _pairs_from_flat_polygon(item.get("polygon", []))
        out.append((str(name), poly))
    return out

def _bbox_from_poly(poly: List[Tuple[float,float]]) -> Tuple[float,float,float,float]:
    """Return (lat_min, lat_max, lon_min, lon_max) from polygon vertices."""
    lats = [p[0] for p in poly]
    lons = [p[1] for p in poly]
    return (float(min(lats)), float(max(lats)), float(min(lons)), float(max(lons)))

def in_any_bbox(lat: float, lon: float, bboxes: List[Tuple[float,float,float,float]]) -> bool:
    if not bboxes:
        return True
    for (lat_min, lat_max, lon_min, lon_max) in bboxes:
        if (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max):
            return True
    return False

def build_grid_navpoints_df(
    bboxes: List[Tuple[float,float,float,float]],
    nx: int,
    ny: int,
    altitude_m: float,
    prefix: str = "GRID",
    region_names: List[str] | None = None,
) -> pd.DataFrame:
    """
    Build a rectangular grid of navpoints for each bbox.
    Grid points are placed at tile centers (nx by ny tiles).
    """
    if nx < 1 or ny < 1:
        raise ValueError("Grid dimensions must satisfy nx>=1 and ny>=1")
    rows: List[Tuple[str, float, float]] = []
    for ridx, (lat_min, lat_max, lon_min, lon_max) in enumerate(bboxes):
        # Ensure ordering (south<north, west<east)
        lat_s, lat_n = (lat_min, lat_max) if lat_min <= lat_max else (lat_max, lat_min)
        lon_w, lon_e = (lon_min, lon_max) if lon_min <= lon_max else (lon_max, lon_min)
        dlat = (lat_n - lat_s) / float(ny)
        dlon = (lon_e - lon_w) / float(nx)

        rname = (region_names[ridx] if region_names and ridx < len(region_names) else f"R{ridx}")
        safe_rname = "".join(ch if (ch.isalnum() or ch in ("-","_")) else "_" for ch in str(rname)).upper()

        for j in range(ny):      # south -> north
            lat = lat_s + (j + 0.5) * dlat
            for i in range(nx):  # west -> east
                lon = lon_w + (i + 0.5) * dlon
                ident = f"{prefix}_{safe_rname}_Y{j:02d}X{i:02d}"
                rows.append((ident, float(lat), float(lon)))

    df = pd.DataFrame(rows, columns=["IDENTIFIER","LAT","LON"])
    df["ALTITUDE"] = float(altitude_m)
    df["IS_AIRPORT"] = 0
    return df.drop_duplicates(subset=["IDENTIFIER"])


def parse_grid_identifier(ident: str) -> Tuple[int, int] | None:
    """
    Parse grid identifiers of the form '*_Y<yy>X<xx>' and return (y, x) as ints.
    Returns None if the identifier does not match the expected pattern.
    """
    m = re.search(r"_Y(\d+)X(\d+)$", str(ident).strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))

def build_grid_edges_8nb(vertices: pd.DataFrame) -> List[Tuple[int, int, float]]:
    """
    Build 8-neighborhood edges BETWEEN GRID NAVPOINTS ONLY.
    i.e., connect (x,y) <-> (x2,y2) iff |x-x2|<=1 and |y-y2|<=1, excluding self.
    Distances are computed using chord_distance_3d_m on the per-vertex ALTITUDE.
    """
    g = vertices[vertices["IS_AIRPORT"].astype(int) == 0].copy()
    g["_yx"] = g["IDENTIFIER"].map(parse_grid_identifier)
    g = g[g["_yx"].notna()]
    if g.empty:
        return []
    # map (y,x) -> global vertex index
    yx_to_idx: Dict[Tuple[int,int], int] = {tuple(yx): int(i) for i, yx in zip(g.index, g["_yx"])}
    lat = vertices["LAT"].to_numpy(dtype=float)
    lon = vertices["LON"].to_numpy(dtype=float)
    alt = vertices["ALTITUDE"].to_numpy(dtype=float)
    edges: List[Tuple[int,int,float]] = []
    for (y, x), i in yx_to_idx.items():
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                j = yx_to_idx.get((y + dy, x + dx))
                if j is None or j <= i:
                    continue
                d = chord_distance_3d_m(lat[i], lon[i], alt[i], lat[j], lon[j], alt[j])
                edges.append((i, j, float(d)))
    return edges

def build_airport_to_nav_edges(
    vertices: pd.DataFrame,
    *,
    k_nearest: int = 3,
    max_edge_m: float | None = None,
) -> List[Tuple[int, int, float]]:
    """
    Attach airports to navpoints by connecting each airport to its closest navpoint(s).
    - Uses BallTree (haversine) if available; falls back to brute force otherwise.
    - Distances for edges are computed using chord_distance_3d_m (3D with ALTITUDE).
    - If max_edge_m is provided, we prefer neighbors within that radius, but always
      ensure at least one attachment edge per airport (to its closest navpoint).
    """
    ap_mask = vertices["IS_AIRPORT"].astype(int) == 1
    nav_mask = vertices["IS_AIRPORT"].astype(int) == 0
    ap_idx = vertices.index[ap_mask].to_numpy(dtype=int)
    nav_idx = vertices.index[nav_mask].to_numpy(dtype=int)
    if ap_idx.size == 0 or nav_idx.size == 0:
        return []

    lat = vertices["LAT"].to_numpy(dtype=float)
    lon = vertices["LON"].to_numpy(dtype=float)
    alt = vertices["ALTITUDE"].to_numpy(dtype=float)

    kq = int(max(1, min(k_nearest, nav_idx.size)))
    edges: List[Tuple[int, int, float]] = []

    # Prefer BallTree (fast for large grids), fallback to brute force.
    try:
        from sklearn.neighbors import BallTree
        nav_rad = np.deg2rad(np.column_stack([lat[nav_idx], lon[nav_idx]]))
        tree = BallTree(nav_rad, metric="haversine")
        ap_rad = np.deg2rad(np.column_stack([lat[ap_idx], lon[ap_idx]]))
        dist_rad, nn = tree.query(ap_rad, k=kq, return_distance=True)

        for row, ai in enumerate(ap_idx):
            chosen: List[Tuple[int, float]] = []
            # Try to keep within max_edge_m if provided
            for r in range(kq):
                nj_local = int(nn[row, r])
                aj = int(nav_idx[nj_local])
                d_m = chord_distance_3d_m(lat[ai], lon[ai], alt[ai], lat[aj], lon[aj], alt[aj])
                if (max_edge_m is None) or (d_m <= float(max_edge_m)):
                    chosen.append((aj, float(d_m)))
            # Guarantee at least one attachment (closest) even if outside max_edge_m
            if not chosen:
                aj = int(nav_idx[int(nn[row, 0])])
                d_m = chord_distance_3d_m(lat[ai], lon[ai], alt[ai], lat[aj], lon[aj], alt[aj])
                chosen = [(aj, float(d_m))]
            for aj, d_m in chosen:
                u, v = (ai, aj) if ai < aj else (aj, ai)
                edges.append((u, v, float(d_m)))
        return edges

    except Exception:
        nav_latlonalt = np.column_stack([lat[nav_idx], lon[nav_idx], alt[nav_idx]]).astype(float)
        for ai in ap_idx:
            d_all = chord_distance_3d_m_vec(nav_latlonalt, lat[ai], lon[ai], alt[ai]).astype(float)
            order = np.argsort(d_all)
            chosen: List[Tuple[int, float]] = []
            for pos in order[:kq]:
                aj = int(nav_idx[int(pos)])
                d_m = float(d_all[int(pos)])
                if (max_edge_m is None) or (d_m <= float(max_edge_m)):
                    chosen.append((aj, d_m))
            if not chosen:
                aj = int(nav_idx[int(order[0])])
                chosen = [(aj, float(d_all[int(order[0])]))]
            for aj, d_m in chosen:
                u, v = (ai, aj) if ai < aj else (aj, ai)
                edges.append((u, v, float(d_m)))
        return edges





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
    altitude_m: float = 0.0,
    airport_include: set[str] | None = None,
    # Grid navpoint mode (if enabled, ignore fix.dat/nav.dat and generate artificial grid navpoints)
    grid_only: bool = False,
    grid_nx: int = 0,
    grid_ny: int = 0,
    grid_bboxes: List[Tuple[float,float,float,float]] | None = None,
    grid_prefix: str = "GRID",
    grid_region_names: List[str] | None = None,
    # If True, region filtering uses bbox(es) instead of polygon ray-cast (useful for grid mode)
    region_filter_use_bbox: bool = False,
    region_bboxes: List[Tuple[float,float,float,float]] | None = None,
    ) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: IDENTIFIER, LAT, LON, ALTITUDE (float, meters)
    """
    # Airports
    ap = load_ourairports_df(airports_csv, icao_only=icao_only)

    ap = ap.rename(columns={"ident":"IDENTIFIER","lat":"LAT","lon":"LON"})
    # In grid mode: airports at FL0 (0m). Otherwise: keep legacy uniform-altitude behavior.
    ap["ALTITUDE"] = 0.0 if grid_only else float(altitude_m)
    ap["IS_AIRPORT"] = 1

    # Optional: hard inclusion list for airports (ICAO codes)
    if airport_include:
        # IDENTIFIER is already uppercased in loader, but keep this defensive
        ap["IDENTIFIER"] = ap["IDENTIFIER"].astype(str).str.strip().str.upper()
        ap = ap[ap["IDENTIFIER"].isin(set(airport_include))]
 

    # If OD provided: keep only airports that appear in origin or destination
    #if od_filter is not None and not od_filter.empty:
    #    ap = ap[ap["IDENTIFIER"].isin(set(od_filter["origin"]).union(set(od_filter["destination"])))]

    # Navpoints
    nav_cols = ["IDENTIFIER","LAT","LON","ALTITUDE","IS_AIRPORT"]
    if grid_only:
        if not grid_bboxes:
            raise ValueError("grid_only=True but no grid_bboxes were provided.")
        nav = build_grid_navpoints_df(
            bboxes=grid_bboxes,
            nx=int(grid_nx),
            ny=int(grid_ny),
            altitude_m=float(altitude_m),
            prefix=str(grid_prefix or "GRID"),
            region_names=grid_region_names,
        )
        nav = nav[nav_cols]
    else:
        fix_path = nav_dir / "fix.dat"
        nav_path = nav_dir / "nav.dat"
        nav_frames = []
        if fix_path.exists():
            fdf = parse_fix(fix_path).rename(columns={"ident":"IDENTIFIER","lat":"LAT","lon":"LON"})
            fdf["ALTITUDE"] = float(altitude_m)
            fdf["IS_AIRPORT"] = 0
            nav_frames.append(fdf)
        if nav_path.exists():
            ndf = parse_nav(nav_path).rename(columns={"ident":"IDENTIFIER","lat":"LAT","lon":"LON"})
            ndf["ALTITUDE"] = float(altitude_m)
            ndf["IS_AIRPORT"] = 0
            nav_frames.append(ndf)
        nav = pd.concat(nav_frames, ignore_index=True) if nav_frames else pd.DataFrame(columns=nav_cols)

    # Merge, preferring airport rows on IDENTIFIER collisions
    allv = pd.concat([ap, nav], ignore_index=True)

    allv.drop_duplicates(subset=["IDENTIFIER"], keep="first", inplace=True)  # airports first

    # Important: in non-grid mode, legacy behavior was "uniform altitude for all vertices".
    # In grid mode, we need airports at 0 and navpoints at altitude_m. We already set airports
    # to 0.0 above; grid/navpoints set to altitude_m. Nothing else to do here.
    # Region filter
    if region_filter_use_bbox and region_bboxes:
        mask = allv.apply(lambda r: in_any_bbox(float(r["LAT"]), float(r["LON"]), region_bboxes), axis=1)
        allv = allv[mask]
    elif regions:
        mask = allv.apply(lambda r: in_any_region(float(r["LAT"]), float(r["LON"]), regions), axis=1)
        allv = allv[mask]

    # Clean NaNs and outliers
    allv = allv.dropna(subset=["IDENTIFIER","LAT","LON"]).copy()
    allv["IDENTIFIER"] = allv["IDENTIFIER"].astype(str).str.strip().str.upper()

    # Reindex to have stable 0..N-1 ids
    allv.reset_index(drop=True, inplace=True)
    return allv[["IDENTIFIER","LAT","LON","ALTITUDE","IS_AIRPORT"]]


def _neighbors_within_maxdist_balltree(latlonalt: np.ndarray, max_edge_m: float, progress_msg: str = "Neighbor search"):
    """
    Use scikit-learn BallTree with haversine metric (expects radians).
    Returns (nbr_idx, nbr_dst, pair_list) like the brute-force version.
    Distances returned in meters.
    """
    try:
        from sklearn.neighbors import BallTree
    except Exception as e:
        raise RuntimeError("BallTree requested but scikit-learn is not available. Install scikit-learn.") from e

    N = latlonalt.shape[0]
    rad = np.deg2rad(latlonalt[:, :2])  # columns: [lat, lon] in radians
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
        # 3D distances to candidates
        dd3d_all = chord_distance_3d_m_vec(latlonalt, latlonalt[i,0], latlonalt[i,1], latlonalt[i,2])
        # remove self and apply 3D cutoff
        mask = (ii != i)
        ii = ii[mask]
        dd3d = dd3d_all[ii].astype(float)
        mask = (dd3d <= max_edge_m)
        ii = ii[mask]; dd3d = dd3d[mask]
        nbr_idx[i] = ii
        nbr_dst[i] = dd3d

        # collect i<j
        js = ii[ii > i]
        # map each j to its 3D distance
        pair_list.extend((i, int(j), float(dd3d[np.where(ii == j)[0][0]])) for j in js)

        if not use_tqdm:
            now = time()
            if now - last_print > 5:
                print(f"  processed {i+1}/{N} nodes")
                last_print = now

    return nbr_idx, nbr_dst, pair_list

def _neighbors_within_maxdist(latlonalt: np.ndarray, max_edge_m: float, progress_msg: str = "Neighbor search"):
    """
    For each point i, compute neighbors j!=i within max_edge_m, and distances.
    Returns:
      nbr_idx:  list of np.array(int) neighbors for each i
      nbr_dst:  list of np.array(float) distances (meters) aligned with nbr_idx
      pair_list: list of (i,j,dij) for all i<j within max_edge_m (candidate edges)
    """
    N = latlonalt.shape[0]
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
        d = chord_distance_3d_m_vec(latlonalt, latlonalt[i,0], latlonalt[i,1], latlonalt[i,2])
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

def _neighbors_within_maxdist_indexed(latlonalt: np.ndarray, max_edge_m: float, method: str, progress_msg: str):
    method = (method or "balltree").lower()
    if method == "balltree":
        return _neighbors_within_maxdist_balltree(latlonalt, max_edge_m, progress_msg=progress_msg)
    elif method == "bruteforce":
        return _neighbors_within_maxdist(latlonalt, max_edge_m, progress_msg=progress_msg)
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

def write_edges_csv(edges: List[Tuple[int,int,float]], out_dir: Path, idents: List[str]):
    p = out_dir / "edges.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["V0","V1","D"])
        for i,j,d in edges:
            if j < i:
                i,j = j,i
            w.writerow([idents[i], idents[j], f"{d:.3f}"])

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
    u_phi, u_lambda = u_rad[:,0:1], u_rad[:,1:2]
    v_phi, v_lambda = v_rad[None,:,0], v_rad[None,:,1]
    d_phi = v_phi - u_phi
    d_lambda = v_lambda - u_lambda
    a = np.sin(d_phi/2.0)**2 + np.cos(u_phi)*np.cos(v_phi)*np.sin(d_lambda/2.0)**2
    return 2.0 * EARTH_R_M * np.arcsin(np.sqrt(a))

def _closest_pair_between_sets(idx_a: np.ndarray, idx_b: np.ndarray, latlonalt: np.ndarray) -> Tuple[int,int,float]:
    """Return (ia, ib, d_m) for the closest pair across two components using BallTree."""
    from sklearn.neighbors import BallTree

    A = latlonalt[idx_a, :2]
    B = latlonalt[idx_b, :2]
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
    q_node = int(q_idx[k])
    t_node = int(t_idx[int(nn[k,0])])
    # Recompute as 3D distance using altitudes
    qa = latlonalt[q_node]; ta = latlonalt[t_node]
    d_m = chord_distance_3d_m(qa[0], qa[1], qa[2], ta[0], ta[1], ta[2])

    if flip:
        return t_node, q_node, d_m
    return q_node, t_node, d_m

def ensure_connected(latlonalt: np.ndarray,
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
    N = latlonalt.shape[0]
    G = _graph_from_edges(N, edges_kept)
    _, comps = _component_labels(G)
    if len(comps) <= 1:
        return edges_kept, 0

    print(f"[connectivity] Components before: {len(comps)}")
    C = _centroids(latlonalt[:, :2], comps)
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
        ia, ib, dm = _closest_pair_between_sets(np.asarray(comps[cu]), np.asarray(comps[cv]), latlonalt)
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
    p.add_argument("--airport-include", type=str, default=None,
                   help="Optional: include ONLY these airports (ICAO). "
                        "Either a comma/space-separated list like 'LOWW,EDDM' "
                        "or a path to a text/CSV file containing ICAO codes.")
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
    # Altitude control (uniform for all vertices)
    p.add_argument("--altitude", type=float, default=0.0,
                   help="Uniform altitude value for ALL vertices (default 0). Interpreted via --altitude-unit.")
    p.add_argument("--altitude-unit", type=str, default="m", choices=["m","fl"],
                   help="Unit for --altitude: meters ('m', default) or flight levels ('fl', e.g., --altitude 350 --altitude-unit fl).")

    # --- Grid navpoint mode ---
    p.add_argument("--grid-navpoints", type=str, default="false",
                   help="If true, ignore fix.dat/nav.dat and generate artificial grid navpoints (airports are still included).")
    p.add_argument("--grid-nx", type=int, default=0,
                   help="Number of grid navpoints west->east (required if --grid-navpoints true).")
    p.add_argument("--grid-ny", type=int, default=0,
                   help="Number of grid navpoints south->north (required if --grid-navpoints true).")
    p.add_argument("--grid-prefix", type=str, default="GRID",
                   help="Identifier prefix for generated grid navpoints (default 'GRID').")
    p.add_argument("--grid-region-name", type=str, default=None,
                   help="Optional: pick a single region by name from config for grid generation (case-insensitive).")
    p.add_argument("--grid-bounds", type=float, nargs=4, default=None,
                   metavar=("LAT_SOUTH","LON_WEST","LAT_NORTH","LON_EAST"),
                   help="Optional explicit bbox for grid generation (overrides config regions).")
    p.add_argument("--grid-rect-filter", type=str, default="true",
                   help="If true and grid mode is enabled, filter vertices by the region bbox (not polygon). Default true.")

    return p.parse_args()

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    args.icao_only = True if str(args.icao_only).strip().lower() in ("true","t","1","yes","on") else False
    airport_include = parse_airport_include_spec(args.airport_include)

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
            f"_apinc{len(airport_include) if airport_include else 0}"
            f"_nav-{args.navdir.name}_ap-{args.ourairports.name}"
        )
        exp_dir = out_root / f"{ts}__{tag}"
    exp_dir.mkdir(parents=True, exist_ok=True)
 

    # 1) Load config regions (polygons + names)
    region_items = load_region_items_from_config(args.config) if args.config else []
    regions = [poly for (_, poly) in region_items]
    if regions:
        print(f"[1/6] Loaded {len(regions)} geographic region(s) from {args.config}")

    # Resolve grid mode
    grid_enabled = str(args.grid_navpoints).strip().lower() in ("true","t","1","yes","on")
    if grid_enabled:
        if int(args.grid_nx) < 1 or int(args.grid_ny) < 1:
            raise ValueError("Grid mode enabled but --grid-nx/--grid-ny are not >= 1.")
        # Determine bboxes (explicit overrides config)
        grid_bboxes: List[Tuple[float,float,float,float]] = []
        grid_region_names: List[str] = []
        if args.grid_bounds is not None:
            lat_s, lon_w, lat_n, lon_e = [float(x) for x in args.grid_bounds]
            lat_min, lat_max = (lat_s, lat_n) if lat_s <= lat_n else (lat_n, lat_s)
            lon_min, lon_max = (lon_w, lon_e) if lon_w <= lon_e else (lon_e, lon_w)
            grid_bboxes = [(lat_min, lat_max, lon_min, lon_max)]
            grid_region_names = ["BBOX"]
        else:
            if not region_items:
                raise ValueError("Grid mode enabled but no --grid-bounds provided and no regions found in --config.")
            # Optional: choose one region by name
            if args.grid_region_name:
                want = str(args.grid_region_name).strip().lower()
                region_items_sel = [(n,p) for (n,p) in region_items if str(n).strip().lower() == want]
                if not region_items_sel:
                    known = ", ".join([n for (n,_) in region_items])
                    raise ValueError(f"--grid-region-name '{args.grid_region_name}' not found. Known: {known}")
                region_items_use = region_items_sel
            else:
                region_items_use = region_items
            for (nm, poly) in region_items_use:
                grid_bboxes.append(_bbox_from_poly(poly))
                grid_region_names.append(str(nm))

        # Region filter mode in grid: bbox by default
        region_filter_use_bbox = str(args.grid_rect_filter).strip().lower() in ("true","t","1","yes","on")
        region_bboxes = grid_bboxes if region_filter_use_bbox else None


    # 2) Load OD pairs & aircrafts (optional)
    od_df = load_od_pairs(args.od_file)
    if not od_df.empty:
        print(f"[2/6] Loaded OD pairs: {len(od_df):,} rows (restricting airports to observed ICAOs).")

    # 3) Build vertices (airports + navpoints OR airports + grid-navpoints) with region filtering
    print(f"[3/6] Loading airports and navpoints...")
    altitude_m = to_altitude_m(args.altitude, args.altitude_unit)
    print(f"{altitude_m} = {args.altitude} --> {args.altitude_unit}")

    if airport_include:
        print(f"[3/6] Airport include-list enabled: {len(airport_include)} ICAO(s)")

    vertices = build_vertices(
        airports_csv=args.ourairports,
        nav_dir=args.navdir,
        regions=regions,
        icao_only=args.icao_only,
        od_filter=od_df if not od_df.empty else None,
        altitude_m = altitude_m,
        airport_include = airport_include,
        grid_only=bool(grid_enabled),
        grid_nx=int(args.grid_nx) if grid_enabled else 0,
        grid_ny=int(args.grid_ny) if grid_enabled else 0,
        grid_bboxes=(grid_bboxes if grid_enabled else None),
        grid_prefix=str(args.grid_prefix or "GRID"),
        grid_region_names=(grid_region_names if grid_enabled else None),
        region_filter_use_bbox=(region_filter_use_bbox if grid_enabled else False),
        region_bboxes=(region_bboxes if grid_enabled else None),
    )
    N = len(vertices)
    if N == 0:
        raise RuntimeError("No vertices after filtering. Check inputs/regions.")
    print(f"      Kept {N:,} vertices.")

    # 4) Save vertices.csv
    write_vertices_csv(vertices, exp_dir)
    print(f"[4/6] Wrote vertices.csv -> {exp_dir/'vertices.csv'}")
 
    latlonalt = vertices[["LAT","LON","ALTITUDE"]].to_numpy(dtype=float)

    if grid_enabled:
        # Grid override: enforce local 8-neighborhood connectivity between grid points only.
        # Airports remain in vertices.csv but are excluded from grid adjacency edges.
        print("[5/6] Grid mode enabled: building 8-neighborhood edges between grid navpoints (no RNG/Gabriel).")
        edges_kept = build_grid_edges_8nb(vertices)
        print(f"      Kept edges (grid 8-neighborhood): {len(edges_kept):,}")

        # Attach airports to closest navpoints (to avoid detached airport components)
        ap_attach = build_airport_to_nav_edges(
            vertices,
            k_nearest=4,
            max_edge_m=float(args.max_edge_km) * 1000.0,
        )
        if ap_attach:
            seen = {(min(i, j), max(i, j)) for (i, j, _) in edges_kept}
            added = 0
            for (i, j, d) in ap_attach:
                u, v = (i, j) if i < j else (j, i)
                if (u, v) in seen:
                    continue
                edges_kept.append((u, v, float(d)))
                seen.add((u, v))
                added += 1
            print(f"      Added airport attachment edges: {added:,}")
        # Keep step numbering stable for logs
        print("[6/6] Skipped RNG/Gabriel edge test in grid mode.")
    else:
        # 5) Neighbor search (pre-candidate edges within max distance) with index
        print(f"[5/6] Neighbor search within {args.max_edge_km:g} km using '{args.neighbor_index}'...")
        nbr_idx, nbr_dst, pair_list = _neighbors_within_maxdist_indexed(
            latlonalt, max_edge_m=args.max_edge_km*1000.0, method=args.neighbor_index, progress_msg="Neighbor search"
        )
        print(f"      Candidate pairs: {len(pair_list):,}")

        # 6) Edge test (RNG / Gabriel)
        print(f"[6/6] Testing candidate edges with '{args.criterion}' criterion...")
        edges_kept = build_edges_rng_or_gabriel(
            latlon=latlonalt,
            nbr_idx=nbr_idx,
            nbr_dst=nbr_dst,
            pair_list=pair_list,
            criterion=args.criterion.lower(),
            progress_interval_s=int(args.progress_interval),
        )
        print(f"      Kept edges: {len(edges_kept):,}")
 

    # 7) Enforce connectivity (optional, default on)
    if do_connect and (not grid_enabled):
        print("[7/7] Enforcing connectivity...")
        edges_kept, n_added = ensure_connected(
            latlonalt=latlonalt,
            edges_kept=edges_kept,
            k_centroid_nn=int(args.centroid_knn),
            method=args.connectivity_method.lower(),
        )
        # quick report
        G_final = _graph_from_edges(len(vertices), edges_kept)
        k_final = nx.number_connected_components(G_final)
        print(f"      Components after: {k_final} (added {n_added} bridging edge(s))")
    else:
        if grid_enabled:
            print("[7/7] Skipped connectivity enforcement in grid mode (grid is already connected by construction).")
        else:
            print("[7/7] Skipped connectivity enforcement by user request.")


    # Write edges.csv (using IDENTIFIER names instead of numeric indices)
    write_edges_csv(edges_kept, exp_dir, vertices["IDENTIFIER"].astype(str).tolist())

    # Persist a run_config.json for traceability (paths as strings)
    try:
        import json
        payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        payload["resolved_out_dir"] = str(exp_dir.resolve())
        payload["stats"] = {
            "num_vertices": int(N),
            "num_edges_written": int(len(edges_kept)),
            "enforce_connected": bool(do_connect),
            "altitude_m": float(altitude_m),
        }
        with open(exp_dir / "run_config.json", "w") as fh:
            json.dump(payload, fh, indent=2)
    except Exception:
        pass

    print(f"Done. Wrote edges.csv and vertices.csv to {exp_dir.resolve()}")
 

if __name__ == "__main__":
    main()
