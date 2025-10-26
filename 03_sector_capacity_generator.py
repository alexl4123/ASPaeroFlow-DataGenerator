#!/usr/bin/env python3
"""
Lightweight capacity & atomic sector assignment generator

Reads vertices.csv (from your navpoint graph build) and assigns a capacity per
vertex (interpreted as the dual "atomic sector"):
  - Airports: --cap-airport (default 60000)
  - En-route: --cap-enroute (default 60)

Airports are detected by intersecting IDENTIFIERs with the ICAO codes present
in OurAirports airports.csv (fallback: 4-letter ICAO regex if file not found).

Output: sectors.csv with columns:
  Sector_ID,Capacity
where Sector_ID comes from an ID column in vertices.csv (no renumbering).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import re
import numpy as np
from collections import deque
import pandas as pd

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate sectors.csv with per-vertex capacities.")
    p.add_argument("--path", type=Path, default=Path("./navgraph_out"),
                   help="Directory that contains vertices.csv; sectors.csv will be written here too.")
    p.add_argument("--ourairports", type=Path, default=Path("./ourairports/airports.csv"),
                   help="Path to OurAirports airports.csv (for ICAO detection).")
    p.add_argument("--cap-enroute", type=int, default=60, help="Capacity for en-route sectors (default 60).")
    p.add_argument("--cap-airport", type=int, default=60000, help="Capacity for airport sectors (default 60000).")
    p.add_argument("--sector-default-navaid-size", type=int, default=10,
                   help="Default number n of navaids per sector; sectors are built as connected subgraphs (BFS growth).")
    p.add_argument("--out", type=Path, default=Path("./navgraph_out/world/sectors.csv"), help="Output CSV path.")
    return p.parse_args()

def load_icao_set(ourairports_csv: Path) -> set[str]:
    """
    Return set of ICAO codes from OurAirports. If file is missing, return empty set.
    """
    if not ourairports_csv.exists():
        print(f"[WARN] OurAirports file not found at {ourairports_csv}. Falling back to ICAO regex.", file=sys.stderr)
        return set()
    df = pd.read_csv(ourairports_csv, dtype="string", low_memory=False)
    icao_cols = [c for c in ["icao_code","ident","gps_code"] if c in df.columns]
    if not icao_cols:
        return set()
    cand = pd.concat([df[c] for c in icao_cols], ignore_index=True)
    s = cand.dropna().astype(str).str.strip().str.upper()
    pat = re.compile(r"^[A-Z]{4}$")
    return set(x for x in s if pat.match(x))

def _detect_edge_cols(df: pd.DataFrame) -> tuple[str,str]:
    """
    Heuristically detect (u,v) column names in edges.csv.
    Tries common pairs, then falls back to the first two columns.
    """
    lc = {c.lower(): c for c in df.columns}
    pairs = [
        ("src","dst"), ("source","target"), ("src_id","dst_id"),
        ("from_id","to_id"), ("from","to"), ("u","v"), ("a","b"),
    ]
    for a,b in pairs:
        if a in lc and b in lc:
            return lc[a], lc[b]
    # fallback: first two columns
    if len(df.columns) < 2:
        raise ValueError("edges.csv must have at least two columns for endpoints.")
    return df.columns[0], df.columns[1]

def _build_adjacency(edge_df: pd.DataFrame, id_col: str, allowed_ids: set) -> dict:
    """
    Build undirected adjacency among allowed_ids from edge_df.
    Unknown endpoints (not in allowed_ids) are ignored.
    """
    ucol, vcol = _detect_edge_cols(edge_df)
    u = edge_df[ucol].astype(object)
    v = edge_df[vcol].astype(object)
    adj = {i: set() for i in allowed_ids}
    for a,b in zip(u, v):
        if a in allowed_ids and b in allowed_ids:
            adj[a].add(b)
            adj[b].add(a)
    return adj

def _partition_connected(enroute_ids: list, adj: dict, order: dict, n: int) -> list[list]:
    """
    Greedy BFS-based partition into connected groups of size ~n.
    Deterministic by following the vertex order from vertices.csv.
    If a connected component has < n nodes, it forms a smaller sector.
    """
    unassigned = set(enroute_ids)
    groups = []
    # seed order = appearance in vertices.csv
    def neighbors_sorted(x):
        # deterministic by original order
        return sorted((y for y in adj.get(x, []) if y in unassigned), key=lambda z: order.get(z, 10**12))

    for seed in enroute_ids:
        if seed not in unassigned:
            continue
        group = []
        q = deque([seed])
        unassigned.remove(seed)
        while q and len(group) < n:
            x = q.popleft()
            group.append(x)
            for y in neighbors_sorted(x):
                if y in unassigned:
                    unassigned.remove(y)
                    q.append(y)
        groups.append(group)
    return groups


def main():
    args = parse_args()

    vertices_path = args.path / "vertices.csv"
    out_path = args.path / "sectors.csv"
    edges_path = args.path / "edges.csv"

    if not vertices_path.exists():
        raise FileNotFoundError(f"vertices.csv not found: {vertices_path}")
 
    print("[1/3] Loading vertices...")
    vdf = pd.read_csv(vertices_path, dtype={"IDENTIFIER":"string"})

    if "IDENTIFIER" not in vdf.columns:
        raise ValueError("vertices.csv must have an IDENTIFIER column.")
    vdf["IDENTIFIER"] = vdf["IDENTIFIER"].astype("string").str.strip().str.upper()
    N = len(vdf)
    print(f"       {N:,} vertices found.")

    # Determine which column to use as Sector_ID (no renumbering)
    id_candidates = [
        "ID", "Id", "id",
        "Vertex_ID", "VERTEX_ID", "vertex_id",
        "Sector_ID", "SECTOR_ID", "sector_id",
        # fall back to the human identifier if no numeric ID present
        "IDENTIFIER"
    ]
    id_col = next((c for c in id_candidates if c in vdf.columns), None)
    if id_col is None:
        # Absolute fallback (not expected with your pipeline)
        print("[WARN] No ID-like column found in vertices.csv; falling back to row index.",
              file=sys.stderr)
        vdf["_ROW_INDEX"] = np.arange(N)
        id_col = "_ROW_INDEX"
    print(f"       Using '{id_col}' as Sector_ID source.")

    print("[2/3] Detecting airports...")
    icao_set = load_icao_set(args.ourairports)
    if icao_set:
        is_airport = vdf["IDENTIFIER"].isin(icao_set)
        method = "OurAirports match"
    else:
        # Fallback: 4-letter ICAO-looking identifiers
        is_airport = vdf["IDENTIFIER"].str.match(r"^[A-Z]{4}$", na=False)
        method = "ICAO regex fallback"
    n_airports = int(is_airport.sum())
    print(f"       Airports detected: {n_airports:,} ({method}).")

    # Build connected navaid->sector assignment for ENROUTE vertices
    print("[3/4] Building connected navaid-sector assignment...")
    # Determine processing order (deterministic: as in vertices.csv)
    order = {k: i for i, k in enumerate(vdf[id_col].tolist())}
    # Separate airport vs en-route sets
    enroute_mask = ~is_airport
    enroute_ids = vdf.loc[enroute_mask, id_col].astype(object).tolist()
    airport_ids = vdf.loc[is_airport, id_col].astype(object).tolist()

    if not edges_path.exists():
        print(f"[WARN] edges.csv not found at {edges_path}. "
              f"Falling back to simple contiguous chunks of size n={args.sector_default_navaid_size} (connectivity not guaranteed).",
              file=sys.stderr)
        groups = [enroute_ids[i:i+args.sector_default_navaid_size] for i in range(0, len(enroute_ids), args.sector_default_navaid_size)]
    else:
        print("       Loading edges...")
        edf = pd.read_csv(edges_path, dtype=object, low_memory=False)
        enroute_set = set(enroute_ids)
        adj = _build_adjacency(edf, id_col=id_col, allowed_ids=enroute_set)
        groups = _partition_connected(enroute_ids, adj, order, n=args.sector_default_navaid_size)
    print(f"       Created {len(groups)} connected en-route sectors (target size n={args.sector_default_navaid_size}).")

    # Create human-friendly sector IDs
    sector_ids = [f"SECTOR_{i:06d}" for i in range(len(groups))]
    enroute_assign = {nid: sid for sid, grp in zip(sector_ids, groups) for nid in grp}
    airport_assign = {aid: f"SECTOR_AIRPORT_{aid}" for aid in airport_ids}
    assign_map = {**enroute_assign, **airport_assign}

    # Write navaid->sector assignment
    navaid_ids_series = vdf[id_col].astype(object)
    sector_series = navaid_ids_series.map(assign_map)
    # Guarantee full coverage: any unassigned navaid becomes its own singleton sector
    missing = sector_series.isna()
    if missing.any():
        missing_count = int(missing.sum())
        print(f"[WARN] {missing_count:,} navaids had no assigned sector; "
              f"assigning each to its own singleton sector.", file=sys.stderr)
        sector_series.loc[missing] = navaid_ids_series.loc[missing].map(lambda x: f"SECTOR_SINGLE_{x}")
    nsdf = pd.DataFrame({
        "Navaid_ID": navaid_ids_series,
        "Sector_ID": sector_series
    })
    if nsdf["Sector_ID"].isna().any():
        raise AssertionError("Internal error: some navaids still lack a sector assignment.")


    nsdf_path = args.path / "navaid_sector_assignment.csv"
    nsdf.to_csv(nsdf_path, index=False)

    print("[4/4] Writing sectors.csv (atomic capacities)...")

    capacities = np.where(is_airport.to_numpy(), args.cap_airport, args.cap_enroute)
    out_df = pd.DataFrame({
        "Sector_ID": vdf[id_col],
        "Capacity": capacities.astype(int)
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(out_path, index=False)
    print(f"Done. Wrote {len(out_df):,} rows to {out_path.resolve()} and {len(nsdf):,} rows to {nsdf_path.resolve()}")


if __name__ == "__main__":
    main()




