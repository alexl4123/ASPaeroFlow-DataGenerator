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
import pandas as pd

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate sectors.csv with per-vertex capacities.")
    p.add_argument("--path", type=Path, default=Path("./navgraph_out"),
                   help="Directory that contains vertices.csv; sectors.csv will be written here too.")
    p.add_argument("--ourairports", type=Path, default=Path("./ourairports/airports.csv"),
                   help="Path to OurAirports airports.csv (for ICAO detection).")
    p.add_argument("--cap-enroute", type=int, default=60, help="Capacity for en-route sectors (default 60).")
    p.add_argument("--cap-airport", type=int, default=60000, help="Capacity for airport sectors (default 60000).")
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

def main():
    args = parse_args()

    vertices_path = args.path / "vertices.csv"
    out_path = args.path / "sectors.csv"
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

    print("[3/3] Writing sectors.csv...")
    capacities = np.where(is_airport.to_numpy(), args.cap_airport, args.cap_enroute)
    out_df = pd.DataFrame({
        "Sector_ID": vdf[id_col],
        "Capacity": capacities.astype(int)
    })


    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomic navaid->sector assignment (each vertex maps to itself)
    nsdf = pd.DataFrame({"Navaid_ID": vdf[id_col], "Sector_ID": vdf[id_col]})
    nsdf_path = args.path  / "navaid_sector_assignment.csv"
    nsdf.to_csv(nsdf_path, index=False)


    out_df.to_csv(out_path, index=False)
    print(f"Done. Wrote {len(out_df):,} rows to {out_path.resolve()}")

if __name__ == "__main__":
    main()




