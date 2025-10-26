#!/usr/bin/env python3
"""
Transform pipeline outputs → optimizer format.

Input experiment folder layout (produced by run_pipeline.py):
  <exp_in>/
    model/
    navgraph/                 # requires: vertices.csv, edges.csv, sectors.csv, (airports.csv?), navaid_sector_assignment.csv
    DATA_S<scale>_<seed>/     # one or many
      flights.csv             # (flight_id, aircraft_id, origin, destination, departure_time)
      filed_flights.csv       # (Flight_ID, Position, Time)  -- Position may be IDENTIFIER or vertex id
      aircrafts.csv|aircraft.csv
      run_config.json         # has "number_flights" and the per-data "seed"

Output (per data sample), into: <out_root>/<experiment_name>/<NNNNNNN_SEEDxx>/ :
  airplane_flight_assignment.csv  (Airplane_ID,Flight_ID)
  airplanes.csv                   (Airplane_ID,Speed_kts)
  airports.csv                    (Airport_Vertex)
  flights.csv                     (Flight_ID,Position,Time)
  graph_edges.csv                 (source,target,dist_m)
  navaid_sector_assignment.csv    (Navaid_ID,Sector_ID)
  sectors.csv                     (Sector_ID,Capacity)

Also writes mapping files to <NNNNNNN_SEEDxx>/mappings/ :
  id_maps.json            # {airplane_id: {to_int:{}, to_str:{}}, flight_id:{...}}
  vertex_map.csv          # Vertex_ID,IDENTIFIER    (full vertex dictionary)

Usage:
  python 05_transform_for_optimizer.py \
      --in-exp-dir unparsed_experiment_data/DACH-2019-06-15 \
      --out-root   experiment_data

Notes:
  • We keep navgraph vertex IDs as-is (the row index used in edges.csv).
  • If sectors/assignments list IDENTIFIER strings, we map via vertices.csv.
  • If filed_flights.csv has IDENTIFIER positions, we map them too.
  • airplane/flight integer IDs are assigned by first appearance order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
import sys

# --------------- helpers ---------------

def _find_aircrafts_csv(d: Path) -> Path:
    for name in ("aircrafts.csv", "aircraft.csv"):
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No aircrafts.csv / aircraft.csv in {d}")

def _norm_edges_cols(edf: pd.DataFrame) -> pd.DataFrame:
    low = {str(c).lower(): c for c in edf.columns}
    if {"v0","v1","d"} <= low.keys():
        edf = edf.rename(columns={low["v0"]:"source", low["v1"]:"target", low["d"]:"dist_m"})
    elif {"source","target","dist_m"} <= low.keys():
        edf = edf.rename(columns={low["source"]:"source", low["target"]:"target", low["dist_m"]:"dist_m"})
    elif {"source","target","d"} <= low.keys():
        edf = edf.rename(columns={low["source"]:"source", low["target"]:"target", low["d"]:"dist_m"})
    else:
        # fallback: first three columns
        cols = list(edf.columns)
        if len(cols) < 3:
            raise ValueError("edges.csv must have ≥3 columns.")
        edf = edf.rename(columns={cols[0]:"source", cols[1]:"target", cols[2]:"dist_m"})
    return edf[["source","target","dist_m"]]

def _read_vertices(navgraph_dir: Path) -> Tuple[pd.DataFrame, Dict[str,int], Dict[int,str]]:
    vpath = navgraph_dir / "vertices.csv"
    if not vpath.exists():
        raise FileNotFoundError(f"Missing vertices.csv in {navgraph_dir}")
    vdf = pd.read_csv(vpath)
    if "IDENTIFIER" not in vdf.columns:
        raise ValueError("vertices.csv must contain IDENTIFIER column.")
    id_to_vid: Dict[str,int] = {}
    vid_to_id: Dict[int,str] = {}
    for i, ident in enumerate(vdf["IDENTIFIER"].astype(str).str.strip().str.upper()):
        id_to_vid[ident] = i
        vid_to_id[i] = ident
    return vdf, id_to_vid, vid_to_id

def _safe_int_series(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """try cast to int; returns (ints, is_numeric_mask)"""
    # allow strings like "123"
    tmp = pd.to_numeric(s, errors="coerce")
    mask = tmp.notna()
    ints = tmp.fillna(-1).astype(int)
    return ints, mask

def _first_appearance_index(values: pd.Series) -> Dict[str,int]:
    """Assign 0..K-1 by first occurrence order."""
    mapping: Dict[str,int] = {}
    nxt = 0
    for v in values:
        v = str(v)
        if v not in mapping:
            mapping[v] = nxt
            nxt += 1
    return mapping

def _write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

# --------------- main transform per DATA_* ---------------

def transform_one_sample(exp_in: Path, data_dir: Path, out_root: Path, experiment_name: str):
    """
    exp_in: path to experiment root (contains navgraph/, DATA_S…/)
    data_dir: path to a specific DATA_S… folder
    """
    nav_dir = exp_in / "navgraph"
    if not nav_dir.exists():
        raise FileNotFoundError(f"No navgraph directory at {nav_dir}")

    # --- read navgraph dictionary & edges
    vdf, ident_to_vid, vid_to_ident = _read_vertices(nav_dir)

    epath = nav_dir / "edges.csv"
    if not epath.exists():
        raise FileNotFoundError(f"Missing edges.csv in {nav_dir}")

    edf = pd.read_csv(epath)
    edf = _norm_edges_cols(edf)
    # endpoints may be numeric vertex ids **or** IDENTIFIER strings
    s_ints, s_isnum = _safe_int_series(edf["source"])
    t_ints, t_isnum = _safe_int_series(edf["target"])
    if s_isnum.all() and t_isnum.all():
        edf["source"] = s_ints.astype(int)
        edf["target"] = t_ints.astype(int)
    else:
        # map IDENTIFIER → vertex id using vertices.csv dictionary
        s_ids = edf["source"].astype(str).str.strip().str.upper().map(ident_to_vid)
        t_ids = edf["target"].astype(str).str.strip().str.upper().map(ident_to_vid)
        # sanity check for unknown identifiers
        if s_ids.isna().any() or t_ids.isna().any():
            bad_s = edf.loc[s_ids.isna(), "source"].head(5).tolist()
            bad_t = edf.loc[t_ids.isna(), "target"].head(5).tolist()
            raise KeyError(
                f"edges.csv contains IDENTIFIERs not found in vertices.csv. "
                f"Examples source={bad_s} target={bad_t}"
            )
        edf["source"] = s_ids.astype(int)
        edf["target"] = t_ids.astype(int)
    edf["dist_m"] = pd.to_numeric(edf["dist_m"], errors="coerce").astype(float)
    edf["dist_m"] = edf["dist_m"].round(decimals=0).astype(int)

    # sectors
    spath = nav_dir / "sectors.csv"
    if not spath.exists():
        raise FileNotFoundError(f"Missing sectors.csv in {nav_dir}")
    sdf = pd.read_csv(spath)
    if not {"Sector_ID","Capacity"}.issubset(set(sdf.columns)):
        raise ValueError("sectors.csv must have columns Sector_ID,Capacity")
    # Map Sector_ID (string IDENTIFIER?) → vertex id if needed
    s_ints, s_isnum = _safe_int_series(sdf["Sector_ID"])
    if s_isnum.all():
        sdf["Sector_ID"] = s_ints
    else:
        # expect IDENTIFIER strings
        sid = sdf["Sector_ID"].astype(str).str.strip().str.upper()
        try:
            sdf["Sector_ID"] = sid.map(lambda x: ident_to_vid[x])
        except KeyError as e:
            raise KeyError(f"Unknown Sector_ID '{e.args[0]}' not found in vertices.csv") from e
    sdf["Capacity"] = pd.to_numeric(sdf["Capacity"], errors="raise").astype(int)
    sdf = sdf[["Sector_ID","Capacity"]].sort_values("Sector_ID")

    # navaid → sector (atomic)
    ns_path = nav_dir / "navaid_sector_assignment.csv"
    if not ns_path.exists():
        raise FileNotFoundError(f"Missing navaid_sector_assignment.csv in {nav_dir}")
    nsdf = pd.read_csv(ns_path)
    if not {"Navaid_ID","Sector_ID"} <= set(nsdf.columns):
        raise ValueError("navaid_sector_assignment.csv must have columns Navaid_ID,Sector_ID")
    """
    # map both columns to ints
    for col in ("Navaid_ID","Sector_ID"):
        ints, isnum = _safe_int_series(nsdf[col])
        if isnum.all():
            nsdf[col] = ints
        else:
            nsdf[col] = nsdf[col].astype(str).str.strip().str.upper().map(ident_to_vid)
    nsdf = nsdf[["Navaid_ID","Sector_ID"]].sort_values(["Navaid_ID","Sector_ID"])
    """

    # Map ONLY Navaid_ID → vertex id; keep Sector_ID as opaque label (e.g., SECTOR_000024, SECTOR_AIRPORT_LOAL)
    navaid_ints, navaid_isnum = _safe_int_series(nsdf["Navaid_ID"])
    if navaid_isnum.all():
        nsdf["Navaid_ID"] = navaid_ints.astype(int)
    else:
        nsdf["Navaid_ID"] = nsdf["Navaid_ID"].astype(str).str.strip().str.upper().map(ident_to_vid)
        if nsdf["Navaid_ID"].isna().any():
            bad = nsdf.loc[nsdf["Navaid_ID"].isna(), "Navaid_ID"].head(5).tolist()
            raise KeyError(f"navaid_sector_assignment.csv has unknown Navaid_IDs not in vertices.csv. Examples: {bad}")
    # Sector_ID remains string; just normalize whitespace (do NOT upper in case labels are caseful)
    nsdf["Sector_ID"] = nsdf["Sector_ID"].astype(str).str.strip()
    if (nsdf["Sector_ID"] == "").any():
        raise ValueError("navaid_sector_assignment.csv contains empty Sector_ID values.")
    nsdf = nsdf[["Navaid_ID","Sector_ID"]].sort_values(["Navaid_ID","Sector_ID"], kind="mergesort")

    # airports list
    ap_path = nav_dir / "airports.csv"
    if ap_path.exists():
        adf = pd.read_csv(ap_path)
        # accept either numeric ids or identifiers
        col = list(adf.columns)[0]
        ints, isnum = _safe_int_series(adf[col])
        if isnum.all():
            ap_ids = sorted(ints.tolist())
        else:
            ap_ids = sorted(adf[col].astype(str).str.strip().str.upper().map(ident_to_vid).tolist())
    else:
        # derive from vertices.csv if IS_AIRPORT present
        if "IS_AIRPORT" in vdf.columns:
            ap_ids = [i for i, flag in enumerate(vdf["IS_AIRPORT"].fillna(0).astype(int).tolist()) if flag]
        else:
            print("[WARN] airports.csv missing and vertices.csv has no IS_AIRPORT — emitting empty airports.csv", file=sys.stderr)
            ap_ids = []
    airports_df = pd.DataFrame({"Airport_Vertex": ap_ids})

    # --- read DATA sample
    rc_path = data_dir / "run_config.json"
    if not rc_path.exists():
        raise FileNotFoundError(f"Missing run_config.json in {data_dir}")
    with open(rc_path, "r") as fh:
        rc = json.load(fh) or {}
    n_flights = int(rc.get("number_flights", 0))
    seed_val  = str(rc.get("seed", "NA"))

    flights_path = data_dir / "flights.csv"
    filed_path   = data_dir / "filed_flights.csv"
    if not flights_path.exists():
        raise FileNotFoundError(f"Missing flights.csv in {data_dir}")
    if not filed_path.exists():
        raise FileNotFoundError(f"Missing filed_flights.csv in {data_dir}")
    fl = pd.read_csv(flights_path)
    filed = pd.read_csv(filed_path)

    # normalize column names
    flc = {c.lower(): c for c in fl.columns}
    need_cols = ["flight_id","aircraft_id","origin","destination","departure_time"]
    if not set(need_cols) <= set(flc.keys()):
        missing = [c for c in need_cols if c not in flc]
        raise ValueError(f"flights.csv missing columns: {missing}")
    fl = fl.rename(columns={flc[k]: k for k in need_cols})

    # Build integer ID maps (by first appearance in flights.csv)
    flight_to_int = _first_appearance_index(fl["flight_id"].astype(str))
    aircraft_to_int = _first_appearance_index(fl["aircraft_id"].astype(str))

    # Reduce to flights that actually have filed trajectories
    filed_cols = {c.lower(): c for c in filed.columns}
    if not {"flight_id","position","time"} <= set(filed_cols.keys()):
        raise ValueError("filed_flights.csv must contain Flight_ID/flight_id, Position, Time")
    filed = filed.rename(columns={
        filed_cols.get("flight_id","Flight_ID"): "Flight_ID",
        filed_cols.get("position","Position"):   "Position",
        filed_cols.get("time","Time"):           "Time",
    })
    # Ensure Position → vertex id
    pos_ints, pos_isnum = _safe_int_series(filed["Position"])
    if pos_isnum.all():
        filed["Position"] = pos_ints.astype(int)
    else:
        # map IDENTIFIER → vertex id
        filed["Position"] = filed["Position"].astype(str).str.strip().str.upper().map(ident_to_vid)

    # Coerce time to int
    filed["Time"] = pd.to_numeric(filed["Time"], errors="raise").astype(int)

    # Keep only filed flights that we know from flights.csv map
    filed["_fid_str"] = filed["Flight_ID"].astype(str)
    keep_mask = filed["_fid_str"].isin(flight_to_int.keys())
    dropped = (~keep_mask).sum()
    if dropped:
        print(f"[WARN] Dropping {dropped} filed rows with unknown Flight_ID.", file=sys.stderr)
    filed = filed[keep_mask].copy()

    # Map Flight_ID → int
    filed["Flight_ID"] = filed["_fid_str"].map(flight_to_int)
    filed = filed.drop(columns=["_fid_str"])
    filed = filed[["Flight_ID","Position","Time"]].sort_values(["Flight_ID","Time","Position"], kind="mergesort")

    # Airplanes & assignment
    ac_path = _find_aircrafts_csv(data_dir)
    ac = pd.read_csv(ac_path)
    acc = {c.lower(): c for c in ac.columns}
    if not {"aircraft_id","speed_kts"} <= set(acc.keys()):
        raise ValueError("aircrafts.csv must have aircraft_id, speed_kts")
    ac = ac.rename(columns={acc["aircraft_id"]: "aircraft_id", acc["speed_kts"]: "Speed_kts"})
    ac["Airplane_ID"] = ac["aircraft_id"].astype(str).map(aircraft_to_int)
    airplanes_df = ac[["Airplane_ID","Speed_kts"]].sort_values("Airplane_ID")

    # assignment from flights.csv (only those appearing in filed)
    present_fids_str = set(filed["Flight_ID"].map({v:k for k,v in flight_to_int.items()}).tolist())
    fl_present = fl[fl["flight_id"].astype(str).isin(present_fids_str)].copy()
    fl_present["Airplane_ID"] = fl_present["aircraft_id"].astype(str).map(aircraft_to_int)
    fl_present["Flight_ID"]   = fl_present["flight_id"].astype(str).map(flight_to_int)
    assignment_df = fl_present[["Airplane_ID","Flight_ID"]].drop_duplicates().sort_values(["Airplane_ID","Flight_ID"])

    # Build output folder name "NNNNNNN_SEEDxx"
    # Use number_flights from config (padded), but fall back to actual unique flight count if missing
    if n_flights <= 0:
        n_flights = int(fl["flight_id"].nunique())
    folder_name = f"{n_flights:07d}_SEED{seed_val}"
    out_dir = out_root / experiment_name / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write the seven optimizer files
    _write_csv(assignment_df,              out_dir / "airplane_flight_assignment.csv")
    _write_csv(airplanes_df,               out_dir / "airplanes.csv")
    _write_csv(airports_df,                out_dir / "airports.csv")
    _write_csv(filed,                      out_dir / "flights.csv")
    _write_csv(edf[["source","target","dist_m"]], out_dir / "graph_edges.csv")
    _write_csv(nsdf,                       out_dir / "navaid_sector_assignment.csv")
    _write_csv(sdf,                        out_dir / "sectors.csv")

    # Store mappings for round-trip conversion
    maps_dir = out_dir / "mappings"
    maps_dir.mkdir(exist_ok=True)
    # flight + airplane maps (both directions)
    to_int_f = flight_to_int
    to_str_f = {v:k for k,v in to_int_f.items()}
    to_int_a = aircraft_to_int
    to_str_a = {v:k for k,v in to_int_a.items()}
    id_maps = {
        "flight_id":   {"to_int": to_int_f, "to_str": to_str_f},
        "airplane_id": {"to_int": to_int_a, "to_str": to_str_a},
    }
    with open(maps_dir / "id_maps.json", "w") as fh:
        json.dump(id_maps, fh, indent=2)
    # full vertex dictionary (id -> IDENTIFIER)
    vmap_df = pd.DataFrame(
        {"Vertex_ID": list(vid_to_ident.keys()), "IDENTIFIER": [vid_to_ident[i] for i in vid_to_ident.keys()]}
    ).sort_values("Vertex_ID")
    _write_csv(vmap_df, maps_dir / "vertex_map.csv")

    # Keep a tiny manifest for traceability
    out_manifest = {
        "source_experiment": str(exp_in.resolve()),
        "source_data_dir": str(data_dir.resolve()),
        "navgraph_dir": str(nav_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "n_flights": n_flights,
        "seed": seed_val,
        "files": [
            "airplane_flight_assignment.csv", "airplanes.csv", "airports.csv",
            "flights.csv", "graph_edges.csv", "navaid_sector_assignment.csv", "sectors.csv"
        ]
    }
    with open(out_dir / "transform_manifest.json", "w") as fh:
        json.dump(out_manifest, fh, indent=2)

    print(f"[✓] Wrote optimizer bundle → {out_dir}")

# --------------- CLI ---------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transform pipeline outputs to optimizer format.")
    p.add_argument("--in-exp-dir", type=Path, required=True,
                   help="Experiment folder from pipeline (contains navgraph/ and DATA_* subfolders).")
    p.add_argument("--out-root", type=Path, default=Path("experiment_data"),
                   help="Root folder for optimizer bundles (default: experiment_data).")
    p.add_argument("--experiment-name", type=str, default=None,
                   help="Name of the experiment folder under out-root. "
                        "Default: basename of --in-exp-dir.")
    p.add_argument("--select", type=str, default="DATA_*",
                   help="Glob to choose which data subfolders to convert (default: DATA_*)")
    return p.parse_args()

def main():
    a = parse_args()

    exp_in = a.in_exp_dir
    if not exp_in.exists():
        raise FileNotFoundError(f"Experiment folder not found: {exp_in}")

    experiment_name = a.experiment_name or exp_in.name
    out_root = a.out_root

    # Find DATA_* subfolders
    data_dirs = sorted([p for p in exp_in.glob(a.select) if p.is_dir()])
    if not data_dirs:
        raise RuntimeError(f"No data subfolders matching '{a.select}' in {exp_in}")

    print(f"[i] Experiment: {experiment_name}")
    print(f"[i] Inputs: {len(data_dirs)} data sample(s)")
    for d in data_dirs:
        print(f"    - {d.name}")

    for d in data_dirs:
        transform_one_sample(exp_in=exp_in, data_dir=d, out_root=out_root, experiment_name=experiment_name)

    print(f"\n[✓] All done. Output root: { (out_root / experiment_name).resolve() }")

if __name__ == "__main__":
    main()
