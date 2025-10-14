#!/usr/bin/env python3
"""
Flight OD/temporal model builder (CLI)

Reads OpenSky-like flight CSVs, filters to either a single target day **or** an inclusive
date range (possibly across multiple CSVs in a folder), and exports model artifacts:
  - airport_bins: per-airport expected departures per time bin
  - od_time_model: per-(origin, bin) destination probabilities
  - tat_dist: turnaround time samples (minutes)
  - od_dur_dist: observed durations per OD (minutes)
  - global_dest_freq: global destination popularity

Each artifact is written as a CSV into the output directory.

Usage (defaults match the original script):
    python build_model.py \
        --csv-path flightlist_20190601_20190630.csv \
        --target-day 2019-06-15 \
        --chunksize 250000 \
        --bin-min 60 \
        --smooth-win 3 \
        --epsilon 2.0 \
        --alpha 0.5 \
        --global-backoff 0.05 \
        --min-tat 0 --max-tat 60 \
        --min-dur 1 --max-dur 900 \
        --smoothing false \
        --icao-only true \
        --min-samples-per-od 1 \
        --out-dir ./model_out

Or, for a date range across multiple files inside a folder:
    python build_model.py \
        --csv-path ./flightlist_summer \
        --date-start 2019-06-15 --date-end 2019-07-15 \
        --bin-min 60 --smoothing false --icao-only true \
        --out-dir ./model_out

Using a JSON config (CLI options override config values):
    python build_model.py \
        --config ./config.json \
        --date-start 2019-06-15 --date-end 2019-07-15

"""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timezone
import json

from typing import Dict, Tuple

import numpy as np
import pandas as pd


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    def str2bool(v: str) -> bool:
        if isinstance(v, bool):
            return v
        v = v.strip().lower()
        if v in ("yes", "true", "t", "1", "y", "on"):
            return True
        if v in ("no", "false", "f", "0", "n", "off"):
            return False
        raise argparse.ArgumentError(None, f"Invalid boolean: {v}")


    # ---- first pass to get --config (no help to avoid conflicts) ----
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    # ---- load config if provided ----
    cfg: dict = {}
    if pre_args.config is not None:
        if not pre_args.config.exists():
            raise FileNotFoundError(f"Config file not found: {pre_args.config}")
        with open(pre_args.config, "r") as fh:
            cfg = json.load(fh) or {}

    def cfg_get(key: str, default):
        return cfg.get(key, default)

    p = argparse.ArgumentParser(description="Build OD/time model and export artifacts to CSV.")
    p.add_argument("--config", type=Path, default=None, help="Path to JSON config file.")
    p.add_argument("--csv-path", type=Path,
                    default=Path(cfg_get("csv-path", "flightlist_20190601_20190630.csv")),
                    help="Path to ourairports airports.csv (used to verify ICAO codes).")
    p.add_argument("--ourairports-path", type=Path,
                    default=Path(cfg_get("ourairports-path", "./ourairports/airports.csv")),
                    help="Path to ourairports airports.csv (used to verify ICAO codes).")
    p.add_argument("--target-day", type=str,
                   default=cfg_get("target-day", None), help="(Legacy) UTC day YYYY-MM-DD")
    p.add_argument("--date-start", type=str,
                   default=cfg_get("date-start", None), help="UTC inclusive start date YYYY-MM-DD")
    p.add_argument("--date-end",   type=str,
                   default=cfg_get("date-end", None), help="UTC inclusive end date YYYY-MM-DD")
    p.add_argument("--chunksize", type=int, default=int(cfg_get("chunksize", 250_000)))
 

    # Model inputs (defaults = original script)
    p.add_argument("--bin-min", type=int, default=int(cfg_get("bin-min", 60)),
                    help="Minutes per time bin")
    p.add_argument("--smooth-win", type=int, default=int(cfg_get("smooth-win", 3)),
                    help="Rolling window (bins) for smoothing")
    p.add_argument("--epsilon", type=float, default=float(cfg_get("epsilon", 2.0)),
                    help="Laplace noise scale via 1/epsilon; <=0 disables")
    p.add_argument("--alpha", type=float, default=float(cfg_get("alpha", 0.5)),
                    help="Dirichlet +alpha smoothing for OD")
    p.add_argument("--global-backoff", type=float, default=float(cfg_get("global-backoff", 0.05)),
                    help="Mixture weight with global dest freq")
    p.add_argument("--min-tat", type=float, default=float(cfg_get("min-tat", 0)),
                    help="Min turnaround minutes")
    p.add_argument("--max-tat", type=float, default=float(cfg_get("max-tat", 60)),
                    help="Max turnaround minutes")
    p.add_argument("--min-dur", type=float, default=float(cfg_get("min-dur", 1)),
                    help="Min duration minutes")
    p.add_argument("--max-dur", type=float, default=float(cfg_get("max-dur", 900)),
                    help="Max duration minutes")
    p.add_argument("--smoothing", type=str, default=str(cfg_get("smoothing", "false")), help="true/false")
    p.add_argument("--icao-only", type=str, default=str(cfg_get("icao-only", "true")), help="true/false")
    p.add_argument("--verify-ourairports", type=str, default=str(cfg_get("verify-ourairports", "true")),
                    help="true/false: verify airports exist in ourairports file (default true).")
    p.add_argument("--min-samples-per-od", type=int, default=int(cfg_get("min-samples-per-od", 1)),
                    help="Minimum samples for OD-specific durations")
    p.add_argument("--seed", type=int, default=(cfg_get("seed", None)),
                    help="Random seed (for noise/sampling reproducibility)")
    p.add_argument("--out-dir", type=Path, default=Path(cfg_get("out-dir", "./model_out")),
                    help="Base output directory; an auto-named experiment subfolder will be created inside.")

    p.add_argument("--flat-out", action="store_true",
                    help="Write files directly into --out-dir (disable auto-named subfolder).")

    args = p.parse_args()
    args.smoothing = str2bool(args.smoothing)
    args.icao_only = str2bool(args.icao_only)
    args.verify_ourairports = str2bool(args.verify_ourairports)

    # Pass-through of config-only features (e.g., geographic regions)
    args.considered_geographic_regions = cfg.get("considered_geographic_regions", None)
    args.loaded_config = str(pre_args.config) if pre_args.config is not None else None

    return args


# -------------------------
# Data loading & filtering
# -------------------------
USECOLS = [
    "callsign","number","aircraft_uid","typecode",
    "origin","destination","firstseen","lastseen","day",
    # present in some file variants
    "latitude_1","longitude_1","altitude_1","latitude_2","longitude_2","altitude_2",
]
DROP_COLS = ["number","latitude_1","longitude_1","altitude_1","latitude_2","longitude_2","altitude_2"]
BAD_TOKENS = {"", "NAN", "NONE", "NULL"}

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Helper to access a column by case-insensitive name; returns empty Series if missing."""
    matches = [c for c in df.columns if c.lower() == name]
    return df[matches[0]] if matches else pd.Series(dtype="string")

def load_ourairports_df(path: Path) -> pd.DataFrame:
    """
    Load OurAirports CSV and normalize to columns:
      - icao: 4-letter code (from icao_code or ident/gps_code if looks like ICAO)
      - lat: float latitude
      - lon: float longitude
    """
    raw = pd.read_csv(path, dtype="string", low_memory=False)
    s_icao = _col(raw, "icao_code")
    s_ident = _col(raw, "ident")
    s_gps = _col(raw, "gps_code")
    cand = pd.concat([s_icao, s_ident, s_gps], ignore_index=True)
    icao = cand.dropna().astype(str).str.strip().str.upper()
    pat = r"^[A-Z]{4}$"
    icao = icao[icao.str.match(pat)]
    icao = icao.drop_duplicates().rename("icao")
    # coordinates (prefer latitude_deg/longitude_deg)
    lat = pd.to_numeric(_col(raw, "latitude_deg"), errors="coerce").rename("lat")
    lon = pd.to_numeric(_col(raw, "longitude_deg"), errors="coerce").rename("lon")
    # Keep one row per airport where possible
    df = pd.DataFrame({"icao": _col(raw, "icao_code")}).copy()
    if df["icao"].isna().all():
        df["icao"] = _col(raw, "ident")
    if df["icao"].isna().all():
        df["icao"] = _col(raw, "gps_code")
    df["icao"] = df["icao"].astype("string").str.strip().str.upper()
    df["lat"] = lat
    df["lon"] = lon
    df = df.dropna(subset=["icao","lat","lon"])
    df = df[df["icao"].str.match(pat)]
    df = df.drop_duplicates(subset=["icao"])
    # Ensure only ICAOs that appear in the overall candidate set (guards weird files)
    df = df[df["icao"].isin(set(icao))]

    return df[["icao","lat","lon"]]

def _pairs_from_flat_polygon(flat: list) -> list[tuple[float,float]]:
    if not isinstance(flat, list) or len(flat) < 6 or len(flat) % 2 != 0:
        raise ValueError("Polygon must be a flat list [lat0,lon0,...,latN,lonN] with N>=2")
    it = iter(flat)
    return [(float(lat), float(lon)) for lat, lon in zip(it, it)]

def _point_in_poly(lat: float, lon: float, poly: list[tuple[float,float]]) -> bool:
    # Ray casting in (lon=x, lat=y) space
    # The algorithm is a typical implementation of the ray-casting algorithm
    # (counting number of ray intersections until we are out of the polygon)
    # --> This algorithm counts the number of lines to the right of the point
    # If this number is even = outside, if it is odd = inside
    # The ((y1 > y) != (y2 > y)) states that the y-point must be y1 < y < y2 or the other way around
    # The (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1) is a reformulation of the linear representation:
    #   -> y1,x1 and y2,x2 define a linear function 
    #   -> it is: y' = x' * (y2-y1)/(x2-x1) + d
    #   -> d = y1 - x1 * (y2-y1)/(x2-x1)
    # ==> Observe: if y > x * (y2-y1)/(x2-x1) + d; then the line is to the right
    #   -> Transform to y > x * (y2-y1)/(x2-x1) + y1 - x1 * (y2-y1)/(x2-x1)
    #   -> Rewrite to: (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
    #   -> Add 1e-15 in the denominator to avoid 0-division, and done!

    x, y = lon, lat
    inside = False
    n = len(poly)
    for i in range(n):
        y1, x1 = poly[i][0], poly[i][1]
        y2, x2 = poly[(i+1) % n][0], poly[(i+1) % n][1]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1):
            inside = not inside
    return inside

def filter_icao_by_regions(oa_df: pd.DataFrame, regions_cfg: list[dict]) -> set[str]:
    if not regions_cfg:
        return set(oa_df["icao"])
    polys = [ _pairs_from_flat_polygon(r.get("polygon", [])) for r in regions_cfg ]
    mask_any = np.zeros(len(oa_df), dtype=bool)
    for poly in polys:
        mask_any |= oa_df.apply(lambda r: _point_in_poly(float(r["lat"]), float(r["lon"]), poly), axis=1).to_numpy()
    return set(oa_df.loc[mask_any, "icao"])


def load_filtered(csv_path: Path,
                  target_day: str | None,
                  # date range (for averaging meta)
                  date_start: str | None,
                  date_end: str | None,
                  chunksize: int) -> pd.DataFrame:
    """
    Load one or more CSVs and filter rows to (a) a single target day, or (b) an inclusive
    date range. In both cases we only keep flights that start and land on the same UTC day.
    """
    if (date_start is None) ^ (date_end is None):
        raise ValueError("Both --date-start and --date-end must be provided (or neither).")
    use_range = (date_start is not None and date_end is not None)

    # Collect CSV paths
    csv_paths = []
    if csv_path.is_dir():
        csv_paths = sorted(csv_path.glob("*.csv"))
    else:
        csv_paths = [csv_path]
    if not csv_paths:
        return pd.DataFrame(columns=[c for c in USECOLS if c not in DROP_COLS])

    filtered_chunks: list[pd.DataFrame] = []
    for path in csv_paths:
        for chunk in pd.read_csv(
            path,
            usecols=USECOLS,
            dtype="string",
            chunksize=chunksize,
            low_memory=False,
        ):
            first_day = chunk["firstseen"].str.slice(0, 10)
            last_day  = chunk["lastseen"].str.slice(0, 10)
            same_day = (first_day == last_day)
            if target_day is not None:
                mask = same_day & (first_day == target_day)
            else:
                # range is inclusive on both ends
                mask = same_day & (first_day >= date_start) & (first_day <= date_end)

            if not mask.any():
                continue

            df = chunk.loc[mask].copy()
            df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

            # Clean origin/destination
            for c in ["origin", "destination"]:
                df[c] = df[c].str.strip().str.upper()
                df[c] = df[c].mask(df[c].isin(BAD_TOKENS))

            df = df.dropna(subset=["origin", "destination"])
            if not df.empty:
                filtered_chunks.append(df)

    if filtered_chunks:
        return pd.concat(filtered_chunks, ignore_index=True)
    # empty frame with expected columns
    return pd.DataFrame(columns=[c for c in USECOLS if c not in DROP_COLS])


# -------------------------
# Modeling
# -------------------------
def build_models(
    df_raw: pd.DataFrame,
    *,
    bin_min: int,
    smooth_win: int,
    epsilon: float,
    alpha: float,
    global_backoff: float,
    min_tat: float,
    max_tat: float,
    min_dur: float,
    max_dur: float,
    smoothing: bool,
    icao_only: bool,
    min_samples_per_od: int,
    allowed_icao: set[str] | None,
    date_start: str | None,
    date_end: str | None,
    seed: int | None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]], np.ndarray, Dict[Tuple[str,str], np.ndarray], pd.Series]:
    rng = np.random.default_rng(seed)

    df = df_raw.copy()

    # Timestamps and basic cleaning
    df["origin"] = df["origin"].astype(str).str.strip().str.upper()
    df["destination"] = df["destination"].astype(str).str.strip().str.upper()
    df["firstseen_dt"] = pd.to_datetime(df["firstseen"], utc=True, errors="coerce")
    df["lastseen_dt"]  = pd.to_datetime(df["lastseen"],  utc=True, errors="coerce")
    df = df.dropna(subset=["firstseen_dt","lastseen_dt","origin","destination"])

    # ICAO filter (A-Z{4})
    if icao_only:
        pat_icao = r"^[A-Z]{4}$"
        df = df[df["origin"].str.match(pat_icao) & df["destination"].str.match(pat_icao)]
    # Optional ourairports whitelist filter (applies regardless of icao_only)
    if allowed_icao is not None:
        df = df[df["origin"].isin(allowed_icao) & df["destination"].isin(allowed_icao)]
        # If everything vanished, keep empty df to flow through gracefully

 


    # Minute-of-day bins
    minute_of_day = (df["firstseen_dt"].dt.hour * 60 + df["firstseen_dt"].dt.minute).astype(int)
    n_bins = (24*60) // bin_min
    bin_idx = (minute_of_day // bin_min).clip(0, n_bins-1)
    df["_bin"] = bin_idx
    df["_date"] = df["firstseen_dt"].dt.floor("D")
    
    """
    # Per-airport dep counts per bin
    dep_counts = (
        df.groupby(["origin","_bin"])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=range(n_bins), fill_value=0)
    )
    """

    # Per-airport dep counts per bin, averaged across days in the selection
    # 1) count per (origin, date, bin)
    daily = (
        df.groupby(["origin", "_date", "_bin"])
          .size()
    )
    daily_pivot = (
        daily.unstack("_bin", fill_value=0)
             .reindex(columns=range(n_bins), fill_value=0)
    )
    # 2) build full calendar day index for averaging (inclusive)
    if date_start is not None and date_end is not None:
        start = pd.to_datetime(date_start, utc=True).normalize()
        end   = pd.to_datetime(date_end,   utc=True).normalize()
        all_days = pd.date_range(start=start, end=end, freq="D", tz="UTC")
    else:
        # single-day mode: whatever is present
        all_days = pd.Index(sorted(daily_pivot.index.get_level_values("_date").unique()))
    # 3) ensure zeros for missing (origin, day) combinations
    origins = pd.Index(sorted(df["origin"].unique()))
    full_idx = pd.MultiIndex.from_product([origins, all_days], names=["origin","_date"])
    daily_pivot = daily_pivot.reindex(full_idx, fill_value=0)
    # 4) average over days
    dep_counts = daily_pivot.groupby(level="origin").mean()

    if smoothing:
        # Optional Laplace noise (non-negative)
        if epsilon and epsilon > 0:
            noise = rng.laplace(loc=0.0, scale=1.0/epsilon, size=dep_counts.shape)
            dep_counts = (dep_counts + noise).clip(lower=0)

        # Rolling mean over bins
        dep_counts = dep_counts.T.rolling(window=smooth_win, center=True, min_periods=1).mean().T

    # Global destination popularity q(d)
    global_dest_freq = (df.groupby("destination").size()).pipe(lambda s: s / s.sum())

    # OD counts per (o, b, d)
    od_counts_t = df.groupby(["origin","_bin","destination"]).size().rename("cnt").reset_index()

    # Assemble OD-time model with +alpha smoothing and optional global backoff (on-support)
    od_time_model: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    for (org, b), g in od_counts_t.groupby(["origin","_bin"]):
        g = g.set_index("destination")["cnt"].astype(float)

        # +alpha smoothing
        p = g.add(alpha)
        p = p / p.sum()

        if smoothing and global_backoff > 0:
            # Mix with global destination popularity limited to current support
            gf = global_dest_freq.reindex(p.index).fillna(0.0)
            gf = gf / gf.sum() if gf.sum() > 0 else p
            p = (1.0 - global_backoff) * p + global_backoff * gf

        p = p / p.sum()
        od_time_model.setdefault(org, {})[int(b)] = (p.index.to_numpy(), p.values)

    # Turnaround-time distribution (mins) from chained legs
    tat_mins = []
    for ac, g in df.sort_values("firstseen_dt").groupby("aircraft_uid"):
        if len(g) < 2:
            continue
        g = g.sort_values("firstseen_dt")
        prev = g.shift(1)
        chained = (prev["destination"] == g["origin"])
        delta = (g.loc[chained, "firstseen_dt"] - prev.loc[chained, "lastseen_dt"]).dt.total_seconds() / 60.0
        if not delta.empty:
            tat_mins.extend(delta.values.tolist())

    tat = np.array(tat_mins, dtype=float)
    tat = tat[np.isfinite(tat)]
    tat = tat[(tat >= min_tat) & (tat <= max_tat)]
    if tat.size == 0:
        tat = np.array([0.0, 15.0, 30.0, 45.0], dtype=float)

    # Duration distributions
    dur_series = ((df["lastseen_dt"] - df["firstseen_dt"]).dt.total_seconds() / 60.0).astype(float)
    df_dur = df.assign(dur_min=dur_series)
    df_dur = df_dur[(df_dur["dur_min"] >= min_dur) & (df_dur["dur_min"] <= max_dur)].copy()

    # Global fallback pool
    dur_dist = df_dur["dur_min"].to_numpy()
    if dur_dist.size == 0:
        dur_dist = np.array([60.0, 90.0, 120.0, 180.0], dtype=float)

    # Speed thresholds from global durations (tertiles)
    q33 = float(np.quantile(dur_dist, 1/3))
    q66 = float(np.quantile(dur_dist, 2/3))

    # OD-specific pools
    od_dur_dist: Dict[Tuple[str,str], np.ndarray] = {}
    for (o, d), g in df_dur.groupby(["origin", "destination"]):
        vals = g["dur_min"].to_numpy(dtype=float)
        if vals.size >= min_samples_per_od:
            od_dur_dist[(o, d)] = vals
        else:
            od_dur_dist[(o, d)] = dur_dist

    return dep_counts, od_time_model, tat, od_dur_dist, global_dest_freq, (q33, q66)


# -------------------------
# Saving artifacts (CSV)
# -------------------------
def save_artifacts(
    *,
    out_dir: Path,
    bin_min: int,
    airport_bins: pd.DataFrame,
    od_time_model: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    tat_dist: np.ndarray,
    od_dur_dist: Dict[Tuple[str,str], np.ndarray],
    global_dest_freq: pd.Series,
    dur_tertiles: Tuple[float,float],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # airport_bins: long format
    bins_long = (
        airport_bins
        .reset_index()
        .melt(id_vars="origin", var_name="bin", value_name="rate")
        .sort_values(["origin","bin"], kind="mergesort")
    )
    bins_long.to_csv(out_dir / "airport_bins.csv", index=False)

    # od_time_model: explode (origin, bin) → rows of (dest, prob)
    rows = []
    for o, bins in od_time_model.items():
        for b, (dests, probs) in bins.items():
            for d, p in zip(dests, probs):
                rows.append((o, b, d, float(p)))
    odm = pd.DataFrame(rows, columns=["origin","bin","destination","prob"]).sort_values(["origin","bin","destination"])
    odm.to_csv(out_dir / "od_time_model.csv", index=False)

    # tat_dist
    pd.DataFrame({"tat_min": tat_dist}).to_csv(out_dir / "tat_dist.csv", index=False)

    # od_dur_dist with speed_kts via global tertiles
    q33, q66 = dur_tertiles
    def duration_to_speed_kts(d: float) -> int:
        if d < q33: return 430
        if d < q66: return 450
        return 480

    # od_dur_dist: flatten to rows
    rows = []
    for (o, d), arr in od_dur_dist.items():
        for v in arr:
            rows.append((o, d, float(v), int(duration_to_speed_kts(float(v)))))
    odd = pd.DataFrame(rows, columns=["origin","destination","duration_min","speed_kts"]).sort_values(["origin","destination","duration_min"])
    odd.to_csv(out_dir / "od_dur_dist.csv", index=False)

    # global_dest_freq
    gdf = global_dest_freq.rename("freq").reset_index().rename(columns={"index":"destination"})
    gdf.to_csv(out_dir / "global_dest_freq.csv", index=False)

    # Optional metadata (useful for traceability)
    meta = pd.DataFrame(
        [{"artifact":"airport_bins","bin_minutes":bin_min},
         {"artifact":"od_time_model","bin_minutes":bin_min}]
    )
    meta.to_csv(out_dir / "metadata.csv", index=False)


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    # Resolve ourairports verification
    allowed_icao: set[str] | None = None
    if args.verify_ourairports:
        default_oa_path = Path("./ourairports/airports.csv")
        is_default_path = (args.ourairports_path.resolve() == default_oa_path.resolve())
        if args.ourairports_path.exists():
            try:

                oa_df = load_ourairports_df(args.ourairports_path)
                # Optional geographic region restriction from config
                regions_cfg = args.considered_geographic_regions or []
                if regions_cfg:
                    region_icao = filter_icao_by_regions(oa_df, regions_cfg)
                    if not region_icao:
                        print("[WARNING] - Geographic region filter matched 0 airports; ignoring region restriction.")
                        allowed_icao = set(oa_df["icao"])
                    else:
                        allowed_icao = region_icao
                else:
                    allowed_icao = set(oa_df["icao"])

            except Exception as e:
                raise RuntimeError(f"Failed to load ourairports file at {args.ourairports_path}: {e}") from e
        else:
            if is_default_path:
                print(f"[WARNING] - Could not verify airports from ourairports "
                      f"(assumed to be located in {args.ourairports_path}). "
                      f"Proceeding without external airport verification.")
            else:
                raise FileNotFoundError(f"OurAirports file not found: {args.ourairports_path}")


    # Validate date selection
    if args.date_start and args.date_end:
        target_day = None
    elif args.target_day:
        target_day = args.target_day
    else:
        # default to legacy single-day if nothing provided
        target_day = "2019-06-15"

    df = load_filtered(
        args.csv_path,
        target_day=target_day,
        date_start=args.date_start,
        date_end=args.date_end,
        chunksize=args.chunksize,
    )

    airport_bins, od_time_model, tat_dist, od_dur_dist, global_dest_freq, dur_tertiles = build_models(
        df,
        bin_min=args.bin_min,
        smooth_win=args.smooth_win,
        epsilon=args.epsilon,
        alpha=args.alpha,
        global_backoff=args.global_backoff,
        min_tat=args.min_tat,
        max_tat=args.max_tat,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        smoothing=args.smoothing,
        icao_only=args.icao_only,
        min_samples_per_od=args.min_samples_per_od,
        allowed_icao=allowed_icao,
        date_start=args.date_start,
        date_end=args.date_end,
        seed=args.seed,
    )

    # --- build auto-named experiment directory ---
    if args.flat_out:
        exp_dir = args.out_dir
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")

        if args.date_start and args.date_end:
            date_tag = f"range-{args.date_start}_to_{args.date_end}"
        else:
            date_tag = f"day-{target_day}"

        tag = (
            f"{date_tag}"
            f"_bin{args.bin_min}"
            f"_smooth{'T' if args.smoothing else 'F'}w{args.smooth_win}"
            f"_eps{args.epsilon:g}"
            f"_a{args.alpha:g}"
            f"_back{args.global_backoff:g}"
            f"_tat{int(args.min_tat)}-{int(args.max_tat)}"
            f"_dur{int(args.min_dur)}-{int(args.max_dur)}"
            f"_icao{'T' if args.icao_only else 'F'}"
            f"{'_seed'+str(args.seed) if args.seed is not None else ''}"
        )

        exp_dir = args.out_dir / f"{ts}__{tag}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    # persist full CLI args for traceability
    with open(exp_dir / "run_config.json", "w") as fh:
        # dump a clean, JSON-serializable view (Paths as strings)
        payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        # Include regions config explicitly (already in args but ensure JSON-friendly)
        if "considered_geographic_regions" in payload and payload["considered_geographic_regions"] is not None:
            payload["considered_geographic_regions"] = payload["considered_geographic_regions"]
        json.dump(payload, fh, indent=2)

    save_artifacts(
        out_dir=exp_dir,
        bin_min=args.bin_min,
        airport_bins=airport_bins,
        od_time_model=od_time_model,
        tat_dist=tat_dist,
        od_dur_dist=od_dur_dist,
        global_dest_freq=global_dest_freq,
        dur_tertiles=dur_tertiles,
    )

    print(f"Done. Artifacts written to: {exp_dir.resolve()}")

if __name__ == "__main__":
    main()
