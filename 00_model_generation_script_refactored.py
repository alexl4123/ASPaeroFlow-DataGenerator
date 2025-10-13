#!/usr/bin/env python3
"""
Flight OD/temporal model builder (CLI)

Reads OpenSky-like flight CSVs, filters to a target day, and exports model artifacts:
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

"""
from __future__ import annotations

import argparse
from pathlib import Path
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

    p = argparse.ArgumentParser(description="Build OD/time model and export artifacts to CSV.")
    p.add_argument("--csv-path", type=Path, default=Path("flightlist_20190601_20190630.csv"))
    p.add_argument("--target-day", type=str, default="2019-06-15", help="UTC day YYYY-MM-DD")
    p.add_argument("--chunksize", type=int, default=250_000)

    # Model inputs (defaults = original script)
    p.add_argument("--bin-min", type=int, default=60, help="Minutes per time bin")
    p.add_argument("--smooth-win", type=int, default=3, help="Rolling window (bins) for smoothing")
    p.add_argument("--epsilon", type=float, default=2.0, help="Laplace noise scale via 1/epsilon; <=0 disables")
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet +alpha smoothing for OD")
    p.add_argument("--global-backoff", type=float, default=0.05, help="Mixture weight with global dest freq")
    p.add_argument("--min-tat", type=float, default=0, help="Min turnaround minutes")
    p.add_argument("--max-tat", type=float, default=60, help="Max turnaround minutes")
    p.add_argument("--min-dur", type=float, default=1, help="Min duration minutes")
    p.add_argument("--max-dur", type=float, default=900, help="Max duration minutes")
    p.add_argument("--smoothing", type=str, default="false", help="true/false")
    p.add_argument("--icao-only", type=str, default="true", help="true/false")
    p.add_argument("--min-samples-per-od", type=int, default=1, help="Minimum samples for OD-specific durations")
    p.add_argument("--seed", type=int, default=None, help="Random seed (for noise/sampling reproducibility)")
    p.add_argument("--out-dir", type=Path, default=Path("./model_out"))

    args = p.parse_args()
    args.smoothing = str2bool(args.smoothing)
    args.icao_only = str2bool(args.icao_only)
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


def load_filtered(csv_path: Path, target_day: str, chunksize: int) -> pd.DataFrame:
    filtered_chunks = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=USECOLS,
        dtype="string",
        chunksize=chunksize,
        low_memory=False,
    ):
        first_day = chunk["firstseen"].str.slice(0, 10)
        last_day  = chunk["lastseen"].str.slice(0, 10)
        mask = (first_day == target_day) & (last_day == target_day)
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

    # Minute-of-day bins
    minute_of_day = (df["firstseen_dt"].dt.hour * 60 + df["firstseen_dt"].dt.minute).astype(int)
    n_bins = (24*60) // bin_min
    bin_idx = (minute_of_day // bin_min).clip(0, n_bins-1)
    df["_bin"] = bin_idx

    # Per-airport dep counts per bin
    dep_counts = (
        df.groupby(["origin","_bin"])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=range(n_bins), fill_value=0)
    )

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

    # OD-specific pools
    od_dur_dist: Dict[Tuple[str,str], np.ndarray] = {}
    for (o, d), g in df_dur.groupby(["origin", "destination"]):
        vals = g["dur_min"].to_numpy(dtype=float)
        if vals.size >= min_samples_per_od:
            od_dur_dist[(o, d)] = vals
        else:
            od_dur_dist[(o, d)] = dur_dist

    return dep_counts, od_time_model, tat, od_dur_dist, global_dest_freq


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

    # od_dur_dist: flatten to rows
    rows = []
    for (o, d), arr in od_dur_dist.items():
        for v in arr:
            rows.append((o, d, float(v)))
    odd = pd.DataFrame(rows, columns=["origin","destination","duration_min"]).sort_values(["origin","destination"])
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

    df = load_filtered(args.csv_path, args.target_day, args.chunksize)

    airport_bins, od_time_model, tat_dist, od_dur_dist, global_dest_freq = build_models(
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
        seed=args.seed,
    )

    save_artifacts(
        out_dir=args.out_dir,
        bin_min=args.bin_min,
        airport_bins=airport_bins,
        od_time_model=od_time_model,
        tat_dist=tat_dist,
        od_dur_dist=od_dur_dist,
        global_dest_freq=global_dest_freq,
    )

    print(f"Done. Artifacts written to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
