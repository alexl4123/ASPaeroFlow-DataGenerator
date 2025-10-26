#!/usr/bin/env python3
"""
Synthetic flight day generator

Loads model artifacts from an output directory (default: ./model_out)
and generates a synthetic day of flights using:
  - airport_bins.csv
  - od_time_model.csv
  - tat_dist.csv
  - od_dur_dist.csv
  - global_dest_freq.csv
  - metadata.csv (optional, to recover bin_minutes)

Outputs a CSV of flights:
  flight_id, aircraft_id, origin, destination, departure_time (UTC ISO-8601)
"""

from __future__ import annotations

import argparse
import heapq
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np
import pandas as pd


# -------------------------
# Loading artifacts
# -------------------------

def load_bin_minutes(model_dir: Path, default: int = 60) -> int:
    meta_path = model_dir / "metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        # Prefer airport_bins row if present; else any bin_minutes entry
        for pref in ("airport_bins", "od_time_model"):
            m = meta[meta["artifact"] == pref]
            if not m.empty and "bin_minutes" in m.columns:
                val = m["bin_minutes"].iloc[0]
                try:
                    return int(val)
                except Exception:
                    pass
        if "bin_minutes" in meta.columns and not meta["bin_minutes"].isna().all():
            try:
                return int(meta["bin_minutes"].dropna().iloc[0])
            except Exception:
                pass
    return default


def load_airport_bins(model_dir: Path, bin_min: int) -> pd.DataFrame:
    """
    Returns a wide DataFrame: index=origin, columns=bin (0..n_bins-1), values=lambda (expected dep count)
    """
    df = pd.read_csv(model_dir / "airport_bins.csv")
    if df.empty:
        # Return an empty, correctly-shaped wide table
        n_bins = (24 * 60) // bin_min
        return pd.DataFrame(columns=range(n_bins)).set_index(pd.Index([], name="origin"))
    df["bin"] = df["bin"].astype(int)
    n_bins = (24 * 60) // bin_min
    wide = (
        df.pivot(index="origin", columns="bin", values="rate")
          .reindex(columns=range(n_bins), fill_value=0.0)
          .fillna(0.0)
    )
    wide.columns.name = None
    return wide


def load_od_time_model(model_dir: Path) -> Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """
    Returns dict: origin -> { bin -> (dests ndarray[str], probs ndarray[float]) }
    """
    df = pd.read_csv(model_dir / "od_time_model.csv")
    if df.empty:
        return {}
    df["bin"] = df["bin"].astype(int)

    od_model: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    for (o, b), g in df.groupby(["origin", "bin"], sort=False):
        dests = g["destination"].astype(str).to_numpy()
        probs = g["prob"].astype(float).to_numpy()
        s = probs.sum()
        if s > 0:
            probs = probs / s
        od_model.setdefault(o, {})[int(b)] = (dests, probs)
    return od_model


def load_tat(model_dir: Path) -> np.ndarray:
    df = pd.read_csv(model_dir / "tat_dist.csv")
    col = "tat_min" if "tat_min" in df.columns else df.columns[0]
    arr = df[col].astype(float).to_numpy()
    if arr.size == 0:
        arr = np.array([0.0, 15.0, 30.0, 45.0], dtype=float)
    return arr


def load_od_durations(model_dir: Path) -> Tuple[Dict[Tuple[str, str], np.ndarray], np.ndarray]:
    """
    Returns:
      od_map: dict[(o,d)] -> (durations[], speeds[])
      glob_dur: durations[] (global fallback)
      glob_spd: speeds[] (global fallback; if missing speeds, infer by tertiles)
    """

    df = pd.read_csv(model_dir / "od_dur_dist.csv")
    if df.empty:
        # strong fallback
        fallback_d = np.array([60.0, 90.0, 120.0, 180.0], dtype=float)
        # classify by tertiles of fallback
        q33, q66 = np.quantile(fallback_d, [1/3, 2/3])
        def sp(x): return 430 if x<q33 else (450 if x<q66 else 480)
        fallback_s = np.array([sp(x) for x in fallback_d], dtype=int)
        return {}, fallback_d, fallback_s

    df["duration_min"] = df["duration_min"].astype(float)
    if "speed_kts" not in df.columns:
        # Back-compat: infer speeds via tertiles from observed durations
        q33, q66 = np.quantile(df["duration_min"].to_numpy(), [1/3, 2/3])
        df["speed_kts"] = np.where(df["duration_min"] < q33, 430,
                            np.where(df["duration_min"] < q66, 450, 480)).astype(int)

    od_map: Dict[Tuple[str,str], Tuple[np.ndarray, np.ndarray]] = {}
    for (o, d), g in df.groupby(["origin","destination"], sort=False):
        od_map[(str(o), str(d))] = (
            g["duration_min"].to_numpy(dtype=float),
            g["speed_kts"].to_numpy(dtype=int),
        )

    glob_dur = df["duration_min"].to_numpy(dtype=float)
    glob_spd = df["speed_kts"].to_numpy(dtype=int)
    return od_map, glob_dur, glob_spd


def load_global_dest_freq(model_dir: Path) -> pd.Series:
    df = pd.read_csv(model_dir / "global_dest_freq.csv")
    if df.empty:
        return pd.Series([], dtype=float)
    # Column may be either 'destination'/'freq' or 'index'/'freq' depending on CSV writer
    if "destination" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "destination"})
    s = pd.Series(df["freq"].astype(float).to_numpy(), index=df["destination"].astype(str))
    s = s / s.sum() if s.sum() > 0 else s
    return s


# -------------------------
# Sampling helpers
# -------------------------

def _sample_time_in_bin(day_str: str, bin_index: int, bin_min: int, rng: np.random.Generator) -> datetime:
    day0 = datetime.fromisoformat(day_str).replace(tzinfo=timezone.utc)
    start = day0 + timedelta(minutes=bin_index * bin_min)
    offs_sec = rng.random() * (bin_min * 60.0)  # uniform within bin
    return start + timedelta(seconds=float(offs_sec))


def _pick_dest_time(
    origin: str,
    dep_time: datetime,
    od_time_model: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    global_dest_freq: pd.Series,
    bin_min: int,
    rng: np.random.Generator,
) -> str:
    b = (dep_time.hour * 60 + dep_time.minute) // bin_min
    b = int(max(0, min(b, (24 * 60) // bin_min - 1)))
    tb = od_time_model.get(origin, {})
    destination = None

    iter_ = 0
    while iter_ < 200:
        if b in tb:
            dests, probs = tb[b]
            destination = str(rng.choice(dests, p=(probs / probs.sum() if probs.sum() > 0 else None)))
        # Fallbacks: any other bin for this origin, else global
        if tb:
            dests, probs = next(iter(tb.values()))
            destination = str(rng.choice(dests, p=(probs / probs.sum() if probs.sum() > 0 else None)))
        if not global_dest_freq.empty:
            destination =  str(rng.choice(global_dest_freq.index.to_numpy(), p=global_dest_freq.values))

        if destination != origin:
            break
        else:
            iter_ += 1

    if destination == origin:
        print(f"[WARN]: DEST=ORIGIN for {origin}")

    # Hard fallback if everything is empty
    return destination

"""
def _pick_duration_od(
    origin: str,
    dest: str,
    od_dur_dist: Dict[Tuple[str, str], np.ndarray],
    dur_dist: np.ndarray,
    rng: np.random.Generator,
) -> float:
    pool = od_dur_dist.get((origin, dest), dur_dist)
    return float(rng.choice(pool)) if pool.size else float(rng.choice(dur_dist))
"""

def _pick_duration_and_speed(origin: str, dest: str, od_map, glob_dur: np.ndarray, glob_spd: np.ndarray, rng: np.random.Generator):
    dur_arr, spd_arr = od_map.get((origin, dest), (glob_dur, glob_spd))
    if dur_arr.size == 0: dur_arr, spd_arr = glob_dur, glob_spd
    idx = rng.integers(0, dur_arr.size)
    return float(dur_arr[idx]), int(spd_arr[idx])


def _pick_turnaround(tat_dist: np.ndarray, rng: np.random.Generator) -> float:
    return float(rng.choice(tat_dist)) if tat_dist.size else 0.0


# -------------------------
# Generation
# -------------------------

def generate_synthetic_day(
    *,
    day_str: str,
    airport_bins: pd.DataFrame,
    od_time_model: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    global_dest_freq: pd.Series,
    tat_dist: np.ndarray,
    od_dur_dist: Dict[Tuple[str, str], np.ndarray],
    dur_dist: np.ndarray,
    glob_spd: np.ndarray,
    bin_min: int,
    scale: float = 1.0,
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      flight_id, aircraft_id, origin, destination, departure_time  (ISO 8601, UTC)
    """
    rng = np.random.default_rng(seed)

    # 1) Build all departure events via Poisson sampling per airport×bin
    events: list[tuple[datetime, str]] = []
    n_bins = (24 * 60) // bin_min
    # Ensure airport_bins has all bins as columns
    if airport_bins.shape[1] != n_bins:
        airport_bins = airport_bins.reindex(columns=range(n_bins), fill_value=0.0)

    for origin, row in airport_bins.iterrows():
        lam = np.maximum(row.values.astype(float) * float(scale), 0.0)
        k = rng.poisson(lam)  # samples per bin
        for b, kk in enumerate(k):
            for _ in range(int(kk)):
                events.append((_sample_time_in_bin(day_str, b, bin_min, rng), str(origin)))

    # Sort departures by time to process chaining
    events.sort(key=lambda x: x[0])

    # 2) Aircraft pools per airport: min-heaps keyed by ready_time
    available: Dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    aircraft_speed: Dict[str, int] = {}
    flights = []
    ac_counter = 0
    fl_counter = 0

    for dep_time, origin in events:
        # Pop one ready aircraft if any (ready_time <= dep_time)
        ac_id = None
        heap = available[origin]
        while heap and heap[0][0] <= dep_time:
            _, cand_id = heapq.heappop(heap)
            ac_id = cand_id
            break
        if ac_id is None:
            ac_counter += 1
            ac_id = f"AC{ac_counter:06d}"

        # Route + duration
        dest = _pick_dest_time(origin, dep_time, od_time_model, global_dest_freq, bin_min, rng)
        dur_min, speed_kts = _pick_duration_and_speed(origin, dest, od_dur_dist, dur_dist, glob_spd, rng)
        arr_time = dep_time + timedelta(minutes=dur_min)

        # Assign aircraft speed on first use
        if ac_id not in aircraft_speed:
            aircraft_speed[ac_id] = speed_kts

        # Turnaround & push aircraft into dest heap
        tat_min = _pick_turnaround(tat_dist, rng)
        ready_time = arr_time + timedelta(minutes=tat_min)
        heapq.heappush(available[dest], (ready_time, ac_id))

        # Record flight
        fl_counter += 1
        flights.append({
            "flight_id": f"F{fl_counter:07d}",
            "aircraft_id": ac_id,
            "origin": origin,
            "destination": dest,
            "departure_time": dep_time.isoformat().replace("+00:00", "Z"),
        })

    flights_df = pd.DataFrame(flights)
    aircrafts_df = pd.DataFrame({"aircraft_id": list(aircraft_speed.keys()),
                                 "speed_kts": [aircraft_speed[a] for a in aircraft_speed]})

    return flights_df, aircrafts_df

# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a synthetic flight day from model artifacts.")
    p.add_argument("--model-dir", type=Path, default=Path("./model_out"), help="Directory with exported artifacts")
    p.add_argument("--day", type=str, default="2019-06-15", help="UTC day YYYY-MM-DD to simulate")
    p.add_argument("--scale", type=float, default=1.0, help="Scale factor for departures")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument("--bin-min", type=int, default=None, help="Override bin minutes (otherwise read from metadata)")
    p.add_argument("--out-dir", type=Path, default=Path("./data_out"), help="Default output folder")
    p.add_argument("--flat-out", action="store_true",
                   help="Write files directly into the parent folder of --out (disable auto-named subfolder).")

    return p.parse_args()


def main():
    args = parse_args()
    model_dir = args.model_dir
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    bin_min = args.bin_min or load_bin_minutes(model_dir, default=60)

    airport_bins = load_airport_bins(model_dir, bin_min)
    od_time_model = load_od_time_model(model_dir)
    tat_dist = load_tat(model_dir)
    od_dur_dist, dur_dist, glob_spd = load_od_durations(model_dir)
    global_dest_freq = load_global_dest_freq(model_dir)

    flights_df, aircrafts_df = generate_synthetic_day(
        day_str=args.day,
        airport_bins=airport_bins,
        od_time_model=od_time_model,
        global_dest_freq=global_dest_freq,
        tat_dist=tat_dist,
        od_dur_dist=od_dur_dist,
        dur_dist=dur_dist,
        glob_spd=glob_spd,
        bin_min=bin_min,
        scale=args.scale,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # --- build auto-named experiment directory ---
    if args.flat_out:
        exp_dir = args.out_dir
        out_flights = exp_dir / "flights.csv"
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
        tag = (
            f"day-{args.day}"
            f"_scale{args.scale:g}"
            f"_bin{bin_min}"
            f"{'_seed'+str(args.seed) if args.seed is not None else ''}"
            f"_model-{args.model_dir.name}"
        )
        exp_dir = args.out_dir / f"{ts}__{tag}"
        out_flights = exp_dir / "flights.csv"
    exp_dir.mkdir(parents=True, exist_ok=True)
    out_dir_str = str(args.out_dir)
    # persist full CLI + resolved values
    with open(exp_dir / "run_config.json", "w") as fh:
        cfg = {}
        var_dict = dict(vars(args))
        for var in var_dict.keys():
            cfg[var] = str(var_dict[var])
        cfg["model_dir"] = str(args.model_dir)
        cfg["out"] = out_dir_str
        cfg["resolved_bin_min"] = bin_min
        cfg["number_flights"] = flights_df.shape[0]
        
        json.dump(cfg, fh, indent=2)

    # before saving flights.csv
    s = flights_df["departure_time"].astype(str).str.strip()
    # clean any legacy ".Z" artifacts just in case
    s = s.str.replace(r"\.Z$", "Z", regex=True)

    flights_df["departure_time"] = pd.to_datetime(
        s, format="ISO8601", utc=True, errors="raise"
    )

    # choose how you want to serialize (seconds precision here)
    flights_df.to_csv(out_flights, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")

    aircrafts_path = exp_dir / "aircrafts.csv"
    aircrafts_df.to_csv(aircrafts_path, index=False)

    print(f"Wrote {len(flights_df):,} flights to {str(out_flights)}")
    print(f"Wrote {len(aircrafts_df):,} aircraft rows to {aircrafts_path.resolve()}")

if __name__ == "__main__":
    main()
