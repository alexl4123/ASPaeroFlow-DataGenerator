#!/usr/bin/env python3
"""
ASPaeroFlow end-to-end pipeline

Priority of parameters:
  1) CLI flags (highest)
  2) Config file values (--config)
  3) Built-in defaults (lowest)

If --experiment-name (or config.experiment_name) is provided, all outputs go to:
  unparsed_experiment_data/<experiment_name>/
…so repeated runs reuse artifacts and can be skipped. If not provided, a
timestamped tag is used (as before).

Steps:
  1) Build model (skipped if artifacts already exist)
  2) Generate flights
  3) Build/ensure-connected navgraph (skipped if vertices & edges exist)
  4) Add sector capacities
  5) Generate filed flight plans
  6) Write manifest.json
"""

from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable, List, Tuple

# -------------------------
# utils
# -------------------------
def run(cmd, cwd: Path | None = None):
    print(f"[RUN] {' '.join(str(x) for x in cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

def file_exists(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except Exception:
        return False

def str2bool(v) -> bool:
    if isinstance(v, bool): return v
    s = str(v).strip().lower()
    if s in ("y","yes","true","1","on"):  return True
    if s in ("n","no","false","0","off"): return False
    raise argparse.ArgumentTypeError(f"Invalid bool: {v}")

def _cfg_get(cfg: dict, key: str, default=None):
    # accept both hyphen and underscore keys
    if key in cfg: return cfg[key]
    alt = key.replace("-", "_")
    if alt in cfg: return cfg[alt]
    return default

def read_region_names(cfg_path: Path | None) -> list[str]:
    if not cfg_path or not cfg_path.exists():
        return ["world"]
    try:
        with open(cfg_path, "r") as fh:
            cfg = json.load(fh) or {}
        regs = cfg.get("considered_geographic_regions", [])
        names = []
        for i, r in enumerate(regs):
            nm = r.get("region-name") or r.get("region_name") or f"poly{i+1}"
            names.append(str(nm).replace(",", "_"))
        return names or ["world"]
    except Exception:
        return ["world"]

def derive_exp_dir(out_root: Path, experiment_name: str | None,
                   cfg_path: Path | None, day: str | None,
                   scale: float, criterion: str, max_edge_km: float,
                   neighbor_index: str) -> Path:
    if experiment_name:
        return out_root / experiment_name
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    region_tag = ",".join(read_region_names(cfg_path))
    if len(region_tag) > 60:
        region_tag = region_tag[:57] + "..."
    tag = (
        f"regions-{region_tag}"
        f"_day-{(day or 'NA')}"
        f"_scale{scale:g}"
        f"_crit-{criterion}"
        f"_max{int(max_edge_km)}km"
        f"_idx-{neighbor_index}"
    )
    return out_root / f"{ts}__{tag}"

def _parse_list_float(v):
    if v is None:
        return None
    if isinstance(v, list):
        return [float(x) for x in v]
    s = str(v).strip()
    if not s:
        return None
    return [float(x) for x in s.replace(";", ",").split(",")]

def _parse_list_int(v):
    if v is None:
        return None
    if isinstance(v, list):
        return [int(x) for x in v]
    s = str(v).strip()
    if not s:
        return None
    return [int(x) for x in s.replace(";", ",").split(",")]

def _ds_name(scale: float, seed: int) -> str:
    # e.g., scale 0.001 -> "S0p001"; negative -> 'm' for minus
    sc = f"{scale:g}".replace(".", "p").replace("-", "m")
    return f"DATA_S{sc}_{seed}"

def _pair_grid(scales: List[float], seeds: List[int]) -> List[Tuple[float, int]]:
    pairs = []
    for s in scales:
        for r in seeds:
            pairs.append((s, r))
    return pairs


# -------------------------
# config-aware arg parsing
# -------------------------
def parse_args() -> argparse.Namespace:
    # pre-parse to learn --config and --experiment-name defaults from config
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre.add_argument("--experiment-name", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    cfg: dict = {}
    if pre_args.config and pre_args.config.exists():
        with open(pre_args.config, "r") as fh:
            cfg = json.load(fh) or {}

    def C(key, default=None):
        return _cfg_get(cfg, key, default)

    # Now the real parser with defaults sourced from config where available
    p = argparse.ArgumentParser(description="ASPaeroFlow end-to-end pipeline")

    # Common/config
    p.add_argument("--config", type=Path, default=pre_args.config,
                   help="JSON with considered_geographic_regions (shared by model & navgraph)")
    p.add_argument("--experiment-name", type=str, default=C("experiment-name", pre_args.experiment_name),
                   help="Stable experiment folder name under --out-root")
    p.add_argument("--out-root", type=Path, default=Path(C("out-root", "unparsed_experiment_data")),
                   help="Root folder for experiment outputs")

    # Model builder (00)
    p.add_argument("--csv-path", type=Path, default=Path(C("csv_path", "flightlist_20190601_20190630.csv")))
    p.add_argument("--ourairports", type=Path, default=Path(C("ourairports", "./ourairports/airports.csv")))
    p.add_argument("--bin-min", type=int, default=int(C("bin-min", 60)))
    p.add_argument("--target-day", type=str, default=C("target-day", None))
    p.add_argument("--date-start", type=str, default=C("date-start", None))
    p.add_argument("--date-end",   type=str, default=C("date-end",   None))
    p.add_argument("--model-chunksize", type=int, default=int(C("model-chunksize", 250_000)))
    p.add_argument("--seed", type=int, default=int(C("seed", 42)))

    # Data generator (01)
    p.add_argument("--day", type=str, default=C("day", None),
                   help="UTC YYYY-MM-DD to simulate (defaults to --target-day or 2019-06-15)")
    p.add_argument("--scale", type=float, default=float(C("scale", 1.0)))
    # Multi-sampling
    p.add_argument("--flight-scales", type=str, default=None,
                   help="Comma-separated list of scales (overrides config flight-scales)")
    p.add_argument("--flight-seeds", type=str, default=None,
                   help="Comma-separated list of seeds (overrides config flight-seeds)")
 

    # Navgraph (02)
    p.add_argument("--navdir", type=Path, default=Path(C("navdir", "./test_navpoints")))
    p.add_argument("--criterion", type=str, choices=["rng","gabriel"], default=C("criterion", "rng"))
    p.add_argument("--max-edge-km", type=float, default=float(C("max-edge-km", 350.0)))
    p.add_argument("--neighbor-index", type=str, choices=["balltree","bruteforce"], default=C("neighbor_index", "balltree"))
    p.add_argument("--centroid-knn", type=int, default=int(C("centroid-knn", 12)))
    p.add_argument("--enforce-connected", type=str, default=C("enforce-connected", "true"),
                   help="true/false; ensure a single connected component (default true)")

    # Sectors (03)
    p.add_argument("--cap-enroute", type=int, default=int(C("cap-enroute", 60)))
    p.add_argument("--cap-airport", type=int, default=int(C("cap-airport", 60000)))

    # Flight plans (04)
    p.add_argument("--time-granularity", type=int, default=int(C("time-granularity", 1)),
                   help="1=hourly slots, 4=15-min slots, ...")

    # Behavior flags
    p.add_argument("--skip-nav-od-filter", type=str, default=C("skip-nav-od-filter", "false"),
                   help="true/false; if true, do NOT restrict navgraph airports to OD file")
    p.add_argument("--force-rebuild-model", type=str, default=C("force-rebuild-model", "false"))
    p.add_argument("--force-rebuild-graph", type=str, default=C("force-rebuild-graph", "false"))

    args = p.parse_args()

    # normalize booleans
    args.enforce_connected   = str2bool(args.enforce_connected)
    args.skip_nav_od_filter  = str2bool(args.skip_nav_od_filter)
    args.force_rebuild_model = str2bool(args.force_rebuild_model)
    args.force_rebuild_graph = str2bool(args.force_rebuild_graph)

    # derive multi-sampling lists with proper precedence: CLI > config > fallback
    cfg_scales = _cfg_get(cfg, "flight-scales", None)
    cfg_seeds  = _cfg_get(cfg, "flight-seeds",  None)
    cli_scales = _parse_list_float(args.flight_scales)
    cli_seeds  = _parse_list_int(args.flight_seeds)
    args.scales = cli_scales if cli_scales is not None else (_parse_list_float(cfg_scales) if cfg_scales is not None else [args.scale])
    args.seeds  = cli_seeds  if cli_seeds  is not None else (_parse_list_int(cfg_seeds)  if cfg_seeds  is not None else [args.seed])
    # ensure unique order-preserving
    args.scales = list(dict.fromkeys(args.scales)); args.seeds = list(dict.fromkeys(args.seeds))

    return args

# -------------------------
# main
# -------------------------
def main():
    a = parse_args()

    a.out_root.mkdir(parents=True, exist_ok=True)

    sim_day = a.day or a.target_day or "2019-06-15"
    exp_dir = derive_exp_dir(
        a.out_root, a.experiment_name, a.config, sim_day, a.scale,
        a.criterion, a.max_edge_km, a.neighbor_index
    )
    (exp_dir / "model").mkdir(parents=True, exist_ok=True)
    # per-dataset folders will live directly under exp_dir (DATA_S<...>_<seed>)
    (exp_dir / "navgraph").mkdir(parents=True, exist_ok=True)

    model_dir = exp_dir / "model"
    # data subfolders will be created per (scale,seed)
    nav_dir   = exp_dir / "navgraph"

    # 1) MODEL (00) — skip if artifacts exist and not forced
    model_needed = a.force_rebuild_model or not all([
        file_exists(model_dir / "airport_bins.csv"),
        file_exists(model_dir / "od_time_model.csv"),
        file_exists(model_dir / "tat_dist.csv"),
        file_exists(model_dir / "od_dur_dist.csv"),
        file_exists(model_dir / "global_dest_freq.csv"),
    ])
    if model_needed:
        cmd = [
            "python", "00_model_generation_script_refactored.py",
            "--csv-path", str(a.csv_path),
            "--ourairports-path", str(a.ourairports),
            "--chunksize", str(a.model_chunksize),
            "--bin-min", str(a.bin_min),
            "--seed", str(a.seed),
            "--out-dir", str(model_dir),
            "--flat-out",
            "--verify-ourairports", "true",
            "--icao-only", "true",
        ]
        if a.config:      cmd += ["--config", str(a.config)]
        if a.date_start:  cmd += ["--date-start", a.date_start]
        if a.date_end:    cmd += ["--date-end",   a.date_end]
        if a.target_day:  cmd += ["--target-day", a.target_day]
        run(cmd)
    else:
        print("[SKIP] Model artifacts present → step 1 skipped.")

    # 2) DATA (01)
    # 2.A) DATA (01)
    # Build full (scale,seed) grid
    ds_pairs = _pair_grid(a.scales, a.seeds)
    if not ds_pairs:
        ds_pairs = [(a.scale, a.seed)]
    first_scale, first_seed = ds_pairs[0]
    first_ds_dir = exp_dir / _ds_name(first_scale, first_seed)
    first_ds_dir.mkdir(parents=True, exist_ok=True)

    # 2.B) First DATA sample (01) to allow OD restriction for navgraph (if requested)
    print(f"[2/6] Generating first dataset for navgraph OD reference: scale={first_scale:g}, seed={first_seed}")
    run([
        "python", "01_data_generation_script_refactored.py",
        "--model-dir", str(model_dir),
        "--day", sim_day,
        "--scale", str(first_scale),
        "--seed", str(first_seed),
        "--out-dir", str(first_ds_dir),
        "--flat-out",
    ])


    # 3) NAVGRAPH (02) — skip if vertices & edges exist and not forced
    graph_needed = a.force_rebuild_graph or not all([
        file_exists(nav_dir / "vertices.csv"),
        file_exists(nav_dir / "edges.csv"),
    ])
    if graph_needed:
        cmd = [
            "python", "02_graph_generator.py",
            "--ourairports", str(a.ourairports),
            "--navdir", str(a.navdir),
            "--criterion", a.criterion,
            "--max-edge-km", str(a.max_edge_km),
            "--neighbor-index", a.neighbor_index,
            "--centroid-knn", str(a.centroid_knn),
            "--enforce-connected", "true" if a.enforce_connected else "false",
            "--out-dir", str(nav_dir),
            "--flat-out",
        ]

        if a.config:
            cmd += ["--config", str(a.config)]
        if not a.skip_nav_od_filter:
            od_file = first_ds_dir / "flights.csv"
            if file_exists(od_file):
                cmd += ["--od-file", str(od_file)]

        run(cmd)
    else:
        print("[SKIP] Navgraph present → step 3 skipped.")

    # 4) SECTOR CAPACITIES (03)
    # NOTE: Corrected call — script expects '--path', not '--vertices-dir'
    run([
        "python", "03_sector_capacity_generator.py",
        "--path", str(nav_dir),
        "--cap-enroute", str(a.cap_enroute),
        "--cap-airport", str(a.cap_airport),
    ])

    # 5) DATASETS + FILED PLANS for all (scale,seed) pairs
    datasets = []
    for idx, (scale, seed) in enumerate(ds_pairs):
        ds_dir = exp_dir / _ds_name(scale, seed)
        if not ds_dir.exists():
            ds_dir.mkdir(parents=True, exist_ok=True)
        # Regenerate data if missing (idempotent)
        flights_csv = ds_dir / "flights.csv"
        if not file_exists(flights_csv):
            print(f"[5/{idx+1}] Generating data: scale={scale:g}, seed={seed}")
            run([
                "python", "01_data_generation_script_refactored.py",
                "--model-dir", str(model_dir),
                "--day", sim_day,
                "--scale", str(scale),
                "--seed", str(seed),
                "--out-dir", str(ds_dir),
                "--flat-out",
            ])
        else:
            print(f"[SKIP] Data present for {ds_dir.name}")

        # Filed flight plans (always ensure present)
        print(f"[5/{idx+1}] Generating filed plans for {ds_dir.name}")
        run([
            "python", "04_simplified_filed_flight_plan_generator.py",
            "--data-dir", str(ds_dir),
            "--navgraph-dir", str(nav_dir),
            "--time-granularity", str(a.time_granularity),
        ])
        datasets.append(str(ds_dir.resolve()))


    # 6) manifest
    manifest = {
        "experiment_dir": str(exp_dir.resolve()),
        "experiment_name": a.experiment_name,
        "config": str(a.config.resolve()) if a.config else None,
        "regions": read_region_names(a.config),
        "model_dir": str(model_dir.resolve()),
        "navgraph_dir": str(nav_dir.resolve()),
        "datasets": datasets,
        "parameters": {
            "day": sim_day,
            "scale": a.scale,
            "bin_min": a.bin_min,
            "criterion": a.criterion,
            "max_edge_km": a.max_edge_km,
            "neighbor_index": a.neighbor_index,
            "centroid_knn": a.centroid_knn,
            "enforce_connected": a.enforce_connected,
            "cap_enroute": a.cap_enroute,
            "cap_airport": a.cap_airport,
            "time_granularity": a.time_granularity,
            "seed": a.seed,
            "csv_path": str(a.csv_path),
            "ourairports": str(a.ourairports),
            "navdata": str(a.navdir),
            "date_range": [a.date_start, a.date_end],
            "target_day": a.target_day,
        },
        "artifacts": {
            "model": ["airport_bins.csv","od_time_model.csv","tat_dist.csv","od_dur_dist.csv","global_dest_freq.csv"],
            "navgraph": ["vertices.csv","edges.csv","sectors.csv"],
        }
    }
    with open(exp_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n[✓] Pipeline complete.\n→ {exp_dir.resolve()}")

if __name__ == "__main__":
    main()
