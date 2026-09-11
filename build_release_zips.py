#!/usr/bin/env python3
"""
Build the release archives: one zip per time granularity, each containing the parsed instances
with the capacity sweep MATERIALISED (every PCAP level as a full, ready-to-solve instance).

Layout inside <root>_TG<n>.zip:

    experiment_data_V2_large_scaling_TG<n>/
        <EXPERIMENT>/
            PCAP010/<scale>_SEED<seed>/{flights.csv, sectors.csv, ...}
            ...
            PCAP100/<scale>_SEED<seed>/...

Each PCAP directory is a complete instance: the base parsed dataset with that level's sectors.csv
substituted. No post-processing needed by the user, which is the point of materialising rather
than shipping overlays.

    python3 build_release_zips.py --data-root 20260910_data_cluster --out-dir release
    python3 build_release_zips.py --data-root ... --granularities 60 --dry-run
"""
import argparse, os, shutil, subprocess, sys, zipfile
from pathlib import Path

LEVELS = [f"PCAP{p:03d}" for p in range(10, 101, 10)]


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}"
        n /= 1024


def tree_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def build_one(data_root: Path, tg: int, staging: Path, dry: bool) -> Path | None:
    parsed = data_root / f"experiment_data_V2_large_scaling_TG{tg}"
    overlays = data_root / f"capacity_overlays_V2_large_scaling_TG{tg}"
    if not parsed.is_dir():
        print(f"[skip] {parsed} not found")
        return None
    stage = staging / parsed.name
    if stage.exists():
        shutil.rmtree(stage)
    exps = sorted(d for d in parsed.iterdir() if d.is_dir())
    print(f"\n=== TG={tg}: {len(exps)} experiments x {len(LEVELS)} levels ===")
    for e in exps:
        ov = overlays / e.name
        if not ov.is_dir():
            print(f"  [WARN] no overlays for {e.name}; shipping base only")
            if not dry:
                shutil.copytree(e, stage / e.name / "BASE")
            continue
        for lvl in LEVELS:
            src_lvl = ov / lvl
            if not src_lvl.is_dir():
                print(f"  [WARN] {e.name}: missing {lvl}")
                continue
            if dry:
                continue
            for ds in sorted(src_lvl.glob("*")):
                if not (ds / "sectors.csv").exists():
                    continue
                out = stage / e.name / lvl / ds.name
                shutil.copytree(e / ds.name, out)
                shutil.copy2(ds / "sectors.csv", out / "sectors.csv")
        print(f"  {e.name[:58]:<58} done")
    return stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("release"))
    ap.add_argument("--staging", type=Path, default=None,
                    help="scratch dir for the materialised tree (default: <out-dir>/_staging)")
    ap.add_argument("--granularities", default="1,4,15,60")
    ap.add_argument("--keep-staging", action="store_true")
    ap.add_argument("--skip-parsed", action="store_true",
                    help="only archive the unparsed roots (parsed zips already built)")
    ap.add_argument("--unparsed", choices=["none", "per-granularity", "single"],
                    default="none",
                    help="also archive the unparsed roots. They are stage 00-04 output (demand "
                         "model, navgraph, per-dataset flights/filed_flights) -- not needed to "
                         "solve an instance, but required to reproduce generation or to run the "
                         "F1/F2 model audits, which cannot work on parsed data.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    data_root = a.data_root.expanduser()
    out_dir = a.out_dir.expanduser(); out_dir.mkdir(parents=True, exist_ok=True)
    staging = (a.staging or out_dir / "_staging").expanduser(); staging.mkdir(parents=True, exist_ok=True)

    gran = [int(x) for x in a.granularities.split(",") if x.strip()]
    total_raw = total_zip = 0
    for tg in ([] if a.skip_parsed else gran):
        stage = build_one(data_root, tg, staging, a.dry_run)
        if stage is None or a.dry_run:
            continue
        raw = tree_size(stage); total_raw += raw
        zpath = out_dir / f"{stage.name}.zip"
        print(f"  zipping {human(raw)} -> {zpath.name} ...", flush=True)
        # -1 (fast) is the right trade here: these CSVs are highly compressible and the archive is
        # large, so maximum compression costs a lot of time for a few percent.
        subprocess.run(["zip", "-r", "-q", "-1", str(zpath.resolve()), stage.name],
                       cwd=staging, check=True)
        z = zpath.stat().st_size; total_zip += z
        print(f"  {zpath.name}: {human(raw)} raw -> {human(z)} zipped ({raw/z:.1f}x)")
        if not a.keep_staging:
            shutil.rmtree(stage)
    if a.unparsed != "none" and not a.dry_run:
        roots = [data_root / f"unparsed_experiment_data_V2_large_scaling_TG{tg}" for tg in gran]
        roots = [r for r in roots if r.is_dir()]
        if not roots:
            print("\n[WARN] no unparsed roots found")
        elif a.unparsed == "per-granularity":
            print()
            for r in roots:
                raw = tree_size(r); total_raw += raw
                zp = out_dir / f"{r.name}.zip"
                print(f"  zipping {human(raw)} -> {zp.name} ...", flush=True)
                subprocess.run(["zip", "-r", "-q", "-1", str(zp.resolve()), r.name],
                               cwd=r.parent, check=True)
                z = zp.stat().st_size; total_zip += z
                print(f"  {zp.name}: {human(raw)} raw -> {human(z)} zipped ({raw/z:.1f}x)")
        else:
            raw = sum(tree_size(r) for r in roots); total_raw += raw
            zp = out_dir / "unparsed_experiment_data_V2_large_scaling_ALL.zip"
            print(f"\n  zipping {human(raw)} -> {zp.name} ...", flush=True)
            subprocess.run(["zip", "-r", "-q", "-1", str(zp.resolve())] + [r.name for r in roots],
                           cwd=roots[0].parent, check=True)
            z = zp.stat().st_size; total_zip += z
            print(f"  {zp.name}: {human(raw)} raw -> {human(z)} zipped ({raw/z:.1f}x)")

    if total_raw:
        print(f"\nTOTAL: {human(total_raw)} raw -> {human(total_zip)} zipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
