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

With --instance-info-commit, instance_info.json (write_instance_info.py) and MANIFEST.csv are
written into the staged tree BEFORE zipping; --docs copies README, licences etc. into the root of
every archive, parsed and unparsed. --family small archives experiment_data_V2_small_scaling
(no capacity sweep) and its unparsed root instead:

    python3 build_release_zips.py --data-root <cluster data> --out-dir instances --staging <scratch> \
        --granularities 60 --instance-info-commit <sha> --docs README.md LICENSE-DATA.txt ...
    python3 build_release_zips.py --data-root <cluster data> --out-dir generation_artefacts \
        --skip-parsed --unparsed per-granularity --docs README.md ...
    python3 build_release_zips.py --family small --data-root <cluster data> --out-dir ... \
        --instance-info-commit <sha> --unparsed per-granularity --docs ...
"""
import argparse, csv, os, shutil, subprocess, sys, zipfile
from pathlib import Path

import write_instance_info as wii

LEVELS = [f"PCAP{p:03d}" for p in range(10, 101, 10)]
SMALL_ROOT = "experiment_data_V2_small_scaling"
MANIFEST_COLUMNS = {
    "large": ["path", "region", "experiment", "time_granularity", "timesteps_per_day",
              "minutes_per_timestep", "flights", "seed", "capacity_percent", "nominal_capacity",
              "sector_capacity_per_timestep", "licence"],
    "small": ["path", "region", "time_granularity", "timesteps_per_day", "minutes_per_timestep",
              "flights", "seed", "enroute_capacity_per_timestep"],
}


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}"
        n /= 1024


def tree_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def strip_cr(root: Path) -> int:
    """Rewrite CRLF as LF in the STAGED tree (never in the source). Returns the file count.

    A generated CSV is written by a single writer, so a CR in the header implies CRLF throughout;
    probing the first 8 KiB keeps this from re-reading gigabytes. The release verifier rescans
    every archive entry for CR afterwards, so a miss here cannot reach the upload unnoticed.
    """
    n = 0
    for f in root.rglob("*"):
        if f.is_symlink() or not f.is_file():
            continue
        with open(f, "rb") as fh:
            if b"\r" not in fh.read(8192):
                continue
        st = f.stat()
        body = f.read_bytes().replace(b"\r\n", b"\n")
        if b"\r" in body:
            raise SystemExit(f"{f}: carriage return that is not part of a CRLF; not normalising")
        f.write_bytes(body)
        os.utime(f, (st.st_atime, st.st_mtime))
        n += 1
    return n


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
        if not dry and (ov / "nominal_capacities.csv").exists():
            (stage / "nominal_capacities").mkdir(parents=True, exist_ok=True)
            shutil.copy2(ov / "nominal_capacities.csv", stage / "nominal_capacities" / f"{e.name}.csv")
    return stage


def finish_stage(stage: Path, family: str, commit: str | None, docs: list[Path]) -> None:
    """Write instance_info.json + MANIFEST.csv into the staged tree and copy the documents."""
    if commit:
        rows = []
        for inst, experiment, pcap in wii.walk(stage):
            info = wii.build(inst, experiment, pcap, commit)
            (inst / "instance_info.json").write_text(wii.serialise(info))
            rows.append({
                "path": inst.relative_to(stage.parent).as_posix(), "region": info["region"],
                "experiment": experiment, "time_granularity": info["time_granularity_bins_per_hour"],
                "timesteps_per_day": info["timesteps_per_day"],
                "minutes_per_timestep": f"{info['minutes_per_timestep']:g}",
                "flights": info["flights"], "seed": info["seed"],
                "capacity_percent": info.get("capacity_level_percent_of_nominal"),
                "nominal_capacity": info.get("nominal_capacity"),
                "sector_capacity_per_timestep": info.get("sector_capacity_per_timestep"),
                "enroute_capacity_per_timestep": info.get("enroute_capacity_per_timestep"),
                "licence": info["licence"].split()[0].rstrip(";"),   # SPDX id
            })
        rows.sort(key=lambda r: r["path"])
        with open(stage / "MANIFEST.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS[family], extrasaction="ignore",
                               lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {len(rows)} instance_info.json + MANIFEST.csv (generator_commit {commit})")
    for d in docs:
        shutil.copy2(d, stage / d.name)


def zip_unparsed(root: Path, zp: Path, staging: Path, docs: list[Path], lf: bool = False) -> None:
    """Zip an unparsed root; with docs, zip a view of symlinks + documents instead of the root.

    zip follows symlinks unless given -y, so the archive stores the real files and the source
    tree is never written to. With lf the view holds real copies instead, because line endings
    are normalised in it and the source must stay untouched.
    """
    zp.unlink(missing_ok=True)
    if not docs and not lf:
        subprocess.run(["zip", "-r", "-q", "-1", str(zp.resolve()), root.name],
                       cwd=root.parent, check=True)
        return
    view = staging / root.name
    if view.is_symlink() or view.exists():
        raise SystemExit(f"refusing to reuse existing staging path {view}")
    view.mkdir(parents=True)
    try:
        for child in sorted(root.iterdir()):
            if lf and child.is_dir():
                shutil.copytree(child, view / child.name)
            elif lf:
                shutil.copy2(child, view / child.name)
            else:
                (view / child.name).symlink_to(child.resolve())
        if lf:
            print(f"  LF: normalised {strip_cr(view)} files in the staged copy", flush=True)
        for d in docs:
            shutil.copy2(d, view / d.name)
        subprocess.run(["zip", "-r", "-q", "-1", str(zp.resolve()), view.name],
                       cwd=staging, check=True)
    finally:
        for p in view.iterdir():          # unlink the links themselves, never their targets
            if p.is_symlink():
                p.unlink()
        shutil.rmtree(view)


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
    ap.add_argument("--family", choices=["large", "small"], default="large",
                    help="large: materialised TG archives; small: experiment_data_V2_small_scaling")
    ap.add_argument("--instance-info-commit", default=None,
                    help="write instance_info.json and MANIFEST.csv into the staged parsed tree "
                         "before zipping, recording this generator commit")
    ap.add_argument("--docs", type=Path, nargs="*", default=[],
                    help="files copied into the root of every archive (README.md, licences, ...)")
    ap.add_argument("--lf", action="store_true",
                    help="rewrite CRLF as LF in the staged tree before zipping (the source tree "
                         "is never written to). v2.0.0 of the generator wrote CRLF in the sweep's "
                         "sectors.csv/nominal_capacities.csv and in the unparsed navgraph/edges.csv "
                         "and DATA_*/aircrafts.csv; 2d69f1f writes LF everywhere.")
    a = ap.parse_args()

    data_root = a.data_root.expanduser()
    out_dir = a.out_dir.expanduser(); out_dir.mkdir(parents=True, exist_ok=True)
    staging = (a.staging or out_dir / "_staging").expanduser(); staging.mkdir(parents=True, exist_ok=True)
    docs = [d.expanduser() for d in a.docs]
    for d in docs:
        if not d.is_file():
            raise SystemExit(f"--docs: not a file: {d}")

    if a.lf and a.unparsed == "single":
        raise SystemExit("--lf is not implemented for --unparsed single (it zips the source root)")

    gran = [int(x) for x in a.granularities.split(",") if x.strip()]
    total_raw = total_zip = 0
    for tg in ([] if a.skip_parsed else (gran if a.family == "large" else [None])):
        if a.family == "large":
            stage = build_one(data_root, tg, staging, a.dry_run)
        else:
            stage = staging / SMALL_ROOT
            if stage.exists():
                shutil.rmtree(stage)
            if not a.dry_run:
                shutil.copytree(data_root / SMALL_ROOT, stage)
        if stage is None or a.dry_run:
            continue
        if a.lf:
            print(f"  LF: normalised {strip_cr(stage)} files in the staged tree", flush=True)
        finish_stage(stage, a.family, a.instance_info_commit, docs)
        raw = tree_size(stage); total_raw += raw
        zpath = out_dir / f"{stage.name}.zip"
        zpath.unlink(missing_ok=True)     # never update a stale archive in place
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
        if a.family == "small":
            roots = [data_root / f"unparsed_{SMALL_ROOT}"]
        else:
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
                zip_unparsed(r, zp, staging, docs, a.lf)
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
