#!/usr/bin/env python3
"""
compare.py -- diff two manifests written by fingerprint.py.

    python tests/regression/compare.py BASELINE.json CANDIDATE.json

Exit status
    0   the candidate matches the baseline
    1   the candidate differs (files added, removed or changed)
    2   usage / IO error

Every changed CSV is classified:

    CONTENT     the multiset of rows changed.  The columns whose hashes moved are
                named; columns that did not move are named too, so a refactor that
                perturbs one field is immediately localised.
    ORDER-ONLY  the rows are identical as a multiset, only their order in the file
                changed.  This is a REAL difference and fails by default.  It can be
                downgraded to a warning with --allow-row-reorder, which prints a
                loud banner saying the check ran weakened and why.
    SHAPE       the row or column count changed.

If the trees themselves are still on disk, pass --baseline-root / --candidate-root
and the report will quote actual differing rows rather than the two sample rows the
manifest carries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MAX_EXAMPLES = 4


def load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"[compare] cannot read {path}: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except json.JSONDecodeError as exc:
        print(f"[compare] {path} is not valid JSON: {exc}", file=sys.stderr)
        raise SystemExit(2)
    if "files" not in data:
        print(f"[compare] {path} is not a fingerprint manifest", file=sys.stderr)
        raise SystemExit(2)
    return data


def read_rows(root: Path | None, rel: str) -> list[str] | None:
    if root is None:
        return None
    path = root / rel
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    return lines[1:] if lines else []


def example_rows(base_root, cand_root, rel, base_entry, cand_entry) -> list[str]:
    """A few concrete rows illustrating the change."""
    base_rows = read_rows(base_root, rel)
    cand_rows = read_rows(cand_root, rel)

    if base_rows is not None and cand_rows is not None:
        base_set, cand_set = set(base_rows), set(cand_rows)
        only_base = [r for r in base_rows if r not in cand_set][:MAX_EXAMPLES]
        only_cand = [r for r in cand_rows if r not in base_set][:MAX_EXAMPLES]
        out = []
        for row in only_base:
            out.append(f"    - baseline only : {row[:200]}")
        for row in only_cand:
            out.append(f"    + candidate only: {row[:200]}")
        if not out:
            # Same multiset of rows: show the first position where order diverges.
            for idx, (b, c) in enumerate(zip(base_rows, cand_rows)):
                if b != c:
                    out.append(f"    first divergence at data row {idx + 1}:")
                    out.append(f"    - baseline : {b[:200]}")
                    out.append(f"    + candidate: {c[:200]}")
                    break
        return out

    out = []
    for row in base_entry.get("sample_rows") or []:
        out.append(f"    - baseline sample : {row}")
    for row in cand_entry.get("sample_rows") or []:
        out.append(f"    + candidate sample: {row}")
    if out:
        out.append("    (manifest samples only; pass --baseline-root/--candidate-root "
                   "for real differing rows)")
    return out


def describe_change(rel, base, cand, base_root, cand_root) -> tuple[str, list[str]]:
    """Return (classification, report lines)."""
    lines: list[str] = []

    if base.get("kind") != "csv" or cand.get("kind") != "csv":
        lines.append(f"    bytes {base.get('bytes')} -> {cand.get('bytes')}")
        return "CONTENT", lines

    b_rows, c_rows = base.get("rows"), cand.get("rows")
    b_cols, c_cols = base.get("columns"), cand.get("columns")

    if b_rows != c_rows:
        lines.append(f"    row count {b_rows} -> {c_rows}")
    if b_cols != c_cols:
        lines.append(f"    columns {b_cols} -> {c_cols}")
        return "SHAPE", lines
    if b_rows != c_rows:
        lines.extend(example_rows(base_root, cand_root, rel, base, cand))
        return "SHAPE", lines

    same_multiset = (
        base.get("sha256_sorted") is not None
        and base.get("sha256_sorted") == cand.get("sha256_sorted")
    )

    b_colhash = base.get("column_sha256") or {}
    c_colhash = cand.get("column_sha256") or {}
    moved = [c for c in (b_cols or []) if b_colhash.get(c) != c_colhash.get(c)]
    stable = [c for c in (b_cols or []) if c not in moved]

    if same_multiset:
        lines.append(f"    rows identical as a multiset ({b_rows} rows); only the "
                     f"order in the file changed")
        lines.extend(example_rows(base_root, cand_root, rel, base, cand))
        return "ORDER-ONLY", lines

    if moved:
        lines.append(f"    columns that changed : {', '.join(moved)}")
    if stable:
        lines.append(f"    columns unchanged    : {', '.join(stable)}")
    if not moved and not stable:
        lines.append("    (column-level detail unavailable; file was not parseable "
                     "as a table)")
    lines.extend(example_rows(base_root, cand_root, rel, base, cand))
    return "CONTENT", lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Diff two fingerprint.py manifests.")
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--baseline-root", type=Path, default=None,
                    help="Baseline output tree, if still on disk, for real example rows.")
    ap.add_argument("--candidate-root", type=Path, default=None,
                    help="Candidate output tree, for real example rows.")
    ap.add_argument("--allow-row-reorder", action="store_true",
                    help="WEAKENS THE CHECK: treat row-order-only differences as "
                         "warnings instead of failures.")
    ap.add_argument("--quiet", action="store_true",
                    help="Only print the summary and any differences.")
    args = ap.parse_args()

    base = load(args.baseline)
    cand = load(args.candidate)

    print("=" * 78)
    print(f"regression compare: {base.get('label')}")
    print(f"  baseline : {args.baseline}")
    print(f"  candidate: {args.candidate}")
    print("=" * 78)

    if args.allow_row_reorder:
        print()
        print("!!! WEAKENED CHECK ACTIVE -- --allow-row-reorder was passed.")
        print("!!! Files whose rows are identical as a multiset but appear in a")
        print("!!! different order are reported as warnings and DO NOT fail this")
        print("!!! run.  This hides exactly the class of difference that stage 04's")
        print("!!! `for aircraft in list(set(flights[\"aircraft_id\"]))` produces")
        print("!!! (04_simplified_filed_flight_plan_generator.py:613).  Only use it")
        print("!!! when you have decided that ordering is not part of the contract.")
        print()

    b_env = base.get("env") or {}
    c_env = cand.get("env") or {}
    for key in ("PYTHONHASHSEED", "python", "pandas"):
        if b_env.get(key) != c_env.get(key):
            print(f"[warn] {key} differs: baseline={b_env.get(key)!r} "
                  f"candidate={c_env.get(key)!r}")
    if b_env.get("PYTHONHASHSEED") != c_env.get("PYTHONHASHSEED"):
        print("[warn] -> aircrafts.csv row order is NOT stable across different "
              "hash seeds; differences below may be an artefact of that, not of "
              "the code change under test.  Re-run with PYTHONHASHSEED pinned.")

    b_files, c_files = base["files"], cand["files"]
    b_names, c_names = set(b_files), set(c_files)

    removed = sorted(b_names - c_names)
    added = sorted(c_names - b_names)
    common = sorted(b_names & c_names)

    changed: list[tuple[str, str, list[str]]] = []
    for rel in common:
        if b_files[rel].get("sha256") != c_files[rel].get("sha256"):
            kind, lines = describe_change(
                rel, b_files[rel], c_files[rel], args.baseline_root, args.candidate_root
            )
            changed.append((rel, kind, lines))

    if removed:
        print(f"\n--- REMOVED ({len(removed)}) " + "-" * 40)
        for rel in removed:
            print(f"  - {rel}")
    if added:
        print(f"\n--- ADDED ({len(added)}) " + "-" * 42)
        for rel in added:
            print(f"  + {rel}")
    if changed:
        print(f"\n--- CHANGED ({len(changed)}) " + "-" * 40)
        for rel, kind, lines in changed:
            print(f"  ~ [{kind}] {rel}")
            for line in lines:
                print(line)

    order_only = [c for c in changed if c[1] == "ORDER-ONLY"]
    real = [c for c in changed if c[1] != "ORDER-ONLY"]

    print("\n" + "=" * 78)
    print(f"files compared : {len(common)}")
    print(f"added          : {len(added)}")
    print(f"removed        : {len(removed)}")
    print(f"changed        : {len(changed)}  "
          f"({len(real)} content/shape, {len(order_only)} row-order-only)")

    failed = bool(added or removed or real)
    if order_only:
        if args.allow_row_reorder:
            print(f"WARNING: {len(order_only)} file(s) differ by row order only and "
                  f"were NOT counted as failures because --allow-row-reorder was "
                  f"passed.  THIS RUN USED A WEAKENED CHECK.")
        else:
            failed = True

    if failed:
        print("RESULT: MISMATCH")
        print("=" * 78)
        return 1

    print("RESULT: MATCH")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
