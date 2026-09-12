#!/usr/bin/env python3
"""
fingerprint.py -- deterministic content manifest for a generator output tree.

Walks an output tree produced by run_pipeline.py (and/or 05_transform_for_optimizer.py)
and writes a JSON manifest mapping relative path -> content hashes plus enough CSV
structure that a later diff can say WHAT changed, not merely THAT something changed.

Per file the manifest records:

  sha256          SHA-256 of the file after normalisation (see NORMALISATION below)
  bytes           size of the normalised content
  kind            "csv" | "json" | "other"

and, additionally, for every parseable CSV:

  rows            number of data rows (header excluded)
  cols            number of columns
  columns         the column names, in file order
  sha256_sorted   SHA-256 over the header plus the data rows sorted as raw text.
                  Equal sha256_sorted with unequal sha256 means the file was
                  re-ordered but its multiset of rows is unchanged.
  column_sha256   per-column SHA-256, taken over the column's values in the
                  canonical (row-sorted) order, so compare.py can name the
                  columns that actually moved.
  sample_rows     the first two data rows, truncated -- purely so a failure
                  report can show something concrete when the generated tree
                  that produced the baseline is no longer on disk.

NORMALISATION (the only things deliberately ignored)
----------------------------------------------------
1. Absolute filesystem paths inside JSON *string values* are replaced by
   "<ABS>/<basename>".  The generator records its own --out-root, --csv-path and
   per-dataset directories in run_config.json / manifest.json /
   transform_manifest.json.  Those strings change whenever the tree is written to a
   different directory, which every regression run does by construction; they are not
   generator output.  The basename is kept so that a change of *which file* was read
   (e.g. a different flight list) is still caught.  Relative paths -- such as the
   "config" and "ourairports_path" values -- are NOT touched and are compared exactly.
2. Line endings are normalised to "\n".

Nothing else is excluded.  In particular no numeric field, no row ordering, no
"stats" block (num_vertices / num_edges in navgraph/run_config.json) and no CSV
column is normalised away.  The generator writes no generation timestamp into any
artefact, so there is no timestamp exclusion to declare.

Stdlib + pandas only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd

MANIFEST_VERSION = 1
SAMPLE_ROW_CHARS = 200
SAMPLE_ROW_COUNT = 2

NORMALISATION_NOTES = [
    "json:absolute-path-values -> '<ABS>/<basename>' (out-root, csv-path and "
    "per-dataset directories are run-location, not generator output)",
    "all files: CRLF/CR line endings -> LF",
]


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

def _looks_absolute(value: str) -> bool:
    return value.startswith("/") and len(value) > 1 and "/" in value[1:]


def _redact_paths(node):
    """Recursively replace absolute-path string values with '<ABS>/<basename>'."""
    if isinstance(node, dict):
        return {k: _redact_paths(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_redact_paths(v) for v in node]
    if isinstance(node, str) and _looks_absolute(node):
        return "<ABS>/" + node.rstrip("/").rsplit("/", 1)[-1]
    return node


def _normalise_bytes(path: Path, raw: bytes) -> tuple[bytes, str]:
    """Return (normalised_bytes, kind)."""
    if path.suffix.lower() == ".json":
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _normalise_text(raw), "other"
        canonical = json.dumps(
            _redact_paths(parsed), indent=2, sort_keys=True, ensure_ascii=False
        )
        return canonical.encode("utf-8"), "json"

    normalised = _normalise_text(raw)
    kind = "csv" if path.suffix.lower() == ".csv" else "other"
    return normalised, kind


def _normalise_text(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# CSV structure
# --------------------------------------------------------------------------

def _csv_details(normalised: bytes) -> dict | None:
    """Row/column counts, sorted-content hash and per-column hashes, or None."""
    text = normalised.decode("utf-8", errors="replace")
    lines = text.split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    if not lines:
        return {
            "rows": 0,
            "cols": 0,
            "columns": [],
            "sha256_sorted": _sha(b""),
            "column_sha256": {},
            "sample_rows": [],
        }

    header, data_lines = lines[0], lines[1:]

    # Row-order-insensitive hash: header verbatim, data rows sorted as raw text.
    sorted_blob = "\n".join([header] + sorted(data_lines)).encode("utf-8")

    details = {
        "rows": len(data_lines),
        "sha256_sorted": _sha(sorted_blob),
        "sample_rows": [ln[:SAMPLE_ROW_CHARS] for ln in data_lines[:SAMPLE_ROW_COUNT]],
    }

    try:
        frame = pd.read_csv(
            io.StringIO(text), dtype=str, keep_default_na=False, na_filter=False
        )
    except Exception:
        # Unparseable as a table; the whole-file hashes above still apply.
        details.update({"cols": None, "columns": None, "column_sha256": None})
        return details

    columns = [str(c) for c in frame.columns]
    details["cols"] = len(columns)
    details["columns"] = columns

    if len(frame) and columns:
        canonical = frame.sort_values(by=columns, kind="stable")
    else:
        canonical = frame

    details["column_sha256"] = {
        col: _sha("\n".join(canonical[col].astype(str).tolist()).encode("utf-8"))
        for col in columns
    }
    return details


# --------------------------------------------------------------------------
# walk
# --------------------------------------------------------------------------

def fingerprint_tree(root: Path, skip_names: set[str]) -> dict:
    files: dict[str, dict] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name in skip_names:
                continue
            abs_path = Path(dirpath) / name
            rel = abs_path.relative_to(root).as_posix()
            try:
                raw = abs_path.read_bytes()
            except OSError as exc:
                files[rel] = {"error": f"unreadable: {exc}"}
                continue
            normalised, kind = _normalise_bytes(abs_path, raw)
            entry = {
                "sha256": _sha(normalised),
                "bytes": len(normalised),
                "kind": kind,
            }
            if kind == "csv":
                detail = _csv_details(normalised)
                if detail:
                    entry.update(detail)
            files[rel] = entry
    return files


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Emit a deterministic content manifest for a generator output tree."
    )
    ap.add_argument("--root", type=Path, required=True,
                    help="Directory to fingerprint.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Manifest JSON to write ('-' for stdout).")
    ap.add_argument("--label", type=str, default=None,
                    help="Fixture name recorded in the manifest.")
    ap.add_argument("--skip-name", action="append", default=[],
                    help="Basename to exclude (repeatable).")
    args = ap.parse_args()

    if not args.root.is_dir():
        print(f"[fingerprint] not a directory: {args.root}", file=sys.stderr)
        return 2

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "label": args.label or args.root.name,
        "normalisation": NORMALISATION_NOTES,
        # Recorded because stage 04 iterates a Python set (see tests/regression/README.md);
        # a baseline and a candidate taken under different hash seeds are not comparable
        # byte-for-byte.  compare.py warns when these disagree.
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "python": "%d.%d" % sys.version_info[:2],
            "pandas": pd.__version__,
        },
        "files": fingerprint_tree(args.root, set(args.skip_name)),
    }

    blob = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if str(args.out) == "-":
        sys.stdout.write(blob)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(blob, encoding="utf-8")
        print(f"[fingerprint] {len(manifest['files'])} files -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
