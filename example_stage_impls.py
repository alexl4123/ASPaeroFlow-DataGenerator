#!/usr/bin/env python3
"""
Worked examples of alternative stage implementations.

Nothing in the pipeline imports this file. It exists to be pointed at:

    # 1. delegate to the shipped stage 03, announcing itself. Output is identical.
    python run_pipeline.py --config default_configs_small_scaling/30_0_east_asia_3x3.json \\
        --out-root /tmp/demo --stage-impl sectors=example_stage_impls:LoggingSectors

    # 2. really replace it: one flat capacity for every sector, en-route and airport alike.
    python run_pipeline.py --config default_configs_small_scaling/30_0_east_asia_3x3.json \\
        --out-root /tmp/demo2 --stage-impl sectors=example_stage_impls:FlatCapacitySectors

    # 3. the transform stage, which is selected on its own entry point.
    python 05_transform_for_optimizer.py --in-exp-dir /tmp/demo/<REGION> --out-root /tmp/parsed \\
        --stage-impl example_stage_impls:LoggingTransform

    # 4. a stage 03 that does not call the default at all: it clusters by
    #    geography instead of by graph connectivity.
    python run_pipeline.py --config default_configs_small_scaling/30_0_east_asia_3x3.json \\
        --out-root /tmp/demo3 --stage-impl sectors=example_stage_impls:LatitudeBandSectors

Write your own the same way: subclass the parent class for the stage (see
``stage_interfaces.py`` — its docstring lists the files you must write and the
invariants you must respect), implement ``start(self, argv)``, and name your
class as ``module:ClassName`` or ``path/to/file.py:ClassName``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from stage_interfaces import (
    DefaultSectorCapacity,
    DefaultTransform,
    SectorCapacityStage,
    TransformStage,
    argv_to_dict,
)


class LoggingSectors(SectorCapacityStage):
    """The smallest possible alternative: announce yourself, then call the default.

    Useful as a template and as a check that ``--stage-impl`` is wired up — the
    output is byte-identical to a run without it.
    """

    def start(self, argv: Sequence[str]) -> None:
        opts = argv_to_dict(argv)
        print(f"[example] LoggingSectors: navgraph={opts.get('path')} "
              f"cap-enroute={opts.get('cap-enroute')} cap-airport={opts.get('cap-airport')}")
        DefaultSectorCapacity().start(argv)


class FlatCapacitySectors(SectorCapacityStage):
    """A real substitution: give every sector the same capacity.

    The shipped stage 03 gives airport sectors ``--cap-airport`` (effectively
    unlimited) and en-route sectors ``--cap-enroute``. This one runs the default
    to get the clustering, then overwrites every capacity with ``--cap-enroute``,
    so airports become as constrained as the airspace.

    It shows the pattern that most alternative implementations want: keep the
    default's structure, change one rule. The invariants of
    ``SectorCapacityStage`` still hold — ``sectors.csv`` keeps its columns, and
    capacity stays >= 1, which check ``P4`` verifies.
    """

    def start(self, argv: Sequence[str]) -> None:
        DefaultSectorCapacity().start(argv)

        opts = argv_to_dict(argv)
        nav_dir = Path(str(opts["path"]))
        cap = max(1, int(str(opts.get("cap-enroute", 1))))

        path = nav_dir / "sectors.csv"
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))
        header, body = rows[0], rows[1:]
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for row in body:
                w.writerow([row[0], cap])
        print(f"[example] FlatCapacitySectors: set {len(body)} sectors to Capacity={cap}")


class LatitudeBandSectors(SectorCapacityStage):
    """A stage 03 written against the interface, not derived from the shipped one.

    ``LoggingSectors`` and ``FlatCapacitySectors`` both call
    ``DefaultSectorCapacity`` and adjust what it produced. This one does not run
    the shipped stage at all — it reads the navgraph and writes both artefacts
    itself, which is what a third party replacing stage 03 actually has to do.

    The rule it implements
    ----------------------
    The shipped stage grows en-route sectors by breadth-first search over
    ``edges.csv``, so a sector is a connected subgraph. This one ignores the
    edges entirely and clusters by **geography**: sort the en-route vertices
    south to north (ties broken west to east, then by identifier) and cut the
    order into contiguous runs of ``--sector-default-navaid-size``. Each run is
    one latitude band. Airport vertices get a singleton sector each, as in any
    sensible capacity model — an airport's capacity is its own.

    Bands are latitude strips, hence convex in lat/lon by construction, so
    ``--convex-sectors`` needs no separate code path here; the flag is accepted
    and has no effect. ``edges.csv`` is never opened.

    What it takes from ``SectorCapacityStage``
    ------------------------------------------
    Everything, and nothing from ``03_sector_capacity_generator.py``:

    * writes ``sectors.csv`` (``Sector_ID, Capacity``) and
      ``navaid_sector_assignment.csv`` (``Navaid_ID, Sector_ID``) into
      ``--path``, the directory it read;
    * ``--cap-airport`` for airport sectors, ``--cap-enroute`` for en-route
      ones, both floored at 1 so check ``P4`` cannot fail;
    * every vertex of ``vertices.csv`` appears exactly once as ``Navaid_ID``
      (checks ``P5``/``D6``);
    * follows convention ② — ``sectors.csv`` is keyed by **navaid**, one row per
      graph vertex, because stage 05 maps its ``Sector_ID`` column through the
      vertex table. The docstring says a per-cluster ``sectors.csv`` is legal
      too, but it would not survive this pipeline's stage 05.

    Airports are identified from the ``IS_AIRPORT`` column that
    ``NavigationGraphStage`` promises in ``vertices.csv``, rather than by
    re-deriving them from OurAirports. Taking the navgraph at its word is the
    cheaper contract to rely on, and it cannot disagree with the graph.
    """

    def start(self, argv: Sequence[str]) -> None:
        opts = argv_to_dict(argv)
        nav_dir = Path(str(opts["path"]))
        band_size = max(1, int(str(opts.get("sector-default-navaid-size", 10))))
        cap_enroute = max(1, int(str(opts.get("cap-enroute", 60))))
        cap_airport = max(1, int(str(opts.get("cap-airport", 60000))))

        with open(nav_dir / "vertices.csv", newline="") as fh:
            vertices = list(csv.DictReader(fh))
        if not vertices:
            raise RuntimeError(f"no vertices to sectorise in {nav_dir}")
        for col in ("IDENTIFIER", "LAT", "LON", "IS_AIRPORT"):
            if col not in vertices[0]:
                raise ValueError(f"vertices.csv is missing the '{col}' column")

        def is_airport(row) -> bool:
            return str(row["IS_AIRPORT"]).strip().lower() in ("1", "true", "yes", "t")

        # south -> north, ties west -> east, then by identifier so the order is
        # total and does not depend on the order vertices.csv happened to use.
        enroute = sorted(
            ((float(r["LAT"]), float(r["LON"]), str(r["IDENTIFIER"]).strip().upper())
             for r in vertices if not is_airport(r)),
        )

        sector_of = {ident: f"BAND_{i // band_size:06d}"
                     for i, (_lat, _lon, ident) in enumerate(enroute)}
        for row in vertices:
            if is_airport(row):
                ident = str(row["IDENTIFIER"]).strip().upper()
                sector_of[ident] = f"BAND_AIRPORT_{ident}"

        assign_path = nav_dir / "navaid_sector_assignment.csv"
        caps_path = nav_dir / "sectors.csv"
        # pandas writes "\n"; csv.writer defaults to "\r\n", so pin it to match
        # what the rest of the pipeline produces.
        with open(assign_path, "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["Navaid_ID", "Sector_ID"])
            for row in vertices:
                ident = str(row["IDENTIFIER"]).strip().upper()
                w.writerow([ident, sector_of[ident]])

        with open(caps_path, "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["Sector_ID", "Capacity"])
            for row in vertices:
                ident = str(row["IDENTIFIER"]).strip().upper()
                w.writerow([ident, cap_airport if is_airport(row) else cap_enroute])

        n_bands = len({s for s in sector_of.values() if s.startswith("BAND_0")})
        print(f"[example] LatitudeBandSectors: {len(vertices)} vertices -> "
              f"{n_bands} latitude band(s) of <= {band_size} navaids + "
              f"{len(vertices) - len(enroute)} airport sector(s); "
              f"cap-enroute={cap_enroute} cap-airport={cap_airport}")


class LoggingTransform(TransformStage):
    """The same template for stage 05, which carries its own ``--stage-impl``."""

    def start(self, argv: Sequence[str]) -> None:
        opts = argv_to_dict(argv)
        print(f"[example] LoggingTransform: in={opts.get('in-exp-dir')} out={opts.get('out-root')}")
        DefaultTransform().start(argv)
