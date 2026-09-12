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


class LoggingTransform(TransformStage):
    """The same template for stage 05, which carries its own ``--stage-impl``."""

    def start(self, argv: Sequence[str]) -> None:
        opts = argv_to_dict(argv)
        print(f"[example] LoggingTransform: in={opts.get('in-exp-dir')} out={opts.get('out-root')}")
        DefaultTransform().start(argv)
