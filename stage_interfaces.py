#!/usr/bin/env python3
"""
Swappable stage implementations.

Every stage of the pipeline is described here by a parent class with exactly one
required method, ``start(argv)``. The shipped scripts are registered as the
``default`` implementation of their stage, so nothing changes unless a different
implementation is asked for. To substitute your own trajectory generator, graph
builder, capacity rule, … you subclass the stage's parent class, implement
``start``, and name the class on the command line:

    python run_pipeline.py --config <cfg> --stage-impl navgraph=my_graph:MyGraphStage
    python 05_transform_for_optimizer.py --in-exp-dir <dir> --stage-impl my_tf:MyTransform

``SPEC`` is either a registered name (``default``), ``module:ClassName`` for an
importable module, or ``path/to/file.py:ClassName`` for a file. The class must
subclass the parent class of the stage it is registered for; anything else is
rejected with an error naming the interface it had to implement.

What an implementation receives
-------------------------------
``start`` is handed the *canonical argv* for its stage: the exact argument list
the default implementation would pass to its script, with every value already
resolved from the config file and the CLI. It is a list of strings. Flags that
the pipeline only passes conditionally (``--config``, ``--date-start``, …) are
absent when they were not set. Use :func:`argv_to_dict` if you would rather have
a mapping than a list.

Two rules bind every implementation:

* **Write the files your stage's docstring names, with the columns it names.**
  The next stage reads them positionally and by column name; nothing validates
  them for you until ``check_instances.py`` runs at the end.
* **Write them under the directory passed in argv** (``--out-dir``, ``--path``
  or ``--data-dir``, depending on the stage). The pipeline chooses those paths
  and looks for the artefacts there to decide whether a stage can be skipped.

An implementation may shell out, import a library, or call the default and
post-process it — see ``example_stage_impls.py`` for both of the last two.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Sequence, Type

# Stage keys, in pipeline order. These are the names accepted by --stage-impl.
STAGE_KEYS = ("model", "flights", "navgraph", "sectors", "filedplans", "transform")


# --------------------------------------------------------------------------
# parent interfaces
# --------------------------------------------------------------------------

class GeneratorStage(ABC):
    """Parent of every stage interface.

    Subclass one of the six stage classes below, not this one: the stage class
    is what ``--stage-impl`` checks against, and its docstring is the contract.
    """

    #: which stage this interface describes; set on each subclass
    stage_key: str = ""

    @abstractmethod
    def start(self, argv: Sequence[str]) -> None:
        """Run the stage. Raise on failure; the pipeline judges by the exception.

        :param argv: the canonical argument list for this stage (see module
            docstring). Values are strings, already resolved.
        """
        raise NotImplementedError


class DemandModelStage(GeneratorStage):
    """Stage 00 — fit the demand model from a real flight list.

    Reads the OpenSky flight list (``--csv-path``) and the OurAirports snapshot
    (``--ourairports-path``), localised by ``--timezone`` and restricted to
    ``--date-start`` … ``--date-end`` (or ``--target-day``), to the airports
    named by ``--airport-include`` / ``--airport-types``, and to the regions in
    ``--config``.

    Must write, into ``--out-dir``:

    ==========================  =====================================
    file                        columns
    ==========================  =====================================
    ``airport_bins.csv``        ``origin, bin, rate``
    ``od_time_model.csv``       ``origin, bin, destination, prob``
    ``od_dur_dist.csv``         ``origin, destination, duration_min, speed_kts``
    ``tat_dist.csv``            ``tat_min``
    ``global_dest_freq.csv``    ``destination, freq``
    ==========================  =====================================

    Invariants:

    * ``bin`` indexes bins of ``--bin-min`` minutes over one local day, so
      ``bin`` < 1440 / ``--bin-min``. ``rate`` is departures per bin.
    * ``prob`` sums to 1 over ``destination`` within each ``(origin, bin)``.
    * ``speed_kts`` is the speed stage 01 will attach to the flight; stage 04
      recomputes duration from it, so it must be > 0.
    * The airport set written here must be a subset of the vertices stage 02
      marks ``IS_AIRPORT``, or flights will be sampled from airports that have
      no vertex to depart from.
    * All five files must be non-empty: ``run_pipeline`` treats their presence
      as proof the stage ran and skips it on a re-run.
    """

    stage_key = "model"


class FlightScheduleStage(GeneratorStage):
    """Stage 01 — sample a flight schedule from the fitted model.

    Reads the model directory (``--model-dir``) and draws either ``--flights``
    flights or a ``--scale`` multiple of the fitted demand, for the day
    ``--day``, in the frame given by ``--timezone``, under ``--seed``.

    Must write, into ``--out-dir``:

    ==========================  =====================================
    file                        columns
    ==========================  =====================================
    ``flights.csv``             ``flight_id, aircraft_id, origin, destination, departure_time``
    ``aircrafts.csv``           ``aircraft_id, speed_kts``
    ``run_config.json``         at least ``number_flights`` and ``seed``
    ==========================  =====================================

    Invariants:

    * ``origin`` != ``destination``: a self-loop is unrepresentable in the
      point-to-point encoding downstream (check ``C4``).
    * ``origin`` and ``destination`` are ICAO codes present in the model's
      airport set and on the navgraph.
    * every ``aircraft_id`` in ``flights.csv`` appears in ``aircrafts.csv``.
    * ``departure_time`` is an ISO timestamp carrying the ``--timezone`` offset.
    * Stage 04 **rewrites both files in place** (README convention ④). Extra
      columns are tolerated: the shipped stage 04 adds ``src``, ``dst``,
      ``start_slot``, ``speed_kts``.
    """

    stage_key = "flights"


class NavigationGraphStage(GeneratorStage):
    """Stage 02 — build the navigation graph.

    Reads X-Plane ``fix.dat`` / ``nav.dat`` from ``--navdir``, or synthesises a
    ``--grid-nx`` × ``--grid-ny`` grid when ``--grid-navpoints true``, plus the
    airports from ``--ourairports``, and joins vertices under ``--criterion``
    subject to ``--max-edge-km`` and ``--min-dist-vertices-km``.

    Must write, into ``--out-dir``:

    ==========================  =====================================
    file                        columns
    ==========================  =====================================
    ``vertices.csv``            ``IDENTIFIER, LAT, LON, ALTITUDE, IS_AIRPORT``
    ``edges.csv``               ``V0, V1, D``
    ==========================  =====================================

    Invariants:

    * ``V0``/``V1`` are ``IDENTIFIER`` values from ``vertices.csv``; ``D`` is the
      edge length in **metres**. Stage 04 divides it by the aircraft speed to
      cost a traversal, so ``D`` > 0.
    * Edges are undirected and listed once. Stage 04 and the checker both treat
      ``(V0, V1)`` and ``(V1, V0)`` as the same edge.
    * Every airport in the model's airport set is a vertex with ``IS_AIRPORT``
      = 1, and it must be reachable: with ``--enforce-connected true`` the graph
      is a single connected component, and stage 04 raises if it cannot route a
      sampled origin–destination pair.
    * ``IDENTIFIER`` is unique. Stage 05 turns it into the integer vertex id
      that the solver-facing files use, via ``mappings/vertex_map.csv``.
    """

    stage_key = "navgraph"


class SectorCapacityStage(GeneratorStage):
    """Stage 03 — cluster vertices into sectors and give them capacities.

    Reads ``vertices.csv`` and ``edges.csv`` from ``--path`` (the navgraph
    directory) and clusters them into sectors of about
    ``--sector-default-navaid-size`` navaids, convex when ``--convex-sectors 1``.

    Must write, into ``--path`` (the same directory it read):

    ================================  =====================================
    file                              columns
    ================================  =====================================
    ``sectors.csv``                   ``Sector_ID, Capacity``
    ``navaid_sector_assignment.csv``  ``Navaid_ID, Sector_ID``
    ================================  =====================================

    Invariants:

    * **Capacity is per TIMESTEP, not per hour** (README convention ①). At
      TG=60 a timestep is one minute. ``--cap-enroute`` applies to en-route
      sectors and ``--cap-airport`` to airport sectors.
    * ``Capacity`` >= 1. A capacity of 0 is unsatisfiable by construction, since
      a flight occupies its departure sector in its first timestep (check
      ``P4``).
    * Every vertex in ``vertices.csv`` appears exactly once as ``Navaid_ID``,
      and every ``Sector_ID`` used there is declared in ``sectors.csv``
      (checks ``P5``/``D6``).
    * **``sectors.csv`` is keyed by navaid** despite the column name (README
      convention ②): the shipped stage writes one row per graph vertex, and
      ``navaid_sector_assignment.csv`` is what maps vertices onto clusters.
      An implementation that writes one row per cluster instead is legal — the
      checker only requires that the two files agree — but downstream consumers
      that assume convention ② will read it differently.
    """

    stage_key = "sectors"


class FiledFlightPlanStage(GeneratorStage):
    """Stage 04 — route each flight over the graph and time-stamp it.

    Reads ``flights.csv`` and ``aircrafts.csv`` from ``--data-dir`` and the
    graph from ``--navgraph-dir``, and works in timesteps of
    60 / ``--time-granularity`` minutes, resampling under ``--resample-seed``.

    Must write, into ``--data-dir``:

    ==========================  =====================================
    file                        columns
    ==========================  =====================================
    ``filed_flights.csv``       ``Flight_ID, Position, Time``
    ==========================  =====================================

    and may rewrite ``flights.csv`` and ``aircrafts.csv`` in place.

    Invariants — these are what the checker tests, so an implementation that
    breaks them produces instances a solver can reject:

    * ``Position`` is a vertex ``IDENTIFIER`` (or an integer vertex id);
      ``Time`` is an integer timestep.
    * ``Time`` strictly increases along a flight, one row per timestep visited
      (checks ``P6``/``F2``, ``D2``).
    * Consecutive positions are adjacent in ``edges.csv`` or identical — no
      teleporting (checks ``D1``/``F8``).
    * The first and last position of every flight are airport vertices, and
      they differ (checks ``D3``/``F7``, ``C4``).
    * **The 24-hour window is a hard contract** (README convention ③): every
      timestep lies in ``[0, --time-granularity × 24]``. The shipped stage
      enforces it by resampling a nearer destination, and raises rather than
      shipping a short instance (checks ``P3``/``F1``).
    * **Two legs of one airframe are separated by at least one whole timestep**
      — arrive at *t*, depart no earlier than *t+1*. Sharing the boundary slot
      puts the aircraft at two navpoints in one timestep (checks ``D4``/``F4``).
    * The set of ``Flight_ID`` must equal the set in ``flights.csv``: the stage
      may split an airframe's legs onto new aircraft ids, but it may not lose or
      invent a flight.

    Note this stage is **not idempotent**: it rewrites its own input. A re-run
    must start from stage 01, not from here.
    """

    stage_key = "filedplans"


class TransformStage(GeneratorStage):
    """Stage 05 — turn one experiment into solver-ready instance directories.

    Reads the experiment directory ``--in-exp-dir`` (its ``navgraph/`` and every
    subdirectory matching ``--select``) and writes one instance directory per
    data sample under ``--out-root`` / ``--experiment-name``, named
    ``<n_flights zero-padded to 7>_SEED<seed>``.

    Must write, into each instance directory:

    ================================  =====================================
    file                              columns
    ================================  =====================================
    ``flights.csv``                   ``Flight_ID, Position, Time``
    ``airplanes.csv``                 ``Airplane_ID, Speed_kts``
    ``airplane_flight_assignment.csv``  ``Airplane_ID, Flight_ID``
    ``airports.csv``                  ``Airport_Vertex``
    ``graph_edges.csv``               ``source, target, dist_m``
    ``sectors.csv``                   ``Sector_ID, Capacity``
    ``navaid_sector_assignment.csv``  ``Navaid_ID, Sector_ID``
    ``transform_manifest.json``       ``n_flights``, ``seed``, ``files``
    ``mappings/vertex_map.csv``       ``IDENTIFIER, VERTEX_ID``
    ``mappings/id_maps.json``         airplane and flight id dictionaries
    ================================  =====================================

    Invariants:

    * Every identifier is a **contiguous integer from 0**: vertex ids, flight
      ids and airplane ids are all re-indexed here, and ``mappings/`` is the
      only record of what they were.
    * ``n_flights`` and ``seed`` in ``transform_manifest.json`` agree with the
      directory name (check ``P9``), and the directory name's flight count is
      the number of distinct ``Flight_ID`` (checks ``P2``/``F9``).
    * The invariants of stage 04 survive the re-indexing — window, adjacency,
      airport endpoints, aircraft separation. This is the artefact a user
      downloads, so ``check_instances.py`` runs against exactly these files.
    """

    stage_key = "transform"


INTERFACES: Dict[str, Type[GeneratorStage]] = {
    "model": DemandModelStage,
    "flights": FlightScheduleStage,
    "navgraph": NavigationGraphStage,
    "sectors": SectorCapacityStage,
    "filedplans": FiledFlightPlanStage,
    "transform": TransformStage,
}


# --------------------------------------------------------------------------
# the shipped implementations
# --------------------------------------------------------------------------

def _run_script(script: str, argv: Sequence[str]) -> None:
    """Spawn a shipped stage script exactly as the pipeline always has.

    ``run_pipeline.run`` is imported late so that this module can also be used
    from ``05_transform_for_optimizer.py`` without an import cycle.
    """
    from run_pipeline import run  # noqa: PLC0415 -- late by design, see above
    run(["python", script, *[str(x) for x in argv]])


class DefaultDemandModel(DemandModelStage):
    """The shipped stage 00, spawned as ``python 00_…py``."""

    SCRIPT = "00_model_generation_script_refactored.py"

    def start(self, argv: Sequence[str]) -> None:
        _run_script(self.SCRIPT, argv)


class DefaultFlightSchedule(FlightScheduleStage):
    """The shipped stage 01, spawned as ``python 01_…py``."""

    SCRIPT = "01_data_generation_script_refactored.py"

    def start(self, argv: Sequence[str]) -> None:
        _run_script(self.SCRIPT, argv)


class DefaultNavigationGraph(NavigationGraphStage):
    """The shipped stage 02, spawned as ``python 02_…py``."""

    SCRIPT = "02_graph_generator.py"

    def start(self, argv: Sequence[str]) -> None:
        _run_script(self.SCRIPT, argv)


class DefaultSectorCapacity(SectorCapacityStage):
    """The shipped stage 03, spawned as ``python 03_…py``."""

    SCRIPT = "03_sector_capacity_generator.py"

    def start(self, argv: Sequence[str]) -> None:
        _run_script(self.SCRIPT, argv)


class DefaultFiledFlightPlan(FiledFlightPlanStage):
    """The shipped stage 04, spawned as ``python 04_…py``."""

    SCRIPT = "04_simplified_filed_flight_plan_generator.py"

    def start(self, argv: Sequence[str]) -> None:
        _run_script(self.SCRIPT, argv)


class DefaultTransform(TransformStage):
    """The shipped stage 05.

    Unlike the other defaults this one is *not* spawned: stage 05 is its own
    entry point, and when it selects the default implementation it simply runs
    its own code. The class exists so that ``transform`` has a registered
    ``default`` like every other stage, and so that an alternative transform can
    delegate to the shipped one by instantiating it.
    """

    SCRIPT = "05_transform_for_optimizer.py"

    def start(self, argv: Sequence[str]) -> None:
        _run_script(self.SCRIPT, argv)


#: stage key -> {implementation name -> class}
REGISTRY: Dict[str, Dict[str, Type[GeneratorStage]]] = {
    "model": {"default": DefaultDemandModel},
    "flights": {"default": DefaultFlightSchedule},
    "navgraph": {"default": DefaultNavigationGraph},
    "sectors": {"default": DefaultSectorCapacity},
    "filedplans": {"default": DefaultFiledFlightPlan},
    "transform": {"default": DefaultTransform},
}


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

class StageImplError(Exception):
    """Raised when ``--stage-impl`` cannot be honoured."""


def register(stage_key: str, name: str, cls: Type[GeneratorStage]) -> None:
    """Register ``cls`` under ``name`` so ``--stage-impl <stage>=<name>`` finds it."""
    if stage_key not in REGISTRY:
        raise StageImplError(f"unknown stage '{stage_key}'; expected one of {', '.join(STAGE_KEYS)}")
    _check_subclass(stage_key, cls)
    REGISTRY[stage_key][name] = cls


def _check_subclass(stage_key: str, cls: type) -> None:
    iface = INTERFACES[stage_key]
    if not (isinstance(cls, type) and issubclass(cls, iface)):
        raise StageImplError(
            f"{getattr(cls, '__name__', cls)} is not a {iface.__name__}: an implementation of "
            f"stage '{stage_key}' must subclass stage_interfaces.{iface.__name__} and define "
            f"start(self, argv)."
        )


def _import_module(ref: str):
    """Import ``ref`` as a module name, or as a path to a .py file."""
    if ref.endswith(".py"):
        name = "_stage_impl_" + ref.replace("/", "_").replace(".", "_").replace("\\", "_")
        spec = importlib.util.spec_from_file_location(name, ref)
        if spec is None or spec.loader is None:
            raise StageImplError(f"cannot load implementation file: {ref}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    try:
        return importlib.import_module(ref)
    except ImportError as exc:
        raise StageImplError(
            f"cannot import '{ref}' ({exc}). Give an importable module, or a path ending in .py."
        ) from exc


def load(stage_key: str, spec: str | None = None) -> GeneratorStage:
    """Resolve ``spec`` to an instantiated implementation of ``stage_key``.

    ``spec`` may be ``None`` or a registered name (``default``), ``module:Class``
    or ``path/to/file.py:Class``.
    """
    if stage_key not in REGISTRY:
        raise StageImplError(f"unknown stage '{stage_key}'; expected one of {', '.join(STAGE_KEYS)}")
    if spec is None:
        spec = "default"
    if ":" in spec:
        ref, _, cls_name = spec.rpartition(":")
        mod = _import_module(ref)
        cls = getattr(mod, cls_name, None)
        if cls is None:
            raise StageImplError(f"{ref} has no attribute '{cls_name}'")
    else:
        cls = REGISTRY[stage_key].get(spec)
        if cls is None:
            known = ", ".join(sorted(REGISTRY[stage_key]))
            raise StageImplError(
                f"no implementation '{spec}' registered for stage '{stage_key}' (known: {known}). "
                f"Use 'module:ClassName' or 'path/to/file.py:ClassName' for your own."
            )
    _check_subclass(stage_key, cls)
    return cls()


def parse_stage_impl(values: Iterable[str] | None) -> Dict[str, str]:
    """Turn ``["navgraph=my:Cls", …]`` into ``{"navgraph": "my:Cls", …}``."""
    out: Dict[str, str] = {}
    for raw in values or ():
        if "=" not in raw:
            raise StageImplError(
                f"--stage-impl expects STAGE=SPEC, got '{raw}'. "
                f"STAGE is one of {', '.join(STAGE_KEYS)}."
            )
        stage, _, spec = raw.partition("=")
        stage, spec = stage.strip(), spec.strip()
        if stage not in REGISTRY:
            raise StageImplError(
                f"unknown stage '{stage}' in --stage-impl; expected one of {', '.join(STAGE_KEYS)}"
            )
        if not spec:
            raise StageImplError(f"--stage-impl {stage}= needs an implementation")
        out[stage] = spec
    return out


def load_all(values: Iterable[str] | None, stages: Sequence[str] = STAGE_KEYS
             ) -> Dict[str, GeneratorStage]:
    """Resolve every stage in ``stages``, overriding from ``--stage-impl`` values."""
    chosen = parse_stage_impl(values)
    unknown = set(chosen) - set(stages)
    if unknown:
        raise StageImplError(
            f"--stage-impl {', '.join(sorted(unknown))}= is not selectable here; "
            f"this entry point drives {', '.join(stages)}."
        )
    impls = {key: load(key, chosen.get(key)) for key in stages}
    for key, spec in chosen.items():
        print(f"[stage-impl] {key}: {spec} -> {type(impls[key]).__name__}")
    return impls


def argv_to_dict(argv: Sequence[str]) -> Dict[str, object]:
    """Convenience for implementations: ``["--out-dir", "x", "--flat-out"]`` ->
    ``{"out-dir": "x", "flat-out": True}``.

    A flag repeated more than once keeps its last value; a flag with no value is
    ``True``. Leading ``--`` is stripped and hyphens are kept as written.
    """
    out: Dict[str, object] = {}
    key: str | None = None
    for tok in argv:
        tok = str(tok)
        if tok.startswith("--"):
            if key is not None:
                out[key] = True
            key = tok[2:]
        elif key is not None:
            out[key] = tok
            key = None
    if key is not None:
        out[key] = True
    return out


__all__ = [
    "STAGE_KEYS", "INTERFACES", "REGISTRY", "StageImplError",
    "GeneratorStage", "DemandModelStage", "FlightScheduleStage",
    "NavigationGraphStage", "SectorCapacityStage", "FiledFlightPlanStage",
    "TransformStage",
    "DefaultDemandModel", "DefaultFlightSchedule", "DefaultNavigationGraph",
    "DefaultSectorCapacity", "DefaultFiledFlightPlan", "DefaultTransform",
    "register", "load", "load_all", "parse_stage_impl", "argv_to_dict",
]
