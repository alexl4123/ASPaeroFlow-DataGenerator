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

To see what is available without reading this file::

    python run_pipeline.py --list-stage-impls         # every stage, its contract, the examples
    python run_pipeline.py --stage-impl sectors=help  # one stage's contract in full

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

An implementation may shell out, import a library, or call the shipped stage and
post-process what it produced — ``load("sectors").start(argv)`` gets you the
shipped stage 03; ``example_stage_impls.py`` shows both patterns.

One implementation per stage
----------------------------
The class in the shipped script **is** the stage. ``02_graph_generator.py``'s
``NavigationGraphBuilder`` subclasses ``NavigationGraphStage`` and its ``start``
is the real algorithm; it is registered as stage ``navgraph``'s ``default`` and
``run_pipeline.py`` calls it directly, in this process. There is no adapter
class in between and no subprocess, so ``--stage-impl navgraph=default`` and
``--stage-impl navgraph=02_graph_generator.py:NavigationGraphBuilder`` are two
spellings of the same thing.

That script is therefore also the file to read, and the file to copy, when
writing your own:

* ``model``      00_model_generation_script_refactored.py:DemandModelBuilder
* ``flights``    01_data_generation_script_refactored.py:FlightScheduleSampler
* ``navgraph``   02_graph_generator.py:NavigationGraphBuilder
* ``sectors``    03_sector_capacity_generator.py:SectorCapacityGenerator
* ``filedplans`` 04_simplified_filed_flight_plan_generator.py:FiledFlightPlanGenerator
* ``transform``  05_transform_for_optimizer.py:OptimizerTransform

Each interface also carries its reference implementation as ``reference_impl``,
and ``load(stage)`` returns an instance of it.

Two consequences of running in one process
------------------------------------------
* **Stage output is not captured.** Everything a stage prints goes straight to
  the terminal, interleaved with the pipeline's own progress lines, instead of
  being buffered and replayed only if the stage failed.
* **A failing stage raises where it stands.** The original exception propagates
  out of ``run_pipeline.main`` rather than arriving as a
  ``subprocess.CalledProcessError``; the pipeline still exits non-zero, and the
  traceback now names the line that actually failed. An implementation that
  wants the pipeline to stop should raise; one that wants to decline a single
  dataset should ``return`` without writing its artefacts, which leaves
  ``run_pipeline``'s ``file_exists`` guards to skip the rest of that dataset.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
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

    #: ``path/to/file.py:ClassName`` of the shipped reference implementation of
    #: this stage — the working example to read and copy. Set on each subclass.
    reference_impl: str = ""

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

    #: the shipped example to read and copy
    reference_impl = "00_model_generation_script_refactored.py:DemandModelBuilder"

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

    #: the shipped example to read and copy
    reference_impl = "01_data_generation_script_refactored.py:FlightScheduleSampler"

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

    #: the shipped example to read and copy
    reference_impl = "02_graph_generator.py:NavigationGraphBuilder"

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
    * Every vertex in ``vertices.csv`` appears exactly once as ``Navaid_ID``.
    * The two files need **not** share a ``Sector_ID`` namespace here. Checks
      ``P5``/``D6`` — every sector referenced is a sector declared — bind the
      *parsed* instance, not this stage's output: stage 05 re-keys both columns
      through the vertex table before the checker ever sees them. The shipped
      stage 03 writes ``SECTOR_000000`` / ``SECTOR_AIRPORT_RCKH`` into
      ``navaid_sector_assignment.csv`` while ``sectors.csv`` holds
      ``GRID_EAST-ASIA_Y00X00_FL100`` — the two sets are disjoint, with no
      overlap at all, and the instance that comes out is still valid.
    * **``sectors.csv`` is keyed by navaid** despite the column name (README
      convention ②): the shipped stage writes one row per graph vertex, and
      ``navaid_sector_assignment.csv`` is what maps vertices onto clusters.
      An implementation that writes one row per cluster instead is legal — the
      checker only requires that the two files agree — but downstream consumers
      that assume convention ② will read it differently.

      .. TODO:: **Unresolved; for the project owner to decide, not the reader.**
         The paragraph above says a per-cluster ``sectors.csv`` "is legal", and
         ``example_stage_impls.py:LatitudeBandSectors`` — an example written
         against this very interface — says in its own docstring that such a
         file "would not survive this pipeline's stage 05". Both cannot stand:
         stage 05 maps the ``Sector_ID`` column through the vertex table, and a
         cluster name has no vertex to map to. The choice is between
         (a) teaching ``05_transform_for_optimizer.py`` to accept a per-cluster
         ``sectors.csv``, and (b) dropping the promise here and requiring one
         row per vertex. Left as written until that is settled.
    """

    #: the shipped example to read and copy
    reference_impl = "03_sector_capacity_generator.py:SectorCapacityGenerator"

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

    #: the shipped example to read and copy
    reference_impl = "04_simplified_filed_flight_plan_generator.py:FiledFlightPlanGenerator"

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

    #: the shipped example to read and copy
    reference_impl = "05_transform_for_optimizer.py:OptimizerTransform"

    stage_key = "transform"


INTERFACES: Dict[str, Type[GeneratorStage]] = {
    "model": DemandModelStage,
    "flights": FlightScheduleStage,
    "navgraph": NavigationGraphStage,
    "sectors": SectorCapacityStage,
    "filedplans": FiledFlightPlanStage,
    "transform": TransformStage,
}

_HERE = Path(__file__).resolve().parent


class StageImplError(Exception):
    """Raised when ``--stage-impl`` cannot be honoured."""


# --------------------------------------------------------------------------
# the registry
#
# One implementation per stage: the class in the shipped script, registered as
# that stage's ``default``. There is no adapter in between.
# --------------------------------------------------------------------------

#: stage key -> {implementation name -> class}
#:
#: ``default`` is resolved on first use rather than written out here. Every
#: stage script does ``from stage_interfaces import <Stage>Stage``, so naming
#: those classes at this module's scope would be a circular import, and it
#: would drag pandas, scikit-learn and networkx into any process that only
#: wants to parse a ``--stage-impl`` argument. Once :func:`reference_class`,
#: :func:`load` or :func:`implementations` has run for a stage, the entry *is*
#: the script's class.
REGISTRY: Dict[str, Dict[str, Type[GeneratorStage]]] = {key: {} for key in STAGE_KEYS}


# --------------------------------------------------------------------------
# importing an implementation
# --------------------------------------------------------------------------

def _module_for_file(path: str):
    """An already-imported module whose source file is ``path``, or ``None``.

    A stage script launched directly (``python 05_transform_for_optimizer.py``)
    is in ``sys.modules`` as ``__main__``. Executing its file a second time
    under a second name would give a second, unrelated copy of its classes —
    and, for a 68 KB module, a second import of pandas and networkx — so reuse
    whatever is already loaded.
    """
    try:
        target = os.path.realpath(path)
    except OSError:
        return None
    for mod in list(sys.modules.values()):
        src = getattr(mod, "__file__", None)
        if not src:
            continue
        try:
            if os.path.realpath(src) == target:
                return mod
        except OSError:
            continue
    return None


def _resolve_file(ref: str) -> str:
    """Locate a ``.py`` reference: as given, else beside this module.

    The shipped implementations are recorded as bare filenames
    (``02_graph_generator.py``), which used to be resolved by the shell against
    the working directory. Falling back to this module's own directory means
    the pipeline finds its own stages without depending on where it was started.
    """
    if os.path.isabs(ref) or os.path.exists(ref):
        return ref
    beside = _HERE / ref
    return str(beside) if beside.exists() else ref


def _import_module(ref: str):
    """Import ``ref`` as a module name, or as a path to a .py file."""
    if ref.endswith(".py"):
        path = _resolve_file(ref)
        existing = _module_for_file(path)
        if existing is not None:
            return existing
        name = "_stage_impl_" + ref.replace("/", "_").replace(".", "_").replace("\\", "_")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise StageImplError(f"cannot load implementation file: {ref}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except FileNotFoundError as exc:
            sys.modules.pop(name, None)
            raise StageImplError(f"cannot load implementation file: {ref} ({exc})") from exc
        return mod
    try:
        return importlib.import_module(ref)
    except ImportError as exc:
        raise StageImplError(
            f"cannot import '{ref}' ({exc}). Give an importable module, or a path ending in .py."
        ) from exc


def _check_subclass(stage_key: str, cls: type) -> None:
    iface = INTERFACES[stage_key]
    if not (isinstance(cls, type) and issubclass(cls, iface)):
        raise StageImplError(
            f"{getattr(cls, '__name__', cls)} is not a {iface.__name__}: an implementation of "
            f"stage '{stage_key}' must subclass stage_interfaces.{iface.__name__} and define "
            f"start(self, argv)."
        )


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def reference_class(stage_key: str) -> Type[GeneratorStage]:
    """Import and return the shipped implementation class of ``stage_key``.

    That is the class named by ``INTERFACES[stage_key].reference_impl`` — the
    algorithm itself, living in the stage script. The result is cached into
    ``REGISTRY[stage_key]["default"]``.
    """
    if stage_key not in REGISTRY:
        raise StageImplError(f"unknown stage '{stage_key}'; expected one of {', '.join(STAGE_KEYS)}")
    cached = REGISTRY[stage_key].get("default")
    if cached is not None:
        return cached
    spec = INTERFACES[stage_key].reference_impl
    ref, _, cls_name = spec.rpartition(":")
    mod = _import_module(ref)
    cls = getattr(mod, cls_name, None)
    if cls is None:
        raise StageImplError(
            f"the shipped implementation of stage '{stage_key}' is recorded as '{spec}', "
            f"but {ref} defines no '{cls_name}'."
        )
    _check_subclass(stage_key, cls)
    REGISTRY[stage_key]["default"] = cls
    return cls


def implementations(stage_key: str) -> Dict[str, Type[GeneratorStage]]:
    """``{short name -> class}`` for ``stage_key``, with ``default`` resolved."""
    reference_class(stage_key)
    return dict(REGISTRY[stage_key])


def register(stage_key: str, name: str, cls: Type[GeneratorStage]) -> None:
    """Register ``cls`` under ``name`` so ``--stage-impl <stage>=<name>`` finds it.

    Note what this can and cannot do. A class registers only when its module is
    imported, and the only way to get a module outside this repository imported
    is to name it as ``module:ClassName`` — by which point the short name has
    nothing left to do. So short names are **repo-internal**: useful for classes
    this repository already imports, not a plug-in mechanism for third parties.
    ``--list-stage-impls`` says the same thing in the user-facing help.
    """
    if stage_key not in REGISTRY:
        raise StageImplError(f"unknown stage '{stage_key}'; expected one of {', '.join(STAGE_KEYS)}")
    _check_subclass(stage_key, cls)
    REGISTRY[stage_key][name] = cls


def load(stage_key: str, spec: str | None = None) -> GeneratorStage:
    """Resolve ``spec`` to an instantiated implementation of ``stage_key``.

    ``spec`` may be ``None`` or a registered name (``default``), ``module:Class``
    or ``path/to/file.py:Class``. ``load(key)`` is how an alternative
    implementation delegates to the shipped one.
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
    elif spec == "default":
        cls = reference_class(stage_key)
    else:
        cls = REGISTRY[stage_key].get(spec)
        if cls is None:
            known = ", ".join(sorted(implementations(stage_key)))
            raise StageImplError(
                f"no implementation '{spec}' registered for stage '{stage_key}' (known: {known}). "
                f"Use 'module:ClassName' or 'path/to/file.py:ClassName' for your own, or "
                f"'{stage_key}=help' for the contract."
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


def show_contract_requests(values: Iterable[str] | None) -> bool:
    """Handle ``--stage-impl STAGE=help``: print those contracts, report whether any.

    Called before anything is written, so ``--stage-impl sectors=help`` prints
    the specification and the run stops without touching the output tree.
    """
    asked = [key for key, spec in parse_stage_impl(values).items() if spec == HELP_SPEC]
    for i, key in enumerate(asked):
        if i:
            print()
        print(contract_text(key))
    return bool(asked)


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


# --------------------------------------------------------------------------
# discovery — what can I actually pass to --stage-impl?
#
# Nothing below is a hand-written list of implementations, and nothing below
# needs editing when one is added. The contract and its location come from the
# interface class itself; the reference implementation comes from that class's
# ``reference_impl``; the short names come from REGISTRY; and the examples are
# found by importing ``example_stage_impls.py`` and walking ``__subclasses__``.
# --------------------------------------------------------------------------

#: the spec that asks for a stage's contract instead of an implementation
HELP_SPEC = "help"

#: shipped worked examples — imported for discovery, never listed by hand
EXAMPLES_FILE = "example_stage_impls.py"


def _iter_subclasses(cls: type) -> Iterable[type]:
    for sub in cls.__subclasses__():
        yield sub
        yield from _iter_subclasses(sub)


def _load_examples() -> str | None:
    """Import the shipped examples so their classes exist. Returns a note on failure."""
    try:
        _import_module(str(_HERE / EXAMPLES_FILE))
        return None
    except Exception as exc:  # noqa: BLE001 -- a listing must survive a broken example file
        return f"note: could not import {EXAMPLES_FILE} ({exc}); its examples are not listed."


def _where(cls: type) -> str:
    """``file.py:ClassName`` for ``cls``, relative to the repository when possible."""
    try:
        src = inspect.getsourcefile(cls) or "<unknown>"
    except TypeError:
        return f"<unknown>:{cls.__name__}"
    try:
        src = os.path.relpath(src, _HERE)
    except ValueError:
        pass
    return f"{src}:{cls.__name__}"


def _line_of(cls: type) -> int:
    try:
        return inspect.getsourcelines(cls)[1]
    except (OSError, TypeError):
        return 0


def _summary(cls: type) -> str:
    doc = (inspect.getdoc(cls) or "").strip()
    return doc.splitlines()[0] if doc else "(no docstring)"


def alternatives(stage_key: str) -> List[type]:
    """Every loaded implementation of ``stage_key`` other than the shipped one.

    Found by walking ``__subclasses__`` of the interface, so it cannot go stale:
    whatever has been imported and subclasses the interface is listed.
    """
    try:
        shipped: type | None = reference_class(stage_key)
    except StageImplError:
        shipped = None
    seen: set = set()
    out: List[type] = []
    for sub in _iter_subclasses(INTERFACES[stage_key]):
        if sub is shipped or sub in seen:
            continue
        seen.add(sub)
        out.append(sub)
    return sorted(out, key=lambda c: (_where(c), c.__name__))


def contract_text(stage_key: str) -> str:
    """One stage's contract in full: where it is written, and the docstring."""
    if stage_key not in INTERFACES:
        raise StageImplError(f"unknown stage '{stage_key}'; expected one of {', '.join(STAGE_KEYS)}")
    iface = INTERFACES[stage_key]
    src = os.path.relpath(inspect.getsourcefile(iface) or __file__, _HERE)
    lines = [
        f"stage '{stage_key}' -- the contract is the docstring of "
        f"stage_interfaces.{iface.__name__}",
        f"  defined at : {src}:{_line_of(iface)}",
        f"  copy from  : {iface.reference_impl}",
        "",
    ]
    lines += ["  " + ln if ln.strip() else "" for ln in (inspect.getdoc(iface) or "").splitlines()]
    return "\n".join(lines)


def describe_all() -> str:
    """The body of ``--list-stage-impls``."""
    note = _load_examples()
    out: List[str] = [
        "Stage implementations",
        "=====================",
        "",
        "  --stage-impl STAGE=SPEC   select an implementation for one stage",
        "  --stage-impl STAGE=help   print that stage's contract in full",
        "",
        "SPEC is a short name from the table below, 'module:ClassName', or",
        "'path/to/file.py:ClassName'. The class must subclass the stage's",
        "interface; anything else is rejected before the run starts.",
        "",
    ]
    for key in STAGE_KEYS:
        iface = INTERFACES[key]
        src = os.path.relpath(inspect.getsourcefile(iface) or __file__, _HERE)
        out.append(key)
        out.append(f"    {_summary(iface)}")
        out.append(f"  contract    : stage_interfaces.{iface.__name__}  ({src}:{_line_of(iface)})")
        try:
            names = ", ".join(sorted(implementations(key)))
        except StageImplError as exc:
            names = f"<unavailable: {exc}>"
        out.append(f"  short names : {names}")
        out.append(f"  reference   : {iface.reference_impl}")
        alts = alternatives(key)
        if alts:
            out.append("  examples    :")
            for cls in alts:
                out.append(f"      {_where(cls)}")
                out.append(f"          {_summary(cls)}")
        else:
            out.append("  examples    : none shipped for this stage")
        out.append("")
    if note:
        out.append(note)
        out.append("")
    out += [
        "Two frictions to know before you write one",
        "------------------------------------------",
        "* Your file has to run 'from stage_interfaces import <Stage>Stage', so an",
        "  outside implementation is coupled to this repository being importable.",
        "  Run from the repository root, or put the repository on PYTHONPATH.",
        "* Short names are repo-internal. stage_interfaces.register() only takes",
        "  effect once your module has been imported, and the only way to get an",
        "  outside module imported is to name it 'module:ClassName' -- by which",
        "  point the short name has nothing left to do. Select your own code as",
        "  'module:ClassName' or 'path/to/file.py:ClassName'; the short names above",
        "  are for classes this repository already imports.",
    ]
    return "\n".join(out)


__all__ = [
    "STAGE_KEYS", "INTERFACES", "REGISTRY", "HELP_SPEC", "StageImplError",
    "GeneratorStage", "DemandModelStage", "FlightScheduleStage",
    "NavigationGraphStage", "SectorCapacityStage", "FiledFlightPlanStage",
    "TransformStage",
    "register", "load", "load_all", "parse_stage_impl", "argv_to_dict",
    "reference_class", "implementations", "alternatives",
    "contract_text", "describe_all", "show_contract_requests",
]
