# ASPaeroFlow Data Generator

Generates synthetic but realistically structured air traffic scenarios for **Air Traffic Flow and
Capacity Management (ATFCM)** research: flight schedules, a navigation graph, sector definitions
with capacities, and filed flight plans, in a form solvers can consume directly.

Demand, origin–destination structure, flight durations and turnaround times are all estimated
from a real flight list (OpenSky). The output is a *benchmark* dataset built from real data — it
is designed to produce instances that are challenging to solve, and its fidelity to real traffic
is measured and documented rather than assumed (see `FUTURE_WORK.md`).

---

## Requirements

* **Python ≥ 3.10** (the code uses `X | None` annotations)
* `pip install -r requirements.txt` — pandas, numpy, **scikit-learn**, networkx, tqdm

scikit-learn is not optional: `sklearn.neighbors.BallTree` builds the navigation graph.

## Input data

| input | where it goes | needed for | licence |
|---|---|---|---|
| [OpenSky COVID-19 flight dataset](https://zenodo.org/records/5815448), e.g. `flightlist_20190601_20190630.csv` | anywhere; pass `--csv-path` | everything | CC-BY-4.0 |
| [OurAirports `airports.csv`](https://ourairports.com/data/airports.csv) | `./ourairports/` (a patched snapshot is tracked here) | airport set | public domain |
| X-Plane navdata: **`fix.dat`** and **`nav.dat`** | `./test_navpoints/` | only for real-waypoint regions | GPL-2.0-or-later |

The navdata files are named `fix.dat` and `nav.dat` — *not* `earth_fix.dat` / `earth_nav.dat`.
They are only read when building a graph from real waypoints. With `--grid-navpoints true` the
generator synthesises a rectangular grid instead and **no navdata is needed at all**; every
`*_small_scaling` and grid region works this way.

---

## Pipeline

`run_pipeline.py` drives six stages and skips any whose artefacts already exist:

| stage | script | produces |
|---|---|---|
| 00 | `00_model_generation_script_refactored.py` | `model/` — per-airport departure profiles, origin-conditional OD model, duration and turnaround distributions |
| 01 | `01_data_generation_script_refactored.py` | `flights.csv`, `aircrafts.csv` per dataset |
| 02 | `02_graph_generator.py` | `navgraph/` — vertices and edges |
| 03 | `03_sector_capacity_generator.py` | sector clustering and capacities |
| 04 | `04_simplified_filed_flight_plan_generator.py` | `filed_flights.csv` — the routed, time-stamped plan |
| 05 | `05_transform_for_optimizer.py` | the solver-ready instance directory |

Then, optionally:

| script | purpose |
|---|---|
| `06_capacity_sweep.py` | computes each instance's **nominal capacity** (its peak sector occupancy) and emits overlays at 10 %–100 % of it — the PCAP levels in the published dataset |
| `06_bluesky_converter.py` | export to BlueSky |
| `07_check_parsed_experiments_graph_connectedness.py` | sanity check on parsed output |
| `build_release_zips.py` | assembles the release archives, materialising every capacity level |
| `check_instances.py` | **validity checks on a generated instance** — window, adjacency, aircraft separation, sector cover; see below |

### Running it

Everything is driven by a JSON config; the CLI only overrides.

```bash
python run_pipeline.py --config default_configs_large_scaling_tg/04_0_dach_TG60.json
python 05_transform_for_optimizer.py --in-exp-dir unparsed_experiment_data_.../<REGION> \
                                     --out-root experiment_data_...
python 06_capacity_sweep.py --experiment experiment_data_.../<REGION> --overlay-root capacity_overlays_...
```

### Config families

| directory | what it is |
|---|---|
| `default_configs/` | single-region examples |
| `default_configs_TG4/` | the earlier published (v1) instances |
| `default_configs_large_scaling_tg/` | **the shipped large-scale family**: 8 regions × TG {1,4,15,60} |
| `default_configs_small_scaling/` | **the shipped small-scale family**: 5 grid regions, 10–100 flights |
| `default_configs_validation_seeds/` | 10-seed and odd/even hold-out runs for the statistical validation — evaluation only, never shipped |

### Frequently used options

| option | meaning |
|---|---|
| `--config` | JSON config; supplies defaults for everything below |
| `--csv-path` | the OpenSky flight list (file or directory) |
| `--date-start` / `--date-end` | the date range to fit the model on. **Use this**, not `--target-day`, which is the legacy single-day mode |
| `--day-parity {all,odd,even}` | fit on half the calendar days, for held-out validation |
| `--time-granularity` | bins per hour: 1 = 60-minute timesteps, 60 = 1-minute timesteps |
| `--flight-flights` / `--flight-seeds` | the (demand level, seed) grid to generate |
| `--timezone` | hours offset; the model is localised before binning |
| `--grid-navpoints` | synthesise grid vertices instead of reading X-Plane navdata |
| `--criterion {gabriel,rng}`, `--max-edge-km` | navgraph topology |
| `--cap-enroute`, `--cap-airport` | per-timestep capacities written into `sectors.csv` |
| `--resample-seed` | seed for stage 04's destination resampling |

---

## Four conventions that will otherwise surprise you

**① Capacities are per TIMESTEP, not per hour.** `sectors.csv::Capacity` is the number of flights
allowed in a sector during one timestep. At TG=60 that timestep is one minute. Do not divide by
the number of timesteps in an hour.

**② `sectors.csv` is keyed by NAVAID despite the column being called `Sector_ID`.** There is one
row per graph vertex; `navaid_sector_assignment.csv` maps vertices onto sector clusters (DACH at
TG=60: 1,508 vertices onto 182 sectors). How per-vertex capacities compose into a sector capacity
is the solver's choice — the reference optimizer offers `max` (default), `sum` and average-based
rules.

**③ The 24-hour window is a hard contract.** Every flight departs and lands within
`[0, TG × 24]`. Stage 04 enforces it by resampling a nearer destination when a drawn flight will
not fit, and raises rather than silently shipping a short instance. A consequence is that
instances begin and end with empty airspace.

**④ Stage 04 is NOT idempotent.** It rewrites `flights.csv`. Re-running a pipeline must start from
stage 01, not stage 04 — delete the `DATA_*` directories first. Note also that `run_pipeline.run()`
captures subprocess output and prints it only on failure, so stage 04's `[OK] …` confirmation line
never appears in a successful pipeline log. Judge success by the exit code.

---

## Swapping a stage for your own implementation

Each stage has a parent class in `stage_interfaces.py` with one required method, `start(argv)`.
The shipped scripts are registered as that stage's `default`, so **nothing changes unless you ask
for something else**. To use your own trajectory generator, graph builder or capacity rule,
subclass the stage's parent class and name it on the command line.

| stage | key | parent class | default implementation |
|---|---|---|---|
| 00 demand model | `model` | `DemandModelStage` | `00_model_generation_script_refactored.py` |
| 01 flight schedule | `flights` | `FlightScheduleStage` | `01_data_generation_script_refactored.py` |
| 02 navigation graph | `navgraph` | `NavigationGraphStage` | `02_graph_generator.py` |
| 03 sectors and capacities | `sectors` | `SectorCapacityStage` | `03_sector_capacity_generator.py` |
| 04 filed flight plans | `filedplans` | `FiledFlightPlanStage` | `04_simplified_filed_flight_plan_generator.py` |
| 05 transform | `transform` | `TransformStage` | `05_transform_for_optimizer.py` |

Each parent class's docstring **is the contract**: the files an implementation must write, the
columns they carry, and the invariants it must respect. Read it before writing one — nothing
validates your output until `check_instances.py` runs at the end.

```bash
# stages 00-04, on the pipeline driver.  Repeatable; STAGE=SPEC.
python run_pipeline.py --config default_configs_small_scaling/30_0_east_asia_3x3.json \
    --stage-impl sectors=example_stage_impls:FlatCapacitySectors

# stage 05 is its own entry point, so it carries its own selector (SPEC only)
python 05_transform_for_optimizer.py --in-exp-dir unparsed_experiment_data_.../<REGION> \
    --out-root experiment_data_... --stage-impl my_transform:MyTransform
```

`SPEC` is a registered name (`default`), `module:ClassName`, or `path/to/file.py:ClassName`. The
class must subclass that stage's parent class; anything else is rejected before the run starts,
with an error naming the interface it had to implement.

### Worked example

`example_stage_impls.py` ships two alternative stage 03 implementations and one alternative
stage 05. The smallest one just calls the default:

```python
class LoggingSectors(SectorCapacityStage):
    def start(self, argv):
        opts = argv_to_dict(argv)
        print(f"[example] LoggingSectors: navgraph={opts.get('path')} ...")
        DefaultSectorCapacity().start(argv)
```

```
$ python run_pipeline.py --config default_configs_small_scaling/30_0_east_asia_3x3.json \
      --out-root /tmp/demo --flight-flights 10 --flight-seeds 42 \
      --stage-impl sectors=example_stage_impls:LoggingSectors
[stage-impl] sectors: example_stage_impls:LoggingSectors -> LoggingSectors
...
[example] LoggingSectors: navgraph=/tmp/demo/30-0-EAST-ASIA-3x3-V2/navgraph cap-enroute=1 cap-airport=60000
[RUN] python 03_sector_capacity_generator.py --path /tmp/demo/... --cap-enroute 1 ...
```

Its output is byte-identical to a run without the flag — fingerprint both trees with
`tests/regression/fingerprint.py` and `compare.py` reports `RESULT: MATCH`, 27 files, 0 changed. `FlatCapacitySectors` in the same file
is a real substitution: it runs the default for the clustering, then gives every sector
`--cap-enroute`, so airport sectors drop from 60000 to 1 and that change reaches the parsed
instance.

### Traps

* `start(argv)` receives the **canonical argv** for the stage — the exact argument list the
  default would pass to its script, with config-file values and defaults already resolved.
  Conditional flags (`--config`, `--date-start`, …) are simply absent when unset. Use
  `stage_interfaces.argv_to_dict(argv)` if you want a mapping.
* **Write your artefacts where argv says** (`--out-dir`, `--path`, `--data-dir`). `run_pipeline`
  skips a stage whose artefacts already exist and looks for them at exactly those paths.
* Run from the repository root. The default implementations spawn `python <NN_stage>.py` with no
  path, exactly as the pipeline always has.
* `manifest.json` does **not** record which implementation ran. Nothing in the generated output
  distinguishes a substituted stage from the default; track that yourself.

---

## Checking an instance you generated

`check_instances.py` answers one question about a **parsed** instance: is it well formed enough
for a solver to consume? Point it at an instance directory, an experiment, or a whole parsed root.

```bash
python check_instances.py experiment_data_V2_small_scaling
python check_instances.py experiment_data_.../<REGION>/0000100_SEED42 --verbose
```

```
$ python check_instances.py experiment_data_V2_small_scaling
experiment_data_V2_small_scaling
  time granularity : TG=1  (window = 24 timesteps, from .../30-0-EAST-ASIA-3x3-V2/manifest.json)
  instances        : 200
    PASS  30-0-EAST-ASIA-3x3-V2
    PASS  30-1-CENTRAL-EUROPE-5x5-V2
    PASS  30-2-INDIA-4x10-V2
    PASS  30-3-USA-7x7-V2
    PASS  30-4-MAJOR-EUROPE-10x10-V2

--- reported, not failed -------------------------------------------
  P3-clamp  flights ending on the window edge: mean 6.2% [0.0%-23.8%]
  D5        sectors ever over capacity:        mean 31.5% [0.0%-45.5%]

ALL CHECKS PASSED: 200 instance(s)
```

Exit code 0 = all passed, 1 = violations, 2 = usage error, so it gates a generation run directly.

| check | what it requires |
|---|---|
| `P1` | every required file is present |
| `P2`/`F9` | distinct flight count == the number in the directory name |
| `P3`/`F1` | every timestep in `[0, TG × 24]` (`P3` the upper bound, `P3b` departures before t=0) |
| `F3` | no single waypoint-to-waypoint gap exceeds the whole window |
| `P4`/`F6` | every sector capacity ≥ 1 — 0 is unsatisfiable by construction |
| `P5`/`D6`/`F6` | every graph vertex has a sector; every sector used is declared |
| `P6`/`F2` | each flight's `Time` strictly increases |
| `D2` | one position per (flight, timestep) |
| `P6b`/`F6` | every `Position` is a declared navaid |
| `P7` | every airport vertex is on the graph |
| `P8`/`F5` | every flight has exactly one airplane, and it is declared |
| `D1`/`F8` | consecutive positions are graph-adjacent — no teleporting |
| `D3`/`F7` | flights start and end at airport vertices |
| `C4` | no flight returns to its origin airport |
| `D4`/`F4` | two legs of one airframe are ≥ 1 timestep apart |
| `P9` | `transform_manifest.json` agrees with the directory name |

The identifiers are the ones used in `dataset_analysis_JOAS/00_generator_integrity/` and the dev
log — `P4` here is that `P4`. Where two scripts named the same predicate differently they are
reported together (`D4`/`F4` is one check, not two).

Two things are **reported and never fail**, because both are properties of the benchmark rather
than defects:

| reported | why not a failure |
|---|---|
| `P3-clamp` | share of flights ending exactly on the window edge. At TG=1 the one-slot-per-edge cost consumes the window on larger graphs; accepted and documented (`FUTURE_WORK.md` §D1) |
| `D5` | share of sectors ever over capacity. An ATFCM instance is *meant* to exceed capacity — that imbalance is the problem. A **zero** here means the instance is trivially feasible, which is the suspicious case — except on a `PCAP100` overlay, where zero overload *is* the definition of nominal capacity |

### Traps

* It reads the **parsed** tree only. The demand and OD audits, and the unparsed-tree checks
  `C1`–`C3`, `C5`–`C8`, need `model/` and `DATA_*/`, which stage 05 does not carry forward; they
  stay in `dataset_analysis_JOAS/00_generator_integrity/`.
* **`TG` is not recorded in the instance.** The checker takes it from the generator's
  `manifest.json` if the unparsed experiment is still on this machine, else from a `TG<n>` in the
  path, else assumes 1 — and prints which. `P3` is meaningless if that guess is wrong, so pass
  `--time-granularity` when checking instances you moved off the generating machine.
* It checks validity, not fidelity. Nothing here says an instance resembles real traffic; that is
  what `dataset_analysis_JOAS/` is for.

---

## Time granularity: pick it with the graph, not the region

Each edge traversal costs at least one whole timestep. So when a timestep is longer than the time
an aircraft needs to cross one edge, flight duration becomes `hops × timestep` rather than
distance ÷ speed. Graphs with many short edges therefore need a fine granularity:

| graph | hops | faithful from |
|---|---|---|
| coarse grids (e.g. 7×7, ~5.6 hops) | few | TG ≥ 4 |
| Gabriel graphs (e.g. DACH, ~24 hops) | many | TG ≥ 15, ideally 60 |

Measured on DACH: `hops × timestep` predicts 1440 / 360 / 96 minutes at TG = 1 / 4 / 15, observed
1427 / 358 / 98. At TG=60 the effect stops binding and duration follows the real geometry. This
is a property of the topology, not of the region — see `FUTURE_WORK.md` §D1.

---

## Licence

* Code: **MIT** (`license.md`).
* Generated instance data: **CC-BY-4.0**, because it derives from the CC-BY OpenSky flight list
  and that attribution has to travel with it.
* Third-party inputs keep their own licences: OpenSky CC-BY-4.0, X-Plane navdata
  GPL-2.0-or-later, OurAirports public domain. The published dataset records carry a
  `NOTICE-THIRD-PARTY.md` with the full attribution chain.

## Known defects and planned work

`FUTURE_WORK.md` (project root) lists ten measured defects with evidence and fix directions. The
two that matter most: flight duration is a near-deterministic function of distance through only
three speed values, and the demand model has no day-of-week cycle.
