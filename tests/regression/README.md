# Regression harness

An instrument for proving that a change to the generator changed no output.

It regenerates a set of small fixtures from scratch, fingerprints every produced file,
and diffs that against a baseline captured from the unmodified code on branch
`refactor/modular-stages-and-checks` (tip `d887bc9`). It is not a unit-test suite: it
makes no claim about whether the output is *correct*, only about whether it is *the
same*.

```bash
tests/regression/run_fixtures.sh                   # regenerate + compare; exit 0 = unchanged
tests/regression/run_fixtures.sh --only ea3x3      # one fixture (~25 s)
tests/regression/run_fixtures.sh --list            # what each fixture covers
tests/regression/run_fixtures.sh --shipped-check   # also re-derive the published instances
tests/regression/run_fixtures.sh --update-baseline  # ONLY on known-good code
```

`run_fixtures.sh` exits non-zero if any stage fails or any fixture differs, so it can
gate a refactor directly.

---

## The determinism question: verdict **(B)**, with one qualification

**The generator is byte-reproducible, once `PYTHONHASHSEED` is pinned, modulo absolute
paths recorded in its JSON side-files.** Without that pin, exactly one artefact —
`aircrafts.csv` — comes out with its rows in a different order; nothing else varies,
and no value ever varies.

### Evidence

Two runs of `default_configs_small_scaling/30_0_east_asia_3x3.json`, same seeds, into
two different output roots:

| run pair | CSVs compared | byte-identical | differing |
|---|---|---|---|
| hash seed **not** pinned | 130 | 129 | `DATA_S50_42/aircrafts.csv` — same 26 rows, different order |
| `PYTHONHASHSEED=0` | 130 | 130 | none |

Repeated at scale on the full 40-dataset shipped grid: 16 of 40 `aircrafts.csv` files
differed from the shipped artefacts, **all 16 identical as a multiset of rows, zero
content differences**.

### The specific cause

`04_simplified_filed_flight_plan_generator.py:613`

```python
for aircraft in list(set(flights["aircraft_id"])):
```

This iterates a Python `set` of `str`. CPython randomises `str.__hash__` per process
unless `PYTHONHASHSEED` is set, so the order in which aircraft are visited changes from
run to run. The loop body creates split-aircraft ids (`AC000007_0`, …) as new keys in
the `aircraft_speed` dict, and that dict is written out in insertion order at the end of
`main()`. So the visitation order becomes the row order of `aircrafts.csv`.

Only the order changes, never the values: the body filters `flights` down to the one
aircraft it is processing, so the work done for each aircraft is independent of when it
is visited. The later split pass (`for ac, legs in by_ac.items()`, ~line 691) iterates a
`defaultdict` and is insertion-ordered, hence already deterministic.

`run_fixtures.sh` exports `PYTHONHASHSEED=0` so the check is the strong one — exact
bytes. It does **not** work around the issue, and the issue is not hidden: the pinned
value is recorded in every manifest, and `compare.py` warns when a baseline and a
candidate were taken under different seeds.

### Exactly what is normalised away

Two things, both in `fingerprint.py`:

1. **Absolute filesystem paths appearing as JSON string values**, replaced by
   `<ABS>/<basename>`. This affects `model/run_config.json`,
   `navgraph/run_config.json`, `DATA_*/run_config.json`, `manifest.json` and
   `transform_manifest.json`, which record `--out-root`, `--csv-path`, the per-dataset
   directories and the transform's source and output directories. Any two runs write to
   different directories by construction, so these strings cannot agree and are not
   generator output. The **basename is kept**, so swapping the input flight list for a
   different file is still caught. Relative paths — `"config"`,
   `"ourairports_path"` — are *not* touched and are compared exactly.
2. **Line endings**, normalised to `\n`.

That is the whole list. In particular:

* **No timestamp is excluded, because the generator writes none.** Every artefact was
  checked: no `run_config.json`, `manifest.json` or `transform_manifest.json` carries a
  generation time. The timestamped auto-named experiment directory in
  `run_pipeline.derive_exp_dir` fires only when `experiment-name` is absent, and every
  config used here sets it.
* No numeric field, no row ordering, no CSV column and no library-version string is
  normalised away. The `stats` block in `navgraph/run_config.json` (`num_vertices_base`,
  `num_edges_written`, …) is real output and is compared.
* `instance_info.json` is *excluded* — not normalised — from the `--shipped-check`
  comparison only. It is written by `build_release_zips.py`, not by the pipeline, and it
  records `generator_commit`.

### If a difference is row-order-only

`compare.py` classifies it as `ORDER-ONLY` and **fails anyway** by default. Passing
`--allow-row-reorder` downgrades it to a warning and prints a banner in the report
saying the run used a weakened check and why. Do not pass it habitually: it hides
exactly the class of change that a refactor of stage 04's aircraft handling would
produce.

---

## Does the published dataset regenerate from source? **Yes, exactly.**

`--shipped-check` regenerates the full EAST-ASIA-3x3 grid (10–100 flights × 4 seeds, 40
instances) and diffs it against
`../release_upload/small_scaling/instances/experiment_data_V2_small_scaling.zip`.

```
files compared : 400
added          : 0
removed        : 0
changed        : 0  (0 content/shape, 0 row-order-only)
RESULT: MATCH
```

All 400 published files — `flights.csv`, `airplanes.csv`, `airplane_flight_assignment.csv`,
`graph_edges.csv`, `sectors.csv`, `navaid_sector_assignment.csv`, `airports.csv`,
`mappings/*` and `transform_manifest.json` — reproduce byte-for-byte from this source
tree. The `aircrafts.csv` ordering above does not reach the published instances: stage 05
re-derives `airplanes.csv` in its own canonical order.

The intermediate (unparsed) artefacts reproduce too: compared against the tracked
`unparsed_experiment_data_V2_small_scaling/30-0-EAST-ASIA-3x3-V2`, 130 files match except
the 16 row-reordered `aircrafts.csv` noted above.

This confirms that the current branch tip is output-identical to `dataset-v2-generated`
for this config. Takes ~47 s.

---

## Fixtures

| fixture | covers | runtime |
|---|---|---|
| `ea3x3` | grid navpoints, TG=1, 17-vertex graph, `convex-sectors=1`, `airport-include` | 23 s |
| `ce5x5` | grid navpoints, TG=1, denser airport set | 24 s |
| `india4x10` | grid navpoints, TG=1, non-square grid | 23 s |
| `usa7x7` | grid navpoints, TG=1, western-hemisphere longitudes | 25 s |
| `majeur10x10` | grid navpoints, TG=1, largest small-scaling grid | 25 s |
| `ea3x3_tg4` | same region at **TG=4** — stage 04 timestep arithmetic TG=1 does not reach | 21 s |
| `dach_gabriel_tg15` | **real X-Plane waypoints**: `BallTree` neighbourhoods, Gabriel edge criterion, `min-dist-vertices-km`, 1508 vertices / 3442 edges, `convex-sectors=0`, `airport-types`, **TG=15** | 170 s |

Full suite: **~5 min 15 s**, 553 files fingerprinted; **~6 min 20 s** with
`--shipped-check`. Every fixture runs all six stages plus
`05_transform_for_optimizer.py`; nothing is stubbed and no stage is skipped.

For a quick check during a refactor, `--only ea3x3` (25 s) exercises stages 00–05 on the
grid path; add `--only dach_gabriel_tg15` (~3 min) before touching graph generation.

Each fixture is regenerated into a freshly deleted directory. This is required, not
merely tidy: stage 04 rewrites `flights.csv` in place and is **not idempotent**, so
re-running over an existing tree produces different output (README convention ④).

### What is deliberately NOT covered

Be explicit about these before relying on a green run:

* **Stages 06 and 07** — `06_capacity_sweep.py`, `06_bluesky_converter.py`,
  `07_check_parsed_experiments_graph_connectedness.py`. Also `build_release_zips.py`,
  `convert_v1_hourly_capacities.py` and `expand_instances_for_benchmark.sh`.
* **`--scale` mode.** Every fixture drives demand through `--flight-flights`. The
  `ds_pairs` branch in `run_pipeline.main()` — used by `default_configs/world_20190615.json`
  — is never executed.
* **`--day-parity odd|even`.** All fixtures leave it at `all`, which short-circuits the
  hold-out code paths.
* **`--criterion rng`.** Both graph fixtures use `gabriel`.
* **`--neighbor-index bruteforce`** and **`--connectivity-method mst|greedy`**. All
  fixtures use `balltree` and `closest`.
* **TG=60**, and large regions (EUROPE, USA-MAINLAND at 19,610 vertices). Cost.
* **`--target-day` legacy single-day mode**; all fixtures use `--date-start/--date-end`.
* Numerical agreement across **different library versions**. The baseline was taken with
  Python 3.12.3, pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0, networkx 3.6.1.
  `requirements.txt` already warns that `sklearn.neighbors.BallTree` could in principle
  change the graph across versions. `compare.py` prints a warning when the recorded
  versions differ, but a version bump and a code change are not distinguishable from the
  manifests alone.

---

## The pieces

**`fingerprint.py`** — walks an output tree, writes a manifest of relative path →
SHA-256 of the normalised content. For every CSV it also records row and column counts,
the column names, a row-order-insensitive hash (`sha256_sorted`), a per-column hash
taken in canonical row order, and the first two rows as a sample. Stdlib + pandas.

**`compare.py`** — diffs two manifests. Reports added, removed and changed files; for a
changed CSV it names the columns whose hashes moved *and* the columns that did not, and
quotes example rows. Exit 0 = match, 1 = mismatch, 2 = usage error.

```bash
python tests/regression/compare.py tests/regression/baseline/ea3x3.json /path/to/candidate.json
```

**`run_fixtures.sh`** — regenerates every fixture from a clean directory, fails loudly on
any non-zero stage exit, fingerprints, and compares or refreshes the baseline.

**`baseline/`** — the committed manifests. Ground truth, captured from unmodified code.
Regenerate them **only** after deciding a change is intended, and say so explicitly in
the commit that does it.

### Inputs

Overridable by environment variable:

```
CSV_PATH=/home/thinklex/Documents/2026_not_for_dropbox/05_ASPaeroFlow_Data/ASPaeroFlow-DataGenerator/flightlist_20190601_20190630.csv
NAVDIR=/home/thinklex/Documents/2026_not_for_dropbox/05_ASPaeroFlow_Data/ASPaeroFlow-DataGenerator/test_navpoints
SCRATCH=<somewhere outside the repo>
PYTHON=python
```

`test_navpoints/` is **not** in this repository — it lives beside the flight list in the
external data directory. Only `dach_gabriel_tg15` needs it.

`PYTHON` must be an interpreter named `python` on `PATH`: `run_pipeline.py` spawns its
stages as `["python", "<stage>.py", …]`, not `sys.executable`, so a virtualenv that only
provides `python3` will run the stages under the wrong interpreter.

---

## Proof that the harness catches a real change

`02_graph_generator.py:112` was temporarily changed from

```python
EARTH_R_M = 6371008.8  # mean Earth radius (meters)
```

to `6371008.9` — a relative perturbation of 1.6 × 10⁻⁸ — and `--only ea3x3` was re-run.
The harness exited 1 and reported 7 changed files, naming the affected column in each:

```
--- CHANGED (7) ---
  ~ [CONTENT] parsed/30-0-EAST-ASIA-3x3-V2/0000010_SEED13/graph_edges.csv
    columns that changed : dist_m
    columns unchanged    : source, target
    - baseline sample : 0,1,1351188.334
    + candidate sample: 0,1,1351188.356
  ~ [CONTENT] unparsed/30-0-EAST-ASIA-3x3-V2/navgraph/edges.csv
    columns that changed : D
    columns unchanged    : V0, V1
```

The constant was then restored and `git diff` on the pipeline scripts confirmed empty;
re-running the fixture returned `RESULT: MATCH`, exit 0.
