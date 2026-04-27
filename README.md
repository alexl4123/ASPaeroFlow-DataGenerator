# ASPaeroFlow Data Generator

**ASPaeroFlow Data Generator** is a Python-based toolkit to create realistic air traffic scenarios for research and simulations. It builds synthetic flight schedules and airspace structures from historical data (e.g. OpenSky Network), and prepares inputs for strategic Air Traffic Flow and Capacity Management (ATFCM) analyses. In the overall **ASPaeroFlow pipeline**, this data generator is responsible for producing the traffic scenarios – including flights, navigation points, and sector definitions – that can be fed into optimization models or simulation tools.

## Features

- **Synthetic Traffic Scenario Generation:** Construct a full day of flights based on statistical models derived from real data (e.g. OpenSky flight records).
- **OpenSky Data Integration:** Build probabilistic models of departures, destinations, durations, and turnarounds from OpenSky flight listings.
- **Navpoint Graph via Voronoi Partitioning:** Generate a navigation graph of waypoints and airports using geometric criteria (e.g., RNG or Gabriel Graph).
- **Sector Generation and Capacities:** Partition airspace into sectors with configurable size and assign realistic capacities to each.
- **Configurable Parameters:** Full control over flights, traffic scaling, navgraph shape, regions, and temporal resolution.
- **BlueSky Simulation Compatibility:** Convert generated scenarios to `.scn` files for playback in BlueSky ATC simulator.
- **Optimizer-Ready Output:** Output flights, sectors, and navgraph in CSV format for downstream optimization models.

## Installation

**Recommended platform:** Linux (tested). Basic compatibility exists for macOS and Windows (see below).

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YourUsername/ASPaeroFlow-DataGenerator.git
   cd ASPaeroFlow-DataGenerator
   ```

2. **Install Dependencies:**
   ```bash
   pip install numpy pandas networkx tqdm
   ```

3. **Download Required Data:**
   - [OpenSky COVID19 Flight Dataset (cc-by)](https://zenodo.org/records/5815448) →  Place e.g., `flightlist_20190601_20190630.csv` in `./data/`.
   - [OurAirports `airports.csv`](https://ourairports.com/data/airports.csv) → Place in `./ourairports/`.
   - Optionally: `earth_fix.dat` and `earth_nav.dat` from X-Plane → Place in `./test_navpoints/`.

4. **Platform Notes:**
   - macOS: Should work if Python and dependencies installed.
   - Windows: Use WSL or adjust paths and shell calls accordingly.

## Usage

Run the full scenario generator:

```bash
python run_pipeline.py \
  --csv-path data/flightlist_20190601_20190630.csv \
  --target-day 2019-06-15 \
  --config configs/dach_region.json \
  --experiment-name DACH-2019-06-15 \
  --scale 0.5 \
  --seed 42
```

### Common Options

| Option | Description |
|--------|-------------|
| `--csv-path` | Path to OpenSky CSV (or folder of them) |
| `--target-day` | Day to simulate (`YYYY-MM-DD`) |
| `--scale` or `--flights` | Scale traffic by factor or use exact flight count |
| `--experiment-name` | Folder name under `unparsed_experiment_data/` |
| `--grid-navpoints true` | Use synthetic navpoint grid instead of real fixes |
| `--sector-default-navaid-size N` | Approx. navpoints per sector (for grouping) |
| `--time-granularity` | Time slot size (e.g. 4 = 15 min) |
| `--seed` | Random seed for reproducibility |

See `--help` for more.

## Outputs

Generated in `unparsed_experiment_data/<experiment-name>/`

- `model/`: Statistical models (departure rates, durations, etc.)
- `navgraph/`: 
  - `vertices.csv`: Waypoints and airports
  - `edges.csv`: Navgraph edges
  - `sectors.csv`: Sector capacities
  - `navaid_sector_assignment.csv`: Sector grouping (if used)
- `DATA_S*/`: 
  - `flights.csv`, `aircrafts.csv`
  - `filed_flights.csv`: Flight plans (waypoints + timestamps)
  - `manifest.json`: Summary of the run

Optional:
- `experiment_data/`: Transformed version for solvers
- `bluesky/`: `.scn` file for BlueSky simulator

## BlueSky Conversion

```bash
python 06_bluesky_converter.py \
  --data-dir unparsed_experiment_data/DACH-2019-06-15/DATA_S0p5_42 \
  --navgraph-dir unparsed_experiment_data/DACH-2019-06-15/navgraph \
  --time-granularity 60
```

## Repository Structure

| Script | Purpose |
|--------|---------|
| `00_model_generation_script_refactored.py` | Learns statistical traffic model |
| `01_data_generation_script_refactored.py` | Samples synthetic flights |
| `02_graph_generator.py` | Builds navgraph (grid or real fixes) |
| `03_sector_capacity_generator.py` | Assigns sector IDs and capacities |
| `04_simplified_filed_flight_plan_generator.py` | Generates routed flight plans |
| `05_transform_for_optimizer.py` | (Optional) Converts for solver input |
| `06_bluesky_converter.py` | (Optional) Creates BlueSky `.scn` file |
| `07_check_parsed_experiments_graph_connectedness.py` | (Debugging) Checks graph connectivity |

## Development

- **Custom Regions:** Use `--config` JSON with a polygon to restrict to a geographic area.
- **Navgraph Tweaks:** Adjust RNG vs Gabriel Graph, or grid shape and spacing.
- **Sectorization Control:** Use BFS grouping or convex grouping mode.
- **Multiple Scenarios:** Use `--flight-scales` and `--flight-seeds` as lists.
- **Testing:** Run small scenarios first and validate route feasibility.

## License

MIT license with attribution (for details see license.md).

---

*This README describes the ASPaeroFlow-DataGenerator for air traffic research. Outputs are compatible with BlueSky, optimization models, or ASP-based planning.*
*This README was created with the help of generative AI.*

