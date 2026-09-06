# OurAirports snapshot (patched)

`airports.csv` is the OurAirports database snapshot used to build every V2 dataset.
It is tracked here rather than downloaded at run time for two reasons.

**1. It is hand-patched.** OurAirports is a contemporary database; our flight data is
June 2019. Airports that have since closed have had their ICAO code cleared upstream,
so the generator drops them and the historically dominant OD pairs involving them
disappear from the generated data. Berlin Tegel (EDDT) was the case that surfaced this:
upstream it now carries `ident=DE-0876` with an empty `icao_code`, and because
`DE-0876` is not ICAO-shaped, even the per-row ICAO fallback cannot recover it.

Patched back in (`ident` and `icao_code` restored):

| ICAO | upstream ident | note |
|------|----------------|------|
| EDDT | DE-0876 | Berlin Tegel, closed 2020, `type` also restored to `large_airport` |
| ETNJ | DE-0882 | |
| EDEF | DE-0899 | |

**2. Reproducibility.** Re-downloading OurAirports would silently change the airport set
of every regenerated instance. Pin this file; its MD5 is in `airports.csv.md5`.

The other two inputs (`flightlist_20190601_20190630.csv` from the OpenSky COVID-19
dataset, and `test_navpoints/` from BlueSky) are unmodified upstream data and are not
tracked here for size reasons.
