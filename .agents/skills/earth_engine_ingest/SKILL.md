---
name: earth_engine_ingest
description: Fetches cloud-masked Landsat 8/9 imagery, computes NDWI, LST in Celsius, and FAI/NDVI. Use when downloading or extracting satellite data for a lake.
---

# Earth Engine Ingestion Skill

## Goal
Extract calibrated water surface temperature and algal bloom indices over an input basin.

## Execution
Run the bundled script with the target bounding box and date range:
`python .agents/skills/earth_engine_ingest/run.py --bbox <min_lon,min_lat,max_lon,max_lat> --start <YYYY-MM-DD> --end <YYYY-MM-DD>`

Verify that `data/lake_timeseries.csv` is populated with `date`, `lst_c`, `ndvi`, and `ndwi`.
