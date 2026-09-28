---
name: risk_forecaster
description: Simulates forward-looking What-If scenarios under consecutive hot day exceedance and climate warming deltas, generating 30m spatial hotspot risk maps. Use for predicting future bloom development.
---

# Risk Forecasting & Hotspot Mapping Skill

## Goal
Simulate future algal bloom proliferation under specified thermal pre-conditions and map intervention hotspots.

## Execution
Run the scenario simulator:
`python .agents/skills/risk_forecaster/forecast.py --basin oder_river --threshold 22.0 --consecutive-days 5 --delta 2.0 --output data/forecast_risk_map.json`

Verify that `data/forecast_risk_map.json` is generated with 14-day risk trajectories and hotspot coordinates.
