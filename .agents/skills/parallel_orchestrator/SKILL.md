---
name: parallel_orchestrator
description: Master agent that manages, schedules, and executes all specialized agent skills (ingestion, thermal lag analysis, hydraulic advisory, hotspot forecasting, and agile testing) sequentially or in parallel.
---

# Master Parallel Orchestrator Skill

## Goal
Manage the complete multi-agent lifecycle, coordinating data handoffs and parallel task execution across all skills.

## Execution
Run the master parallel orchestrator:
`python .agents/skills/parallel_orchestrator/orchestrate_parallel.py --basin oder_river --mode full`

Verify that all agent outputs are generated and validated in `data/orchestrator_execution_manifest.json`.
