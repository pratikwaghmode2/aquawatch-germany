#!/usr/bin/env python3
"""
Master Parallel Orchestrator Agent
Manages and coordinates all specialized skills:
1. earth_engine_ingest (EO Satellite Data & River Skeleton)
2. thermal_correlation (Heat Accumulation & 3-5 Day Lag)
3. hydraulic_advisory (Weir Flushing & German UBA Directives)
4. risk_forecaster (Parallel Multi-Scenario Climate Simulations)
5. agile_tester (5-Sprint Automated Agile QA Gate)
Outputs to: data/orchestrator_execution_manifest.json
"""

import os
import sys
import time
import json
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args():
    parser = argparse.ArgumentParser(description="Master Parallel Orchestrator for AquaWatch EO multi-agent workflow.")
    parser.add_argument("--basin", type=str, default="oder_river", help="Target basin key (default: oder_river)")
    parser.add_argument("--bbox", type=str, default="14.40,52.20,14.75,52.65", help="River bounding box")
    parser.add_argument("--start", type=str, default="2021-06-01", help="Start date")
    parser.add_argument("--end", type=str, default="2025-08-31", help="End date")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "parallel-scenarios", "test-only"], help="Execution mode")
    return parser.parse_args()


class ParallelOrchestrator:
    def __init__(self, basin: str, bbox: str, start_date: str, end_date: str):
        self.basin = basin
        self.bbox = bbox
        self.start_date = start_date
        self.end_date = end_date
        self.manifest = {
            "orchestrator_version": "2.0.0",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_basin": basin,
            "target_bbox": bbox,
            "agents_executed": [],
            "status": "INITIALIZING"
        }

    def execute_agent_task(self, agent_name: str, script_path: str, args_list: list):
        """Execute a specialized agent script in its subprocess sandbox."""
        cmd = [sys.executable, script_path] + args_list
        start_t = time.time()
        print(f"\n[ORCHESTRATOR] >>> Dispatching Agent: [{agent_name}]")
        print(f"               Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = round(time.time() - start_t, 2)
        success = (result.returncode == 0)

        record = {
            "agent": agent_name,
            "command": " ".join(cmd),
            "duration_sec": duration,
            "status": "SUCCESS" if success else "FAILED",
            "exit_code": result.returncode
        }
        self.manifest["agents_executed"].append(record)

        if success:
            print(f"[ORCHESTRATOR] <<< Agent [{agent_name}] completed successfully in {duration}s.")
        else:
            print(f"[ORCHESTRATOR] [!] Agent [{agent_name}] failed with code {result.returncode}:")
            print(result.stderr)

        return success, result.stdout

    def run_parallel_climate_scenarios(self):
        """Execute 3 'What-If' climate scenarios in parallel using ThreadPoolExecutor."""
        print("\n[ORCHESTRATOR] >>> Launching Parallel Multi-Scenario Simulation (3 Scenarios Concurrently)...")
        scenarios = [
            {"name": "Mild Warming (+1.0°C)", "delta": "1.0", "streak": "3", "wind": "moderate", "out": "data/scenario_mild.json"},
            {"name": "Moderate Heatwave (+2.0°C)", "delta": "2.0", "streak": "5", "wind": "stagnant", "out": "data/scenario_moderate.json"},
            {"name": "Extreme Heatwave (+3.5°C)", "delta": "3.5", "streak": "8", "wind": "stagnant", "out": "data/scenario_extreme.json"}
        ]

        def run_single(sc):
            cmd = [
                sys.executable,
                ".agents/skills/risk_forecaster/forecast.py",
                "--basin", self.basin,
                "--delta", sc["delta"],
                "--consecutive-days", sc["streak"],
                "--wind", sc["wind"],
                "--output", sc["out"]
            ]
            t0 = time.time()
            res = subprocess.run(cmd, capture_output=True, text=True)
            return sc["name"], res.returncode == 0, round(time.time() - t0, 2)

        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_single, sc) for sc in scenarios]
            for f in as_completed(futures):
                name, ok, dur = f.result()
                results.append((name, ok, dur))
                status_str = "SUCCESS" if ok else "FAILED"
                print(f"[ORCHESTRATOR]     Worker [{name}]: {status_str} in {dur}s")

        return all(r[1] for r in results)

    def run_full_pipeline(self):
        t_start = time.time()
        print("=" * 70)
        print("  AquaWatch EO: Master Parallel Orchestrator")
        print(f"  Target: {self.basin.upper()} | Reach BBox: {self.bbox}")
        print("=" * 70)

        # Stage 1: EO Satellite Ingestion Agent
        ok1, _ = self.execute_agent_task(
            "earth_engine_ingest",
            ".agents/skills/earth_engine_ingest/run.py",
            ["--bbox", self.bbox, "--start", self.start_date, "--end", self.end_date]
        )
        if not ok1:
            print("[ORCHESTRATOR] Aborting due to ingestion failure.")
            return False

        # Stage 2: Thermal Dynamics & Biological Lag Analyst
        ok2, _ = self.execute_agent_task(
            "thermal_correlation",
            ".agents/skills/thermal_correlation/analyze.py",
            []
        )
        if not ok2:
            print("[ORCHESTRATOR] Aborting due to thermal analysis failure.")
            return False

        # Stage 3: Transboundary Hydraulic Advisory Planner
        ok3, _ = self.execute_agent_task(
            "hydraulic_advisory",
            ".agents/skills/hydraulic_advisory/advisory.py",
            []
        )

        # Stage 4: Parallel Multi-Scenario Hotspot Forecaster
        ok4 = self.run_parallel_climate_scenarios()

        # Stage 5: Agile QA & Product Verification Tester
        ok5, _ = self.execute_agent_task(
            "agile_tester",
            ".agents/skills/agile_tester/test_suite.py",
            []
        )

        total_duration = round(time.time() - t_start, 2)
        all_passed = ok1 and ok2 and ok3 and ok4 and ok5

        self.manifest["total_duration_sec"] = total_duration
        self.manifest["status"] = "ALL_AGENTS_COMPLETED_SUCCESSFULLY" if all_passed else "COMPLETED_WITH_ERRORS"
        self.manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        manifest_path = "data/orchestrator_execution_manifest.json"
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

        print("\n" + "=" * 70)
        print(f"  MASTER ORCHESTRATION COMPLETE ({total_duration}s)")
        print(f"  Status: {self.manifest['status']}")
        print(f"  [OK] Saved Execution Manifest: {manifest_path}")
        print("=" * 70 + "\n")
        return all_passed


def main():
    args = parse_args()
    orch = ParallelOrchestrator(
        basin=args.basin,
        bbox=args.bbox,
        start_date=args.start,
        end_date=args.end
    )
    if args.mode == "test-only":
        orch.execute_agent_task("agile_tester", ".agents/skills/agile_tester/test_suite.py", [])
    elif args.mode == "parallel-scenarios":
        orch.run_parallel_climate_scenarios()
    else:
        orch.run_full_pipeline()


if __name__ == "__main__":
    main()
