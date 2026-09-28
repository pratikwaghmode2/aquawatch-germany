#!/usr/bin/env python3
"""
Hydraulic Advisory & Water Authority Action Brief Generator
Translates satellite EO thermal indices and bloom risk assessments into
concrete hydraulic management mandates:
- Weir discharge / flushing pulse volume (m³/s)
- Selective deep-water withdrawal
- Drinking water intake protection protocols
- Public health bathing advisories
Outputs to: data/advisory_brief.md
"""

import os
import sys
import json
import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Generate hydraulic advisory brief for water authorities.")
    parser.add_argument("--assessment", type=str, default="data/risk_assessment.json", help="Path to risk assessment JSON")
    parser.add_argument("--output", type=str, default="data/advisory_brief.md", help="Path to output markdown brief")
    return parser.parse_args()


def generate_advisory_markdown(data: dict) -> str:
    risk = data.get("risk_level", "LOW")
    date_str = data.get("assessment_date", datetime.now().strftime("%Y-%m-%d"))
    lst = data.get("current_lst_c", 20.0)
    dhd = data.get("dhd_7d", 0.0)
    streak = data.get("consecutive_days_over_threshold", 0)
    thresh = data.get("threshold_c", 22.0)
    lag = data.get("optimal_lag_days", 4)
    corr = data.get("peak_lag_correlation", 0.65)

    if risk == "CRITICAL":
        alert_badge = "🔴 CRITICAL HAZARD — EMERGENCY PROTOCOL ACTIVE"
        weir_mandate = "Initiate emergency hydraulic flushing pulse: Increase weir/barrage discharge to **45–60 m³/s** to elevate river flow velocity (>0.4 m/s), break thermal stagnation in groyne fields, and flush toxic microalgae (*Prymnesium parvum* / cyanobacteria) downstream."
        intake_mandate = "MANDATORY: Divert municipal and industrial riverbank filtration intakes to deep groundwater reserves. Suspend direct river extraction; engage Powdered Activated Carbon (PAC) at 20 mg/L."
        public_mandate = "IMMEDIATE RIVER RECREATION & FISHING BAN: Prohibit bathing, angling, and pet water access. Notify the German-Polish Bilateral Oder Commission (IKSO/MKOO) for synchronized transboundary emergency response."
        sampling_mandate = "Deploy mobile environmental patrol vessels for 12-hour grab sampling (cell counts, conductivity/salinity, and toxic prymnesin/microcystin bioassays)."
    elif risk == "WARNING":
        alert_badge = "🟠 ELEVATED WARNING — PRECAUTIONARY INTERVENTION"
        weir_mandate = "Prepare controlled hydraulic release: Adjust upstream reservoir releases to **25–35 m³/s** over the next 48 hours to prevent river stagnation and thermocline formation."
        intake_mandate = "Heighten riverbank wellfield monitoring. Restrict upstream industrial saline wastewater discharges to prevent salinity-temperature synergy."
        public_mandate = "POST ADVISORY NOTICES: Warn public against contact with discolored river stretches, backwaters, and foam accumulation."
        sampling_mandate = "Increase cross-sectional river profiling to 48-hour cadence at key border gauge stations."
    elif risk == "WATCH":
        alert_badge = "🟡 ADVISORY WATCH — THERMAL SURVEILLANCE"
        weir_mandate = "Maintain nominal ecological baseline flow (15–20 m³/s). Monitor upstream reservoir storage levels in Poland and Germany."
        intake_mandate = "Standard drinking water monitoring active. Calibrate continuous automated river water quality probes."
        public_mandate = "Unrestricted river recreation; normal navigation and bathing permitted."
        sampling_mandate = "Standard weekly baseline sampling. Re-assess with next Landsat 8/9 overpass."
    else:
        alert_badge = "🟢 LOW RISK / CLEAR WATER — ROUTINE OPERATION"
        weir_mandate = "Standard seasonal river discharge schedule."
        intake_mandate = "Routine riverbank filtration operations."
        public_mandate = "Unrestricted water use."
        sampling_mandate = "Bi-weekly seasonal surveillance."

    md = f"""# 🌊 Water Authority Hydraulic & Ecological Advisory Brief
**Report Generated:** {date_str} | **EO Pipeline:** Landsat 8/9 Level-2 Analysis  
**Operational Status:** {alert_badge}

---

## 1. Executive Limnological Summary
- **Current Lake Surface Water Temperature (LSWT):** `{lst}°C`
- **7-Day Cumulative Heat Load (DHD > 20°C):** `{dhd} °C·days`
- **Consecutive Days Above Critical Threshold ({thresh}°C):** `{streak} days`
- **Observed Biological Bloom Lag:** `{lag} days` (Cross-correlation `r = {corr:.2f}`)
- **Determined Risk Classification:** **{risk}**

> **Biophysical Assessment:**  
> {data.get("summary", "Thermal accumulation exceeds seasonal baseline, accelerating cyanobacteria cell division and vertical buoyancy regulation.")}

---

## 2. Hydraulic & Engineering Directives

### A. Weir & Dam Discharge Management
- **Action:** {weir_mandate}
- **Objective:** Destabilize lake epilimnion stratification, increase flushing velocity in sheltered embayments, and mitigate surface scum accumulation.

### B. Municipal Drinking Water Intake Protection
- **Action:** {intake_mandate}
- **Objective:** Prevent toxic cyanotoxin ingress (microcystin, cylindrospermopsin) into municipal water distribution networks.

### C. Public Health & Recreation Advisory
- **Action:** {public_mandate}
- **Regulatory Standard:** Conforms to WHO Cyanobacteria Recreational Water Guidelines and EU Bathing Water Directive (2006/7/EC).

### D. Limnological Field Verification
- **Action:** {sampling_mandate}

---

## 3. Forward Risk Trajectory & Next Satellite Overpass
- **Forecast Horizon:** Next 7–14 days.
- **Critical Trigger Condition:** If temperatures remain above `{thresh}°C` for more than 4 consecutive days with wind speed `< 2.5 m/s`, expect exponential surface scum proliferation.
- **Next Landsat Observation:** Scheduled in approximately 3–4 days (Landsat 8/9 combined 8-day orbit cadence).

*Authorized by: Basin Hydraulic Control & Environmental Safety Directorate*
"""
    return md


def main():
    args = parse_args()
    print(f"[hydraulic_advisory] Reading assessment from: {args.assessment}")

    if not os.path.exists(args.assessment):
        print(f"[!] Risk assessment file {args.assessment} not found. Run thermal_correlation first!", file=sys.stderr)
        sys.exit(1)

    with open(args.assessment, "r", encoding="utf-8") as f:
        data = json.load(f)

    md_content = generate_advisory_markdown(data)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[hydraulic_advisory] [OK] Advisory brief generated successfully.")
    print(f"[hydraulic_advisory]   Status: [{data.get('risk_level', 'UNKNOWN')}]")
    print(f"[hydraulic_advisory] [OK] Saved to {args.output}")


if __name__ == "__main__":
    main()
