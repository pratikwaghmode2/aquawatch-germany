---
name: thermal_correlation_analyst
description: Calculates Degree Heating Days and cross-correlates water temperature lag against bloom development. Use when analyzing temperature trends or predicting bloom risk.
---

# Thermal Correlation Skill

## Goal
Analyze cumulative thermal thresholds (DHD) and output a risk assessment.

## Execution
Execute the analysis script:
`python .agents/skills/thermal_correlation/analyze.py`

Verify that `data/risk_assessment.json` is generated containing `current_lst_c`, `dhd_7d`, and `risk_level`.
