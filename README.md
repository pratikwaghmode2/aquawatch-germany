# 🌊 AquaWatch EO: Germany Freshwater Algae & Hydraulic Early Warning System

> **EIT Water Hackathon Munich 2026** — *Challenge 4: Earth Observation for Freshwater Quality & Hydraulic Risk Management*

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=pratikwaghmode2/aquawatch-germany&branch=main&mainModule=app.py)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AquaWatch EO is an operational decision-support dashboard for environmental ministries, water utilities, and transboundary river basin authorities in Germany. It integrates multi-annual satellite Earth Observation (Landsat 8/9, Sentinel-2), thermal dynamics, and hydraulic operational levers to detect, forecast, and proactively mitigate toxic cyanobacterial Harmful Algal Blooms (HABs).

---

## 🌟 Key Features

1. **Interactive Germany Basin Map (10 Stations)**
   - **5 Major Rivers:** Rhine, Elbe, Danube, Oder, Weser
   - **5 Critical Lakes/Reservoirs:** Lake Constance (Bodensee), Lake Müritz, Lake Chiemsee, Lake Starnberg, Edersee Reservoir
   - Real-time risk status indicators (Normal, Watch, Warning, Emergency) with coordinates across all German federal states.

2. **Synchronized 4-Parameter Time Series**
   - Clean, uncluttered dual-axis telemetry visualization:
     - 🌡️ **Water Temperature (°C)**
     - 💧 **Water Supply / Inflow ($m^3/s$)**
     - 🏭 **Water Demand / Abstraction ($m^3/s$)**
     - 🟢 **Algae Formation (NDVI / FAI Biomass Index)**
   - Normalized comparison mode & rolling trendlines.

3. **Interactive "What-If" Climate & Hydraulic Scenario Simulator**
   - Live intervention testing with real-time feedback:
     - 🌡️ **Climate Warming ($\Delta T$):** $+0.0^\circ\text{C}$ to $+5.0^\circ\text{C}$
     - 🌊 **Weir Flushing Pulse ($+Q_{\text{flush}}$):** $0$ to $+100\,m^3/s$
     - 🚫 **Agricultural Abstraction Moratorium ($-\text{Demand}\%$):** $0\%$ to $-50\%$
   - One-click presets: *Heatwave Worst Case*, *Maximum Ecological Flushing*, *Extreme Drought*, and *Normal Baseline*.
   - Live mitigation scorecard showing dynamic changes in bloom risk, peak biomass, and water balance ratio.

4. **Multi-Annual Historical Coverage (2021–2025)**
   - 5 years of daily seasonal observations (April–October) capturing summer heatwave peaks and autumn recovery.

---

## 🚀 Quickstart & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/aquawatch-eo-germany.git
cd aquawatch-eo-germany
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ 1-Click Free Cloud Deployment (Streamlit Community Cloud)

1. Fork or push this repository to your GitHub account.
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
3. Click **"New app"**, select your repository, branch (`main`), and set Main file path to `app.py`.
4. Click **"Deploy"**! You will receive a permanent, free public URL (e.g. `https://aquawatch-germany.streamlit.app`) with no passwords or timeouts.

---

## 📂 Project Structure

```text
├── app.py                      # Main Streamlit Dashboard UI & Simulator
├── core/
│   ├── basin_manager.py        # 10 German Basin profiles & multi-annual data generator
│   ├── thermal_dynamics.py     # Degree Heating Days (DHD) & thermal lag cross-correlation
│   ├── ml_models.py            # Random Forest & Gradient Boosting bloom predictors
│   ├── forecasting_engine.py   # 30m spatial hotspot risk propagation
│   └── resolution_adapter.py   # Adaptive sub-pixel river water extraction
├── data/
│   └── basins/                 # 5-year multi-annual benchmark datasets (CSV)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 👥 Hackathon Team & Acknowledgements
- Developed for **EIT Water Hackathon Munich 2026**
- *Challenge 4: Freshwater Algae EO & Early Warning System*
