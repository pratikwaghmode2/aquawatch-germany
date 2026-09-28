"""
AquaWatch EO: Interactive Germany Freshwater Map & 'What-If' Hydraulic Simulator
Interactive Geospatial Map of Germany with River & Lake Early Warning Beacons,
Coupled with the 4-Line Nexus Graph & Interactive 'What-If' Climate & Hydraulic Scenario Simulator.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats

from core.basin_manager import BASIN_PROFILES, generate_basin_timeseries

# Page configuration
st.set_page_config(
    page_title="AquaWatch EO | Germany Map & What-If Simulator",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #06b6d4; margin-bottom: 2px; }
    .sub-title { font-size: 1.05rem; color: #94a3b8; margin-bottom: 16px; }
    .badge-river { background: #082f49; color: #38bdf8; border: 1px solid #0284c7; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; }
    .badge-lake { background: #022c22; color: #34d399; border: 1px solid #059669; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; }
    .kpi-container { background: #0f172a; border-radius: 8px; padding: 12px 16px; border: 1px solid #1e293b; }
    .sim-card { background: #1e1b4b; border: 1px solid #6366f1; border-radius: 8px; padding: 14px; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.title("🌊 AquaWatch Germany")
st.sidebar.caption("Earth Observation for Freshwater Systems")
st.sidebar.markdown("---")

preset_options = {
    "🌊 Oder River (Brandenburg / Poland Border)": "oder_river",
    "🏞️ Lake Constance (Bodensee - Baden-Württ. / Bavaria)": "lake_constance",
    "🏞️ Lake Müggelsee (Berlin Spree Lowland)": "lake_mueggelsee",
    "🌊 Middle Rhine Reach (Mittelrhein / Kaub)": "rhine_river",
    "🌊 Middle Elbe Reach (Magdeburg / Saxony-Anhalt)": "elbe_river",
    "🌊 Bavarian Danube Reach (Donau / Regensburg)": "danube_river",
    "🏞️ Lake Starnberg (Starnberger See - Bavaria)": "lake_starnberg",
    "🏞️ Lake Chiemsee ('Bavarian Sea' - Bavaria)": "lake_chiemsee",
    "🏞️ Bautzen Reservoir (Talsperre Bautzen - Saxony)": "bautzen_reservoir",
    "🏞️ Lake Steinhude (Steinhuder Meer - Lower Saxony)": "steinhuder_meer"
}

selected_preset_label = st.sidebar.selectbox("Select Station to Inspect:", list(preset_options.keys()))
active_basin_key = preset_options[selected_preset_label]
profile = BASIN_PROFILES[active_basin_key]

badge_class = "badge-river" if profile["type"] == "River" else "badge-lake"
st.sidebar.markdown(f"""
<div style="margin-top: 6px;">
    <span class="{badge_class}">{profile['type'].upper()} SYSTEM</span>
</div>
<div style="font-size: 0.85rem; color: #94a3b8; margin-top: 8px;">
    <strong>Location:</strong> {profile.get('state', 'Germany')}<br>
    <strong>Morphology:</strong> {profile['scale_desc']}<br>
    <strong>Jurisdiction:</strong> {profile['jurisdiction']}<br>
    <strong>Primary Taxa:</strong> <em>{profile['primary_taxa']}</em>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Observation Timeline")

timeline_options = {
    "📅 2025 (Active Summer Season)": 2025,
    "📅 2024 (Thunderstorm Flushes)": 2024,
    "📅 2023 (Global Record Warmth)": 2023,
    "🚨 2022 (Historic Heatwave & Oder Disaster)": 2022,
    "📅 2021 (Cool Summer Baseline)": 2021,
    "🌐 All 5 Years (2021–2025 Continuous)": "all"
}

selected_timeline_label = st.sidebar.selectbox("Timeline Window:", list(timeline_options.keys()))
active_timeline_val = timeline_options[selected_timeline_label]

st.sidebar.markdown("---")
st.sidebar.subheader("Visualization Layout")
view_layout = st.sidebar.radio(
    "Display Mode:",
    ["🗺️ Map & Station Graph (Integrated)", "🌐 Fullscreen Germany Map", "📈 Fullscreen 4-Line Graph"],
    index=0
)

# -------------------------------------------------------------
# SESSION STATE INITIALIZATION FOR WHAT-IF SIMULATOR
# -------------------------------------------------------------
if "sim_temp" not in st.session_state:
    st.session_state["sim_temp"] = 0.0
if "sim_flush" not in st.session_state:
    st.session_state["sim_flush"] = 0.0
if "sim_restr" not in st.session_state:
    st.session_state["sim_restr"] = 0

# -------------------------------------------------------------
# DATA ENGINE: LOAD DATA FOR ALL 10 GERMAN STATIONS
# -------------------------------------------------------------
DEFAULT_COORDS = {
    "oder_river": (52.345, 14.550, "Brandenburg (German-Polish Border)"),
    "lake_constance": (47.636, 9.380, "Baden-Württemberg / Bavaria"),
    "lake_mueggelsee": (52.433, 13.630, "Berlin"),
    "bautzen_reservoir": (51.215, 14.453, "Saxony"),
    "danube_river": (48.950, 12.300, "Bavaria (Regensburg / Deggendorf)"),
    "lake_starnberg": (47.900, 11.310, "Bavaria (near Munich)"),
    "lake_chiemsee": (47.880, 12.400, "Bavaria ('Bavarian Sea')"),
    "rhine_river": (50.150, 7.720, "Rhineland-Palatinate / Hesse"),
    "elbe_river": (52.130, 11.630, "Saxony-Anhalt"),
    "steinhuder_meer": (52.480, 9.320, "Lower Saxony")
}

station_records = []
all_dfs = {}

for key, prof in BASIN_PROFILES.items():
    csv_file = f"data/basins/{key}_5years.csv"
    if not os.path.exists(csv_file):
        df_b = generate_basin_timeseries(key)
    else:
        df_b = pd.read_csv(csv_file)
        
    all_dfs[key] = df_b
    
    # Filter by timeline
    if active_timeline_val == "all":
        df_filtered = df_b.copy()
    else:
        df_filtered = df_b[df_b["year"] == active_timeline_val].copy()

    latest_row = df_filtered.iloc[-1]
    temp = float(latest_row["lst_c"])
    supply = float(latest_row["water_supply_m3s"])
    demand = float(latest_row["water_demand_m3s"])
    ndvi = float(latest_row["ndvi"])
    balance = supply - demand
    
    # Risk Classification
    if ndvi >= 0.22 or (temp >= 24.0 and balance < 0):
        status = "CRITICAL (Alert)"
        color = "#ef4444" # Red
    elif ndvi >= 0.12 or temp >= 22.0:
        status = "WARNING (Watch)"
        color = "#f97316" # Orange
    elif temp >= 20.5:
        status = "WATCH (Notice)"
        color = "#eab308" # Yellow
    else:
        status = "LOW (Clear)"
        color = "#10b981" # Green
        
    is_active = (key == active_basin_key)
    marker_size = 26 if is_active else (20 if prof.get("type", "Lake") == "River" else 18)
    
    def_geo = DEFAULT_COORDS.get(key, (51.16, 10.45, "Germany"))
    lat_val = float(prof.get("lat", def_geo[0]))
    lon_val = float(prof.get("lon", def_geo[1]))
    state_val = prof.get("state", def_geo[2])

    station_records.append({
        "key": key,
        "name": prof.get("name", key),
        "short_name": prof.get("name", key).split(" (")[0],
        "type": prof.get("type", "Freshwater"),
        "state": state_val,
        "lat": lat_val,
        "lon": lon_val,
        "temp": temp,
        "supply": supply,
        "demand": demand,
        "ndvi": ndvi,
        "balance": balance,
        "status": status,
        "color": color,
        "marker_size": marker_size,
        "is_active": is_active
    })

map_df = pd.DataFrame(station_records)

# Extract Active Basin Data
df_active_full = all_dfs[active_basin_key]
if active_timeline_val == "all":
    df_display = df_active_full.copy()
    display_title_suffix = "5-Year Continuous Timeline (2021–2025)"
else:
    df_display = df_active_full[df_active_full["year"] == active_timeline_val].copy()
    display_title_suffix = f"Summer {active_timeline_val} Season"

# -------------------------------------------------------------
# MAIN HEADER & NATIONAL OVERVIEW KPIS
# -------------------------------------------------------------
st.markdown('<div class="main-title">AquaWatch Germany: National Freshwater EO Map</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">Interactive Earth Observation & Hydraulic Monitoring Across German Rivers and Lakes | <strong>{display_title_suffix}</strong></div>', unsafe_allow_html=True)

# 4 National Summary Metric Cards
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.metric("📍 Monitored Systems", f"{len(map_df)} Stations", delta="5 Rivers & 5 Lakes Across Germany")

with kpi_col2:
    mean_nat_temp = float(map_df["temp"].mean())
    st.metric("🌡️ National Mean Water Temp", f"{mean_nat_temp:.1f} °C", delta=f"{mean_nat_temp - 20.0:+.1f} °C vs 20°C threshold", delta_color="inverse")

with kpi_col3:
    critical_count = int((map_df["status"].str.contains("CRITICAL|WARNING")).sum())
    st.metric("🚨 Systems on High Alert", f"{critical_count} / {len(map_df)}", delta=f"{critical_count} Elevated Hazard Hotspots", delta_color="inverse" if critical_count > 0 else "normal")

with kpi_col4:
    worst_deficit_row = map_df.loc[map_df["balance"].idxmin()]
    st.metric("⚖️ Max Abstraction Deficit", f"{worst_deficit_row['balance']:.1f} m³/s", delta=f"{worst_deficit_row['short_name']}", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# INTERACTIVE MAP OF GERMANY
# -------------------------------------------------------------
def build_germany_map(df_stations, active_key, sim_status=None, sim_color=None):
    fig_map = go.Figure()

    df_plot = df_stations.copy()
    if sim_status and sim_color:
        idx = df_plot[df_plot["key"] == active_key].index
        if len(idx) > 0:
            df_plot.loc[idx, "status"] = sim_status
            df_plot.loc[idx, "color"] = sim_color

    # 1. Base station markers
    fig_map.add_trace(go.Scattermap(
        lat=df_plot["lat"],
        lon=df_plot["lon"],
        mode="markers+text",
        marker=dict(
            size=df_plot["marker_size"],
            color=df_plot["color"],
            opacity=0.92
        ),
        text=df_plot["short_name"],
        textposition="top right",
        textfont=dict(size=12, color="#f8fafc", family="sans-serif"),
        customdata=df_plot[["name", "type", "state", "status", "temp", "supply", "demand", "ndvi", "balance"]],
        hovertemplate=(
            "<b style='font-size:14px;'>%{customdata[0]}</b><br>" +
            "Type: <b>%{customdata[1]}</b> | Region: <b>%{customdata[2]}</b><br>" +
            "Alert Tier: <b>%{customdata[3]}</b><br>" +
            "──────────────────────────────<br>" +
            "🌡️ Water Temp: <b>%{customdata[4]:.1f} °C</b><br>" +
            "💧 Water Supply: <b>%{customdata[5]:.1f} m³/s</b><br>" +
            "🚰 Water Demand: <b>%{customdata[6]:.1f} m³/s</b><br>" +
            "🌿 Algae Formation: <b>%{customdata[7]:.3f} NDVI</b><br>" +
            "⚖️ Flow Balance: <b>%{customdata[8]:+.1f} m³/s</b>" +
            "<extra></extra>"
        ),
        name="German Freshwater Stations"
    ))

    # 2. Glowing Halo on the Currently Active Station
    active_row = df_plot[df_plot["key"] == active_key].iloc[0]
    halo_color = sim_color if sim_color else "rgba(56, 189, 248, 0.35)"
    fig_map.add_trace(go.Scattermap(
        lat=[active_row["lat"]],
        lon=[active_row["lon"]],
        mode="markers",
        marker=dict(
            size=36,
            color=halo_color,
            symbol="circle"
        ),
        hoverinfo="skip",
        name="Active Selected Station"
    ))

    fig_map.update_layout(
        map=dict(
            style="carto-darkmatter",
            center=dict(lat=51.16, lon=10.45), # Center of Germany
            zoom=5.5
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_dark",
        height=520,
        showlegend=False
    )
    return fig_map


# -------------------------------------------------------------
# 4-PARAMETER GRAPH ENGINE (Temperature, Supply, Demand, Algae)
# -------------------------------------------------------------
def build_4parameter_graph(df_data, basin_prof, line_typ, scale_typ, sim_t, sim_q, sim_r):
    x_idx = np.arange(len(df_data))
    
    # 1. Base arrays
    base_t_arr = df_data["lst_c"]
    base_s_arr = df_data["water_supply_m3s"]
    base_d_arr = df_data["water_demand_m3s"]
    base_a_arr = df_data["ndvi"]

    # 2. Apply What-If Simulation Physics
    is_sim = (sim_t != 0.0 or sim_q > 0.0 or sim_r > 0)
    
    sim_t_arr = base_t_arr + sim_t
    sim_s_arr = base_s_arr + sim_q
    demand_heat = 1.0 + np.maximum(0.0, sim_t * 0.04)
    demand_restr = 1.0 - (sim_r / 100.0)
    sim_d_arr = base_d_arr * demand_heat * demand_restr

    # Algae biophysical response: thermal division acceleration vs hydraulic flushing washout
    algae_thermal = 1.0 + (np.maximum(0.0, sim_t) / 14.0)
    flushing_washout = np.clip((sim_q / 45.0) * 0.48, 0.0, 0.52)
    sim_a_arr = np.clip((base_a_arr * algae_thermal) * (1.0 - flushing_washout), -0.08, 0.65)

    # Linear Regression Trendlines on active data
    slope_t, intercept_t, _, _, _ = stats.linregress(x_idx, sim_t_arr)
    t_lin = slope_t * x_idx + intercept_t
    t_smooth = sim_t_arr.rolling(7, min_periods=1).mean()

    slope_s, intercept_s, _, _, _ = stats.linregress(x_idx, sim_s_arr)
    s_lin = slope_s * x_idx + intercept_s
    s_smooth = sim_s_arr.rolling(7, min_periods=1).mean()

    slope_d, intercept_d, _, _, _ = stats.linregress(x_idx, sim_d_arr)
    d_lin = slope_d * x_idx + intercept_d
    d_smooth = sim_d_arr.rolling(7, min_periods=1).mean()

    slope_a, intercept_a, _, _, _ = stats.linregress(x_idx, sim_a_arr)
    a_lin = slope_a * x_idx + intercept_a
    a_smooth = sim_a_arr.rolling(7, min_periods=1).mean()

    tag = " (Simulated)" if is_sim else ""
    if line_typ == "📈 Trendlines (OLS)":
        s_v, d_v, t_v, a_v = s_lin, d_lin, t_lin, a_lin
        s_lbl = f"💧 Water Supply Trend{tag} ({slope_s * len(df_data):+.1f} m³/s)"
        d_lbl = f"🚰 Water Demand Trend{tag} ({slope_d * len(df_data):+.1f} m³/s)"
        t_lbl = f"🌡️ Temperature Trend{tag} ({slope_t * len(df_data):+.2f} °C)"
        a_lbl = f"🌿 Algae Formation Trend{tag} ({slope_a * len(df_data):+.3f} NDVI)"
    elif line_typ == "🌊 7-Day Moving Trend":
        s_v, d_v, t_v, a_v = s_smooth, d_smooth, t_smooth, a_smooth
        s_lbl = f"💧 Water Supply (7-Day Mean{tag})"
        d_lbl = f"🚰 Water Demand (7-Day Mean{tag})"
        t_lbl = f"🌡️ Temperature (7-Day Mean{tag})"
        a_lbl = f"🌿 Algae Formation (7-Day Mean{tag})"
    else:
        s_v, d_v, t_v, a_v = sim_s_arr, sim_d_arr, sim_t_arr, sim_a_arr
        s_lbl = f"💧 Water Supply{tag} (m³/s)"
        d_lbl = f"🚰 Water Demand{tag} (m³/s)"
        t_lbl = f"🌡️ Temperature{tag} (°C)"
        a_lbl = f"🌿 Algae Formation{tag} (NDVI)"

    fig = go.Figure()

    if scale_typ == "Multi-Axis (m³/s, °C, NDVI)":
        # 1. Water Supply
        fig.add_trace(go.Scatter(
            x=df_data["date"], y=s_v, name=s_lbl,
            line=dict(color="#38bdf8", width=3),
            fill="tozeroy", fillcolor="rgba(56, 189, 248, 0.08)", yaxis="y1"
        ))
        # 2. Water Demand
        fig.add_trace(go.Scatter(
            x=df_data["date"], y=d_v, name=d_lbl,
            line=dict(color="#fb923c", width=3, dash="dash"), yaxis="y1"
        ))
        # 3. Temperature
        fig.add_trace(go.Scatter(
            x=df_data["date"], y=t_v, name=t_lbl,
            line=dict(color="#f87171", width=3), yaxis="y2"
        ))
        # 4. Algae Formation
        fig.add_trace(go.Scatter(
            x=df_data["date"], y=a_v, name=a_lbl,
            line=dict(color="#4ade80", width=3.5),
            fill="tozeroy", fillcolor="rgba(74, 222, 128, 0.15)", yaxis="y3"
        ))

        # If simulation active, render unmitigated reference ghost lines
        if is_sim:
            fig.add_trace(go.Scatter(
                x=df_data["date"], y=base_s_arr,
                name="💧 Baseline Supply (Unmitigated)",
                line=dict(color="rgba(56, 189, 248, 0.35)", width=1.5, dash="dot"),
                yaxis="y1"
            ))
            fig.add_trace(go.Scatter(
                x=df_data["date"], y=base_d_arr,
                name="🚰 Baseline Demand (Unmitigated)",
                line=dict(color="rgba(251, 146, 60, 0.35)", width=1.5, dash="dot"),
                yaxis="y1"
            ))
            fig.add_trace(go.Scatter(
                x=df_data["date"], y=base_a_arr,
                name="🌿 Baseline Algae (Unmitigated)",
                line=dict(color="rgba(74, 222, 128, 0.40)", width=1.5, dash="dot"),
                yaxis="y3"
            ))

        fig.update_layout(
            template="plotly_dark",
            height=540,
            hovermode="x unified",
            margin=dict(l=60, r=70, t=30, b=50),
            xaxis=dict(domain=[0.05, 0.88], title="Observation Date"),
            yaxis=dict(
                title=dict(text="Water Supply & Demand (m³/s)", font=dict(color="#38bdf8", size=13)),
                tickfont=dict(color="#38bdf8"), gridcolor="#1e293b"
            ),
            yaxis2=dict(
                title=dict(text="Water Temperature (°C)", font=dict(color="#f87171", size=13)),
                tickfont=dict(color="#f87171"), overlaying="y", side="right", gridcolor="rgba(0,0,0,0)"
            ),
            yaxis3=dict(
                title=dict(text="Algae Formation (NDVI)", font=dict(color="#4ade80", size=13)),
                tickfont=dict(color="#4ade80"), anchor="free", overlaying="y", side="right", position=0.96,
                gridcolor="rgba(0,0,0,0)", range=[-0.12, max(0.60, float(sim_a_arr.max()) * 1.15)]
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5, font=dict(size=11))
        )
    else:
        # Normalized (0–100%)
        s_min, s_max = sim_s_arr.min(), sim_s_arr.max()
        d_min, d_max = sim_d_arr.min(), sim_d_arr.max()
        t_min, t_max = sim_t_arr.min(), sim_t_arr.max()
        a_min, a_max = sim_a_arr.min(), sim_a_arr.max()

        s_pct = (s_v - s_min) / (s_max - s_min + 1e-4) * 100
        d_pct = (d_v - d_min) / (d_max - d_min + 1e-4) * 100
        t_pct = (t_v - t_min) / (t_max - t_min + 1e-4) * 100
        a_pct = (a_v - a_min) / (a_max - a_min + 1e-4) * 100

        fig.add_trace(go.Scatter(x=df_data["date"], y=s_pct, name="💧 Water Supply (%)", line=dict(color="#38bdf8", width=3)))
        fig.add_trace(go.Scatter(x=df_data["date"], y=d_pct, name="🚰 Water Demand (%)", line=dict(color="#fb923c", width=3, dash="dash")))
        fig.add_trace(go.Scatter(x=df_data["date"], y=t_pct, name="🌡️ Temperature (%)", line=dict(color="#f87171", width=3)))
        fig.add_trace(go.Scatter(x=df_data["date"], y=a_pct, name="🌿 Algae Formation (%)", line=dict(color="#4ade80", width=3.5), fill="tozeroy", fillcolor="rgba(74, 222, 128, 0.15)"))

        fig.update_layout(
            template="plotly_dark",
            height=540,
            hovermode="x unified",
            margin=dict(l=50, r=50, t=30, b=50),
            xaxis=dict(title="Observation Date"),
            yaxis=dict(title="Relative Stress Scale (0–100%)", range=[0, 105], gridcolor="#1e293b"),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5, font=dict(size=11))
        )
        
    return fig, sim_t_arr, sim_s_arr, sim_d_arr, sim_a_arr


# -------------------------------------------------------------
# WHAT-IF SIMULATOR CONTROLS
# -------------------------------------------------------------
sim_panel_container = st.container()

with sim_panel_container:
    with st.expander("🎛️ Interactive 'What-If' Climate & Hydraulic Scenario Simulator", expanded=True):
        st.caption("Test forward-looking limnological interventions: Apply climate warming heatwaves or engage emergency weir flushing & abstraction moratoria.")

        # Preset action buttons
        btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)
        if btn_c1.button("🚨 IPCC Heatwave (+2.0°C)"):
            st.session_state["sim_temp"] = 2.0
            st.session_state["sim_flush"] = 0.0
            st.session_state["sim_restr"] = 0
            st.rerun()
        if btn_c2.button("💧 Weir Flushing (+30 m³/s)"):
            st.session_state["sim_temp"] = 0.0
            st.session_state["sim_flush"] = 30.0
            st.session_state["sim_restr"] = 0
            st.rerun()
        if btn_c3.button("⚖️ Compound Action (+25m³/s, -20%)"):
            st.session_state["sim_temp"] = 1.5
            st.session_state["sim_flush"] = 25.0
            st.session_state["sim_restr"] = 20
            st.rerun()
        if btn_c4.button("🔄 Reset to Baseline (0)"):
            st.session_state["sim_temp"] = 0.0
            st.session_state["sim_flush"] = 0.0
            st.session_state["sim_restr"] = 0
            st.rerun()

        # Sliders
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            val_t = st.slider(
                "🌡️ Temperature Delta (Heatwave)",
                min_value=-1.0, max_value=+4.0,
                value=float(st.session_state["sim_temp"]),
                step=0.5, format="%+.1f °C",
                help="Simulate atmospheric warming or summer heatwave exceedance."
            )
            st.session_state["sim_temp"] = val_t
        with col_s2:
            val_q = st.slider(
                "💧 Weir Flushing Pulse (Reservoir Inflow)",
                min_value=0.0, max_value=50.0,
                value=float(st.session_state["sim_flush"]),
                step=5.0, format="+%.0f m³/s",
                help="Simulate emergency release from upstream reservoirs/barrages to flush stagnant backwaters."
            )
            st.session_state["sim_flush"] = val_q
        with col_s3:
            val_r = st.slider(
                "🚰 Abstraction Moratorium (Irrigation Ban)",
                min_value=0, max_value=40,
                value=int(st.session_state["sim_restr"]),
                step=5, format="-%d%%",
                help="Simulate mandatory restrictions on agricultural irrigation and municipal water use."
            )
            st.session_state["sim_restr"] = val_r

# Active simulation parameters
curr_sim_t = float(st.session_state["sim_temp"])
curr_sim_q = float(st.session_state["sim_flush"])
curr_sim_r = int(st.session_state["sim_restr"])
is_simulation_active = (curr_sim_t != 0.0 or curr_sim_q > 0.0 or curr_sim_r > 0)

# Build Graph and Simulated Curves
graph_4line_fig, sim_t_out, sim_s_out, sim_d_out, sim_a_out = build_4parameter_graph(
    df_display, profile, "📈 Trendlines (OLS)", "Multi-Axis (m³/s, °C, NDVI)",
    curr_sim_t, curr_sim_q, curr_sim_r
)

# Compute Simulation Impact Metrics
peak_base_algae = float(df_display["ndvi"].max())
peak_sim_algae = float(sim_a_out.max())
algae_change_pct = ((peak_sim_algae - peak_base_algae) / (peak_base_algae + 1e-4)) * 100.0

base_def = float((df_display["water_supply_m3s"] - df_display["water_demand_m3s"]).min())
sim_def = float((sim_s_out - sim_d_out).min())
def_recovery = sim_def - base_def

# Simulated Risk Classification for Active Station
if peak_sim_algae >= 0.22 or (sim_t_out.iloc[-1] >= 24.0 and sim_def < 0):
    active_sim_status = "CRITICAL (Alert)"
    active_sim_color = "#ef4444"
elif peak_sim_algae >= 0.12 or sim_t_out.iloc[-1] >= 22.0:
    active_sim_status = "WARNING (Watch)"
    active_sim_color = "#f97316"
elif sim_t_out.iloc[-1] >= 20.5:
    active_sim_status = "WATCH (Notice)"
    active_sim_color = "#eab308"
else:
    active_sim_status = "LOW (Clear)"
    active_sim_color = "#10b981"

# -------------------------------------------------------------
# RENDER ACCORDING TO VIEW MODE
# -------------------------------------------------------------
if view_layout in ["🗺️ Map & Station Graph (Integrated)", "🌐 Fullscreen Germany Map"]:
    col_map_hdr1, col_map_hdr2 = st.columns([3, 1])
    with col_map_hdr1:
        st.subheader("🗺️ National Germany Freshwater Early Warning Map")
        st.caption("Hover over any beacon to view real-time station metrics. Red indicates Critical/Alert tier.")
    with col_map_hdr2:
        st.markdown(f"**Focused Station:** `{profile['name'].split(' (')[0]}`")

    # Map with live simulated beacon update for active station
    germany_map_fig = build_germany_map(
        map_df, active_basin_key,
        sim_status=active_sim_status if is_simulation_active else None,
        sim_color=active_sim_color if is_simulation_active else None
    )
    st.plotly_chart(germany_map_fig, use_container_width=True)

if view_layout in ["🗺️ Map & Station Graph (Integrated)", "📈 Fullscreen 4-Line Graph"]:
    st.markdown("---")

    # Simulation Impact Scorecard if active
    if is_simulation_active:
        st.markdown("#### **⚡ Simulation Outcome & Intervention Impact:**")
        sc_col1, sc_col2, sc_col3, sc_col4 = st.columns(4)
        with sc_col1:
            st.metric(
                "🌿 Algae Peak Response",
                f"{algae_change_pct:+.1f}%",
                delta="Bloom Suppressed" if algae_change_pct < 0 else "Bloom Accelerated",
                delta_color="normal" if algae_change_pct < 0 else "inverse"
            )
        with sc_col2:
            st.metric(
                "💧 Hydraulic Recovery",
                f"{def_recovery:+.1f} m³/s",
                delta="Ecological Flow Restored" if def_recovery > 0 else "Deficit Worsened",
                delta_color="normal" if def_recovery > 0 else "inverse"
            )
        with sc_col3:
            st.metric(
                "🌡️ Simulated Surface Temp",
                f"{sim_t_out.iloc[-1]:.1f} °C",
                delta=f"{curr_sim_t:+.1f} °C Exceedance",
                delta_color="inverse" if curr_sim_t > 0 else "normal"
            )
        with sc_col4:
            st.metric(
                "🚨 Simulated Alert Tier",
                f"{active_sim_status.split(' ')[0]}",
                delta="Intervention Mitigated" if "LOW" in active_sim_status or "WATCH" in active_sim_status else "High Hazard Alert",
                delta_color="normal" if "LOW" in active_sim_status or "WATCH" in active_sim_status else "inverse"
            )
        st.markdown("<br>", unsafe_allow_html=True)
    
    col_g_hdr, col_ctl1, col_ctl2 = st.columns([2, 1, 1])
    with col_g_hdr:
        st.subheader(f"📊 Station Dynamics: {profile['name']}")
        st.caption(f"Showing the 4 coupled parameters over time for **{profile.get('state', 'Germany')}** ({display_title_suffix})")
    with col_ctl1:
        line_type_sel = st.radio(
            "Line Mode (4 Lines):",
            ["📈 Trendlines (OLS)", "📊 Observed Data", "🌊 7-Day Moving Trend"],
            index=0,
            horizontal=True
        )
    with col_ctl2:
        scale_type_sel = st.radio(
            "Vertical Axis:",
            ["Multi-Axis (m³/s, °C, NDVI)", "Normalized (0–100%)"],
            index=0,
            horizontal=True
        )

    # Re-render graph with selected line and scale type
    graph_final, _, _, _, _ = build_4parameter_graph(
        df_display, profile, line_type_sel, scale_type_sel,
        curr_sim_t, curr_sim_q, curr_sim_r
    )
    st.plotly_chart(graph_final, use_container_width=True)

# Station Quick-Data Table expander
with st.expander("📋 View National German Stations Data Matrix"):
    st.dataframe(
        map_df[["name", "type", "state", "status", "temp", "supply", "demand", "ndvi", "balance"]].rename(columns={
            "name": "Station Name",
            "type": "System Type",
            "state": "Federal State / Region",
            "status": "Alert Status",
            "temp": "Temp (°C)",
            "supply": "Supply (m³/s)",
            "demand": "Demand (m³/s)",
            "ndvi": "Algae (NDVI)",
            "balance": "Flow Balance (m³/s)"
        }),
        use_container_width=True
    )
