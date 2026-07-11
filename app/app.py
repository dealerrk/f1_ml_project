import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Strategy Intelligence",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main { background-color: #0f0f13; }

.hero-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-size: 0.95rem;
    color: #8a8a9a;
    margin-bottom: 1.5rem;
    letter-spacing: 0.3px;
}

.driver-card {
    padding: 16px 20px;
    border-radius: 10px;
    margin-bottom: 16px;
    border-left: 4px solid;
}

.metric-card {
    background: #1a1a24;
    border: 1px solid #2a2a38;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}

.metric-label {
    font-size: 0.72rem;
    color: #8a8a9a;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
}

.metric-unit {
    font-size: 0.75rem;
    color: #8a8a9a;
}

.strategy-box {
    padding: 20px 24px;
    border-radius: 12px;
    margin: 12px 0;
    border-left: 5px solid;
}

.strategy-pit {
    background: rgba(239, 68, 68, 0.12);
    border-color: #ef4444;
    color: #fca5a5;
}

.strategy-prepare {
    background: rgba(245, 158, 11, 0.12);
    border-color: #f59e0b;
    color: #fcd34d;
}

.strategy-stay {
    background: rgba(34, 197, 94, 0.12);
    border-color: #22c55e;
    color: #86efac;
}

.strategy-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 6px;
}

.strategy-desc {
    font-size: 0.88rem;
    opacity: 0.85;
    line-height: 1.5;
}

.cluster-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-top: 6px;
}

.model-stat {
    background: #1a1a24;
    border: 1px solid #2a2a38;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
}

.model-stat-label {
    font-size: 0.72rem;
    color: #8a8a9a;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.model-stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 600;
    color: #e4e4f0;
}

.improvement-tag {
    color: #22c55e;
    font-size: 0.8rem;
    font-weight: 600;
}

.section-header {
    font-size: 0.72rem;
    font-weight: 600;
    color: #8a8a9a;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #2a2a38;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD ASSETS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_model():
    path = os.path.join(BASE_DIR, "models", "best_model_driver_aware.pkl")
    return joblib.load(path)

@st.cache_resource
def load_encoders():
    path = os.path.join(BASE_DIR, "models", "label_encoders.pkl")
    return joblib.load(path)

@st.cache_resource
def load_feature_cols():
    path = os.path.join(BASE_DIR, "models", "feature_columns.pkl")
    return joblib.load(path)

@st.cache_data
def load_driver_profiles():
    path = os.path.join(BASE_DIR, "notebooks", "driver_profiles_with_clusters.csv")
    return pd.read_csv(path)

model         = load_model()
label_encoders = load_encoders()
feature_cols  = load_feature_cols()
profiles_df   = load_driver_profiles()

# ─────────────────────────────────────────────
# STATIC MAPPINGS
# ─────────────────────────────────────────────
DRIVER_FULL = {
    "VER": "Max Verstappen",      "PER": "Sergio Perez",
    "LEC": "Charles Leclerc",     "SAI": "Carlos Sainz",
    "HAM": "Lewis Hamilton",      "RUS": "George Russell",
    "NOR": "Lando Norris",        "PIA": "Oscar Piastri",
    "ALO": "Fernando Alonso",     "STR": "Lance Stroll",
    "GAS": "Pierre Gasly",        "OCO": "Esteban Ocon",
    "ALB": "Alexander Albon",     "SAR": "Logan Sargeant",
    "TSU": "Yuki Tsunoda",        "RIC": "Daniel Ricciardo",
    "BOT": "Valtteri Bottas",     "ZHO": "Zhou Guanyu",
    "MAG": "Kevin Magnussen",     "HUL": "Nico Hulkenberg",
    "LAW": "Liam Lawson",         "COL": "Franco Colapinto",
    "DEV": "Nyck de Vries",       "LAT": "Nicholas Latifi",
    "MSC": "Mick Schumacher",     "VET": "Sebastian Vettel",
}

DRIVER_TEAM = {
    "VER": "Red Bull Racing",   "PER": "Red Bull Racing",
    "LEC": "Ferrari",           "SAI": "Ferrari",
    "HAM": "Mercedes",          "RUS": "Mercedes",
    "NOR": "McLaren",           "PIA": "McLaren",
    "ALO": "Aston Martin",      "STR": "Aston Martin",
    "GAS": "Alpine",            "OCO": "Alpine",
    "ALB": "Williams",          "SAR": "Williams",
    "TSU": "RB",                "RIC": "RB",
    "BOT": "Kick Sauber",       "ZHO": "Kick Sauber",
    "MAG": "Haas F1 Team",      "HUL": "Haas F1 Team",
    "LAW": "AlphaTauri",        "COL": "Williams",
    "DEV": "AlphaTauri",        "LAT": "Williams",
    "MSC": "Haas F1 Team",      "VET": "Aston Martin",
}

TEAM_COLOR = {
    "Red Bull Racing": "#3671C6",  "Ferrari":      "#E8002D",
    "Mercedes":        "#27F4D2",  "McLaren":      "#FF8000",
    "Aston Martin":    "#229971",  "Alpine":       "#FF87BC",
    "Williams":        "#64C4FF",  "RB":           "#6692FF",
    "Kick Sauber":     "#52E252",  "Haas F1 Team": "#B6BABD",
    "AlphaTauri":      "#5E8FAA",  "Alfa Romeo":   "#C92D4B",
}

CLUSTER_NAMES = {
    0: ("Balanced Driver",          "#64C4FF"),
    1: ("High-Speed Aggressive",    "#FF4444"),
    2: ("Technical Driver",         "#FFB347"),
    3: ("Fast & Efficient",         "#22c55e"),
    4: ("Conservative Driver",      "#A78BFA"),
    5: ("Smooth Driver",            "#F472B6"),
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_driver_cluster(driver_abbr):
    """Get the most common cluster for a driver across all races."""
    rows = profiles_df[profiles_df["Driver"] == driver_abbr]
    if rows.empty:
        return 3, "Fast & Efficient"
    cluster = int(rows["DrivingStyleCluster"].mode()[0])
    name = rows["DrivingStyle"].mode()[0] if "DrivingStyle" in rows.columns else CLUSTER_NAMES[cluster][0]
    return cluster, name


def encode_input(driver, team, compound, tyre_life, stint,
                 position, track_status, cluster):
    """Build a properly encoded input DataFrame for the model."""
    le_d  = label_encoders["Driver"]
    le_t  = label_encoders["Team"]
    le_c  = label_encoders["Compound"]

    # Safe transform — if unseen label, fallback to 0
    def safe_transform(le, val):
        if val in le.classes_:
            return int(le.transform([val])[0])
        return 0

    row = {
        "Driver":              safe_transform(le_d, driver),
        "Team":                safe_transform(le_t, team),
        "Compound":            safe_transform(le_c, compound),
        "TyreLife":            float(tyre_life),
        "Stint":               float(stint),
        "Position":            float(position),
        "TrackStatus":         float(track_status),
        "DrivingStyleCluster": float(cluster),
    }
    df = pd.DataFrame([row])[feature_cols]
    return df


def predict_lap_time(driver, team, compound, tyre_life, stint,
                     position, track_status, cluster):
    inp = encode_input(driver, team, compound, tyre_life, stint,
                       position, track_status, cluster)
    return float(model.predict(inp)[0])


def simulate_degradation(driver, team, compound, current_tyre_life,
                          stint, position, track_status, cluster,
                          laps_ahead=20, min_stint=5):
    """Simulate future lap times by incrementing TyreLife."""
    results = []
    for i in range(laps_ahead):
        tl = current_tyre_life + i
        lt = predict_lap_time(driver, team, compound, tl, stint,
                              position, track_status, cluster)
        results.append({"TyreLife": tl, "PredictedLapTime": lt})
    return pd.DataFrame(results)


def get_strategy(sim_df, min_stint=5):
    """Model-driven strategy: compare lap+1 vs lap+10 degradation."""
    if len(sim_df) < 10:
        return "stay", 0.0, None

    # Ignore first few laps of a stint (warm-up effect)
    valid = sim_df[sim_df["TyreLife"] >= min_stint]
    if len(valid) < 2:
        return "stay", 0.0, None

    t0 = valid.iloc[0]["PredictedLapTime"]
    t10_idx = min(9, len(valid) - 1)
    t10 = valid.iloc[t10_idx]["PredictedLapTime"]
    delta = t10 - t0

    # Find pit lap: first lap where deg exceeds threshold
    pit_lap = None
    for _, row in valid.iterrows():
        if row["PredictedLapTime"] - t0 > 1.5:
            pit_lap = int(row["TyreLife"])
            break

    if delta > 1.5:
        return "pit", delta, pit_lap
    elif delta > 0.8:
        return "prepare", delta, pit_lap
    else:
        return "stay", delta, pit_lap


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">Race Inputs</div>',
                unsafe_allow_html=True)

    # Only show drivers the model knows
    known_drivers = list(label_encoders["Driver"].classes_)
    display_drivers = {
        abbr: f"{DRIVER_FULL.get(abbr, abbr)} ({abbr})"
        for abbr in known_drivers
        if abbr in DRIVER_FULL
    }
    selected_display = st.selectbox(
        "Driver",
        options=list(display_drivers.values()),
        index=list(display_drivers.values()).index(
            "Max Verstappen (VER)") if "Max Verstappen (VER)" in display_drivers.values() else 0
    )
    driver_abbr = selected_display.split("(")[-1].replace(")", "").strip()

    compound = st.selectbox("Tyre Compound", ["SOFT", "MEDIUM", "HARD"])

    tyre_life = st.slider("Current Tyre Life (laps)", 1, 40, 10)

    stint = st.selectbox("Stint Number", [1, 2, 3, 4], index=0)

    position = st.slider("Current Race Position", 1, 20, 5)

    track_status_label = st.selectbox(
        "Track Status",
        ["🟢 Green Flag", "🟡 Yellow Flag", "🔴 Safety Car"]
    )
    track_status_map = {
        "🟢 Green Flag": 1,
        "🟡 Yellow Flag": 2,
        "🔴 Safety Car": 4
    }
    track_status = track_status_map[track_status_label]

    st.markdown("---")
    predict_btn = st.button("🏁 Run Strategy Analysis", use_container_width=True)

# ─────────────────────────────────────────────
# MAIN — HERO HEADER
# ─────────────────────────────────────────────
st.markdown(
    '<div class="hero-title">🏎 F1 Strategy Intelligence</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="hero-sub">Driver-aware tyre degradation modelling · '
    'Telemetry-based clustering · Real-time pit strategy</div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
# DRIVER CARD
# ─────────────────────────────────────────────
team        = DRIVER_TEAM.get(driver_abbr, "Unknown")
team_color  = TEAM_COLOR.get(team, "#888888")
cluster_id, cluster_name = get_driver_cluster(driver_abbr)
cluster_color = CLUSTER_NAMES.get(cluster_id, ("Unknown", "#888888"))[1]
full_name   = DRIVER_FULL.get(driver_abbr, driver_abbr)

st.markdown(f"""
<div class="driver-card" style="
    background: linear-gradient(135deg, {team_color}18 0%, #1a1a2400 100%);
    border-left-color: {team_color};
    border: 1px solid {team_color}40;
">
    <div style="font-size:0.72rem;color:#8a8a9a;letter-spacing:1px;
                text-transform:uppercase;margin-bottom:2px;">{team}</div>
    <div style="font-size:1.6rem;font-weight:700;color:#fff;
                font-family:'JetBrains Mono',monospace;">{full_name}</div>
    <span class="cluster-badge" style="background:{cluster_color}22;
          color:{cluster_color};border:1px solid {cluster_color}55;">
        🏎 {cluster_name}
    </span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# QUICK METRICS ROW
# ─────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (c1, "Tyre Life",   f"{tyre_life}", "laps"),
    (c2, "Compound",    compound,        ""),
    (c3, "Position",    f"P{position}",  ""),
    (c4, "Stint",       f"#{stint}",     ""),
    (c5, "Track",       track_status_label.split(" ", 1)[1], ""),
]
for col, label, val, unit in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-unit">{unit}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PREDICTION SECTION
# ─────────────────────────────────────────────
if predict_btn:

    # Current lap time prediction
    current_lt = predict_lap_time(
        driver_abbr, team, compound, tyre_life,
        stint, position, track_status, cluster_id
    )

    # Simulate future laps
    sim_df = simulate_degradation(
        driver_abbr, team, compound, tyre_life,
        stint, position, track_status, cluster_id,
        laps_ahead=20
    )

    # Strategy decision
    strategy, delta, pit_lap = get_strategy(sim_df)

    # ── Current lap time ──────────────────────
    st.markdown('<div class="section-header">Current Lap Prediction</div>',
                unsafe_allow_html=True)

    mins = int(current_lt // 60)
    secs = current_lt % 60
    col_lt, col_deg, col_pit = st.columns(3)

    with col_lt:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Lap Time</div>
            <div class="metric-value">{mins}:{secs:06.3f}</div>
            <div class="metric-unit">min:sec.ms</div>
        </div>""", unsafe_allow_html=True)

    with col_deg:
        deg_color = "#ef4444" if delta > 1.5 else "#f59e0b" if delta > 0.8 else "#22c55e"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Degradation (10 laps)</div>
            <div class="metric-value" style="color:{deg_color}">
                +{delta:.3f}s
            </div>
            <div class="metric-unit">lap time loss</div>
        </div>""", unsafe_allow_html=True)

    with col_pit:
        pit_display = f"Lap {pit_lap}" if pit_lap else "Not imminent"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Recommended Pit (Tyre Life)</div>
            <div class="metric-value">{pit_display}</div>
            <div class="metric-unit">tyre life at pit</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Strategy recommendation ───────────────
    st.markdown('<div class="section-header">Strategy Recommendation</div>',
                unsafe_allow_html=True)

    if strategy == "pit":
        st.markdown(f"""
        <div class="strategy-box strategy-pit">
            <div class="strategy-title">🔴 PIT THIS LAP</div>
            <div class="strategy-desc">
                Tyre degradation is severe — predicted {delta:.2f}s lap time loss
                over the next 10 laps. An immediate pit stop will recover
                more time than staying out.
                {f'Optimal pit window: tyre life {pit_lap}.' if pit_lap else ''}
            </div>
        </div>""", unsafe_allow_html=True)

    elif strategy == "prepare":
        st.markdown(f"""
        <div class="strategy-box strategy-prepare">
            <div class="strategy-title">🟡 PREPARE TO PIT — Within 3 Laps</div>
            <div class="strategy-desc">
                Performance is declining — {delta:.2f}s predicted degradation
                over the next 10 laps. Begin boxing preparation.
                {f'Optimal pit window: tyre life {pit_lap}.' if pit_lap else ''}
            </div>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="strategy-box strategy-stay">
            <div class="strategy-title">🟢 CONTINUE STINT</div>
            <div class="strategy-desc">
                Tyre condition is healthy — only {delta:.2f}s predicted
                degradation over the next 10 laps.
                {full_name}'s {cluster_name.lower()} style
                is maintaining pace well on this {compound.lower()} compound.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Degradation forecast chart ────────────
    st.markdown('<div class="section-header">Tyre Degradation Forecast</div>',
                unsafe_allow_html=True)

    fig = go.Figure()

    # Degradation line
    fig.add_trace(go.Scatter(
        x=sim_df["TyreLife"],
        y=sim_df["PredictedLapTime"],
        mode="lines+markers",
        name="Predicted Lap Time",
        line=dict(color=team_color, width=2.5),
        marker=dict(size=5, color=team_color),
    ))

    # Pit lap vertical line
    if pit_lap:
        fig.add_vline(
            x=pit_lap,
            line_dash="dash",
            line_color="rgba(239,68,68,0.3)",
            line_width=1.5,
            annotation_text=f"  Pit @ TL {pit_lap}",
            annotation_font_color="#ef4444",
            annotation_font_size=11,
        )

    # Shaded warning zone
    if len(sim_df) > 0:
        baseline = sim_df["PredictedLapTime"].iloc[0]
        fig.add_hline(
    y=baseline + 1.5,
    line_dash="dot",
    line_color="rgba(239,68,68,0.3)",
    annotation_text="  Pit threshold (+1.5s)",
    annotation_font_color="#ef4444",
    annotation_font_size=10,
)

    fig.update_layout(
        title=dict(
            text=f"{full_name} · {compound} · Stint {stint}",
            font=dict(size=14, color="#e4e4f0"),
        ),
        plot_bgcolor="#0f0f13",
        paper_bgcolor="#0f0f13",
        font=dict(color="#8a8a9a", family="Inter"),
        xaxis=dict(
            title="Tyre Life (laps)",
            gridcolor="#2a2a38",
            color="#8a8a9a",
            showline=True,
            linecolor="#2a2a38",
        ),
        yaxis=dict(
            title="Predicted Lap Time (s)",
            gridcolor="#2a2a38",
            color="#8a8a9a",
            showline=True,
            linecolor="#2a2a38",
        ),
        legend=dict(
            bgcolor="#1a1a24",
            bordercolor="#2a2a38",
            borderwidth=1,
            font=dict(color="#e4e4f0"),
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        height=380,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Future lap table ──────────────────────
    with st.expander("📋 View Full Lap-by-Lap Forecast"):
        display_df = sim_df.copy()
        display_df["Lap Time"] = display_df["PredictedLapTime"].apply(
            lambda x: f"{int(x//60)}:{x%60:06.3f}"
        )
        display_df["vs Current"] = (
            display_df["PredictedLapTime"] - display_df["PredictedLapTime"].iloc[0]
        ).apply(lambda x: f"+{x:.3f}s" if x >= 0 else f"{x:.3f}s")
        display_df["Tyre Life"] = display_df["TyreLife"].astype(int)
        st.dataframe(
            display_df[["Tyre Life", "Lap Time", "vs Current"]],
            use_container_width=True,
            hide_index=True
        )

else:
    # ── Empty state ───────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#8a8a9a;">
        <div style="font-size:3rem;margin-bottom:16px;">🏁</div>
        <div style="font-size:1.1rem;font-weight:600;color:#e4e4f0;
                    margin-bottom:8px;">Ready to analyse</div>
        <div style="font-size:0.88rem;">
            Select your driver, compound, and race conditions
            in the sidebar, then click <strong>Run Strategy Analysis</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL INFO PANEL (always visible at bottom)
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">Model Information</div>',
            unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
model_stats = [
    (m1, "Algorithm",         "Random Forest"),
    (m2, "Baseline R²",       "0.416"),
    (m3, "Driver-Aware R²",   "0.949"),
    (m4, "Improvement",       "+0.533 ↑"),
    (m5, "Top Feature",       "DrivingStyleCluster (78.8%)"),
]
for col, label, val in model_stats:
    with col:
        color = "#22c55e" if "↑" in val else "#e4e4f0"
        st.markdown(f"""
        <div class="model-stat">
            <div class="model-stat-label">{label}</div>
            <div class="model-stat-value" style="color:{color}">{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:24px 0 8px;
            font-size:0.72rem;color:#4a4a5a;letter-spacing:0.5px;">
    Built with FastF1 · Scikit-Learn · Streamlit &nbsp;|&nbsp;
    Training: 11,869 laps · 4 circuits · 2022–2024 seasons &nbsp;|&nbsp;
    Novel feature: Telemetry-based K-Means driver style clustering (k=6)
</div>
""", unsafe_allow_html=True)
