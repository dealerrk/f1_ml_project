import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import os

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="F1 Race Strategy Intelligence",
    page_icon="🏎️",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = joblib.load("models/best_model.pkl")

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🏎️ F1 Race Strategy Intelligence System")

st.markdown("""
Predict Formula 1 lap performance using machine learning and receive a race strategy recommendation based on tyre degradation and race conditions.
""")

st.divider()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("🏁 Race Inputs")



driver_team = {
    "Max Verstappen": "Red Bull",
    "Sergio Perez": "Red Bull",

    "Charles Leclerc": "Ferrari",
    "Carlos Sainz": "Ferrari",

    "Lewis Hamilton": "Mercedes",
    "George Russell": "Mercedes",

    "Lando Norris": "McLaren",
    "Oscar Piastri": "McLaren",

    "Fernando Alonso": "Aston Martin",
    "Lance Stroll": "Aston Martin",

    "Pierre Gasly": "Alpine",
    "Esteban Ocon": "Alpine",

    "Alexander Albon": "Williams",
    "Logan Sargeant": "Williams",

    "Yuki Tsunoda": "RB",
    "Daniel Ricciardo": "RB",

    "Valtteri Bottas": "Kick Sauber",
    "Zhou Guanyu": "Kick Sauber",

    "Kevin Magnussen": "Haas",
    "Nico Hulkenberg": "Haas"
}

team_color = {
    "Ferrari": "#DC0000",
    "Mercedes": "#00D2BE",
    "Red Bull": "#1E41FF",
    "McLaren": "#FF8700",
    "Aston Martin": "#006F62",
    "Alpine": "#0090FF",
    "Williams": "#005AFF",
    "RB": "#6692FF",
    "Kick Sauber": "#52E252",
    "Haas": "#B6BABD"
}


driver = st.sidebar.selectbox(
    "🏎 Driver",
    [
        "Max Verstappen",
        "Sergio Perez",
        "Charles Leclerc",
        "Carlos Sainz",
        "Lewis Hamilton",
        "George Russell",
        "Lando Norris",
        "Oscar Piastri",
        "Fernando Alonso",
        "Lance Stroll",
        "Pierre Gasly",
        "Esteban Ocon",
        "Alexander Albon",
        "Logan Sargeant",
        "Yuki Tsunoda",
        "Daniel Ricciardo",
        "Valtteri Bottas",
        "Zhou Guanyu",
        "Kevin Magnussen",
        "Nico Hulkenberg"
    ]
)
team = driver_team[driver]
color = team_color[team]

st.markdown(
    f"""
<div style="
padding:18px;
border-radius:12px;
background-color:{color};
color:white;
">

## 🏎 {driver}

### Team: {team}

</div>
""",
unsafe_allow_html=True
)

lap = st.sidebar.slider(
    "Current Lap",
    min_value=1,
    max_value=70,
    value=25
)

tyre_life = st.sidebar.slider(
    "Tyre Life",
    min_value=1,
    max_value=35,
    value=15
)

stint = st.sidebar.selectbox(
    "Current Stint",
    [1, 2, 3, 4]
)

position = st.sidebar.slider(
    "Current Position",
    1,
    20,
    5
)
track_options = {
    "🟢 Green Flag": 1,
    "🟡 Yellow Flag": 2,
    "🔴 Safety Car": 4
}

track_choice = st.sidebar.selectbox(
    "Track Status",
    list(track_options.keys())
)

track_status = track_options[track_choice]

compound = st.sidebar.selectbox(
    "Tyre Compound",
    ["SOFT", "MEDIUM", "HARD"]
)

predict = st.sidebar.button("🏁 Predict Strategy")

# ---------------------------------------------------
# DISPLAY VALUES
# ---------------------------------------------------

track_map = {
    1: "🟢 Green Flag",
    2: "🟡 Yellow Flag",
    4: "🔴 Safety Car"
}

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🏁 Lap", lap)

with col2:
    st.metric("🏎 Position", f"P{position}")

with col3:
    st.metric("🛞 Compound", compound)

with col4:
    st.metric("⏱ Tyre Life", f"{tyre_life} laps")

st.divider()

# ---------------------------------------------------
# ENCODING
# ---------------------------------------------------

compound_map = {
    "SOFT": 2,
    "MEDIUM": 1,
    "HARD": 0
}

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if predict:

    fuel_load = max(0.0, 1 - lap / 50)

    stint_progress = tyre_life / 25

    tyre_age_squared = tyre_life ** 2

    input_data = pd.DataFrame({
        "LapNumber": [lap],
        "TyreLife": [tyre_life],
        "Stint": [stint],
        "Position": [position],
        "TrackStatus": [track_status],
        "FuelLoadApprox": [fuel_load],
        "StintProgress": [stint_progress],
        "CompoundEncoded": [compound_map[compound]],
        "FreshTyre": [0],
        "TyreAgeSquared": [tyre_age_squared]
    })

    prediction = model.predict(input_data)[0]

    # ------------------------------------------------
    # HERO PREDICTION
    # ------------------------------------------------

    st.subheader("🏁 Predicted Lap Time")

    st.metric(
        label="Prediction",
        value=f"{prediction:.2f} sec"
    )

    st.divider()

    # ------------------------------------------------
    # RACE STATUS
    # ------------------------------------------------

    st.subheader("🏎 Race Status")

    st.write(f"**Track Condition:** {track_map[track_status]}")

    fuel_remaining = max(0, 100 - lap)

    st.write("⛽ Fuel Remaining")

    st.progress(int(fuel_remaining))

    tyre_wear = min(int((tyre_life / 35) * 100), 100)

    st.write("🛞 Tyre Wear")

    st.progress(tyre_wear)

    st.divider()

    # ------------------------------------------------
    # STRATEGY
    # ------------------------------------------------

    st.subheader("🏁 Strategy Recommendation")

    if tyre_life >= 28 or prediction > 97:

        st.error("🔴 PIT THIS LAP")

        st.write(
            "Tyre degradation is high and lap time has dropped significantly. "
            "An immediate pit stop is recommended."
        )

    elif tyre_life >= 20 or prediction > 95:

        st.warning("🟡 PREPARE TO PIT")

        st.write(
            "Performance is beginning to decrease. Consider pitting within the next few laps."
        )

    else:

        st.success("🟢 CONTINUE STINT")

        st.write(
            "Current tyre condition is healthy. Staying out is the recommended strategy."
        )

    st.divider()

    # ------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------

    st.subheader("📊 Feature Importance")

    image_path = "outputs/figures/feature_importance.png"

    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.info("Feature importance plot not found.")

    st.divider()

    # ------------------------------------------------
    # MODEL INFO
    # ------------------------------------------------

    st.subheader("🤖 Model Information")

    info1, info2, info3 = st.columns(3)

    with info1:
        st.info("""
**Model**

Random Forest Regressor
""")

    with info2:
        st.info("""
**Training Data**

FastF1 Dataset

2022–2024 Seasons
""")

    with info3:
        st.info("""
**Features**

10 Engineered Features

Target: Lap Time
""")

    st.divider()

    st.caption(
        "Built using FastF1 • Scikit-Learn • Streamlit | F1 Race Strategy Intelligence System"
    )