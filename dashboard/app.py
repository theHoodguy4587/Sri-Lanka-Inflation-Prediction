import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "world_bank_data_2025.csv"
DEFAULT_API_URL = "http://api:8000/predict"


st.set_page_config(
    page_title="Inflation Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #f1f5f9;
    }
    .stApp, .stApp p, .stApp label, .stApp span, .stApp div, .stApp li {
        color: #f1f5f9;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 2px solid #14b8a6;
        color: #f1f5f9;
    }
    [data-testid="stSidebar"] * {
        color: #f1f5f9;
    }
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea,
    .stSelectbox [data-baseweb="select"] > div,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        caret-color: #14b8a6;
        background-color: #f8fafc !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }
    .stTextInput input::placeholder,
    .stNumberInput input::placeholder,
    .stTextArea textarea::placeholder,
    [data-baseweb="input"] input::placeholder,
    [data-baseweb="textarea"] textarea::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }
    [data-baseweb="select"] * {
        color: #0f172a !important;
        background-color: #f8fafc !important;
    }
    [data-baseweb="select"] [role="option"] {
        color: #0f172a !important;
        background-color: #f8fafc !important;
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f172a 0%, #14b8a6 50%, #06b6d4 100%);
        color: white;
        box-shadow: 0 20px 50px rgba(20, 184, 166, 0.3);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .hero p {
        margin: 0.4rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    .card {
        background: linear-gradient(135deg, #1e293b 0%, #0f4c3a 100%);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid #14b8a6;
        color: #f1f5f9;
    }
    .card * {
        color: #f1f5f9;
    }
    .small-note {
        color: #94a3b8;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_country_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[[
        "country_name",
        "year",
        "Inflation (CPI %)",
        "GDP Growth (% Annual)",
        "Inflation (GDP Deflator, %)",
        "GDP per Capita (Current USD)",
    ]].copy()
    df.columns = [
        "Country",
        "Year",
        "Inflation_CPI",
        "GDP_Growth",
        "Inflation_GDP_Deflator",
        "GDP_per_Capita",
    ]
    df = df.dropna(subset=["Inflation_CPI"]).sort_values(["Country", "Year"])
    return df


@st.cache_data
def get_country_defaults(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values(["Country", "Year"])

    def last_valid(series: pd.Series) -> float:
        non_null = series.dropna()
        return non_null.iloc[-1] if not non_null.empty else float("nan")

    latest = df_sorted.groupby("Country", as_index=False).agg(
        GDP_Growth=("GDP_Growth", last_valid),
        Inflation_GDP_Deflator=("Inflation_GDP_Deflator", last_valid),
        GDP_per_Capita=("GDP_per_Capita", last_valid),
        Inflation_CPI=("Inflation_CPI", last_valid),
    )
    latest = latest.rename(columns={"Inflation_CPI": "Last_Observed_Inflation"})
    return latest.sort_values("Country")


@st.cache_data
def build_country_lag_defaults(df: pd.DataFrame) -> pd.DataFrame:
    lagged = df.copy()
    lagged["Inflation_Lag1"] = lagged.groupby("Country")["Inflation_CPI"].shift(1)
    lagged["Inflation_Lag2"] = lagged.groupby("Country")["Inflation_CPI"].shift(2)
    lagged["GDP_Growth_Lag1"] = lagged.groupby("Country")["GDP_Growth"].shift(1)
    lagged = lagged.sort_values(["Country", "Year"]).groupby("Country", as_index=False).tail(1)
    return lagged[["Country", "Inflation_Lag1", "Inflation_Lag2", "GDP_Growth_Lag1"]]


@st.cache_data
def get_country_bundle() -> pd.DataFrame:
    country_data = load_country_data()
    defaults = get_country_defaults(country_data)
    lags = build_country_lag_defaults(country_data)
    bundle = defaults.merge(lags, on="Country", how="left")
    return bundle


country_bundle = get_country_bundle()
country_options = country_bundle["Country"].tolist()

st.markdown(
    """
    <div class="hero">
      <h1>Inflation Forecast Dashboard</h1>
            <p>Choose a country, adjust the inputs, and get a clear inflation forecast instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Prediction Controls")
    api_url = st.text_input("API endpoint", value="https://global-inflation-prediction-api.onrender.com/predict")
    selected_country = st.selectbox("Country", country_options, index=0)
    st.caption("Only user-facing macro inputs are shown. Lag features are generated automatically from the selected country's history.")

selected_row = country_bundle.loc[country_bundle["Country"] == selected_country].iloc[0]

lag_values = {
    "Inflation_Lag1": float(selected_row["Inflation_Lag1"]) if pd.notna(selected_row["Inflation_Lag1"]) else 0.0,
    "Inflation_Lag2": float(selected_row["Inflation_Lag2"]) if pd.notna(selected_row["Inflation_Lag2"]) else 0.0,
    "GDP_Growth_Lag1": float(selected_row["GDP_Growth_Lag1"]) if pd.notna(selected_row["GDP_Growth_Lag1"]) else 0.0,
}

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Feature Inputs")
    st.caption("These controls are the inputs the user can understand directly. Lagged features are computed behind the scenes.")

    show_tips = st.toggle("Show input tips", value=False)
    if show_tips:
        st.info(
            "Adjust GDP metrics to test scenarios. The model uses the latest country history for lag features."
        )
        with st.expander("What do these inputs mean?"):
            st.markdown(
                """
                - **GDP Growth**: Annual percentage change in GDP.
                - **Inflation (GDP Deflator)**: Broad inflation measure based on GDP prices.
                - **GDP per Capita**: GDP per person in current USD.
                """
            )

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            gdp_growth = st.number_input(
                "GDP Growth",
                value=float(selected_row["GDP_Growth"]),
                step=0.1,
                format="%.2f",
                help="Annual percent change in GDP. Use scenario values to explore outcomes.",
            )
            inflation_deflator = st.number_input(
                "Inflation (GDP Deflator)",
                value=float(selected_row["Inflation_GDP_Deflator"]),
                step=0.1,
                format="%.2f",
                help="Inflation measured by the GDP deflator. Typically smoother than CPI.",
            )
            gdp_per_capita = st.number_input(
                "GDP per Capita (USD)",
                value=float(selected_row["GDP_per_Capita"]),
                step=100.0,
                format="%.2f",
                help="Income per person in current USD.",
            )
        with c2:
            st.info(
                "Lag features are derived automatically from the latest available observations for the chosen country."
            )
            with st.expander("View auto-generated lag features"):
                st.write(
                    {
                        "Inflation Lag 1": lag_values["Inflation_Lag1"],
                        "Inflation Lag 2": lag_values["Inflation_Lag2"],
                        "GDP Growth Lag 1": lag_values["GDP_Growth_Lag1"],
                    }
                )

        submitted = st.form_submit_button("Forecast Inflation", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Country Snapshot")
    snapshot = pd.DataFrame(
        {
            "Metric": ["GDP Growth", "GDP Deflator", "GDP per Capita", "Last Observed Inflation"],
            "Value": [
                selected_row["GDP_Growth"],
                selected_row["Inflation_GDP_Deflator"],
                selected_row["GDP_per_Capita"],
                selected_row["Last_Observed_Inflation"],
            ],
        }
    )
    st.dataframe(snapshot, use_container_width=True, hide_index=True)
    st.markdown("<div class='small-note'>These values are the latest observed data points for the selected country.</div>", unsafe_allow_html=True)

    gauge_placeholder = st.empty()
    result_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

if submitted:
    payload = {
        "Country": selected_country,
        "GDP_Growth": gdp_growth,
        "Inflation_GDP_Deflator": inflation_deflator,
        "GDP_per_Capita": gdp_per_capita,
        "Inflation_Lag1": lag_values["Inflation_Lag1"],
        "Inflation_Lag2": lag_values["Inflation_Lag2"],
        "GDP_Growth_Lag1": lag_values["GDP_Growth_Lag1"],
    }

    try:
        response = requests.post(api_url, json=payload, timeout=20)
        response.raise_for_status()
        predicted = float(response.json()["predicted_inflation"])

        with result_placeholder.container():
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Predicted Inflation", f"{predicted:.2f}%")
            metric_col2.metric("Country", selected_country)

            st.success("Prediction completed successfully.")

        with gauge_placeholder.container():
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=predicted,
                    number={"suffix": "%"},
                    delta={"reference": selected_row["Last_Observed_Inflation"] if pd.notna(selected_row["Last_Observed_Inflation"]) else 0},
                    gauge={
                        "axis": {"range": [None, max(10, predicted * 1.6, 20)]},
                        "bar": {"color": "#3b82f6"},
                        "steps": [
                            {"range": [0, 5], "color": "#e2e8f0"},
                            {"range": [5, 10], "color": "#cbd5e1"},
                            {"range": [10, max(10, predicted * 1.6, 20)], "color": "#94a3b8"},
                        ],
                    },
                    title={"text": "Forecasted Inflation"},
                )
            )
            fig.update_layout(height=320, margin={"l": 20, "r": 20, "t": 60, "b": 20})
            st.plotly_chart(fig, use_container_width=True)

        comparison = pd.DataFrame(
            {
                "Feature": ["GDP Growth", "GDP Deflator", "GDP per Capita", "Lag 1", "Lag 2", "GDP Growth Lag 1"],
                "Value": [
                    gdp_growth,
                    inflation_deflator,
                    gdp_per_capita,
                    lag_values["Inflation_Lag1"],
                    lag_values["Inflation_Lag2"],
                    lag_values["GDP_Growth_Lag1"],
                ],
            }
        )
        st.bar_chart(comparison.set_index("Feature"))

    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
    except KeyError:
        st.error("The API response did not include 'predicted_inflation'.")

st.caption("Tip: keep the API running with `uvicorn api.app:app --reload` before using the dashboard.")
