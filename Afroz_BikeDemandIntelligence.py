import io
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="Bike Demand Intelligence",
    page_icon="🚲",
    layout="wide"
)

DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


@st.cache_data
def load_data():
    """Download and load the UCI Bike Sharing daily dataset."""
    with urllib.request.urlopen(DATA_URL, timeout=30) as response:
        data = response.read()

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open("day.csv") as f:
            df = pd.read_csv(f)

    df["dteday"] = pd.to_datetime(df["dteday"])
    return df


@st.cache_resource
def train_model(df):
    """Train a simple demand prediction model using chronological data."""
    features = [
        "season", "yr", "mnth", "holiday", "weekday",
        "workingday", "weathersit", "temp", "atemp",
        "hum", "windspeed"
    ]
    target = "cnt"

    data = df.sort_values("dteday").copy()
    split = int(len(data) * 0.80)

    train = data.iloc[:split]
    test = data.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=10,
        min_samples_leaf=2,
        n_jobs=-1
    )

    model.fit(train[features], train[target])
    predictions = model.predict(test[features])

    metrics = {
        "MAE": mean_absolute_error(test[target], predictions),
        "RMSE": np.sqrt(mean_squared_error(test[target], predictions)),
        "R2": r2_score(test[target], predictions)
    }

    importance = (
        pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        })
        .sort_values("Importance", ascending=False)
    )

    test_results = test[["dteday", "cnt"]].copy()
    test_results["Predicted"] = predictions

    return model, metrics, importance, test_results


def make_insights(df, importance):
    """Create transparent, data-grounded business insights."""
    monthly = df.groupby(df["dteday"].dt.month)["cnt"].mean()
    best_month = int(monthly.idxmax())
    worst_month = int(monthly.idxmin())

    seasonal = df.groupby("season")["cnt"].mean()
    best_season = int(seasonal.idxmax())
    worst_season = int(seasonal.idxmin())

    weather = df.groupby("weathersit")["cnt"].mean()
    best_weather = int(weather.idxmax())
    worst_weather = int(weather.idxmin())

    peak_day = df.loc[df["cnt"].idxmax(), "dteday"].strftime("%d %b %Y")
    peak_count = int(df["cnt"].max())

    top_driver = importance.iloc[0]["Feature"]

    return {
        "best_month": best_month,
        "worst_month": worst_month,
        "best_season": best_season,
        "worst_season": worst_season,
        "best_weather": best_weather,
        "worst_weather": worst_weather,
        "peak_day": peak_day,
        "peak_count": peak_count,
        "top_driver": top_driver
    }


def main():
    st.title("🚲 Bike Demand Intelligence")
    st.caption(
        "A simple Python Data Science + AI/ML decision-support project "
        "built around the IBM SkillsBuild project methodology."
    )

    try:
        df = load_data()
    except Exception as exc:
        st.error("The dataset could not be downloaded. Check your internet connection.")
        st.code(str(exc))
        st.stop()

    model, metrics, importance, test_results = train_model(df)
    insights = make_insights(df, importance)

    # -------------------------
    # Executive Overview
    # -------------------------
    st.header("1. Executive Overview")
    st.write("What is happening?")

    total_rentals = int(df["cnt"].sum())
    avg_daily = float(df["cnt"].mean())
    best_day = insights["peak_day"]
    best_day_count = insights["peak_count"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rentals", f"{total_rentals:,}")
    c2.metric("Average Daily Rentals", f"{avg_daily:,.0f}")
    c3.metric("Highest Daily Rentals", f"{best_day_count:,}")
    c4.metric("Model R²", f"{metrics['R2']:.2f}")

    trend = df.groupby("dteday", as_index=False)["cnt"].sum()
    fig = px.line(
        trend,
        x="dteday",
        y="cnt",
        title="Daily Bike Rental Trend",
        labels={"dteday": "Date", "cnt": "Rentals"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Trends
    # -------------------------
    st.header("2. Trend Analysis")
    st.write("How is demand changing?")

    monthly = (
        df.assign(Month=df["dteday"].dt.month)
        .groupby("Month", as_index=False)["cnt"]
        .mean()
    )

    fig_month = px.bar(
        monthly,
        x="Month",
        y="cnt",
        title="Average Daily Rentals by Month",
        labels={"cnt": "Average Rentals"}
    )
    st.plotly_chart(fig_month, use_container_width=True)

    seasonal = df.groupby("season", as_index=False)["cnt"].mean()
    fig_season = px.bar(
        seasonal,
        x="season",
        y="cnt",
        title="Average Rentals by Season",
        labels={"season": "Season Code", "cnt": "Average Rentals"}
    )
    st.plotly_chart(fig_season, use_container_width=True)

    # -------------------------
    # Drivers
    # -------------------------
    st.header("3. Driver Analysis")
    st.write("Which factors are most associated with demand in the prediction model?")

    fig_imp = px.bar(
        importance.head(8).sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Model Features"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.info(
        f"The strongest model feature in this run is **{insights['top_driver']}**. "
        "Feature importance indicates predictive contribution in this model; "
        "it does not by itself prove causation."
    )

    # -------------------------
    # Risk / Opportunity
    # -------------------------
    st.header("4. Risk & Opportunity")
    st.write("What could go wrong, and where could demand grow?")

    r1, r2 = st.columns(2)

    with r1:
        st.subheader("Risk")
        st.write(
            f"Demand is not uniform across the year. The lowest average-demand "
            f"month in this dataset is month **{insights['worst_month']}**, "
            f"which can create a planning risk if staffing or bike availability "
            "is not adjusted for lower-demand periods."
        )

    with r2:
        st.subheader("Opportunity")
        st.write(
            f"The highest average-demand month is month **{insights['best_month']}**. "
            "This provides an opportunity to prepare inventory, staffing and "
            "availability ahead of stronger-demand periods."
        )

    # -------------------------
    # ML Prediction
    # -------------------------
    st.header("5. AI / ML Demand Prediction")
    st.write("A Random Forest regression model predicts daily rental demand.")

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{metrics['MAE']:,.0f}")
    m2.metric("RMSE", f"{metrics['RMSE']:,.0f}")
    m3.metric("R²", f"{metrics['R2']:.2f}")

    fig_pred = px.line(
        test_results,
        x="dteday",
        y=["cnt", "Predicted"],
        title="Actual vs Predicted Demand on Test Data",
        labels={"value": "Rentals", "dteday": "Date"}
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # -------------------------
    # Action
    # -------------------------
    st.header("6. Action")
    st.write("What should management do?")

    actions = [
        f"Prepare capacity and staffing for stronger demand around month {insights['best_month']}.",
        f"Review resource levels for lower-demand periods, especially month {insights['worst_month']}.",
        f"Monitor the model's strongest predictive factor, {insights['top_driver']}, when planning demand.",
        "Use the prediction model as a planning aid rather than as a replacement for operational judgment."
    ]

    for action in actions:
        st.markdown(f"- {action}")

    # -------------------------
    # Methodology
    # -------------------------
    with st.expander("Project Methodology"):
        st.markdown(
            """
            **Data:** UCI Bike Sharing Dataset.

            **KPI:** total rentals and average daily rentals.

            **Trend analysis:** daily, monthly and seasonal demand patterns.

            **Driver analysis:** Random Forest feature importance.

            **Prediction:** chronological 80/20 train-test split with Random Forest Regression.

            **Risk/Opportunity:** demand variability and high-demand periods are translated
            into planning insights.

            **Action:** recommendations are generated directly from the observed analytical
            results.
            """
        )

    st.caption(
        "Dataset source: UCI Machine Learning Repository, Bike Sharing Dataset. "
        "https://archive.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset"
    )


if __name__ == "__main__":
    main()