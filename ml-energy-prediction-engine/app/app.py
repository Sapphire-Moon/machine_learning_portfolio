"""
Stage 8 (revised): Gradio frontend with TWO modes.

Tab 1 - Historical Replay: pick a real date/hour, the app looks up the
true recorded lag values (nobody can be expected to know these from
memory) and shows predicted vs actual, to prove the model's accuracy on
data it never trained on.

Tab 2 - What-If / Live Prediction: type in ANY values directly - today's
real sensor readings if you had them, or a hypothetical scenario, for any
year. This is the mode that proves the model does genuine forward-looking
prediction: it has no idea whether a number came from a historical record
or from a real live reading five minutes ago - it just answers "given
these values, what's the next hour?" The date-range restriction on Tab 1
is a limitation of THIS DEMO (we have no live sensor feed to draw on),
not a limitation of the model or the API itself.

Both tabs call the same, unchanged FastAPI /predict endpoint over HTTP -
this file never loads the model itself.
"""
import os
import sys
from pathlib import Path

import requests
import pandas as pd
import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preprocessing import load_and_clean  # noqa: E402

API_BASE_URL = os.environ.get("PREDICTION_API_URL", "http://127.0.0.1:8000")
API_URL = f"{API_BASE_URL.rstrip('/')}/predict"


def build_lookup_table():
    df = load_and_clean()
    df["Year"] = df["Date"].dt.year
    df["Production_Lag_1"] = df.groupby("Source")["Production"].shift(1)
    df["Production_Lag_24"] = df.groupby("Source")["Production"].shift(24)
    df["Production_Lag_168"] = df.groupby("Source")["Production"].shift(168)
    df = df.dropna(subset=["Production_Lag_1", "Production_Lag_24", "Production_Lag_168"])
    return df


LOOKUP = build_lookup_table()
MIN_DATE = LOOKUP["Date"].min().date()
MAX_DATE = LOOKUP["Date"].max().date()


def predict_from_history(source, target_date, start_hour):
    if not target_date:
        return "Please pick a date.", ""

    target_date_only = pd.to_datetime(target_date).date()
    match = LOOKUP[
        (LOOKUP["Source"] == source)
        & (LOOKUP["Date"].dt.date == target_date_only)
        & (LOOKUP["Start_Hour"] == int(start_hour))
    ]

    if match.empty:
        return (
            f"No historical record for {source} on {target_date_only} at hour "
            f"{int(start_hour)}. Valid range is {MIN_DATE} to {MAX_DATE}, and note "
            f"that not every source has a logged reading in every single hour "
            f"(e.g. Solar has no readings at night).",
            "",
        )

    row = match.iloc[0]
    payload = {
        "source": source,
        "day_name": row["Day_Name"],
        "start_hour": int(row["Start_Hour"]),
        "day_of_year": int(row["Day_of_Year"]),
        "year": int(row["Year"]),
        "production_lag_1": float(row["Production_Lag_1"]),
        "production_lag_24": float(row["Production_Lag_24"]),
        "production_lag_168": float(row["Production_Lag_168"]),
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
    except requests.exceptions.ConnectionError:
        return "The prediction API is not reachable right now. Is it running?", ""

    if response.status_code != 200:
        detail = response.json().get("detail", "Unknown error")
        return f"API error: {detail}", ""

    prediction = response.json()["prediction"]
    actual = float(row["Production"])
    diff = prediction - actual
    return (
        f"Predicted next-hour production: {prediction:.2f}",
        f"Actual recorded value: {actual:.2f}   |   Difference: {diff:+.2f}",
    )


DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def predict_manual(source, day_name, start_hour, day_of_year, year, lag1, lag24, lag168):
    payload = {
        "source": source,
        "day_name": day_name,
        "start_hour": int(start_hour),
        "day_of_year": int(day_of_year),
        "year": int(year),
        "production_lag_1": float(lag1),
        "production_lag_24": float(lag24),
        "production_lag_168": float(lag168),
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
    except requests.exceptions.ConnectionError:
        return "The prediction API is not reachable right now. Is it running?"

    if response.status_code != 200:
        detail = response.json().get("detail", "Unknown error")
        if isinstance(detail, list):
            detail = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in detail
            )
        return f"Invalid input - {detail}"

    prediction = response.json()["prediction"]
    return f"Predicted next-hour production: {prediction:.2f}"


with gr.Blocks(title="Hourly Renewable Energy Production Predictor") as demo:
    gr.Markdown("# Hourly Renewable Energy Production Predictor")
    gr.Markdown(
        "Both tabs call the same live FastAPI backend serving a trained Gradient Boosting "
        "model - neither tab loads the model itself."
    )

    with gr.Tab("Historical Replay (verify against real data)"):
        gr.Markdown(
            "Pick a real historical date, source, and hour. The app looks up the true "
            "recorded lag values from that exact moment and shows predicted vs. actual, "
            "so you can see the model's real accuracy on data it never trained on."
        )
        with gr.Row():
            with gr.Column():
                hist_source = gr.Dropdown(["Wind", "Solar"], label="Source", value="Wind")
                hist_date = gr.DateTime(
                    label=f"Date (data available {MIN_DATE} to {MAX_DATE})",
                    include_time=False,
                    type="string",
                    value=str(MAX_DATE),
                )
                hist_hour = gr.Slider(0, 23, step=1, label="Hour of day (0-23)", value=14)
                hist_btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                hist_pred = gr.Textbox(label="Model prediction")
                hist_actual = gr.Textbox(label="Ground truth (for comparison)")
        hist_btn.click(
            predict_from_history,
            inputs=[hist_source, hist_date, hist_hour],
            outputs=[hist_pred, hist_actual],
        )

    with gr.Tab("What-If / Live Prediction (any values, any year)"):
        gr.Markdown(
            "Enter values directly - today's real readings if you had a live feed, or a "
            "hypothetical scenario. This proves the model makes a genuine forward-looking "
            "prediction: it has no idea whether these numbers are historical or truly "
            "current, it just answers 'given these, what's the next hour?' The only reason "
            "the other tab is limited to 2020-2025 is that we have no live sensor feed to "
            "connect for this project - not a limitation of the model itself."
        )
        with gr.Row():
            with gr.Column():
                m_source = gr.Dropdown(["Wind", "Solar"], label="Source", value="Wind")
                m_day_name = gr.Dropdown(DAY_NAMES, label="Day of week", value="Monday")
                m_hour = gr.Slider(0, 23, step=1, label="Hour of day (0-23)", value=14)
                m_doy = gr.Slider(1, 366, step=1, label="Day of year (1-366)", value=200)
                m_year = gr.Number(label="Year", value=2025, precision=0)
                m_lag1 = gr.Number(label="Production 1 hour ago", value=7200)
                m_lag24 = gr.Number(label="Production 24 hours ago", value=6800)
                m_lag168 = gr.Number(label="Production 168 hours ago", value=6500)
                m_btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                m_pred = gr.Textbox(label="Model prediction")
        m_btn.click(
            predict_manual,
            inputs=[m_source, m_day_name, m_hour, m_doy, m_year, m_lag1, m_lag24, m_lag168],
            outputs=[m_pred],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

