# Hourly Renewable Energy Production Prediction Engine

**Status: core pipeline + API + UI built, tested end to end, and deployed live.**

## Live demo

- **API** (FastAPI, interactive docs): https://energy-prediction-api-d8wf.onrender.com/docs
- **UI** (Gradio): https://huggingface.co/spaces/Sapphire-Moon/hourly-energy-prediction-ui

Note: the free hosting tier spins down after ~15 minutes of inactivity, so the first request after a while can take 30-60 seconds to wake up. That's normal, not a bug.

## What this is

Predicts **next-hour** Wind/Solar energy production from recent historical readings and calendar
information, using a Gradient Boosting model trained on ~51K hourly records (2020-2025).

The UI has two modes:
- **Historical Replay** - pick a real date/source/hour from the dataset, see predicted vs. actual.
- **What-If / Live Prediction** - enter any values directly, for any year, to see the model's genuine forward-looking prediction. This tab has no date restriction, because the model itself has no concept of "past" vs "future" - it just answers "given these inputs, what's the next hour?" The Historical Replay tab is bounded to 2020-2025 purely because that's the only period we have recorded, verifiable data to compare against.

## Running it from scratch

### 0. Prerequisites - installing Python

You need Python 3.10 or 3.11 with `pip`. If you already have it, skip to step 1. Otherwise:

**Windows:**
1. Download the Python 3.11 installer from https://www.python.org/downloads/
2. Run it - on the first screen, check **"Add python.exe to PATH"** before clicking Install.
3. Close and reopen your terminal so PATH updates.
4. Verify: `python --version` and `pip --version`.
5. Note: on Windows, use `python` instead of `python3` in every command below.

**macOS:**
1. Download the Python 3.11 installer from https://www.python.org/downloads/ and run it (or `brew install python@3.11`).
2. Verify: `python3 --version` and `pip3 --version`.

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install python3 python3-pip python3-venv
python3 --version
```

### 1. Clone the repo and enter the project folder
```bash
git clone https://github.com/Sapphire-Moon/machine_learning_portfolio.git
cd machine_learning_portfolio/ml-energy-prediction-engine
```

### 2. (Recommended) create a virtual environment
```bash
python3 -m venv venv
# Mac/Linux:
source venv/bin/activate
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
Installs pandas, scikit-learn, TensorFlow, FastAPI, Gradio, and everything else needed. TensorFlow is the largest download - this can take a few minutes.

### 4. Train the model
```bash
python3 src/train.py
```
**Run this from the project root** (the folder containing `src/`, `data/`, etc.), not from inside `src/` - the scripts look for `data/Energy Production Dataset.csv` relative to wherever you launch Python from.

Model artifacts (`models/production_model.joblib`, `models/preprocessor.joblib`, `models/feature_config.json`) are already included, so this step is optional unless you want to regenerate them - training is deterministic and reproduces the exact same numbers.

### 5. Start the prediction API (keep this terminal running)
```bash
cd api
uvicorn main:app --reload --port 8000
```
Verify at **http://127.0.0.1:8000/docs**, or:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"source":"Wind","day_name":"Monday","start_hour":14,"day_of_year":200,"year":2025,"production_lag_1":7200,"production_lag_24":6800,"production_lag_168":6500}'
```

### 6. Start the Gradio UI (in a NEW second terminal - leave the API running)
```bash
cd app
python3 app.py
```
Open **http://127.0.0.1:7860**. Both tabs (Historical Replay and What-If/Live Prediction) call the API from step 5.

### 7. Run the tests
```bash
python3 -m pytest tests/ -v
```
All 7 should pass.

### 8. Try the external-consumer example
```bash
python3 examples/consume_api.py
```

### Stopping everything
`Ctrl+C` in each terminal.

## Deployment (already done - see Live demo above)

- **API** deployed to [Render](https://render.com): Root Directory `ml-energy-prediction-engine/api`, Build Command `pip install -r requirements.txt`, Start Command `uvicorn main:app --host 0.0.0.0 --port $PORT`. Uses a slim `api/requirements.txt` (no TensorFlow/Gradio - the API only needs to serve the saved model).
- **UI** deployed to [Hugging Face Spaces](https://huggingface.co/spaces): SDK = Gradio, `app_file: app/app.py`, with a `PREDICTION_API_URL` variable set to the Render URL above. Uses a slim `app/requirements.txt`.

### Troubleshooting
- **`FileNotFoundError: data/Energy Production Dataset.csv`** - you ran a script from inside `src/` instead of the project root. `cd` back to the root and run `python3 src/train.py`.
- **`ModuleNotFoundError`** - dependencies aren't installed, or your virtual environment isn't activated.
- **`Address already in use`** - something else is using port 8000 or 7860. Change the port and update the API URL accordingly.
- **Gradio UI says "could not reach the prediction API"** - the API terminal isn't running, or `PREDICTION_API_URL` is wrong.

## Project structure
