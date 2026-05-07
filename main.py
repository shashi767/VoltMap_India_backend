from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from models import (
    forecast_prophet, forecast_ridge, forecast_yearly,
    run_scenario, gap_df, ev_model
)

app = FastAPI(title="VoltMap India API", version="2.0.0")

# ── CORS ──────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://volt-map-india-frontend.vercel.app/","http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schemas ───────────────────────────────────────────────────────
class ForecastRequest(BaseModel):
    start_date : str        # "2025-01-01"
    end_date   : str        # "2050-12-01"
    model      : str = "both"   # "prophet" | "ridge" | "both"

class ScenarioRequest(BaseModel):
    growth_multiplier : float      = 1.4
    budget            : int        = 50000
    focus_states      : List[str]  = []
    target_year       : int        = 2030

class YearlyForecastRequest(BaseModel):
    start_year : int = 2025
    end_year   : int = 2050

# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "VoltMap India API v2 running", "forecast_horizon": "2050"}


@app.post("/api/forecast")
def forecast(req: ForecastRequest):
    """
    Monthly EV forecast using Prophet and/or Ridge regression.
    Accepts any date range — dynamically computed.
    """
    if req.start_date >= req.end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")
    result = {}
    if req.model in ["prophet", "both"]:
        result["prophet"] = forecast_prophet(req.start_date, req.end_date)
    if req.model in ["ridge", "both"]:
        result["ridge"]   = forecast_ridge(req.start_date, req.end_date)
    return result


@app.get("/api/yearly-forecast")
@app.post("/api/yearly-forecast")
def yearly_forecast(req: YearlyForecastRequest = None):
    """
    Annual EV registration totals, dynamically computed via Ridge model.
    Supports GET (defaults 2025-2050) and POST with custom range.
    """
    start = req.start_year if req else 2025
    end   = req.end_year   if req else 2050
    if start < 2025 or end > 2050 or start > end:
        raise HTTPException(
            status_code=400,
            detail="year range must be between 2025 and 2050"
        )
    return forecast_yearly(start, end)


@app.get("/api/gap-analysis")
def gap_analysis():
    """State-level demand vs supply gap scores."""
    STATE_COL = "State name" if "State name" in gap_df.columns else "state"
    return [
        {
            "state"               : row[STATE_COL],
            "demand_score"        : round(float(row["demand_score"]), 1),
            "supply_score"        : round(float(row["supply_score"]), 1),
            "gap_score"           : round(float(row["gap_score"]), 1),
            "population"          : int(row["total_population"]),
            "stations_per_million": round(float(row.get("stations_per_million", 0)), 2),
        }
        for _, row in gap_df.sort_values("gap_score", ascending=False).iterrows()
    ]


@app.get("/api/historical")
def historical():
    """Raw historical monthly EV registrations."""
    return [
        {
            "date"  : row["Date"].strftime("%Y-%m-%d"),
            "value" : int(row["total"]),
            "year"  : int(row["year"]),
            "month" : int(row["month"]),
        }
        for _, row in ev_model.iterrows()
    ]


@app.get("/api/yearly-summary")
def yearly_summary():
    """
    Annual forecast summary 2025-2050 (dynamic, from Ridge model).
    Kept for backwards compatibility — delegates to forecast_yearly().
    """
    return forecast_yearly(2025, 2050)


@app.post("/api/scenario")
def scenario(req: ScenarioRequest):
    """
    Infrastructure scenario simulator.
    Supports any target year from 2025 to 2050.
    """
    if req.target_year < 2025 or req.target_year > 2050:
        raise HTTPException(
            status_code=400,
            detail="target_year must be between 2025 and 2050"
        )
    return run_scenario(
        req.growth_multiplier,
        req.budget,
        req.focus_states,
        req.target_year,
    )


@app.get("/api/states-list")
def states_list():
    """Sorted list of all Indian states in the dataset."""
    STATE_COL = "State name" if "State name" in gap_df.columns else "state"
    return sorted(gap_df[STATE_COL].dropna().tolist())