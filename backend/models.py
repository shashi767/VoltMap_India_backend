import pickle
import numpy as np
import pandas as pd
from pathlib import Path

ARTIFACTS = Path(__file__).parent / "artifacts"

# ── Load models once at startup ───────────────────────────────────────────
with open(ARTIFACTS / "prophet_model.pkl", "rb") as f:
    prophet_model = pickle.load(f)

with open(ARTIFACTS / "ridge_model.pkl", "rb") as f:
    ridge_model = pickle.load(f)

ev_model  = pd.read_csv(ARTIFACTS / "ev_model.csv",   parse_dates=["Date"])
gap_df    = pd.read_csv(ARTIFACTS / "gap_df.csv")

last_t = int(ev_model["t"].max())


# ── Forecast helpers ──────────────────────────────────────────────────────

def forecast_prophet(start_date: str, end_date: str):
    """Forecast using Prophet for a date range."""
    dates  = pd.date_range(start_date, end_date, freq="MS")
    df     = pd.DataFrame({"ds": dates})
    result = prophet_model.predict(df)
    return [
        {
            "date"  : row["ds"].strftime("%Y-%m-%d"),
            "value" : max(round(row["yhat"]), 0),
            "lower" : max(round(row["yhat_lower"]), 0),
            "upper" : round(row["yhat_upper"]),
            "model" : "Prophet"
        }
        for _, row in result.iterrows()
    ]


def forecast_ridge(start_date: str, end_date: str):
    """Forecast using Ridge for a date range."""
    dates = pd.date_range(start_date, end_date, freq="MS")
    rows  = []
    for i, d in enumerate(dates):
        t_val = last_t + i + 1
        rows.append({
            "t"     : t_val,
            "sin12" : np.sin(2 * np.pi * d.month / 12),
            "cos12" : np.cos(2 * np.pi * d.month / 12),
        })
    X    = pd.DataFrame(rows)
    pred = np.maximum(ridge_model.predict(X), 0)
    return [
        {
            "date"  : d.strftime("%Y-%m-%d"),
            "value" : round(float(p)),
            "lower" : round(float(p) * 0.78),
            "upper" : round(float(p) * 1.22),
            "model" : "Ridge"
        }
        for d, p in zip(dates, pred)
    ]


def forecast_yearly(start_year: int = 2025, end_year: int = 2050):
    """
    Dynamically compute yearly totals using the Ridge model.
    Returns a list of dicts with Year, Forecast, Lower, Upper (annual sums).
    This replaces the static yearly_fc.csv and works for any year range.
    """
    start = f"{start_year}-01-01"
    end   = f"{end_year}-12-01"
    monthly = forecast_ridge(start, end)

    yearly: dict[int, dict] = {}
    for row in monthly:
        yr = int(row["date"][:4])
        if yr not in yearly:
            yearly[yr] = {"Year": yr, "Forecast": 0, "Lower": 0, "Upper": 0}
        yearly[yr]["Forecast"] += row["value"]
        yearly[yr]["Lower"]    += row["lower"]
        yearly[yr]["Upper"]    += row["upper"]

    result = []
    for yr in sorted(yearly.keys()):
        d = yearly[yr]
        result.append({
            "Year"       : d["Year"],
            "Forecast"   : round(d["Forecast"]),
            "Lower"      : round(d["Lower"]),
            "Upper"      : round(d["Upper"]),
            "Forecast_M" : round(d["Forecast"] / 1e6, 2),
        })
    return result


def run_scenario(growth_multiplier: float,
                 budget: int,
                 focus_states: list,
                 target_year: int):
    """Run infrastructure scenario simulation (supports any year up to 2050)."""
    STATE_COL = "State name" if "State name" in gap_df.columns else "state"

    # ── Dynamically compute target year forecast using Ridge ──────────────
    try:
        monthly_fc = forecast_ridge(
            f"{target_year}-01-01", f"{target_year}-12-01"
        )
        base_fc = sum(m["value"] for m in monthly_fc)
    except Exception:
        # Final fallback: use yearly summary (also dynamic now)
        yearly = forecast_yearly(target_year, target_year)
        base_fc = yearly[0]["Forecast"] if yearly else 0

    scenario = base_fc * growth_multiplier

    # ── Distribute stations ───────────────────────────────────────────────
    gap_sim = gap_df.copy()
    
    if focus_states:
        n_focus = len(focus_states)
        n_other = max(len(gap_sim) - n_focus, 1)
        
        focus_share = int(budget * 0.60) // n_focus
        other_share = int(budget * 0.40) // n_other
        
        gap_sim["new_stations"] = gap_sim[STATE_COL].apply(
            lambda s: focus_share if s in focus_states else other_share
        )
    else:
        # If no focus states selected, distribute budget equally across all states
        n_total = len(gap_sim)
        equal_share = int(budget) // n_total
        gap_sim["new_stations"] = equal_share

    pop_m               = gap_sim["total_population"] / 1e6
    gap_sim["new_spm"]  = (
        gap_sim["stations_per_million"].fillna(0)
        + gap_sim["new_stations"] / pop_m
    )
    max_spm               = gap_sim["new_spm"].max()
    gap_sim["new_supply"] = gap_sim["new_spm"] / max_spm * 100
    gap_sim["new_demand"] = gap_sim["demand_score"] * growth_multiplier
    gap_sim["new_demand"] = gap_sim["new_demand"] / gap_sim["new_demand"].max() * 100
    gap_sim["new_gap"]    = (gap_sim["new_demand"] - gap_sim["new_supply"]).clip(lower=0)

    return {
        "baseline_forecast" : round(base_fc),
        "scenario_forecast" : round(scenario),
        "growth_multiplier" : growth_multiplier,
        "target_year"       : target_year,
        "states": [
            {
                "state"        : row[STATE_COL],
                "gap_before"   : round(float(row["gap_score"]), 1),
                "gap_after"    : round(float(row["new_gap"]), 1),
                "new_stations" : int(row["new_stations"]),
                "is_focus"     : row[STATE_COL] in focus_states,
            }
            for _, row in gap_sim.sort_values(
                "new_gap", ascending=False
            ).head(20).iterrows()
        ],
    }