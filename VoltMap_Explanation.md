# VoltMap India — Comprehensive Platform Documentation

VoltMap India is a state-of-the-art predictive analytics platform designed to solve geospatial infrastructure imbalances for the Electric Vehicle (EV) market in India. By analyzing disparate data sources ranging from Indian census figures to rolling vehicle registrations, VoltMap delivers prescriptive recommendations for policy planners, governments, and private charging providers.

This document serves as the complete guide to understanding every page, graph, metric, and machine learning scenario within the VoltMap application.

---

## 1. Trajectory Forecast (`/`)
**Purpose:** To provide dynamic, macro-economic insights into India's monthly EV registrations dynamically from the present day up to **December 2050**.

### Interactive Graph: *Monthly EV Registration Forecast*
- **Behavior:** This dual-model Area Chart plots predictions from two separate Machine Learning models over time. Users can interact with the **Forecast Timeline** slider (a custom dual-thumb slider) to seamlessly drag and bound the start and end dates between 2025 and 2050. The chart instantly re-fetches and updates the rendering. 
- **Visuals:** 
  - **Sky Blue Area:** Represents the **Prophet** model.
  - **Neon Violet Area (Dashed):** Represents the **Polynomial Ridge** model.
  - **Dotted Vertical Line:** Marks the year **2030**, serving as a visual reference point for the Indian government's 2030 EV adoption targets.

### Key Metrics & Numbers:
- **[Model] · 2050 Target (e.g., 85K EVs):** The estimated raw monthly volume of EV registrations at the very end of your selected timeline. It tells you exactly how many EVs will be hitting the roads *per month* by that future date.
- **Forecast Growth (e.g., +210%):** The percentage increase in monthly EV registrations from the beginning of your selected timeline to the end. It tells you the aggregate growth velocity.
- **Data Points:** The total number of monthly intervals currently being rendered on the graph based on your slider selection.

---

## 2. Geospatial Gap Analysis (`/gap-map`)
**Purpose:** A geographical and statistical breakdown of where EV charging infrastructure is severely lacking compared to EV adoption demand.

### Interactive UI: *State Cards & Filters*
- **Behavior:** Presents an exhaustive array of state cards classified by prioritization. Users can use the **Sort Options** to rank states by Gap Score, Population, Demand, or Supply. They can also use the **Filters** to isolate states by their severity tier (Critical, Elevated, Balanced).

### Key Metrics & Numbers (per state):
- **Gap Score (e.g., 34.5):** The ultimate infrastructure deficit index. It is calculated as `Demand Score - Supply Score`. A **higher number means a worse deficit** (supply is failing to meet demand).
- **Demand Score:** A dimensionless composite index reflecting EV registration growth, population size, and urban density. Higher = surging EV adoption.
- **Supply Score:** A dimensionless composite index reflecting the existing operational charging infrastructure capacity relative to the state's EV fleet size. Higher = robust charging network.
- **Supply Coverage (e.g., 45%):** A progress bar showing supply as a percentage of demand. `100%` means infrastructure is fully capable of handling current EVs. Lower percentages highlight severe under-capacity.
- **Pop (e.g., 125.0M):** The state's human population in millions. Used by planners to weight the human impact of an infrastructure gap.
- **Stations per million:** The raw number of public EV chargers available per 1 million residents. A standard global metric for infrastructure density.

### Top-Level Dataset KPIs:
- **Critical Deficit:** Count of states with a Gap Score > 30. These require immediate emergency infrastructure funding.
- **Elevated Risk:** Count of states with a Gap Score between 15–30. These require steady investment to prevent critical failure.
- **Balanced Coverage:** Count of states with a Gap Score ≤ 15. Supply is currently meeting demand.

---

## 3. Scenario Simulator (`/simulator`)
**Purpose:** A prescriptive sandbox allowing policy planners to test hypothetical "what-if" infrastructure investments and market scenarios.

### Policy Parameters (The Controls):
- **Demand Growth Multiplier (e.g., 1.5×):** Simulates accelerated or conservative EV adoption relative to the historical baseline. A `1.5×` multiplier forces the algorithm to simulate demand growing 50% faster than expected.
- **Charging Station Budget (e.g., 50,000 stations):** The absolute number of new physical charging stations the government/planner is proposing to build by the target year.
- **Target Year (2025–2050):** The future date for which the simulation outcomes and gap deficits are resolved.
- **Focus States:** Selecting specific states tells the algorithm to prioritize distributing the Charging Station Budget to those territories first before allocating leftovers nationally.

### Interactive Graph: *Before / After Gap Score*
- **Behavior:** A horizontal Bar Chart comparing each state's EV infrastructure gap score before and after applying your simulation parameters.
- **Visuals:** 
  - **Crimson (Red) Bars:** The state's *current* infrastructure gap.
  - **Emerald (Green) Bars:** The state's *projected* infrastructure gap after the new budget is built and new demand is realized. A shorter green bar means your policy successfully mitigated the crisis.

### Key Metrics & Numbers:
- **Baseline Forecast (e.g., 1.24M EVs):** The expected total annual EV registrations for the Target Year under *normal* historical growth trends.
- **Scenario Forecast (e.g., 2.22M EVs):** The new projected total EV registrations after your Demand Growth Multiplier is factored in.
- **Gap Reduction (e.g., 45.0%):** Measures whether your Budget successfully met the Scenario demand. A positive green percentage means the overall national supply gap narrowed by that amount. 
- **Gap Increase (e.g., +250.0%):** If your simulated Demand Growth Multiplier is extremely high and your Budget is too low, demand outpaces supply. The UI dynamically flips to a red "Gap Increase" to warn planners that the deficit has worsened.

---

## 4. Analytics (`/analytics`)
**Purpose:** High-level executive overview of historical context and long-term annual projections.

### Graphs & Behavior:
- **Graph 1: Historical EV Growth (Area Chart):** Plots actual, verified EV registration counts across historical years. Displays the real-world baseline prior to any forecasting. 
- **Graph 2: Ridge Model Forecast (Grouped Bar Chart):** Displays the Ridge Regression's annual aggregate predictions. 
  - **Gold Bars:** The central predicted forecast.
  - **Pink/Green Bars:** The lower and upper statistical confidence bounds, showing the margin of error.

### Key Metrics & Numbers:
- **Historical Records:** The number of historical years analyzed to train the models.
- **Forecast at 2030 / 2050 (e.g., 4.5M EVs):** The absolute number of EVs predicted to be registered *annually* by that specific year.

---

## 5. Machine Learning Models & Scenarios

The frontend connects flawlessly to a Python FastAPI backend powered by pre-compiled `pickle` models trained on Indian `Vahan` vehicle registration datasets. 

1. **Facebook Prophet Extrapolation:** This model uses time-series decomposition. It is highly sensitive to seasonality (e.g., vehicle buying spikes during Indian festival seasons like Diwali) and incorporates exogenous macro "shocks" (like FAME-II subsidy implementations or COVID-19 demand plunges). It provides highly stable medium-term forecasting.
2. **Polynomial Ridge Regression:** Configured to handle chronological bounds cleanly extending into 2050. It applies an L2 regularization penalty to prevent the polynomial curves from exploding into unrealistic exponential numbers when forecasting decades into the future. It captures the non-linear "S-curve" adoption phase of new technologies perfectly.

---

## 6. Conclusion

VoltMap India bridges the gap between complex machine learning data science and actionable civic policy. By unifying historical trajectory forecasting with geospatial supply-demand modeling, the platform ensures that stakeholders do not just see *where* EV adoption is heading, but explicitly *how many* charging stations must be built, *where* they must be placed, and *when* they are needed to prevent critical infrastructure gridlock by 2050.
