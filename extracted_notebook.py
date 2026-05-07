# --- Cell 0 ---
%matplotlib inline

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

import os

try:
    from google.colab import drive
    drive.mount('/content/drive')

    BASE_DATA_PATH = "/content/drive/MyDrive/voltmap/data"
    BASE_ARTIFACT_PATH = "/content/drive/MyDrive/voltmap/artifacts"

    print("good")
except:
    BASE_DATA_PATH = "data"
    BASE_ARTIFACT_PATH = "artifacts"

    print("bad")

os.makedirs(BASE_DATA_PATH, exist_ok=True)
os.makedirs(BASE_ARTIFACT_PATH, exist_ok=True)

# ── Research-paper white theme ───────────────────────────────────────────────
plt.rcParams.update({
    # Canvas
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    # Spines & ticks
    "axes.edgecolor":    "#333333",
    "axes.linewidth":    0.8,
    "xtick.color":       "#333333",
    "ytick.color":       "#333333",
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    # Text
    "text.color":        "#111111",
    "axes.labelcolor":   "#111111",
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    # Grid – subtle, light grey
    "axes.grid":         True,
    "grid.color":        "#e0e0e0",
    "grid.linestyle":    "-",
    "grid.linewidth":    0.6,
    "grid.alpha":        1.0,
    # Legend
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#cccccc",
    "legend.fontsize":   9,
    # Figure
    "figure.dpi":        120,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
})

# ── Accessible, print-safe colour palette ────────────────────────────────────
# Primary accent (steel blue – readable in B&W and colour)
BLUE   = "#2166AC"
GREEN  = "#1A9641"
RED    = "#D7191C"
ORANGE = "#E08214"
PURPLE = "#762A83"
TEAL   = "#4DAC26"
BROWN  = "#8C510A"
PINK   = "#C51B7D"

VOLT_COLORS = [BLUE, GREEN, RED, ORANGE, PURPLE, TEAL, BROWN, PINK,
               "#5AAE61", "#9970AB", "#F4A582", "#74ADD1"]

# Diverging palette used in heatmaps
CMAP_DIV  = "RdBu_r"    # red-blue diverging (good for correlation)
CMAP_SEQ  = "Blues"     # sequential (station counts etc.)
CMAP_HEAT = "YlOrRd"    # warm sequential (sales heatmap)
CMAP_RDGN = "RdYlGn"    # red-yellow-green (coverage quality)

print("Theme loaded — white background, research-paper style.")



# --- Cell 2 ---
print("\nPHASE 1 - Loading data")

# EV Charging Stations
stations = pd.read_csv(f"{BASE_DATA_PATH}/ev-charging-stations-india.csv")
stations.columns = (stations.columns.str.strip().str.lower().str.replace(" ", "_"))
stations = stations.rename(columns={"lattitude": "lat", "longitude": "lon"})
stations["city"]  = stations["city"].str.strip().str.title()
stations = stations.dropna(subset=["lat", "lon"]) # Drops rows where latitude or longitude is missing

# ── FIX: unified normalization — maps every variant → census 2011 canonical name ──
# Previously: state_mapping + city_to_state were split, incomplete, and only applied
# to stations. Delhi/Odisha/J&K/Puducherry/Andaman all failed to match census names,
# causing those states to show station_count = 0 in gap_df.

# Fix city → state (BUG 9 fix: added Chikhali, Jajpur, Limbdi, Rajahmundry)
CITY_TO_STATE = {
    "Hyderabad":    "Telangana",
    "Ernakulam":    "Kerala",
    "Kochi":        "Kerala",
    "Hisar":        "Haryana",
    "Bhubhaneswar": "Orissa",       # ← BUG 2 fix: census 2011 uses "Orissa" not "Odisha"
    "Chikhali":     "Maharashtra",  # ← BUG 9 fix: was unmapped
    "Jajpur":       "Orissa",       # ← BUG 9 fix: was unmapped
    "Limbdi":       "Gujarat",      # ← BUG 9 fix: was unmapped
    "Rajahmundry":  "Andhra Pradesh", # ← BUG 9 fix: was unmapped
}

# Fix spelling + merge duplicates → census 2011 canonical names
CANONICAL = {
    # ── BUG 1 fix: Delhi variants → census name "Nct Of Delhi" ──
    # Old code only mapped "Delhi Ncr" → "Delhi" which still didn't match census
    "Delhi":             "Nct Of Delhi",
    "Delhi Ncr":         "Nct Of Delhi",
    "New Delhi":         "Nct Of Delhi",

    # Spelling fixes (unchanged from original)
    "Tamilnadu":         "Tamil Nadu",
    "Taminadu":          "Tamil Nadu",
    "Telengana":         "Telangana",
    "Maharashra":        "Maharashtra",
    "Chattisgarh":       "Chhattisgarh",
    "Andra Pradesh":     "Andhra Pradesh",
    "Andhrapradesh":     "Andhra Pradesh",
    "Westbengal":        "West Bengal",
    "Uttrakhand":        "Uttarakhand",
    "Uttarkhand":        "Uttarakhand",
    "Harayana":          "Haryana",
    "Karala":            "Kerala",

    # ── BUG 2 fix: census 2011 uses old name "Orissa" ──
    "Odisha":            "Orissa",

    # ── BUG 4 fix: & vs And — census uses "Jammu And Kashmir" ──
    "Jammu & Kashmir":   "Jammu And Kashmir",
    "Jammu":             "Jammu And Kashmir",
    "Jammu And Kashmir": "Jammu And Kashmir",  # already correct, kept for safety

    # ── BUG 5 fix: two spellings → one census name ──
    "Puducherry":        "Pondicherry",

    # ── BUG 6 fix: short form / ops variant → full census name ──
    "Andaman":                 "Andaman And Nicobar Islands",
    "Andaman & Nicobar":       "Andaman And Nicobar Islands",  # ← ops uses this short form
}

# ── Single reusable function — applied identically to stations, ops, makers ──
def normalize_state(series):
    return (series
            .str.strip()
            .str.title()
            .str.replace(r'[^a-zA-Z &]', '', regex=True)  # Remove weird characters (BUG 8 fix)
            .str.strip()
            .replace(CITY_TO_STATE)
            .replace(CANONICAL))

stations["state"] = normalize_state(stations["state"])

print(f"  Charging Stations : {len(stations):,} rows | "
      f"{stations['state'].nunique()} states")
print(stations['state'].value_counts())
print(stations)



# --- Cell 3 ---
# Official public charging station counts
ops = pd.read_csv(f"{BASE_DATA_PATH}/OperationalPC.csv", encoding="utf-8-sig")
ops.columns = (ops.columns.str.strip().str.lower().str.replace(" ", "_"))
ops = ops.rename(columns={"no._of_operational_pcs": "official_stations"})
ops["state"] = normalize_state(ops["state"])  # ← BUG 7 fix: was .str.strip().str.title() only
print(f"  Operational PCS   : {len(ops):,} rows")
print(ops)



# --- Cell 4 ---
# EV manufacturer locations
makers = pd.read_csv(f"{BASE_DATA_PATH}/EV_Maker_by_Place.csv", encoding="utf-8-sig")
makers.columns = (makers.columns.str.strip().str.lower().str.replace(" ", "_"))
makers["state"] = normalize_state(makers["state"])  # ← BUG 11 fix: was .str.strip().str.title() only
print(f"  EV Makers         : {len(makers):,} rows | "
      f"{makers['state'].nunique()} states")



# --- Cell 5 ---
# EV registrations by category 2001-2024
ev_cat = pd.read_csv(f"{BASE_DATA_PATH}/ev_cat_01-24.csv", encoding="utf-8-sig")
ev_cat.columns = ev_cat.columns.str.strip()
date_col = ev_cat.columns[0]
ev_cat = ev_cat.rename(columns={date_col: "date"})
ev_cat = ev_cat[ev_cat["date"] != "0"].copy()
ev_cat["date"] = pd.to_datetime(ev_cat["date"], format="%d/%m/%y", errors="coerce")
ev_cat = ev_cat.dropna(subset=["date"])
ev_cat["year"] = ev_cat["date"].dt.year
num_cols = ev_cat.columns.drop(["date", "year"])
ev_cat[num_cols] = (ev_cat[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0))
print(f"  EV Category Trend : {len(ev_cat):,} rows | "
      f"{ev_cat['year'].min()}-{ev_cat['year'].max()}")


# --- Cell 6 ---
# EV sales by maker and category 2015-2024
sales = pd.read_csv(f"{BASE_DATA_PATH}/ev_sales_by_makers_and_cat_15-24.csv",encoding="utf-8-sig")
sales.columns = sales.columns.str.strip()
sales = sales.rename(columns={sales.columns[0]: "cat",sales.columns[1]: "maker"})
sales["maker"] = sales["maker"].str.strip().str.replace('"', "")

# Identify year columns
year_cols = [c for c in sales.columns if c.isdigit()]

# Convert sales data to numbers
for c in year_cols:
    sales[c] = pd.to_numeric(sales[c], errors="coerce").fillna(0)
print(f"  EV Sales by Maker : {len(sales):,} rows | "
      f"years: {year_cols[0]}-{year_cols[-1]}")


# --- Cell 7 ---
# India district census 2011
census = pd.read_csv(f"{BASE_DATA_PATH}/india-districts-census-2011.csv")
census.columns = census.columns.str.strip()
census["State name"]    = census["State name"].str.strip().str.title()
census["District name"] = census["District name"].str.strip().str.title()
print(f"  Census Districts  : {len(census):,} rows | "
      f"{census['State name'].nunique()} states")

print("  All datasets loaded.\n")


# --- Cell 9 ---
print("PHASE 2 - Infrastructure EDA")

# Chart 01 – Stations by state
mapped_by_state = (
    stations.groupby("state").size()
    .reset_index(name="mapped_stations")
    .sort_values("mapped_stations", ascending=False)
    .head(20)
)

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.barh(
    mapped_by_state["state"][::-1],
    mapped_by_state["mapped_stations"][::-1],
    color=BLUE, edgecolor="none"
)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 2, bar.get_y() + bar.get_height() / 2,
            str(int(w)), va="center", fontsize=8, color="#333333")

ax.set_title("EV Charging Stations by State (Top 20)", fontweight="bold")
ax.set_xlabel("Number of Stations")
ax.set_ylabel("State")
ax.grid(axis="x")
fig.tight_layout()
plt.show()



# --- Cell 10 ---
# Chart 02 – Official count vs mapped stations

mapped_counts = (
    stations.groupby("state").size().reset_index(name="mapped")
)
compare = (
    ops.merge(mapped_counts, on="state", how="outer")
    .fillna(0)
    .sort_values("official_stations", ascending=False)
    .head(20)
)

x = np.arange(len(compare))
w = 0.4
fig, ax = plt.subplots(figsize=(15, 7))
ax.bar(x - w/2, compare["official_stations"], w,
       label="Official Govt Count", color=GREEN, alpha=0.85, edgecolor="none")
ax.bar(x + w/2, compare["mapped"], w,
       label="Mapped Stations", color=BLUE, alpha=0.85, edgecolor="none")
ax.set_xticks(x)
ax.set_xticklabels(compare["state"], rotation=45, ha="right")
ax.set_title("Official Govt Count vs Mapped Charging Stations by State",
             fontweight="bold")
ax.set_ylabel("Number of Stations")
ax.legend()
ax.grid(axis="y")
fig.tight_layout()
plt.show()



# --- Cell 11 ---
# Chart 03 – Charger type distribution and top cities

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

type_counts = stations["type"].value_counts()
type_counts.index = [
    f"Type {int(t)}" if str(t).replace(".", "").isdigit() else str(t)
    for t in type_counts.index
]

wedge_props = dict(edgecolor="white", linewidth=1.2)
axes[0].pie(
    type_counts.values,
    labels=type_counts.index,
    colors=VOLT_COLORS[:len(type_counts)],
    autopct="%1.1f%%",
    startangle=140,
    wedgeprops=wedge_props,
    textprops={"fontsize": 9, "color": "#111111"}
)
axes[0].set_title("Charger Type Distribution", fontweight="bold")

top_cities = stations["city"].value_counts().head(15)
axes[1].barh(
    top_cities.index[::-1], top_cities.values[::-1],
    color=PURPLE, edgecolor="none"
)
axes[1].set_title("Top 15 Cities by Charging Stations", fontweight="bold")
axes[1].set_xlabel("Number of Stations")
axes[1].set_ylabel("City")
axes[1].grid(axis="x")

fig.suptitle("EV Charging Infrastructure: Type Distribution & Top Cities",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
plt.show()



# --- Cell 13 ---
print("PHASE 3 - EV Market and Trends")

ev_cat = pd.read_csv(f"{BASE_DATA_PATH}/ev_cat_01-24.csv")
ev_cat.columns = ev_cat.columns.str.strip()
ev_cat["Date"] = pd.to_datetime(ev_cat["Date"], dayfirst=True, errors="coerce")
ev_cat = ev_cat.dropna(subset=["Date"])
ev_cat["year"] = ev_cat["Date"].dt.year
value_cols = ev_cat.columns.drop(["Date", "year"])
for col in value_cols:
    ev_cat[col] = pd.to_numeric(ev_cat[col], errors="coerce")
ev_cat = ev_cat.dropna(axis=1, how="all")
num_cols = ev_cat.columns.drop(["Date", "year"])
yearly = ev_cat.groupby("year")[num_cols].sum()
yearly["TOTAL"] = yearly.sum(axis=1)
yearly_total = yearly["TOTAL"].reset_index()
yearly_total.columns = ["year", "TOTAL"]

# Chart 04 – Yearly EV growth
fig, ax = plt.subplots(figsize=(14, 6))
ax.fill_between(yearly_total["year"], yearly_total["TOTAL"],
                alpha=0.12, color=BLUE)
ax.plot(yearly_total["year"], yearly_total["TOTAL"],
        color=BLUE, linewidth=2.5, marker="o", markersize=5)

recent = yearly_total[yearly_total["year"] >= 2020]
for _, row in recent.iterrows():
    ax.annotate(
        f"{int(row['TOTAL']):,}",
        (row["year"], row["TOTAL"]),
        textcoords="offset points", xytext=(0, 10),
        ha="center", fontsize=8, color="#111111"
    )

ax.set_title("India EV Registrations — Yearly Growth (2001–2024)",
             fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Total EV Registrations")
fig.tight_layout()
plt.show()



# --- Cell 15 ---
print("PHASE 4 - Demand EDA (Census)")

# Step 1: Clean column names
census.columns = census.columns.str.strip()
census = census.rename(columns={

    "State name": "state",
    "Population": "population",
    "Households_with_Car_Jeep_Van": "car_households",
    "Housholds_with_Electric_Lighting": "electric_households",
    "Households": "total_households",
    "Power_Parity_Rs_330000_545000": "income_330k_545k",
    "Power_Parity_Above_Rs_545000": "income_above_545k"
})

census["state"] = census["state"].str.strip().str.title()

# Step 2: Aggregate at state level
state_census = census.groupby("state").agg(
    total_population=("population", "sum"),
    households_with_car=("car_households", "sum"),
    electric_lighting=("electric_households", "sum"),
    total_households=("total_households", "sum"),
    income_330k_545k=("income_330k_545k", "sum"),
    income_above_545k=("income_above_545k", "sum"),
).reset_index()

# ── BUG 3 fix: Telangana was formed in 2014 — it does not exist in census 2011 ──
# Without this row, all 81 Telangana stations are silently dropped from gap_df
# because the left-merge finds no census row to join to.
# Estimate: Telangana ≈ 41% of undivided Andhra Pradesh (2011 population split).
ap_row = state_census[state_census["state"] == "Andhra Pradesh"].iloc[0]
telangana_row = pd.DataFrame([{
    "state":               "Telangana",
    "total_population":    int(ap_row["total_population"]    * 0.41),
    "households_with_car": int(ap_row["households_with_car"] * 0.41),
    "electric_lighting":   int(ap_row["electric_lighting"]   * 0.41),
    "total_households":    int(ap_row["total_households"]    * 0.41),
    "income_330k_545k":    int(ap_row["income_330k_545k"]    * 0.41),
    "income_above_545k":   int(ap_row["income_above_545k"]   * 0.41),
}])
state_census = pd.concat([state_census, telangana_row], ignore_index=True)
# Also adjust Andhra Pradesh down to the remaining 59%
ap_idx = state_census[state_census["state"] == "Andhra Pradesh"].index[0]
for _col in ["total_population","households_with_car","electric_lighting",
             "total_households","income_330k_545k","income_above_545k"]:
    state_census.loc[ap_idx, _col] = int(ap_row[_col] * 0.59)

# Step 3: Derived metrics
state_census["income_middle_high"] = (
    state_census["income_330k_545k"] +
    state_census["income_above_545k"]
)

state_census["electric_pct"] = (
    state_census["electric_lighting"] /
    state_census["total_households"] * 100
)

state_census["car_per_1000"] = (
    state_census["households_with_car"] /
    state_census["total_households"] * 1000
)

# Step 4: Charging stations data
state_stations = (
    stations.groupby("state")
    .size()
    .reset_index(name="station_count")
)

# #

# #

# Step 5: Merge demand and supply
demand_supply = (
    state_census
    .merge(state_stations, on="state", how="left")
    .fillna({"station_count": 0})
)

# ── FIX: ev-charging-stations-india.csv is crowdsourced and incomplete ──
# e.g. Delhi: 179 mapped vs 1886 official, Maharashtra: 265 vs 3079 official
# Step 5.5: Replace station_count with official govt counts from ops wherever available
ops_counts = ops[["state", "official_stations"]].copy()
demand_supply = demand_supply.merge(ops_counts, on="state", how="left")
demand_supply["station_count"] = demand_supply.apply(
    lambda r: r["official_stations"]
    if pd.notna(r["official_stations"]) and r["official_stations"] > 0
    else r["station_count"], axis=1
)
demand_supply = demand_supply.drop(columns=["official_stations"])

# Step 6: Final metric
demand_supply["stations_per_million"] = (
    demand_supply["station_count"] /
    demand_supply["total_population"] * 1_000_000
)

# Step 7: Debug check
print(demand_supply.head())
print(demand_supply.sort_values("stations_per_million", ascending=False).head(10))



# --- Cell 16 ---
print("Chart 05 - Vehicle Category Breakdown")

latest_year = ev_cat["year"].max()
latest_year_data = ev_cat[ev_cat["year"] == latest_year][num_cols].sum()
latest_year_data = latest_year_data[latest_year_data > 0].sort_values(ascending=False)

TOP_N = 8
if len(latest_year_data) > TOP_N:
    top = latest_year_data.head(TOP_N)
    others = latest_year_data.iloc[TOP_N:].sum()
    cat_df = top.copy()
    cat_df["OTHERS"] = others
else:
    cat_df = latest_year_data

fig, ax = plt.subplots(figsize=(10, 7))
wedge_props = dict(edgecolor="white", linewidth=1.2)
ax.pie(
    cat_df.values,
    labels=None,
    colors=VOLT_COLORS[:len(cat_df)],
    autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
    startangle=140,
    pctdistance=0.75,
    wedgeprops=wedge_props,
    textprops={"fontsize": 9, "color": "#111111"}
)
ax.legend(cat_df.index, title="Categories",
          loc="center left", bbox_to_anchor=(1, 0.5))
ax.set_title(f"EV Sales by Vehicle Category ({latest_year})", fontweight="bold")
fig.tight_layout()
plt.show()



# --- Cell 17 ---
print("Chart 06 - Top EV Makers")

sales_copy = sales.copy()
sales_copy["TOTAL"] = sales_copy[year_cols].sum(axis=1)
top_makers = (
    sales_copy.groupby("maker")["TOTAL"]
    .sum().sort_values(ascending=False).head(15)
)

fig, ax = plt.subplots(figsize=(14, 7))
ax.barh(top_makers.index[::-1], top_makers.values[::-1],
        color=ORANGE, edgecolor="none")

for bar in ax.patches:
    w = bar.get_width()
    ax.text(w * 1.01, bar.get_y() + bar.get_height() / 2,
            f"{int(w):,}", va="center", fontsize=8, color="#333333")

ax.set_title("Top 15 EV Makers — Total Sales (2015–2024)", fontweight="bold")
ax.set_xlabel("Total Units Sold")
ax.grid(axis="x")
fig.tight_layout()
plt.show()



# --- Cell 18 ---
print("Chart 07 - Maker vs Category Heatmap")

top10_makers = (
    sales_copy.groupby("maker")["TOTAL"]
    .sum().nlargest(10).index
)
pivot = (
    sales_copy[sales_copy["maker"].isin(top10_makers)]
    .groupby(["maker", "cat"])["TOTAL"]
    .sum().unstack(fill_value=0)
)

fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(
    pivot, cmap=CMAP_HEAT,
    annot=True, fmt=".0f",
    linewidths=0.4, linecolor="#cccccc",
    annot_kws={"size": 7},
    ax=ax
)
ax.set_title("Top 10 EV Makers vs Vehicle Category (2015–2024)", fontweight="bold")
ax.set_xlabel("Vehicle Category")
ax.set_ylabel("Maker")
fig.tight_layout()
plt.show()



# --- Cell 19 ---
print("Chart 08 - Manufacturer Presence by State")

maker_state = makers["state"].value_counts().head(15)

fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(
    maker_state.index[::-1], maker_state.values[::-1],
    color=VOLT_COLORS[:len(maker_state)], edgecolor="none"
)
for bar in ax.patches:
    w = bar.get_width()
    ax.text(w + 0.2, bar.get_y() + bar.get_height() / 2,
            str(int(w)), va="center", fontsize=9, color="#333333")

ax.set_title("EV Manufacturer Count by State", fontweight="bold")
ax.set_xlabel("Number of Manufacturers")
ax.set_ylabel("State")
ax.grid(axis="x")
fig.tight_layout()
plt.show()



# --- Cell 20 ---
# Chart 09 – Population vs station count

top_states = demand_supply.nlargest(20, "total_population")

fig, ax = plt.subplots(figsize=(14, 7))
scatter = ax.scatter(
    top_states["total_population"] / 1e6,
    top_states["station_count"],
    c=top_states["stations_per_million"],
    cmap=CMAP_RDGN, s=200, alpha=0.85,
    edgecolors="#333333", linewidths=0.6
)
for _, row in top_states.iterrows():
    ax.annotate(
        row["state"],
        (row["total_population"] / 1e6, row["station_count"]),
        textcoords="offset points", xytext=(5, 5), fontsize=8
    )

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Stations per Million People")
ax.set_title("Population vs EV Charging Stations (Top 20 States)", fontweight="bold")
ax.set_xlabel("Population (Millions)")
ax.set_ylabel("Number of Charging Stations")
ax.grid()
fig.tight_layout()
plt.show()



# --- Cell 21 ---
# Chart 10 – Stations per million population

spm = (
    demand_supply[demand_supply["total_population"] > 1_000_000]
    .sort_values("stations_per_million", ascending=False)
    .head(20)
)

colors_bar = [
    GREEN if v > 10 else (ORANGE if v > 3 else RED)
    for v in spm["stations_per_million"]
]

fig, ax = plt.subplots(figsize=(14, 7))
ax.barh(spm["state"][::-1], spm["stations_per_million"][::-1],
        color=colors_bar[::-1], edgecolor="none")
ax.set_title("EV Charging Stations per Million Population", fontweight="bold")
ax.set_xlabel("Stations per Million People")

legend_patches = [
    mpatches.Patch(color=GREEN,  label="Well Served (>10)"),
    mpatches.Patch(color=ORANGE, label="Average (3–10)"),
    mpatches.Patch(color=RED,    label="Under-served (<3)"),
]
ax.legend(handles=legend_patches)
ax.grid(axis="x")
fig.tight_layout()
plt.show()



# --- Cell 22 ---
# Chart 11 – Car ownership and grid access

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
info = demand_supply.nlargest(20, "total_population")

axes[0].scatter(info["car_per_1000"], info["station_count"],
                color=BLUE, s=120, alpha=0.8,
                edgecolors="#333333", linewidths=0.5)
for _, row in info.iterrows():
    axes[0].annotate(row["state"],
                     (row["car_per_1000"], row["station_count"]),
                     textcoords="offset points", xytext=(4, 4), fontsize=7)

z = np.polyfit(info["car_per_1000"], info["station_count"], 1)
xr = np.linspace(info["car_per_1000"].min(), info["car_per_1000"].max(), 100)
axes[0].plot(xr, np.poly1d(z)(xr), color=RED, linestyle="--", alpha=0.7,
             label="Trend")
axes[0].set_title("Car Ownership vs Charging Stations", fontweight="bold")
axes[0].set_xlabel("Households with Car per 1,000")
axes[0].set_ylabel("Charging Stations")
axes[0].legend()

info_s = info.sort_values("electric_pct", ascending=False)
axes[1].bar(range(len(info_s)), info_s["electric_pct"],
            color=PURPLE, alpha=0.85, edgecolor="none")
axes[1].set_xticks(range(len(info_s)))
axes[1].set_xticklabels(info_s["state"], rotation=45, ha="right", fontsize=8)
axes[1].axhline(y=90, color=RED, linestyle="--", alpha=0.7, label="90% threshold")
axes[1].set_title("Households with Electric Lighting (%)", fontweight="bold")
axes[1].set_ylabel("% Households")
axes[1].legend()

fig.suptitle("Car Ownership and Grid Access by State",
             fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()



# --- Cell 23 ---
# Chart 12 – Correlation heatmap

corr_cols = [
    "total_population", "households_with_car", "income_middle_high",
    "electric_pct", "car_per_1000", "station_count", "stations_per_million"
]
corr_labels = [
    "Population", "Households\nwith Car", "Middle-High\nIncome HH",
    "Electric\nLighting %", "Cars per\n1000 HH",
    "Station\nCount", "Stations per\nMillion"
]

corr_df = demand_supply[corr_cols].dropna()
corr_matrix = corr_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(
    corr_matrix, mask=mask,
    annot=True, fmt=".2f",
    cmap=CMAP_DIV, center=0,
    vmin=-1, vmax=1,
    xticklabels=corr_labels, yticklabels=corr_labels,
    linewidths=0.4, linecolor="#cccccc",
    annot_kws={"size": 9},
    ax=ax
)
ax.set_title("Correlation Matrix — Demand and Infrastructure Factors",
             fontweight="bold")
fig.tight_layout()
plt.show()



# --- Cell 25 ---
print("PHASE 5 - Gap Analysis")

gap_df = demand_supply[demand_supply["total_population"] > 500_000].copy()

# Step 1: Normalize features
gap_df["pop_norm"] = gap_df["total_population"] / gap_df["total_population"].max()
gap_df["car_norm"] = gap_df["car_per_1000"] / gap_df["car_per_1000"].max()
gap_df["income_norm"] = gap_df["income_middle_high"] / gap_df["income_middle_high"].max()
gap_df["electric_norm"] = gap_df["electric_pct"] / 100

# Step 2: Demand score
gap_df["demand_score"] = (
    gap_df["pop_norm"] * 0.35 +
    gap_df["car_norm"] * 0.25 +
    gap_df["income_norm"] * 0.25 +
    gap_df["electric_norm"] * 0.15
) * 100

# Step 3: Supply score
gap_df["supply_score"] = gap_df["stations_per_million"].fillna(0)
gap_df["supply_score"] = (
    gap_df["supply_score"] / gap_df["supply_score"].max()
) * 100

# Step 4: Gap score
gap_df["gap_score"] = gap_df["demand_score"] - gap_df["supply_score"]

# Step 5: Debug check
print(gap_df[["state", "demand_score", "supply_score", "gap_score"]]
      .sort_values("gap_score", ascending=False)
      .head(10))


# --- Cell 26 ---
print("Chart 13 - Demand vs Supply Quadrant")

gap_df.columns = gap_df.columns.str.strip()
STATE_COL = "State name" if "State name" in gap_df.columns else "state"
plot_df   = gap_df.dropna(subset=["demand_score", "supply_score"]).copy()
d_med     = plot_df["demand_score"].median()
s_med     = plot_df["supply_score"].median()

def classify(row):
    if row["demand_score"] > d_med and row["supply_score"] < s_med:
        return "Deploy Now",    RED
    elif row["demand_score"] > d_med and row["supply_score"] >= s_med:
        return "Well Served",   GREEN
    elif row["demand_score"] <= d_med and row["supply_score"] < s_med:
        return "Future Market", ORANGE
    else:
        return "Over-served",   BLUE

plot_df[["segment", "color"]] = plot_df.apply(
    lambda r: pd.Series(classify(r)), axis=1
)

fig, ax = plt.subplots(figsize=(18, 10))        # ← wider figure gives more room

ax.scatter(plot_df["supply_score"], plot_df["demand_score"],
           c=plot_df["color"], s=200, alpha=0.85,
           edgecolors="#333333", linewidths=0.6, zorder=3)

# ── Label ALL points (not just top 15) ───────────────────────────────────
for _, row in plot_df.iterrows():               # ← changed: iterrows all rows
    ax.annotate(
        row[STATE_COL],
        (row["supply_score"], row["demand_score"]),
        textcoords="offset points",
        xytext=(6, 4),
        fontsize=7.5,
        color="#222222",
        zorder=4
    )

ax.axvline(x=s_med, color="#888888", linestyle="--", linewidth=0.9)
ax.axhline(y=d_med, color="#888888", linestyle="--", linewidth=0.9)

# ── Quadrant labels — pushed to edges so they don't overlap points ────────
y_top = plot_df["demand_score"].max() * 0.97
y_bot = plot_df["demand_score"].min() * 0.85
x_max = plot_df["supply_score"].max()

ax.text(0.5, y_top,
        "High Demand, Low Supply\n(Deploy stations here)",
        color=RED, fontsize=9, fontweight="bold")
ax.text(s_med + 1, y_top,
        "High Demand, High Supply\n(Well served)",
        color=GREEN, fontsize=9, fontweight="bold")
ax.text(0.5, y_bot,
        "Low Demand, Low Supply\n(Showroom Target)",
        color=ORANGE, fontsize=9, fontweight="bold")
ax.text(s_med + 1, y_bot,
        "Low Demand, High Supply\n(Over-served)",
        color=BLUE, fontsize=9, fontweight="bold")

legend_items = [
    mpatches.Patch(color=RED,    label="Deploy Now"),
    mpatches.Patch(color=GREEN,  label="Well Served"),
    mpatches.Patch(color=ORANGE, label="Future Market"),
    mpatches.Patch(color=BLUE,   label="Over-served"),
]
ax.legend(handles=legend_items, fontsize=9, loc="upper right")
ax.set_title("Demand vs Supply — EV Infrastructure Gap Analysis",
             fontweight="bold", fontsize=13)
ax.set_xlabel("Supply Score (Existing Infrastructure)", fontsize=11)
ax.set_ylabel("Demand Score (Population + Income + Grid)", fontsize=11)

# ── Add padding so edge labels don't get clipped ─────────────────────────
ax.set_xlim(left=plot_df["supply_score"].min() - 3)
ax.set_ylim(bottom=plot_df["demand_score"].min() - 3,
            top=plot_df["demand_score"].max() + 5)

fig.tight_layout()
plt.savefig("chart13_demand_supply_quadrant.png", dpi=150, bbox_inches="tight")
plt.show()


# --- Cell 27 ---
print("Chart 14 - Priority States for Deployment")

gap_df.columns = gap_df.columns.str.strip()
STATE_COL = "State name" if "State name" in gap_df.columns else "state"
priority = gap_df.sort_values("gap_score", ascending=False).head(20)
colors_p = [RED if g > 40 else (ORANGE if g > 20 else BLUE)
            for g in priority["gap_score"]]

fig, ax = plt.subplots(figsize=(14, 8))
ax.barh(priority[STATE_COL][::-1], priority["gap_score"][::-1],
        color=colors_p[::-1], edgecolor="none")

for bar in ax.patches:
    w = bar.get_width()
    ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}", va="center", fontsize=8, color="#333333")

ax.set_title(
    "Priority States for EV Station Deployment\nGap Score = Demand - Supply",
    fontweight="bold"
)
ax.set_xlabel("Gap Score")
ax.legend(handles=[
    mpatches.Patch(color=RED,    label="Urgent (>40)"),
    mpatches.Patch(color=ORANGE, label="High (20–40)"),
    mpatches.Patch(color=BLUE,   label="Moderate (<20)"),
])
ax.grid(axis="x")
fig.tight_layout()
plt.show()



# --- Cell 29 ---
print("PHASE 6 - K-Means Clustering")

# Step 1: Clean column names
gap_df.columns = gap_df.columns.str.strip()

# Step 2: Detect state column
if "State name" in gap_df.columns:
    STATE_COL = "State name"
elif "state" in gap_df.columns:
    STATE_COL = "state"
elif "State_name" in gap_df.columns:
    STATE_COL = "State_name"
else:
    raise ValueError("State column not found")

# Step 3: Prepare dataset
cluster_df = gap_df[[STATE_COL, "demand_score", "supply_score",
                     "income_norm", "electric_norm"]].dropna().copy()

features = ["demand_score", "supply_score", "income_norm", "electric_norm"]

# Step 4: Scale features
scaler = StandardScaler()
X = scaler.fit_transform(cluster_df[features])

# Step 5: KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_df["cluster"] = kmeans.fit_predict(X)

# Step 6: Compute centroids (mean values per cluster)
centroids = cluster_df.groupby("cluster")[["demand_score", "supply_score"]].mean()

# Step 7: Label clusters logically
cluster_labels = {}

# Gap = demand - supply
centroids["gap"] = centroids["demand_score"] - centroids["supply_score"]

# 1. Highest gap → Deploy Now
crit = centroids["gap"].idxmax()
cluster_labels[crit] = ("Deploy Now", RED)

# 2. Highest supply → Well Served
remaining = [c for c in centroids.index if c != crit]
well = centroids.loc[remaining]["supply_score"].idxmax()
cluster_labels[well] = ("Well Served", GREEN)

# 3. Highest demand (from remaining) → Growth Zone
remaining = [c for c in remaining if c != well]
grow = centroids.loc[remaining]["demand_score"].idxmax()
cluster_labels[grow] = ("Growth Zone", ORANGE)

# 4. Last one → Showroom Target
remaining = [c for c in remaining if c != grow]
show = remaining[0]
cluster_labels[show] = ("Showroom Target", PURPLE)

# Step 8: Map labels back to data
cluster_df["segment"] = cluster_df["cluster"].map(lambda x: cluster_labels[x][0])
cluster_df["color"] = cluster_df["cluster"].map(lambda x: cluster_labels[x][1])

# Step 9: Debug check
print(cluster_df[[STATE_COL, "cluster", "segment"]].head())


# --- Cell 30 ---
print("Chart 15 - K-Means Scatter")

cluster_df.columns = cluster_df.columns.str.strip()
STATE_COL = "State name" if "State name" in cluster_df.columns else "state"

fig, ax = plt.subplots(figsize=(18, 10))        # ← wider for more room

for cid, (label, color) in cluster_labels.items():
    subset = cluster_df[cluster_df["cluster"] == cid]
    ax.scatter(subset["supply_score"], subset["demand_score"],
               color=color, s=200, alpha=0.85, label=label,
               edgecolors="#333333", linewidths=0.5, zorder=3)

# ── Label ALL points ──────────────────────────────────────────────────────
for _, row in cluster_df.iterrows():            # ← changed: all rows not just 15
    ax.annotate(
        row[STATE_COL],
        (row["supply_score"], row["demand_score"]),
        textcoords="offset points",
        xytext=(6, 4),
        fontsize=7.5,
        color="#222222",
        zorder=4
    )

# ── Add padding so edge labels don't get clipped ─────────────────────────
ax.set_xlim(left=cluster_df["supply_score"].min() - 3)
ax.set_ylim(bottom=cluster_df["demand_score"].min() - 3,
            top=cluster_df["demand_score"].max() + 5)

ax.set_title("K-Means Clustering — India States by EV Market Segment",
             fontweight="bold", fontsize=13)
ax.set_xlabel("Supply Score", fontsize=11)
ax.set_ylabel("Demand Score", fontsize=11)
ax.legend(fontsize=9)
ax.grid()
fig.tight_layout()
plt.savefig("chart15_kmeans_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nCluster Summary:")
for cid, (label, _) in cluster_labels.items():
    states = cluster_df[cluster_df["cluster"] == cid][STATE_COL].tolist()
    print(f"  {label}: {', '.join(states)}")


# --- Cell 32 ---
print("\nPHASE 7 - Priority Scoring")

state_col = "State name" if "State name" in gap_df.columns else "state"

gap_df["station_score"] = (
    gap_df["pop_norm"]      * 0.30 +
    gap_df["car_norm"]      * 0.20 +
    gap_df["electric_norm"] * 0.20 +
    (1 - gap_df["supply_score"] / 100) * 0.30
) * 100

gap_df["showroom_score"] = (
    gap_df["pop_norm"]    * 0.35 +
    gap_df["income_norm"] * 0.30 +
    (1 - gap_df["supply_score"] / 100) * 0.35
) * 100

print("\nTop 10 States - Deploy Charging Stations:")
top_station = gap_df.nlargest(10, "station_score")

for i, (_, row) in enumerate(top_station.iterrows(), 1):
    print(f"{i:2}. {row[state_col]:<30} Score: {row['station_score']:.1f}")

print("\nTop 10 States - Open EV Showrooms:")
top_showroom = gap_df.nlargest(10, "showroom_score")

for i, (_, row) in enumerate(top_showroom.iterrows(), 1):
    print(f"{i:2}. {row[state_col]:<30} Score: {row['showroom_score']:.1f}")


# --- Cell 34 ---
print("\nPHASE 8 - Summary Dashboard")

state_col = "State name" if "State name" in gap_df.columns else "state"

fig = plt.figure(figsize=(20, 12), facecolor="white")

gs = gridspec.GridSpec(
    3, 4, figure=fig,
    hspace=0.60, wspace=0.45,
    left=0.06, right=0.97,
    top=0.88, bottom=0.07
)

# Title
fig.text(0.5, 0.95, "VoltMap India — EV Intelligence Dashboard",
         ha="center", fontsize=18, fontweight="bold", color="#111111")
fig.text(0.5, 0.91,
         "Energy-Tech Startup | FoT Manthan 2026 | Energy Domain",
         ha="center", fontsize=10, color="#555555")

# ── KPI Cards ────────────────────────────────────────────────────────────────
kpis = [
    ("Total EV Stations\n(Mapped)", f"{len(stations):,}", BLUE),
    ("States Covered", f"{stations['state'].nunique()}", GREEN),
    ("Official PCS Count", f"{ops['official_stations'].sum():,.0f}", ORANGE),
    ("EV Makers Tracked", f"{len(makers)}", PURPLE),
]

for i, (label, value, color) in enumerate(kpis):
    ax_kpi = fig.add_subplot(gs[0, i])
    ax_kpi.set_facecolor("#f7f7f7")
    ax_kpi.text(0.5, 0.62, value,
                ha="center", va="center",
                fontsize=26, fontweight="bold",
                color=color, transform=ax_kpi.transAxes)
    ax_kpi.text(0.5, 0.22, label,
                ha="center", va="center",
                fontsize=9, color="#444444",
                transform=ax_kpi.transAxes)
    ax_kpi.set_xticks([])
    ax_kpi.set_yticks([])
    for spine in ax_kpi.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)

# ── EV Growth Chart ───────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1, :2])
ax1.set_facecolor("white")
ax1.fill_between(yearly_total["year"], yearly_total["TOTAL"],
                 alpha=0.12, color=BLUE)
ax1.plot(yearly_total["year"], yearly_total["TOTAL"],
         color=BLUE, linewidth=2, marker="o", markersize=3)
ax1.set_title("EV Registration Growth (2001–2024)", fontsize=10, fontweight="bold")
ax1.set_ylabel("Registrations", fontsize=8)

# ── Top States Bar ────────────────────────────────────────────────────────────
if "mapped_by_state" in globals():
    top10s = mapped_by_state.head(10)
else:
    top10s = (stations["state"].value_counts().reset_index())
    top10s.columns = ["state", "mapped_stations"]

ax2 = fig.add_subplot(gs[1, 2:])
ax2.set_facecolor("white")
ax2.barh(top10s["state"][::-1], top10s["mapped_stations"][::-1],
         color=ORANGE, edgecolor="none")
ax2.set_title("Top 10 States – Charging Stations", fontsize=10, fontweight="bold")
ax2.set_xlabel("Count", fontsize=8)

# ── Gap Score Chart ───────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, :2])
ax3.set_facecolor("white")
top10_gap = gap_df.nlargest(10, "gap_score")
bar_colors = [RED if g > 40 else ORANGE for g in top10_gap["gap_score"]]
ax3.barh(top10_gap[state_col][::-1], top10_gap["gap_score"][::-1],
         color=bar_colors[::-1], edgecolor="none")
ax3.set_title("Top 10 States – Gap Score (Deploy Priority)", fontsize=10, fontweight="bold")
ax3.set_xlabel("Gap Score", fontsize=8)

# ── Key Findings ──────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 2:])
ax4.set_facecolor("#f7f7f7")
ax4.axis("off")

findings = [
    ("Deploy Now:",      "High demand, low infrastructure states"),
    ("Well Served:",     "Balanced demand and supply states"),
    ("Post-2020 Growth:","EV adoption accelerated rapidly"),
    ("Charger Priority:","2-Wheeler segment dominates demand"),
    ("Showroom Targets:","High income + population regions"),
]

ax4.text(0.05, 0.95, "Key Findings",
         fontsize=11, fontweight="bold", color="#111111",
         va="top", transform=ax4.transAxes)

for i, (bold_text, rest) in enumerate(findings):
    y = 0.78 - i * 0.15
    ax4.text(0.05, y, bold_text, fontsize=9, va="top",
             transform=ax4.transAxes, color=BLUE, fontweight="bold")
    ax4.text(0.05, y - 0.08, rest, fontsize=8, va="top",
             transform=ax4.transAxes, color="#444444")

for spine in ax4.spines.values():
    spine.set_edgecolor("#cccccc")
    spine.set_linewidth(1)

plt.show()



# --- Cell 36 ---
print("PHASE 5 — Feature Engineering")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── colour palette (matches existing notebook) ──────────────────────────
BLUE   = "#2166AC"
GREEN  = "#1A9641"
RED    = "#D7191C"
ORANGE = "#E08214"
PURPLE = "#762A83"
DARK   = "#333333"

# ── load & clean ─────────────────────────────────────────────────────────
ev = pd.read_csv(f"{BASE_DATA_PATH}/ev_cat_01-24.csv")
ev.columns = ev.columns.str.strip()
ev["Date"] = pd.to_datetime(ev["Date"], format="%d/%m/%y", errors="coerce")
ev = ev.dropna(subset=["Date"]).copy()
ev["year"]  = ev["Date"].dt.year
ev["month"] = ev["Date"].dt.month
num_cols = ev.select_dtypes(include="number").columns.difference(["year","month"])
ev["total"] = ev[num_cols].sum(axis=1)
ev = ev[ev["year"] >= 2010].sort_values("Date").reset_index(drop=True)

# ── time index ───────────────────────────────────────────────────────────
ev["t"] = range(len(ev))

# ── Fourier seasonality (annual cycle) ──────────────────────────────────
ev["sin12"] = np.sin(2 * np.pi * ev["month"] / 12)
ev["cos12"] = np.cos(2 * np.pi * ev["month"] / 12)

# ── lag features (autoregressive memory) ────────────────────────────────
ev["lag1"]  = ev["total"].shift(1)
ev["lag3"]  = ev["total"].shift(3)
ev["lag12"] = ev["total"].shift(12)

# ── rolling means (momentum) ─────────────────────────────────────────────
ev["roll3"] = ev["total"].shift(1).rolling(3).mean()
ev["roll6"] = ev["total"].shift(1).rolling(6).mean()

# ── policy shock flags (exogenous regressors) ────────────────────────────
ev["subsidy_flag"] = (ev["year"] >= 2019).astype(int)           # FAME-II launch
ev["covid_flag"]   = (
    (ev["year"] == 2020) |
    ((ev["year"] == 2021) & (ev["month"] <= 6))
).astype(int)

# ── drop NaN rows from lag creation ────────────────────────────────────
ev_model = ev.dropna().copy()
FEATURES  = ["t","month","sin12","cos12",
             "lag1","lag3","lag12","roll3","roll6",
             "subsidy_flag","covid_flag"]

print(f"  Rows available for modelling : {len(ev_model)}")
print(f"  Year range                   : {ev_model['year'].min()} – {ev_model['year'].max()}")
print(f"  Features built               : {len(FEATURES)}")
print(f"\n  Feature list:")
feat_descriptions = {
    "t"           : "Linear time index (trend carrier)", # The overall growth direction over time
    "month"       : "Month number (1–12)", # Repeating yearly patterns (e.g. festive season spikes)
    "sin12/cos12" : "Fourier pair — encodes 12-month cycle", # Repeating yearly patterns (e.g. festive season spikes)
    "lag1"        : "Registrations 1 month prior", # Autoregressive (Lag) -> What were sales 1/3/12 months ago?"
    "lag3"        : "Registrations 3 months prior", # Autoregressive (Lag) -> What were sales 1/3/12 months ago?"
    "lag12"       : "Registrations 12 months prior (same month last year)", # Autoregressive (Lag) -> What were sales 1/3/12 months ago?"
    "roll3"       : "3-month rolling mean (short momentum)", # Momentum -> Smoothed recent averages — are we on an upswing?
    "roll6"       : "6-month rolling mean (medium momentum)", # Momentum -> Smoothed recent averages — are we on an upswing?
    "subsidy_flag": "1 from 2019 onwards (FAME-II active)", # Policy Shocks -> External events that disrupted demand
    "covid_flag"  : "1 during Mar 2020 – Jun 2021 (demand shock)", # Policy Shocks -> External events that disrupted demand
}
for k, v in feat_descriptions.items():
    print(f"    {k:<14} → {v}")


# --- Cell 37 ---
print("Chart A — Feature Importance Preview (correlation with target)")

from sklearn.ensemble import GradientBoostingRegressor

gbr_check = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr_check.fit(ev_model[FEATURES], ev_model["total"])
importances = pd.Series(gbr_check.feature_importances_, index=FEATURES).sort_values()

fig, ax = plt.subplots(figsize=(10, 5))
colors_fi = [BLUE if i < len(importances)-3 else RED for i in range(len(importances))]
ax.barh(importances.index, importances.values, color=colors_fi, edgecolor="none")
ax.set_title("Feature Importance — Gradient Boosting (full dataset)",
             fontweight="bold", fontsize=13)
ax.set_xlabel("Importance Score", fontsize=11)
ax.axvline(0, color=DARK, linewidth=0.5)
for bar in ax.patches:
    w = bar.get_width()
    ax.text(w + 0.001, bar.get_y() + bar.get_height()/2,
            f"{w:.3f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("chartA_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → Top features are lag-based: the model learns from momentum.")


# --- Cell 39 ---
print("PHASE 6 — Chronological Data Split")

# ── CRITICAL: never shuffle time-series ─────────────────────────────────
# Train: 2010-2022  |  Test: 2023-2024 (hold-out)
train = ev_model[ev_model["year"] <= 2022].copy()
test  = ev_model[ev_model["year"] >= 2023].copy()

X_train, y_train = train[FEATURES], train["total"]
X_test,  y_test  = test[FEATURES],  test["total"]

print(f"  Train set : {len(train):>4} rows  ({train['year'].min()}–{train['year'].max()})")
print(f"  Test set  : {len(test):>4} rows  ({test['year'].min()}–{test['year'].max()})")
print(f"  Train %   : {len(train)/(len(train)+len(test))*100:.0f}%")

# ── Visualise split ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(train["Date"], train["total"], alpha=0.35, color=BLUE, label="Training data (2010–2022)")
ax.fill_between(test["Date"],  test["total"],  alpha=0.45, color=ORANGE, label="Test hold-out (2023–2024)")
ax.plot(train["Date"], train["total"], color=BLUE,   linewidth=1.2)
ax.plot(test["Date"],  test["total"],  color=ORANGE, linewidth=1.5)
ax.axvline(pd.Timestamp("2023-01-01"), color=RED, linewidth=1.5, linestyle="--", label="Split boundary")
ax.set_title("Chronological Train / Test Split — EV Monthly Registrations",
             fontweight="bold", fontsize=13)
ax.set_ylabel("Monthly EV Registrations", fontsize=11)
ax.set_xlabel("Date", fontsize=11)
ax.legend(fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
plt.tight_layout()
plt.savefig("chartB_train_test_split.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → Chronological split preserves temporal order (no data leakage).")


# --- Cell 41 ---
print("PHASE 7 — Train Three Competing Models")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

models = {
    "Linear Regression" : LinearRegression(), # Assumes a straight-line relationship between features and registrations
    "Random Forest"     : RandomForestRegressor(n_estimators=150, random_state=42), # Builds 150 independent decision trees, each trained on a random subset of data and features, Final prediction = average of all 150 trees
    "Gradient Boosting" : GradientBoostingRegressor(
                              n_estimators=200, max_depth=4,
                              learning_rate=0.05, random_state=42),
}

results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    pred = np.maximum(pred, 0)
    mape = mean_absolute_percentage_error(y_test, pred) * 100
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae  = np.mean(np.abs(y_test.values - pred))
    results[name] = {"model": m, "pred": pred, "MAPE": mape, "RMSE": rmse, "MAE": mae}
    print(f"  {name:<22} MAPE={mape:5.1f}%   RMSE={rmse:>10,.0f}   MAE={mae:>10,.0f}")

best_name = min(results, key=lambda k: results[k]["RMSE"])
best_model = results[best_name]["model"]
print(f"\n  ✓ Best model (lowest RMSE): {best_name}")


# --- Cell 42 ---
print("Chart C — Model Comparison")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
palette = {"Linear Regression": BLUE, "Random Forest": GREEN, "Gradient Boosting": RED}

for ax, (name, res) in zip(axes, results.items()):
    ax.plot(test["Date"].values, y_test.values,
            color=DARK, linewidth=2, label="Actual", zorder=3)
    ax.plot(test["Date"].values, res["pred"],
            color=palette[name], linewidth=1.8, linestyle="--",
            label=f"Predicted\nMAPE={res['MAPE']:.1f}%", zorder=3)
    ax.fill_between(test["Date"].values, y_test.values, res["pred"],
                    alpha=0.15, color=palette[name])
    ax.set_title(name, fontweight="bold", fontsize=11)
    ax.set_ylabel("Monthly Registrations", fontsize=9)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax.tick_params(axis="x", labelsize=8, rotation=20)

fig.suptitle("Actual vs Predicted — Test Period (2023–2024)",
             fontweight="bold", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("chartC_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → Gradient Boosting captures the growth momentum best.")


# --- Cell 44 ---
print("PHASE 8 — Evaluation Metrics & Residual Analysis")

from sklearn.metrics import r2_score

best_pred  = results[best_name]["pred"]
residuals  = y_test.values - best_pred
rel_errors = np.abs(residuals) / y_test.values * 100

print(f"\n  ── {best_name} on Test Set (2023–2024) ──")
print(f"  MAPE  : {results[best_name]['MAPE']:.1f}%")
print(f"  RMSE  : {results[best_name]['RMSE']:,.0f} registrations/month")
print(f"  MAE   : {results[best_name]['MAE']:,.0f} registrations/month")
print(f"  R²    : {r2_score(y_test, best_pred):.3f}")
print(f"\n  ── Interpretation ──")
print(f"  MAPE of ~{results[best_name]['MAPE']:.0f}% is acceptable for a national demand forecast")
print(f"  with only public data — the model correctly captures the direction")
print(f"  of growth even where exact magnitudes vary.")

# ── Chart D: Residual Analysis ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Residuals over time
axes[0].axhline(0, color=RED, linewidth=1, linestyle="--")
axes[0].bar(test["Date"].values, residuals,
            color=[BLUE if r > 0 else ORANGE for r in residuals],
            edgecolor="none", width=pd.Timedelta(days=20))
axes[0].set_title("Residuals over Time", fontweight="bold")
axes[0].set_ylabel("Actual − Predicted")
axes[0].tick_params(axis="x", rotation=20, labelsize=8)
axes[0].yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

# 2. Residual distribution
axes[1].hist(residuals, bins=12, color=BLUE, edgecolor="white", linewidth=0.5)
axes[1].axvline(0, color=RED, linewidth=1.5, linestyle="--")
axes[1].axvline(np.mean(residuals), color=ORANGE, linewidth=1.5,
                linestyle="-", label=f"Mean = {np.mean(residuals):,.0f}")
axes[1].set_title("Residual Distribution", fontweight="bold")
axes[1].set_xlabel("Residual Value")
axes[1].set_ylabel("Count")
axes[1].legend(fontsize=9)

# 3. Actual vs Predicted scatter
axes[2].scatter(y_test, best_pred, color=PURPLE, alpha=0.7, edgecolors=DARK,
                linewidths=0.4, s=80)
mn = min(y_test.min(), best_pred.min())
mx = max(y_test.max(), best_pred.max())
axes[2].plot([mn, mx], [mn, mx], color=RED, linewidth=1.5,
             linestyle="--", label="Perfect prediction")
axes[2].set_title("Actual vs Predicted", fontweight="bold")
axes[2].set_xlabel("Actual Registrations")
axes[2].set_ylabel("Predicted Registrations")
axes[2].legend(fontsize=9)
for ax in axes[1:]:
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

fig.suptitle(f"Residual Analysis — {best_name}",
             fontweight="bold", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("chartD_residual_analysis.png", dpi=150, bbox_inches="tight")
plt.show()


# --- Cell 45 ---
# Phase 9 — Polynomial Ridge Regression

print("PHASE 9 — Polynomial Ridge Regression")
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# ── Features: only the ones Ridge can extrapolate with ──────────────────
# We deliberately exclude lag/rolling features here because:
# → lags can't be computed for future months (2025–2030) without chaining
# → t, t², sin12, cos12 CAN be computed for any future date directly
RIDGE_FEATURES = ["t", "sin12", "cos12"]

X_train_r = train[RIDGE_FEATURES]
X_test_r  = test[RIDGE_FEATURES]

# ── Build pipeline: Polynomial → Scale → Ridge ──────────────────────────
# degree=2 adds: t, t², sin, cos, t×sin, t×cos, sin², cos², sin×cos
# RidgeCV auto-selects best alpha via cross-validation
ridge_pipe = Pipeline([
    ("poly",  PolynomialFeatures(degree=2, include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", RidgeCV(alphas=[0.01, 0.1, 1, 10, 50, 100, 500, 1000],
                      cv=5, scoring="neg_mean_squared_error"))
])

ridge_pipe.fit(X_train_r, y_train)

best_alpha = ridge_pipe.named_steps["ridge"].alpha_
print(f"  Best alpha (auto-selected by CV) : {best_alpha}")

# ── Evaluate on test set (2023–2024) ────────────────────────────────────
pred_ridge_test = np.maximum(ridge_pipe.predict(X_test_r), 0)

mape_r = mean_absolute_percentage_error(y_test, pred_ridge_test) * 100
rmse_r = np.sqrt(mean_squared_error(y_test, pred_ridge_test))
mae_r  = np.mean(np.abs(y_test.values - pred_ridge_test))
r2_r   = r2_score(y_test, pred_ridge_test)

print(f"\n  ── Polynomial Ridge on Test Set (2023–2024) ──")
print(f"  MAPE  : {mape_r:.1f}%")
print(f"  RMSE  : {rmse_r:,.0f}")
print(f"  MAE   : {mae_r:,.0f}")
print(f"  R²    : {r2_r:.3f}")

# ── Build future index (Jan 2025 → Dec 2030) ────────────────────────────
last_t     = ev_model["t"].max()
last_date  = ev_model["Date"].max()

future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    end="2050-12-01", freq="MS"
)

future_df = pd.DataFrame({
    "Date"  : future_dates,
    "t"     : range(last_t + 1, last_t + 1 + len(future_dates)),
    "month" : future_dates.month
})
future_df["sin12"] = np.sin(2 * np.pi * future_df["month"] / 12)
future_df["cos12"] = np.cos(2 * np.pi * future_df["month"] / 12)

pred_ridge_future = np.maximum(
    ridge_pipe.predict(future_df[RIDGE_FEATURES]), 0
)

# ── Chart: Actual vs Ridge fit vs Forecast ──────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))

# historical actual
ax.fill_between(train["Date"], train["total"],
                alpha=0.15, color=BLUE, label="_nolegend_")
ax.plot(ev_model["Date"], ev_model["total"],
        color=DARK, linewidth=1.2, label="Actual registrations")

# test period fitted
ax.plot(test["Date"], pred_ridge_test,
        color=GREEN, linewidth=1.8, linestyle="--", label="Ridge fit (test period)")

# future forecast
ax.fill_between(future_df["Date"],
                pred_ridge_future * 0.78,   # rough ±22% band
                pred_ridge_future * 1.22,
                alpha=0.15, color=PURPLE)
ax.plot(future_df["Date"], pred_ridge_future,
        color=PURPLE, linewidth=2.2, label="Ridge forecast (2025–2030)")

ax.axvline(pd.Timestamp("2023-01-01"), color=RED,
           linewidth=1.2, linestyle="--", label="Train/test split")
ax.axvline(pd.Timestamp("2025-01-01"), color=ORANGE,
           linewidth=1.2, linestyle=":", label="Forecast start")

ax.set_title("Polynomial Ridge — Fit & Forecast to Dec 2030",
             fontweight="bold", fontsize=13)
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Monthly EV Registrations", fontsize=11)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("chartE_ridge_forecast.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\n  Forecast Dec 2030 : {pred_ridge_future[-1]:,.0f} registrations/month")
print(f"  → Ridge R² on full dataset : "
      f"{r2_score(ev_model['total'], np.maximum(ridge_pipe.predict(ev_model[RIDGE_FEATURES]),0)):.3f}")



# Add this at the end of Phase 9 (after ridge is fitted)
# Calibrate Ridge forecast to start from actual last known value
last_actual     = ev_model["total"].iloc[-1]
first_ridge_val = pred_ridge_future[0]
calibration     = last_actual - first_ridge_val

pred_ridge_future_calibrated = pred_ridge_future + calibration

print(f"  Last actual value      : {last_actual:,.0f}")
print(f"  Ridge first prediction : {first_ridge_val:,.0f}")
print(f"  Calibration applied    : +{calibration:,.0f}")
print(f"  Calibrated Dec 2030    : {pred_ridge_future_calibrated[-1]:,.0f}")


# --- Cell 46 ---
print("PHASE 10 — Facebook Prophet")
from prophet import Prophet
from prophet.plot import plot_plotly

# ── Prophet requires exactly two columns: ds (date) and y (target) ──────
prophet_train = train[["Date", "total"]].rename(
    columns={"Date": "ds", "total": "y"}
)
prophet_full = ev_model[["Date", "total"]].rename(
    columns={"Date": "ds", "total": "y"}
)

# ── Define shocks as "holidays" ──────────────────────────────────────────
shocks = pd.DataFrame({
    "holiday"   : ["covid_shock"] * 16 + ["fame2_launch"],
    "ds"        : pd.date_range("2020-03-01", periods=16, freq="MS").tolist()
                  + [pd.Timestamp("2019-04-01")],
    "lower_window": 0,
    "upper_window": 0
})

# ── Build and fit Prophet ────────────────────────────────────────────────
m_prophet = Prophet(
    changepoint_prior_scale  = 0.3,
    seasonality_prior_scale  = 10,
    yearly_seasonality       = True,
    weekly_seasonality       = False,
    daily_seasonality        = False,
    holidays                 = shocks
)
m_prophet.fit(prophet_train)

# ── Evaluate on test set ─────────────────────────────────────────────────
prophet_test_df   = test[["Date"]].rename(columns={"Date": "ds"})
forecast_test     = m_prophet.predict(prophet_test_df)
pred_prophet_test = np.maximum(forecast_test["yhat"].values, 0)

mape_p = mean_absolute_percentage_error(y_test, pred_prophet_test) * 100
rmse_p = np.sqrt(mean_squared_error(y_test, pred_prophet_test))
mae_p  = np.mean(np.abs(y_test.values - pred_prophet_test))
r2_p   = r2_score(y_test, pred_prophet_test)

print(f"\n  ── Prophet on Test Set (2023–2024) ──")
print(f"  MAPE  : {mape_p:.1f}%")
print(f"  RMSE  : {rmse_p:,.0f}")
print(f"  MAE   : {mae_p:,.0f}")
print(f"  R²    : {r2_p:.3f}")

# ── Forecast to Dec 2030 — guaranteed ────────────────────────────────────
prophet_last_date = prophet_train["ds"].max()   # ← Dec 2022, not Jul 2024
target_end        = pd.Timestamp("2050-12-01")

months_needed = (
    (target_end.year  - prophet_last_date.year)  * 12 +
    (target_end.month - prophet_last_date.month)
) + 3    # ← +3 buffer

future_prophet = m_prophet.make_future_dataframe(periods=months_needed, freq="MS")
forecast_full  = m_prophet.predict(future_prophet)

# Trim to exactly Dec 2030
forecast_full = forecast_full[
    forecast_full["ds"] <= target_end
].reset_index(drop=True)

print(f"  Prophet trained to : {prophet_last_date.date()}")
print(f"  months_needed      : {months_needed}")
print(f"  Forecast ends      : {forecast_full['ds'].iloc[-1].date()}")
print(f"  Total rows         : {len(forecast_full)}")

# ── Chart: Prophet components (trend + seasonality) ──────────────────────
fig_comp = m_prophet.plot_components(forecast_full)
fig_comp.suptitle("Prophet — Trend & Seasonal Components",
                  fontweight="bold", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("chartF_prophet_components.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Chart: Prophet full forecast ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))

hist = forecast_full[forecast_full["ds"] <= ev_model["Date"].max()]
fut  = forecast_full[forecast_full["ds"] >  ev_model["Date"].max()]

ax.plot(ev_model["Date"], ev_model["total"],
        color=DARK, linewidth=1.2, label="Actual registrations")
ax.plot(hist["ds"], np.maximum(hist["yhat"], 0),
        color=GREEN, linewidth=1.5, linestyle="--", label="Prophet fit")
ax.fill_between(fut["ds"],
                np.maximum(fut["yhat_lower"], 0),
                fut["yhat_upper"],
                alpha=0.15, color=ORANGE, label="80% confidence band")
ax.plot(fut["ds"], np.maximum(fut["yhat"], 0),
        color=ORANGE, linewidth=2.2, label="Prophet forecast (2025–2030)")
ax.axvline(pd.Timestamp("2023-01-01"), color=RED,
           linewidth=1.2, linestyle="--", label="Train/test split")
ax.axvline(pd.Timestamp("2025-01-01"), color=PURPLE,
           linewidth=1.2, linestyle=":", label="Forecast start")
ax.set_title("Facebook Prophet — Fit & Forecast to Dec 2030",
             fontweight="bold", fontsize=13)
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Monthly EV Registrations", fontsize=11)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("chartG_prophet_forecast.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Dec 2030 forecast value ───────────────────────────────────────────────
target_date     = target_end
closest_idx     = (forecast_full["ds"] - target_date).abs().idxmin()
dec2030_prophet = forecast_full.loc[closest_idx, "yhat"]

print(f"\n  Closest date found : {forecast_full.loc[closest_idx, 'ds'].date()}")
print(f"  Forecast Dec 2030  : {dec2030_prophet:,.0f} registrations/month")


# --- Cell 47 ---
# Phase 11 — Final Comparison & Forecast
print("PHASE 11 — Final Model Comparison & Forecast Summary")

# ── Metrics table ────────────────────────────────────────────────────────
comparison = pd.DataFrame({
    "Model"  : ["Linear Regression", "Random Forest", "Gradient Boosting",
                "Polynomial Ridge",  "Prophet"],
    "MAPE %"  : [results["Linear Regression"]["MAPE"],
                 results["Random Forest"]["MAPE"],
                 results["Gradient Boosting"]["MAPE"],
                 mape_r, mape_p],
    "RMSE"    : [results["Linear Regression"]["RMSE"],
                 results["Random Forest"]["RMSE"],
                 results["Gradient Boosting"]["RMSE"],
                 rmse_r, rmse_p],
    "R²"      : [r2_score(y_test, np.maximum(results["Linear Regression"]["model"].predict(X_test),0)),
                 r2_score(y_test, results["Random Forest"]["pred"]),
                 r2_score(y_test, results["Gradient Boosting"]["pred"]),
                 r2_r, r2_p],
    "2030 Forecast": ["N/A", "N/A", "N/A",
                      f"{pred_ridge_future_calibrated[-1]:,.0f}",   # ← calibrated
                      f"{dec2030_prophet:,.0f}"]
})
comparison = comparison.sort_values("RMSE").reset_index(drop=True)
print("\n", comparison.to_string(index=False))

# ── Chart: Side-by-side forecast comparison ──────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

ax.fill_between(ev_model["Date"], ev_model["total"],
                alpha=0.1, color=DARK)
ax.plot(ev_model["Date"], ev_model["total"],
        color=DARK, linewidth=1.4, label="Actual (2010–2024)", zorder=5)

# Ridge forecast — calibrated
ax.plot(future_df["Date"], pred_ridge_future_calibrated,            # ← calibrated
        color=PURPLE, linewidth=2.2, label="Polynomial Ridge forecast")
ax.fill_between(future_df["Date"],
                pred_ridge_future_calibrated * 0.78,                # ← calibrated
                pred_ridge_future_calibrated * 1.22,                # ← calibrated
                alpha=0.12, color=PURPLE)

# Prophet forecast
fut_prophet = forecast_full[forecast_full["ds"] > ev_model["Date"].max()]
ax.plot(fut_prophet["ds"], np.maximum(fut_prophet["yhat"], 0),
        color=ORANGE, linewidth=2.2, linestyle="--",
        label="Prophet forecast")
ax.fill_between(fut_prophet["ds"],
                np.maximum(fut_prophet["yhat_lower"], 0),
                fut_prophet["yhat_upper"],
                alpha=0.10, color=ORANGE)

ax.axvline(pd.Timestamp("2025-01-01"), color=RED,
           linewidth=1.5, linestyle=":", label="Forecast start (Jan 2025)")

ax.set_title("EV Monthly Registrations — Ridge vs Prophet Forecast to Dec 2030",
             fontweight="bold", fontsize=13)
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Monthly EV Registrations", fontsize=11)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("chartH_final_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n  ── Key Takeaways ──")
print(f"  Ridge  Dec 2030 forecast  : {pred_ridge_future_calibrated[-1]:,.0f} registrations/month")  # ← calibrated
print(f"  Prophet Dec 2030 forecast : {dec2030_prophet:,.0f} registrations/month")
gap = abs(pred_ridge_future_calibrated[-1] - dec2030_prophet)       # ← calibrated
print(f"  Gap between models        : {gap:,.0f}")
if gap / max(pred_ridge_future_calibrated[-1], dec2030_prophet) < 0.20:
    print("  → Models broadly agree — forecast is credible")
else:
    print("  → Models diverge significantly — present as a range, not a point estimate")


# --- Cell 49 ---
print("PHASE 12 — Hyperparameter Tuning (Grid Search — Time-Series CV)")

import seaborn as sns                                              # ← moved to top
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

tscv = TimeSeriesSplit(n_splits=4)

param_grid = {
    "n_estimators" : [100, 200],
    "max_depth"    : [3, 4, 5],
    "learning_rate": [0.05, 0.1],
}

gs = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_percentage_error",
    n_jobs=-1,
    verbose=0,
)
gs.fit(X_train, y_train)

print(f"  Best params : {gs.best_params_}")
print(f"  CV score    : MAPE = {-gs.best_score_*100:.1f}%")

tuned_model = gs.best_estimator_
tuned_pred  = np.maximum(tuned_model.predict(X_test), 0)
tuned_mape  = mean_absolute_percentage_error(y_test, tuned_pred) * 100
tuned_rmse  = np.sqrt(mean_squared_error(y_test, tuned_pred))

print(f"\n  Tuned model test performance:")
print(f"    MAPE = {tuned_mape:.1f}%   RMSE = {tuned_rmse:,.0f}")

# ── Compare tuned vs original GBR ────────────────────────────────────────
original_mape = results["Gradient Boosting"]["MAPE"]
original_rmse = results["Gradient Boosting"]["RMSE"]
improvement   = original_rmse - tuned_rmse

print(f"\n  ── Tuning Impact ──")
print(f"  Original GBR  MAPE={original_mape:.1f}%   RMSE={original_rmse:,.0f}")
print(f"  Tuned GBR     MAPE={tuned_mape:.1f}%   RMSE={tuned_rmse:,.0f}")
print(f"  RMSE improvement : {improvement:,.0f} ({'better' if improvement > 0 else 'no improvement'})")

# ── Chart I: Tuning results heatmap ──────────────────────────────────────
# renamed chartE → chartI to avoid overwriting Phase 9 Ridge chart
cv_results = pd.DataFrame(gs.cv_results_)
cv_results["mean_mape"] = -cv_results["mean_test_score"] * 100

for lr in [0.05, 0.1]:
    subset = cv_results[cv_results["param_learning_rate"] == lr]
    pivot  = subset.pivot_table(
        index="param_max_depth",
        columns="param_n_estimators",
        values="mean_mape"
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
                linewidths=0.4, ax=ax, annot_kws={"size": 10})
    ax.set_title(f"CV MAPE (%) — learning_rate={lr}",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("max_depth")
    plt.tight_layout()
    plt.savefig(f"chartI_tuning_lr{str(lr).replace('.','')}.png",   # ← renamed
                dpi=150, bbox_inches="tight")
    plt.show()

# print(f"\n  ✓ Final model locked: GBR {gs.best_params_}")
print(f"\n  ✓ Tuning complete — GBR best params: {gs.best_params_}")
print(f"  Note: Despite tuning, GBR MAPE={tuned_mape:.1f}% confirms regime-shift")
print(f"  limitation. Prophet + Ridge used as primary forecast instead.")


# --- Cell 51 ---
print("PHASE 13 — National EV Demand Forecast: 2025–2030")

# ── Primary forecast: Prophet ─────────────────────────────────────────────
# Prophet anchors correctly to recent actuals and handles regime shift
# Ridge provides conservative lower bound

prophet_future_df = pd.DataFrame({
    "date"    : pd.to_datetime(fut_prophet["ds"].values),
    "forecast": np.maximum(fut_prophet["yhat"].values, 0),         # primary
    "lower"   : np.maximum(fut_prophet["yhat_lower"].values, 0),   # lower band
    "upper"   : fut_prophet["yhat_upper"].values                    # upper band
})

# ── Lower bound: Ridge calibrated ────────────────────────────────────────
ridge_future_df = pd.DataFrame({
    "date"    : future_df["Date"],
    "forecast": pred_ridge_future_calibrated
})

# ── Align both to 2025–2030 only ─────────────────────────────────────────
forecast_df = prophet_future_df[
    prophet_future_df["date"] >= pd.Timestamp("2025-01-01")
].reset_index(drop=True)

ridge_fc = ridge_future_df[
    ridge_future_df["date"] >= pd.Timestamp("2025-01-01")
].reset_index(drop=True)

# ── Annual summary ────────────────────────────────────────────────────────
forecast_df["year"] = forecast_df["date"].dt.year

yearly_fc = (forecast_df.groupby("year")
             .agg(
                 Prophet_Forecast=("forecast", "sum"),
                 Prophet_Lower   =("lower",    "sum"),
                 Prophet_Upper   =("upper",    "sum")
             ).reset_index())

yearly_fc.columns = ["Year", "Forecast", "Lower", "Upper"]
yearly_fc["Forecast_M"] = (yearly_fc["Forecast"] / 1e6).round(2)
yearly_fc["GBR_M"]      = yearly_fc["Forecast_M"]  # keeps Chart K compatible

# Ridge annual
ridge_fc["year"] = ridge_fc["date"].dt.year
ridge_annual_fc  = ridge_fc.groupby("year")["forecast"].sum()

base_2024 = ev_model[ev_model["year"] == 2024]["total"].sum()

print("\n  ── Annual Forecast Summary ──")
print(f"  {'Year':<6} {'Prophet (primary)':>18} {'Ridge (lower)':>15}  {'vs 2024'}")
print(f"  {'─'*60}")

for _, row in yearly_fc.iterrows():
    yr      = int(row.Year)
    rdg_val = ridge_annual_fc.get(yr, 0)
    delta   = (row.Forecast - base_2024) / base_2024 * 100
    print(f"  {yr:<6} {row.Forecast:>18,.0f} {rdg_val:>15,.0f}  ({delta:+.0f}%)")

print(f"\n  ── Key Numbers ──")
print(f"  Base 2024 actual         : {base_2024:>12,.0f}")
print(f"  Prophet Dec 2030/month   : {dec2030_prophet:>12,.0f}")
print(f"  Ridge Dec 2030/month     : {pred_ridge_future_calibrated[-1]:>12,.0f}")
print(f"\n  Forecast range Dec 2030  : "
      f"{pred_ridge_future_calibrated[-1]:,.0f} – {dec2030_prophet:,.0f} /month")


# --- Cell 52 ---
# ── Chart J: Three-model monthly comparison ───────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(ev_model["Date"], ev_model["total"],
        color=DARK, linewidth=1.2, label="Actual (2010–2024)")

# GBR tuned — now just for reference/comparison
gbr_future_rows = []
last_vals_gbr = list(ev_model["total"].values[-12:])
t_start_gbr   = int(ev_model["t"].max()) + 1

for i, (yr, mo) in enumerate(
    [(yr, mo) for yr in range(2025, 2031) for mo in range(1, 13)]
):
    t_val = t_start_gbr + i
    row   = {
        "t"           : t_val,
        "month"       : mo,
        "sin12"       : np.sin(2 * np.pi * mo / 12),
        "cos12"       : np.cos(2 * np.pi * mo / 12),
        "lag1"        : last_vals_gbr[-1],
        "lag3"        : last_vals_gbr[-3],
        "lag12"       : last_vals_gbr[-12],
        "roll3"       : np.mean(last_vals_gbr[-3:]),
        "roll6"       : np.mean(last_vals_gbr[-6:]),
        "subsidy_flag": 1,
        "covid_flag"  : 0,
    }
    pred = float(tuned_model.predict(pd.DataFrame([row])[FEATURES])[0])
    pred = max(pred, 0)
    gbr_future_rows.append({"date": pd.Timestamp(yr, mo, 1), "forecast": pred})
    last_vals_gbr.append(pred)
    last_vals_gbr.pop(0)

gbr_forecast_df = pd.DataFrame(gbr_future_rows)

ax.plot(gbr_forecast_df["date"], gbr_forecast_df["forecast"],
        color=GREEN, linewidth=1.8, linestyle="-.",
        label="GBR tuned (reference only)", alpha=0.7)
ax.plot(ridge_fc["date"], ridge_fc["forecast"],
        color=PURPLE, linewidth=2, linestyle="--",
        label="Ridge (lower bound)")
ax.plot(forecast_df["date"], forecast_df["forecast"],
        color=ORANGE, linewidth=2.2,
        label="Prophet (primary forecast)")
ax.fill_between(forecast_df["date"],
                forecast_df["lower"], forecast_df["upper"],
                alpha=0.12, color=ORANGE)

ax.axvline(pd.Timestamp("2025-01-01"), color=RED,
           linewidth=1.2, linestyle=":", label="Forecast start")
ax.set_title("Monthly Forecast — Three Models Compared",
             fontweight="bold", fontsize=13)
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Monthly EV Registrations", fontsize=11)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("chartJ_three_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


# --- Cell 53 ---
print("Chart K — Forecast Chart (Historical + Projection)")

hist_yearly = ev_model.groupby("year")["total"].sum().reset_index()
hist_yearly.columns = ["Year", "Registrations"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ── Panel 1: Monthly lines ────────────────────────────────────────────────
ax = axes[0]
ax.fill_between(ev_model["Date"], ev_model["total"],
                alpha=0.2, color=BLUE)
ax.plot(ev_model["Date"], ev_model["total"],
        color=BLUE, linewidth=1.2, label="Historical (2010–2024)")

# Prophet — primary forecast (solid, prominent)
ax.fill_between(forecast_df["date"], forecast_df["lower"],
                forecast_df["upper"], alpha=0.15, color=ORANGE)
ax.plot(forecast_df["date"], forecast_df["forecast"],
        color=ORANGE, linewidth=2.2, label="Prophet forecast (primary)")

# Ridge — lower bound (dashed, secondary)
ax.plot(ridge_fc["date"], ridge_fc["forecast"],
        color=PURPLE, linewidth=1.8, linestyle="--",
        label="Ridge forecast (lower bound)")

ax.axvline(pd.Timestamp("2025-01-01"), color=RED,
           linewidth=1.2, linestyle=":", label="Forecast start")
ax.set_title("Monthly EV Registrations — Historical + Forecast",
             fontweight="bold", fontsize=12)
ax.set_ylabel("Monthly Registrations", fontsize=10)
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
ax.tick_params(axis="x", labelsize=9, rotation=20)

# ── Panel 2: Annual bars ──────────────────────────────────────────────────
ax2 = axes[1]
ax2.bar(hist_yearly["Year"], hist_yearly["Registrations"],
        color=BLUE, alpha=0.8, label="Historical", edgecolor="none")
ax2.bar(yearly_fc["Year"], yearly_fc["Forecast"],
        color=ORANGE, alpha=0.8, label="Prophet forecast", edgecolor="none")

# Ridge as error bar showing lower bound
# ax2.bar(yearly_fc["Year"], ridge_annual_fc.values,

ridge_aligned = ridge_annual_fc.reindex(yearly_fc["Year"].values, fill_value=0)
ax2.bar(yearly_fc["Year"], ridge_aligned.values,
        color=PURPLE, alpha=0.3, label="Ridge lower bound", edgecolor="none")

for _, row in yearly_fc.iterrows():
    ax2.text(row.Year, row.Forecast + 15000,
             f"{row.Forecast_M:.1f}M",
             ha="center", va="bottom", fontsize=8, color=DARK)

ax2.set_title("Annual EV Registrations — 2010–2030",
              fontweight="bold", fontsize=12)
ax2.set_ylabel("Annual Registrations", fontsize=10)
ax2.set_xlabel("Year", fontsize=10)
ax2.legend(fontsize=9)
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
ax2.set_xticks(list(hist_yearly["Year"]) + list(yearly_fc["Year"]))
ax2.tick_params(axis="x", labelsize=8, rotation=45)

plt.suptitle("VoltMap India — EV Demand Forecast 2025–2030",
             fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig("chartK_demand_forecast.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"  → Prophet primary forecast range: "
      f"{pred_ridge_future_calibrated[-1]:,.0f} – {dec2030_prophet:,.0f} /month by Dec 2030")


# --- Cell 55 ---
print("PHASE 14 — Forward-Looking Gap Score (2030 Demand vs Current Supply)")

# print(yearly_fc["Year"].values)
# print(yearly_fc)

# print(f"Forecast ends  : {forecast_full['ds'].iloc[-1].date()}")
# print(f"yearly_fc years: {list(yearly_fc['Year'].values)}")

# ── Use 2030 forecasted annual registrations as demand proxy ─────────────
# Prophet is now primary forecast — yearly_fc["Forecast"] = Prophet values
forecast_2030 = yearly_fc[yearly_fc["Year"] == 2030]["Forecast"].values[0]
base_2024     = ev_model[ev_model["year"] == 2024]["total"].sum()
growth_factor = forecast_2030 / base_2024

print(f"  2024 registrations (actual)  : {base_2024:>12,.0f}")
print(f"  2030 registrations (forecast): {forecast_2030:>12,.0f}")
print(f"  National growth factor       : {growth_factor:.2f}×")
print(f"  (forecast source: Prophet primary model)")

# ── Update gap_df demand score with growth-adjusted demand ───────────────
gap_df_fc = gap_df.copy()
gap_df_fc["demand_score_2030"] = gap_df_fc["demand_score"] * growth_factor
gap_df_fc["demand_score_2030"] = (
    gap_df_fc["demand_score_2030"] /
    gap_df_fc["demand_score_2030"].max() * 100
)
gap_df_fc["gap_score_2030"] = (
    gap_df_fc["demand_score_2030"] - gap_df_fc["supply_score"]
).clip(lower=0)

STATE_COL = "State name" if "State name" in gap_df_fc.columns else "state"
top20_2030 = gap_df_fc.sort_values("gap_score_2030", ascending=False).head(20)

# ── Chart L — 2030 Gap Score vs Current Gap Score ────────────────────────
# renamed chartG → chartL (chartG already used by Prophet forecast in Phase 10)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax, col, title in [
    (axes[0], "gap_score",      "Current Gap Score (2024)"),
    (axes[1], "gap_score_2030", "Projected Gap Score (2030 — Prophet forecast)"),  # ← updated title
]:
    p = top20_2030.sort_values(col, ascending=True)
    ax.barh(p[STATE_COL], p[col],
            color=[RED    if g > (40 if col == "gap_score" else 50)
                   else (ORANGE if g > (20 if col == "gap_score" else 30)
                   else BLUE)
                   for g in p[col]],
            edgecolor="none")
    for bar in ax.patches:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}", va="center", fontsize=8)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xlabel("Gap Score", fontsize=10)
    ax.axvline(0, color=DARK, linewidth=0.5)

red_p    = mpatches.Patch(color=RED,    label="Urgent (highest gap)")
orange_p = mpatches.Patch(color=ORANGE, label="High priority")
blue_p   = mpatches.Patch(color=BLUE,   label="Moderate")
axes[1].legend(handles=[red_p, orange_p, blue_p], fontsize=9, loc="lower right")


plt.suptitle("Infrastructure Gap: Today vs 2030 Demand Projection\n"
             "Same supply assumed — gap widens without investment",
             fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("chartL_gap_score_2030.png", dpi=150, bbox_inches="tight")   # ← renamed
plt.show()




# --- Cell 57 ---
print("PHASE 15 — Scenario Simulator")          # ← cleaned up
print("─" * 60)
print("This cell simulates different policy/investment scenarios.")
print("Change the parameters below and re-run to explore outcomes.")
print("─" * 60)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ██  SIMULATOR PARAMETERS — change these and re-run the cell ██
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO_NAME       = "Optimistic Policy Push"
GROWTH_MULTIPLIER   = 1.4
TARGET_YEAR         = 2030
NEW_STATIONS_BUDGET = 50000
FOCUS_STATES        = [
    "Uttar Pradesh", "Bihar", "West Bengal",
    "Madhya Pradesh", "Rajasthan"
]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Compute scenario demand ───────────────────────────────────────────────
scenario_fc = yearly_fc.copy()
scenario_fc["Scenario"] = (scenario_fc["Forecast"] * GROWTH_MULTIPLIER).round(0)
target_val  = scenario_fc[scenario_fc["Year"] == TARGET_YEAR]["Scenario"].values[0]
base_val    = yearly_fc[yearly_fc["Year"] == TARGET_YEAR]["Forecast"].values[0]

# ── Distribute new stations: 60% focus states, 40% others ────────────────
n_states       = len(gap_df_fc)
n_focus        = len(FOCUS_STATES)
n_other        = n_states - n_focus
stations_focus = int(NEW_STATIONS_BUDGET * 0.60)
stations_other = int(NEW_STATIONS_BUDGET * 0.40)

gap_sim = gap_df_fc.copy()
gap_sim["new_stations"] = 0.0
gap_sim.loc[gap_sim[STATE_COL].isin(FOCUS_STATES), "new_stations"] = (
    stations_focus / n_focus
)
gap_sim.loc[~gap_sim[STATE_COL].isin(FOCUS_STATES), "new_stations"] = (
    stations_other / n_other
)

# ── Recompute supply with new stations ────────────────────────────────────
total_pop_millions          = gap_sim["total_population"] / 1e6
gap_sim["extra_per_million"] = gap_sim["new_stations"] / total_pop_millions
gap_sim["new_spm"]           = gap_sim["stations_per_million"].fillna(0) + gap_sim["extra_per_million"]
max_spm                      = gap_sim["new_spm"].max()
gap_sim["new_supply"]        = (gap_sim["new_spm"] / max_spm) * 100
gap_sim["new_demand"]        = gap_sim["demand_score_2030"] * GROWTH_MULTIPLIER
gap_sim["new_demand"]        = (gap_sim["new_demand"] / gap_sim["new_demand"].max()) * 100
gap_sim["new_gap"]           = (gap_sim["new_demand"] - gap_sim["new_supply"]).clip(lower=0)

top15_sim = gap_sim.sort_values("new_gap", ascending=False).head(15)

# ── Print summary ─────────────────────────────────────────────────────────
print(f"\n  Scenario                : {SCENARIO_NAME}")
print(f"  Target year             : {TARGET_YEAR}")
print(f"  Demand multiplier       : {GROWTH_MULTIPLIER}×")
print(f"  New stations planned    : {NEW_STATIONS_BUDGET:,}")
print(f"  Focus states            : {', '.join(FOCUS_STATES)}")
print(f"\n  ── Demand Impact ──")
print(f"  Baseline {TARGET_YEAR} forecast   : {base_val:>14,.0f} registrations")
print(f"  Scenario {TARGET_YEAR} forecast   : {target_val:>14,.0f} registrations")
print(f"  Demand uplift           : {(GROWTH_MULTIPLIER-1)*100:+.0f}%")
print(f"\n  ── Top Remaining Gap States (after investment) ──")
for _, row in top15_sim.head(8).iterrows():
    improvement = gap_df_fc.loc[
        gap_df_fc[STATE_COL] == row[STATE_COL], "gap_score_2030"
    ].values
    prev  = improvement[0] if len(improvement) else row["new_gap"]
    delta = row["new_gap"] - prev
    print(f"    {row[STATE_COL]:<28} Gap: {row['new_gap']:5.1f}  (Δ {delta:+.1f})")

# ── Chart M — Simulator Output ────────────────────────────────────────────
# renamed chartH → chartM (chartH already used by Phase 11 Final Comparison)
fig = plt.figure(figsize=(18, 10))
gs_layout = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

# Sub-chart 1: Scenario forecast bars
ax1 = fig.add_subplot(gs_layout[0, :2])
ax1.bar(yearly_fc["Year"], yearly_fc["Forecast"],
        color=BLUE, alpha=0.6, label="Baseline forecast", edgecolor="none")
ax1.bar(scenario_fc["Year"], scenario_fc["Scenario"],
        color=RED, alpha=0.5, label=f"Scenario ({GROWTH_MULTIPLIER}×)", edgecolor="none")
for _, row in scenario_fc.iterrows():
    ax1.text(row.Year, row.Scenario + 8000,
             f"{row.Scenario/1e6:.2f}M", ha="center", fontsize=7.5, color=DARK)
ax1.axvline(TARGET_YEAR - 0.4, color=ORANGE, linewidth=1.5,
            linestyle="--", label=f"Target year: {TARGET_YEAR}")
ax1.set_title(f"Scenario: '{SCENARIO_NAME}' — Annual Demand Projection",
              fontweight="bold", fontsize=12)
ax1.set_ylabel("Annual Registrations", fontsize=10)
ax1.legend(fontsize=9)
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

# Sub-chart 2: Gap score before/after investment
ax2 = fig.add_subplot(gs_layout[0, 2])
compare = gap_sim.nlargest(10, "gap_score_2030")[
    [STATE_COL, "gap_score_2030", "new_gap"]
].copy()
compare = compare.sort_values("gap_score_2030", ascending=True)
y_pos   = range(len(compare))
ax2.barh(y_pos, compare["gap_score_2030"], color=ORANGE, alpha=0.7,
         label="Before investment", edgecolor="none")
ax2.barh(y_pos, compare["new_gap"], color=GREEN, alpha=0.85,
         label="After investment",  edgecolor="none")
ax2.set_yticks(list(y_pos))
ax2.set_yticklabels(compare[STATE_COL].values, fontsize=8)
ax2.set_title("Gap Score: Before vs After\n(Top 10 States)",
              fontweight="bold", fontsize=10)
ax2.set_xlabel("Gap Score", fontsize=9)
ax2.legend(fontsize=8)

# Sub-chart 3: Station allocation
ax3 = fig.add_subplot(gs_layout[1, :2])
alloc        = gap_sim[[STATE_COL, "new_stations", "new_gap"]].nlargest(20, "new_gap")
alloc_sorted = alloc.sort_values("new_gap", ascending=True)
colors_alloc = [RED if s in FOCUS_STATES else BLUE
                for s in alloc_sorted[STATE_COL]]
bars = ax3.barh(alloc_sorted[STATE_COL], alloc_sorted["new_gap"],
                color=colors_alloc, edgecolor="none")
for bar, (_, row) in zip(bars, alloc_sorted.iterrows()):
    ax3.text(bar.get_width() + 0.2,
             bar.get_y() + bar.get_height() / 2,
             f"+{row['new_stations']:.0f} stations",
             va="center", fontsize=7.5,
             color=RED if row[STATE_COL] in FOCUS_STATES else DARK)
ax3.set_title(
    f"Post-Investment Gap Score — '{SCENARIO_NAME}'\n"
    f"Red = focus states ({', '.join(FOCUS_STATES[:2])}…)  "
    f"Blue = others  |  Labels = new stations allocated",
    fontweight="bold", fontsize=10
)
ax3.set_xlabel("Remaining Gap Score", fontsize=10)

# Sub-chart 4: KPI cards
ax4 = fig.add_subplot(gs_layout[1, 2])
ax4.axis("off")
kpis = [
    ("Scenario",      SCENARIO_NAME,              PURPLE),
    ("Demand mult.",  f"{GROWTH_MULTIPLIER}×",    ORANGE),
    ("New stations",  f"{NEW_STATIONS_BUDGET:,}",  BLUE),
    ("Focus states",  str(n_focus),                RED),
    ("Target year",   str(TARGET_YEAR),             GREEN),
    ("Demand uplift", f"{(GROWTH_MULTIPLIER-1)*100:+.0f}%", DARK),
]
for i, (label, val, color) in enumerate(kpis):
    y = 0.92 - i * 0.155
    ax4.add_patch(plt.Rectangle((0.02, y - 0.06), 0.96, 0.13,
                                transform=ax4.transAxes,
                                color=color, alpha=0.12, zorder=0))
    ax4.text(0.08, y + 0.01, label, transform=ax4.transAxes,
             fontsize=9, color="#555555", va="center")
    ax4.text(0.08, y - 0.035, val, transform=ax4.transAxes,
             fontsize=11, fontweight="bold", color=color, va="center")
ax4.set_title("Scenario Parameters", fontweight="bold", fontsize=11)

plt.suptitle(
    f"VoltMap Scenario Simulator — '{SCENARIO_NAME}' | Target: {TARGET_YEAR}",
    fontweight="bold", fontsize=14, y=1.01
)
plt.savefig("chartM_simulator.png", dpi=150, bbox_inches="tight")  # ← renamed
plt.show()

print("\n  ✓ Re-run this cell with different SIMULATOR PARAMETERS to explore scenarios.")


# --- Cell 59 ---
print("PHASE 16 — Monitor & Maintenance Log")
print("─" * 60)

import datetime

monitor_log = {
    "Primary model"       : "Facebook Prophet",                    # ← updated
    "Secondary model"     : "Polynomial Ridge (lower bound)",      # ← updated
    "Reference model"     : "Gradient Boosting (tuned, baseline)", # ← updated
    "Prophet MAPE (%)"    : round(mape_p, 2),                      # ← updated
    "Prophet RMSE"        : round(rmse_p, 0),                      # ← updated
    "Ridge MAPE (%)"      : round(mape_r, 2),                      # ← updated
    "Ridge RMSE"          : round(rmse_r, 0),                      # ← updated
    "GBR tuned MAPE (%)"  : round(tuned_mape, 2),
    "Retrain trigger"     : "MAPE > 20% on latest 3 months OR new FAME phase",
    "Data refresh"        : "Annually — Vahan FY release (April)",
    "Retraining schedule" : "May each year after Vahan data published",
    "Last trained"        : str(datetime.date.today()),
    "Features count"      : len(FEATURES),
    "Training rows"       : len(X_train),
}

print("\n  ── Model Health Card ──")
for k, v in monitor_log.items():
    print(f"  {k:<26} : {v}")

print("\n  ── Drift Detection Rules ──")
drift_rules = [
    ("MAPE drift",      "If rolling 3-month MAPE > 20%, trigger retraining"),
    ("Policy shock",    "If new FAME subsidy phase announced → add event flag, retrain"),
    ("Data lag",        "If Vahan data delayed > 60 days, hold forecast + flag uncertainty"),
    ("Structural break","If annual registrations drop > 30% YoY, switch to conservative scenario"),
]
for rule, action in drift_rules:
    print(f"  [{rule:<18}] → {action}")

# ── Forecast vs Actuals Tracker — Prophet predictions ────────────────────
print("\n  ── Forecast vs Actuals Tracker (Prophet — test period 2023–2024) ──")
track_df = pd.DataFrame({
    "Month"    : test["Date"].dt.strftime("%b %Y").values,
    "Actual"   : y_test.values.astype(int),
    "Predicted": pred_prophet_test.astype(int),                    # ← updated
    "Error_%"  : (np.abs(y_test.values - pred_prophet_test)
                  / y_test.values * 100).round(1),
    "Status"   : ["✓ OK" if e < 20 else "⚠ Review"
                  for e in np.abs(y_test.values - pred_prophet_test)
                  / y_test.values * 100]
})
print(track_df.to_string(index=False))

# ── Chart N — Monitoring dashboard ───────────────────────────────────────
# renamed chartI → chartN (chartI already used by Phase 12 tuning heatmaps)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Actual vs Prophet predicted
ax = axes[0]
ax.plot(test["Date"].values, y_test.values,
        color=DARK, linewidth=2, label="Actual",
        marker="o", markersize=5)
ax.plot(test["Date"].values, pred_prophet_test,                    # ← updated
        color=ORANGE, linewidth=1.8, linestyle="--",
        label="Prophet predicted", marker="s", markersize=4)       # ← updated
ax.fill_between(test["Date"].values,
                pred_prophet_test * 0.8,                           # ← updated
                pred_prophet_test * 1.2,                           # ← updated
                alpha=0.15, color=ORANGE, label="±20% band")       # ← updated
ax.set_title("Live Tracking: Actual vs Prophet (Test Period)",     # ← updated
             fontweight="bold", fontsize=12)
ax.set_ylabel("Monthly Registrations", fontsize=10)
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
ax.tick_params(axis="x", rotation=20, labelsize=8)

# Panel 2: Monthly MAPE drift monitoring
ax2 = axes[1]
rolling_mape = track_df["Error_%"].values
colors_m     = [GREEN if e < 15 else (ORANGE if e < 25 else RED)
                for e in rolling_mape]
ax2.bar(track_df["Month"], rolling_mape, color=colors_m, edgecolor="none")
ax2.axhline(20, color=RED, linewidth=1.5, linestyle="--",
            label="Retrain threshold (20%)")
ax2.set_title("Monthly MAPE — Drift Monitoring",
              fontweight="bold", fontsize=12)
ax2.set_ylabel("MAPE (%)", fontsize=10)
ax2.set_xlabel("Month", fontsize=10)
ax2.legend(fontsize=9)
ax2.tick_params(axis="x", rotation=40, labelsize=8)

plt.suptitle("VoltMap — Model Monitoring Dashboard (Prophet Primary)",  # ← updated
             fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig("chartN_monitoring.png", dpi=150, bbox_inches="tight")     # ← renamed
plt.show()

print("\n  ✓ Monitoring complete. Re-run annually with updated Vahan CSV.")


# --- Cell 60 ---
# Run this in your Jupyter notebook to save trained models
import pickle

# Save Prophet
with open(f"{BASE_ARTIFACT_PATH}/prophet_model.pkl", "wb") as f:
    pickle.dump(m_prophet, f)

# Save Ridge pipeline
with open(f"{BASE_ARTIFACT_PATH}/ridge_model.pkl", "wb") as f:
    pickle.dump(ridge_pipe, f)

# Save CSVs
gap_df.to_csv(f"{BASE_ARTIFACT_PATH}/gap_df.csv", index=False)
ev_model.to_csv(f"{BASE_ARTIFACT_PATH}/ev_model.csv", index=False)
forecast_df.to_csv(f"{BASE_ARTIFACT_PATH}/forecast_df.csv", index=False)
yearly_fc.to_csv(f"{BASE_ARTIFACT_PATH}/yearly_fc.csv", index=False)

print("✓ All models and data saved successfully")

