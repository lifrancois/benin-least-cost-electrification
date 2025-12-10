#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

DATA_DIR = Path(r"C:\VIDA_benin_case")

settlements_path = DATA_DIR / "Benin_settlement_properties.geojson"
lines_path       = DATA_DIR / "Benin_existing_transmission_lines_2017.geojson"

settlements = gpd.read_file(settlements_path)
lines       = gpd.read_file(lines_path)

print(len(settlements), "settlements")
print(len(lines), "lignes")
print(settlements.crs, lines.crs)
settlements.head()


# In[2]:


if settlements.crs is None:
    settlements = settlements.set_crs(epsg=4326)
if lines.crs is None:
    lines = lines.set_crs(epsg=4326)

settlements = settlements.to_crs(epsg=3857)
lines       = lines.to_crs(epsg=3857)

settlements["population_clean"] = settlements["population"].fillna(0).clip(lower=0)
settlements["households"] = settlements["population_clean"] / 5

if "num_connections" in settlements.columns:
    settlements["num_connections_clean"] = settlements["num_connections"].fillna(
        settlements["households"]
    )
else:
    settlements["num_connections_clean"] = settlements["households"]

settlements["has_school"] = settlements.get("num_education_facilities", 0).fillna(0) > 0
settlements["has_health"] = settlements.get("num_health_facilities", 0).fillna(0) > 0

print(settlements[[
    "population_clean","households",
    "num_connections_clean","has_school","has_health"
]].head())


# In[3]:


buffer_radius_km = 2
buffer_radius_m  = buffer_radius_km * 1000

buffer_geom = settlements.geometry.buffer(buffer_radius_m)

sett_buf = gpd.GeoDataFrame(
    settlements[["population_clean"]].copy(),
    geometry=buffer_geom,
    crs=settlements.crs
)

join = gpd.sjoin(
    sett_buf[["geometry"]],
    settlements[["population_clean", "geometry"]],
    how="left",
    predicate="intersects"
)

pop_sum = join.groupby(level=0)["population_clean"].sum()
area_km2 = np.pi * (buffer_radius_km ** 2)
density = (pop_sum / area_km2).reindex(settlements.index).fillna(0)

settlements["pop_density_km2"] = density

print(settlements["pop_density_km2"].describe())


# In[4]:


settlements["dist_grid_km"] = settlements.geometry.apply(
    lambda x: lines.distance(x).min() / 1000
)

print(settlements["dist_grid_km"].describe())


# In[5]:


def base_electrification_rate(dist_km: float) -> float:
    if dist_km < 5:
        return 0.6
    elif dist_km < 15:
        return 0.4
    else:
        return 0.2


def demand_time_series(row):
    hh   = row["households"]
    dist = row["dist_grid_km"]

    base_rate = base_electrification_rate(dist)

    extra = 0.0
    if row["has_school"]:
        extra += SCHOOL_DEMAND
    if row["has_health"]:
        extra += HEALTH_DEMAND

    annual = []
    discounted = 0.0

    for t in range(T):
        elec_rate_t = min(1.0, base_rate + 0.02 * t)
        kwh_hh_t    = BASE_KWH * ((1 + GROWTH) ** t)
        d_t = elec_rate_t * hh * kwh_hh_t + extra

        annual.append(d_t)
        discounted += d_t / ((1 + r) ** (t + 1))

    return pd.Series({
        "demand_year_0":  annual[0],
        "demand_year_14": annual[-1],
        "discounted_demand": discounted
    })


# In[6]:


T = 15
r = 0.08
BASE_KWH = 350
GROWTH = 0.05
SCHOOL_DEMAND = 5000
HEALTH_DEMAND = 8000

# ... def base_electrification_rate / def demand_time_series ...

demand_df = settlements.apply(demand_time_series, axis=1)
settlements = pd.concat([settlements, demand_df], axis=1)

print(settlements[[
    "dist_grid_km","households",
    "demand_year_0","demand_year_14","discounted_demand"
]].head())


# In[7]:


# ----------------- 6. COST PARAMETERS -----------------
params = {
    "grid": {
        "line_cost_per_km": 24000,
        "lv_connection_cost": 650,
        "service_connection_cost": 200,
        "opex_rate": 0.04,
    },
    "mini_grid": {
        "gen_cost_per_kw": 3400,
        "storage_cost_per_kwh": 350,
        "lv_cost_per_connection": 550,
        "opex_rate": 0.08,
        "load_factor": 0.35,
    },
    "shs": {
        "capex_per_system": 700,
        "opex_rate": 0.04,
    }
}

def npc(capex, opex_annual, r=0.08, T=15):
    return capex + sum(opex_annual / ((1 + r) ** (t + 1)) for t in range(T))

def lcoe(capex, opex_annual, discounted_demand, r=0.08, T=15):
    if discounted_demand <= 0:
        return np.nan
    return npc(capex, opex_annual, r, T) / discounted_demand

# ----------------- 7. TECHNO-ECONOMIC MODEL -----------------
def cost_model(row):
    n_conn      = float(row["num_connections_clean"])
    dist_grid   = float(row["dist_grid_km"])
    density     = float(row["pop_density_km2"])
    disc_demand = float(row["discounted_demand"])

    # ----- Grid -----
    p = params["grid"]
    capex_grid = (
        p["line_cost_per_km"] * dist_grid +
        p["lv_connection_cost"] * n_conn +
        p["service_connection_cost"] * n_conn
    )
    opex_grid = capex_grid * p["opex_rate"]
    lcoe_grid = lcoe(capex_grid, opex_grid, disc_demand)

    # ----- Mini-grid -----
    p = params["mini_grid"]
    discount_factor_sum = sum(1 / ((1 + r) ** (t + 1)) for t in range(T))
    annual_avg_demand = disc_demand / discount_factor_sum if disc_demand > 0 else 0

    peak_kw = annual_avg_demand / (p["load_factor"] * 8760) if annual_avg_demand > 0 else 0
    battery_kwh = peak_kw * 6  # 6h autonomy

    capex_mg = (
        p["gen_cost_per_kw"] * peak_kw +
        p["storage_cost_per_kwh"] * battery_kwh +
        p["lv_cost_per_connection"] * n_conn
    )
    opex_mg = capex_mg * p["opex_rate"]
    lcoe_mg = lcoe(capex_mg, opex_mg, disc_demand)

    # ----- SHS -----
    p = params["shs"]
    capex_shs = p["capex_per_system"] * n_conn
    opex_shs  = capex_shs * p["opex_rate"]
    lcoe_shs  = lcoe(capex_shs, opex_shs, disc_demand)

   # Very low-density areas -> mini-grid not suitable, SHS preferred
    if density < 80:
        lcoe_mg *= 1.30      # penalize mini-grid
        lcoe_shs *= 0.95     # slight bonus for SHS

    # Intermediate density (80–160):
    # mini-grid can be relevant if far from the grid
    if (density >= 80) and (density < 160) and (dist_grid > 8):
        lcoe_mg *= 0.90      # small bonus for mini-grid
        lcoe_shs *= 1.05

    # High density (>160): mini-grids more suitable than SHS
    if density >= 160:
        lcoe_mg *= 0.80
        lcoe_shs *= 1.10

    # Close to the grid + very dense -> grid can become competitive again
    if (dist_grid < 5) and (density >= 160):
        lcoe_grid *= 0.75
        lcoe_mg   *= 0.95
        lcoe_shs  *= 1.15

    # Far from MV lines -> grid penalized
    if dist_grid > 10:
        lcoe_grid *= 1.40


    options = {
        "grid":      lcoe_grid,
        "mini_grid": lcoe_mg,
        "shs":       lcoe_shs,
    }

    best = min(options, key=lambda k: options[k] if not np.isnan(options[k]) else np.inf)

    return pd.Series({
        "lcoe_grid":      lcoe_grid,
        "lcoe_mini_grid": lcoe_mg,
        "lcoe_shs":       lcoe_shs,
        "best_option":    best
    })


# In[8]:


# ---- Cost + best option ----
costs = settlements.apply(cost_model, axis=1)
settlements_scen1 = pd.concat([settlements, costs], axis=1)

print("OK : settlements_scen1 created")
print(settlements_scen1.columns)




# In[9]:


print(settlements_scen1["best_option"].value_counts(normalize=True) * 100)


# In[10]:


settlements_scen1[["pop_density_km2","dist_grid_km",
                   "lcoe_grid","lcoe_mini_grid","lcoe_shs"]].describe()


# In[12]:


# ---------------------------------------------------------
# EXPORT SCENARIO 1 (CSV + GEOJSON)
# ---------------------------------------------------------

# 1) Reproject to WGS84 for QGIS / GeoJSON
settlements_scen1_export = settlements_scen1.to_crs(epsg=4326)

# 2) Useful columns for analysis and mapping
cols = [
    "identifier" if "identifier" in settlements_scen1_export.columns else settlements_scen1_export.index.name,
    "population_clean",
    "households",
    "num_connections_clean",
    "pop_density_km2",
    "dist_grid_km",
    "demand_year_0",
    "demand_year_14",
    "discounted_demand",
    "lcoe_grid",
    "lcoe_mini_grid",
    "lcoe_shs",
    "best_option",
    "geometry"
]

# Some datasets do not contain an 'identifier' column
cols = [c for c in cols if c in settlements_scen1_export.columns]

# 3) Export CSV
csv_path = DATA_DIR / "benin_scenario1_results_clean.csv"
settlements_scen1_export.drop(columns="geometry").to_csv(csv_path, index=False)

# 4) Export GEOJSON
geojson_path = DATA_DIR / "benin_scenario1_results_clean.geojson"
settlements_scen1_export.to_file(geojson_path, driver="GeoJSON")

print("✅ EXPORT TERMINÉ")
print("📄 CSV  :", csv_path)
print("🗺️  GEOJSON :", geojson_path)


# In[13]:


# =========================================================
# SCENARIO 2: Mini-grid & Grid Support Program
# - 30% CAPEX subsidy for mini-grids
# - 20% reduction in grid extension cost
# - SHS unchanged
# =========================================================

from copy import deepcopy

# 1) Copy the parameters from Scenario 1
params_scen2 = deepcopy(params)

# 2) Apply policy assumptions
# → 30% subsidy on mini-grid CAPEX (generation + storage + LV)
params_scen2["mini_grid"]["gen_cost_per_kw"]        *= 0.70
params_scen2["mini_grid"]["storage_cost_per_kwh"]   *= 0.70
params_scen2["mini_grid"]["lv_cost_per_connection"] *= 0.70

# → 20% reduction in grid extension cost
params_scen2["grid"]["line_cost_per_km"] *= 0.80


# 3) Techno-economic model for Scenario 2
def cost_model_scen2(row):
    n_conn      = float(row["num_connections_clean"])
    dist_grid   = float(row["dist_grid_km"])
    density     = float(row["pop_density_km2"])
    disc_demand = float(row["discounted_demand"])

    # ----- Grid -----
    p = params_scen2["grid"]
    capex_grid = (
        p["line_cost_per_km"] * dist_grid +
        p["lv_connection_cost"] * n_conn +
        p["service_connection_cost"] * n_conn
    )
    opex_grid = capex_grid * p["opex_rate"]
    lcoe_grid = lcoe(capex_grid, opex_grid, disc_demand)

    # ----- Mini-grid -----
    p = params_scen2["mini_grid"]
    discount_factor_sum = sum(1 / ((1 + r) ** (t + 1)) for t in range(T))
    annual_avg_demand = disc_demand / discount_factor_sum if disc_demand > 0 else 0

    peak_kw = annual_avg_demand / (p["load_factor"] * 8760) if annual_avg_demand > 0 else 0
    battery_kwh = peak_kw * 6  # 6h autonomy

    capex_mg = (
        p["gen_cost_per_kw"] * peak_kw +
        p["storage_cost_per_kwh"] * battery_kwh +
        p["lv_cost_per_connection"] * n_conn
    )
    opex_mg = capex_mg * p["opex_rate"]
    lcoe_mg = lcoe(capex_mg, opex_mg, disc_demand)

    # ----- SHS -----
    p = params_scen2["shs"]
    capex_shs = p["capex_per_system"] * n_conn
    opex_shs  = capex_shs * p["opex_rate"]
    lcoe_shs  = lcoe(capex_shs, opex_shs, disc_demand)

  # ----- Realistic adjustments (same rules as Scenario 1) -----
    # Very low-density → mini-grid penalized, SHS favored
    if density < 80:
        lcoe_mg *= 1.30
        lcoe_shs *= 0.95

   # Intermediate density (80–160) → mini-grid relevant if far from the grid
    if (density >= 80) and (density < 160) and (dist_grid > 8):
        lcoe_mg *= 0.90
        lcoe_shs *= 1.05

    # High density (>160) → mini-grid more suitable than SHS
    if density >= 160:
        lcoe_mg *= 0.80
        lcoe_shs *= 1.10

    # Close to grid + very dense → grid becomes competitive
    if (dist_grid < 5) and (density >= 160):
        lcoe_grid *= 0.75
        lcoe_mg   *= 0.95
        lcoe_shs  *= 1.15

    # Very far from MV lines → grid penalized
    if dist_grid > 10:
        lcoe_grid *= 1.40

    # ----- Select the least-cost option -----
    options = {
        "grid":      lcoe_grid,
        "mini_grid": lcoe_mg,
        "shs":       lcoe_shs,
    }

    best = min(options, key=lambda k: options[k] if not np.isnan(options[k]) else np.inf)

    return pd.Series({
        "lcoe_grid_s2":      lcoe_grid,
        "lcoe_mini_grid_s2": lcoe_mg,
        "lcoe_shs_s2":       lcoe_shs,
        "best_option_s2":    best
    })


# 4) Compute costs for Scenario 2
costs_s2 = settlements.apply(cost_model_scen2, axis=1)
settlements_scen2 = pd.concat([settlements, costs_s2], axis=1)

print("Répartition des options – Scénario 2 (%):")
print(settlements_scen2["best_option_s2"].value_counts(normalize=True) * 100)


# 5) Export (CSV + GeoJSON)
settlements_scen2_export = settlements_scen2.to_crs(epsg=4326)

csv_s2 = DATA_DIR / "benin_scenario2_results.csv"
geojson_s2 = DATA_DIR / "benin_scenario2_results.geojson"

settlements_scen2_export.drop(columns="geometry").to_csv(csv_s2, index=False)
settlements_scen2_export.to_file(geojson_s2, driver="GeoJSON")

print("✅ Scénario 2 exporté :")
print("   CSV     :", csv_s2)
print("   GeoJSON :", geojson_s2)


# In[14]:


print("=== Scénario 1 – Baseline ===")
print(settlements_scen1["best_option"].value_counts(normalize=True) * 100)

print("\n=== Scénario 2 – Politique mini-grids + grid ===")
print(settlements_scen2["best_option_s2"].value_counts(normalize=True) * 100)


# In[ ]:




