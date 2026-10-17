# Shared feature code for Bengaluru Smart Mobility Intelligence (used by the notebook AND app.py)
import json, os, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROADS = json.load(open(os.path.join(HERE, "roads.json")))

def slug(s):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(s)).strip("_").lower()

ROAD_COLS = ["road_" + slug(r) for r in ROADS]
OPS = ["traffic_volume", "average_speed", "travel_time_index", "congestion_level", "road_capacity_utilization",
       "incident_reports", "public_transport_usage", "traffic_signal_compliance", "parking_usage",
       "pedestrian_and_cyclist_count"]
WX = ["rain_sum", "temp_mean", "humidity_mean", "wind_max", "cloud_mean"]
WX_NEXT = [c + "_next" for c in WX]
LAGS = ["cong_lag1", "cong_roll3", "cong_roll7", "cong_std7", "sat_share7", "vol_roll7"]
CAL_N = ["dow_n_sin", "dow_n_cos", "weekend_n", "month_n", "monsoon_n"]      # calendar of the TARGET day (tomorrow)
CAL_T = ["dow_sin", "dow_cos", "weekend", "month", "monsoon"]                # calendar of TODAY

# Task A: forecast congestion tomorrow
A_BASE = OPS + ["roadwork"] + WX + WX_NEXT + CAL_N + ROAD_COLS
A_LAG = LAGS
A_ENG = ["pressure", "vol_x_cong", "speed_x_cong", "incident_rate", "weekend_n_x_vol", "roadwork_x_vol", "rain_next_x_vol"]
# Task B: estimate congestion now from its drivers (no congestion-derived inputs)
B_DRIVERS = ["traffic_volume", "incident_reports", "public_transport_usage", "traffic_signal_compliance",
             "parking_usage", "pedestrian_and_cyclist_count", "roadwork"] + WX + CAL_T + ROAD_COLS
B_ENG = ["incident_rate", "weekend_x_vol", "roadwork_x_vol", "rain_x_vol"]
B_SENSORS = ["average_speed", "travel_time_index", "road_capacity_utilization"]   # co-measured with congestion: reported, not deployed
CONGESTION_DERIVED = {"congestion_level", "vol_x_cong", "speed_x_cong"}           # must never appear in Task B inputs

def build_features(df):
    # df needs: date, road, the OPS columns, roadwork, WX, WX_NEXT and (for Task A) the LAGS columns
    d = df.copy()
    t = pd.to_datetime(d["date"]); n = t + pd.Timedelta(days=1)
    d["dow"] = t.dt.dayofweek
    d["dow_sin"] = np.sin(2 * np.pi * d["dow"] / 7); d["dow_cos"] = np.cos(2 * np.pi * d["dow"] / 7)
    d["weekend"] = (d["dow"] >= 5).astype(int); d["month"] = t.dt.month
    d["monsoon"] = t.dt.month.isin([6, 7, 8, 9]).astype(int)
    dn = n.dt.dayofweek
    d["dow_n_sin"] = np.sin(2 * np.pi * dn / 7); d["dow_n_cos"] = np.cos(2 * np.pi * dn / 7)
    d["weekend_n"] = (dn >= 5).astype(int); d["month_n"] = n.dt.month
    d["monsoon_n"] = n.dt.month.isin([6, 7, 8, 9]).astype(int)
    vol = d["traffic_volume"]
    d["pressure"] = vol * d["road_capacity_utilization"] / 1e5               # load x how full the road already is
    d["vol_x_cong"] = vol * d["congestion_level"] / 1e5
    d["speed_x_cong"] = d["average_speed"] * d["congestion_level"] / 100
    d["incident_rate"] = d["incident_reports"] / (vol / 1e4).clip(lower=1)    # incidents per 10k vehicles
    d["weekend_x_vol"] = d["weekend"] * vol / 1e4
    d["weekend_n_x_vol"] = d["weekend_n"] * vol / 1e4
    d["roadwork_x_vol"] = d["roadwork"] * vol / 1e4
    d["rain_x_vol"] = d["rain_sum"] * vol / 1e5
    d["rain_next_x_vol"] = d["rain_sum_next"] * vol / 1e5
    for r, c in zip(ROADS, ROAD_COLS):
        d[c] = (d["road"] == r).astype(int)
    return d
