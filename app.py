import os, sys, re, json, time, tempfile, datetime
import numpy as np, pandas as pd
import joblib
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

st.set_page_config(page_title="Bengaluru Smart Mobility", page_icon="🚦", layout="wide")
BANDS = [(0, 50, "Light"), (50, 75, "Moderate"), (75, 90, "Heavy"), (90, 101, "Severe")]
def band(x):
    return next(n for lo, hi, n in BANDS if lo <= x < hi)

def show(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)

def table(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)

@st.cache_resource
def load_assets():
    return joblib.load(os.path.join(HERE, "model.pkl")), json.load(open(os.path.join(HERE, "meta.json")))

@st.cache_data
def load_tables():
    rd = lambda f, **k: pd.read_csv(os.path.join(HERE, "data", f), **k)
    return dict(traffic=rd("traffic_clean.csv", parse_dates=["date"]), crash=rd("crashes_clean.csv"), btp=rd("btp_clean.csv"),
                road_stats=rd("road_stats.csv"), area_stats=rd("area_stats.csv"), hotspots=rd("hotspots.csv"))

bundle, meta = load_assets()
T = load_tables()
DEFAULTS, ROAD_AREA, RNG = meta["road_defaults"], meta["road_area"], meta["input_ranges"]
import mobility_report as mr

def estimate(road, date, ov):
    return mr.estimate_now(bundle, meta, road, date, ov)

def forecast(road, date, ov):
    return mr.forecast_tomorrow(bundle, meta, road, date, ov)

# plain-language names for what affects traffic
FRIENDLY = {"traffic_volume": "Number of vehicles on the road", "pedestrian_and_cyclist_count": "Pedestrians and cyclists",
            "incident_reports": "Accidents and breakdowns", "incident_rate": "Accidents per 10,000 vehicles",
            "weekend_x_vol": "Weekend traffic load", "weekend": "Weekend vs weekday", "dow_sin": "Day of the week", "dow_cos": "Day of the week",
            "parking_usage": "Roadside parking", "roadwork_x_vol": "Roadwork combined with heavy traffic", "roadwork": "Roadwork",
            "temp_mean": "Temperature", "cloud_mean": "Weather", "humidity_mean": "Weather", "wind_max": "Weather", "rain_sum": "Rainfall",
            "rain_x_vol": "Rain combined with heavy traffic", "traffic_signal_compliance": "Signal compliance",
            "public_transport_usage": "Public-transport use", "month": "Month of the year", "monsoon": "Monsoon season"}
def friendly(name):
    return "Which road it is" if name.startswith("road_") else FRIENDLY.get(name, name.replace("_", " ").capitalize())

def plain_insights():
    tr, out = T["traffic"], []
    a = T["area_stats"].sort_values("pressure_index", ascending=False).iloc[0]
    rs = T["road_stats"].sort_values("congestion", ascending=False).iloc[0]
    out.append(f"**{a['area']}** is the area under the most traffic pressure, and **{rs['road']}** ({rs['area']}) is the most congested road.")
    out.append(f"Roads were at full gridlock (score 100) on **{tr['saturated'].mean():.0%}** of recorded days.")
    wk = tr.assign(w=tr["date"].dt.dayofweek >= 5).groupby("w")["congestion_level"].mean()
    d = wk.get(True, np.nan) - wk.get(False, np.nan)
    out.append("Weekends and weekdays look about the same." if abs(d) < 2 else f"Weekends are about **{abs(d):.0f} points {'busier' if d > 0 else 'quieter'}** than weekdays.")
    inc = tr.groupby(tr["incident_reports"].clip(upper=3))["cong_vs_road"].mean()
    out.append(f"On days with 3 or more accidents/breakdowns, a road is about **{inc.iloc[-1] - inc.iloc[0]:.0f} points more congested** than on days with none.")
    rw = tr.groupby("roadwork")["cong_vs_road"].mean(); dr = rw.get(1.0, 0) - rw.get(0.0, 0)
    out.append("Roadwork shows no clear extra congestion in this data." if abs(dr) < 2 else f"Roads with active roadwork are about **{abs(dr):.0f} points {'more' if dr > 0 else 'less'} congested** than usual.")
    wc = tr.groupby("weather_conditions")["cong_vs_road"].mean()
    out.append("The recorded weather conditions show no clear effect on congestion." if wc.max() - wc.min() < 2 else f"Congestion varies by about {wc.max() - wc.min():.0f} points between weather conditions (highest in {wc.idxmax().lower()}).")
    r = tr["public_transport_usage"].corr(tr["cong_vs_road"])
    out.append("Public-transport use shows no clear link to congestion." if abs(r) < 0.05 else f"Higher public-transport use goes with {'more' if r > 0 else 'less'} congestion.")
    cr = T["crash"].iloc[-1]; bt = T["btp"].iloc[-1]
    out.append(f"Road safety: **{int(cr['Killed']):,} people** were killed in **{int(cr['Total']):,} crashes** in {int(cr['Year'])}.")
    out.append(f"Enforcement: **{bt['Total Cases Booked'] / 1e6:.2f} million** traffic cases were booked in {int(bt['YEAR'])}; **{bt['automation_share_pct']:.0f}%** were caught by cameras.")
    return out

# ------------------------------------------------------------------ header and sidebar
tr_all = T["traffic"]
st.title("🚦 Bengaluru Smart Mobility")
st.caption("Know where traffic will be heavy, what is causing it, and where to act first.")
with st.sidebar:
    st.header("How to read the scores")
    st.write("**Congestion score** runs from 0 (free-flowing) to 100 (gridlock).")
    st.write("🟢 0–50 Light  \n🟡 50–75 Moderate  \n🟠 75–90 Heavy  \n🔴 90–100 Severe")
    st.divider()
    st.write(f"**Coverage:** {tr_all['road'].nunique()} roads in {tr_all['area'].nunique()} areas")
    st.write(f"**Traffic records:** {tr_all['date'].min():%b %Y} – {tr_all['date'].max():%b %Y}")
    st.caption("Estimates are guidance for planning, not guarantees. Conditions on the day can differ.")

tab_pred, tab_map, tab_cctv, tab_traffic, tab_safety, tab_enf, tab_mob, tab_about = st.tabs(
    ["🔮 Predict Traffic", "🗺️ City Map", "📹 Camera View", "📍 Traffic Overview", "🛡️ Road Safety", "👮 Enforcement", "🚶 Mobility", "ℹ️ Insights & About"])

# ------------------------------------------------------------------ predict
with tab_pred:
    left, right = st.columns([1, 1.2])
    with left:
        road = st.selectbox("Road / intersection", meta["roads"])
        st.caption(f"Area: {ROAD_AREA[road]}")
        date = st.date_input("Date", value=datetime.date.today())
        d = DEFAULTS[road]
        def num(label, col, step):
            lo, hi = RNG[col]
            return st.slider(label, float(lo), float(hi), float(np.clip(d[col], lo, hi)), step=step, key=f"{col}_{road}")
        ov = {"traffic_volume": num("Vehicles on the road", "traffic_volume", 100.0),
              "incident_reports": num("Accidents / breakdowns", "incident_reports", 1.0),
              "public_transport_usage": num("Public-transport use (%)", "public_transport_usage", 1.0),
              "traffic_signal_compliance": num("Signal compliance (%)", "traffic_signal_compliance", 1.0),
              "parking_usage": num("Roadside parking (%)", "parking_usage", 1.0),
              "pedestrian_and_cyclist_count": num("Pedestrians and cyclists", "pedestrian_and_cyclist_count", 5.0),
              "rain_sum": num("Rainfall (mm)", "rain_sum", 0.5), "temp_mean": num("Temperature (°C)", "temp_mean", 0.5)}
        ov["roadwork"] = 1.0 if st.checkbox("Roadwork in progress", value=bool(d["roadwork"]), key=f"rw_{road}") else 0.0
        go_btn = st.button("Predict", type="primary")
    with right:
        if go_btn:
            est = estimate(road, date, ov)
            fc, p_sat = forecast(road, date, ov)
            c1, c2, c3 = st.columns(3)
            c1.metric("Congestion with these conditions", f"{est:.0f}/100", band(est))
            c2.metric("Expected tomorrow", f"{fc:.0f}/100", band(fc))
            c3.metric("Chance of a severe jam tomorrow", f"{p_sat:.0%}")
            st.progress(min(max(p_sat, 0.0), 1.0))
            typical = float(DEFAULTS[road]["cong_roll7"])
            st.info(f"With these conditions, **{road}** is expected to be **{band(est).lower()}** (score {est:.0f}/100). "
                    f"Over the past week it averaged {typical:.0f}. Tomorrow looks **{band(fc).lower()}** ({fc:.0f}/100), with a **{p_sat:.0%}** chance of a severe jam.")
            base = est
            trials = {"Roadwork starts": {"roadwork": 1.0}, "Roadwork ends": {"roadwork": 0.0},
                      "3 more accidents / breakdowns": {"incident_reports": ov["incident_reports"] + 3}, "20% more vehicles": {"traffic_volume": ov["traffic_volume"] * 1.2},
                      "20% fewer vehicles": {"traffic_volume": ov["traffic_volume"] * 0.8}, "Heavy rain (30 mm)": {"rain_sum": 30.0},
                      "20% more public-transport use": {"public_transport_usage": ov["public_transport_usage"] * 1.2}}
            sens = pd.Series({k: estimate(road, date, {**ov, **v}) - base for k, v in trials.items()}).sort_values()
            fig = go.Figure(go.Bar(y=sens.index, x=sens.values, orientation="h", marker_color=["#C44E52" if v > 0 else "#4C72B0" for v in sens.values]))
            fig.update_layout(title="What would change things (points of congestion)", height=340, margin=dict(l=10, r=10, t=40, b=10))
            show(fig)
            if base >= 99:
                st.caption("This road is already at full gridlock (100), so extra load cannot push it higher. Try fewer vehicles to see what helps.")
        else:
            st.write("Set the conditions on the left and press **Predict**.")
        with st.expander("What affects traffic the most?"):
            imp = pd.Series(meta["importance"]["whatif"])
            grp = imp.groupby([friendly(k) for k in imp.index]).sum().sort_values(ascending=False).head(6)
            share = 100 * grp / grp.sum()
            for name, v in share.items():
                st.write(f"**{name}** — {v:.0f}% of the influence")
                st.progress(float(v) / 100)

# ------------------------------------------------------------------ city map
LOC_AREAS = {"Indiranagar": (12.9784, 77.6408), "M.G. Road": (12.9756, 77.6068), "Koramangala": (12.9352, 77.6245), "Jayanagar": (12.9308, 77.5838),
             "Hebbal": (13.0358, 77.5970), "Whitefield": (12.9698, 77.7500), "Yeshwanthpur": (13.0285, 77.5409), "Electronic City": (12.8452, 77.6602)}
# approximate positions of the monitored junctions - edit here if you have exact coordinates
LOC_ROADS = {"100 Feet Road": (12.9719, 77.6412), "CMH Road": (12.9780, 77.6387), "Trinity Circle": (12.9731, 77.6200), "Anil Kumble Circle": (12.9755, 77.6035),
             "Sony World Junction": (12.9367, 77.6269), "Sarjapur Road": (12.9246, 77.6372), "Silk Board Junction": (12.9172, 77.6228), "Jayanagar 4th Block": (12.9250, 77.5830),
             "South End Circle": (12.9381, 77.5802), "Hebbal Flyover": (13.0382, 77.5973), "Ballari Road": (13.0520, 77.5950), "ITPL Main Road": (12.9857, 77.7380),
             "Marathahalli Bridge": (12.9569, 77.7011), "Tumkur Road": (13.0290, 77.5320), "Yeshwanthpur Circle": (13.0236, 77.5505), "Hosur Road": (12.8560, 77.6510)}
CITY_CENTER = (12.9716, 77.5946)

def hav_km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(2 * 6371 * np.arcsin(np.sqrt(h)))

@st.cache_data(show_spinner=False)
def map_table(date_iso, rain):
    rows = []
    for r in meta["roads"]:
        if r not in LOC_ROADS:
            continue
        ov = {"rain_sum": float(rain)}
        est = mr.estimate_now(bundle, meta, r, date_iso, ov)
        fc, p = mr.forecast_tomorrow(bundle, meta, r, date_iso, ov)
        rep = mr.intelligence_report(bundle, meta, r, date_iso, ov)
        rows.append(dict(road=r, area=ROAD_AREA[r], lat=LOC_ROADS[r][0], lon=LOC_ROADS[r][1], today=est, tomorrow=fc, p_severe=p,
                         attention=rep["priority_score"], tier=rep["priority_tier"].title()))
    return pd.DataFrame(rows)

def secret_or_env(name):
    # looks for the key in: environment variable, Streamlit secrets, a secrets.toml next to app.py (in .streamlit/ or streamlit/), or tomtom_key.txt
    v = os.environ.get(name)
    if v:
        return v.strip()
    try:
        v = st.secrets.get(name)
        if v:
            return str(v).strip()
    except Exception:
        pass
    for rel in (".streamlit/secrets.toml", "streamlit/secrets.toml", "secrets.toml"):
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            try:
                m = re.search(r"^\s*" + re.escape(name) + r"\s*=\s*[\"']([^\"']+)[\"']", open(p, encoding="utf-8-sig").read(), re.M)
                if m:
                    return m.group(1).strip()
            except Exception:
                pass
    p = os.path.join(HERE, "tomtom_key.txt")
    if name == "TOMTOM_API_KEY" and os.path.exists(p):
        return open(p, encoding="utf-8-sig").read().strip() or None
    return None

TOMTOM_TILES = {"classic": "https://api.tomtom.com/traffic/map/4/tile/flow/relative/{z}/{x}/{y}.png?key=KEY",
                "orbis": "https://api.tomtom.com/maps/orbis/traffic/flow/raster/tile/{z}/{x}/{y}?apiVersion=2&key=KEY&style=light"}

def tile_xy(lat, lon, z):
    n = 2 ** z
    return int((lon + 180) / 360 * n), int((1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n)

@st.cache_data(ttl=600, show_spinner=False)
def tomtom_probe(key):
    # asks TomTom for one Bengaluru tile to check the key really works; returns (endpoint kind or None, reason)
    import urllib.request, urllib.error
    x, y = tile_xy(CITY_CENTER[0], CITY_CENTER[1], 12)
    last = "network"
    for kind, tpl in TOMTOM_TILES.items():
        url = tpl.replace("{z}", "12").replace("{x}", str(x)).replace("{y}", str(y)).replace("KEY", key)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                if r.status == 200 and "image" in r.headers.get("Content-Type", ""):
                    return kind, "ok"
        except urllib.error.HTTPError as e:
            last = str(e.code)
        except Exception:
            last = "network"
    return None, last

def tomtom_problem(reason):
    if reason == "network":
        return "Couldn't reach TomTom. Check your internet connection and try again."
    if reason in ("401", "403"):
        return ("TomTom did not accept this key. Check that you copied all of it, that the Traffic API is ticked for this key in your TomTom dashboard, "
                "that no domain restriction blocks localhost, and that today's free limit has not been used up.")
    return f"TomTom returned an unexpected answer ({reason}). Check the key and try again."

def flow_url_for(kind, key, detail, thickness, dark):
    if kind == "classic":
        style = "relative-delay" if detail == "Only slow roads" else "relative"
        return f"https://api.tomtom.com/traffic/map/4/tile/flow/{style}/{{z}}/{{x}}/{{y}}.png?key={key}&thickness={int(thickness)}"
    return f"https://api.tomtom.com/maps/orbis/traffic/flow/raster/tile/{{z}}/{{x}}/{{y}}?apiVersion=2&key={key}&style={'dark' if dark else 'light'}"

@st.cache_data(ttl=3600, show_spinner=False)
def geocode(q):
    # free OpenStreetMap place search, limited to the Bengaluru area
    import urllib.request, urllib.parse
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": q + ", Bengaluru", "format": "json", "limit": 1, "viewbox": "77.40,13.20,77.85,12.75", "bounded": 1})
    req = urllib.request.Request(url, headers={"User-Agent": "BengaluruSmartMobility/1.0 (student project)"})
    with urllib.request.urlopen(req, timeout=8) as r:
        data = json.loads(r.read().decode("utf-8"))
    return (float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("display_name", q)) if data else None

def city_map(df, col, label, center, zoom, my=None, flow_url=None, base="open-street-map", show_markers=True, size=28, pin=None):
    new = hasattr(go, "Scattermap")
    Sc = go.Scattermap if new else go.Scattermapbox
    dark = base == "carto-darkmatter"
    fig = go.Figure()
    fig.add_trace(Sc(lat=[v[0] for v in LOC_AREAS.values()], lon=[v[1] for v in LOC_AREAS.values()], mode="text", text=list(LOC_AREAS.keys()),
                     textfont=dict(size=13, color="#d0d0d0" if dark else "#555"), hoverinfo="skip", showlegend=False))
    has = df[col].notna()
    hover = [f"<b>{r.road}</b><br>{r.area}<br>Congestion today: {r.today:.0f}/100 ({mr.band(r.today)})<br>Tomorrow: {r.tomorrow:.0f}/100<br>"
             f"Chance of a severe jam tomorrow: {r.p_severe:.0%}<br>Attention needed: {r.tier}" for r in df.itertuples()]
    if show_markers:
        fig.add_trace(Sc(lat=df.lat[has], lon=df.lon[has], mode="markers+text", text=[f"{v:.0f}" for v in df[col][has]], textposition="middle center",
                         textfont=dict(size=10, color="white"), hovertext=[h for h, k in zip(hover, has) if k], hoverinfo="text",
                         marker=dict(size=size, color=df[col][has], colorscale="RdYlGn_r", cmin=0, cmax=100, showscale=True,
                                     colorbar=dict(title=label, thickness=12, len=0.6)), showlegend=False))
        if (~has).any():
            fig.add_trace(Sc(lat=df.lat[~has], lon=df.lon[~has], mode="markers", marker=dict(size=int(size * 0.5), color="#9aa0a6"),
                             hovertext=[f"<b>{r.road}</b><br>No camera reading yet" for r in df[~has].itertuples()], hoverinfo="text", showlegend=False))
    if my:
        fig.add_trace(Sc(lat=[my[0]], lon=[my[1]], mode="markers+text", text=["You"], textposition="top center", marker=dict(size=16, color="#1a73e8"), hoverinfo="skip", showlegend=False))
    if pin:
        fig.add_trace(Sc(lat=[pin[0]], lon=[pin[1]], mode="markers+text", text=[pin[2].split(",")[0]], textposition="top center",
                         textfont=dict(size=12, color="#ff5252"), marker=dict(size=16, color="#ea4335"), hoverinfo="skip", showlegend=False))
    layers = []
    if flow_url:
        layers.append(dict(sourcetype="raster", below="traces", opacity=0.9, source=[flow_url]))
    geo = dict(style=base, center=dict(lat=center[0], lon=center[1]), zoom=zoom, layers=layers)
    fig.update_layout(**({"map": geo} if new else {"mapbox": geo}), margin=dict(l=0, r=0, t=0, b=0), height=680)
    return fig

with tab_map:
    st.subheader("🗺️ Bengaluru traffic map")
    cmap1, cmap2, cmap3 = st.columns([1.5, 1, 1])
    layer = cmap1.radio("Show", ["Congestion today", "Outlook for tomorrow", "Attention needed", "Camera readings"], horizontal=True)
    jump = cmap2.selectbox("Zoom to", ["All Bengaluru"] + list(LOC_AREAS.keys()) + [r for r in meta["roads"] if r in LOC_ROADS])
    rain_map = cmap3.slider("Rainfall today (mm)", 0.0, 40.0, 0.0, 1.0)
    o1, o2, o3 = st.columns([1.1, 1, 1.5])
    style_name = o1.selectbox("Map style", ["Dark (night mode)", "Light (Google-style)", "Standard (OpenStreetMap)"])
    base = {"Dark (night mode)": "carto-darkmatter", "Light (Google-style)": "carto-positron", "Standard (OpenStreetMap)": "open-street-map"}[style_name]
    show_markers = o2.checkbox("Show junction circles", value=True)
    query = o3.text_input("Search a place in Bengaluru", placeholder="e.g. Lalbagh, Majestic, Manyata Tech Park")
    mt = map_table(datetime.date.today().isoformat(), rain_map)
    cams = st.session_state.get("cam_scores", {})
    mt["camera"] = mt["road"].map(lambda r: cams.get(r, {}).get("score"))
    col, lab = {"Congestion today": ("today", "Congestion"), "Outlook for tomorrow": ("tomorrow", "Tomorrow"),
                "Attention needed": ("attention", "Attention"), "Camera readings": ("camera", "Camera")}[layer]
    if jump == "All Bengaluru":
        center, zoom = CITY_CENTER, 10.4
    elif jump in LOC_AREAS:
        center, zoom = LOC_AREAS[jump], 13.2
    else:
        center, zoom = LOC_ROADS[jump], 14.5
    pin = None
    if query.strip():
        try:
            pin = geocode(query.strip())
            if pin:
                center, zoom = (pin[0], pin[1]), 15.0
            else:
                st.warning("Couldn't find that place in Bengaluru. Try a more specific name.")
        except Exception:
            st.warning("Place search isn't available right now. Please try again in a moment.")
    my_pos = None
    try:
        from streamlit_geolocation import streamlit_geolocation
        with st.expander("📍 Show my location"):
            st.caption("Press the button and allow location access in your browser.")
            g = streamlit_geolocation()
            if g and g.get("latitude") and g.get("longitude"):
                my_pos = (g["latitude"], g["longitude"])
    except Exception:
        pass
    stored_key = secret_or_env("TOMTOM_API_KEY")
    flow_key = (stored_key or st.session_state.get("tomtom_key") or "").strip()
    kind, flow_msg = None, None
    if flow_key:
        kind, reason = tomtom_probe(flow_key)
        if not kind:
            flow_msg = tomtom_problem(reason)
    flow_on, detail, thick = False, "All roads", 6
    if kind:
        t1, t2, t3 = st.columns([1.2, 1.3, 1.2])
        flow_on = t1.checkbox("Show live traffic on every road", value=True)
        detail = t2.radio("Traffic detail", ["All roads", "Only slow roads"], horizontal=True, help="'Only slow roads' hides roads where traffic is moving normally, so the map looks cleaner.")
        thick = t3.slider("Line thickness", 1, 15, 6)
    flow_url = flow_url_for(kind, flow_key, detail, thick, base == "carto-darkmatter") if (kind and flow_on) else None
    show(city_map(mt, col, lab, center, zoom, my_pos, flow_url, base, show_markers, 22 if flow_url else 28, pin))
    with st.expander("🚦 Live traffic for every road", expanded=not kind):
        if stored_key:
            st.write(" so live traffic starts automatically.")
        else:
            st.write("Live road-by-road colours (green, orange, red, like Google Maps) come from a traffic provider. TomTom's free plan needs no credit card and includes 50,000 map-tile requests a day.")
            st.markdown("1. Go to **developer.tomtom.com** and click **Sign up** (free).  \n"
                        "2. Confirm your email and log in.  \n"
                        "3. Open **Keys** and click **Add a new key** (or use the default key). Give it a name and make sure **Traffic API** is ticked.  \n"
                        "4. Copy the key and paste it below, or save it permanently in a file named `.streamlit/secrets.toml` containing  \n"
                        "`TOMTOM_API_KEY = \"your_key\"`")
            st.text_input("Paste your TomTom key", type="password", key="tomtom_key")
        if flow_msg:
            st.error(flow_msg)
        elif kind:
            st.success("Live traffic is on. Zoom and pan the map to see traffic on every road; it updates as you reload the map.")
        # st.caption("Keep your key private: do not share the app link while a key is entered, and do not upload the key to GitHub.")
    ref_pt, ref_name = (my_pos, "you") if my_pos else ((pin[0], pin[1]), pin[2].split(",")[0]) if pin else (None, None)
    if ref_pt:
        mt["km"] = [hav_km(ref_pt, (a, b)) for a, b in zip(mt["lat"], mt["lon"])]
        near = mt.sort_values("km").iloc[0]
        st.info(f"Nearest monitored location to {ref_name}: **{near['road']}** ({near['km']:.1f} km away) - congestion {near['today']:.0f}/100 ({mr.band(near['today']).lower()}).")
    if layer == "Camera readings" and not cams:
        st.info("No camera readings yet. In the Camera View tab, choose the road a camera watches and press Analyse - its reading will appear here.")
    st.caption("Circles are the monitored junctions (the colour and number show the score); positions are approximate. The scores are estimates for each monitored junction, "
               "not a live feed from every road in the city. The road colours come from TomTom's live traffic when a key is set.")
    st.subheader("All monitored locations")
    tbl = mt[["road", "area", "today", "tomorrow", "p_severe", "tier"]].copy()
    tbl["p_severe"] = (tbl["p_severe"] * 100).round(0)
    table(tbl.sort_values("today", ascending=False).rename(columns={"road": "Location", "area": "Area", "today": "Congestion today", "tomorrow": "Tomorrow",
                                                                    "p_severe": "Chance of severe jam tomorrow (%)", "tier": "Attention needed"}).round(0))

# ------------------------------------------------------------------ camera
with tab_cctv:
    st.subheader("📹 Camera view: how heavy is the traffic right now?")
    st.caption("Upload a CCTV picture or video, or connect a live camera. The system spots and counts vehicles, measures how much road space they take up, and rates the congestion from 0 to 100.")
    WEIGHTS, REFP = os.path.join(HERE, "yolo_bmd45.pt"), os.path.join(HERE, "data", "pcu_reference.json")
    vision_ok, det, ref = False, None, None
    if not (os.path.exists(WEIGHTS) and os.path.exists(REFP)):
        st.warning("The camera analysis files are missing. Please complete the camera set-up step to enable this tab.")
    else:
        try:
            import vision_utils as vu
            @st.cache_resource
            def load_detector():
                return vu.load_detector(WEIGHTS), vu.load_reference(REFP)
            det, ref = load_detector()
            vision_ok = True
        except Exception:
            st.error("Camera analysis is not available on this system yet. Please contact the administrator.")
    if vision_ok:
        c1, c2, c3 = st.columns([1.3, 1, 1])
        source = c1.radio("Camera source", ["Sample CCTV clip", "Sample frames", "Upload image", "Upload video", "Live stream URL"])
        NO_ROAD = "- Not linked to a road (camera score only) -"
        road_c = c2.selectbox("Which road does this camera watch?", [NO_ROAD] + meta["roads"], key="cctv_road",
                              help="A camera cannot tell the system where it is installed. Choose the road this camera actually watches to combine what it sees with that road's usual traffic, accident risk and signal compliance. Leave it unlinked to see the camera's own score only.")
        conf = c3.slider("Detection sensitivity", 0.10, 0.80, 0.25, 0.05, help="Lower finds more vehicles but may include false ones; higher is stricter.")
        sample_files = sorted(os.listdir(os.path.join(HERE, "data", "sample_frames"))) if os.path.isdir(os.path.join(HERE, "data", "sample_frames")) else []
        sample_choice = up_img = up_vid = None
        every_s, stream_url, auto = 2.0, "", False
        if source == "Sample frames" and sample_files:
            sample_choice = st.selectbox("Picture", sample_files)
        elif source == "Upload image":
            up_img = st.file_uploader("CCTV picture", type=["jpg", "jpeg", "png"])
        elif source == "Upload video":
            up_vid = st.file_uploader("CCTV video", type=["mp4", "avi", "mov", "mkv"])
            every_s = st.slider("Check one frame every (seconds)", 0.5, 10.0, 2.0, 0.5)
        elif source == "Live stream URL":
            stream_url = st.text_input("Camera link (rtsp://..., http://..., or a webcam number like 0)", key="stream_url")
            st.caption("Use the camera's *viewing* address, not a publishing/upload address. If the camera needs a login, put it in the link, e.g. rtsp://username:password@192.168.0.20:554/stream1. "
                       "You can also type a webcam number (0) or the path of a video file on this computer.")
            auto = st.checkbox("Refresh every 5 seconds")
        trigger = st.button("Capture and analyse" if source == "Live stream URL" else "Analyse", type="primary")
        if source == "Live stream URL" and auto and stream_url:
            trigger = True

        def collect():
            if source == "Sample CCTV clip":
                return [(f"{t:.0f}s", fr) for t, fr in vu.video_frames(os.path.join(HERE, "data", "sample_cctv.mp4"), every_s=1.0, max_frames=12)]
            if source == "Sample frames":
                return [(sample_choice, np.asarray(Image.open(os.path.join(HERE, "data", "sample_frames", sample_choice)).convert("RGB")))] if sample_choice else []
            if source == "Upload image":
                return [(up_img.name, np.asarray(Image.open(up_img).convert("RGB")))] if up_img else []
            if source == "Upload video":
                if not up_vid:
                    return []
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(up_vid.name)[1], delete=False) as tf:
                    tf.write(up_vid.read())
                return [(f"{t:.0f}s", fr) for t, fr in vu.video_frames(tf.name, every_s=every_s, max_frames=24)]
            if source == "Live stream URL" and stream_url:
                return [(time.strftime("%H:%M:%S"), vu.grab_stream_frame(stream_url))]
            return []

        if trigger:
            try:
                frames = collect()
                if not frames:
                    st.info("Choose or upload something to analyse first.")
                else:
                    prog, results = st.progress(0.0), []
                    for k, (lab, fr) in enumerate(frames):
                        results.append(vu.detect_frame(det, fr, ref, conf=conf))
                        prog.progress((k + 1) / len(frames))
                    st.session_state["cctv"] = dict(labels=[l for l, _ in frames], results=results)
            except Exception:
                if source == "Live stream URL":
                    st.error("Couldn't connect to that camera link. Common reasons: it is a publishing (upload) address rather than a viewing address; the camera needs a username and password; "
                             "the camera is offline; or it is a web page / YouTube / Google Drive / WhatsApp link. For a video file, choose 'Upload video' instead.")
                else:
                    st.error("Could not read that file. Check it is a valid picture or video and try again.")
        S = st.session_state.get("cctv")
        if S:
            res = S["results"]
            vision = vu.aggregate(res)
            linked = road_c != NO_ROAD
            rep = mr.intelligence_report(bundle, meta, road_c, pd.Timestamp(datetime.date.today()), None, vision) if linked else None
            if linked:
                st.session_state.setdefault("cam_scores", {})[road_c] = dict(score=vision["visual_index"], time=time.strftime("%H:%M:%S"))
            if vision["n_vehicles"] == 0:
                st.warning("No vehicles were detected in this picture. If the picture looks grey or blurry, the camera is still starting up - try again in a few seconds. "
                           "A camera aimed at an empty road or the road surface will also show none.")
            cols = st.columns(4 if linked else 3)
            cols[0].metric("Vehicles seen", f"{vision['n_vehicles']:.0f}")
            cols[1].metric("Road space taken (car-equivalents)", f"{vision['pcu']:.1f}")
            cols[2].metric("Camera congestion score", f"{vision['visual_index']:.0f}/100", mr.band(vision["visual_index"]))
            if linked:
                cols[3].metric(f"Attention needed — {road_c}", rep["priority_tier"].title(), f"{rep['priority_score']:.0f}/100")
                st.caption("The picture, counts and camera score come only from the camera. The road you chose adds that road's usual traffic, accident risk and signal compliance to decide how urgent the situation is.")
            else:
                st.caption("Showing the camera's own score only. To see how urgent the situation is, choose the road this camera watches in the box above.")
            idx = 0
            if len(res) > 1:
                idx = st.slider("Frame", 0, len(res) - 1, int(np.argmax([r_["visual_index"] for r_ in res])))
                tl = pd.DataFrame({"time": S["labels"], "Camera congestion score": [r_["visual_index"] for r_ in res]})
                show(px.line(tl, x="time", y="Camera congestion score", markers=True, title="How congestion changes across the video", labels={"time": "Time in video"}))
            cc1, cc2 = st.columns([1.6, 1])
            cc1.image(res[idx]["annotated"], caption=f"Frame {S['labels'][idx]} — {res[idx]['n_vehicles']} vehicles detected", channels="RGB")
            counts = pd.DataFrame({"Vehicle type": list(res[idx]["counts"].keys()), "Count": list(res[idx]["counts"].values())}).sort_values("Count")
            if len(counts):
                show(px.bar(counts, x="Count", y="Vehicle type", orientation="h", title="Vehicles by type (this frame)"))
            st.subheader("Situation summary")
            if linked:
                st.info(f"**{road_c}** ({rep['area']}): the camera rates traffic at **{vision['visual_index']:.0f}/100**, while the usual outlook for this road is **{rep['expected_congestion']:.0f}/100**. "
                        f"Combined, congestion is rated **{rep['congestion_used']:.0f}/100**. About **{rep['expected_incidents']:.1f}** accidents/breakdowns are expected on a day like this.")
                st.write("**Suggested actions**")
                for line in rep["recommendations"]:
                    st.write("•", line)
            else:
                st.info(f"The camera rates the traffic it sees at **{vision['visual_index']:.0f}/100** ({mr.band(vision['visual_index']).lower()}), with **{vision['n_vehicles']:.0f}** vehicles taking up about **{vision['pcu']:.1f}** car-equivalents of road space.")
                if vision["visual_index"] >= 85:
                    st.write("• The camera shows heavy traffic right now - consider checking this location.")
            st.caption("One camera shows one stretch of road, not the whole corridor.")
            if source == "Live stream URL" and auto:
                time.sleep(5)
                st.rerun()

# ------------------------------------------------------------------ overview
with tab_traffic:
    a = T["area_stats"].sort_values("pressure_index")
    show(px.bar(a, x="pressure_index", y="area", orientation="h", title="Traffic pressure by area (100 = highest)", labels={"pressure_index": "Pressure score", "area": ""}))
    dow = tr_all.assign(dow=tr_all["date"].dt.day_name()).groupby("dow")["congestion_level"].mean().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    show(px.bar(dow, title="Average congestion by day of the week", labels={"value": "Congestion score", "dow": ""}))
    cb = tr_all.assign(b=pd.cut(tr_all["road_capacity_utilization"], 12)).groupby("b", observed=True).agg(x=("road_capacity_utilization", "mean"), y=("congestion_level", "mean"), n=("congestion_level", "size")).reset_index(drop=True)
    cb = cb[cb["n"] >= 20]
    show(px.line(cb, x="x", y="y", markers=True, title="The fuller the road, the higher the congestion", labels={"x": "How full the road is (%)", "y": "Congestion score"}))
    st.subheader("Where to act first")
    hs = T["hotspots"][["area", "road", "congestion", "saturated_share", "compliance", "attention_score", "priority"]].copy()
    hs["saturated_share"] = (hs["saturated_share"] * 100).round(0)
    hs["priority"] = hs["priority"].map({"Act now": "Act now", "Elevated": "Watch closely", "Monitor": "Routine"}).fillna(hs["priority"])
    table(hs.rename(columns={"area": "Area", "road": "Road", "congestion": "Average congestion", "saturated_share": "Days at gridlock (%)",
                             "compliance": "Signal compliance (%)", "attention_score": "Attention score", "priority": "Level"}).round(1))

# ------------------------------------------------------------------ safety
with tab_safety:
    cr = T["crash"]
    k1, k2, k3 = st.columns(3)
    k1.metric(f"People killed ({int(cr['Year'].iloc[-1])})", f"{int(cr['Killed'].iloc[-1]):,}")
    k2.metric("Crashes", f"{int(cr['Total'].iloc[-1]):,}")
    k3.metric("Deaths per 100 crashes", f"{cr['killed_per_100_crashes'].iloc[-1]:.1f}")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Crashes and deaths each year", "Deaths per 100 crashes"))
    fig.add_trace(go.Bar(x=cr["Year"], y=cr["Total"], name="Crashes"), 1, 1)
    fig.add_trace(go.Scatter(x=cr["Year"], y=cr["Killed"], name="Killed"), 1, 1)
    fig.add_trace(go.Scatter(x=cr["Year"], y=cr["killed_per_100_crashes"], name="Deaths per 100 crashes", mode="lines+markers"), 1, 2)
    show(fig)
    st.caption("The most recent year may be incomplete; read the latest change with care.")

# ------------------------------------------------------------------ enforcement
with tab_enf:
    bt = T["btp"]
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Cases booked ({int(bt['YEAR'].iloc[-1])})", f"{bt['Total Cases Booked'].iloc[-1] / 1e6:.2f} M")
    k2.metric("Fines collected (₹ crore)", f"{bt['Total Fine Collected'].iloc[-1] / 1e7:,.1f}")
    k3.metric("Caught by cameras", f"{bt['automation_share_pct'].iloc[-1]:.0f}%")
    fig = go.Figure()
    for col, nm in [("M.V.Act Cases", "Motor Vehicles Act"), ("K.P.Act Cases", "Karnataka Police Act"), ("Towing Cases", "Towing"), ("K.T.C Act Cases", "K.T.C Act"), ("Automation cases", "Camera-detected")]:
        fig.add_trace(go.Bar(x=bt["YEAR"], y=bt[col].fillna(0), name=nm))
    fig.update_layout(barmode="stack", title="Traffic cases booked each year")
    show(fig)
    show(px.bar(bt, x="YEAR", y=bt["Total Fine Collected"] / 1e7, title="Fines collected each year (₹ crore)", labels={"y": "₹ crore", "YEAR": "Year"}))

# ------------------------------------------------------------------ mobility
with tab_mob:
    rs = T["road_stats"]
    for col, ttl in [("public_transport", "Public-transport use (%) by road"), ("pedestrian_and_cyclist_count", "Pedestrians and cyclists by road"),
                     ("parking_usage", "Roadside parking (%) by road"), ("compliance", "Signal compliance (%) by road")]:
        s = rs.sort_values(col)
        show(px.bar(s, x=col, y="road", orientation="h", title=ttl, labels={col: "", "road": ""}))

# ------------------------------------------------------------------ about
with tab_about:
    st.subheader("What the data shows")
    for line in plain_insights():
        st.write("•", line)
    st.subheader("About this system")
    st.write("This tool brings together Bengaluru's traffic records, weather, road-crash statistics, traffic-police enforcement figures and CCTV pictures to help answer: "
             "**Where is traffic heavy? What is causing it? Where should we act first?**")
    st.write("• **Predict Traffic** estimates congestion for a road under chosen conditions and gives an outlook for tomorrow.  \n"
             "• **City Map** shows every monitored junction on a map of Bengaluru, coloured by traffic.  \n"
             "• **Camera View** rates congestion from a CCTV picture, video or live camera.  \n"
             "• **Traffic Overview, Road Safety, Enforcement and Mobility** show the wider picture.")
    st.caption("Limits: based on 16 roads over about two and a half years. Traffic from one day to the next is hard to predict beyond a road's usual level, so tomorrow's outlook is guidance, not a promise. Findings describe patterns in the records, not proven causes.")
