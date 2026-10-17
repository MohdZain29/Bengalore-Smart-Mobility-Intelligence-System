# Shared prediction + intelligence-report logic (used by the notebook AND the Streamlit app)
import numpy as np, pandas as pd
import mobility_features as mf

BANDS = [(0, 50, "Light"), (50, 75, "Moderate"), (75, 90, "Heavy"), (90, 101, "Severe")]
PRIORITY_W = (0.45, 0.35, 0.20)     # congestion, incident risk, signal non-compliance

def band(x):
    return next(n for lo, hi, n in BANDS if lo <= x < hi)

def make_row(meta, road, date, ov=None):
    ov = ov or {}
    d = dict(meta["road_defaults"][road])
    d.update({k: v for k, v in ov.items() if k in meta["input_cols"]})
    if "rain_sum" in ov and "rain_sum_next" not in ov:
        d["rain_sum_next"] = ov["rain_sum"]
    return pd.DataFrame([{**d, "road": road, "date": pd.Timestamp(date)}])

def estimate_now(bundle, meta, road, date, ov=None):
    f = mf.build_features(make_row(meta, road, date, ov))
    return float(np.clip(bundle["whatif"].predict(f[bundle["feats_whatif"]])[0], 0, 100))

def forecast_tomorrow(bundle, meta, road, date, ov=None):
    f = mf.build_features(make_row(meta, road, date, ov))
    return (float(np.clip(bundle["forecast"].predict(f[bundle["feats_forecast"]])[0], 0, 100)),
            float(bundle["alert"].predict_proba(f[bundle["feats_alert"]])[0, 1]))

def intelligence_report(bundle, meta, road, date, ov=None, vision=None, vision_alpha=0.3):
    # vision: dict with visual_index (0-100), pcu, n_vehicles, counts  (from vision_utils.detect_frame / aggregated frames)
    ov = ov or {}
    exp_c, p_sat = forecast_tomorrow(bundle, meta, road, date, ov)
    nxt = pd.Timestamp(date) + pd.Timedelta(days=1)
    d = meta["road_defaults"][road]
    tmr = make_row(meta, road, nxt, {**ov, "congestion_level": exp_c, "rain_sum": ov.get("rain_sum_next", d["rain_sum_next"]), "temp_mean": d["temp_mean_next"]})
    exp_inc = float(bundle["incident"].predict(mf.build_features(tmr)[bundle["feats_incident"]])[0])
    p95 = bundle["inc_p95"]
    compliance = float(ov.get("traffic_signal_compliance", d["traffic_signal_compliance"]))
    cong_used = exp_c
    if vision is not None:
        cong_used = (1 - vision_alpha) * exp_c + vision_alpha * vision["visual_index"]
    w1, w2, w3 = PRIORITY_W
    priority = 100 * (w1 * cong_used / 100 + w2 * min(1.0, exp_inc / p95 if p95 > 0 else 0.0) + w3 * float(np.clip(1 - compliance / 100, 0, 1)))
    tier = "HIGH" if priority >= 65 else "MEDIUM" if priority >= 45 else "LOW"
    recs = []
    if cong_used >= 90 or p_sat >= 0.65:
        recs.append("Deploy peak-hour marshals / review signal timing on this corridor")
    if vision is not None and vision["visual_index"] >= 85:
        recs.append(f"Camera shows a heavy vehicle load right now (visual index {vision['visual_index']:.0f}/100): consider immediate diversion or marshal deployment")
    if exp_inc >= p95:
        recs.append("Pre-position a quick-response patrol (incident risk in the top 5% of history)")
    if compliance < 75:
        recs.append(f"Focus signal-violation enforcement (compliance {compliance:.0f}%)")
    if float(ov.get("rain_sum_next", d["rain_sum_next"])) >= 10:
        recs.append("Heavy rain forecast: waterlogging and low-visibility cautions")
    if not recs:
        recs.append("Routine monitoring")
    enf = meta["enforcement"]
    return dict(road=road, area=meta["road_area"][road], target_date=nxt.date().isoformat(), expected_congestion=exp_c, band=band(exp_c),
                p_saturated=p_sat, expected_incidents=exp_inc, fatal_exposure_index=exp_inc * meta["safety"]["fatal_share"],
                priority_score=priority, priority_tier=tier, recommendations=recs, congestion_used=cong_used,
                btp_context=dict(cases_per_day=enf["cases_per_day"], automation_share=enf["automation_share"]),
                vision=None if vision is None else {k: vision[k] for k in ("counts", "n_vehicles", "pcu", "visual_index")})

def format_report(r):
    s = [f"{r['road']} ({r['area']}) — outlook for {r['target_date']}",
         f"  Congestion forecast : {r['expected_congestion']:.0f}/100 [{r['band']}]   P(saturated) = {r['p_saturated']:.0%}",
         f"  Expected incidents  : {r['expected_incidents']:.2f}   fatal-exposure index {r['fatal_exposure_index']:.3f}",
         f"  Enforcement priority: {r['priority_score']:.0f}/100 -> {r['priority_tier']}"]
    if r["vision"]:
        s.append(f"  Camera frame        : {r['vision']['n_vehicles']:.0f} vehicles, {r['vision']['pcu']:.1f} PCU, visual index {r['vision']['visual_index']:.0f}/100")
    s.append("  Actions: " + "; ".join(r["recommendations"]))
    s.append(f"  BTP context         : ~{r['btp_context']['cases_per_day']:,.0f} cases/day citywide, {r['btp_context']['automation_share']:.0%} automated")
    return "\n".join(s)
