# Bengaluru Smart Mobility Intelligence — Project Report

## 1. Problem
Use real Bengaluru traffic, weather, road-safety and enforcement data to (A) forecast next-day congestion, (B) estimate congestion under given conditions, and (C) turn the data into planning intelligence.

## 2. Data and roles
| Dataset | Rows | Role |
|---|---|---|
| Traffic (16 roads, 2022-01-01 to 2024-08-09) | 8,936 | prediction + insight |
| Weather (hourly) | 34,344 | feature source |
| Road crashes 2007-2025 | 19 | road-safety insight |
| BTP cases and fines | 15 | enforcement insight |

## 3. Data-quality and leakage audit
* The earlier "next-day" target mixed horizons: only 70% of rows were truly the next day, and 50% of values sit at the ceiling of 100.
* Interpolated 'today' rows leak the next day: the same model scores far higher with them (see the leakage demonstration) and is not real skill. They are excluded.
* `Environmental Impact` duplicates `Traffic Volume` (r = 1.0) and is dropped.

## 4. Method
Chronological split (development to 2024-03-16, test from 2024-03-17), expanding-window time-series cross-validation, random-search tuning, one-standard-error simplicity rule for model choice.

## 5. Results
**Task A — tomorrow's congestion:** final model Ridge (tuned): MAE 15.25 · RMSE 19.74 · R² 0.179. The per-road average alone scores R² 0.179; persistence scores -0.663. Alert classifier (Logistic regression): test AUC 0.718.

**Task B — what-if estimator:** final model Random Forest: MAE 2.81 · RMSE 4.20 · R² 0.969 (cross-validated R² 0.966). Reality check: traffic volume alone reaches R² 0.968 — in this dataset congestion is almost a saturating function of volume, so the other drivers add little.

## 6. Explainability
* traffic_volume: a higher value increases predicted congestion (mean |SHAP| 15.79) — more vehicles on the road raise congestion
* pedestrian_and_cyclist_count: a higher value decreases predicted congestion (mean |SHAP| 3.75) — pedestrian-heavy streets are typically lower-vehicle streets, so the link can be negative
* incident_reports: a higher value increases predicted congestion (mean |SHAP| 1.36) — breakdowns and accidents block lanes and slow the whole corridor
* incident_rate: a higher value decreases predicted congestion (mean |SHAP| 0.78) — incidents per 10k vehicles (risk normalised for exposure)
* weekend_x_vol: a higher value increases predicted congestion (mean |SHAP| 0.69)
* weekend: a higher value decreases predicted congestion (mean |SHAP| 0.13) — weekend travel differs from the weekday office commute

## 7. Mobility insights
* Highest traffic pressure: Koramangala (index 100/100) among areas; Sony World Junction (Koramangala) among roads (index 100).
* 44% of road-days sit at the congestion ceiling of 100; saturation ranges from 3% to 77% across roads — WHICH road it is explains 19% of next-day congestion variance.
* Time patterns: weekday differences are not detectable (p=0.90, range 1.1 pts); monthly not detectable (p=0.77); seasonal not detectable (p=0.50).
* Capacity utilisation and congestion move together (Spearman +0.80): congested roads are the ones that are already full.
* Incidents: within-road correlation with congestion +0.36.
* Roadwork effect on congestion vs the road's own average: -0.8 pts (95% CI -2.2 to +0.7), not detectable.
* Weather: label effect not detectable (p=0.58); measured rainfall effect not detectable (p=0.77).
* Public transport usage vs congestion: raw Spearman -0.01, within-road -0.01.
* Locations needing most attention: 100 Feet Road, Sony World Junction, Sarjapur Road.
* Road safety: 878 killed in 2025 vs peak 981 in 2007; fatality rate 17.9 killed per 100 crashes.
* Enforcement: 6.99 million cases in 2025, of which 88% automated; forecast 8.09 million next year.

## 8. Vision — CCTV vehicle detection
YOLO11n trained on BMD-45 (3000 training images, 30 epochs, 640px, T4 GPU): validation mAP50 0.557, mAP50-95 0.455. On held-out frames the PCU load estimated from detections has MAE 2.32 PCU and correlation 0.93 with the annotated load; the visual congestion index is within 10 points of the annotation-based index on average. Weakest class: Mini-bus; strongest: Three-wheeler. The index is fused into the intelligence report and served in the app's CCTV tab (image, video, live stream).

## 9. Deep-learning comparison
GRU R² 0.088, LSTM R² 0.120 vs per-road mean 0.133 on identical leak-free 7-day windows: sequence models do not find a temporal pattern the simple models miss.

## 10. Deployment
Streamlit app (`app.py`) with prediction, interpretation, a CCTV detection tab, five analysis tabs and model explainability; models in `model.pkl`; shared feature code in `mobility_features.py`.

## 11. Limitations
The CCTV index covers one camera view and is relative to annotated frames. Tomorrow's congestion is largely the road's usual level plus noise, which caps forecast R²; the what-if model is an estimator, not a forecast; congestion is capped at 100; small yearly tables; observational data (associations, not causation).
