# Bengaluru Smart Mobility Intelligence

Real Bengaluru traffic, weather, road-safety and traffic-enforcement data turned into congestion forecasts, a what-if congestion estimator and planning insights.

## What it does
* **What-if estimator** (R² 0.97 on a held-out later period; traffic volume alone reaches 0.97, so it is mostly a volume-to-congestion curve): estimates congestion from volume, incidents, roadwork, public-transport usage, weather and calendar for a chosen road.
* **Tomorrow forecast + alert**: expected congestion and the probability that tomorrow saturates at 100.
* **Mobility intelligence**: pressure by area, time patterns, capacity utilisation, incidents, weather, roadwork, hotspots, road safety, enforcement.
* **CCTV detection tab**: a YOLO detector trained on BMD-45 (mAP50 0.56) finds vehicles in an uploaded image, a video clip or a live stream URL, converts the load to a PCU-weighted visual congestion index and fuses it into the road's intelligence report.
* **Streamlit app** with prediction, interpretation, charts, CCTV detection and model explainability.

## Folder structure
```
bengaluru_smart_mobility/
├── app.py                  Streamlit application
├── mobility_features.py    shared feature code (training and serving)
├── mobility_report.py      shared report logic (forecast, risk, priority, vision fusion)
├── vision_utils.py         YOLO detection, PCU load, video / stream helpers
├── yolo_bmd45.pt           trained YOLO weights (BMD-45)
├── model.pkl               saved models (what-if estimator, forecast, alert, incident risk)
├── scaler.pkl              scaler for the numeric driver inputs
├── meta.json               metrics, importances, defaults, insights
├── roads.json
├── requirements.txt
├── data/                   cleaned data for the dashboard + original CSVs
├── models/  figures/  docs/  notebooks/
└── dashboard.html          standalone interactive dashboard
```

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (free)
The CCTV tab needs PyTorch + OpenCV, so prefer a host with enough memory:
1. **Hugging Face Spaces** (recommended): create a Space with SDK *Streamlit* and upload this folder (the 2 GB-RAM CPU tier is enough for the small YOLO model).
2. **Streamlit Community Cloud**: push this folder to GitHub and set the main file to `app.py` (memory is tighter; the app still works because the detector loads lazily, and the other tabs do not need PyTorch).
3. Live RTSP cameras must be reachable from the host (a public HTTP/MJPEG URL, or run the app on a machine inside the camera network).

## Honest limitations
Tomorrow's congestion on a road is mostly its usual level plus day-to-day noise (road average alone gives R² 0.18); the high-R² model is a what-if *estimator*, not a forecast. The CCTV index describes one camera view, is relative to annotated BMD-45 frames and is not geo-linked to the traffic table (the user pairs a camera with a road). Data covers 16 roads and about 2.5 years; congestion is capped at 100; the yearly crash and enforcement tables are small and their relationships are descriptive, not causal.
