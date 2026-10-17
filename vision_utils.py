# YOLO vehicle detection -> PCU load -> visual congestion index (shared by the notebook AND the Streamlit app)
import json, os, time
from collections import Counter
import numpy as np

# RTSP over TCP avoids dropped packets (the usual cause of grey / smeared frames) and FFmpeg decoder chatter is silenced
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

CLASS_NAMES = ["Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler", "Two-wheeler", "LCV", "Mini-bus",
               "Tempo-traveller", "Bicycle", "Van", "Other"]
# approximate IRC-style urban passenger-car-unit factors; tune to policy
PCU = {"Hatchback": 1.0, "Sedan": 1.0, "SUV": 1.0, "MUV": 1.0, "Van": 1.0, "Other": 1.0, "Bus": 2.2, "Truck": 2.2, "Mini-bus": 2.2,
       "LCV": 1.4, "Tempo-traveller": 1.4, "Three-wheeler": 1.2, "Two-wheeler": 0.5, "Bicycle": 0.4}
PCU_BY_ID = np.array([PCU[c] for c in CLASS_NAMES], dtype=float)

def load_detector(weights):
    from ultralytics import YOLO
    return YOLO(weights)

def load_reference(path):
    return np.sort(np.array(json.load(open(path)), dtype=float))

def visual_index(pcu, ref):
    # percentile (0-100) of a frame's PCU load within the reference distribution of annotated frames
    return float(100.0 * np.searchsorted(ref, pcu, side="right") / len(ref))

def detect_frame(model, image, ref, conf=0.25, iou=0.5, imgsz=640):
    # image: file path | PIL image | numpy array in RGB order
    if isinstance(image, np.ndarray):
        image = np.ascontiguousarray(image[..., ::-1])            # ultralytics expects BGR for arrays
    r = model.predict(image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    cls = r.boxes.cls.cpu().numpy().astype(int)
    pcu = float(PCU_BY_ID[cls].sum()) if len(cls) else 0.0
    counts = {CLASS_NAMES[k]: int(v) for k, v in sorted(Counter(cls.tolist()).items())}
    return dict(counts=counts, n_vehicles=int(len(cls)), pcu=pcu, visual_index=visual_index(pcu, ref),
                annotated=np.ascontiguousarray(r.plot()[..., ::-1]))      # RGB for display

def aggregate(frames):
    # summarise several per-frame detections into one 'vision' dict for the intelligence report
    total = Counter()
    for f in frames:
        total.update(f["counts"])
    return dict(counts=dict(total), n_vehicles=float(np.mean([f["n_vehicles"] for f in frames])),
                pcu=float(np.mean([f["pcu"] for f in frames])), visual_index=float(np.mean([f["visual_index"] for f in frames])))

def video_frames(path, every_s=2.0, max_frames=24):
    # sample frames from a video file every `every_s` seconds -> list of (timestamp_s, RGB ndarray)
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * every_s)))
    out, i = [], 0
    while len(out) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            out.append((i / fps, frame[..., ::-1].copy()))
        i += 1
    cap.release()
    return out

def grab_stream_frame(url, warmup=12, max_wait_s=6.0, timeout_ms=8000):
    # grab one RGB frame from a live camera link (rtsp:// rtmp:// http(s):// MJPEG/HLS), a webcam number, or a local video file.
    # Live H.264 streams start mid-sequence: the first frames after connecting are often grey/smeared until the next full
    # (key) frame arrives, so we read for a moment and keep the first properly decoded, non-flat frame.
    import cv2
    url = str(url).strip()
    if url.isdigit():
        cap = cv2.VideoCapture(int(url))
    elif url.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")):
        try:   # force the FFmpeg backend (a '%' in a link is not mistaken for an image-sequence pattern) and fail fast
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms])
        except TypeError:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    elif os.path.exists(url):
        cap = cv2.VideoCapture(url)
    else:
        raise IOError("not a camera link: use rtsp://, http:// or https://, or a webcam number")
    if not cap.isOpened():
        raise IOError("could not open the camera link")
    frame, n, t0 = None, 0, time.time()
    while time.time() - t0 < max_wait_s:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        frame = f
        if n >= warmup and float(f.std()) > 12.0:      # decoded, not a flat grey frame
            break
    cap.release()
    if frame is None:
        raise IOError("connected, but no picture was received")
    return frame[..., ::-1].copy()

def write_clip(images, out_path, fps=1.0, size=(960, 540)):
    # build a short demo clip from still frames (PIL images)
    import cv2
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for im in images:
        vw.write(np.asarray(im.convert("RGB").resize(size))[..., ::-1])
    vw.release()