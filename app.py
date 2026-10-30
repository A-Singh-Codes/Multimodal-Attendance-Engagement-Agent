import io
import json
import math
import os
import sqlite3
import tempfile
from datetime import datetime, date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import librosa

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "classpulse.db"

st.set_page_config(
    page_title="ClassPulse AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------- UI theme --------------------------
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #080d18; color: #e8eefb; }
[data-testid="stHeader"] { background: rgba(8,13,24,.92); }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#0b1427 0%,#101b31 100%);
  border-right: 1px solid #203252;
}
[data-testid="stSidebar"] > div { padding-top: 1.15rem; }
.block-container { max-width: 1500px; padding-top: 1.2rem; padding-bottom: 3rem; }
h1,h2,h3 { color: #f4f7ff !important; letter-spacing: -.025em; }
p, label, .stMarkdown { color: #d0daed; }
.hero {
  padding: 1.45rem 1.65rem; border-radius: 22px; margin-bottom: 1rem;
  background: radial-gradient(circle at 85% 10%,rgba(77,142,255,.42),transparent 34%),
  linear-gradient(120deg,#11244a 0%,#173d82 55%,#1d58b8 100%);
  border: 1px solid rgba(118,166,255,.28);
}
.hero h1 { margin: .25rem 0 .35rem 0; font-size: 2.15rem; color: white !important; }
.hero p { margin: 0; color: #dbe8ff; }
.eyebrow { color:#8fb9ff; text-transform:uppercase; font-size:.72rem; font-weight:800; letter-spacing:.14em; }
.panel, .metric-card {
  background: linear-gradient(145deg,#101a2d,#0d1627);
  border: 1px solid #233858; border-radius: 16px; padding: 1rem 1.1rem;
}
.metric-card { min-height: 112px; }
.metric-label { color:#93a8c9; font-size:.78rem; margin-bottom:.4rem; letter-spacing:.05em; }
.metric-value { color:#f5f8ff; font-size:1.7rem; font-weight:800; line-height:1.15; }
.metric-hint { color:#77a7ff; font-size:.75rem; margin-top:.35rem; }
.pill { display:inline-block; padding:.25rem .65rem; border-radius:999px; background:#122b4c; border:1px solid #2d568d; color:#a8ccff; font-size:.75rem; font-weight:700; }
div.stButton > button, div.stDownloadButton > button, div.stFormSubmitButton > button {
  border-radius: 10px; border: 1px solid #2d69c8; font-weight:700;
  background: linear-gradient(100deg,#1d4ed8,#2879ee); color:white; min-height:2.55rem;
}
div.stButton > button:hover, div.stDownloadButton > button:hover { border-color:#8ab7ff; color:white; filter:brightness(1.1); }
.stTextInput input, .stTextArea textarea, .stDateInput input, [data-baseweb="select"] > div {
  background:#0d1729 !important; color:#e8eefb !important; border-color:#2a3c5b !important;
}
[data-testid="stFileUploader"] { background:#0d1729; border:1px dashed #36517a; border-radius:14px; padding:.5rem; }
[data-testid="stDataFrame"] { border:1px solid #253957; border-radius:12px; overflow:hidden; }
div[data-testid="stAlert"] { border-radius:12px; }
hr { border-color:#233452; }
.muted { color:#91a3c1; font-size:.85rem; }
.brand { font-size:1.25rem; font-weight:850; color:#f3f7ff; }
.sidebar-sub { color:#8fa8d0; font-size:.76rem; margin-top:.1rem; }
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, hint=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-hint">{hint}</div></div>',
        unsafe_allow_html=True,
    )


# -------------------------- Local database --------------------------
def db_connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db_connect() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            class_name TEXT NOT NULL,
            session_date TEXT NOT NULL,
            video_name TEXT,
            audio_name TEXT,
            report_json TEXT NOT NULL
        )
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS attendance_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            track_id TEXT NOT NULL,
            first_seen REAL NOT NULL,
            last_seen REAL NOT NULL,
            observed_seconds REAL NOT NULL,
            detection_count INTEGER NOT NULL,
            status TEXT NOT NULL,
            student_name TEXT DEFAULT '',
            roll_number TEXT DEFAULT '',
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
        """)


init_db()


def save_session_to_db(result):
    report = export_report(result)
    with db_connect() as con:
        cur = con.execute(
            "INSERT INTO sessions(created_at,class_name,session_date,video_name,audio_name,report_json) VALUES(?,?,?,?,?,?)",
            (
                datetime.now().isoformat(timespec="seconds"),
                result.get("class_name", "Class"),
                result.get("session_date", str(date.today())),
                result.get("video_name"),
                result.get("audio_name"),
                json.dumps(report, ensure_ascii=False),
            ),
        )
        session_id = cur.lastrowid
        for track in result.get("tracks", []):
            con.execute(
                """INSERT INTO attendance_reviews(session_id,track_id,first_seen,last_seen,observed_seconds,
                   detection_count,status,student_name,roll_number) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    session_id, track["Track ID"], track["First seen (s)"], track["Last seen (s)"],
                    track.get("Observed presence estimate (s)", 0.0), track["Detections"], track["Attendance signal"],
                    track.get("Student name", ""), track.get("Roll number", "")
                ),
            )
    return session_id


def get_session_history(limit=30):
    with db_connect() as con:
        rows = con.execute(
            "SELECT id,created_at,class_name,session_date,video_name,audio_name FROM sessions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def load_session_report(session_id):
    with db_connect() as con:
        row = con.execute("SELECT report_json FROM sessions WHERE id=?", (int(session_id),)).fetchone()
    return json.loads(row["report_json"]) if row else None


# -------------------------- Upload/session state --------------------------
def named_buffer(payload, name):
    buffer = io.BytesIO(payload)
    buffer.name = name
    return buffer


def save_video_upload():
    f = st.session_state.get("video_upload_widget")
    if f is not None:
        st.session_state["saved_video"] = {"name": f.name, "bytes": f.getvalue()}


def save_audio_upload():
    f = st.session_state.get("audio_upload_widget")
    if f is not None:
        st.session_state["saved_audio"] = {"name": f.name, "bytes": f.getvalue()}


def clear_video():
    st.session_state.pop("saved_video", None)
    st.session_state.pop("video_upload_widget", None)


def clear_audio():
    st.session_state.pop("saved_audio", None)
    st.session_state.pop("audio_upload_widget", None)


def roster_parse(text):
    output = []
    seen = set()
    for line in text.splitlines():
        if "," not in line:
            continue
        roll, name = [x.strip() for x in line.split(",", 1)]
        if roll and name and roll not in seen:
            output.append({"Roll number": roll, "Student name": name})
            seen.add(roll)
    return output


# -------------------------- Video: face detection + anonymous tracks --------------------------
def analyze_video(upload, sample_seconds=1.0, max_samples=300, min_presence_ratio=0.2, min_detections=3):
    suffix = Path(upload.name).suffix or ".mp4"
    tmp_path = None
    cap = None
    samples = []
    tracks = {}
    preview = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload.getvalue())
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return None, [], [], None, "Could not open this video. Try MP4 or MOV."
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        interval = max(1, int(fps * sample_seconds))
        if not hasattr(cv2, "CascadeClassifier"):
            return None, [], [], None, "This OpenCV build does not support Haar face detection. Install opencv-python-headless==4.11.0.86."
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            return None, [], [], None, "Could not load the OpenCV face detector."
        index, sample_number = 0, 0
        previous_centres = {}
        max_track_gap = max(2.5, sample_seconds * 2.5)
        while sample_number < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if index % interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Histogram equalization improves contrast for uneven classroom lighting.
                gray = cv2.equalizeHist(gray)
                detections = detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                timestamp = index / fps if fps > 0 else 0.0
                h_img, w_img = frame.shape[:2]
                centres = []
                for (x, y, w, h) in detections:
                    centres.append({
                        "bbox": (int(x), int(y), int(w), int(h)),
                        "centre": (float(x + w / 2), float(y + h / 2)),
                    })
                # Greedy nearest-centre association. Tracks are anonymous and may fragment.
                unmatched = set(tracks.keys())
                assignments = []
                for det in centres:
                    cx, cy = det["centre"]
                    candidates = []
                    for tid in unmatched:
                        t = tracks[tid]
                        if timestamp - t["last_seen"] > max_track_gap:
                            continue
                        px, py = t["last_centre"]
                        dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                        max_dist = max(70.0, 0.18 * math.sqrt(w_img * w_img + h_img * h_img))
                        if dist <= max_dist:
                            candidates.append((dist, tid))
                    if candidates:
                        _, chosen = min(candidates)
                        unmatched.remove(chosen)
                    else:
                        chosen = f"T{len(tracks) + 1:02d}"
                        tracks[chosen] = {
                            "track_id": chosen, "first_seen": timestamp, "last_seen": timestamp,
                            "detections": 0, "seen_times": [], "centres": [], "last_centre": (cx, cy),
                            "movements": [], "max_face_area_ratio": 0.0
                        }
                    t = tracks[chosen]
                    if t["centres"]:
                        px, py = t["last_centre"]
                        movement = math.sqrt((cx - px) ** 2 + (cy - py) ** 2) / max(math.sqrt(w_img*w_img + h_img*h_img), 1)
                        t["movements"].append(float(movement))
                    t["last_seen"] = timestamp
                    t["last_centre"] = (cx, cy)
                    t["detections"] += 1
                    t["seen_times"].append(timestamp)
                    t["centres"].append((cx, cy))
                    x, y, w, h = det["bbox"]
                    t["max_face_area_ratio"] = max(t["max_face_area_ratio"], (w*h)/max(w_img*h_img, 1))
                    assignments.append((chosen, det["bbox"]))
                samples.append({
                    "Time (seconds)": round(timestamp, 2),
                    "Faces detected": int(len(detections)),
                    "Face visible": bool(len(detections) > 0),
                    "Anonymous track IDs": ", ".join(tid for tid, _ in assignments),
                })
                if preview is None:
                    marked = frame.copy()
                    for tid, (x, y, w, h) in assignments:
                        cv2.rectangle(marked, (x, y), (x+w, y+h), (255, 170, 35), 2)
                        cv2.putText(marked, tid, (x, max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,170,35), 2)
                    preview = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)
                sample_number += 1
            index += 1
        if not samples:
            return None, [], [], None, "No video frames could be sampled."
        rows = []
        # Estimate presence from the time span and sampling coverage of each anonymous track,
        # not from detections divided by the entire video duration. A student may arrive late
        # or leave early, so whole-video ratio is a misleading attendance test.
        for tid, t in tracks.items():
            count = t["detections"]
            span = max(0.0, t["last_seen"] - t["first_seen"])
            observed = min(duration, span + sample_seconds) if count else 0.0
            expected_samples = max(1, int(round(span / max(sample_seconds, 0.001))) + 1)
            coverage_ratio = min(1.0, count / expected_samples)
            min_span = max(sample_seconds * (min_detections - 1), 1.0)
            attendance_signal = (
                "Sustained presence signal"
                if count >= min_detections and span >= min_span and coverage_ratio >= min_presence_ratio
                else "Brief/uncertain signal"
            )
            movements = t["movements"]
            avg_move = float(np.mean(movements)) if movements else 0.0
            if avg_move < 0.012:
                movement_label = "Low detected movement"
            elif avg_move < 0.04:
                movement_label = "Moderate detected movement"
            else:
                movement_label = "Higher detected movement"
            rows.append({
                "Track ID": tid,
                "First seen (s)": round(t["first_seen"], 2),
                "Last seen (s)": round(t["last_seen"], 2),
                "Observed presence estimate (s)": round(observed, 1),
                "Track span (s)": round(span, 1),
                "Detection coverage (%)": round(100 * coverage_ratio, 1),
                "Detections": count,
                "Movement index": round(avg_move, 4),
                "Movement pattern": movement_label,
                "Attendance signal": attendance_signal,
                "Student name": "",
                "Roll number": "",
            })
        summary = {
            "duration": round(duration, 2),
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "samples": len(samples),
            "track_count": len(tracks),
            "face_visible_percent": round(100 * sum(x["Face visible"] for x in samples) / len(samples), 1),
            "min_presence_ratio": min_presence_ratio,
            "min_detections": min_detections,
            "sample_seconds": sample_seconds,
        }
        return summary, samples, rows, preview, None
    except Exception as exc:
        return None, [], [], None, f"Video analysis failed: {exc}"
    finally:
        if cap is not None:
            cap.release()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# -------------------------- Audio: energy / quiet segments --------------------------
def analyze_audio(upload, chunk_seconds=0.5):
    try:
        raw = io.BytesIO(upload.getvalue())
        y, sr = librosa.load(raw, sr=None, mono=True)
        if len(y) == 0 or sr <= 0:
            return None, None, "No audio data was found."
        hop = max(512, int(sr * chunk_seconds))
        frame_length = min(2048, max(512, 2**int(math.log2(hop))))
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
        # Use robust quantile threshold; report it transparently.
        threshold = max(float(np.percentile(rms, 35)), 0.008)
        active = rms > threshold
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=hop)[0][:len(rms)]
        table = pd.DataFrame({
            "Time (seconds)": np.round(times, 2),
            "RMS energy": np.round(rms, 5),
            "Zero-crossing rate": np.round(zcr, 4),
            "Audio state": np.where(active, "Activity", "Quiet"),
        })
        # Segments are consecutive states, merged into intervals.
        segments = []
        start = 0
        state = bool(active[0])
        for i in range(1, len(active)):
            if bool(active[i]) != state:
                segments.append({
                    "Start (s)": round(float(times[start]), 2),
                    "End (s)": round(float(times[i-1] + chunk_seconds), 2),
                    "State": "Activity" if state else "Quiet",
                })
                start, state = i, bool(active[i])
        segments.append({
            "Start (s)": round(float(times[start]), 2),
            "End (s)": round(float(times[-1] + chunk_seconds), 2),
            "State": "Activity" if state else "Quiet",
        })
        summary = {
            "duration": round(len(y)/sr, 2),
            "sample_rate": int(sr),
            "active_percent": round(float(np.mean(active)*100), 1),
            "mean_rms": round(float(np.mean(rms)), 5),
            "threshold": round(threshold, 5),
            "chunk_seconds": chunk_seconds,
            "activity_segments": int(sum(1 for s in segments if s["State"] == "Activity")),
            "quiet_segments": int(sum(1 for s in segments if s["State"] == "Quiet")),
        }
        return summary, table, segments, None
    except Exception as exc:
        return None, None, None, f"Audio analysis failed: {exc}. WAV is the most reliable format."


# -------------------------- Offline recommendation engine --------------------------
def build_insights(result):
    insights = []
    vs = result.get("video_summary")
    tracks = result.get("tracks", [])
    audio = result.get("audio_summary")
    v_samples = result.get("video_samples", [])
    a_table = result.get("audio_table")
    if vs:
        if vs["face_visible_percent"] < 35:
            insights.append({
                "priority": "Review",
                "title": "Check camera coverage",
                "why": f"Faces were detected in {vs['face_visible_percent']:.1f}% of sampled frames. Camera angle, distance, lighting or occlusion may explain this.",
                "action": "Move the camera to a stable, wide view and repeat a short test. Do not interpret missing faces as absence without teacher review."
            })
        else:
            insights.append({
                "priority": "Observation",
                "title": "Review visibility dips",
                "why": f"Faces were visible in {vs['face_visible_percent']:.1f}% of sampled frames.",
                "action": "Open the video timeline and check intervals with zero detections before using the attendance signal."
            })
        if tracks:
            sustained = sum(1 for t in tracks if t["Attendance signal"] == "Sustained presence signal")
            insights.append({
                "priority": "Review",
                "title": "Verify anonymous attendance tracks",
                "why": f"{sustained} of {len(tracks)} anonymous track(s) passed the configured sustained-presence rule.",
                "action": "Review the preview and track timeline. Map tracks to names only when a teacher can verify the evidence."
            })
        movement_vals = [t["Movement index"] for t in tracks]
        if movement_vals and max(movement_vals) < .012:
            insights.append({
                "priority": "Try next session",
                "title": "Add a quick participation check",
                "why": "Detected face-centre movement was low in this clip. Low movement is not proof of inattention.",
                "action": "Ask a short question, use a one-minute pair discussion, or run a quick knowledge check to directly assess understanding."
            })
    if audio:
        if audio["active_percent"] < 20:
            insights.append({
                "priority": "Try next session",
                "title": "Consider a short discussion prompt",
                "why": f"Only {audio['active_percent']:.1f}% of audio chunks crossed the sound-activity threshold.",
                "action": "If this was intended to be interactive, add a question, think-pair-share, or learner reflection. Quiet lecture segments may be completely appropriate."
            })
        elif audio["active_percent"] > 80:
            insights.append({
                "priority": "Review",
                "title": "Review high sound activity",
                "why": f"{audio['active_percent']:.1f}% of audio chunks crossed the sound-activity threshold; this measures energy, not speech quality.",
                "action": "Check whether the activity came from discussion, background noise, or the instructor. Use clear turn-taking if needed."
            })
        else:
            insights.append({
                "priority": "Observation",
                "title": "Review activity and quiet intervals",
                "why": f"Audio activity crossed the threshold in {audio['active_percent']:.1f}% of chunks.",
                "action": "Compare active intervals with the class plan to see where questions or discussion may have occurred."
            })
    if vs and audio:
        insights.append({
            "priority": "Observation",
            "title": "Compare the two timelines",
            "why": "Video visibility/movement and audio energy describe different signals and can disagree.",
            "action": "Use the combined timeline to identify periods worth reviewing. Treat this as a session activity view, not an individual engagement score."
        })
    insights.append({
        "priority": "Ethical use",
        "title": "Keep teacher review in the loop",
        "why": "Anonymous face tracks can split or merge when people move, overlap, or leave the frame.",
        "action": "Do not automatically penalize or grade a learner from this prototype. Confirm attendance using a teacher or a consented, verified identification method."
    })
    return insights


def export_report(result):
    return {
        "project": "ClassPulse AI — Multimodal Attendance & Session Activity",
        "created_at": result.get("created_at", datetime.now().isoformat(timespec="seconds")),
        "class": result.get("class_name", ""),
        "session_date": result.get("session_date", ""),
        "video_file": result.get("video_name"),
        "audio_file": result.get("audio_name"),
        "video_summary": result.get("video_summary"),
        "video_samples": result.get("video_samples", []),
        "anonymous_tracks_and_attendance_signals": result.get("tracks", []),
        "audio_summary": result.get("audio_summary"),
        "audio_samples": result.get("audio_table").to_dict(orient="records") if isinstance(result.get("audio_table"), pd.DataFrame) else [],
        "audio_segments": result.get("audio_segments", []),
        "recommendations": build_insights(result),
        "interpretation_limits": [
            "Face detection is not named identity recognition.",
            "Anonymous tracking may fragment or merge tracks.",
            "Head pose is not estimated in this version.",
            "Audio energy is not speech transcription or speaker identification.",
            "Activity indicators are not validated attention, learning, or emotion measures.",
            "Attendance signals require teacher verification.",
        ],
    }


# -------------------------- Navigation --------------------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "roster_text" not in st.session_state:
    st.session_state.roster_text = "CS01, Student One\nCS02, Student Two\nCS03, Student Three"
if "nav_target" not in st.session_state:
    st.session_state.nav_target = "Overview"

with st.sidebar:
    st.markdown('<div class="brand">🎓 ClassPulse <span style="color:#72a8ff">AI</span></div><div class="sidebar-sub">OFFLINE CLASSROOM WORKSPACE</div>', unsafe_allow_html=True)
    st.markdown("---")
    pages = ["Overview", "Analyze session", "Attendance review", "Activity analytics", "Teaching suggestions", "Session history & exports"]
    default_index = pages.index(st.session_state.nav_target) if st.session_state.nav_target in pages else 0
    page = st.radio("WORKSPACE", pages, index=default_index, key="page_navigation")
    st.session_state.nav_target = page
    st.markdown("---")
    st.markdown('<div class="eyebrow">SESSION DETAILS</div>', unsafe_allow_html=True)
    class_name = st.text_input("Class / subject", value=st.session_state.get("class_name", "Computer Science"), key="class_name_input")
    session_date = st.date_input("Session date", value=st.session_state.get("session_date", date.today()), key="session_date_input")
    st.session_state["class_name"] = class_name
    st.session_state["session_date"] = session_date
    st.caption("Media is analyzed locally. Only summaries and review records are saved to a local SQLite database.")
    st.markdown("---")
    st.markdown('<span class="pill">● Local-first</span>', unsafe_allow_html=True)

analysis = st.session_state.analysis
has_analysis = bool(analysis and (analysis.get("video_summary") or analysis.get("audio_summary")))
class_display = analysis.get("class_name", class_name) if analysis else class_name
date_display = analysis.get("session_date", str(session_date)) if analysis else str(session_date)

st.markdown("""
<div class="hero">
  <div class="eyebrow">MULTIMODAL CLASSROOM ANALYTICS</div>
  <h1>ClassPulse AI</h1>
  <p>Track anonymous visual presence, inspect audio activity, and turn session signals into practical teaching actions.</p>
</div>
""", unsafe_allow_html=True)


# -------------------------- Pages --------------------------
if page == "Overview":
    st.markdown(f'<span class="pill">● {class_display}</span> &nbsp; <span class="pill">📅 {date_display}</span>', unsafe_allow_html=True)
    st.markdown("## Session overview")
    st.caption("Run a local analysis, then review evidence and export a session report. No cloud AI or external API is used.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("ROSTER", str(len(roster_parse(st.session_state.roster_text))), "Students listed")
    with col2:
        metric_card("VIDEO DURATION", f"{analysis['video_summary']['duration']:.1f}s" if has_analysis and analysis.get("video_summary") else "—", "Recorded clip")
    with col3:
        metric_card("ANONYMOUS TRACKS", str(len(analysis.get("tracks", []))) if has_analysis and analysis.get("video_summary") else "—", "Not named identities")
    with col4:
        metric_card("AUDIO ACTIVITY", f"{analysis['audio_summary']['active_percent']:.1f}%" if has_analysis and analysis.get("audio_summary") else "—", "Sound-energy proxy")
    st.markdown("")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown('<div class="panel"><div class="eyebrow">NEXT STEP</div><h3>Analyze a session</h3><p>Upload a short video and/or audio file. Set the sustained-presence rule, run analysis, then review the evidence before confirming attendance.</p></div>', unsafe_allow_html=True)
        if st.button("＋ Analyze a session", use_container_width=True):
            st.session_state.nav_target = "Analyze session"
            st.rerun()
    with right:
        st.markdown('<div class="panel"><div class="eyebrow">PROJECT PRINCIPLE</div><h3>Privacy by design</h3><p>Media is processed locally. Face tracks are anonymous; no facial identity matching, emotion recognition, or cloud model is used.</p></div>', unsafe_allow_html=True)
    if has_analysis:
        st.success(f"Latest analysis: {analysis.get('video_name') or 'No video'} / {analysis.get('audio_name') or 'No audio'}")
        if analysis.get("session_db_id"):
            st.caption(f"Saved local session record #{analysis['session_db_id']}.")
    else:
        st.info("No analysis yet. Start with a 10–30 second sample for a quick check.")

elif page == "Analyze session":
    st.markdown("## Analyze a classroom session")
    st.caption("Supported video: MP4, MOV, AVI, MKV. Audio: WAV, MP3, M4A, OGG. The uploaded media stays in this running app session.")
    with st.expander("Student roster (used for manual review only)", expanded=False):
        roster_text = st.text_area("One student per line: Roll number, Student name", key="roster_text", height=115)
        st.caption(f"{len(roster_parse(roster_text))} roster entries. This app does not match faces to student names.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel"><h3>🎥 Classroom video</h3><p class="muted">Detect visible faces and associate detections into anonymous tracks.</p></div>', unsafe_allow_html=True)
        st.file_uploader("Choose video", type=["mp4", "mov", "avi", "mkv"], key="video_upload_widget", on_change=save_video_upload)
        sv = st.session_state.get("saved_video")
        if sv:
            st.success(f"Ready: {sv['name']} · {len(sv['bytes'])/1024/1024:.1f} MB")
            st.button("Remove video", on_click=clear_video, key="remove_video_btn")
        else:
            st.caption("No video selected.")
    with c2:
        st.markdown('<div class="panel"><h3>🎙️ Classroom audio</h3><p class="muted">Measure sound energy and identify quiet/activity intervals.</p></div>', unsafe_allow_html=True)
        st.file_uploader("Choose audio", type=["wav", "mp3", "m4a", "ogg"], key="audio_upload_widget", on_change=save_audio_upload)
        sa = st.session_state.get("saved_audio")
        if sa:
            st.success(f"Ready: {sa['name']} · {len(sa['bytes'])/1024/1024:.1f} MB")
            st.button("Remove audio", on_click=clear_audio, key="remove_audio_btn")
        else:
            st.caption("No audio selected.")
    with st.expander("Analysis controls", expanded=True):
        sample_seconds = st.select_slider("Sample video every", options=[0.5, 1.0, 1.5, 2.0, 3.0], value=1.0, format_func=lambda x: f"{x:g} seconds")
        max_samples = st.slider("Maximum video samples", 30, 600, 180, step=30)
        presence_ratio = st.slider("Minimum track detection coverage", 0.1, 0.8, 0.4, step=0.1, help="Minimum share of expected samples covered within an anonymous track's own time span. Presence remains an estimate and requires teacher verification.")
        min_detections = st.slider("Minimum detections per track", 2, 10, 3)
    if st.button("▶ Analyze uploaded media", type="primary", use_container_width=True):
        saved_video = st.session_state.get("saved_video")
        saved_audio = st.session_state.get("saved_audio")
        if not saved_video and not saved_audio:
            st.warning("Upload a video, an audio file, or both.")
        else:
            result = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "class_name": class_name,
                "session_date": str(session_date),
                "video_name": saved_video["name"] if saved_video else None,
                "audio_name": saved_audio["name"] if saved_audio else None,
                "video_summary": None, "video_samples": [], "tracks": [],
                "video_preview": None, "audio_summary": None, "audio_table": None,
                "audio_segments": [], "roster": roster_parse(st.session_state.roster_text),
            }
            if saved_video:
                with st.spinner("Sampling video and tracking anonymous face detections..."):
                    vsummary, vsamples, tracks, preview, error = analyze_video(
                        named_buffer(saved_video["bytes"], saved_video["name"]), sample_seconds, max_samples, presence_ratio, min_detections
                    )
                if error:
                    st.error(error)
                else:
                    result["video_summary"], result["video_samples"], result["tracks"], result["video_preview"] = vsummary, vsamples, tracks, preview
            if saved_audio:
                with st.spinner("Measuring audio energy and building activity segments..."):
                    asum, atable, segments, error = analyze_audio(named_buffer(saved_audio["bytes"], saved_audio["name"]))
                if error:
                    st.error(error)
                else:
                    result["audio_summary"], result["audio_table"], result["audio_segments"] = asum, atable, segments
            if result["video_summary"] or result["audio_summary"]:
                st.session_state.analysis = result
                try:
                    db_id = save_session_to_db(result)
                    st.session_state.analysis["session_db_id"] = db_id
                except Exception as exc:
                    st.warning(f"Analysis finished, but the local history record could not be saved: {exc}")
                st.success("Analysis complete. Open Attendance review, Activity analytics, or Teaching suggestions.")
                st.rerun()

elif page == "Attendance review":
    st.markdown("## Attendance review")
    st.caption("Presence candidates use repeated detections, track duration, and detection coverage within that track's own time span. Track IDs are anonymous and require teacher verification.")
    if not has_analysis or not analysis.get("video_summary"):
        st.info("Run a video analysis first. Audio-only sessions cannot create visual attendance signals.")
    else:
        tracks = analysis.get("tracks", [])
        if not tracks:
            st.warning("No face tracks were detected. Check the video angle, lighting, or clip format.")
        else:
            st.markdown("### Detected anonymous tracks")
            st.dataframe(pd.DataFrame(tracks), use_container_width=True, hide_index=True)
            st.markdown("### Teacher verification")
            roster = roster_parse(st.session_state.roster_text)
            roster_names = ["Unassigned"] + [f"{r['Roll number']} — {r['Student name']}" for r in roster]
            edited = pd.DataFrame(tracks)
            if "Teacher decision" not in edited.columns:
                edited["Teacher decision"] = edited["Attendance signal"].map(
                    lambda x: "Review needed" if x == "Brief/uncertain signal" else "Confirm present"
                )
            edited["Teacher decision"] = edited["Teacher decision"].astype(str)
            edited["Student mapping"] = edited.apply(
                lambda row: f"{row.get('Roll number','')} — {row.get('Student name','')}".strip(" —") or "Unassigned", axis=1
            )
            config = {
                "Teacher decision": st.column_config.SelectboxColumn("Teacher decision", options=["Review needed", "Confirm present", "Confirm absent"]),
                "Student mapping": st.column_config.SelectboxColumn("Student mapping", options=roster_names),
            }
            cols = ["Track ID", "First seen (s)", "Last seen (s)", "Observed presence estimate (s)", "Track span (s)", "Detection coverage (%)", "Detections", "Attendance signal", "Teacher decision", "Student mapping"]
            edited_df = st.data_editor(edited[cols], column_config=config, disabled=cols[:6], hide_index=True, use_container_width=True, key="attendance_editor")
            if st.button("Save teacher review", type="primary"):
                track_updates = edited_df.to_dict(orient="records")
                mapping = {x["Track ID"]: x for x in track_updates}
                for tr in analysis["tracks"]:
                    update = mapping.get(tr["Track ID"], {})
                    tr["Teacher decision"] = update.get("Teacher decision", "Review needed")
                    selected = update.get("Student mapping", "Unassigned")
                    if selected != "Unassigned" and " — " in selected:
                        roll, name = selected.split(" — ", 1)
                        tr["Roll number"], tr["Student name"] = roll, name
                    else:
                        tr["Roll number"], tr["Student name"] = "", ""
                st.session_state.analysis = analysis
                st.success("Teacher review saved for the current session.")
                st.rerun()
            export_att = pd.DataFrame(analysis["tracks"])
            st.download_button("Download attendance evidence CSV", export_att.to_csv(index=False).encode("utf-8"), "attendance-evidence.csv", "text/csv")
            st.warning("Anonymous tracks can split or merge. Confirm identities visually before mapping tracks to student names.")

elif page == "Activity analytics":
    st.markdown("## Activity analytics")
    st.caption("This page shows measurable activity signals—not a validated measure of attention, learning, or emotion.")
    if not has_analysis:
        st.info("Upload and analyze media first.")
    else:
        vs = analysis.get("video_summary")
        aus = analysis.get("audio_summary")
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("VIDEO SAMPLES", str(vs["samples"]) if vs else "—", "Sampled timestamps")
        with c2:
            metric_card("FACE-VISIBLE SAMPLES", f"{vs['face_visible_percent']:.1f}%" if vs else "—", "Visibility proxy")
        with c3:
            metric_card("AUDIO ACTIVITY", f"{aus['active_percent']:.1f}%" if aus else "—", "RMS threshold proxy")
        if vs:
            vdf = pd.DataFrame(analysis["video_samples"])
            st.markdown("### Visible faces over time")
            st.line_chart(vdf.set_index("Time (seconds)")["Faces detected"], height=240)
            if analysis.get("video_preview") is not None:
                with st.expander("Review first sampled frame"):
                    st.image(analysis["video_preview"], caption="Detected face boxes and anonymous track IDs. This is not identity recognition.", use_container_width=True)
            with st.expander("Inspect video sample records"):
                st.dataframe(vdf, use_container_width=True, hide_index=True)
            tdf = pd.DataFrame(analysis.get("tracks", []))
            if not tdf.empty:
                st.markdown("### Movement estimates by anonymous track")
                st.bar_chart(tdf.set_index("Track ID")["Movement index"], height=220)
                st.caption("Movement index = average face-centre displacement normalized by frame diagonal between detections. Camera motion can affect it.")
        if aus and analysis.get("audio_table") is not None:
            adf = analysis["audio_table"]
            st.markdown("### Audio energy over time")
            st.area_chart(adf.set_index("Time (seconds)")["RMS energy"], height=230)
            st.caption(f"Threshold used: {aus['threshold']:.5f} RMS · Activity chunks: {aus['activity_segments']} · Quiet chunks: {aus['quiet_segments']}")
            if analysis.get("audio_segments"):
                st.markdown("### Audio intervals")
                st.dataframe(pd.DataFrame(analysis["audio_segments"]), use_container_width=True, hide_index=True)
            with st.expander("Inspect audio sample records"):
                st.dataframe(adf, use_container_width=True, hide_index=True)
        if vs and aus and analysis.get("audio_table") is not None:
            st.markdown("### Combined session timeline")
            vdf = pd.DataFrame(analysis["video_samples"])
            vseries = vdf.set_index("Time (seconds)")["Faces detected"].rename("Visible faces")
            at = analysis["audio_table"].copy()
            # Normalize each series to 0..1 for visual comparison only.
            vnorm = vseries / max(float(vseries.max()), 1.0)
            anorm = at.set_index("Time (seconds)")["RMS energy"]
            anorm = anorm / max(float(anorm.max()), 1e-9)
            combo = pd.concat([vnorm.rename("Normalized face count"), anorm.rename("Normalized audio energy")], axis=1).sort_index().interpolate(limit_direction="both").fillna(0)
            st.line_chart(combo, height=260)
            st.caption("Both lines are normalized to 0–1 for comparison. This is not a fused engagement score.")

elif page == "Teaching suggestions":
    st.markdown("## Teaching suggestions")
    st.caption("An offline, explainable rules engine turns session signals into review prompts. It does not use a trained generative AI model.")
    if not has_analysis:
        st.info("Run an analysis first.")
    else:
        for item in build_insights(analysis):
            with st.container(border=True):
                st.markdown(f"**{item['title']}**  ·  `{item['priority']}`")
                st.write(item["why"])
                st.markdown(f"**Suggested action:** {item['action']}")
        st.markdown("### Teacher reflection")
        reflection = st.text_area("Notes for the next session", value=analysis.get("teacher_reflection", ""), placeholder="What went well? What should change next time?")
        if st.button("Save reflection"):
            analysis["teacher_reflection"] = reflection
            st.session_state.analysis = analysis
            st.success("Reflection saved in this running session.")
        st.caption("Suggestions are prompts for a teacher to consider, not automatic decisions about learners.")

elif page == "Session history & exports":
    st.markdown("## Session history & exports")
    st.caption("Session summaries and reports are stored in a local SQLite database in this project folder. Uploaded media itself is not stored in the database.")
    history = get_session_history()
    if history:
        hdf = pd.DataFrame(history)
        st.dataframe(hdf.rename(columns={
            "id": "Session ID", "created_at": "Analyzed at", "class_name": "Class",
            "session_date": "Session date", "video_name": "Video file", "audio_name": "Audio file"
        }), use_container_width=True, hide_index=True)
        ids = [row["id"] for row in history]
        selected_id = st.selectbox("Choose a saved report", ids, format_func=lambda x: f"Session #{x} — {next((r['class_name'] for r in history if r['id']==x), 'Class')} ({next((r['session_date'] for r in history if r['id']==x), '')})")
        saved_report = load_session_report(selected_id)
        st.download_button("Download selected report (JSON)", json.dumps(saved_report, indent=2, ensure_ascii=False), file_name=f"classpulse-session-{selected_id}.json", mime="application/json", use_container_width=True)
    else:
        st.info("No saved sessions yet. Run an analysis and it will be saved locally.")
    if has_analysis:
        st.markdown("### Current session report")
        report = export_report(analysis)
        st.download_button("Download current session report (JSON)", json.dumps(report, indent=2, ensure_ascii=False), file_name="classpulse-session-report.json", mime="application/json")
        if analysis.get("tracks"):
            st.download_button("Download current attendance CSV", pd.DataFrame(analysis["tracks"]).to_csv(index=False).encode("utf-8"), file_name="classpulse-attendance.csv", mime="text/csv")
    st.markdown("### Local project status")
    st.write("**Implemented:** local video face detection, approximate anonymous tracking, sustained-presence signal, face-centre movement estimate, audio energy segmentation, explainable suggestions, teacher review, local report history.")
    st.write("**Not implemented:** facial identity recognition, reliable head-pose estimation, speech transcription, speaker identification, trained engagement model, live webcam processing.")
    with st.expander("Privacy and limitations"):
        st.write("The SQLite database stores reports and review records locally. It does not store uploaded media. The local database file is created in the project folder. Remove `classpulse.db` to erase saved local history. Anonymous tracks can split or merge; camera motion and lighting affect estimates.")
