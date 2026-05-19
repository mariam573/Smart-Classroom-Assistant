"""Smart Classroom Assistant — Flask Web App"""
import os, csv, sys, uuid, base64, threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging, warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from Project import (
    load_known_faces, load_student_registry,
    run_attendance_session, save_attendance_csv,
    build_topic_classifier, capture_speech,
    classify_question, save_question_log,
    ATTENDANCE_CSV, QUESTIONS_CSV, FACES_DIR,
)

app = Flask(__name__)
app.secret_key = "classroom_2025"

# ── Global state ─────────────────────────────────────────────
tasks     = {}
scan_lock = threading.Lock()
classifier = None
registry   = {}

def initialize():
    global classifier, registry
    load_known_faces(FACES_DIR)
    classifier = build_topic_classifier()
    registry   = load_student_registry()

# ── Pages ─────────────────────────────────────────────────────
@app.route("/")
def index():       return render_template("index.html")

@app.route("/attendance")
def attendance():  return render_template("attendance.html")

@app.route("/question")
def question():    return render_template("question.html")

@app.route("/dashboard")
def dashboard():   return render_template("dashboard.html")

# ── Stats ─────────────────────────────────────────────────────
@app.route("/api/stats")
def api_stats():
    today = datetime.now().strftime("%Y-%m-%d")
    att_today, q_total, students = 0, 0, set()
    if os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                students.add(r.get("student_id",""))
                if r.get("date","") == today: att_today += 1
    if os.path.exists(QUESTIONS_CSV):
        with open(QUESTIONS_CSV, newline="", encoding="utf-8") as f:
            q_total = sum(1 for _ in csv.DictReader(f))
    return jsonify({"today_attendance": att_today,
                    "total_students": len([s for s in students if s and s!="UNKNOWN"]),
                    "total_questions": q_total})

# ── History ───────────────────────────────────────────────────
@app.route("/api/attendance-history")
def api_att_history():
    rows = []
    if os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    return jsonify(list(reversed(rows)))

@app.route("/api/questions-history")
def api_q_history():
    rows = []
    if os.path.exists(QUESTIONS_CSV):
        with open(QUESTIONS_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    return jsonify(list(reversed(rows)))

@app.route("/api/registry")
def api_registry():
    return jsonify([{"student_id": k, "name": v} for k,v in registry.items()])

# ── Attendance scan ───────────────────────────────────────────
@app.route("/api/start-scan", methods=["POST"])
def api_start_scan():
    if scan_lock.locked():
        return jsonify({"error": "Scan already running"}), 409
    tid = str(uuid.uuid4())
    tasks[tid] = {"status": "running", "result": None}

    def run():
        with scan_lock:
            try:
                import cv2
                att, unk, unk_mood = run_attendance_session(FACES_DIR, duration_seconds=15, headless=True)
                result = {
                    "recognized": [{"id":s,"name":i["name"],"mood":i["mood"]} for s,i in att.items()],
                    "has_unknown": unk is not None,
                    "unknown_frame_b64": None,
                    "unknown_mood": unk_mood,
                }
                if unk is not None:
                    _, buf = cv2.imencode(".jpg", unk)
                    result["unknown_frame_b64"] = base64.b64encode(buf).decode()
                tasks[tid] = {"status": "done", "result": result}
            except Exception as e:
                tasks[tid] = {"status": "error", "result": str(e)}

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"task_id": tid})

@app.route("/api/scan-status/<tid>")
def api_scan_status(tid):
    return jsonify(tasks.get(tid, {"status": "not_found", "result": None}))

@app.route("/api/save-attendance", methods=["POST"])
def api_save_attendance():
    data = request.get_json()
    att  = {s["id"]: {"name": s["name"], "mood": s["mood"]} for s in data.get("recognized",[])}
    if att: save_attendance_csv(att, datetime.now().isoformat())
    return jsonify({"success": True, "saved": len(att)})

@app.route("/api/register-student", methods=["POST"])
def api_register_student():
    global registry
    import glob
    data = request.get_json()
    sid, name, fb64 = data.get("student_id","").strip(), data.get("name","").strip(), data.get("frame_b64","")
    mood = data.get("mood", "neutral")
    if not sid or not name:
        return jsonify({"success": False, "error": "Missing ID or name"}), 400
    try:
        import cv2, numpy as np
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(fb64), np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(FACES_DIR, f"{sid}_{name}.jpg"), frame)
        # Clear DeepFace pkl cache so next scan picks up the new photo
        for pkl in glob.glob(os.path.join(FACES_DIR, "*.pkl")):
            try: os.remove(pkl)
            except OSError: pass
        save_attendance_csv({sid: {"name": name, "mood": mood}}, datetime.now().isoformat())
        registry[sid] = name
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ── Speech / Q&A ──────────────────────────────────────────────
@app.route("/api/start-recording", methods=["POST"])
def api_start_recording():
    tid = str(uuid.uuid4())
    tasks[tid] = {"status": "running", "result": None}

    def run():
        try:
            q = capture_speech()
            if q:
                subj = classify_question(classifier, q)
                tasks[tid] = {"status":"done","result":{"question":q,"subject":subj}}
            else:
                tasks[tid] = {"status":"done","result":{"question":None,"subject":None,"error":"No speech detected"}}
        except Exception as e:
            tasks[tid] = {"status":"error","result":str(e)}

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"task_id": tid})

@app.route("/api/recording-status/<tid>")
def api_recording_status(tid):
    return jsonify(tasks.get(tid, {"status": "not_found", "result": None}))

@app.route("/api/save-question", methods=["POST"])
def api_save_question():
    d = request.get_json()
    save_question_log({"timestamp": datetime.now().isoformat(),
                       "student_id":   d.get("student_id","UNKNOWN"),
                       "student_name": d.get("student_name","Unknown"),
                       "question":     d.get("question",""),
                       "subject":      d.get("subject","")})
    return jsonify({"success": True})

if __name__ == "__main__":
    initialize()
    app.run(debug=False, port=5000, threaded=True)
