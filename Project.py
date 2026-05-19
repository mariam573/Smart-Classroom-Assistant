import os
import sys

# Force UTF-8 output on Windows (avoids charmap errors with special chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import cv2
import csv
import json
import time
import threading
import joblib
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from deepface import DeepFace

# ─────────────────────────────────────────────
# MODULE 1: COMPUTER VISION (Attendance + Emotion)
# ─────────────────────────────────────────────

FACES_DIR        = "known_faces"
ATTENDANCE_CSV   = "attendance.csv"
ATTENDANCE_FIELDS = ["date", "session_timestamp", "student_id", "name", "mood", "status"]


def load_known_faces(faces_dir=FACES_DIR):
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)
        print(f"[INFO] Created '{faces_dir}/' — add student photos named <ID>_<Name>.jpg")
    print(f"[CV] Using face database folder: {faces_dir}")
    return faces_dir


def _parse_identity(filepath):
    stem = os.path.splitext(os.path.basename(filepath))[0]  # e.g. S001_Ahmed
    if "_" in stem:
        student_id, name = stem.split("_", 1)
    else:
        student_id = stem
        name = stem
    name = name.replace("_", " ")
    return student_id.strip(), name.strip()


def _analyze_frame(frame_small, faces_dir, result_holder):
    import tempfile

    student_id = "UNKNOWN"
    name       = "Unknown"
    tmp_path   = os.path.join(tempfile.gettempdir(), "tmp_frame.jpg")
    cv2.imwrite(tmp_path, frame_small)

    try:
        dfs = DeepFace.find(
            img_path=tmp_path,
            db_path=faces_dir,
            enforce_detection=False,
            silent=True
        )
        if len(dfs) > 0 and len(dfs[0]) > 0:
            best_match = dfs[0].iloc[0]["identity"]
            student_id, name = _parse_identity(best_match)
    except Exception:
        pass

    mood = "neutral"
    try:
        result = DeepFace.analyze(
            frame_small,
            actions=["emotion"],
            enforce_detection=False,
            silent=True
        )
        mood = result[0]["dominant_emotion"]
    except Exception:
        pass

    result_holder["student_id"] = student_id
    result_holder["name"]       = name
    result_holder["mood"]       = mood
    result_holder["new_result"] = True
    result_holder["busy"]       = False


def run_attendance_session(faces_dir, duration_seconds=15, headless=False):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    attendance     = {}    # {student_id: {name, mood}}
    unknown_frame  = None  # single snapshot of the first unrecognised face
    unknown_mood   = "neutral"  # mood at the moment of unknown detection
    start          = datetime.now()
    frame_count    = 0

    print(f"\n[CV] Starting attendance session ({duration_seconds}s)...")
    print("[CV] Press 'q' to stop early.")

    result_holder = {
        "student_id": "UNKNOWN",
        "name":       "Scanning...",
        "mood":       "—",
        "busy":       False,
        "new_result": False,
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        if frame_count % 15 == 0 and not result_holder["busy"]:
            small = cv2.resize(frame, (320, 240))
            result_holder["busy"] = True
            threading.Thread(
                target=_analyze_frame,
                args=(small, faces_dir, result_holder),
                daemon=True,
            ).start()

        sid  = result_holder["student_id"]
        name = result_holder["name"]
        mood = result_holder["mood"]

        if result_holder["new_result"]:
            result_holder["new_result"] = False
            if sid != "UNKNOWN" and name not in ("Unknown", "Scanning..."):
                # Known face — record attendance
                attendance[sid] = {"name": name, "mood": mood}
            elif sid == "UNKNOWN" and name == "Unknown" and unknown_frame is None:
                # New/unknown face — capture ONE snapshot + mood then stop
                unknown_frame = frame.copy()
                unknown_mood  = mood
                print("[CV] Unknown face detected -- will register after session.")

        color = (0, 255, 0) if name not in ("Unknown", "Scanning...") else (0, 0, 255)
        label = f"[{sid}] {name} | {mood}"

        if not headless:
            cv2.putText(frame, label, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Smart Classroom - Attendance", frame)

        if (datetime.now() - start).seconds >= duration_seconds:
            break
        if not headless and cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if headless:
            time.sleep(0.033)   # ~30 fps pacing without cv2.waitKey

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    for sid, info in attendance.items():
        print(f"[CV] OK Recognised: [{sid}] {info['name']} | Mood: {info['mood']}")

    return attendance, unknown_frame, unknown_mood


def _clear_deepface_cache(faces_dir):
    import glob
    for pkl in glob.glob(os.path.join(faces_dir, "*.pkl")):
        try:
            os.remove(pkl)
        except OSError:
            pass


def register_face(frame, faces_dir, registry, mood="neutral"):
    window_title = "Unknown Face -- Register"
    cv2.imshow(window_title, frame)
    cv2.waitKey(500)

    print("\n[CV] Unknown face shown in window.")
    student_id = input("  Student ID   (e.g. 22100512) or SKIP: ").strip()

    if not student_id or student_id.upper() == "SKIP":
        cv2.destroyAllWindows()
        print("  [CV] Skipped.")
        return None

    if student_id in registry:
        name = registry[student_id]
        print(f"  [CV] Student [{student_id}] {name} found in registry -- photo saved.")
    else:
        name = input("  Student Name (e.g. Ahmed):         ").strip()
        if not name:
            cv2.destroyAllWindows()
            print("  [CV] No name entered -- skipped.")
            return None

    filename = f"{student_id}_{name}.jpg"
    filepath = os.path.join(faces_dir, filename)
    cv2.imwrite(filepath, frame)
    cv2.destroyAllWindows()

    # Invalidate DeepFace embedding cache so the new photo is picked up next scan
    _clear_deepface_cache(faces_dir)

    print(f"  [CV] Saved [{student_id}] {name} -> {filepath}")
    return student_id, {"name": name, "mood": mood}


def save_attendance_csv(attendance, session_ts, path=ATTENDANCE_CSV):
    file_exists = os.path.exists(path)
    date_str    = datetime.now().strftime("%Y-%m-%d")

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ATTENDANCE_FIELDS)
        if not file_exists:
            writer.writeheader()
        for sid, info in attendance.items():
            writer.writerow({
                "date":              date_str,
                "session_timestamp": session_ts,
                "student_id":        sid,
                "name":              info["name"],
                "mood":              info["mood"],
                "status":            "Present",
            })

    print(f"[CV] Attendance saved → {path}  ({len(attendance)} student(s))")


def load_student_registry(path=ATTENDANCE_CSV):
    registry = {}
    if not os.path.exists(path):
        return registry
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid  = row.get("student_id", "").strip()
            name = row.get("name", "").strip()
            if sid and sid != "UNKNOWN" and name:
                registry[sid] = name
    print(f"[CV] Registry loaded — {len(registry)} known student(s) from past sessions.")
    return registry


# ─────────────────────────────────────────────
# MODULE 2: SPEECH RECOGNITION
# ─────────────────────────────────────────────

def capture_speech(timeout=5, phrase_limit=10):
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    mic        = sr.Microphone()

    print("\n[SPEECH] Calibrating microphone...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        print("[SPEECH] Mic ready -- speak your question now!")
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
        except sr.WaitTimeoutError:
            print("[SPEECH] No speech detected within timeout.")
            return None

    try:
        text = recognizer.recognize_google(audio)
        print(f"[SPEECH] Transcribed: '{text}'")
        return text
    except Exception:
        return None


# ─────────────────────────────────────────────
# MODULE 3: NLP CLASSIFIER
# ─────────────────────────────────────────────

CLASSIFIER_PATH   = "topic_classifier.joblib"
TRAINING_DATA_CSV = "training_data.csv"


def build_topic_classifier():
    if os.path.exists(CLASSIFIER_PATH):
        pipeline = joblib.load(CLASSIFIER_PATH)
        print("[NLP] Classifier loaded from disk.")
        return pipeline

    if not os.path.exists(TRAINING_DATA_CSV):
        raise FileNotFoundError(
            f"[NLP] Training data file '{TRAINING_DATA_CSV}' not found. "
            "Create it with columns: question,subject"
        )

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    texts, labels = [], []
    with open(TRAINING_DATA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", "").strip()
            s = row.get("subject", "").strip()
            if q and s:
                texts.append(q)
                labels.append(s)

    if not texts:
        raise ValueError("[NLP] No valid rows found in training_data.csv.")

    print(f"[NLP] Loaded {len(texts)} training examples from '{TRAINING_DATA_CSV}'.")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
        ("clf",   MultinomialNB()),
    ])

    pipeline.fit(texts, labels)
    joblib.dump(pipeline, CLASSIFIER_PATH)
    print("[NLP] Classifier trained and saved to disk.")
    return pipeline


def classify_question(classifier, question_text):
    if not question_text:
        return "Unknown"
    return classifier.predict([question_text])[0]


# ─────────────────────────────────────────────
# MODULE 4: QUESTION LOGGING
# ─────────────────────────────────────────────

QUESTIONS_CSV    = "student_questions.csv"
QUESTIONS_FIELDS = ["timestamp", "student_id", "student_name", "question", "subject"]


def save_question_log(record, path=QUESTIONS_CSV):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QUESTIONS_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" SMART CLASSROOM ASSISTANT")
    print("=" * 60)

    faces_dir  = load_known_faces()
    classifier = build_topic_classifier()

    # Load all students who have attended before (from attendance.csv)
    registry = load_student_registry()

    # ── Attendance Phase ──────────────────────────────────────────
    session_ts = datetime.now().isoformat()
    attendance = {}   # {student_id: {name, mood}} for THIS session

    print("\n[SESSION] Attendance Phase — scan one student at a time.")
    print("          Type 'no' when everyone is scanned to move to Q&A.\n")

    try:
        while True:
            att, unknown_frame, unknown_mood = run_attendance_session(faces_dir, duration_seconds=15)

            # Merge all recognised students into session attendance
            for sid, info in att.items():
                if sid not in attendance:
                    attendance[sid] = info
                    print(f"[CV] OK Recognised: [{sid}] {info['name']} -- marked Present")

            # Handle unknown face: register inline
            if unknown_frame is not None:
                result = register_face(unknown_frame, faces_dir, registry, mood=unknown_mood)
                if result:
                    sid, info = result
                    attendance[sid] = info
                    registry[sid]   = info["name"]
                    print(f"[CV] OK Registered & marked Present: [{sid}] {info['name']}")

            choice = input("\n[SESSION] Scan another student? (yes / no to start Q&A): ").strip().lower()
            if choice not in ("yes", "y"):
                break

    except KeyboardInterrupt:
        print("\n[CV] Attendance phase interrupted — saving what was captured.")
        session_ts = session_ts if 'session_ts' in locals() else datetime.now().isoformat()

    # Save this session's attendance to CSV
    save_attendance_csv(attendance, session_ts)

    # Build a name → student_id lookup for the Q&A phase
    name_to_id = {info["name"].lower(): sid for sid, info in attendance.items()}

    # ── Q&A Phase ─────────────────────────────────────────────────
    print("\n[SESSION] Q&A Phase started.  Press Ctrl+C to finish.\n")

    try:
        while True:
            input("Press ENTER when a student is ready to ask a question...")
            question = capture_speech()

            if not question:
                print("[SESSION] No question captured — try again.")
                continue

            # Classify and immediately show the subject
            subject = classify_question(classifier, question)
            print(f"\n[NLP] Subject detected: {subject}")

            # Identify the student
            raw_name = input("Who is asking the question? ").strip()
            sid = name_to_id.get(raw_name.lower(), "UNKNOWN")

            record = {
                "timestamp":    datetime.now().isoformat(),
                "student_id":   sid,
                "student_name": raw_name or "Unknown",
                "question":     question,
                "subject":      subject,
            }

            save_question_log(record)
            print(f"[LOG] Saved → {QUESTIONS_CSV}  [{sid}] {raw_name or 'Unknown'} | {subject}")

    except KeyboardInterrupt:
        print("\n[SESSION] Q&A ended by user.")

    print("[DONE] All logs saved.")


if __name__ == "__main__":
    main()