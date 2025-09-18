from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import os
import uuid
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import json

# ---------------------------
# Initialize Firebase
# ---------------------------
# Expect Firebase key JSON in environment variable
firebase_key_json = os.environ.get("FIREBASE_KEY_JSON")
if not firebase_key_json:
    raise ValueError("FIREBASE_KEY_JSON environment variable not set")

cred_dict = json.loads(firebase_key_json)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ---------------------------
# Initialize Flask
# ---------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB

# ---------------------------
# Load YOLO model
# ---------------------------
model = YOLO("best.pt")  # Ensure this file is in the root directory

# ---------------------------
# IoU function (for internal use)
# ---------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# ---------------------------
# Route: detect potholes
# ---------------------------
@app.route("/detect", methods=["POST"])
def detect_potholes():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    # Save video temporarily
    video_file = request.files["video"]
    video_id = str(uuid.uuid4())
    video_path = f"{video_id}.mp4"
    video_file.save(video_path)

    # Run YOLO tracking
    results = model.track(video_path, tracker="botsort.yaml", verbose=False)

    unique_potholes = []
    total_risk = 0
    frame_count = len(results)

    for frame_result in results:
        frame = frame_result.orig_img.copy()
        for box in frame_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tracker_id = getattr(box, "id", None) or getattr(box, "tracker_id", None)

            if tracker_id is not None and tracker_id not in unique_potholes:
                unique_potholes.append(tracker_id)

            # Risk contribution
            area = (x2 - x1) * (y2 - y1)
            normalized_area = area / (frame.shape[1] * frame.shape[0])
            total_risk += normalized_area

    # Remove temporary video
    os.remove(video_path)

    # Compute risk score
    risk_score = min((total_risk / frame_count) * 100, 100)
    hazard_detected = len(unique_potholes) > 0

    # Generate report JSON
    report_data = {
        "video_id": video_id,
        "timestamp": datetime.utcnow().isoformat(),
        "hazard_detected": "pothole" if hazard_detected else "none",
        "risk_score": round(risk_score, 2),
        "total_unique_potholes": len(unique_potholes),
        "total_frames": frame_count
    }

    # Store report in Firestore
    db.collection("reports").document(video_id).set(report_data)

    return jsonify(report_data)

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
