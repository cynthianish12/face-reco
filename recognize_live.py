# recognize_live_multi.py
import cv2
import mediapipe as mp
import numpy as np
import json
import os

MODEL_PATH = "lbph_model.yml"
LABELS_PATH = "labels.json"
FACE_SIZE = (200, 200)
CONFIDENCE_THRESHOLD = 80  # LBPH distance threshold: lower = better

mp_face = mp.solutions.face_detection

def load_labels(path):
    with open(path, "r") as f:
        data = json.load(f)
    return {v: k for k, v in data.items()}  # id -> name

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print("[ERROR] Model or labels not found. Run train_lbph.py first.")
        exit(1)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    id2name = load_labels(LABELS_PATH)

    cap = cv2.VideoCapture(0)

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = fd.process(rgb)

            if results.detections:
                h, w, _ = frame.shape

                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box

                    # Bounding box conversion
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)
                    x2 = x1 + bw
                    y2 = y1 + bh

                    # Add margin around face
                    margin_x = int(0.2 * bw)
                    margin_y = int(0.2 * bh)
                    x1 = max(0, x1 - margin_x)
                    y1 = max(0, y1 - margin_y)
                    x2 = min(w, x2 + margin_x)
                    y2 = min(h, y2 + margin_y)

                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue

                    # Preprocess for LBPH
                    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(gray, FACE_SIZE)

                    # LBPH prediction
                    label_id, confidence = recognizer.predict(face_resized)
                    name = id2name.get(label_id, "Unknown")

                    # Color: green=good match, red=bad match
                    color = (0, 255, 0) if confidence < CONFIDENCE_THRESHOLD else (0, 0, 255)
                    label_text = f"{name} ({confidence:.1f})"

                    # Draw bounding box & label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Multi-Face Recognition (Press q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
