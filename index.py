import cv2
import mediapipe as mp
import os
import time

DATASET_DIR = "dataset"
FPS_LIMIT = 5  # capture rate (frames per second)
CAPTURE_PER_PERSON = 100  # target images per label
FACE_SIZE = (200, 200)  # size to save

mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

os.makedirs(DATASET_DIR, exist_ok=True)

def collect_for_label(label):
    label_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        print(f"[INFO] Starting capture for label '{label}'. Press 'q' to stop early.")
        saved = len([f for f in os.listdir(label_dir) if f.lower().endswith(".png")])
        last_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            now = time.time()
            if now - last_time < 1.0 / FPS_LIMIT:
                # show but skip processing to reduce load
                cv2.imshow("Collect (Press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            last_time = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb)

            if results.detections:
                # choose largest detection (closest face)
                det = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
                bbox = det.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1 = int(max(0, bbox.xmin * w))
                y1 = int(max(0, bbox.ymin * h))
                x2 = int(min(w, (bbox.xmin + bbox.width) * w))
                y2 = int(min(h, (bbox.ymin + bbox.height) * h))

                # expand a little margin
                margin_x = int(0.2 * (x2 - x1))
                margin_y = int(0.2 * (y2 - y1))
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(w, x2 + margin_x)
                y2 = min(h, y2 + margin_y)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face_resized = cv2.resize(face, FACE_SIZE)
                saved += 1
                fname = os.path.join(label_dir, f"img_{saved:03d}.png")
                cv2.imwrite(fname, face_resized)
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame, f"{label} #{saved}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.imshow("Collect (Press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # stop once we have enough
            if saved >= CAPTURE_PER_PERSON:
                print(f"[INFO] Collected {saved} images for label '{label}'.")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    label = input("Enter label (person name or id) to collect images for: ").strip()
    if label:
        collect_for_label(label)
    else:
        print("Label empty, exiting.")
