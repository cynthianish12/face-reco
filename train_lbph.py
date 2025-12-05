# train_lbph.py
import cv2
import os
import json
import numpy as np

DATASET_DIR = "dataset"
MODEL_PATH = "lbph_model.yml"
LABELS_PATH = "labels.json"
FACE_SIZE = (200, 200)

def load_images_and_labels(dataset_dir):
    images = []
    labels = []
    label_map = {}
    next_label = 0
    for entry in sorted(os.listdir(dataset_dir)):
        folder = os.path.join(dataset_dir, entry)
        if not os.path.isdir(folder):
            continue
        label_name = entry
        if label_name not in label_map:
            label_map[label_name] = next_label
            next_label += 1
        lbl = label_map[label_name]
        for f in sorted(os.listdir(folder)):
            if not (f.lower().endswith(".png") or f.lower().endswith(".jpg") or f.lower().endswith(".jpeg")):
                continue
            path = os.path.join(folder, f)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, FACE_SIZE)
            images.append(img)
            labels.append(lbl)
    return images, labels, label_map

if __name__ == "__main__":
    print("[INFO] Loading dataset...")
    images, labels, label_map = load_images_and_labels(DATASET_DIR)
    if len(images) == 0:
        print("[ERROR] No images found. Run collect_dataset.py first.")
        exit(1)
    print(f"[INFO] Found {len(images)} face images with {len(label_map)} labels.")

    images_np = [np.array(i, dtype=np.uint8) for i in images]
    labels_np = np.array(labels, dtype=np.int32)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("[INFO] Training LBPH recognizer (this may take a few seconds)...")
    recognizer.train(images_np, labels_np)
    recognizer.write(MODEL_PATH)
    print(f"[INFO] Saved model to {MODEL_PATH}")

    # save label map (name -> id)
    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f)
    print(f"[INFO] Saved labels map to {LABELS_PATH}")
