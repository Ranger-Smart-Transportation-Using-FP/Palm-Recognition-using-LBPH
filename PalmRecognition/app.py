import os
import cv2
import numpy as np
import pickle
import time

# Directories and files
DATA_DIR     = "palm_dataset"
TRAINER_FILE = "palm_trainer.yml"
LABELS_FILE  = "palm_labels.pickle"

# Capture settings
SAMPLE_COUNT = 20     # total samples to collect
TIME_INTERVAL = 1.0   # seconds between auto-captures (if enabled)

# ————————————————————————— Registration —————————————————————————
def register_user(name: str, samples: int = SAMPLE_COUNT, auto_capture: bool = False):
    """
    Capture `samples` palm images for `name` using a centered ROI.
    If auto_capture=True, captures a sample every TIME_INTERVAL seconds when palm is detected.
    Otherwise, press 'c' or 'C' to capture manually.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cam = cv2.VideoCapture(0)
    count = 0
    last_time = time.time()
    print(f"[+] Registering '{name}'. Place your palm in the box.")

    while count < samples:
        ret, frame = cam.read()
        if not ret:
            break
        # Mirror for convenience
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        # Dynamic ROI: half of smaller dimension
        size = int(min(h, w) * 0.5)
        x1 = w//2 - size//2
        y1 = h//2 - size//2
        x2 = x1 + size
        y2 = y1 + size

        # Draw ROI and instructions
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text = f"Samples: {count}/{samples} | Press 'c' to capture"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Register Palm", frame)

        key = cv2.waitKey(1) & 0xFF
        grab = False
        # Manual capture
        if key in (ord('c'), ord('C')):
            grab = True
        # Auto capture if enabled
        elif auto_capture and (time.time() - last_time) >= TIME_INTERVAL:
            grab = True
            last_time = time.time()
        # Quit
        elif key == ord('q'):
            break

        if grab:
            roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            filename = os.path.join(DATA_DIR, f"{name}_{count+1}.jpg")
            cv2.imwrite(filename, gray)
            count += 1
            print(f"[+] Captured sample {count}/{samples}")

    cam.release()
    cv2.destroyAllWindows()

    if count > 0:
        train_model()
        print(f"[+] Registration complete for '{name}'. Model updated.")
    else:
        print("[!] No samples captured. Registration aborted.")

# ————————————————————————— Training —————————————————————————
def train_model():
    """
    Scan the dataset folder, assign numeric labels, train LBPH, and save.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    label_ids = {}
    current_id = 0
    x_train, y_labels = [], []

    for fn in os.listdir(DATA_DIR):
        if not fn.lower().endswith('.jpg'):
            continue
        name = fn.split('_')[0]
        if name not in label_ids:
            label_ids[name] = current_id
            current_id += 1

        img_path = os.path.join(DATA_DIR, fn)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        x_train.append(img)
        y_labels.append(label_ids[name])

    # Persist mapping and model
    with open(LABELS_FILE, 'wb') as f:
        pickle.dump(label_ids, f)
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save(TRAINER_FILE)
    print("[*] Palm model trained and saved.")

# ————————————————————————— Recognition/Login —————————————————————————
def recognize_user(draw: bool = True, threshold: float = 50.0):
    """
    Detect and recognize a palm. Returns the user name or 'Unknown'.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)
    with open(LABELS_FILE, 'rb') as f:
        label_ids = pickle.load(f)
    id_to_name = {v: k for k, v in label_ids.items()}

    cam = cv2.VideoCapture(0)
    name_found = "Unknown"
    print("[*] Starting palm recognition. Place your palm in the box.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        size = int(min(h, w) * 0.5)
        x1 = w//2 - size//2
        y1 = h//2 - size//2
        x2 = x1 + size
        y2 = y1 + size

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        label_id, conf = recognizer.predict(gray)
        if conf < threshold:
            name_found = id_to_name.get(label_id, "Unknown")
            color = (0, 255, 0)
        else:
            name_found = "Unknown"
            color = (0, 0, 255)

        if draw:
            text = f"{name_found} ({conf:.1f})"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.imshow("Recognize Palm", frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'),) or name_found != "Unknown":
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Detected: {name_found}")
    return name_found

# ————————————————————————— Entry Point —————————————————————————
if __name__ == "__main__":
    mode = input("Enter mode (register/login): ").strip().lower()
    if mode == "register":
        user = input("Enter name to register: ").strip()
        register_user(user)
    elif mode == "login":
        recognize_user(draw=True)
    else:
        print("Unknown mode. Use 'register' or 'login'.")
