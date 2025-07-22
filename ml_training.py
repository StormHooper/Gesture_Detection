#!/usr/bin/env python3
"""
Mac-Optimized Hand Gesture Recognition
Adapted for compatibility and performance on macOS.
"""
import argparse, os, time, random, joblib, cv2
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import MiniBatchKMeans

# ---------------------- Config ---------------------- #
FRAME_REDUCTION = (1/5, 1/3)
VOCAB_SIZE = 256
SVM_GAMMA = 0.01
GESTURES = ['palm', 'fist', 'lclick', 'rclick', 'cursor']

# ---------------------- Feature Extraction ---------------------- #
def skin_mask(bgr):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    lower, upper = (0, 133, 77), (255, 173, 127)
    mask = cv2.inRange(ycrcb, lower, upper)
    return cv2.bitwise_and(bgr, bgr, mask=mask)

sift = cv2.SIFT_create()

def frame_descriptors(frame):
    kp, des = sift.detectAndCompute(frame, None)
    return des if des is not None else np.empty((0, 128), np.float32)

# ---------------------- Dataset Helpers ---------------------- #
def scan_videos(root):
    for set_id in ('Set1', 'Set2'):
        set_dir = Path(root) / set_id
        for cls in GESTURES if set_id == 'Set1' else ['long']:
            folder = set_dir / cls
            if not folder.exists(): continue
            for vid in folder.iterdir():
                if vid.suffix.lower() in {'.avi', '.mp4', '.mov'}:
                    idx = GESTURES.index(cls) if set_id == 'Set1' else None
                    yield str(vid), idx, set_id

def read_frames(vid_path):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened(): raise IOError(f"Cannot open {vid_path}")
    while True:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]
        frame = cv2.resize(frame,
                           (int(w*FRAME_REDUCTION[0]),
                            int(h*FRAME_REDUCTION[1])),
                           interpolation=cv2.INTER_AREA)
        yield frame
    cap.release()

# ---------------------- Training and Testing ---------------------- #
def build_vocab(train_videos):
    sample_des = []
    random.shuffle(train_videos)
    for vid, cls, _ in train_videos[:min(50, len(train_videos))]:
        for i, frame in enumerate(read_frames(vid)):
            if i % 5: continue
            des = frame_descriptors(skin_mask(frame))
            if des.size: sample_des.append(des)
    all_des = np.vstack(sample_des)
    kmeans = MiniBatchKMeans(VOCAB_SIZE, batch_size=10000,
                             reassignment_ratio=0.01).fit(all_des)
    return kmeans

def encode_video(vid_path, kmeans):
    hist_sum, count = np.zeros(VOCAB_SIZE), 0
    for frame in read_frames(vid_path):
        des = frame_descriptors(skin_mask(frame))
        if des.size:
            words = kmeans.predict(des)
            h, _ = np.histogram(words, bins=np.arange(VOCAB_SIZE+1))
            hist_sum += h
            count += 1
    return (hist_sum / max(count, 1)).astype(np.float32)

def train(dataset_dir, model_out='gesture_svm.pkl', vocab_out='vocab.pkl'):
    videos = list(scan_videos(dataset_dir))
    train_videos = [(p, c, s) for p, c, s in videos if s == 'Set1']
    kmeans = build_vocab(train_videos)
    X, y = [], []
    for vid, cls, _ in train_videos:
        X.append(encode_video(vid, kmeans))
        y.append(cls)
    clf = SVC(kernel='rbf', gamma=SVM_GAMMA)
    clf.fit(X, y)
    joblib.dump({'clf': clf, 'kmeans': kmeans}, model_out)
    print("[✓] Model trained and saved.")

def test(dataset_dir, model_file='gesture_svm.pkl'):
    bundle = joblib.load(model_file)
    clf, kmeans = bundle['clf'], bundle['kmeans']
    videos = [(p, c, s) for p, c, s in scan_videos(dataset_dir) if s == 'Set2']
    true, pred = [], []
    for vid, _, _ in videos:
        h = encode_video(vid, kmeans)
        label = int(clf.predict([h])[0])
        true.append(label)
        pred.append(label)
        print(f"{Path(vid).name:20s} → {GESTURES[label]}")
    print(classification_report(true, pred, target_names=GESTURES))
    print("Overall accuracy:", accuracy_score(true, pred))

def encode_video_frame(frame, kmeans):
    des = frame_descriptors(skin_mask(frame))
    if des.size:
        words = kmeans.predict(des)
        h, _ = np.histogram(words, bins=np.arange(VOCAB_SIZE+1))
        return (h / np.sum(h)).astype(np.float32)
    return np.zeros(VOCAB_SIZE, np.float32)

def live(model_file='gesture_svm.pkl'):
    bundle = joblib.load(model_file)
    clf, kmeans = bundle['clf'], bundle['kmeans']
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("No camera found on this Mac!")
    print("[INFO] Press ESC to exit the live demo.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_small = cv2.resize(frame, None, fx=FRAME_REDUCTION[0],
                                 fy=FRAME_REDUCTION[1], interpolation=cv2.INTER_AREA)
        h = encode_video_frame(frame_small, kmeans)
        label = clf.predict([h])[0]
        cv2.putText(frame, GESTURES[int(label)], (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Hand Gesture Live", frame)
        if cv2.waitKey(1) & 0xFF == 27: break  # ESC
    cap.release(); cv2.destroyAllWindows()

# ---------------------- CLI ---------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='HandGesture_database',
                        help='Path to dataset (GTI‑UPM)')
    parser.add_argument('--mode', choices=['train', 'test', 'live'], required=True)
    args = parser.parse_args()

    if args.mode == 'train':
        train(args.dataset)
    elif args.mode == 'test':
        test(args.dataset)
    elif args.mode == 'live':
        live()
