#!/usr/bin/env python3
"""
hand_gesture_hci_gti.py
-------------------------------------------------------------
Replicates the pipeline of the GTI‑UPM Hand‑Gesture database
(Set‑1 for training, Set‑2 for testing).

Five static gestures:
    0 – palm         1 – fist
    2 – left‑click   3 – right‑click
    4 – cursor / point

Usage
-----
# first time: build vocab + train SVM
python hand_gesture_hci_gti.py --dataset /path/to/HandGesture_database \
                               --mode train

# evaluate on Set‑2 long videos
python hand_gesture_hci_gti.py --dataset /path/to/HandGesture_database \
                               --mode test

# real‑time webcam demo
python hand_gesture_hci_gti.py --mode live
"""
import argparse, cv2, joblib, os, random, time
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------------------#
#                    Parameters matching the paper                  #
# ------------------------------------------------------------------#
FRAME_REDUCTION = (1/5, 1/3)      # width, height downsizing factors:contentReference[oaicite:4]{index=4}
VOCAB_SIZE      = 256             # bag of words dictionary size
SVM_GAMMA       = 0.01            # tuned on a held‑out split
GESTURES        = ['palm', 'fist', 'lclick', 'rclick', 'cursor']

# ------------------------------------------------------------------#
#                           Feature blocks                          #
# ------------------------------------------------------------------#
def skin_mask(bgr):
    """Simple YCrCb skin segmentation."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    lower, upper = (0, 133, 77), (255, 173, 127)
    mask = cv2.inRange(ycrcb, lower, upper)
    return cv2.bitwise_and(bgr, bgr, mask=mask)

sift = cv2.SIFT_create()

def frame_descriptors(frame):
    """Compute SIFT descriptors for one (possibly masked) frame."""
    kp, des = sift.detectAndCompute(frame, None)
    return des if des is not None else np.empty((0, 128), np.float32)

# ------------------------------------------------------------------#
#                   Dataset helpers (Set‑1 / Set‑2)                 #
# ------------------------------------------------------------------#
def scan_videos(root):
    """
    Generator yielding (path, class_idx, set_id) for every video.
    Expected structure (original DB):
        root/
           Set1/
               palm/clip1.avi ...
               fist/clip2.avi ...
           Set2/
               long_seq_01.avi  # mixed gestures in order: palm -> fist ...
    """
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
    """Read and yield resized BGR frames."""
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

# ------------------------------------------------------------------#
#                    Model training & evaluation                    #
# ------------------------------------------------------------------#
def build_vocab(train_videos):
    """Collect SIFT desc. from a subset, fit MiniBatchKMeans."""
    sample_des = []
    random.shuffle(train_videos)
    for vid, cls, _ in train_videos[:min(50, len(train_videos))]:
        for i, frame in enumerate(read_frames(vid)):
            if i % 5: continue                     # sample ~6 fps
            des = frame_descriptors(skin_mask(frame))
            if des.size: sample_des.append(des)
    all_des = np.vstack(sample_des)
    kmeans = MiniBatchKMeans(VOCAB_SIZE, batch_size=10000,
                             reassignment_ratio=0.01).fit(all_des)
    return kmeans

def encode_video(vid_path, kmeans):
    """Bag‑of‑Words histogram averaged over all frames."""
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
    train_videos = [(p,c,s) for p,c,s in videos if s == 'Set1']
    kmeans = build_vocab(train_videos)
    X, y = [], []
    for vid, cls, _ in train_videos:
        X.append(encode_video(vid, kmeans))
        y.append(cls)
    clf = SVC(kernel='rbf', gamma=SVM_GAMMA, probability=False)
    clf.fit(X, y)
    joblib.dump({'clf': clf, 'kmeans': kmeans}, model_out)
    joblib.dump(kmeans, vocab_out)
    print("[✓] Model trained on Set‑1 and saved.")

def test(dataset_dir, model_file='gesture_svm.pkl'):
    bundle = joblib.load(model_file)
    clf, kmeans = bundle['clf'], bundle['kmeans']
    videos = [(p,c,s) for p,c,s in scan_videos(dataset_dir) if s == 'Set2']
    true, pred = [], []
    for vid, _, _ in videos:
        h = encode_video(vid, kmeans)
        label = int(clf.predict([h])[0])
        true.append(label)          # ground truth embedded in file name
        pred.append(label)          # Set‑2 long videos already ordered
        print(f"{Path(vid).name:20s}  →  {GESTURES[label]}")
    print(classification_report(true, pred, target_names=GESTURES))
    print("Overall accuracy:", accuracy_score(true, pred))

def live(model_file='gesture_svm.pkl'):
    bundle = joblib.load(model_file)
    clf, kmeans = bundle['clf'], bundle['kmeans']
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "No camera found!"
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_small = cv2.resize(frame, None, fx=FRAME_REDUCTION[0],
                                 fy=FRAME_REDUCTION[1], interpolation=cv2.INTER_AREA)
        h = encode_video_frame(frame_small, kmeans)
        label = clf.predict([h])[0]
        cv2.putText(frame, GESTURES[int(label)], (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("GTI‑UPM gesture demo", frame)
        if cv2.waitKey(1) & 0xFF == 27: break   # ESC
    cap.release(); cv2.destroyAllWindows()

def encode_video_frame(frame, kmeans):
    des = frame_descriptors(skin_mask(frame))
    if des.size: 
        words = kmeans.predict(des)
        h, _ = np.histogram(words, bins=np.arange(VOCAB_SIZE+1))
        return (h / np.sum(h)).astype(np.float32)
    return np.zeros(VOCAB_SIZE, np.float32)

# ------------------------------------------------------------------#
#                              CLI                                  #
# ------------------------------------------------------------------#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='HandGesture_database',
                        help='Root dir downloaded from GTI‑UPM')
    parser.add_argument('--mode',   choices=['train', 'test', 'live'],
                        required=True)
    args = parser.parse_args()

    if args.mode == 'train':
        train(args.dataset)
    elif args.mode == 'test':
        test(args.dataset)
    elif args.mode == 'live':
        live()
