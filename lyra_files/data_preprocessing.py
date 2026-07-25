"""
data_preprocessing.py
─────────────────────
Multi-speaker LipNet data pipeline.
- Supports multiple speakers in one dataset
- Loads preprocessed .npy files for fast training
- Falls back to MediaPipe mouth detection if .npy not found
- Fixed 40-class vocab (blank=0, space=1)
"""

import os
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from CONFIG import (
    BASE_PATH, NPY_BASE, SPEAKERS, TRAIN_SPLIT,
    char_to_idx, idx_to_char, VOCAB_SIZE
)

# ── Vocabulary helpers ─────────────────────────────────────────────────────────

def encode(tokens: list) -> torch.Tensor:
    """Map a list of characters to a 1-D LongTensor of indices."""
    indices = [char_to_idx[c] for c in tokens if c in char_to_idx]
    return torch.tensor(indices, dtype=torch.long)


def decode(indices) -> str:
    """Map indices back to string. Skips blank (index 0)."""
    chars = [idx_to_char[int(i)] for i in indices if int(i) != 0]
    return "".join(chars)


def ctc_decode(pred: torch.Tensor) -> list:
    """
    Greedy CTC decode on a batch of argmax sequences.
    pred : (B, T) integer tensor from torch.argmax(logits, dim=-1)
    Returns a list of decoded strings with proper word spacing.
    """
    results = []
    for seq in pred:
        out, prev = [], -1
        for tok in seq.tolist():
            if tok != prev and tok != 0:   # 0 = CTC blank, skip
                out.append(tok)
            prev = tok
        text = decode(out)
        text = " ".join(text.split())      # clean up extra spaces
        text = text.strip()
        results.append(text)
    return results


# ── File helpers ───────────────────────────────────────────────────────────────

def get_valid_filenames(video_dir: str, align_dir: str) -> list:
    """
    Return sorted list of base filenames that have both .mpg and .align files.
    Skips non-.mpg files silently.
    """
    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    if not os.path.exists(align_dir):
        raise FileNotFoundError(f"Align directory not found: {align_dir}")

    valid, skipped = [], 0
    for f in sorted(os.listdir(video_dir)):
        if not f.endswith(".mpg"):
            skipped += 1
            continue
        base       = f[:-4]
        align_path = os.path.join(align_dir, f"{base}.align")
        if not os.path.exists(align_path):
            skipped += 1
            continue
        valid.append(base)

    if skipped:
        print(f"[INFO] {video_dir}: skipped {skipped}, found {len(valid)} valid")
    return valid


# ── Video loading ──────────────────────────────────────────────────────────────

def capture_frames(video_path: str) -> np.ndarray:
    """
    Extract mouth-crop frames using MediaPipe face detection.
    Uses 40 lip landmarks for accurate centered mouth crop.
    Falls back to last successful crop if MediaPipe loses the face.
    Returns ndarray (T, 46, 100).
    """
    # Full 40-point lip landmark set (outer + inner lip)
    LIP_LANDMARKS = [
        # Outer lip
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
        # Inner lip
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95
    ]

    try:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )
        use_mp = True
    except (ImportError, AttributeError):
        face_mesh = None
        use_mp    = False

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    n_frames  = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frames    = []
    last_crop = None   # stores last successful MediaPipe crop

    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w = frame.shape[:2]
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        crop = None

        if use_mp:
            rgb     = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                # Get all 40 lip landmark coordinates
                xs = [int(lm[i].x * w) for i in LIP_LANDMARKS]
                ys = [int(lm[i].y * h) for i in LIP_LANDMARKS]

                # Find mouth center
                cx = (min(xs) + max(xs)) // 2
                cy = (min(ys) + max(ys)) // 2

                # Fixed-size centered crop
                half_w, half_h = 55, 28
                x1 = max(cx - half_w, 0);  x2 = min(cx + half_w, w)
                y1 = max(cy - half_h, 0);  y2 = min(cy + half_h, h)
                crop      = cv.resize(gray[y1:y2, x1:x2], (100, 46))
                last_crop = crop   # save for fallback

        if crop is None:
            # Reuse last successful crop — much better than hardcoded coordinates
            # Only fall back to hardcoded if no successful crop yet (first frames)
            crop = last_crop if last_crop is not None \
                else cv.resize(gray[190:236, 100:200], (100, 46))

        frames.append(crop)

    cap.release()
    if face_mesh:
        face_mesh.close()

    if not frames:
        raise ValueError(f"No frames extracted from: {video_path}")
    return np.array(frames, dtype=np.float32)


def capture_frames_any(video_path: str) -> np.ndarray:
    """Same as capture_frames — used by Streamlit app for real-world videos."""
    return capture_frames(video_path)


def reduce_frames(frames: np.ndarray) -> torch.Tensor:
    """Normalise to zero mean / unit std."""
    mean = frames.mean()
    std  = frames.std()
    std  = std if std > 1e-6 else 1.0
    return torch.from_numpy((frames - mean) / std)


def preprocess_speaker(speaker: str):
    """
    Run MediaPipe on all videos for a speaker and save as .npy files.
    Call this once per speaker before training.
    """
    import warnings
    warnings.filterwarnings('ignore')

    video_dir = os.path.join(BASE_PATH, speaker)
    npy_dir   = os.path.join(NPY_BASE, speaker)
    os.makedirs(npy_dir, exist_ok=True)

    videos = sorted([f for f in os.listdir(video_dir) if f.endswith('.mpg')])
    total  = len(videos)
    saved  = 0

    print(f"[{speaker}] Preprocessing {total} videos...")
    for i, fname in enumerate(videos):
        base     = fname[:-4]
        out_path = os.path.join(npy_dir, f"{base}.npy")

        if os.path.exists(out_path):
            saved += 1
            continue

        try:
            frames = capture_frames(os.path.join(video_dir, fname))
            np.save(out_path, frames)
            saved += 1
        except Exception as e:
            print(f"  [SKIP] {fname}: {e}")

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{total} done ({saved} saved)")

    print(f"[{speaker}] Done — {saved}/{total} files ready")
    return npy_dir


def load_video_fast(speaker: str, base: str) -> torch.Tensor:
    """
    Load video from .npy if available across multiple NPY base paths.
    Searches all paths in CONFIG.NPY_BASES then CONFIG.NPY_BASE.
    Falls back to .mpg decoding with MediaPipe if not found anywhere.
    Returns tensor (T, H, W).
    """
    import CONFIG as _cfg

    # Collect all npy base paths to search
    npy_bases = []
    if hasattr(_cfg, 'NPY_BASES'):
        npy_bases.extend(_cfg.NPY_BASES)
    if hasattr(_cfg, 'NPY_BASE') and _cfg.NPY_BASE:
        npy_bases.append(_cfg.NPY_BASE)

    # Search all paths
    for npy_base in npy_bases:
        npy_path = os.path.join(npy_base, speaker, f"{base}.npy")
        if os.path.exists(npy_path):
            return reduce_frames(np.load(npy_path))

    # Fallback to .mpg + MediaPipe
    print(f"[WARN] No .npy found for {speaker}/{base} — using slow fallback")
    mpg_path = os.path.join(BASE_PATH, speaker, f"{base}.mpg")
    return reduce_frames(capture_frames(mpg_path))


# ── Annotation loading ─────────────────────────────────────────────────────────

def split_line(annotation_path: str) -> list:
    """Parse a GRID .align file. Returns list of words with leading spaces."""
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        if parts[2] != "sil":
            tokens += [" ", parts[2]]
    return tokens


def split_words(word_tokens: list) -> list:
    """Flatten words into individual characters."""
    return [ch for word in word_tokens for ch in list(word)]


def load_annot(filename_or_path: str, from_path: bool = False) -> torch.Tensor:
    """Load annotation and return encoded LongTensor."""
    if from_path:
        path = filename_or_path
    else:
        # Legacy single-speaker support
        from CONFIG import SPEAKERS
        speaker   = SPEAKERS[0]
        align_dir = os.path.join(BASE_PATH, speaker, "align")
        path      = os.path.join(align_dir, f"{filename_or_path}.align")
    words = split_line(path)
    chars = split_words(words)
    return encode(chars)


def load_annot_speaker(speaker: str, base: str) -> torch.Tensor:
    """Load annotation for a specific speaker and base filename."""
    path = os.path.join(BASE_PATH, speaker, "align", f"{base}.align")
    return encode(split_words(split_line(path)))


# ── Multi-speaker Dataset ──────────────────────────────────────────────────────

class LipDataset(Dataset):
    """
    Multi-speaker LipNet dataset.
    Loads from all speakers in CONFIG.SPEAKERS.
    Uses preprocessed .npy files when available for fast loading.
    Caches all annotations in memory at init for fast __getitem__.
    """
    def __init__(self, split: str = "train"):
        super().__init__()
        self.samples = []
        self.cache   = {}   # annotation cache — loaded once at init

        for speaker in SPEAKERS:
            video_dir = os.path.join(BASE_PATH, speaker)
            align_dir = os.path.join(BASE_PATH, speaker, "align")

            if not os.path.exists(video_dir):
                print(f"[WARN] Speaker not found: {speaker}")
                continue

            files = get_valid_filenames(video_dir, align_dir)
            n     = len(files)
            cut   = int(TRAIN_SPLIT * n)

            split_files = files[:cut] if split == "train" else files[cut:]

            for f in split_files:
                self.samples.append((speaker, f))
                key = f"{speaker}/{f}"
                if key not in self.cache:
                    self.cache[key] = load_annot_speaker(speaker, f)

        print(f"[LipDataset] {split}: {len(self.samples)} samples "
              f"from {len(SPEAKERS)} speakers — "
              f"{len(self.cache)} annotations cached")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        speaker, base = self.samples[idx]
        video = load_video_fast(speaker, base).float().unsqueeze(-1)  # (T,H,W,1)
        label = self.cache[f"{speaker}/{base}"].long()                 # instant from cache
        return video, label


# ── Collate ────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Pads a batch of (video, label) pairs.
    Returns:
        videos_padded  – (B, T_max, H, W, 1)  float32
        labels_padded  – (B, L_max)            long
        video_lengths  – (B,)
        label_lengths  – (B,)
    """
    videos, labels = zip(*batch)

    video_lengths = torch.tensor([v.shape[0] for v in videos], dtype=torch.long)
    label_lengths = torch.tensor([l.shape[0] for l in labels], dtype=torch.long)

    videos_padded = pad_sequence(videos, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)

    return videos_padded, labels_padded, video_lengths, label_lengths
