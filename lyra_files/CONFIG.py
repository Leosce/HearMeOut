import os
from pathlib import Path

import torch

# ── Environment ────────────────────────────────────────────────────────────────
ENVIRONMENT = os.environ.get("LYRA_ENVIRONMENT", "local")   # "local", "kaggle", or "colab"

LOCAL_ROOT = Path(__file__).resolve().parents[1]

if ENVIRONMENT == "kaggle":
    PROJECT_DIR = "/kaggle/input/datasets/minamichelb/lipreading-py"
    BASE_PATH   = "/kaggle/input/datasets/jedidiahangekouakou/grid-corpus-dataset-for-training-lipnet/data"
    MODEL_PATH  = "/kaggle/working/models"
    NPY_BASE    = "/kaggle/input/lipnet-npy-preprocessed"  # uploaded dataset
elif ENVIRONMENT == "colab":
    PROJECT_DIR = "/content/drive/MyDrive/LipNet"
    BASE_PATH   = "/content/drive/MyDrive/data/data"
    MODEL_PATH  = "/content/drive/MyDrive/models"
    NPY_BASE    = "/content/drive/MyDrive/npy"
else:
    PROJECT_DIR = str(Path(__file__).resolve().parent)
    BASE_PATH   = str(LOCAL_ROOT / "lyra_files" / "data")
    MODEL_PATH  = str(LOCAL_ROOT / "runs" / "lyra_models")
    NPY_BASE    = str(LOCAL_ROOT / "runs" / "lyra_npy")

os.makedirs(MODEL_PATH, exist_ok=True)
# Note: NPY_BASE is not created here because it may point to a
# read-only Kaggle input directory. Created only when preprocessing.

# ── Speakers ───────────────────────────────────────────────────────────────────
# Group 1 — start here (s1-s5), expand later
SPEAKERS = [
    "s1_processed",
    "s2_processed",
    "s3_processed",
    "s4_processed",
    "s5_processed",
]
# Uncomment to add more speakers:
# SPEAKERS += ["s6_processed", "s7_processed", "s8_processed", "s9_processed", "s10_processed"]
# SPEAKERS += [f"s{i}_processed" for i in range(11, 34) if i != 21]

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS      = 100
BATCH_SIZE  = 8
LR          = 5e-4
TRAIN_SPLIT = 0.9

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_TYPE = "lipnet"

# ── Hardware ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Vocabulary (40 classes) ────────────────────────────────────────────────────
# index 0  = CTC blank token
# index 1  = space (word separator)
# index 2+ = letters and symbols
VOCAB       = [""] + [" "] + list("abcdefghijklmnopqrstuvwxyz'?!123456789")
VOCAB_SIZE  = len(VOCAB)          # 40
char_to_idx = {c: i for i, c in enumerate(VOCAB)}
idx_to_char = {i: c for i, c in enumerate(VOCAB)}
