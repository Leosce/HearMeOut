"""Decode a video with Lyra using the original LipNet preprocessing files.

The Lyra training/inference files in ``lyra_files`` are the source of truth:

* ``data_preprocessing.capture_frames_any`` extracts 46x100 mouth crops.
* ``data_preprocessing.reduce_frames`` applies zero-mean/unit-std normalization.
* ``data_preprocessing.ctc_decode`` uses the training vocabulary from CONFIG.
* ``model.LipNet`` expects tensors shaped (B, T, 46, 100, 1).

This runner keeps the desktop app aligned with that pipeline and writes
``hypotheses.txt`` plus ``prediction.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LYRA_FILES = PROJECT_ROOT / "lyra_files"
DEFAULT_EXTRACTED_CHECKPOINT = PROJECT_ROOT / "best_model.pt" / "best_model"
DEFAULT_RUNTIME_CHECKPOINT = PROJECT_ROOT / "runs" / "grid_lipnet_checkpoint_runtime.pt"

if str(LYRA_FILES) not in sys.path:
    sys.path.insert(0, str(LYRA_FILES))

try:
    from CONFIG import VOCAB, VOCAB_SIZE  # type: ignore
    from data_preprocessing import capture_frames_any, ctc_decode, reduce_frames  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(f"Could not load Lyra files from {LYRA_FILES}: {exc}") from exc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


GRID_COMMANDS = {"bin", "lay", "place", "set"}
GRID_COLORS = {"blue", "green", "red", "white"}
GRID_PREPOSITIONS = {"at", "by", "in", "with"}
GRID_LETTERS = set("abcdefghijklmnopqrstuvwxyz")
GRID_DIGITS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
GRID_ADVERBS = {"again", "now", "please", "soon"}
GRID_WORDS = GRID_COMMANDS | GRID_COLORS | GRID_PREPOSITIONS | GRID_LETTERS | GRID_DIGITS | GRID_ADVERBS


class LipNet(nn.Module):
    """Exact Lyra LipNet architecture from lyra_files/model.py for inference."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.drop1 = nn.Dropout3d(0.3)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.drop2 = nn.Dropout3d(0.3)

        self.conv3 = nn.Conv3d(64, 96, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False)
        self.bn3 = nn.BatchNorm3d(96)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.drop3 = nn.Dropout3d(0.3)

        self._cnn_feat = 96 * 5 * 12

        self.gru1 = nn.GRU(self._cnn_feat, 256, bidirectional=True, batch_first=True)
        self.ln1 = nn.LayerNorm(512)
        self.drop4 = nn.Dropout(dropout)

        self.gru2 = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        self.ln2 = nn.LayerNorm(512)
        self.drop5 = nn.Dropout(dropout)

        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B,T,H,W,C), got {tuple(x.shape)}")
        if x.shape[-1] != 1:
            raise ValueError(f"Expected channel-last grayscale input, got {tuple(x.shape)}")

        x = x.permute(0, 4, 1, 2, 3)
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))

        batch, channels, time_steps, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(batch, time_steps, -1)

        x, _ = self.gru1(x)
        x = self.drop4(F.relu(self.ln1(x)))
        x, _ = self.gru2(x)
        x = self.drop5(F.relu(self.ln2(x)))
        return self.fc(x)


def runtime_checkpoint_from_extracted(src: Path, dst: Path) -> Path:
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(
            f"GRID LipNet checkpoint folder was not found: {src}. "
            "Expected the extracted PyTorch folder best_model.pt/best_model."
        )
    newest_src = max(p.stat().st_mtime for p in src.rglob("*") if p.is_file())
    if dst.exists() and dst.stat().st_mtime >= newest_src:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    with ZipFile(tmp, "w", ZIP_DEFLATED) as zf:
        for path in src.rglob("*"):
            if path.is_file():
                zf.write(path, Path(src.name) / path.relative_to(src))
    tmp.replace(dst)
    return dst


def resolve_checkpoint(path: Path | None) -> Path:
    if path:
        candidate = path.expanduser().resolve()
        if candidate.is_dir():
            src = candidate / "best_model" if (candidate / "best_model").is_dir() else candidate
            return runtime_checkpoint_from_extracted(src, DEFAULT_RUNTIME_CHECKPOINT)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"GRID LipNet checkpoint was not found: {candidate}")
    return runtime_checkpoint_from_extracted(DEFAULT_EXTRACTED_CHECKPOINT, DEFAULT_RUNTIME_CHECKPOINT)


def write_crop_preview(frames: np.ndarray, path: Path) -> None:
    if frames.size == 0:
        return
    sample_count = min(10, len(frames))
    indices = [round(i) for i in np.linspace(0, len(frames) - 1, sample_count)]
    strip = np.concatenate([frames[i] for i in indices], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(path), strip)


def preprocess_video(video: Path, preview_path: Path | None = None) -> tuple[torch.Tensor, dict[str, object]]:
    crops = capture_frames_any(str(video))
    if preview_path is not None:
        write_crop_preview(crops.astype(np.uint8), preview_path)
    tensor_3d = reduce_frames(crops).float()
    tensor = tensor_3d.unsqueeze(-1).unsqueeze(0)
    return tensor, {
        "pipeline": "lyra_files.data_preprocessing.capture_frames_any -> reduce_frames",
        "raw_frames": int(crops.shape[0]),
        "selected_frames": int(crops.shape[0]),
        "crop_shape": list(crops.shape[1:]),
        "tensor_shape": list(tensor.shape),
        "crop_size": "100x46",
        "normalize": "zero_mean_unit_std",
        "vocab_source": "lyra_files.CONFIG.VOCAB",
    }


def grid_text_score(text: str) -> int:
    words = text.lower().split()
    score = 0
    score += 4 * sum(1 for word in words if word in GRID_WORDS)
    if len(words) == 6:
        score += 8
        slots = [GRID_COMMANDS, GRID_COLORS, GRID_PREPOSITIONS, GRID_LETTERS, GRID_DIGITS, GRID_ADVERBS]
        score += 6 * sum(1 for word, slot in zip(words, slots) if word in slot)
    score += min(text.count(" "), 6)
    score -= sum(1 for char in text if char in "'?!123456789")
    return score


def predict(args: argparse.Namespace) -> dict[str, object]:
    ckpt_path = resolve_checkpoint(args.ckpt)
    state = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state

    model = LipNet(vocab_size=VOCAB_SIZE)
    model.load_state_dict(state_dict, strict=True)
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model.to(device).eval()

    tensor, prep = preprocess_video(args.video, args.out / "preprocessed_preview.jpg")
    tensor = tensor.to(device)
    started = time.time()
    with torch.no_grad():
        logits = model(tensor)
        pred_ids = torch.argmax(logits, dim=-1).cpu()
    decoded = ctc_decode(pred_ids)
    raw_text = decoded[0].strip() if decoded else ""
    text = raw_text
    top = [{"label": raw_text, "profile": "lyra_training_decoder", "score": grid_text_score(raw_text)}]

    return {
        "text": text,
        "confidence": None,
        "model": "grid_lipnet",
        "raw_model_text": raw_text,
        "device": str(device),
        "frames": int(prep["raw_frames"]),
        "preprocessing": prep,
        "checkpoint_epoch": state.get("epoch") if isinstance(state, dict) else None,
        "checkpoint_val_loss": state.get("val_loss") if isinstance(state, dict) else None,
        "checkpoint_speaker": state.get("speaker") if isinstance(state, dict) else None,
        "runtime_ms": round((time.time() - started) * 1000, 1),
        "blank_index": 0,
        "vocab_profile": "lyra_training",
        "vocab": "".join(VOCAB[1:]),
        "top": top,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--check-only", action="store_true")

    # Compatibility arguments accepted by old wrappers. Lyra now always uses
    # the exact training preprocessing from lyra_files, so these do not alter
    # inference.
    parser.add_argument("--frames", type=int, default=75)
    parser.add_argument("--blank-index", type=int, default=0)
    parser.add_argument("--crop-mode", default="training")
    parser.add_argument("--frame-mode", default="training")
    parser.add_argument("--normalize", default="training")
    parser.add_argument("--resize-interp", default="linear")
    parser.add_argument("--vocab-mode", default="lyra_training")
    args = parser.parse_args()

    args.video = args.video.expanduser().resolve()
    args.out = (args.out or PROJECT_ROOT / "runs" / f"grid_lipnet_{args.video.stem}").resolve()
    if not args.video.exists():
        raise FileNotFoundError(f"Input video was not found: {args.video}")
    ckpt_path = resolve_checkpoint(args.ckpt)

    if args.check_only:
        print(f"[check] video          {args.video}")
        print(f"[check] checkpoint     {ckpt_path}")
        print(f"[check] out            {args.out}")
        print("[check] pipeline       lyra_files.capture_frames_any -> reduce_frames")
        print("[check] crop size      100x46")
        print(f"[check] vocab          {''.join(VOCAB[1:])}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    result = predict(args)
    (args.out / "hypotheses.txt").write_text(str(result["text"]) + "\n", encoding="utf-8")
    (args.out / "prediction.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("===== GRID LIPNET PREDICTION =====")
    print(result["text"] or "(empty prediction)")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Outputs saved to: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"[grid-lipnet] {exc}")
