"""Decode a prerecorded video with the local LRW-AR word classifier.

The checkpoint stored at ``best_model.pth`` is a 23-class Arabic word-level
lip-reading model documented in the local project reports. In normal mode it
predicts one isolated Arabic word from a short video. In phrase mode it uses the
audio track only to split word-sized clips, classifies each clip, and joins the
predicted words.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
DEFAULT_LABELS = [
    "من",
    "في",
    "باسم",
    "بعد",
    "اليوم",
    "الدولة",
    "الذي",
    "تنظيم",
    "الرئيس",
    "السعودية",
    "علي",
    "صالح",
    "طائرات",
    "شخصا",
    "شمال",
    "الموجز",
    "عن",
    "عدد",
    "عليكم",
    "الحكومة",
    "قناة",
    "قطر",
    "قتل",
]


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path and path.exists():
            return path.resolve()
    return None


def ffmpeg_candidates() -> list[Path]:
    candidates = [
        PROJECT_ROOT / "tools" / "ffmpeg.exe",
        PROJECT_ROOT / "tools" / "ffmpeg-8.1-essentials_build" / "bin" / "ffmpeg.exe",
    ]
    found = shutil.which("ffmpeg")
    if found:
        candidates.append(Path(found))
    return candidates


def resolve_ffmpeg(value: Path | None) -> str:
    if value:
        candidate = value.expanduser().resolve()
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"ffmpeg was not found: {candidate}")
    local = first_existing(ffmpeg_candidates())
    return str(local or "ffmpeg")


def run_checked(cmd: list[str]) -> None:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-2000:] or f"Command failed: {' '.join(cmd)}")


def extract_audio(video: Path, wav_path: Path, ffmpeg: str, sample_rate: int = 16000) -> None:
    run_checked([
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(wav_path),
    ])


def read_wav_mono(wav_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise RuntimeError(f"Expected 16-bit WAV audio, got sample width {sample_width}")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio /= 32768.0
    return audio, rate


def parse_threshold(value: str, db_values: np.ndarray) -> float:
    text = str(value).strip().lower()
    finite = db_values[np.isfinite(db_values)]
    if finite.size == 0:
        return -35.0
    if text == "auto":
        peak = float(np.max(finite))
        noise = float(np.percentile(finite, 20))
        threshold = max(noise + 10.0, peak - 35.0, -45.0)
        return min(threshold, -18.0)
    return float(text)


def audio_activity_segments(
    video: Path,
    ffmpeg: str,
    work_dir: Path,
    threshold_db: str,
    window_ms: int,
    min_silence_ms: int,
    min_word_ms: int,
    pre_roll_ms: int,
    post_roll_ms: int,
) -> tuple[list[dict[str, float]], dict[str, float | int | str]]:
    wav_path = work_dir / "audio_mono_16k.wav"
    try:
        extract_audio(video, wav_path, ffmpeg)
    except RuntimeError as exc:
        return [], {"reason": "audio_extract_failed", "error": str(exc)[-500:]}
    audio, rate = read_wav_mono(wav_path)
    if audio.size == 0:
        return [], {"reason": "empty_audio", "sample_rate": rate}

    window = max(1, int(rate * window_ms / 1000))
    count = int(math.ceil(audio.size / window))
    rms_values: list[float] = []
    for i in range(count):
        chunk = audio[i * window:(i + 1) * window]
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        rms_values.append(20.0 * math.log10(max(rms, 1e-6)))
    db = np.array(rms_values, dtype=np.float32)
    threshold = parse_threshold(threshold_db, db)
    voiced = db >= threshold

    min_silence_windows = max(1, int(math.ceil(min_silence_ms / window_ms)))
    min_word_s = min_word_ms / 1000.0
    pre_s = pre_roll_ms / 1000.0
    post_s = post_roll_ms / 1000.0
    audio_duration = audio.size / float(rate)

    segments: list[dict[str, float]] = []
    active_start: int | None = None
    silence_start: int | None = None
    for i, is_voiced in enumerate(voiced.tolist()):
        if is_voiced:
            if active_start is None:
                active_start = i
            silence_start = None
            continue
        if active_start is None:
            continue
        if silence_start is None:
            silence_start = i
        if i - silence_start + 1 >= min_silence_windows:
            end_window = silence_start
            start_s = max(0.0, active_start * window_ms / 1000.0 - pre_s)
            end_s = min(audio_duration, end_window * window_ms / 1000.0 + post_s)
            if end_s - start_s >= min_word_s:
                segments.append({"start": round(start_s, 3), "end": round(end_s, 3)})
            active_start = None
            silence_start = None

    if active_start is not None:
        start_s = max(0.0, active_start * window_ms / 1000.0 - pre_s)
        end_s = min(audio_duration, len(voiced) * window_ms / 1000.0 + post_s)
        if end_s - start_s >= min_word_s:
            segments.append({"start": round(start_s, 3), "end": round(end_s, 3)})

    return segments, {
        "sample_rate": rate,
        "window_ms": window_ms,
        "threshold_db": round(threshold, 2),
        "peak_db": round(float(np.max(db)), 2) if db.size else -120.0,
        "min_silence_ms": min_silence_ms,
        "min_word_ms": min_word_ms,
        "pre_roll_ms": pre_roll_ms,
        "post_roll_ms": post_roll_ms,
        "duration_s": round(audio_duration, 3),
    }


def cut_video_segment(src: Path, dst: Path, start_s: float, end_s: float, ffmpeg: str) -> None:
    duration = max(0.05, end_s - start_s)
    run_checked([
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c",
        "copy",
        str(dst),
    ])


def cap_long_segments(segments: list[dict[str, float]], max_word_ms: int) -> list[dict[str, float]]:
    if max_word_ms <= 0:
        return segments
    max_s = max_word_ms / 1000.0
    capped: list[dict[str, float]] = []
    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        if end - start <= max_s:
            capped.append(segment)
            continue
        cursor = start
        while cursor < end:
            next_end = min(end, cursor + max_s)
            if next_end - cursor >= 0.05:
                capped.append({"start": round(cursor, 3), "end": round(next_end, 3)})
            cursor = next_end
    return capped


class SpatioTemporalEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv3d_1 = nn.Conv3d(1, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv3d_2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.relu(self.bn1(self.conv3d_1(x))))
        x = self.drop(self.relu(self.bn2(self.conv3d_2(x))))
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels: int = 576, reduction: int = 16) -> None:
        super().__init__()
        hidden = channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, channels = x.shape
        pooled = self.avg_pool(x.reshape(batch * frames, channels, 1)).reshape(batch * frames, channels)
        weights = self.fc(pooled).reshape(batch, frames, channels)
        return x * weights


class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights, dim=1)
        return context, weights


class ArabicLipreadingModel(nn.Module):
    def __init__(self, num_classes: int = 23) -> None:
        super().__init__()
        self.spatiotemporal_embed = SpatioTemporalEmbed()
        self.skip_projection = nn.Sequential(nn.Linear(64, 576))

        backbone = mobilenet_v3_small(weights=None)
        first_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            64,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        self.backbone = nn.Sequential(backbone.features)

        self.channel_attention = ChannelAttention(576)
        self.lstm = nn.LSTM(
            input_size=576,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_attention = TemporalAttention(512, 256)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1088, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, 96, 96)
        embedded = self.spatiotemporal_embed(x)
        skip = embedded.mean(dim=(-1, -2)).permute(0, 2, 1)
        skip = self.skip_projection(skip)

        bsz, channels, frames, height, width = embedded.shape
        frame_batch = embedded.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channels, height, width)
        features = self.backbone(frame_batch)
        features = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
        features = features.reshape(bsz, frames, -1)
        features = self.channel_attention(features) + skip

        lstm_out, _ = self.lstm(features)
        temporal_context, weights = self.temporal_attention(lstm_out)
        visual_context = torch.sum(features * weights, dim=1)
        return self.classifier(torch.cat([temporal_context, visual_context], dim=1))


def resolve_path(label: str, value: Path | None, candidates: list[Path]) -> Path:
    if value:
        candidate = value.expanduser().resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"{label} was not found: {candidate}")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"{label} was not found. Checked: {', '.join(map(str, candidates))}")


def fix_frame_count(mouth_frames: list[np.ndarray], target: int) -> list[np.ndarray]:
    if not mouth_frames:
        raise RuntimeError("No frames were decoded from the video.")
    total = len(mouth_frames)
    if total < target:
        return mouth_frames + [mouth_frames[-1]] * (target - total)
    if total > target:
        start = (total - target) // 2
        return mouth_frames[start:start + target]
    return mouth_frames


def select_notebook_frames(frames: list[np.ndarray], target: int) -> list[np.ndarray]:
    if not frames:
        raise RuntimeError("No frames were decoded from the video.")
    if len(frames) > target:
        start = (len(frames) - target) // 2
        return frames[start:start + target]
    return frames


def crop_mouth_notebook(frame: np.ndarray, detector: dlib.fhog_object_detector, predictor: dlib.shape_predictor) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if faces:
        face = faces[0]
        shape = predictor(gray, face)
        pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)])
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        pad = 15
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(gray.shape[1], x2 + pad)
        y2 = min(gray.shape[0], y2 + pad)
        crop = gray[y1:y2, x1:x2]
        if crop.size:
            return cv2.resize(crop, (96, 96))

    h, w = gray.shape
    fallback = gray[int(h * 0.55):int(h * 0.85), int(w * 0.25):int(w * 0.75)]
    if fallback.size == 0:
        fallback = gray
    return cv2.resize(fallback, (96, 96))


def preprocess_video(video: Path, predictor_path: Path, target_frames: int = 29) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(predictor_path))
    selected_frames = select_notebook_frames(frames, target_frames)
    mouth_frames_list = [crop_mouth_notebook(frame, detector, predictor).astype(np.float32) for frame in selected_frames]
    mouth_frames = np.stack(fix_frame_count(mouth_frames_list, target_frames), axis=0).astype(np.float32)
    std = float(mouth_frames.std())
    if std < 1e-6:
        std = 1.0
    mouth_frames = (mouth_frames - float(mouth_frames.mean())) / std
    tensor = torch.from_numpy(mouth_frames).unsqueeze(0).unsqueeze(0)
    return tensor


def load_model(args: argparse.Namespace) -> tuple[list[str], dict, ArabicLipreadingModel, torch.device]:
    labels = DEFAULT_LABELS
    ckpt = torch.load(str(args.ckpt), map_location="cpu")
    model = ArabicLipreadingModel(num_classes=len(labels))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model.to(device).eval()
    return labels, ckpt, model, device


def predict_clip(
    video: Path,
    face_predictor: Path,
    labels: list[str],
    model: ArabicLipreadingModel,
    device: torch.device,
    top_k: int,
) -> dict[str, object]:
    video_tensor = preprocess_video(video, face_predictor).to(device)
    with torch.no_grad():
        logits = model(video_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu()
    top_values, top_indices = torch.topk(probs, k=min(top_k, len(labels)))
    top = [
        {
            "label": labels[int(index)],
            "confidence": round(float(value) * 100.0, 2),
            "class_index": int(index),
        }
        for value, index in zip(top_values, top_indices)
    ]
    return {
        "text": top[0]["label"],
        "confidence": top[0]["confidence"],
        "top": top,
    }


def predict(args: argparse.Namespace) -> dict[str, object]:
    labels, ckpt, model, device = load_model(args)
    result = predict_clip(args.video, args.face_predictor, labels, model, device, args.top_k)
    return {
        **result,
        "model": "lrw_ar_word",
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_acc": ckpt.get("val_acc"),
        "device": str(device),
    }


def predict_phrase(args: argparse.Namespace) -> dict[str, object]:
    labels, ckpt, model, device = load_model(args)
    segment_dir = args.out / "audio_word_segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    segments, audio_info = audio_activity_segments(
        args.video,
        args.ffmpeg,
        segment_dir,
        args.audio_threshold_db,
        args.audio_window_ms,
        args.min_silence_ms,
        args.min_word_ms,
        args.pre_roll_ms,
        args.post_roll_ms,
    )
    original_segment_count = len(segments)
    segments = cap_long_segments(segments, args.max_word_ms)
    if original_segment_count != len(segments):
        audio_info["long_segments_capped"] = 1
        audio_info["original_segment_count"] = original_segment_count
        audio_info["capped_segment_count"] = len(segments)
        audio_info["max_word_ms"] = args.max_word_ms
    if not segments:
        audio_info["fallback"] = "no_audio_segments_detected"
        result = predict_clip(args.video, args.face_predictor, labels, model, device, args.top_k)
        return {
            **result,
            "top": [
                {
                    "label": result["text"],
                    "confidence": result["confidence"],
                    "segment_index": 0,
                    "start": None,
                    "end": None,
                }
            ],
            "segments": [
                {
                    "index": 0,
                    "start": None,
                    "end": None,
                    "duration": None,
                    "text": result["text"],
                    "confidence": result["confidence"],
                    "clip": str(args.video),
                    "top": result["top"],
                }
            ],
            "audio_segments": [],
            "audio_info": audio_info,
            "model": "lrw_ar_phrase_audio",
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_val_acc": ckpt.get("val_acc"),
            "device": str(device),
        }

    words: list[str] = []
    word_results: list[dict[str, object]] = []
    top_summary: list[dict[str, object]] = []
    confidences: list[float] = []
    for i, segment in enumerate(segments):
        clip_path = segment_dir / f"word_{i:02d}.mp4"
        start = float(segment["start"])
        end = float(segment["end"])
        if end <= start:
            continue
        cut_video_segment(args.video, clip_path, start, end, args.ffmpeg)
        result = predict_clip(clip_path, args.face_predictor, labels, model, device, args.top_k)
        word = str(result["text"])
        confidence = float(result["confidence"])
        words.append(word)
        confidences.append(confidence)
        entry = {
            "index": i,
            "start": start,
            "end": end,
            "duration": round(end - start, 3),
            "text": word,
            "confidence": confidence,
            "clip": str(clip_path),
            "top": result["top"],
        }
        word_results.append(entry)
        top_summary.append({
            "label": word,
            "confidence": confidence,
            "segment_index": i,
            "start": start,
            "end": end,
        })

    text = " ".join(words).strip()
    return {
        "text": text,
        "confidence": round(float(np.mean(confidences)), 2) if confidences else None,
        "top": top_summary,
        "segments": word_results,
        "audio_segments": segments,
        "audio_info": audio_info,
        "model": "lrw_ar_phrase_audio",
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_acc": ckpt.get("val_acc"),
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--face-predictor", type=Path, default=None)
    parser.add_argument("--ffmpeg", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--phrase-mode", action="store_true")
    parser.add_argument("--audio-threshold-db", default="auto")
    parser.add_argument("--audio-window-ms", type=int, default=30)
    parser.add_argument("--min-silence-ms", type=int, default=500)
    parser.add_argument("--min-word-ms", type=int, default=250)
    parser.add_argument("--max-word-ms", type=int, default=2200)
    parser.add_argument("--pre-roll-ms", type=int, default=120)
    parser.add_argument("--post-roll-ms", type=int, default=160)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    args.video = resolve_path("input video", args.video, [])
    args.ckpt = resolve_path("LRW-AR checkpoint", args.ckpt, [PROJECT_ROOT / "best_model.pth"])
    args.face_predictor = resolve_path(
        "dlib 68-point face predictor",
        args.face_predictor,
        [PROJECT_ROOT / "data" / "assets" / "shape_predictor_68_face_landmarks.dat"],
    )
    args.ffmpeg = resolve_ffmpeg(args.ffmpeg)
    args.out = (args.out or PROJECT_ROOT / "runs" / f"lrw_ar_{args.video.stem}").resolve()

    if args.check_only:
        print(f"[check] video          {args.video}")
        print(f"[check] checkpoint     {args.ckpt}")
        print(f"[check] face predictor {args.face_predictor}")
        print(f"[check] ffmpeg         {args.ffmpeg}")
        print(f"[check] phrase mode    {args.phrase_mode}")
        print(f"[check] out            {args.out}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    result = predict_phrase(args) if args.phrase_mode else predict(args)
    (args.out / "hypotheses.txt").write_text(str(result["text"]) + "\n", encoding="utf-8")
    (args.out / "prediction.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("===== LRW-AR PHRASE PREDICTION =====" if args.phrase_mode else "===== LRW-AR WORD PREDICTION =====")
    print(result["text"])
    if result.get("confidence") is not None:
        print(f"confidence: {result['confidence']}%")
    if args.phrase_mode:
        segments = result.get("segments") or []
        print(f"segments: {len(segments)}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Outputs saved to: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"[lrw-ar] {exc}")
