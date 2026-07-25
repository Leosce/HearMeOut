"""Colab-style standalone AV-HuBERT VSR decoder for arbitrary mp4 clips.

Requires only:
  - shape_predictor_68_face_landmarks.dat    (dlib HOG-based)
  - 20words_mean_face.npy                    (reference mean face)
  - a fine-tuned AV-HuBERT VSR checkpoint    (e.g. base_vox_433h.pt)

The checkpoint already contains the SentencePiece tokenizer config, so
no separate spm model file is needed.

Usage (PowerShell):
  $env:AVHUBERT_ROOT = "E:\\claude\\av_hubert"
  python scripts\\decode_colab.py ^
      --video "A1 English Listening Practice - Language Learning.mp4" ^
      --ckpt E:\\data\\ckpt\\base_vox_433h.pt ^
      --face-predictor E:\\data\\assets\\shape_predictor_68_face_landmarks.dat ^
      --mean-face E:\\data\\assets\\20words_mean_face.npy ^
      --avhubert-root E:\\claude\\av_hubert ^
      --out runs\\decode_a1_english
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from live_lipread import (  # noqa: E402
    Decoder,
    PROJECT_ROOT,
    _bootstrap_repo_paths,
    _first_existing,
    _landmarks_type_2d,
    _path_candidates,
    _resolve_path,
)


def configure_stable_runtime(enabled: bool) -> None:
    if not enabled:
        return
    os.environ["HMO_STABLE_MODE"] = "1"
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(0)
    try:
        import numpy as np

        np.random.seed(0)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[stable] warning: could not fully configure deterministic Torch mode: {exc}")


def ffmpeg_resample(src: Path, dst: Path, ffmpeg: str = "ffmpeg") -> None:
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vf", "fps=25,format=yuv420p",
        "-c:v", "libx264", "-crf", "18",
        "-an",
        str(dst),
    ]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def detect_landmarks_fa(video_path: Path, device: str = "auto"):
    """GPU 68-landmark detection via `face_alignment` (SFD detector + FAN)."""
    import cv2
    import face_alignment
    import numpy as np
    from tqdm import tqdm

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    landmarks_type = _landmarks_type_2d(face_alignment)
    if landmarks_type is None:
        raise RuntimeError("installed face_alignment has no 2D landmarks enum")
    fa = face_alignment.FaceAlignment(
        landmarks_type, device=device, flip_input=False,
    )

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    landmarks = []
    n_hit = n_miss = 0
    pbar = tqdm(total=total, desc="[fa-gpu] landmarks", unit="f")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = fa.get_landmarks(rgb)
        if not preds:
            landmarks.append(None)
            n_miss += 1
        else:
            best = max(preds, key=lambda p: np.ptp(p[:, 0]) * np.ptp(p[:, 1]))
            landmarks.append(best.astype(np.float32))
            n_hit += 1
        pbar.set_postfix(hit=n_hit, miss=n_miss)
        pbar.update(1)
    pbar.close()
    cap.release()
    print(f"[roi] fa(GPU) usage: hit={n_hit} miss={n_miss}")
    return landmarks


def detect_landmarks(video_path: Path, face_predictor_path: Path,
                     cnn_detector_path: Path | None = None):
    import cv2
    import dlib

    hog_detector = dlib.get_frontal_face_detector()
    cnn_detector = None
    if cnn_detector_path is not None and Path(cnn_detector_path).exists():
        cnn_detector = dlib.cnn_face_detection_model_v1(str(cnn_detector_path))
    predictor = dlib.shape_predictor(str(face_predictor_path))

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    landmarks = []
    n_hog = n_cnn = n_miss = 0
    from tqdm import tqdm
    pbar = tqdm(total=total, desc="[dlib] landmarks", unit="f")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = hog_detector(gray, 1)
        used = "hog"
        if len(rects) == 0 and cnn_detector is not None:
            mmod_rects = cnn_detector(frame, 1)
            rects = [r.rect for r in mmod_rects]
            used = "cnn"
        if len(rects) == 0:
            landmarks.append(None)
            n_miss += 1
        else:
            rect = max(rects, key=lambda r: r.width() * r.height())
            shape = predictor(gray, rect)
            pts = [[shape.part(i).x, shape.part(i).y] for i in range(68)]
            import numpy as np
            landmarks.append(np.array(pts, dtype=np.float32))
            if used == "hog":
                n_hog += 1
            else:
                n_cnn += 1
        pbar.set_postfix(hog=n_hog, cnn=n_cnn, miss=n_miss)
        pbar.update(1)
    pbar.close()
    cap.release()
    print(f"[roi] detector usage: hog={n_hog} cnn={n_cnn} miss={n_miss}")
    return landmarks


def preprocess_roi(video_path: Path, roi_path: Path, face_predictor_path: Path,
                   mean_face_path: Path, avhubert_root: Path,
                   cnn_detector_path: Path | None = None,
                   detector: str = "hog",
                   ffmpeg: str = "ffmpeg",
                   device: str = "auto") -> None:
    # Make av_hubert preparation module importable
    _bootstrap_repo_paths(avhubert_root)
    sys.path.insert(0, str(avhubert_root / "avhubert"))
    import numpy as np
    from preparation.align_mouth import (  # type: ignore
        landmarks_interpolate, crop_patch, write_video_ffmpeg,
    )

    print(f"[roi] detecting landmarks on {video_path.name} (detector={detector}) ...")
    if detector == "fa":
        landmarks = detect_landmarks_fa(video_path, device=device)
    else:
        landmarks = detect_landmarks(video_path, face_predictor_path, cnn_detector_path)
    n_total = len(landmarks)
    n_missing = sum(1 for l in landmarks if l is None)
    print(f"[roi] frames={n_total} missing={n_missing}")
    if n_total == 0:
        raise RuntimeError("no frames decoded from video")
    preprocessed = landmarks_interpolate(landmarks)
    if preprocessed is None:
        raise RuntimeError("all frames are missing faces — cannot align mouth ROI")

    mean_face = np.load(str(mean_face_path))
    # Parameters matching the colab demo / av_hubert preparation
    STD_SIZE = (256, 256)
    STABLE_POINTS = [33, 36, 39, 42, 45]
    WIN = 12
    CROP_H = 96
    CROP_W = 96
    START_IDX = 48
    STOP_IDX = 68

    rois = crop_patch(
        str(video_path), preprocessed, mean_face,
        STABLE_POINTS, STD_SIZE, WIN, START_IDX, STOP_IDX, CROP_H, CROP_W,
    )
    assert rois is not None, "crop_patch returned None"
    write_video_ffmpeg(rois, str(roi_path), ffmpeg=ffmpeg)
    print(f"[roi] wrote {roi_path} (shape {rois.shape})")


def decode(roi_path: Path, ckpt_path: Path, avhubert_root: Path,
           out_dir: Path, spm_model: Path) -> str:
    """Decode one ROI clip using the same robust model loader as live mode."""
    decoder = Decoder(ckpt_path, avhubert_root, spm_model)
    hyp_str = decoder.decode_video(roi_path).strip()
    print(f"[hyp] {hyp_str}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hypotheses.txt").write_text(hyp_str + "\n", encoding="utf-8")
    return hyp_str


def resolve_runtime_paths(args) -> None:
    candidates = _path_candidates()
    args.video = _resolve_path(
        "input video", args.video, "AVHUBERT_VIDEO",
        [PROJECT_ROOT / "new-video.mp4", PROJECT_ROOT / "Lip.mp4",
         PROJECT_ROOT / "test-video.mp4"],
    )
    args.ckpt = _resolve_path(
        "AV-HuBERT checkpoint", args.ckpt, "AVHUBERT_CKPT",
        candidates["ckpt"],
    )
    args.face_predictor = _resolve_path(
        "dlib 68-point face predictor", args.face_predictor,
        "AVHUBERT_FACE_PREDICTOR", candidates["face_predictor"],
    )
    args.mean_face = _resolve_path(
        "AV-HuBERT mean-face .npy", args.mean_face, "AVHUBERT_MEAN_FACE",
        candidates["mean_face"],
    )
    args.spm_model = _resolve_path(
        "AV-HuBERT SentencePiece model", args.spm_model,
        "AVHUBERT_SPM_MODEL", candidates["spm_model"],
    )
    args.avhubert_root = _resolve_path(
        "AV-HuBERT source root", args.avhubert_root, "AVHUBERT_ROOT",
        candidates["avhubert_root"],
    )
    ffmpeg_arg = Path(args.ffmpeg).expanduser() if args.ffmpeg else None
    ffmpeg_path = _resolve_path(
        "ffmpeg.exe", ffmpeg_arg, "FFMPEG",
        candidates["ffmpeg"], required=False,
    )
    args.ffmpeg = str(ffmpeg_path or args.ffmpeg or "ffmpeg")
    if args.out is None:
        args.out = PROJECT_ROOT / "runs" / f"decode_{args.video.stem}"
    else:
        args.out = args.out.resolve()
    _bootstrap_repo_paths(args.avhubert_root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path,
                    help="Input mp4. Defaults to $AVHUBERT_VIDEO or a sample in this repo.")
    ap.add_argument("--ckpt", type=Path,
                    help="AV-HuBERT checkpoint. Defaults to local data/ckpt.")
    ap.add_argument("--face-predictor", type=Path,
                    help="dlib 68-point predictor. Defaults to local data/assets.")
    ap.add_argument("--mean-face", type=Path,
                    help="20words_mean_face.npy. Defaults to local data/assets.")
    ap.add_argument("--spm-model", type=Path,
                    help="spm_unigram1000.model. Defaults to local data/assets.")
    ap.add_argument("--cnn-detector", type=Path, default=None,
                    help="Optional dlib mmod_human_face_detector.dat used as "
                         "fallback when the HOG frontal detector misses.")
    ap.add_argument("--detector", choices=["hog", "fa"], default="hog",
                    help="Landmark detector: 'hog' = dlib HOG+68pt (CPU), "
                         "'fa' = face_alignment SFD+FAN (GPU).")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="Device used by --detector fa. Auto uses CUDA when available.")
    ap.add_argument("--avhubert-root", type=Path,
                    help="AV-HuBERT source tree. Defaults to this repo's av_hubert folder.")
    ap.add_argument("--out", type=Path,
                    help="Output directory. Defaults to runs/decode_<video-name>.")
    local_ffmpeg = _first_existing(_path_candidates()["ffmpeg"])
    ap.add_argument("--ffmpeg", default=str(local_ffmpeg or shutil.which("ffmpeg") or "ffmpeg"))
    ap.add_argument("--skip-resample", action="store_true",
                    help="Assume video is already 25fps yuv420p")
    ap.add_argument("--check-only", action="store_true",
                    help="Resolve paths and exit without decoding.")
    ap.add_argument("--stable", action="store_true",
                    help="Aurora stable mode: CPU/HOG landmarking and deterministic Torch settings.")
    args = ap.parse_args()
    args.stable = bool(args.stable or os.environ.get("HMO_STABLE_MODE") == "1")
    if args.stable:
        args.detector = "hog"
        args.device = "cpu"
    configure_stable_runtime(args.stable)

    try:
        resolve_runtime_paths(args)
    except FileNotFoundError as exc:
        sys.exit(f"[config] {exc}")
    if args.check_only:
        print(f"[check] video          {args.video}")
        print(f"[check] checkpoint     {args.ckpt}")
        print(f"[check] face predictor {args.face_predictor}")
        print(f"[check] mean face      {args.mean_face}")
        print(f"[check] spm model      {args.spm_model}")
        print(f"[check] av_hubert      {args.avhubert_root}")
        print(f"[check] ffmpeg         {args.ffmpeg}")
        print(f"[check] out            {args.out}")
        print(f"[check] stable         {'enabled' if args.stable else 'disabled'}")
        return

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    resampled = out / "resampled_25fps.mp4"
    roi = out / "mouth_roi.mp4"

    if args.skip_resample:
        resampled = args.video
    else:
        ffmpeg_resample(args.video, resampled, args.ffmpeg)
    preprocess_roi(resampled, roi, args.face_predictor, args.mean_face,
                   args.avhubert_root, args.cnn_detector, args.detector,
                   args.ffmpeg, args.device)
    text = decode(roi, args.ckpt, args.avhubert_root, out, args.spm_model)
    print("\n===== PREDICTED TRANSCRIPT =====")
    print(text)
    print(f"\nOutputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
