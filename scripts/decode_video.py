"""Decode one or more prerecorded videos with the local AV-HuBERT pipeline.

This is a compatibility entrypoint around ``decode_colab.py``. It keeps the
older ``--inputs`` interface, but uses the same path auto-detection and ROI
code as the working single-video decoder.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from decode_colab import (  # noqa: E402
    PROJECT_ROOT,
    _bootstrap_repo_paths,
    _first_existing,
    _path_candidates,
    _resolve_path,
    decode,
    ffmpeg_resample,
    preprocess_roi,
)


def _input_candidates() -> list[Path]:
    return [
        PROJECT_ROOT / "new-video.mp4",
        PROJECT_ROOT / "Lip.mp4",
        PROJECT_ROOT / "test-video.mp4",
    ]


def resolve_runtime_paths(args: argparse.Namespace) -> None:
    candidates = _path_candidates()
    raw_inputs = args.inputs or ([args.video] if args.video else [None])
    args.inputs = [
        _resolve_path("input video", src, "AVHUBERT_VIDEO", _input_candidates())
        for src in raw_inputs
    ]

    if args.assets_dir:
        assets = args.assets_dir.expanduser()
        candidates["face_predictor"].insert(
            0, assets / "shape_predictor_68_face_landmarks.dat",
        )
        candidates["mean_face"].insert(0, assets / "20words_mean_face.npy")

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
        stem = args.inputs[0].stem if len(args.inputs) == 1 else "batch"
        args.out = PROJECT_ROOT / "runs" / f"decode_{stem}"
    else:
        args.out = args.out.resolve()
    _bootstrap_repo_paths(args.avhubert_root)


def decode_one(video: Path, args: argparse.Namespace, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    resampled = out_dir / "resampled_25fps.mp4"
    roi = out_dir / "mouth_roi.mp4"

    if args.skip_resample:
        resampled = video
    else:
        ffmpeg_resample(video, resampled, args.ffmpeg)
    preprocess_roi(
        resampled,
        roi,
        args.face_predictor,
        args.mean_face,
        args.avhubert_root,
        args.cnn_detector,
        args.detector,
        args.ffmpeg,
        args.device,
    )
    return decode(roi, args.ckpt, args.avhubert_root, out_dir, args.spm_model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path,
                    help="Single input mp4. Defaults to $AVHUBERT_VIDEO or a sample in this repo.")
    ap.add_argument("--inputs", nargs="+", type=Path,
                    help="One or more input mp4 files.")
    ap.add_argument("--ckpt", type=Path,
                    help="AV-HuBERT checkpoint. Defaults to local data/ckpt.")
    ap.add_argument("--face-predictor", type=Path,
                    help="dlib 68-point predictor. Defaults to local data/assets.")
    ap.add_argument("--mean-face", type=Path,
                    help="20words_mean_face.npy. Defaults to local data/assets.")
    ap.add_argument("--spm-model", type=Path,
                    help="spm_unigram1000.model. Defaults to local data/assets.")
    ap.add_argument("--assets-dir", type=Path,
                    help="Compatibility option: directory containing predictor and mean-face assets.")
    ap.add_argument("--cnn-detector", type=Path, default=None,
                    help="Optional dlib mmod_human_face_detector.dat fallback.")
    ap.add_argument("--detector", choices=["hog", "fa"], default="hog",
                    help="Landmark detector: 'hog' = dlib HOG+68pt, 'fa' = face_alignment.")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="Device used by --detector fa. Auto uses CUDA when available.")
    ap.add_argument("--avhubert-root", type=Path,
                    help="AV-HuBERT source tree. Defaults to this repo's av_hubert folder.")
    ap.add_argument("--out", type=Path,
                    help="Output directory. Defaults to runs/decode_<video-name>.")
    local_ffmpeg = _first_existing(_path_candidates()["ffmpeg"])
    ap.add_argument("--ffmpeg", default=str(local_ffmpeg or shutil.which("ffmpeg") or "ffmpeg"))
    ap.add_argument("--skip-resample", action="store_true",
                    help="Assume the input video is already 25 fps yuv420p.")
    ap.add_argument("--check-only", action="store_true",
                    help="Resolve local paths and exit without decoding.")
    args = ap.parse_args()

    try:
        resolve_runtime_paths(args)
    except FileNotFoundError as exc:
        sys.exit(f"[config] {exc}")

    if args.check_only:
        for video in args.inputs:
            print(f"[check] video          {video}")
        print(f"[check] checkpoint     {args.ckpt}")
        print(f"[check] face predictor {args.face_predictor}")
        print(f"[check] mean face      {args.mean_face}")
        print(f"[check] spm model      {args.spm_model}")
        print(f"[check] av_hubert      {args.avhubert_root}")
        print(f"[check] ffmpeg         {args.ffmpeg}")
        print(f"[check] out            {args.out}")
        return

    summary = []
    for index, video in enumerate(args.inputs):
        out_dir = args.out if len(args.inputs) == 1 else args.out / f"{index:03d}_{video.stem}"
        print("\n" + "=" * 70)
        print(f">>> decoding {video}")
        print("=" * 70)
        text = decode_one(video, args, out_dir)
        summary.append((video, out_dir, text))
        print("\n===== PREDICTED TRANSCRIPT =====")
        print(text)
        print(f"\nOutputs saved to: {out_dir.resolve()}")

    if len(summary) > 1:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.txt").write_text(
            "\n".join(f"{video}\t{out_dir}\t{text}" for video, out_dir, text in summary) + "\n",
            encoding="utf-8",
        )
        print(f"\nBatch summary saved to: {(args.out / 'summary.txt').resolve()}")


if __name__ == "__main__":
    main()
