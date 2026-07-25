"""Scan a video for face-present windows, split into short clips, and decode
each clip with decode_colab.py, printing the hypothesis as soon as each
clip finishes.

Usage (PowerShell):
  $env:Path = "E:\\claude\\tools;" + $env:Path
  $py = "C:\\Users\\LAPTOP WORLD\\miniconda3\\envs\\avhubert\\python.exe"
  & $py scripts\\split_and_decode.py ^
      --video E:\\claude\\test-video.mp4 ^
      --ckpt E:\\data\\ckpt\\base_vox_433h.pt ^
      --face-predictor E:\\data\\assets\\shape_predictor_68_face_landmarks.dat ^
      --mean-face E:\\data\\assets\\20words_mean_face.npy ^
      --avhubert-root E:\\claude\\av_hubert ^
      --out E:\\claude\\runs\\split_test
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from live_lipread import (  # noqa: E402
    PROJECT_ROOT,
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


def _probe_numeric_values(ffprobe: str, video: Path, args: list[str]) -> list[float]:
    try:
        raw = subprocess.check_output(
            [ffprobe, "-v", "error", *args, str(video)],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace")
    except (OSError, subprocess.CalledProcessError):
        return []

    values = []
    for line in raw.splitlines():
        token = line.strip().split(",")[-1].strip()
        if not token or token.upper() == "N/A":
            continue
        try:
            value = float(token)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return values


def probe_duration(video: Path, ffmpeg: str) -> float | None:
    # ffprobe is shipped alongside ffmpeg in the same folder.
    ffprobe = str(Path(ffmpeg).with_name("ffprobe.exe"))
    probes = [
        ["-show_entries", "format=duration", "-of", "default=nw=1:nk=1"],
        ["-select_streams", "v:0", "-show_entries", "stream=duration", "-of", "default=nw=1:nk=1"],
        ["-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "csv=p=0"],
    ]
    for probe_args in probes:
        values = _probe_numeric_values(ffprobe, video, probe_args)
        if values:
            return max(values)

    try:
        import cv2

        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps > 0 and frames > 0:
            return float(frames / fps)
    except Exception:
        pass

    return None


def scan_face_timeline(video: Path, stride: int = 5, detector: str = "fa",
                       face_predictor: Path | None = None,
                       device: str = "auto"):
    """Return (fps, total_frames, face_flags) where face_flags[i] is True if a
    face was detected in frame `i * stride`.
    """
    import cv2
    from tqdm import tqdm

    fa = None
    hog_detector = None
    if detector == "fa":
        import face_alignment
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
        detector_label = f"fa-{device}"
    else:
        import dlib
        hog_detector = dlib.get_frontal_face_detector()
        detector_label = "hog"

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    reported_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    progress_total = reported_total if 0 < reported_total < 1_000_000_000 else None

    flags = []
    idx = 0
    pbar = tqdm(total=progress_total, desc=f"[scan] faces {detector_label}", unit="f")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            if detector == "fa":
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preds = fa.get_landmarks(rgb)
                flags.append(bool(preds))
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flags.append(bool(hog_detector(gray, 1)))
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return fps, idx, flags, stride


def build_segments(fps: float, flags, stride: int,
                   min_s: float = 1.0, max_s: float = 6.0, pad_s: float = 0.2):
    """Collapse consecutive True flags into (start_s, end_s) segments and
    split any segment longer than `max_s` into ~max_s slices.
    """
    segs = []
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < n and flags[j]:
            j += 1
        start_s = max(0.0, (i * stride) / fps - pad_s)
        end_s = (j * stride) / fps + pad_s
        if end_s - start_s >= min_s:
            # split long runs
            t = start_s
            while end_s - t > max_s:
                segs.append((t, t + max_s))
                t += max_s
            if end_s - t >= min_s:
                segs.append((t, end_s))
        i = j
    return segs


def cut_clip(src: Path, dst: Path, start_s: float, dur_s: float, ffmpeg: str):
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_s:.3f}", "-i", str(src), "-t", f"{dur_s:.3f}",
        "-vf", "fps=25,format=yuv420p",
        "-c:v", "libx264", "-crf", "18", "-an", str(dst),
    ]
    subprocess.run(cmd, check=True)


def resolve_runtime_paths(args) -> None:
    candidates = _path_candidates()
    args.video = _resolve_path(
        "input video", args.video, "AVHUBERT_VIDEO",
        [PROJECT_ROOT / "test-video.mp4", PROJECT_ROOT / "new-video.mp4",
         PROJECT_ROOT / "Lip.mp4"],
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
        args.out = PROJECT_ROOT / "runs" / f"split_{args.video.stem}"
    else:
        args.out = args.out.resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path,
                    help="Input mp4. Defaults to $AVHUBERT_VIDEO or a sample in this repo.")
    ap.add_argument("--ckpt", type=Path,
                    help="AV-HuBERT checkpoint. Defaults to local data/ckpt.")
    ap.add_argument("--face-predictor", type=Path,
                    help="dlib 68-point predictor. Defaults to local data/assets.")
    ap.add_argument("--mean-face", type=Path,
                    help="20words_mean_face.npy. Defaults to local data/assets.")
    ap.add_argument("--avhubert-root", type=Path,
                    help="AV-HuBERT source tree. Defaults to this repo's av_hubert folder.")
    ap.add_argument("--out", type=Path,
                    help="Output directory. Defaults to runs/split_<video-name>.")
    local_ffmpeg = _first_existing(_path_candidates()["ffmpeg"])
    ap.add_argument("--ffmpeg", default=str(local_ffmpeg or "ffmpeg"))
    ap.add_argument("--detector", choices=["hog", "fa"], default="fa")
    ap.add_argument("--scan-device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--stride", type=int, default=5,
                    help="Scan every Nth frame for face presence.")
    ap.add_argument("--min-s", type=float, default=1.0)
    ap.add_argument("--max-s", type=float, default=6.0)
    ap.add_argument("--check-only", action="store_true",
                    help="Resolve local paths and exit without processing.")
    ap.add_argument("--stable", action="store_true",
                    help="Aurora stable mode: CPU/HOG face scan, stride 1, deterministic Torch settings.")
    args = ap.parse_args()
    args.stable = bool(args.stable or os.environ.get("HMO_STABLE_MODE") == "1")
    if args.stable:
        args.detector = "hog"
        args.scan_device = "cpu"
        args.stride = 1
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
        print(f"[check] av_hubert      {args.avhubert_root}")
        print(f"[check] ffmpeg         {args.ffmpeg}")
        print(f"[check] out            {args.out}")
        print(f"[check] stable         {'enabled' if args.stable else 'disabled'}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    clips_dir = args.out / "clips"
    clips_dir.mkdir(exist_ok=True)

    duration = probe_duration(args.video, args.ffmpeg)
    duration_text = f"{duration:.2f}s" if duration is not None else "unknown"
    print(f"[probe] {args.video.name}  duration={duration_text}")
    fps, total, flags, stride = scan_face_timeline(
        args.video, stride=args.stride, detector=args.detector,
        face_predictor=args.face_predictor, device=args.scan_device,
    )
    print(f"[scan] fps={fps:.2f} frames={total} sampled={len(flags)} "
          f"face_hits={sum(flags)}")

    segs = build_segments(fps, flags, stride,
                          min_s=args.min_s, max_s=args.max_s)
    if not segs:
        print("[split] no face segments found - aborting.")
        sys.exit(2)

    print(f"[split] {len(segs)} segment(s):")
    for k, (s, e) in enumerate(segs):
        print(f"   clip_{k:02d}  {s:7.2f}s - {e:7.2f}s  ({e-s:5.2f}s)")

    # cut all clips first
    clip_paths = []
    for k, (s, e) in enumerate(segs):
        dst = clips_dir / f"clip_{k:02d}.mp4"
        cut_clip(args.video, dst, s, e - s, args.ffmpeg)
        clip_paths.append((k, s, e, dst))

    # decode each clip sequentially, printing result right after finish
    decoder = Path(__file__).parent / "decode_colab.py"
    summary = []
    for k, s, e, clip in clip_paths:
        print("\n" + "=" * 70)
        print(f">>> clip_{k:02d}  [{s:.2f}s - {e:.2f}s]  {clip.name}")
        print("=" * 70)
        run_dir = args.out / f"clip_{k:02d}"
        cmd = [
            sys.executable, str(decoder),
            "--video", str(clip),
            "--ckpt", str(args.ckpt),
            "--face-predictor", str(args.face_predictor),
            "--mean-face", str(args.mean_face),
            "--avhubert-root", str(args.avhubert_root),
            "--detector", args.detector,
            "--device", args.scan_device,
            "--skip-resample",
            "--out", str(run_dir),
            "--ffmpeg", args.ffmpeg,
        ]
        if args.stable:
            cmd.append("--stable")
        rc = subprocess.call(cmd)
        hyp_file = run_dir / "hypotheses.txt"
        hyp = hyp_file.read_text(encoding="utf-8").strip() if hyp_file.exists() else "(no output)"
        print(f"\n>>> clip_{k:02d} result (rc={rc}):\n    {hyp}")
        summary.append((k, s, e, hyp))

    print("\n" + "#" * 70)
    print("# FINAL SUMMARY")
    print("#" * 70)
    for k, s, e, hyp in summary:
        print(f"clip_{k:02d}  {s:6.2f}-{e:6.2f}s : {hyp}")
    (args.out / "summary.txt").write_text(
        "\n".join(f"clip_{k:02d}\t{s:.2f}\t{e:.2f}\t{h}" for k, s, e, h in summary) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
