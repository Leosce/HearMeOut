"""Live webcam lip reading with AV-HuBERT + TTS + Arabic translation + HCI report.

Features (all toggleable via CLI flags):
  1. Live real-time webcam lip reading (OpenCV)
  2. Text-to-speech of predicted words (MMS-TTS or pyttsx3 fallback)
  3. Arabic translation overlay (Helsinki-NLP/opus-mt-en-ar)
  4. HCI evaluation report (PDF + JSON via reportlab + matplotlib)

Controls:
  SPACE  force-decode the current buffer
  Q/ESC  quit and write report

Required new deps (install once):
    pip install transformers sentencepiece sounddevice pyttsx3 reportlab matplotlib Pillow
    pip install arabic-reshaper python-bidi   # optional, for proper Arabic RTL shaping

Run (PowerShell):
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\\run_live_lipread.ps1 -CheckOnly
    .\\run_live_lipread.ps1
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from argparse import Namespace
from collections import deque
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# HuggingFace tqdm writes to stderr; on some Windows consoles that handle is
# invalid (WinError 6) and crashes the import. Disable progress bars + advisory
# warnings BEFORE any transformers / huggingface_hub import.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")


def _ensure_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _bootstrap_repo_paths(avhubert_root: Path | None = None) -> None:
    """Make the vendored fairseq/AV-HuBERT trees importable from any cwd."""
    fairseq_root = PROJECT_ROOT / "fairseq"
    if (fairseq_root / "fairseq").exists():
        _ensure_path(fairseq_root)
    if avhubert_root and (avhubert_root / "avhubert").exists():
        _ensure_path(avhubert_root / "avhubert")


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        try:
            if candidate and candidate.exists():
                return candidate.resolve()
        except OSError:
            continue
    return None


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _resolve_path(label: str, current: Path | None, env_var: str,
                  candidates: list[Path], required: bool = True) -> Path | None:
    search: list[Path] = []
    if current is not None:
        search.append(current.expanduser())
    env_candidate = _env_path(env_var)
    if env_candidate is not None:
        search.append(env_candidate)
    search.extend(candidates)

    found = _first_existing(search)
    if found is not None:
        return found
    if not required:
        return None

    tried = "\n".join(f"  - {p}" for p in search)
    raise FileNotFoundError(
        f"Cannot find {label}.\n"
        f"Pass it explicitly, set ${env_var}, or put it in one of:\n{tried}"
    )


def _path_candidates() -> dict[str, list[Path]]:
    repo_data = PROJECT_ROOT / "data"
    sibling_claude = PROJECT_ROOT.parent / "claude" / "data"
    return {
        "ckpt": [
            repo_data / "ckpt" / "base_vox_433h.pt",
            sibling_claude / "ckpt" / "base_vox_433h.pt",
            Path("D:/data/ckpt/base_vox_433h.pt"),
            Path("E:/data/ckpt/base_vox_433h.pt"),
            Path("D:/claude/data/ckpt/base_vox_433h.pt"),
            Path("E:/claude/data/ckpt/base_vox_433h.pt"),
        ],
        "face_predictor": [
            repo_data / "assets" / "shape_predictor_68_face_landmarks.dat",
            sibling_claude / "assets" / "shape_predictor_68_face_landmarks.dat",
            Path("D:/data/assets/shape_predictor_68_face_landmarks.dat"),
            Path("E:/data/assets/shape_predictor_68_face_landmarks.dat"),
            Path("D:/claude/data/assets/shape_predictor_68_face_landmarks.dat"),
            Path("E:/claude/data/assets/shape_predictor_68_face_landmarks.dat"),
        ],
        "mean_face": [
            repo_data / "assets" / "20words_mean_face.npy",
            sibling_claude / "assets" / "20words_mean_face.npy",
            Path("D:/data/assets/20words_mean_face.npy"),
            Path("E:/data/assets/20words_mean_face.npy"),
            Path("D:/claude/data/assets/20words_mean_face.npy"),
            Path("E:/claude/data/assets/20words_mean_face.npy"),
        ],
        "spm_model": [
            repo_data / "assets" / "spm_unigram1000.model",
            sibling_claude / "assets" / "spm_unigram1000.model",
            Path("D:/data/assets/spm_unigram1000.model"),
            Path("E:/data/assets/spm_unigram1000.model"),
            Path("D:/claude/data/assets/spm_unigram1000.model"),
            Path("E:/claude/data/assets/spm_unigram1000.model"),
        ],
        "avhubert_root": [
            PROJECT_ROOT / "av_hubert",
            PROJECT_ROOT.parent / "claude" / "av_hubert",
            Path("D:/claude/av_hubert"),
            Path("E:/claude/av_hubert"),
        ],
        "ffmpeg": [
            PROJECT_ROOT / "tools" / "ffmpeg.exe",
            PROJECT_ROOT / "tools" / "ffmpeg-8.1-essentials_build" / "bin" / "ffmpeg.exe",
            PROJECT_ROOT.parent / "claude" / "tools" / "ffmpeg.exe",
            Path("D:/claude/tools/ffmpeg.exe"),
            Path("E:/claude/tools/ffmpeg.exe"),
        ],
    }


def _landmarks_type_2d(face_alignment_module):
    landmarks_type = face_alignment_module.LandmarksType
    return getattr(landmarks_type, "TWO_D",
                   getattr(landmarks_type, "_2D", None))


def _resolve_runtime_paths(args) -> None:
    candidates = _path_candidates()
    args.ckpt = _resolve_path(
        "AV-HuBERT checkpoint", args.ckpt, "AVHUBERT_CKPT",
        candidates["ckpt"],
    )
    args.face_predictor = _resolve_path(
        "dlib 68-point face predictor", args.face_predictor,
        "AVHUBERT_FACE_PREDICTOR", candidates["face_predictor"],
        required=False,
    )
    args.mean_face = _resolve_path(
        "AV-HuBERT mean-face .npy", args.mean_face, "AVHUBERT_MEAN_FACE",
        candidates["mean_face"],
    )
    args.spm_model = _resolve_path(
        "AV-HuBERT SentencePiece model", getattr(args, "spm_model", None),
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
    if ffmpeg_path is not None:
        args.ffmpeg = str(ffmpeg_path)
    _bootstrap_repo_paths(args.avhubert_root)


def _check_import(name: str, import_name: str | None = None,
                  required: bool = True) -> bool:
    module_name = import_name or name
    try:
        __import__(module_name)
        print(f"[check] ok      {name}")
        return True
    except Exception as exc:  # noqa: BLE001
        level = "missing" if required else "optional"
        print(f"[check] {level:<7} {name}: {exc}")
        return not required


def _check_runtime(args) -> bool:
    ok = True
    print(f"[check] project       {PROJECT_ROOT}")
    print(f"[check] python        {sys.executable}")
    for label, value in [
        ("checkpoint", args.ckpt),
        ("mean face", args.mean_face),
        ("spm model", args.spm_model),
        ("av_hubert", args.avhubert_root),
        ("ffmpeg", args.ffmpeg),
    ]:
        print(f"[check] {label:<12} {value}")
    if args.face_predictor:
        print(f"[check] face predictor {args.face_predictor}")

    for name in [
        "cv2", "numpy", "torch", "fairseq", "face_alignment",
        "sentencepiece", "skimage", "python_speech_features",
    ]:
        ok = _check_import(name) and ok
    if not args.no_report:
        for name in ["reportlab", "matplotlib", "PIL"]:
            ok = _check_import(name) and ok
    if not args.no_tts:
        tts_ok = (_check_import("transformers", required=False)
                  and _check_import("sounddevice", required=False))
        pyttsx_ok = _check_import("pyttsx3", required=False)
        if not (tts_ok or pyttsx_ok):
            print("[check] optional TTS has no usable backend; run with --no-tts")
    if not args.no_arabic:
        ok = _check_import("transformers") and ok
        _check_import("arabic_reshaper", required=False)
        _check_import("bidi", required=False)

    try:
        import numpy as np
        mean_face = np.load(str(args.mean_face))
        if mean_face.shape != (68, 2):
            ok = False
            print(f"[check] bad     mean-face shape {mean_face.shape}, expected (68, 2)")
        else:
            print("[check] ok      mean-face shape (68, 2)")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[check] missing mean-face load failed: {exc}")

    try:
        subprocess.run(
            [str(args.ffmpeg), "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        print("[check] ok      ffmpeg executable")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[check] missing ffmpeg executable: {exc}")

    align_mouth = args.avhubert_root / "avhubert" / "preparation" / "align_mouth.py"
    if align_mouth.exists():
        print("[check] ok      AV-HuBERT align_mouth.py")
    else:
        ok = False
        print(f"[check] missing {align_mouth}")

    return ok


# ---------------------------------------------------------------------------
# AV-HuBERT decoder wrapper (model loaded ONCE)
# ---------------------------------------------------------------------------
class Decoder:
    """Holds an AV-HuBERT model + generator + task in memory and decodes
    any ROI mp4 fed to it via `decode_video(roi_mp4)`.
    """

    def __init__(self, ckpt_path: Path, avhubert_root: Path,
                 spm_model: Path | None = None):
        _bootstrap_repo_paths(avhubert_root)
        import torch
        import torch.nn as nn
        from fairseq import tasks, utils as fs_utils
        from fairseq.dataclass.utils import convert_namespace_to_omegaconf
        from fairseq.dataclass.configs import GenerationConfig
        from omegaconf import OmegaConf

        user_dir = str(avhubert_root / "avhubert")
        _ensure_path(Path(user_dir))
        fs_utils.import_user_module(Namespace(user_dir=user_dir))

        # placeholder dir; we will overwrite the tsv every decode
        self._data_dir = Path(tempfile.mkdtemp(prefix="avh_live_"))
        self._spm_model = spm_model
        self._write_manifest(n_frames=25, roi_mp4=None)
        if spm_model is not None:
            shutil.copy2(str(spm_model), str(self._data_dir / spm_model.name))
        spm_for_task = self._data_dir / (spm_model.name if spm_model else "spm_unigram1000.model")

        print(f"[decode] loading checkpoint {ckpt_path} ...")
        try:
            state = torch.load(str(ckpt_path), map_location="cpu",
                               weights_only=False)
        except TypeError:
            state = torch.load(str(ckpt_path), map_location="cpu")
        saved_cfg = state.get("cfg")
        if saved_cfg is None:
            saved_cfg = convert_namespace_to_omegaconf(state["args"])
        elif isinstance(saved_cfg, dict):
            saved_cfg = OmegaConf.create(saved_cfg)

        OmegaConf.set_struct(saved_cfg, False)
        saved_cfg.task.data = str(self._data_dir)
        saved_cfg.task.label_dir = str(self._data_dir)
        saved_cfg.task.modalities = ["video"]
        saved_cfg.task.max_sample_size = 10_000
        saved_cfg.task.min_sample_size = 0
        saved_cfg.task.tokenizer_bpe_model = str(spm_for_task)
        try:
            w2v_args = saved_cfg.model.w2v_args
            if w2v_args is not None:
                OmegaConf.set_struct(w2v_args, False)
                w2v_args.task.data = str(self._data_dir)
                w2v_args.task.label_dir = str(self._data_dir)
                w2v_args.task.tokenizer_bpe_model = str(spm_for_task)
        except Exception:
            pass

        task = tasks.setup_task(saved_cfg.task, from_checkpoint=True)
        model = task.build_model(saved_cfg.model)
        ckpt_weights = state["model"]

        # Newer local Fairseq can build a few modules with shapes that differ
        # from this checkpoint. Match the modules to the checkpoint before load.
        ckpt_weights.pop("encoder.w2v_model.label_embs_concat", None)
        tensors = {**dict(model.named_parameters()), **dict(model.named_buffers())}
        mismatched = {
            name for name, value in ckpt_weights.items()
            if name in tensors and value.shape != tensors[name].shape
        }
        if mismatched:
            print(f"[decode] resizing {len(mismatched)} checkpoint-shaped module(s)")
        for param_name in sorted(mismatched):
            ckpt_shape = tuple(ckpt_weights[param_name].shape)
            parts = param_name.split(".")
            parent = model
            for part in parts[:-2]:
                parent = getattr(parent, part)
            child_name = parts[-2]
            attr_name = parts[-1]
            child = getattr(parent, child_name)
            if isinstance(child, nn.Linear) and attr_name == "weight":
                out_features, in_features = ckpt_shape
                setattr(parent, child_name,
                        nn.Linear(in_features, out_features,
                                  bias=child.bias is not None))
            elif isinstance(child, nn.Embedding) and attr_name == "weight":
                num_embeddings, embedding_dim = ckpt_shape
                setattr(parent, child_name,
                        nn.Embedding(num_embeddings, embedding_dim,
                                     padding_idx=child.padding_idx))
            else:
                try:
                    setattr(parent, child_name,
                            nn.Parameter(torch.zeros(ckpt_shape),
                                         requires_grad=True))
                except Exception as exc:  # noqa: BLE001
                    print(f"[decode] warning: could not resize {param_name}: {exc}")

        missing, unexpected = model.load_state_dict(ckpt_weights, strict=False)
        missing = [k for k in missing if "label_embs_concat" not in k]
        unexpected = [k for k in unexpected if "label_embs_concat" not in k]
        if missing:
            print(f"[decode] missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"[decode] unexpected keys ({len(unexpected)}): {unexpected[:5]}")

        self._torch = torch
        self._fs_utils = fs_utils
        self._model = model
        self._use_cuda = torch.cuda.is_available() and os.environ.get("HMO_STABLE_MODE") != "1"
        if self._use_cuda:
            self._model = self._model.cuda().eval()
        else:
            self._model = self._model.eval()
        self._saved_cfg = saved_cfg
        self._task = task
        self._models = [self._model]

        gen_cfg = GenerationConfig(beam=20, lenpen=1.0)
        self._generator = task.build_generator(self._models, gen_cfg)
        self._model_name = Path(ckpt_path).name
        print(f"[decode] ready ({self._model_name})")

    def _write_manifest(self, n_frames: int, roi_mp4: Path | None) -> None:
        roi_text = str(roi_mp4.resolve()) if roi_mp4 else "/"
        n_audio = int(16000 * n_frames / 25)
        (self._data_dir / "test.tsv").write_text(
            "/\n" +
            f"live-0\t{roi_text}\tNone\t{n_frames}\t{n_audio}\n",
            encoding="utf-8",
        )
        (self._data_dir / "test.wrd").write_text("DUMMY\n", encoding="utf-8")
        (self._data_dir / "test.km").write_text(
            " ".join(["0"] * max(1, n_frames)) + "\n",
            encoding="utf-8",
        )
        if not (self._data_dir / "dict.wrd.txt").exists():
            if self._spm_model is not None and self._spm_model.exists():
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                sp.Load(str(self._spm_model))
                pieces = [sp.IdToPiece(i) for i in range(4, sp.GetPieceSize())]
                (self._data_dir / "dict.wrd.txt").write_text(
                    "\n".join(f"{piece} 1" for piece in pieces) + "\n",
                    encoding="utf-8",
                )
            else:
                (self._data_dir / "dict.wrd.txt").write_text(
                    "\n".join(f"{i} 1" for i in range(996)) + "\n",
                    encoding="utf-8",
                )
        if not (self._data_dir / "dict.km.txt").exists():
            (self._data_dir / "dict.km.txt").write_text(
                "\n".join(f"{i} 1" for i in range(2000)) + "\n",
                encoding="utf-8",
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    def decode_video(self, roi_mp4: Path) -> str:
        import cv2
        num_frames = int(cv2.VideoCapture(str(roi_mp4)).get(cv2.CAP_PROP_FRAME_COUNT))
        if num_frames < 5:
            return ""
        self._write_manifest(n_frames=num_frames, roi_mp4=roi_mp4)
        # Rebuild dataset for the new ROI file
        self._task.load_dataset("test", task_cfg=self._saved_cfg.task)
        dataset = self._task.dataset("test")
        sample = dataset.collater([dataset[0]])
        if self._use_cuda:
            sample = self._fs_utils.move_to_cuda(sample)

        with self._torch.no_grad():
            hypos = self._task.inference_step(self._generator, self._models, sample)
        best = hypos[0][0]
        tokens = best["tokens"].int().cpu()
        return dataset.label_processors[0].decode(tokens).strip()


# ---------------------------------------------------------------------------
# Mouth ROI from buffered (frame, landmarks) pairs — reuses av_hubert helpers
# ---------------------------------------------------------------------------
def buffer_to_roi_mp4(frames_bgr, landmarks, mean_face, roi_path: Path,
                      tmp_clip_path: Path, avhubert_root: Path,
                      ffmpeg: str = "ffmpeg") -> int:
    """Write buffered frames+landmarks to a mouth-ROI mp4 (96x96 RGB, 25fps).
    Returns number of ROI frames; 0 if ROI extraction failed.
    """
    import cv2
    import numpy as np

    _bootstrap_repo_paths(avhubert_root)
    from preparation.align_mouth import (  # type: ignore
        landmarks_interpolate, crop_patch, write_video_ffmpeg,
    )

    if not frames_bgr:
        return 0

    # Write the raw clip (25 fps, libx264, no audio) — required by crop_patch
    h, w = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(
        str(tmp_clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h),
    )
    for f in frames_bgr:
        writer.write(f)
    writer.release()

    preprocessed = landmarks_interpolate(landmarks)
    if preprocessed is None:
        return 0

    STD_SIZE = (256, 256)
    STABLE_POINTS = [33, 36, 39, 42, 45]
    WIN = 12
    CROP_H = CROP_W = 96
    START_IDX, STOP_IDX = 48, 68
    try:
        rois = crop_patch(
            str(tmp_clip_path), preprocessed, mean_face,
            STABLE_POINTS, STD_SIZE, WIN, START_IDX, STOP_IDX, CROP_H, CROP_W,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[roi] crop_patch failed: {exc}")
        return 0
    if rois is None or len(rois) == 0:
        return 0
    write_video_ffmpeg(rois, str(roi_path), ffmpeg=ffmpeg)
    return int(len(rois))


def _best_face_landmarks(preds):
    if not preds:
        return None
    import numpy as np
    best = max(preds, key=lambda p: np.ptp(p[:, 0]) * np.ptp(p[:, 1]))
    return best.astype(np.float32)


class CameraStream:
    """Continuously read the webcam so expensive model work cannot starve it."""

    def __init__(self, cap, target_fps: float = 25.0,
                 history_seconds: float = 10.0):
        self._cap = cap
        self._target_fps = max(1.0, float(target_fps))
        # Keep enough raw frames for the longest utterance plus pre-roll.
        maxlen = int(max(90, history_seconds * max(self._target_fps, 30.0) * 2))
        self._history = deque(maxlen=maxlen)
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            ts = time.perf_counter()
            with self._lock:
                self._latest = (ts, frame)
                self._history.append((ts, frame.copy()))

    def latest(self):
        with self._lock:
            if self._latest is None:
                return None
            ts, frame = self._latest
            return ts, frame.copy()

    def latest_ts(self) -> float | None:
        with self._lock:
            return self._latest[0] if self._latest is not None else None

    def frames_between(self, start_ts: float, end_ts: float,
                       target_fps: float, max_frames: int | None = None) -> list:
        if end_ts < start_ts:
            start_ts, end_ts = end_ts, start_ts
        with self._lock:
            items = [(ts, frame) for ts, frame in self._history
                     if start_ts <= ts <= end_ts]
        if not items:
            return []
        period = 1.0 / max(1.0, float(target_fps))
        wanted = max(1, int((end_ts - start_ts) / period) + 1)
        if max_frames is not None:
            wanted = min(wanted, max_frames)
        selected = []
        idx = 0
        for n in range(wanted):
            target_ts = start_ts + n * period
            while (idx + 1 < len(items)
                   and abs(items[idx + 1][0] - target_ts)
                   <= abs(items[idx][0] - target_ts)):
                idx += 1
            selected.append(items[idx][1].copy())
        return selected


class LiveLandmarkWorker(threading.Thread):
    """Runs face_alignment away from the UI loop; drops stale requests."""

    def __init__(self, fa, fa_lock: threading.Lock):
        super().__init__(daemon=True)
        self._fa = fa
        self._fa_lock = fa_lock
        self._jobs = queue.Queue(maxsize=1)
        self.results = queue.Queue()
        self._busy = threading.Event()
        self._stop = threading.Event()

    def submit(self, ts: float, frame_bgr) -> bool:
        if self._busy.is_set():
            return False
        self._busy.set()
        try:
            self._jobs.put_nowait((ts, frame_bgr.copy()))
            return True
        except queue.Full:
            self._busy.clear()
            return False

    def run(self) -> None:
        import cv2
        while not self._stop.is_set():
            try:
                ts, frame_bgr = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                with self._fa_lock:
                    preds = self._fa.get_landmarks(rgb)
                self.results.put((ts, _best_face_landmarks(preds)))
            except Exception as exc:  # noqa: BLE001
                self.results.put((ts, None, exc))
            finally:
                self._busy.clear()

    def stop(self) -> None:
        self._stop.set()


def detect_landmarks_for_frames(frames_bgr, fa, fa_lock: threading.Lock,
                                log=None):
    import cv2
    landmarks = []
    hits = 0
    misses = 0
    with fa_lock:
        for i, frame in enumerate(frames_bgr, start=1):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lm = _best_face_landmarks(fa.get_landmarks(rgb))
            landmarks.append(lm)
            if lm is None:
                misses += 1
            else:
                hits += 1
            if log and (i == len(frames_bgr) or i % 25 == 0):
                log("roi", f"landmarks {i}/{len(frames_bgr)} "
                           f"hit={hits} miss={misses}")
    return landmarks, hits, misses


# ---------------------------------------------------------------------------
# TTS — prefers MMS-TTS, falls back to pyttsx3
# ---------------------------------------------------------------------------
class TTSEngine:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._kind = None
        self._mms_model = None
        self._mms_tokenizer = None
        self._pyttsx = None
        self._sd = None
        self._lock = threading.Lock()
        if not enabled:
            return
        # Try MMS-TTS
        try:
            from transformers import VitsModel, AutoTokenizer
            import sounddevice as sd
            import torch  # noqa: F401
            print("[tts] loading facebook/mms-tts-eng from local cache ...")
            self._mms_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng", local_files_only=True)
            self._mms_model = VitsModel.from_pretrained("facebook/mms-tts-eng", local_files_only=True)
            self._sd = sd
            self._kind = "mms"
            print("[tts] mms-tts ready")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[tts] mms-tts unavailable ({exc}); falling back to pyttsx3")
        # Fallback
        try:
            import pyttsx3
            self._pyttsx = pyttsx3.init()
            self._kind = "pyttsx3"
            print("[tts] pyttsx3 ready")
        except Exception as exc:  # noqa: BLE001
            print(f"[tts] pyttsx3 unavailable ({exc}); TTS disabled")
            self.enabled = False

    def speak_async(self, text: str):
        if not self.enabled or not text.strip():
            return
        threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _speak(self, text: str):
        try:
            with self._lock:
                if self._kind == "mms":
                    import torch
                    inputs = self._mms_tokenizer(text, return_tensors="pt")
                    with torch.no_grad():
                        out = self._mms_model(**inputs).waveform
                    audio = out.squeeze().cpu().numpy()
                    sr = self._mms_model.config.sampling_rate
                    self._sd.play(audio, sr)
                    self._sd.wait()
                elif self._kind == "pyttsx3":
                    self._pyttsx.say(text)
                    self._pyttsx.runAndWait()
        except Exception as exc:  # noqa: BLE001
            print(f"[tts] playback error: {exc}")


# ---------------------------------------------------------------------------
# Arabic translator
# ---------------------------------------------------------------------------
class Translator:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._tokenizer = None
        self._model = None
        self._cache: dict[str, str] = {}
        if not enabled:
            return
        try:
            from transformers import MarianMTModel, MarianTokenizer
            import torch
            name = "Helsinki-NLP/opus-mt-en-ar"
            print(f"[trans] loading {name} from local cache ...")
            self._tokenizer = MarianTokenizer.from_pretrained(name, local_files_only=True)
            self._model = MarianMTModel.from_pretrained(name, local_files_only=True)
            if torch.cuda.is_available():
                self._model = self._model.cuda()
            self._model.eval()
            self._torch = torch
            print("[trans] ready")
        except Exception as exc:  # noqa: BLE001
            print(f"[trans] unavailable ({exc}); translation disabled")
            self.enabled = False

    def translate(self, text: str) -> str:
        if not self.enabled or not text.strip():
            return ""
        if text in self._cache:
            return self._cache[text]
        try:
            inputs = self._tokenizer([text], return_tensors="pt", padding=True)
            if self._torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with self._torch.no_grad():
                out = self._model.generate(**inputs, max_length=128, num_beams=4)
            ar = self._tokenizer.decode(out[0], skip_special_tokens=True)
            self._cache[text] = ar
            return ar
        except Exception as exc:  # noqa: BLE001
            print(f"[trans] error: {exc}")
            return ""


def reshape_arabic(text: str) -> str:
    """Apply arabic-reshaper + python-bidi for correct RTL rendering."""
    if not text:
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


# ---------------------------------------------------------------------------
# Frame overlay (English via OpenCV, Arabic via PIL)
# ---------------------------------------------------------------------------
class Overlay:
    def __init__(self, arabic_font: Path, enabled_arabic: bool):
        self._enabled_arabic = enabled_arabic
        self._font = None
        if enabled_arabic:
            try:
                from PIL import ImageFont
                self._font = ImageFont.truetype(str(arabic_font), 32)
            except Exception as exc:  # noqa: BLE001
                print(f"[overlay] cannot load font {arabic_font}: {exc}")
                self._enabled_arabic = False

    @staticmethod
    def _wrap_cv2(text: str, max_width: int, font, scale: float,
                  thickness: int, max_lines: int = 2) -> list[str]:
        import cv2
        if not text:
            return []
        lines: list[str] = []
        cur = ""
        for word in text.split():
            trial = f"{cur} {word}".strip()
            width = cv2.getTextSize(trial, font, scale, thickness)[0][0]
            if cur and width > max_width:
                lines.append(cur)
                cur = word
                if len(lines) >= max_lines:
                    break
            else:
                cur = trial
        if cur and len(lines) < max_lines:
            lines.append(cur)
        return lines[:max_lines]

    def draw(self, frame_bgr, english: str, arabic: str, status: str,
             roi_thumb=None):
        import cv2
        import numpy as np

        h, w = frame_bgr.shape[:2]
        # Status (top-left)
        cv2.putText(frame_bgr, status, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        eng_lines = self._wrap_cv2(english, max(80, w - 28), font, 0.72, 2)
        shaped = ""
        ar_size = (0, 0)
        if self._enabled_arabic and arabic and self._font is not None:
            from PIL import Image, ImageDraw
            shaped = reshape_arabic(arabic)
            try:
                dummy = Image.new("RGB", (8, 8))
                draw_dummy = ImageDraw.Draw(dummy)
                bbox = draw_dummy.textbbox((0, 0), shaped, font=self._font)
                ar_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            except Exception:
                ar_size = (min(420, w - 32), 32)

        if eng_lines or shaped:
            eng_h = 28 * len(eng_lines)
            ar_h = ar_size[1] + 12 if shaped else 0
            panel_h = max(48, 18 + eng_h + ar_h)
            y0 = max(36, h - panel_h)
            panel = frame_bgr.copy()
            cv2.rectangle(panel, (0, y0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(panel, 0.62, frame_bgr, 0.38, 0, frame_bgr)
            y = y0 + 26
            for line in eng_lines:
                cv2.putText(frame_bgr, line, (12, y), font, 0.72,
                            (50, 220, 255), 2, cv2.LINE_AA)
                y += 28

        # Arabic via PIL (RTL), below the English rows.
        if shaped:
            from PIL import Image, ImageDraw
            pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            x = max(12, w - ar_size[0] - 12)
            y = h - ar_size[1] - 12
            draw.text((x, y), shaped, font=self._font, fill=(255, 255, 255))
            frame_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        # ROI thumbnail (top-right)
        if roi_thumb is not None:
            th_h, th_w = roi_thumb.shape[:2]
            scale = 1
            thumb = cv2.resize(roi_thumb, (th_w * scale, th_h * scale))
            if thumb.ndim == 2:
                thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
            frame_bgr[10:10 + thumb.shape[0],
                      w - 10 - thumb.shape[1]:w - 10] = thumb
        return frame_bgr


# ---------------------------------------------------------------------------
# HCI report
# ---------------------------------------------------------------------------
class HCIReport:
    def __init__(self, model_name: str):
        self.start = time.time()
        self.events: list[dict] = []
        self.attempts = 0
        self.successes = 0
        self.tts_count = 0
        self.latencies_ms: list[float] = []
        self.frames_per_decode: list[int] = []
        self.model_name = model_name

    def log_decode(self, english: str, arabic: str, latency_ms: float,
                   n_frames: int, success: bool):
        self.attempts += 1
        if success:
            self.successes += 1
            self.latencies_ms.append(latency_ms)
            self.frames_per_decode.append(n_frames)
        self.events.append({
            "t_rel": round(time.time() - self.start, 3),
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "english": english,
            "arabic": arabic,
            "latency_ms": round(latency_ms, 1),
            "frames": n_frames,
            "success": success,
        })

    def note_tts(self):
        self.tts_count += 1

    def _system_specs(self) -> dict:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(no CUDA)"
        return {
            "os": platform.platform(),
            "python": platform.python_version(),
            "gpu": gpu,
            "torch": torch.__version__,
            "model": self.model_name,
        }

    def save(self, pdf_path: Path, json_path: Path):
        duration = time.time() - self.start
        specs = self._system_specs()
        summary = {
            "duration_s": round(duration, 2),
            "attempts": self.attempts,
            "successes": self.successes,
            "tts_count": self.tts_count,
            "avg_latency_ms": round(sum(self.latencies_ms) / len(self.latencies_ms), 1)
                              if self.latencies_ms else 0.0,
            "avg_frames_per_decode": round(sum(self.frames_per_decode) /
                                           len(self.frames_per_decode), 1)
                                     if self.frames_per_decode else 0.0,
        }
        # JSON
        json_path.write_text(json.dumps({
            "specs": specs, "summary": summary, "events": self.events,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[report] json -> {json_path}")
        # PDF
        self._render_pdf(pdf_path, specs, summary)
        print(f"[report] pdf  -> {pdf_path}")

    def _render_pdf(self, pdf_path: Path, specs: dict, summary: dict):
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, Image)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                                title="Live Lip Reading System")
        story = []
        story.append(Paragraph("<b>Live Lip Reading System</b>", styles["Title"]))
        story.append(Paragraph(
            f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
            styles["Normal"]))
        story.append(Spacer(1, 12))
        # Specs
        story.append(Paragraph("<b>System</b>", styles["Heading2"]))
        spec_rows = [[k, str(v)] for k, v in specs.items()]
        t = Table([["Field", "Value"]] + spec_rows, hAlign="LEFT",
                  colWidths=[120, 360])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))
        # Summary
        story.append(Paragraph("<b>Session metrics</b>", styles["Heading2"]))
        sm_rows = [[k, str(v)] for k, v in summary.items()]
        t = Table([["Metric", "Value"]] + sm_rows, hAlign="LEFT",
                  colWidths=[200, 280])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))
        # Latency chart
        if self.latencies_ms:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.bar(range(1, len(self.latencies_ms) + 1), self.latencies_ms,
                   color="#4477aa")
            ax.set_xlabel("Decode #")
            ax.set_ylabel("Latency (ms)")
            ax.set_title("Per-decode latency")
            ax.grid(True, axis="y", alpha=0.3)
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=140)
            plt.close(fig)
            buf.seek(0)
            story.append(Image(buf, width=460, height=200))
            story.append(Spacer(1, 12))
        # Transcript table
        story.append(Paragraph("<b>Hypotheses transcript</b>", styles["Heading2"]))
        # Use a Unicode font so Arabic renders
        try:
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            arial_path = Path("C:/Windows/Fonts/arial.ttf")
            if arial_path.exists():
                pdfmetrics.registerFont(TTFont("Arial", str(arial_path)))
                arabic_font_name = "Arial"
            else:
                arabic_font_name = "Helvetica"
        except Exception:
            arabic_font_name = "Helvetica"

        rows = [["t (s)", "English", "Arabic", "lat (ms)", "frames"]]
        for ev in self.events:
            rows.append([
                f"{ev['t_rel']:.1f}",
                ev["english"] or "—",
                ev["arabic"] or "—",
                f"{ev['latency_ms']:.0f}",
                str(ev["frames"]),
            ])
        t = Table(rows, hAlign="LEFT",
                  colWidths=[45, 200, 180, 50, 45], repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (2, 1), (2, -1), arabic_font_name),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t)
        doc.build(story)


# ---------------------------------------------------------------------------
# Main live loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    help="AV-HuBERT checkpoint. Defaults to local machine paths.")
    ap.add_argument("--face-predictor", type=Path,
                    help="Optional dlib 68-point predictor; kept for parity with decode_colab.")
    ap.add_argument("--mean-face", type=Path,
                    help="20words_mean_face.npy. Defaults to local machine paths.")
    ap.add_argument("--spm-model", type=Path,
                    help="spm_unigram1000.model. Defaults to local machine paths.")
    ap.add_argument("--avhubert-root", type=Path,
                    help="AV-HuBERT source tree. Defaults to this repo's av_hubert folder.")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--capture-width", type=int, default=640,
                    help="Requested webcam width. Lower values keep landmark "
                         "tracking responsive on laptop GPUs.")
    ap.add_argument("--capture-height", type=int, default=480,
                    help="Requested webcam height.")
    ap.add_argument("--target-fps", type=float, default=25.0,
                    help="Frame rate fed to AV-HuBERT and used for live "
                         "utterance buffering.")
    ap.add_argument("--history-seconds", type=float, default=10.0,
                    help="Seconds of raw camera history retained for decoding.")
    ap.add_argument("--live-landmark-interval", type=float, default=0.08,
                    help="Seconds between live gate landmark requests. The "
                         "camera still captures continuously between requests.")
    ap.add_argument("--pre-roll-frames", type=int, default=8,
                    help="Raw frames to include before the live gate detects "
                         "speech start.")
    ap.add_argument("--post-roll-frames", type=int, default=5,
                    help="Raw frames to include after the live gate detects "
                         "speech end.")
    ap.add_argument("--buffer-frames", type=int, default=25,
                    help=argparse.SUPPRESS)
    ap.add_argument("--max-utt-frames", type=int, default=100,
                    help="Maximum frames buffered in one utterance before "
                         "force-decode (25fps -> 4s).")
    ap.add_argument("--activity-frames", type=int, default=3,
                    help="Need this many consecutive 'open' frames to START "
                         "an utterance.")
    ap.add_argument("--silence-frames", type=int, default=10,
                    help="This many consecutive 'closed' frames END an "
                         "utterance and trigger decode. 25fps -> 0.4s.")
    ap.add_argument("--activity-threshold", type=float, default=0.08,
                    help="Mouth-opening ratio (inner-lip dist / face width) "
                         "above which the mouth counts as 'open'. Lower it "
                         "if the gate never triggers, raise it if you get "
                         "spurious decodes when silent.")
    ap.add_argument("--min-utt-frames", type=int, default=10,
                    help="Discard utterances shorter than this many frames.")
    local_ffmpeg = _first_existing(_path_candidates()["ffmpeg"])
    ap.add_argument("--ffmpeg", default=str(local_ffmpeg or shutil.which("ffmpeg") or "ffmpeg"))
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--no-arabic", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--arabic-font", type=Path,
                    default=Path("C:/Windows/Fonts/arial.ttf"))
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ap.add_argument("--report-out", type=Path,
                    default=Path(f"runs/live/report_{ts}.pdf"))
    ap.add_argument("--log-every", type=int, default=30,
                    help="When IDLE/SPEAKING, print a status line every N "
                         "frames so you can watch the gate breathe. 0=off.")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress periodic gate logs (still print events).")
    ap.add_argument("--check-only", action="store_true",
                    help="Validate local paths/imports and exit without opening the camera.")
    args = ap.parse_args()

    try:
        _resolve_runtime_paths(args)
    except FileNotFoundError as exc:
        sys.exit(f"[config] {exc}")

    if args.check_only:
        sys.exit(0 if _check_runtime(args) else 1)

    import cv2
    import face_alignment
    import numpy as np

    print("[init] loading face_alignment (GPU) ...")
    landmarks_type = _landmarks_type_2d(face_alignment)
    if landmarks_type is None:
        sys.exit("[err] installed face_alignment has no 2D landmarks enum")
    fa = face_alignment.FaceAlignment(
        landmarks_type,
        device="cuda" if _cuda_ok() else "cpu",
        flip_input=False,
    )
    fa_lock = threading.Lock()
    decoder = Decoder(args.ckpt, args.avhubert_root, args.spm_model)
    tts = TTSEngine(enabled=not args.no_tts)
    translator = Translator(enabled=not args.no_arabic)
    overlay = Overlay(args.arabic_font, enabled_arabic=not args.no_arabic)
    report = HCIReport(model_name=decoder.model_name)

    mean_face = np.load(str(args.mean_face))
    work_dir = Path(tempfile.mkdtemp(prefix="avh_live_io_"))
    tmp_clip = work_dir / "clip.mp4"
    tmp_roi = work_dir / "roi.mp4"

    cap = _open_camera(args.camera, args.capture_width, args.capture_height,
                       args.target_fps)
    if cap is None:
        print(f"[err] cannot open any camera (tried index {args.camera} on "
              f"DSHOW/MSMF/ANY). Plug in a webcam or pick another --camera idx.")
        sys.exit(2)
    stream = CameraStream(cap, target_fps=args.target_fps,
                          history_seconds=args.history_seconds)
    stream.start()
    landmark_worker = LiveLandmarkWorker(fa, fa_lock)
    landmark_worker.start()

    print("[live] mouth-activity gate is ON.")
    print("       SPACE=force-decode current buffer   Q/ESC=quit")
    print(f"       activity_thresh={args.activity_threshold} "
          f"start_frames={args.activity_frames} "
          f"silence_frames={args.silence_frames} "
          f"min={args.min_utt_frames} max={args.max_utt_frames} "
          f"target_fps={args.target_fps:g}")

    # Logging helper -- one timestamp + tag per line
    def log(tag: str, msg: str):
        t = dt.datetime.now().strftime("%H:%M:%S")
        print(f"{t} [{tag}] {msg}", flush=True)

    log("init", "ready, waiting for first frame")

    last_roi_thumb = None
    last_english = ""
    last_arabic = ""
    status = "warming up..."
    decode_lock = threading.Lock()
    decoding = threading.Event()

    # mouth-activity gating state
    state = "IDLE"               # IDLE -> SPEAKING -> (decode) -> IDLE
    open_streak = 0              # consecutive frames mouth was open
    closed_streak = 0            # consecutive frames mouth was closed
    last_openness = 0.0
    last_lm = None
    last_lm_ts = 0.0
    gate_label = "IDLE  (waiting for landmarks)"
    utterance_start_ts: float | None = None
    utterance_end_ts: float | None = None

    def mouth_openness(lm) -> float:
        """Inner-lip vertical opening normalized by face width.
        Inner lips: 61..67. Use top mid (62) vs bottom mid (66) for vertical,
        and 60 vs 64 (outer mouth corners) for horizontal scale, fall back to
        face width 0..16 if mouth points are noisy.
        Returns 0.0 if landmarks unavailable.
        """
        if lm is None:
            return 0.0
        top = lm[62]
        bot = lm[66]
        v = float(((top[0] - bot[0]) ** 2 + (top[1] - bot[1]) ** 2) ** 0.5)
        # face horizontal extent (jaw 0 -> 16)
        face_w = float(abs(lm[16, 0] - lm[0, 0]) + 1e-6)
        return v / face_w

    def submit_decode(frames_copy, reason: str):
        if decoding.is_set():
            log("gate", f"DROP decode request ({reason}) - decoder busy")
            return
        decoding.set()
        log("gate", f"SUBMIT decode  frames={len(frames_copy)}  reason={reason}")

        def worker():
            nonlocal last_english, last_arabic, last_roi_thumb, status
            t0 = time.time()
            try:
                log("roi", f"detecting landmarks on {len(frames_copy)} buffered frames ...")
                lms_copy, hits, misses = detect_landmarks_for_frames(
                    frames_copy, fa, fa_lock, log=log,
                )
                if hits == 0:
                    report.log_decode("", "", (time.time() - t0) * 1000,
                                      len(frames_copy), success=False)
                    status = "no face landmarks"
                    log("roi", f"FAIL  no landmarks hit (miss={misses})")
                    return
                log("roi", f"extracting mouth ROI hit={hits} miss={misses} ...")
                n_roi = buffer_to_roi_mp4(
                    frames_copy, lms_copy, mean_face, tmp_roi, tmp_clip,
                    args.avhubert_root, ffmpeg=args.ffmpeg,
                )
                if n_roi < 5:
                    report.log_decode("", "", (time.time() - t0) * 1000,
                                      len(frames_copy), success=False)
                    status = f"no ROI ({n_roi} frames)"
                    log("roi", f"FAIL  only {n_roi} ROI frames produced")
                    return
                log("roi", f"OK  {n_roi} ROI frames -> {tmp_roi.name}")
                log("decode", "running AV-HuBERT beam search ...")
                english = decoder.decode_video(tmp_roi)
                latency_ms = (time.time() - t0) * 1000
                arabic = ""
                if english:
                    log("decode", f"HYP  '{english}'  ({latency_ms:.0f} ms)")
                    if translator.enabled:
                        log("trans", "translating to Arabic ...")
                        arabic = translator.translate(english)
                        log("trans", f"AR   '{arabic}'")
                else:
                    log("decode", f"EMPTY hypothesis  ({latency_ms:.0f} ms)")
                with decode_lock:
                    last_english = english
                    last_arabic = arabic
                    rcap = cv2.VideoCapture(str(tmp_roi))
                    ok, rf = rcap.read()
                    rcap.release()
                    if ok:
                        last_roi_thumb = cv2.resize(rf, (140, 140))
                report.log_decode(english, arabic, latency_ms, n_roi,
                                  success=bool(english))
                status = f"decode ok ({latency_ms:.0f} ms, {reason})"
                if english and tts.enabled:
                    log("tts", f"speaking: '{english}'")
                    tts.speak_async(english)
                    report.note_tts()
            except Exception as exc:  # noqa: BLE001
                log("decode", f"ERROR {exc!r}")
                status = f"error: {exc}"
            finally:
                decoding.clear()

        threading.Thread(target=worker, daemon=True).start()

    def flush_and_decode(reason: str, end_ts: float | None = None):
        nonlocal state, open_streak, closed_streak
        nonlocal utterance_start_ts, utterance_end_ts, gate_label
        latest_ts = stream.latest_ts() or time.perf_counter()
        stop_ts = end_ts if end_ts is not None else latest_ts
        if reason not in ("max-window", "forced"):
            stop_ts = min(latest_ts, stop_ts + args.post_roll_frames / args.target_fps)
        start_ts = utterance_start_ts
        if start_ts is None:
            start_ts = stop_ts - args.max_utt_frames / args.target_fps
        start_ts = max(0.0, start_ts)
        frames = stream.frames_between(
            start_ts, stop_ts, target_fps=args.target_fps,
            max_frames=args.max_utt_frames,
        )
        n = len(frames)
        if n >= args.min_utt_frames and not decoding.is_set():
            log("gate", f"END utterance ({reason})  buf={n}  -> decode")
            submit_decode(frames, reason)
        elif n > 0:
            log("gate", f"END utterance ({reason})  buf={n} < min "
                       f"{args.min_utt_frames}  -> DISCARD")
        else:
            log("gate", f"END utterance ({reason})  empty buffer")
        state = "IDLE"
        open_streak = 0
        closed_streak = 0
        utterance_start_ts = None
        utterance_end_ts = None
        gate_label = "IDLE  (waiting for speech)"

    win = "Live lip reading (mouth-activity gated; Q/ESC quit, SPACE force)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frame_idx = 0
    no_face_streak = 0
    last_landmark_submit = 0.0

    def handle_landmark(lm_ts: float, lm) -> None:
        nonlocal state, open_streak, closed_streak, last_openness
        nonlocal last_lm, last_lm_ts, no_face_streak, gate_label
        nonlocal utterance_start_ts, utterance_end_ts

        if lm is not None:
            last_lm = lm
            last_lm_ts = lm_ts
            if no_face_streak > 0 and not args.quiet:
                log("face", f"face reacquired after {no_face_streak} miss frames")
            no_face_streak = 0
        else:
            no_face_streak += 1
            if no_face_streak in (1, 25, 75) and not args.quiet:
                log("face", f"no face for {no_face_streak} landmark request(s)")

        openness = mouth_openness(lm)
        last_openness = openness
        is_open = openness >= args.activity_threshold

        if state == "IDLE":
            if is_open:
                open_streak += 1
                if open_streak >= args.activity_frames:
                    state = "SPEAKING"
                    utterance_start_ts = max(
                        0.0,
                        lm_ts - args.pre_roll_frames / args.target_fps,
                    )
                    utterance_end_ts = None
                    closed_streak = 0
                    log("gate", f"START utterance  m={openness:.3f} "
                                f">= {args.activity_threshold}")
            else:
                open_streak = 0
            gate_label = (f"IDLE  (open={open_streak}/"
                          f"{args.activity_frames}  m={openness:.2f})")
            return

        if is_open:
            closed_streak = 0
        else:
            closed_streak += 1
        start_ts = utterance_start_ts or lm_ts
        est_frames = int(max(0.0, lm_ts - start_ts) * args.target_fps)
        gate_label = (f"SPEAKING  buf~{est_frames}  "
                      f"sil={closed_streak}/{args.silence_frames}  "
                      f"m={openness:.2f}")
        if closed_streak >= args.silence_frames:
            utterance_end_ts = lm_ts
            flush_and_decode("silence", end_ts=lm_ts)
        elif est_frames >= args.max_utt_frames:
            utterance_end_ts = lm_ts
            flush_and_decode("max-window", end_ts=lm_ts)

    try:
        while True:
            latest = stream.latest()
            if latest is None:
                time.sleep(0.01)
                continue
            ts_frame, frame = latest
            frame_idx += 1

            now = time.perf_counter()
            if (not decoding.is_set()
                    and now - last_landmark_submit >= args.live_landmark_interval):
                if landmark_worker.submit(ts_frame, frame):
                    last_landmark_submit = now

            while True:
                try:
                    result = landmark_worker.results.get_nowait()
                except queue.Empty:
                    break
                if len(result) == 3:
                    lm_ts, _, exc = result
                    log("face", f"landmark error at {lm_ts:.3f}: {exc!r}")
                    handle_landmark(lm_ts, None)
                else:
                    lm_ts, lm = result
                    handle_landmark(lm_ts, lm)

            if last_lm is not None and ts_frame - last_lm_ts < 1.0:
                for x, y in last_lm[48:68]:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 200, 255), -1)

            # periodic gate log (so you can watch it breathe even when nothing happens)
            if (args.log_every and not args.quiet
                    and frame_idx % args.log_every == 0):
                log("gate", gate_label)

            display_status = (gate_label if not decoding.is_set()
                              else f"DECODING  ({status})")

            with decode_lock:
                e, a, t = last_english, last_arabic, last_roi_thumb
            out = overlay.draw(frame, e, a, display_status, roi_thumb=t)
            # visualize openness meter (bar on the left edge)
            bar_h = int(min(1.0, last_openness / max(args.activity_threshold * 3,
                                                     0.001)) * 200)
            is_open = last_openness >= args.activity_threshold
            color = (0, 200, 0) if is_open else (90, 90, 90)
            cv2.rectangle(out, (5, 220 - bar_h), (15, 220), color, -1)
            cv2.rectangle(out,
                          (5, 220 - int(args.activity_threshold /
                                        max(args.activity_threshold * 3,
                                            0.001) * 200)),
                          (15, 220 - int(args.activity_threshold /
                                         max(args.activity_threshold * 3,
                                             0.001) * 200) + 1),
                          (0, 0, 255), 2)
            cv2.imshow(win, out)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                log("ui", "Q/ESC -> quit")
                if state == "SPEAKING":
                    flush_and_decode("quit", end_ts=stream.latest_ts())
                break
            if key == 32:  # SPACE -> force-decode whatever is buffered
                if state == "SPEAKING":
                    log("ui", "SPACE -> force decode")
                    flush_and_decode("forced", end_ts=stream.latest_ts())
                else:
                    log("ui", "SPACE pressed but IDLE (nothing buffered)")
    finally:
        landmark_worker.stop()
        stream.stop()
        cap.release()
        cv2.destroyAllWindows()
        if not args.no_report:
            args.report_out.parent.mkdir(parents=True, exist_ok=True)
            json_path = args.report_out.with_suffix(".json")
            report.save(args.report_out, json_path)


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _open_camera(preferred: int, width: int = 640, height: int = 480,
                 fps: float = 25.0):
    """Try (preferred, then 0..3) across (DSHOW, MSMF, ANY). Return a working
    `cv2.VideoCapture` or None."""
    import cv2
    backends = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF",  cv2.CAP_MSMF),
        ("ANY",   cv2.CAP_ANY),
    ]
    indices = [preferred] + [i for i in (0, 1, 2, 3) if i != preferred]
    for idx in indices:
        for name, backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok, _ = cap.read()
                if ok:
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"[cam] opened index={idx} backend={name} "
                          f"{actual_w}x{actual_h}@{actual_fps:.1f}")
                    return cap
                cap.release()
    return None


if __name__ == "__main__":
    main()
