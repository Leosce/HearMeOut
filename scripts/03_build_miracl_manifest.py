"""Build speaker-disjoint {train,valid,test}.{tsv,wrd} for MIRACL-VC1.

Input:
  <root>/file.list            produced by 02_run_roi_pipeline.sh, each line =
                              "<speaker>/<words|phrases>/<label_id>/<instance>"
  <root>/nframes.video        int per line, parallel to file.list
  <root>/video/<id>.mp4       88x88 grayscale 25 fps clips

Output:
  <out>/train.tsv  <out>/train.wrd
  <out>/valid.tsv  <out>/valid.wrd
  <out>/test.tsv   <out>/test.wrd

Manifest (tsv) format expected by AV-HuBERT's AVHubertDataset:
  <tab-separated>
  line 0:  <root_dir>                           (used as prefix for col 2,3)
  line N:  <id>\t<video_rel>\t<audio_or_None>\t<n_video>\t<n_audio>

MIRACL-VC1 has no usable audio so column 3 = "None" and column 5 = 0.

Speaker split (matches plan):
  train  = 12 speakers
  valid  =  1 speaker
  test   =  2 speakers
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

# Default speaker-disjoint split (override with --split if needed).
DEFAULT_TRAIN = ["F01", "F02", "F04", "F05", "F06", "F07", "F08", "F09",
                 "F10", "F11", "M01", "M02"]
DEFAULT_VALID = ["M04"]
DEFAULT_TEST = ["M07", "M08"]


def load_labels(labels_tsv: Path) -> dict[tuple[str, str], str]:
    """Return {(kind, id) -> canonical transcript} from docs/labels_miracl_vc1.tsv."""
    mapping: dict[tuple[str, str], str] = {}
    with labels_tsv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mapping[(row["kind"], row["id"])] = row["transcript"].strip().lower()
    return mapping


def parse_id(clip_id: str) -> tuple[str, str, str, str]:
    """<speaker>/<kind>/<label_id>/<instance>  ->  (spk, kind, label_id, inst)."""
    parts = clip_id.split("/")
    if len(parts) != 4:
        raise ValueError(f"Unexpected id structure: {clip_id!r}")
    return tuple(parts)  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="MIRACL-VC1 preprocessed root")
    ap.add_argument("--out", required=True, help="Output manifest directory")
    ap.add_argument("--labels", default="docs/labels_miracl_vc1.tsv")
    ap.add_argument("--train-spk", nargs="+", default=DEFAULT_TRAIN)
    ap.add_argument("--valid-spk", nargs="+", default=DEFAULT_VALID)
    ap.add_argument("--test-spk", nargs="+", default=DEFAULT_TEST)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    ids = (root / "file.list").read_text().splitlines()
    nframes = [int(x) for x in (root / "nframes.video").read_text().splitlines()]
    if len(ids) != len(nframes):
        raise RuntimeError(
            f"file.list ({len(ids)}) and nframes.video ({len(nframes)}) disagree"
        )

    labels = load_labels(Path(args.labels))

    splits: dict[str, list[tuple[str, int, str]]] = {
        "train": [], "valid": [], "test": [],
    }
    spk_to_split = {s: "train" for s in args.train_spk}
    spk_to_split.update({s: "valid" for s in args.valid_spk})
    spk_to_split.update({s: "test" for s in args.test_spk})

    dropped = 0
    for clip_id, n in zip(ids, nframes):
        spk, kind, label_id, _inst = parse_id(clip_id)
        split = spk_to_split.get(spk)
        if split is None:
            dropped += 1
            continue
        transcript = labels.get((kind, label_id))
        if transcript is None:
            dropped += 1
            continue
        splits[split].append((clip_id, n, transcript))

    video_root = root / "video"
    for name, rows in splits.items():
        tsv_path = out / f"{name}.tsv"
        wrd_path = out / f"{name}.wrd"
        with tsv_path.open("w", encoding="utf-8", newline="") as ftsv, \
             wrd_path.open("w", encoding="utf-8", newline="") as fwrd:
            ftsv.write(str(video_root) + "\n")
            for clip_id, n, transcript in rows:
                video_rel = f"{clip_id}.mp4"
                # id, video_rel, audio (None), n_video, n_audio
                ftsv.write(f"{clip_id}\t{video_rel}\tNone\t{n}\t0\n")
                fwrd.write(transcript + "\n")
        print(f"{name}: {len(rows)} clips  ->  {tsv_path}")

    if dropped:
        print(f"Dropped {dropped} clips (speaker not in split or unknown label)")


if __name__ == "__main__":
    main()
