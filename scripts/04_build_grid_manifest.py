"""Build {train,valid}.{tsv,wrd} for GRID after running the AV-HuBERT ROI
pipeline on the GRID corpus (same tooling as MIRACL-VC1; see
02_run_roi_pipeline.sh).

GRID transcripts come from the official alignment files (*.align) whose 4th
column per line is the word; we concatenate non-silence words lowercased.

Input:
  <root>/file.list         "<speaker>/<uid>"
  <root>/nframes.video     parallel to file.list
  <root>/video/<id>.mp4    88x88 grayscale 25 fps clips
  <root>/align/<speaker>/<uid>.align     provided by GRID

Split (plan):
  train = S1 .. S30
  valid = S31 .. S33
  (no test set written here; evaluate retention on LRS3 test instead.)
"""
from __future__ import annotations

import argparse
from pathlib import Path


def read_align(path: Path) -> str:
    """Return the GRID sentence from an .align file, lowercased, no 'sil'."""
    words: list[str] = []
    for line in path.read_text().splitlines():
        toks = line.strip().split()
        if len(toks) >= 3 and toks[2].lower() != "sil":
            words.append(toks[2].lower())
    return " ".join(words)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-spk", nargs="+",
                    default=[f"s{i}" for i in range(1, 31)])
    ap.add_argument("--valid-spk", nargs="+",
                    default=[f"s{i}" for i in range(31, 34)])
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    ids = (root / "file.list").read_text().splitlines()
    nframes = [int(x) for x in (root / "nframes.video").read_text().splitlines()]

    spk_to_split = {s: "train" for s in args.train_spk}
    spk_to_split.update({s: "valid" for s in args.valid_spk})

    buckets: dict[str, list[tuple[str, int, str]]] = {"train": [], "valid": []}
    dropped = 0
    for clip_id, n in zip(ids, nframes):
        spk = clip_id.split("/")[0]
        split = spk_to_split.get(spk)
        if split is None:
            dropped += 1
            continue
        align_path = root / "align" / f"{clip_id}.align"
        if not align_path.exists():
            dropped += 1
            continue
        transcript = read_align(align_path)
        if not transcript:
            dropped += 1
            continue
        buckets[split].append((clip_id, n, transcript))

    video_root = root / "video"
    for name, rows in buckets.items():
        tsv_path = out / f"{name}.tsv"
        wrd_path = out / f"{name}.wrd"
        with tsv_path.open("w", encoding="utf-8", newline="") as ftsv, \
             wrd_path.open("w", encoding="utf-8", newline="") as fwrd:
            ftsv.write(str(video_root) + "\n")
            for clip_id, n, transcript in rows:
                ftsv.write(f"{clip_id}\t{clip_id}.mp4\tNone\t{n}\t0\n")
                fwrd.write(transcript + "\n")
        print(f"{name}: {len(rows)} clips  ->  {tsv_path}")

    if dropped:
        print(f"Dropped {dropped} clips")


if __name__ == "__main__":
    main()
