"""Build a joint {train,valid}.{tsv,wrd} by concatenating LRS3-trainval,
MIRACL-VC1 (oversampled), and GRID manifests.

Each source manifest has its own root directory on tsv line 0. Because
AV-HuBERT's AVHubertDataset only supports one root per manifest, we rewrite
entries to use ABSOLUTE paths and set the joint manifest's root line to "/"
(or any existing directory — it is ignored when relative paths are absolute).

Usage:
  python scripts/05_build_joint_manifest.py \
      --lrs3 <dir with train.tsv/valid.tsv/train.wrd/valid.wrd> \
      --miracl <same> --grid <same> \
      --out data/joint --miracl-repeat 3
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def load_manifest(folder: Path, split: str) -> tuple[str, list[str], list[str]]:
    tsv_lines = (folder / f"{split}.tsv").read_text().splitlines()
    wrd_lines = (folder / f"{split}.wrd").read_text().splitlines()
    root = tsv_lines[0]
    rows = tsv_lines[1:]
    if len(rows) != len(wrd_lines):
        raise RuntimeError(f"{folder}/{split}: tsv/wrd length mismatch")
    return root, rows, wrd_lines


def absolutize(rows: list[str], root: str) -> list[str]:
    """Rewrite the video (col 2) and audio (col 3) columns to absolute paths."""
    out: list[str] = []
    for row in rows:
        parts = row.split("\t")
        # id, video_rel, audio_rel, n_video, n_audio
        if not os.path.isabs(parts[1]):
            parts[1] = str(Path(root) / parts[1])
        if parts[2] != "None" and not os.path.isabs(parts[2]):
            parts[2] = str(Path(root) / parts[2])
        out.append("\t".join(parts))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs3", required=True)
    ap.add_argument("--miracl", required=True)
    ap.add_argument("--grid", default=None, help="Optional")
    ap.add_argument("--out", required=True)
    ap.add_argument("--miracl-repeat", type=int, default=3,
                    help="Oversampling factor for MIRACL-VC1 train split")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid"):
        joined_rows: list[str] = []
        joined_wrds: list[str] = []

        sources: list[tuple[str, Path, int]] = [
            ("lrs3", Path(args.lrs3), 1),
            ("miracl", Path(args.miracl), args.miracl_repeat if split == "train" else 1),
        ]
        if args.grid:
            sources.append(("grid", Path(args.grid), 1))

        for name, folder, repeat in sources:
            try:
                root, rows, wrds = load_manifest(folder, split)
            except FileNotFoundError:
                print(f"[{split}] skipping {name} (no {split}.tsv)")
                continue
            rows = absolutize(rows, root)
            for _ in range(repeat):
                joined_rows.extend(rows)
                joined_wrds.extend(wrds)
            print(f"[{split}] {name}: +{len(rows) * repeat} rows (x{repeat})")

        tsv_path = out / f"{split}.tsv"
        wrd_path = out / f"{split}.wrd"
        with tsv_path.open("w", encoding="utf-8", newline="") as ftsv:
            ftsv.write("/\n")  # placeholder root; per-row paths are absolute
            for row in joined_rows:
                ftsv.write(row + "\n")
        wrd_path.write_text("\n".join(joined_wrds) + "\n", encoding="utf-8")

        print(f"[{split}] total: {len(joined_rows)} rows -> {tsv_path}")


if __name__ == "__main__":
    main()
