"""Download + arrange the training images into ImageFolder layout.

Result:
  data/damage/{train,val,test}/{damaged,not_damaged}/*.jpg
  data/severity/{train,val,test}/{minor,moderate,severe}/*.jpg

Source: Kaggle "Car Damage Detection" (anujms/car-damage-detection)
  data1a/ -> 00-damage, 01-whole        (damaged vs not damaged)
  data3a/ -> 01-minor, 02-moderate, 03-severe (severity)

Needs a Kaggle API token in ~/.kaggle/kaggle.json (kaggle.com -> Settings -> Create New Token).
Add your own insurer photos with --extra path/to/folder (sub-folders named like the classes).
"""
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
RAW = PROJECT_DIR / "raw"
OUT = PROJECT_DIR / "data"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

DAMAGE_MAP = {"00-damage": "damaged", "01-whole": "not_damaged"}
SEVERITY_MAP = {"01-minor": "minor", "02-moderate": "moderate", "03-severe": "severe"}


def download() -> None:
    target = RAW / "car-damage-detection"
    if target.exists():
        return
    target.mkdir(parents=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", "anujms/car-damage-detection", "-p", str(target), "--unzip"], check=True)


def collect(root: Path, mapping: dict[str, str]) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {v: [] for v in mapping.values()}
    for folder in root.rglob("*"):
        if folder.is_dir() and folder.name in mapping:
            found[mapping[folder.name]] += [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT]
    return found


def split_copy(task: str, files: dict[str, list[Path]], seed: int = 42) -> None:
    rnd = random.Random(seed)
    for label, paths in files.items():
        paths = sorted(set(paths))
        rnd.shuffle(paths)
        n = len(paths)
        if n == 0:
            print(f"{task:9s} {label:12s} total=0")
            continue
        parts = {"train": paths[: int(n * 0.75)], "val": paths[int(n * 0.75): int(n * 0.9)], "test": paths[int(n * 0.9):]}
        for split, items in parts.items():
            dest = OUT / task / split / label
            dest.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(items):
                shutil.copy2(src, dest / f"{label}_{i:05d}{src.suffix.lower()}")
        print(f"{task:9s} {label:12s} total={n} train={len(parts['train'])} val={len(parts['val'])} test={len(parts['test'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", type=Path, help="Folder with your own labelled photos (sub-folders = class names)")
    ap.add_argument(
        "--severity-extra",
        type=Path,
        default=PROJECT_DIR / "dataset" / "severity",
        help="Severity image folder (default: dataset/severity)",
    )
    args = ap.parse_args()
    download()
    base = RAW / "car-damage-detection"
    damage = collect(base, DAMAGE_MAP)
    severity = collect(base, SEVERITY_MAP)
    extra_folders = []
    if args.extra:
        extra_folders.append(args.extra)
    if args.severity_extra:
        extra_folders.append(args.severity_extra)

    for extra_arg in extra_folders:
        extra = extra_arg if extra_arg.is_absolute() else PROJECT_DIR / extra_arg
        if not extra.is_dir():
            print(f"Warning: extra folder not found: {extra}")
            continue
        for cls_dir in extra.iterdir():
            if not cls_dir.is_dir():
                continue
            imgs = [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT]
            (damage if cls_dir.name in damage else severity).setdefault(cls_dir.name, []).extend(imgs)
    if OUT.exists():
        shutil.rmtree(OUT)
    split_copy("damage", damage)
    split_copy("severity", severity)


if __name__ == "__main__":
    main()
