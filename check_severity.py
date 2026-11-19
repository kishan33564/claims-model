from pathlib import Path

base_path = Path(__file__).resolve().parent / "dataset" / "severity"

classes = ["minor", "moderate", "severe"]

for cls in classes:
    folder = base_path / cls

    if folder.is_dir():
        images = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        print(cls, ":", len(images), "images")
    else:
        print(cls, ": Folder not found")