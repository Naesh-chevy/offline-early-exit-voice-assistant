"""Turn the raw datasets into small manifest csvs + label maps in data/prepared/.

usage: python data/prepare.py sc|fsc|slurp
Audio files stay where they are, manifests only point at them.
"""
import argparse
import csv
import glob
import json
import os
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio.labels import build_label_map, save_label_maps  # noqa: E402

RAW = Path("data/raw")
OUT = Path("data/prepared")


def write_csv(path: Path, rows: list, cols: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def prepare_sc(wake_word: str) -> None:
    root = RAW / "speech_commands_v2"
    val = set((root / "validation_list.txt").read_text().split())
    test = set((root / "testing_list.txt").read_text().split())
    splits = {"train": [], "valid": [], "test": []}
    for wav in sorted(root.glob("*/*.wav")):
        word = wav.parent.name
        if word.startswith("_"):
            continue
        rel = f"{word}/{wav.name}"
        split = "valid" if rel in val else "test" if rel in test else "train"
        splits[split].append({"path": str(wav), "word": word, "is_wake": int(word == wake_word)})
    for split, rows in splits.items():
        write_csv(OUT / f"sc_{split}.csv", rows, ["path", "word", "is_wake"])
        print(f"sc {split}: {len(rows)} clips, {sum(r['is_wake'] for r in rows)} wake")


def fsc_rows_from_csv(root: Path, split: str) -> list:
    """Original fsc layout: data/<split>_data.csv + wavs/."""
    rows = []
    with open(root / "data" / f"{split}_data.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"path": str(root / r["path"]), "action": r["action"], "object": r["object"],
                         "location": r["location"], "text": r["transcription"]})
    return rows


def fsc_rows_from_tars(src: Path, root: Path, split: str) -> list:
    """Zenodo webdataset copy: tars of <id>.wav + <id>.json. Wavs are extracted once into root/wavs/speakers."""
    wav_dir = root / "wavs" / "speakers"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for tar_path in sorted(glob.glob(str(src / f"fluentspeechcommands_{split}_*.tar"))):
        labels = {}
        with tarfile.open(tar_path) as tar:
            for m in tar:
                if not m.isfile():
                    continue
                stem, ext = os.path.splitext(os.path.basename(m.name))   # base name only, never trust tar paths
                data = tar.extractfile(m).read()
                if ext == ".wav":
                    (wav_dir / f"{stem}.wav").write_bytes(data)
                elif ext == ".json":
                    labels[stem] = json.loads(data)
        for stem, j in sorted(labels.items()):
            rows.append({"path": str(wav_dir / f"{stem}.wav"), "action": j["action"], "object": j["object"],
                         "location": j["location"], "text": j["transcription"]})
    return rows


def prepare_fsc() -> None:
    root = RAW / "fsc"
    tars = RAW / "fsc_zenodo"
    cols = ["path", "action", "object", "location", "text"]
    use_csv = (root / "data" / "train_data.csv").exists()
    all_rows = {}
    for split in ["train", "valid", "test"]:
        rows = fsc_rows_from_csv(root, split) if use_csv else fsc_rows_from_tars(tars, root, split)
        all_rows[split] = rows
        write_csv(OUT / f"fsc_{split}.csv", rows, cols)
        print(f"fsc {split}: {len(rows)} clips")
    # label maps from all splits so ids never shift
    flat = [r for rows in all_rows.values() for r in rows]
    maps = {k: build_label_map(r[k] for r in flat) for k in ["action", "object", "location"]}
    save_label_maps(maps, str(OUT / "fsc_labels.json"))
    print({k: len(v) for k, v in maps.items()})


def prepare_slurp() -> None:
    root = RAW / "slurp"
    cols = ["path", "action", "text", "slurp_id"]
    for split, name in [("train", "train.jsonl"), ("valid", "devel.jsonl"), ("test", "test.jsonl")]:
        rows = []
        with open(root / "slurp" / name) as f:
            for line in f:
                d = json.loads(line)
                if d["scenario"] != "iot":
                    continue
                for rec in d["recordings"]:
                    rows.append({"path": str(root / "slurp_real" / rec["file"]), "action": d["action"],
                                 "text": d["sentence"], "slurp_id": d["slurp_id"]})
        write_csv(OUT / f"slurp_iot_{split}.csv", rows, cols)
        print(f"slurp iot {split}: {len(rows)} recordings")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sc", "fsc", "slurp"])
    ap.add_argument("--wake_word", default="marvin")
    args = ap.parse_args()
    {"sc": lambda: prepare_sc(args.wake_word), "fsc": prepare_fsc, "slurp": prepare_slurp}[args.what]()
