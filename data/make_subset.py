"""Copy a small random subset of a prepared split into one folder + zip it, so it can go on a pi.

usage: python data/make_subset.py --split test --n 200 --out C:/Users/panas/Desktop/fsc_test_200
The folder gets wavs/ and manifest.csv (paths relative to the folder). The zip is made next to it.
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prepared", default="data/prepared")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(f"{args.prepared}/fsc_{args.split}.csv", newline="")))
    random.Random(args.seed).shuffle(rows)    # fixed seed so everyone gets the same clips
    rows = rows[:args.n]

    out = Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    kept = []
    for r in rows:
        src = Path(r["path"].replace("\\", "/"))     # manifest was written on windows
        shutil.copy(src, out / "wavs" / src.name)
        kept.append({**r, "path": f"wavs/{src.name}"})
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kept[0].keys()))
        w.writeheader()
        w.writerows(kept)
    zip_path = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    size_mb = Path(zip_path).stat().st_size / 1e6
    print(f"{len(kept)} clips from {args.split}, zip {zip_path} ({size_mb:.1f} MB)")
