"""Download helpers. Only speech commands v2 (2.3 GB) can be fetched automatically.

FSC needs a manual download (fluent.ai form) and SLURP audio is big (several GB), so for
those we just print what to do. Nothing is downloaded unless you run this on purpose.
"""
import argparse
import tarfile
import urllib.request
from pathlib import Path

SC_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


def download_sc(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / "speech_commands_v0.02.tar.gz"
    if not tar_path.exists():
        print("downloading speech commands v2, about 2.3 GB")
        urllib.request.urlretrieve(SC_URL, tar_path)
    with tarfile.open(tar_path) as tar:
        tar.extractall(out_dir / "speech_commands_v2", filter="data")
    print("done:", out_dir / "speech_commands_v2")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sc", "fsc", "slurp"])
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    if args.what == "sc":
        download_sc(Path(args.out))
    elif args.what == "fsc":
        print("FSC: the fluent.ai download link is dead. Use the zenodo copy, record 14722453 (5 tar files, 2.3 GB).")
        print(f"put the tars in {args.out}/fsc_zenodo/ , then run: python data/prepare.py fsc")
    else:
        print("SLURP: annotations at https://github.com/pswietojanski/slurp (dataset/slurp/*.jsonl),")
        print("audio from the zenodo link in that readme (slurp_real, several GB, check disk first).")
        print(f"put them in {args.out}/slurp/ as slurp/*.jsonl and slurp_real/*.flac")
